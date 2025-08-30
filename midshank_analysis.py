import os
import re
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from euler_angle_calculations import quaternion_conjugation


def parse_subject_info_matlab(path: str) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Parse a MATLAB subject info .m file (e.g., P_004_Info.m) to extract:
    - serial_suffix_to_index: map like {"76E":0, ...}
    - num_mapping: map of sensor name -> index (e.g., {"MidShankLat":2, ...})
    - file_format: dict with AccColumns (1-based inclusive range), RVelColumns, QuatColumnOne, RowStart, FR
    """
    serial_suffix_to_index: Dict[str, int] = {}
    num_mapping: Dict[str, int] = {}
    file_format: Dict[str, int] = {}

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Serial comments: % Serial 783 = 2;
    for m in re.finditer(r"%\s*Serial\s+([0-9A-Fa-f]{3})\s*=\s*(\d+)\s*;", text):
        serial_suffix_to_index[m.group(1).upper()] = int(m.group(2))

    # num.X = N; lines
    for m in re.finditer(r"num\.([A-Za-z0-9_]+)\s*=\s*(\d+)\s*;", text):
        num_mapping[m.group(1)] = int(m.group(2))

    # File format
    m = re.search(r"AccColumns\s*=\s*([0-9]+)\s*:\s*([0-9]+)\s*;", text)
    if m:
        file_format['AccStart'] = int(m.group(1))
        file_format['AccEnd'] = int(m.group(2))
    m = re.search(r"RVelColumns\s*=\s*([0-9]+)\s*:\s*([0-9]+)\s*;", text)
    if m:
        file_format['GyrStart'] = int(m.group(1))
        file_format['GyrEnd'] = int(m.group(2))
    m = re.search(r"QuatColumnOne\s*=\s*([0-9]+)\s*;", text)
    if m:
        file_format['QuatColumnOne'] = int(m.group(1))
    m = re.search(r"RowStart\s*=\s*(\d+)\s*;", text)
    if m:
        file_format['RowStart'] = int(m.group(1))
    m = re.search(r"FR\s*=\s*(\d+)\s*;", text)
    if m:
        file_format['FR'] = int(m.group(1))

    return serial_suffix_to_index, num_mapping, file_format


def find_files_for_serial(folder: str, serial_suffix: str) -> List[str]:
    serial_suffix = serial_suffix.upper()
    out = []
    for name in os.listdir(folder):
        if name.lower().endswith('.txt') and name.upper().endswith(serial_suffix + '.TXT'):
            out.append(os.path.join(folder, name))
    return sorted(out)


def read_xsens_txt(path: str, row_start_1based: int) -> np.ndarray:
    df = pd.read_csv(path, sep='\t', header=None, skiprows=row_start_1based - 1)
    return df.values


def extract_columns(data: np.ndarray, start_1b: int, end_1b: int) -> np.ndarray:
    return data[:, (start_1b - 1):end_1b]


def quat_series_rotate_vectors(quats_wxyz: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Rotate 3xN vector series by Nx4 quaternions [w,x,y,z]. Returns 3xN.
    """
    n = quats_wxyz.shape[0]
    out = np.zeros((3, n))
    if vectors.shape[0] == n:
        # vectors is Nx3
        for i in range(n):
            out[:, i] = quaternion_conjugation(quats_wxyz[i, :], vectors[i, :])
    else:
        # vectors is 3 or 3xN (broadcast 3-vector if needed)
        if vectors.ndim == 1:
            vec = vectors
            for i in range(n):
                out[:, i] = quaternion_conjugation(quats_wxyz[i, :], vec)
        else:
            for i in range(n):
                out[:, i] = quaternion_conjugation(quats_wxyz[i, :], vectors[:, i])
    return out


def estimate_axes_from_data(acc: np.ndarray, gyr: np.ndarray, fr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate SI via gravity during a likely standing period, ML via PCA of gyro in steady walking,
    AP via cross product. Lightweight heuristic: use first 5s for standing, middle segment for gait.
    Returns (SI, ML, AP) as 3-vectors.
    """
    n = acc.shape[0]
    # Standing: first 5 seconds or first 300 samples
    stand_end = min(n, max(100, int(5 * fr)))
    g_vec = acc[:stand_end, :].mean(axis=0)
    si = g_vec / (np.linalg.norm(g_vec) + 1e-9)

    # Gait window: middle third
    s = n // 3
    e = 2 * n // 3
    gyr_mid = gyr[s:e, :]
    # Smooth simple moving average (5-point)
    if gyr_mid.shape[0] >= 5:
        w = np.array([1, 2, 3, 2, 1])
        w = w / w.sum()
        gm = np.vstack([
            np.convolve(gyr_mid[:, i], w, mode='same') for i in range(3)
        ]).T
    else:
        gm = gyr_mid
    # PCA first component
    # Compute covariance eigenvectors
    X = gm - gm.mean(axis=0)
    C = (X.T @ X) / max(1, X.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(C)
    ml = eigvecs[:, np.argmax(eigvals)]
    ml = ml / (np.linalg.norm(ml) + 1e-9)
    # AP via cross and re-orthogonalize
    ap = np.cross(ml, si)
    ap = ap / (np.linalg.norm(ap) + 1e-9)
    ml = np.cross(si, ap)
    ml = ml / (np.linalg.norm(ml) + 1e-9)
    return si, ml, ap


def main():
    # Paths
    repo_root = os.path.dirname(os.path.abspath(__file__))
    subj_info = os.path.join(repo_root, 'Subject Information', 'Subject Information', 'P_004_Info.m')
    trial_dir = os.path.join(repo_root, 'Subject IMU Data Text Files -1', 'Trial Data P_004')

    serial_to_idx, num_map, fmt = parse_subject_info_matlab(subj_info)

    # Target sensors: MidShankLat (serial suffix from index 2), and ShinBone (index 4)
    # Assumption: "mid shank frontal" corresponds to ShinBone (anterior tibial crest placement)
    idx_midshanklat = num_map.get('MidShankLat')
    idx_shinbone = num_map.get('ShinBone')
    # Reverse lookup serial suffixes
    idx_to_serial_suffix = {v: k for k, v in serial_to_idx.items()}
    serial_midshank = idx_to_serial_suffix[idx_midshanklat]
    serial_shinbone = idx_to_serial_suffix[idx_shinbone]

    files_mid = find_files_for_serial(trial_dir, serial_midshank)
    files_shin = find_files_for_serial(trial_dir, serial_shinbone)
    if not files_mid or not files_shin:
        raise FileNotFoundError('Could not find required sensor files in trial directory.')

    # Use first occurrence of each time block
    f_mid = files_mid[0]
    f_shin = files_shin[0]

    data_mid = read_xsens_txt(f_mid, fmt['RowStart'])
    data_shin = read_xsens_txt(f_shin, fmt['RowStart'])

    # Extract raw columns (convert 1-based to slices)
    acc_mid = extract_columns(data_mid, fmt['AccStart'], fmt['AccEnd'])
    gyr_mid = extract_columns(data_mid, fmt['GyrStart'], fmt['GyrEnd'])
    quat_mid = extract_columns(data_mid, fmt['QuatColumnOne'], fmt['QuatColumnOne'] + 3)

    acc_shin = extract_columns(data_shin, fmt['AccStart'], fmt['AccEnd'])
    gyr_shin = extract_columns(data_shin, fmt['GyrStart'], fmt['GyrEnd'])

    fr = int(fmt.get('FR', 60))
    t_mid = np.arange(acc_mid.shape[0]) / fr
    t_shin = np.arange(acc_shin.shape[0]) / fr

    # 1) Plot raw data (acc and gyro) for the two sensors
    out_dir = trial_dir
    plt.figure(figsize=(12, 8))
    for i, axlbl in enumerate(['X', 'Y', 'Z']):
        plt.subplot(2, 3, i + 1)
        plt.plot(t_mid, acc_mid[:, i], label='MidShankLat Acc_' + axlbl)
        plt.plot(t_shin, acc_shin[:, i], label='ShinBone Acc_' + axlbl, alpha=0.8)
        plt.grid(True)
        plt.legend()
        if i == 0:
            plt.title('Raw Acceleration')
        plt.subplot(2, 3, 3 + i + 1)
        plt.plot(t_mid, gyr_mid[:, i], label='MidShankLat Gyr_' + axlbl)
        plt.plot(t_shin, gyr_shin[:, i], label='ShinBone Gyr_' + axlbl, alpha=0.8)
        plt.grid(True)
        plt.legend()
        if i == 0:
            plt.title('Raw Gyro')
    plt.tight_layout()
    raw_png = os.path.join(out_dir, 'midshank_raw_P004.png')
    plt.savefig(raw_png, dpi=150)
    plt.close()

    # 2) Sensor orientation (estimate axes from data for MidShankLat)
    si_mid, ml_mid, ap_mid = estimate_axes_from_data(acc_mid, gyr_mid, fr)

    # 3) Apply rotation (use quaternions provided by Xsens to rotate accelerations to global frame)
    # Normalize quats to unit length for safety
    quat_mid_n = quat_mid / (np.linalg.norm(quat_mid, axis=1, keepdims=True) + 1e-9)
    acc_mid_global = quat_series_rotate_vectors(quat_mid_n, acc_mid)

    # 4) Compare with simple ground truth: during the initial standing period, gravity should align with -SI.
    # We'll use the magnitude of acc and the projection along estimated SI vs. quaternion-rotated Z.
    stand_end = min(acc_mid.shape[0], max(100, int(5 * fr)))
    # Compute mean global acceleration vector during standing
    g_est_vec = acc_mid_global[:, :stand_end].mean(axis=1)
    g_est_vec = g_est_vec / (np.linalg.norm(g_est_vec) + 1e-9)

    # Angle between quaternion-derived gravity and estimated SI
    cosang = float(np.clip(np.dot(g_est_vec, si_mid), -1.0, 1.0))
    angle_deg = np.degrees(np.arccos(cosang))

    # 5) Plot curves: raw, a lightly filtered version, rotated (global), and a reference gravity line
    # Simple moving average smoothing for visualization
    def smooth3(x: np.ndarray, k: int = 11) -> np.ndarray:
        if k <= 1:
            return x
        k = min(k, len(x))
        w = np.ones(k) / k
        return np.convolve(x, w, mode='same')

    plt.figure(figsize=(12, 10))
    for i, axlbl in enumerate(['X', 'Y', 'Z']):
        plt.subplot(3, 1, i + 1)
        raw = acc_mid[:, i]
        proc = smooth3(raw, 21)
        rot = acc_mid_global[i, :]
        plt.plot(t_mid, raw, color='gray', alpha=0.5, label=f'Raw Acc {axlbl}')
        plt.plot(t_mid, proc, color='C0', label=f'Processed Acc {axlbl}')
        plt.plot(t_mid, rot, color='C1', label=f'Rotated Acc (global) {axlbl}')
        # Ground truth reference: project rotated acceleration on gravity direction during standing
        g_dot = np.dot(acc_mid_global.T, g_est_vec)
        plt.plot(t_mid, g_dot, color='C3', alpha=0.7, label='Ground truth (proj on gravity)')
        plt.grid(True)
        plt.legend()
        if i == 0:
            plt.title(f'MidShankLat Acc Rotation (angle SI vs quat-gravity ~ {angle_deg:.1f} deg)')
        plt.xlabel('Time (s)')
        plt.ylabel('m/s^2')
    plt.tight_layout()
    rot_png = os.path.join(out_dir, 'midshank_rotation_P004.png')
    plt.savefig(rot_png, dpi=150)
    plt.close()

    print('Saved plots:')
    print(' -', raw_png)
    print(' -', rot_png)
    print('Note: Assumed "mid shank frontal" corresponds to ShinBone sensor for comparison.')


if __name__ == '__main__':
    main()
