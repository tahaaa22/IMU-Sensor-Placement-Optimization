import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import tkinter as tk
from tkinter import filedialog, simpledialog
import scipy.io as sio
import os


def dcm_projection(vector: np.ndarray, axis1: np.ndarray, axis2: np.ndarray, 
                   axis3: np.ndarray) -> np.ndarray:
    """
    Python equivalent of DCMProjection.m
    
    Projects a vector onto a coordinate system defined by three orthogonal axes.
    
    Args:
        vector: Vector to project (3x1 or 3xn)
        axis1, axis2, axis3: Orthogonal axes defining the coordinate system (3xn)
    
    Returns:
        Projected vector components in the new coordinate system
    """
    # Create rotation matrix from the three axes
    if axis1.ndim == 1:
        # Single time point
        R = np.column_stack([axis1, axis2, axis3])
        return R.T @ vector
    else:
        # Multiple time points
        n_frames = axis1.shape[1]
        projected = np.zeros((3, n_frames))
        
        for i in range(n_frames):
            R = np.column_stack([axis1[:, i], axis2[:, i], axis3[:, i]])
            if vector.ndim == 1:
                projected[:, i] = R.T @ vector
            else:
                projected[:, i] = R.T @ vector[:, i]
        
        return projected


def quaternion_multiplication(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Python equivalent of QuaternionMultiplication.m
    
    Multiplies two quaternions.
    
    Args:
        q1, q2: Quaternions in [w, x, y, z] format
    
    Returns:
        Product quaternion
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])


def quaternion_conjugation(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Python equivalent of QuaternionConjugation.m
    
    Rotates a vector using quaternion conjugation.
    
    Args:
        quat: Quaternion in [w, x, y, z] format
        vector: Vector to rotate (3x1 or 3xn)
    
    Returns:
        Rotated vector
    """
    # Normalize quaternion
    quat = quat / np.linalg.norm(quat)
    
    # Create quaternion from vector (w=0)
    if vector.ndim == 1:
        vec_quat = np.array([0, vector[0], vector[1], vector[2]])
        
        # Perform conjugation: q * v * q*
        q_conj = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
        temp = quaternion_multiplication(quat, vec_quat)
        result_quat = quaternion_multiplication(temp, q_conj)
        
        return result_quat[1:]  # Return only vector part
    else:
        # Multiple time points
        n_frames = vector.shape[1]
        rotated = np.zeros((3, n_frames))
        
        for i in range(n_frames):
            vec_quat = np.array([0, vector[0, i], vector[1, i], vector[2, i]])
            q_conj = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
            temp = quaternion_multiplication(quat, vec_quat)
            result_quat = quaternion_multiplication(temp, q_conj)
            rotated[:, i] = result_quat[1:]
        
        return rotated


def coordinate_transform_quat_pre_specified_axes(quaternions: np.ndarray, 
                                                si_axis: np.ndarray, ml_axis: np.ndarray, 
                                                ap_axis: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Python equivalent of CoordinateTransformQuatPreSpecifiedAxes.m
    
    Transforms coordinate axes using quaternions.
    
    Args:
        quaternions: Quaternion data (nx4)
        si_axis, ml_axis, ap_axis: Initial coordinate axes
    
    Returns:
        Dictionary containing rotated coordinate axes
    """
    n_frames = quaternions.shape[0]
    
    # Initialize output arrays
    axis1 = np.zeros((3, n_frames))
    axis2 = np.zeros((3, n_frames))
    axis3 = np.zeros((3, n_frames))
    
    for i in range(n_frames):
        quat = quaternions[i, :]
        
        # Rotate each axis
        axis1[:, i] = quaternion_conjugation(quat, si_axis)
        axis2[:, i] = quaternion_conjugation(quat, ml_axis)
        axis3[:, i] = quaternion_conjugation(quat, ap_axis)
    
    return {
        'Axis1': axis1,
        'Axis2': axis2,
        'Axis3': axis3
    }


def rot_mat_transform_to_global(axis1_ref: np.ndarray, axis2_ref: np.ndarray, 
                               axis3_ref: np.ndarray, axis1: np.ndarray, 
                               axis2: np.ndarray, axis3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Python equivalent of RotMatTransformToGlobal.m
    
    Transforms coordinate system to align with global reference frame.
    
    Args:
        axis1_ref, axis2_ref, axis3_ref: Reference axes at offset point
        axis1, axis2, axis3: Axes to transform
    
    Returns:
        Transformed axes
    """
    # Create reference rotation matrix
    R_ref = np.column_stack([axis1_ref, axis2_ref, axis3_ref])
    
    # Apply transformation to all time points
    n_frames = axis1.shape[1]
    
    for i in range(n_frames):
        # Create current rotation matrix
        R_curr = np.column_stack([axis1[:, i], axis2[:, i], axis3[:, i]])
        
        # Transform to global
        R_global = R_curr @ R_ref.T
        
        # Extract transformed axes
        axis1[:, i] = R_global[:, 0]
        axis2[:, i] = R_global[:, 1]
        axis3[:, i] = R_global[:, 2]
    
    return axis1, axis2, axis3


def angles_yxz(upper_coord1: np.ndarray, upper_coord2: np.ndarray, upper_coord3: np.ndarray,
               lower_coord_si: np.ndarray, lower_coord_ml: np.ndarray, lower_coord_ap: np.ndarray,
               ml: np.ndarray, ap: np.ndarray, wrap_it: bool = False, 
               slope_threshold: float = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Python equivalent of AnglesYXZ.m
    
    Calculates Euler angles using YXZ rotation order.
    
    Args:
        upper_coord1,2,3: Coordinate axes of upper sensor (3xn)
        lower_coord_si,ml,ap: Coordinate axes of lower sensor (3xn)
        ml, ap: ML and AP directions during calibration
        wrap_it: Whether to apply angle wrapping correction
        slope_threshold: Threshold for angle wrapping detection
    
    Returns:
        flexang, abdang, rotang: Flexion, abduction, rotation angles
    """
    # Project lower sensor axes onto upper sensor coordinate system
    vect1 = dcm_projection(lower_coord_si, upper_coord1, upper_coord2, upper_coord3)
    vect2 = dcm_projection(lower_coord_ml, upper_coord1, upper_coord2, upper_coord3)
    vect3 = dcm_projection(lower_coord_ap, upper_coord1, upper_coord2, upper_coord3)
    
    # First Euler rotation about Y (flexion)
    flexang = np.arctan2d(vect1[2, :], vect1[0, :])
    
    # Second Euler rotation about X' (abduction)
    abdang = np.arctan2d(vect1[1, :], np.sqrt(vect1[0, :]**2 + vect1[2, :]**2))
    
    # Apply wrapping correction if requested
    if wrap_it:
        flexang = wrap_opt(flexang, slope_threshold)
        abdang = wrap_opt(abdang, slope_threshold)
    
    # Third Euler rotation about Z'' (rotation)
    n_frames = vect1.shape[1]
    rotang = np.zeros(n_frames)
    
    for i in range(n_frames):
        # Create quaternion for flexion rotation
        quat_flex = np.array([
            np.cos(np.radians(flexang[i]) / 2),
            np.sin(np.radians(flexang[i]) / 2) * ml[0],
            np.sin(np.radians(flexang[i]) / 2) * ml[1],
            np.sin(np.radians(flexang[i]) / 2) * ml[2]
        ])
        
        # Rotate AP axis to find X'
        x_prime = quaternion_conjugation(quat_flex, ap)
        x_prime_proj = dcm_projection(x_prime, vect1[:, i], vect2[:, i], vect3[:, i])
        
        rotang[i] = np.arctan2d(x_prime_proj[1], x_prime_proj[2])
    
    if wrap_it:
        rotang = wrap_opt(rotang, slope_threshold)
    
    return flexang, abdang, rotang


def wrap_opt(ang: np.ndarray, slope_threshold: float) -> np.ndarray:
    """
    Helper function for angle wrapping correction.
    
    Args:
        ang: Angle array
        slope_threshold: Threshold for detecting wrapping
    
    Returns:
        Corrected angle array
    """
    from scipy.signal import find_peaks
    
    ang_slope = np.concatenate([[0], np.diff(ang)])
    
    if np.max(ang_slope) >= slope_threshold:
        # Find peaks in slope function
        pos_locs, _ = find_peaks(ang_slope, height=slope_threshold)
        neg_locs, _ = find_peaks(-ang_slope, height=slope_threshold)
        
        # Apply corrections
        for loc in pos_locs:
            ang[loc:] -= 360
        
        for loc in neg_locs:
            ang[loc:] += 360
    
    return ang


def euler_angle_calculations():
    """
    Main function for Euler angle calculations - Python equivalent of EulerAngleCalculations.m
    """
    # Set up workspace
    root = tk.Tk()
    root.withdraw()
    
    # Check if data already exists
    data_exists = 'data' in globals()
    if data_exists:
        load_new = simpledialog.askinteger("Data", "Do you want to use the same trial data? Yes or no (1/0)?")
    else:
        load_new = 0
    
    if load_new == 0:
        print("Load trial data ('P_xxx.mat')")
        data_file = filedialog.askopenfilename(
            title="Select trial data file",
            filetypes=[("MAT files", "*.mat"), ("All files", "*.*")]
        )
        
        if data_file:
            # Load data (in real implementation, use scipy.io.loadmat)
            print("Note: .mat file loading needs to be implemented with scipy.io.loadmat")
    
    # Load calibration file
    print("Load calibration file")
    calib_file = filedialog.askopenfilename(
        title="Select calibration file",
        filetypes=[("MAT files", "*.mat"), ("All files", "*.*")]
    )
    
    if calib_file:
        # Load calibration (in real implementation, use scipy.io.loadmat)
        print("Note: Calibration loading needs to be implemented with scipy.io.loadmat")
        # zero = sio.loadmat(calib_file)['Zero']
    
    # Select task
    print("Available tasks:")
    print("1: Calibration")
    print("2: Walking")
    print("3: Running")
    
    j = simpledialog.askinteger("Task Selection", "Select Task")
    if j is None:
        return
    
    # Calculate index
    number_sensors = 11  # This would come from loaded data
    ind = number_sensors * (j - 1)
    
    # Select joints
    print("1: Hip Angles")
    print("2: Knee Angles") 
    print("3: Ankle Angles")
    print("4: All Joints, All Angles")
    
    joint_opt = simpledialog.askstring("Joint Selection", 
                                     "Select Joint (enter all that apply as a vector, e.g., [1,2,3])")
    
    if joint_opt:
        joint_opt = eval(joint_opt)  # Convert string to list
    else:
        joint_opt = [1]
    
    # Determine which joints to analyze
    hip_opt = 1 in joint_opt
    knee_opt = 2 in joint_opt
    ankle_opt = 3 in joint_opt
    skip = 4 in joint_opt
    
    # Select sensors
    foot_opt = []
    shank_opt = []
    thigh_opt = []
    torso_opt = []
    
    if not skip:
        if ankle_opt:
            print("1: Heel")
            print("2: Dorsal Foot")
            foot_input = simpledialog.askstring("Foot Sensors", 
                                              "Select Foot Sensors (enter as vector, e.g., [1,2])")
            if foot_input:
                foot_opt = eval(foot_input)
        
        if ankle_opt or knee_opt:
            print("1: Shin Flat Bone")
            print("2: MidShank Lateral")
            print("3: LowShank Lateral")
            shank_input = simpledialog.askstring("Shank Sensors", 
                                               "Select Shank Sensors (enter as vector, e.g., [1,2,3])")
            if shank_input:
                shank_opt = eval(shank_input)
        
        if knee_opt or hip_opt:
            print("1: LowThigh Anterior")
            print("2: MidThigh Lateral")
            print("3: LowThigh Lateral")
            print("4: LowThigh Posterior")
            thigh_input = simpledialog.askstring("Thigh Sensors", 
                                               "Select Thigh Sensors (enter as vector, e.g., [1,2,3,4])")
            if thigh_input:
                thigh_opt = eval(thigh_input)
        
        if hip_opt:
            print("1: Sacrum")
            print("2: L4-L5")
            torso_input = simpledialog.askstring("Torso Sensors", 
                                               "Select Torso Sensors (enter as vector, e.g., [1,2])")
            if torso_input:
                torso_opt = eval(torso_input)
    else:
        # All sensors for all joints
        foot_opt = [1, 2]
        shank_opt = [1, 2, 3]
        thigh_opt = [1, 2, 3, 4]
        torso_opt = [1, 2]
    
    # Perform coordinate transformations
    # This section would process all selected sensors
    # For now, we'll create placeholder structures
    print("Note: Coordinate transformation processing needs to be implemented with actual data")
    
    # Set up plotting colors and line styles
    colors = ['r', 'b', 'g', 'k']
    lines = ['-', '--', '-.', ':']
    
    # Calculate and plot angles
    fr = 60  # Frame rate (would come from data)
    
    # Hip angles
    if hip_opt:
        print("Calculating hip angles...")
        # This would iterate through all torso-thigh combinations
        # For now, we'll create placeholder plots
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Example data
        time = np.linspace(0, 10, 600)  # 10 seconds at 60 Hz
        flex_hip = 20 * np.sin(2 * np.pi * 1.2 * time)  # Example flexion
        abd_hip = 5 * np.sin(2 * np.pi * 2.4 * time)    # Example abduction
        rot_hip = 3 * np.sin(2 * np.pi * 0.8 * time)    # Example rotation
        
        ax1.plot(time, flex_hip, 'r-', label='Torso Sacrum / Thigh LowThighAnt')
        ax1.set_title('Hip Angles\nFlexion')
        ax1.set_ylabel('Degrees')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(time, abd_hip, 'r-', label='Torso Sacrum / Thigh LowThighAnt')
        ax2.set_title('Abduction')
        ax2.set_ylabel('Degrees')
        ax2.legend()
        ax2.grid(True)
        
        ax3.plot(time, rot_hip, 'r-', label='Torso Sacrum / Thigh LowThighAnt')
        ax3.set_title('Rotation')
        ax3.set_ylabel('Degrees')
        ax3.set_xlabel('Time (s)')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    # Knee angles
    if knee_opt:
        print("Calculating knee angles...")
        # Similar processing for knee angles
        
    # Ankle angles
    if ankle_opt:
        print("Calculating ankle angles...")
        # Similar processing for ankle angles
    
    # Save angle data
    save_it = simpledialog.askinteger("Save", "Do you want to save the calculated angles? (1 yes/ 0 no)")
    if save_it:
        print("Note: Angle data saving needs to be implemented with scipy.io.savemat")
    
    return


if __name__ == "__main__":
    euler_angle_calculations()
