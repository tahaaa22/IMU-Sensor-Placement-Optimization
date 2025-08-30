import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import tkinter as tk
from tkinter import filedialog, simpledialog
import scipy.io as sio
import os
from scipy import interpolate


def resampling(original_data: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    """
    Python equivalent of resampling.m
    
    Resamples data to a different sampling frequency.
    
    Args:
        original_data: Original time series data
        original_fs: Original sampling frequency
        target_fs: Target sampling frequency
    
    Returns:
        Resampled data
    """
    # Calculate time vectors
    original_time = np.arange(len(original_data)) / original_fs
    target_time = np.arange(0, original_time[-1], 1/target_fs)
    
    # Use scipy interpolation for resampling
    if original_data.ndim == 1:
        f_interp = interpolate.interp1d(original_time, original_data, 
                                       kind='linear', bounds_error=False, fill_value='extrapolate')
        resampled_data = f_interp(target_time)
    else:
        # Handle multi-dimensional data
        resampled_data = np.zeros((len(target_time), original_data.shape[1]))
        for i in range(original_data.shape[1]):
            f_interp = interpolate.interp1d(original_time, original_data[:, i], 
                                           kind='linear', bounds_error=False, fill_value='extrapolate')
            resampled_data[:, i] = f_interp(target_time)
    
    return resampled_data


def comparisons_tool():
    """
    Main function for comparing IMU angles to optical motion capture data.
    Python equivalent of Comparisons_Tool.m
    """
    # Set up workspace
    root = tk.Tk()
    root.withdraw()
    
    # Load Optical Motion Capture Data
    print("Load Optical Motion Capture Data")
    optical_exists = 'optical_data' in globals()
    if optical_exists:
        load_optical = simpledialog.askinteger("Optical Data", 
                                             "Do you want to load new optical data? Yes or no (1/0)?")
    else:
        load_optical = 1
    
    if load_optical:
        optical_file = filedialog.askopenfilename(
            title="Select optical motion capture data file",
            filetypes=[("MAT files", "*.mat"), ("All files", "*.*")]
        )
        
        if optical_file:
            # Load optical data (in real implementation, use scipy.io.loadmat)
            print("Note: Optical data loading needs to be implemented with scipy.io.loadmat")
            # optical_data = sio.loadmat(optical_file)
    
    # Select task from optical data
    print("Available optical tasks:")
    print("1: Walking")
    print("2: Running")
    print("3: Jumping")
    
    optical_task = simpledialog.askinteger("Optical Task", "Select optical task")
    if optical_task is None:
        return
    
    # Load IMU angle data
    print("Load IMU angle data")
    imu_exists = 'imu_data' in globals()
    if imu_exists:
        load_imu = simpledialog.askinteger("IMU Data", 
                                         "Do you want to load new IMU data? Yes or no (1/0)?")
    else:
        load_imu = 1
    
    if load_imu:
        imu_file = filedialog.askopenfilename(
            title="Select IMU angle data file",
            filetypes=[("MAT files", "*.mat"), ("All files", "*.*")]
        )
        
        if imu_file:
            # Load IMU data (in real implementation, use scipy.io.loadmat)
            print("Note: IMU data loading needs to be implemented with scipy.io.loadmat")
            # imu_data = sio.loadmat(imu_file)
    
    # Get frame rates
    print("Enter frame rates:")
    imu_fps = simpledialog.askfloat("Frame Rate", "IMU frame rate (FPS):")
    optical_fps = simpledialog.askfloat("Frame Rate", "Optical frame rate (FPS):")
    
    if imu_fps is None or optical_fps is None:
        return
    
    # Ask about offset
    offset_opt = simpledialog.askinteger("Offset", 
                                        "Do you want to offset IMU data to match optical data on frame 1? (1 yes/ 0 no)")
    
    # For demonstration, we'll create example data
    # In real implementation, this would come from the loaded files
    
    # Example data generation
    time_imu = np.linspace(0, 10, int(imu_fps * 10))  # 10 seconds
    time_optical = np.linspace(0, 10, int(optical_fps * 10))
    
    # Generate example IMU angles
    imu_hip_flex = 25 * np.sin(2 * np.pi * 1.2 * time_imu) + 5 * np.random.randn(len(time_imu))
    imu_hip_abd = 8 * np.sin(2 * np.pi * 2.4 * time_imu) + 2 * np.random.randn(len(time_imu))
    imu_hip_rot = 3 * np.sin(2 * np.pi * 0.8 * time_imu) + 1 * np.random.randn(len(time_imu))
    
    imu_knee_flex = 60 * np.sin(2 * np.pi * 1.2 * time_imu) + 10 * np.random.randn(len(time_imu))
    imu_knee_abd = 2 * np.sin(2 * np.pi * 2.4 * time_imu) + 1 * np.random.randn(len(time_imu))
    imu_knee_rot = 1 * np.sin(2 * np.pi * 0.8 * time_imu) + 0.5 * np.random.randn(len(time_imu))
    
    imu_ankle_flex = 15 * np.sin(2 * np.pi * 1.2 * time_imu) + 3 * np.random.randn(len(time_imu))
    imu_ankle_abd = 1 * np.sin(2 * np.pi * 2.4 * time_imu) + 0.5 * np.random.randn(len(time_imu))
    imu_ankle_rot = 0.5 * np.sin(2 * np.pi * 0.8 * time_imu) + 0.2 * np.random.randn(len(time_imu))
    
    # Generate example optical angles (slightly different for comparison)
    optical_hip_flex = 24 * np.sin(2 * np.pi * 1.2 * time_optical) + 3 * np.random.randn(len(time_optical))
    optical_hip_abd = 7.5 * np.sin(2 * np.pi * 2.4 * time_optical) + 1.5 * np.random.randn(len(time_optical))
    optical_hip_rot = 2.8 * np.sin(2 * np.pi * 0.8 * time_optical) + 0.8 * np.random.randn(len(time_optical))
    
    optical_knee_flex = 58 * np.sin(2 * np.pi * 1.2 * time_optical) + 8 * np.random.randn(len(time_optical))
    optical_knee_abd = 1.8 * np.sin(2 * np.pi * 2.4 * time_optical) + 0.8 * np.random.randn(len(time_optical))
    optical_knee_rot = 0.9 * np.sin(2 * np.pi * 0.8 * time_optical) + 0.4 * np.random.randn(len(time_optical))
    
    optical_ankle_flex = 14.5 * np.sin(2 * np.pi * 1.2 * time_optical) + 2.5 * np.random.randn(len(time_optical))
    optical_ankle_abd = 0.9 * np.sin(2 * np.pi * 2.4 * time_optical) + 0.4 * np.random.randn(len(time_optical))
    optical_ankle_rot = 0.45 * np.sin(2 * np.pi * 0.8 * time_optical) + 0.15 * np.random.randn(len(time_optical))
    
    # Apply offset if requested
    if offset_opt:
        # Offset IMU data to match optical data at frame 1
        imu_hip_flex += (optical_hip_flex[0] - imu_hip_flex[0])
        imu_hip_abd += (optical_hip_abd[0] - imu_hip_abd[0])
        imu_hip_rot += (optical_hip_rot[0] - imu_hip_rot[0])
        
        imu_knee_flex += (optical_knee_flex[0] - imu_knee_flex[0])
        imu_knee_abd += (optical_knee_abd[0] - imu_knee_abd[0])
        imu_knee_rot += (optical_knee_rot[0] - imu_knee_rot[0])
        
        imu_ankle_flex += (optical_ankle_flex[0] - imu_ankle_flex[0])
        imu_ankle_abd += (optical_ankle_abd[0] - imu_ankle_abd[0])
        imu_ankle_rot += (optical_ankle_rot[0] - imu_ankle_rot[0])
    
    # Resample IMU data to match optical frame rate
    imu_hip_flex_resampled = resampling(imu_hip_flex, imu_fps, optical_fps)
    imu_hip_abd_resampled = resampling(imu_hip_abd, imu_fps, optical_fps)
    imu_hip_rot_resampled = resampling(imu_hip_rot, imu_fps, optical_fps)
    
    imu_knee_flex_resampled = resampling(imu_knee_flex, imu_fps, optical_fps)
    imu_knee_abd_resampled = resampling(imu_knee_abd, imu_fps, optical_fps)
    imu_knee_rot_resampled = resampling(imu_knee_rot, imu_fps, optical_fps)
    
    imu_ankle_flex_resampled = resampling(imu_ankle_flex, imu_fps, optical_fps)
    imu_ankle_abd_resampled = resampling(imu_ankle_abd, imu_fps, optical_fps)
    imu_ankle_rot_resampled = resampling(imu_ankle_rot, imu_fps, optical_fps)
    
    # Create comparison plots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Hip angles
    axes[0, 0].plot(time_optical, optical_hip_flex, 'b-', label='Optical', linewidth=2)
    axes[0, 0].plot(time_optical, imu_hip_flex_resampled, 'r--', label='IMU', linewidth=2)
    axes[0, 0].set_title('Hip Flexion/Extension')
    axes[0, 0].set_ylabel('Degrees')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(time_optical, optical_hip_abd, 'b-', label='Optical', linewidth=2)
    axes[0, 1].plot(time_optical, imu_hip_abd_resampled, 'r--', label='IMU', linewidth=2)
    axes[0, 1].set_title('Hip Abduction/Adduction')
    axes[0, 1].set_ylabel('Degrees')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(time_optical, optical_hip_rot, 'b-', label='Optical', linewidth=2)
    axes[0, 2].plot(time_optical, imu_hip_rot_resampled, 'r--', label='IMU', linewidth=2)
    axes[0, 2].set_title('Hip Internal/External Rotation')
    axes[0, 2].set_ylabel('Degrees')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Knee angles
    axes[1, 0].plot(time_optical, optical_knee_flex, 'b-', label='Optical', linewidth=2)
    axes[1, 0].plot(time_optical, imu_knee_flex_resampled, 'r--', label='IMU', linewidth=2)
    axes[1, 0].set_title('Knee Flexion/Extension')
    axes[1, 0].set_ylabel('Degrees')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(time_optical, optical_knee_abd, 'b-', label='Optical', linewidth=2)
    axes[1, 1].plot(time_optical, imu_knee_abd_resampled, 'r--', label='IMU', linewidth=2)
    axes[1, 1].set_title('Knee Abduction/Adduction')
    axes[1, 1].set_ylabel('Degrees')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    axes[1, 2].plot(time_optical, optical_knee_rot, 'b-', label='Optical', linewidth=2)
    axes[1, 2].plot(time_optical, imu_knee_rot_resampled, 'r--', label='IMU', linewidth=2)
    axes[1, 2].set_title('Knee Internal/External Rotation')
    axes[1, 2].set_ylabel('Degrees')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    # Ankle angles
    axes[2, 0].plot(time_optical, optical_ankle_flex, 'b-', label='Optical', linewidth=2)
    axes[2, 0].plot(time_optical, imu_ankle_flex_resampled, 'r--', label='IMU', linewidth=2)
    axes[2, 0].set_title('Ankle Dorsiflexion/Plantarflexion')
    axes[2, 0].set_ylabel('Degrees')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    axes[2, 1].plot(time_optical, optical_ankle_abd, 'b-', label='Optical', linewidth=2)
    axes[2, 1].plot(time_optical, imu_ankle_abd_resampled, 'r--', label='IMU', linewidth=2)
    axes[2, 1].set_title('Ankle Inversion/Eversion')
    axes[2, 1].set_ylabel('Degrees')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    axes[2, 2].plot(time_optical, optical_ankle_rot, 'b-', label='Optical', linewidth=2)
    axes[2, 2].plot(time_optical, imu_ankle_rot_resampled, 'r--', label='IMU', linewidth=2)
    axes[2, 2].set_title('Ankle Internal/External Rotation')
    axes[2, 2].set_ylabel('Degrees')
    axes[2, 2].set_xlabel('Time (s)')
    axes[2, 2].legend()
    axes[2, 2].grid(True)
    
    plt.tight_layout()
    plt.suptitle('IMU vs Optical Motion Capture Comparison', fontsize=16, y=0.98)
    plt.show()
    
    # Calculate correlation coefficients
    print("\nCorrelation Analysis:")
    print("=" * 50)
    
    # Hip correlations
    hip_flex_corr = np.corrcoef(optical_hip_flex, imu_hip_flex_resampled)[0, 1]
    hip_abd_corr = np.corrcoef(optical_hip_abd, imu_hip_abd_resampled)[0, 1]
    hip_rot_corr = np.corrcoef(optical_hip_rot, imu_hip_rot_resampled)[0, 1]
    
    print(f"Hip Flexion/Extension: r = {hip_flex_corr:.3f}")
    print(f"Hip Abduction/Adduction: r = {hip_abd_corr:.3f}")
    print(f"Hip Internal/External Rotation: r = {hip_rot_corr:.3f}")
    
    # Knee correlations
    knee_flex_corr = np.corrcoef(optical_knee_flex, imu_knee_flex_resampled)[0, 1]
    knee_abd_corr = np.corrcoef(optical_knee_abd, imu_knee_abd_resampled)[0, 1]
    knee_rot_corr = np.corrcoef(optical_knee_rot, imu_knee_rot_resampled)[0, 1]
    
    print(f"\nKnee Flexion/Extension: r = {knee_flex_corr:.3f}")
    print(f"Knee Abduction/Adduction: r = {knee_abd_corr:.3f}")
    print(f"Knee Internal/External Rotation: r = {knee_rot_corr:.3f}")
    
    # Ankle correlations
    ankle_flex_corr = np.corrcoef(optical_ankle_flex, imu_ankle_flex_resampled)[0, 1]
    ankle_abd_corr = np.corrcoef(optical_ankle_abd, imu_ankle_abd_resampled)[0, 1]
    ankle_rot_corr = np.corrcoef(optical_ankle_rot, imu_ankle_rot_resampled)[0, 1]
    
    print(f"\nAnkle Dorsiflexion/Plantarflexion: r = {ankle_flex_corr:.3f}")
    print(f"Ankle Inversion/Eversion: r = {ankle_abd_corr:.3f}")
    print(f"Ankle Internal/External Rotation: r = {ankle_rot_corr:.3f}")
    
    # Calculate RMS errors
    print("\nRMS Error Analysis:")
    print("=" * 50)
    
    # Hip RMS errors
    hip_flex_rms = np.sqrt(np.mean((optical_hip_flex - imu_hip_flex_resampled)**2))
    hip_abd_rms = np.sqrt(np.mean((optical_hip_abd - imu_hip_abd_resampled)**2))
    hip_rot_rms = np.sqrt(np.mean((optical_hip_rot - imu_hip_rot_resampled)**2))
    
    print(f"Hip Flexion/Extension: RMS = {hip_flex_rms:.2f}°")
    print(f"Hip Abduction/Adduction: RMS = {hip_abd_rms:.2f}°")
    print(f"Hip Internal/External Rotation: RMS = {hip_rot_rms:.2f}°")
    
    # Knee RMS errors
    knee_flex_rms = np.sqrt(np.mean((optical_knee_flex - imu_knee_flex_resampled)**2))
    knee_abd_rms = np.sqrt(np.mean((optical_knee_abd - imu_knee_abd_resampled)**2))
    knee_rot_rms = np.sqrt(np.mean((optical_knee_rot - imu_knee_rot_resampled)**2))
    
    print(f"\nKnee Flexion/Extension: RMS = {knee_flex_rms:.2f}°")
    print(f"Knee Abduction/Adduction: RMS = {knee_abd_rms:.2f}°")
    print(f"Knee Internal/External Rotation: RMS = {knee_rot_rms:.2f}°")
    
    # Ankle RMS errors
    ankle_flex_rms = np.sqrt(np.mean((optical_ankle_flex - imu_ankle_flex_resampled)**2))
    ankle_abd_rms = np.sqrt(np.mean((optical_ankle_abd - imu_ankle_abd_resampled)**2))
    ankle_rot_rms = np.sqrt(np.mean((optical_ankle_rot - imu_ankle_rot_resampled)**2))
    
    print(f"\nAnkle Dorsiflexion/Plantarflexion: RMS = {ankle_flex_rms:.2f}°")
    print(f"Ankle Inversion/Eversion: RMS = {ankle_abd_rms:.2f}°")
    print(f"Ankle Internal/External Rotation: RMS = {ankle_rot_rms:.2f}°")
    
    return


if __name__ == "__main__":
    comparisons_tool()
