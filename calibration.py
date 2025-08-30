import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import tkinter as tk
from tkinter import filedialog, simpledialog
import scipy.io as sio
import os


def axis_estimation(acc: np.ndarray, rot: np.ndarray, grav_span: List[int], 
                   steady_span: List[int], guess: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Python equivalent of Axis_Estimation.m
    
    Uses gravitational acceleration vector while subject is stationary to determine SI direction
    and principal component analysis (PCA) of rotational velocity to estimate ML direction.
    
    Args:
        acc: Acceleration signal
        rot: Rotational velocity signal  
        grav_span: Span of indices when subject is standing still
        steady_span: Span of indices when subject is walking at steady pace
        guess: Initial guess of ML vector direction
    
    Returns:
        SI: Unit vector aligned with SI direction (superior is positive)
        ML: Unit vector aligned with ML direction (lateral is positive) 
        AP: Unit vector aligned with AP direction (anterior is positive)
    """
    
    # Finding Superior Inferior Axis Through Gravity
    gravity_start = int(grav_span[0])
    gravity_end = int(grav_span[1])
    
    gravity = np.mean(acc[gravity_start:gravity_end+1, :], axis=0)
    si = gravity / np.linalg.norm(gravity)
    
    # Finding Medial Lateral through Principal Component Analysis (PCA)
    # Apply low pass moving average filter
    rot_filt = rot.copy()
    filter_weights = np.array([1, 2, 3, 2, 1])
    filter_span = (len(filter_weights) - 1) // 2
    filter_mean = np.sum(filter_weights)
    
    for zz in range(filter_span, len(rot) - filter_span):
        rot_filt[zz] = np.sum(rot[zz-filter_span:zz+filter_span+1] * filter_weights) / filter_mean
    
    rot = rot_filt
    
    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(rot[steady_span[0]:steady_span[1]+1, :])
    pca_coef = pca.components_.T
    
    # Check direction of first PCA component
    tolerance = 0.5
    if (np.all(np.abs(guess - pca_coef[:, 0]) < tolerance)):
        ml_f = pca_coef[:, 0]
    elif (np.all(np.abs(-1 * guess - pca_coef[:, 0]) < tolerance)):
        ml_f = -1 * pca_coef[:, 0]
    else:
        ml_f = pca_coef[:, 0]
        print("Warning: PCA axis does not align well with guess, assuming correct direction!")
    
    # Normalize and ensure orthogonality
    ml_f = ml_f / np.linalg.norm(ml_f)
    ap = np.cross(si, ml_f)
    ap = ap / np.linalg.norm(ap)
    
    ml = ml_f
    
    return si, ml, ap


def axis_estimation_si_only(acc: np.ndarray, grav_span: List[int]) -> np.ndarray:
    """
    Python equivalent of Axis_Estimation_SI_Only.m
    
    Only estimates the SI axis using gravitational acceleration.
    
    Args:
        acc: Acceleration signal
        grav_span: Span of indices when subject is standing still
    
    Returns:
        SI: Unit vector aligned with SI direction
    """
    gravity_start = int(grav_span[0])
    gravity_end = int(grav_span[1])
    
    gravity = np.mean(acc[gravity_start:gravity_end+1, :], axis=0)
    si = gravity / np.linalg.norm(gravity)
    
    return si


def axis_estimation_stationary(acc: np.ndarray, range_stand: List[int], 
                              range_sit: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Python equivalent of Axis_Estimation_stationary.m
    
    Uses two static poses to estimate anatomical axes.
    
    Args:
        acc: Acceleration signal
        range_stand: Span when subject is standing still
        range_sit: Span when subject is seated
    
    Returns:
        SI, ML, AP: Unit vectors for anatomical directions
    """
    # SI axis from standing pose
    stand_start = int(range_stand[0])
    stand_end = int(range_stand[1])
    gravity_stand = np.mean(acc[stand_start:stand_end+1, :], axis=0)
    si = gravity_stand / np.linalg.norm(gravity_stand)
    
    # Co-planar vector from seated pose
    sit_start = int(range_sit[0])
    sit_end = int(range_sit[1])
    gravity_sit = np.mean(acc[sit_start:sit_end+1, :], axis=0)
    coplanar = gravity_sit / np.linalg.norm(gravity_sit)
    
    # ML axis from cross product
    ml = np.cross(si, coplanar)
    ml = ml / np.linalg.norm(ml)
    
    # AP axis from cross product
    ap = np.cross(ml, si)
    ap = ap / np.linalg.norm(ap)
    
    # Ensure orthogonality
    ml = np.cross(si, ap)
    ml = ml / np.linalg.norm(ml)
    ap = np.cross(ml, si)
    ap = ap / np.linalg.norm(ap)
    
    return si, ml, ap


def stationary_calibration_ranges(acc_heel: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Python equivalent of StationaryCalibrationRanges.m
    
    Interactive function to define ranges of still standing and still sitting.
    
    Args:
        acc_heel: Heel acceleration data
    
    Returns:
        range_stand: Range of indices for standing still
        range_sit: Range of indices for sitting still
    """
    plt.figure(figsize=(12, 8))
    plt.plot(acc_heel)
    plt.title('Heel Acceleration - Click to define ranges')
    plt.xlabel('Frame')
    plt.ylabel('Acceleration')
    plt.grid(True)
    
    print("Click on the starting and ending points for still standing")
    points_stand = plt.ginput(2, timeout=0)
    range_stand = [int(points_stand[0][0]), int(points_stand[1][0])]
    
    print("Click on the starting and ending points for still sitting (or press Enter to skip)")
    try:
        points_sit = plt.ginput(2, timeout=10)
        range_sit = [int(points_sit[0][0]), int(points_sit[1][0])]
    except:
        range_sit = [0, 0]  # Skip sitting calibration
    
    plt.close()
    
    return range_stand, range_sit


def zero_velocity_finder(acc_net: np.ndarray, rot_net: np.ndarray, 
                        range_stand: List[int], range_sit: List[int]) -> Tuple[List[int], int]:
    """
    Python equivalent of ZeroVelocityFinder.m
    
    Finds instances of flat foot during gait.
    
    Args:
        acc_net: Net acceleration across all sensors
        rot_net: Net rotational velocity across all sensors
        range_stand: Range for standing still
        range_sit: Range for sitting still
    
    Returns:
        instances_initial: Initial estimates of flat foot instances
        threshold: Threshold for coincident points
    """
    from scipy.signal import find_peaks
    
    # Find minima in rotational velocity
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(rot_net)
    plt.title('Bulk Rotational Velocity - Click below threshold for minima detection')
    plt.ylabel('Rotational Velocity')
    
    print("Click below the threshold for rotational velocity minima detection")
    threshold_rot = plt.ginput(1, timeout=0)[0][1]
    
    # Find minima
    minima_rot, _ = find_peaks(-rot_net, height=-threshold_rot)
    
    # Find minima in acceleration
    plt.subplot(2, 1, 2)
    plt.plot(acc_net)
    plt.title('Bulk Acceleration - Click below threshold for minima detection')
    plt.ylabel('Acceleration')
    
    print("Click below the threshold for acceleration minima detection")
    threshold_acc = plt.ginput(1, timeout=0)[0][1]
    
    # Find minima
    minima_acc, _ = find_peaks(-acc_net, height=-threshold_acc)
    
    plt.close()
    
    # Find coincident points
    print("Enter threshold (in frames) for treating points as coincident:")
    threshold = int(input())
    
    instances_initial = []
    for rot_min in minima_rot:
        for acc_min in minima_acc:
            if abs(rot_min - acc_min) <= threshold:
                instances_initial.append(rot_min)
                break
    
    return instances_initial, threshold


def angle_corrected(rot_heel: np.ndarray, instances_initial: List[int], 
                   threshold: int) -> Tuple[np.ndarray, List[int]]:
    """
    Python equivalent of Angle_Corrected.m
    
    Generates drift-corrected angle estimate using integrated rotational velocity.
    
    Args:
        rot_heel: Heel rotational velocity
        instances_initial: Initial flat foot instances
        threshold: Threshold for coincident points
    
    Returns:
        heel: Drift-corrected angle estimate
        instances: Refined flat foot instances
    """
    # Integrate rotational velocity
    angle_raw = np.cumsum(rot_heel, axis=0)
    
    # Apply linear drift correction between flat foot points
    instances = instances_initial.copy()
    heel = angle_raw.copy()
    
    for i in range(1, len(instances)):
        start_idx = instances[i-1]
        end_idx = instances[i]
        
        # Linear drift correction
        drift = angle_raw[end_idx] - angle_raw[start_idx]
        frames = end_idx - start_idx
        
        for j in range(start_idx, end_idx + 1):
            correction = drift * (j - start_idx) / frames
            heel[j] = angle_raw[j] - correction
    
    # Refine flat foot points by finding minimal slope
    instances_refined = []
    for instance in instances:
        # Look for minimal slope near the initial guess
        window = 10
        start_search = max(0, instance - window)
        end_search = min(len(heel), instance + window)
        
        slopes = np.abs(np.diff(heel[start_search:end_search+1], axis=0))
        min_slope_idx = np.argmin(slopes)
        refined_instance = start_search + min_slope_idx
        instances_refined.append(refined_instance)
    
    return heel, instances_refined


def defining_steady_state(heel: np.ndarray, instances: List[int]) -> List[List[int]]:
    """
    Python equivalent of DefiningSteadyState.m
    
    Defines regions of steady state gait.
    
    Args:
        heel: Drift-corrected heel angle
        instances: Flat foot instances
    
    Returns:
        range_steady: List of steady state ranges
    """
    plt.figure(figsize=(12, 8))
    plt.plot(heel)
    plt.plot(instances, heel[instances], 'ro', label='Flat foot points')
    plt.title('Corrected Heel Angle - Define steady state regions')
    plt.xlabel('Frame')
    plt.ylabel('Angle')
    plt.legend()
    plt.grid(True)
    
    print(f"Numbered flat foot points: {list(range(1, len(instances)+1))}")
    print("Enter number of steady state regions:")
    num_regions = int(input())
    
    range_steady = []
    for i in range(num_regions):
        print(f"Enter start and end point numbers for region {i+1} (e.g., 2 7):")
        start_end = input().split()
        start_idx = instances[int(start_end[0]) - 1]
        end_idx = instances[int(start_end[1]) - 1]
        range_steady.append([start_idx, end_idx])
    
    plt.close()
    
    return range_steady


def calibration():
    """
    Main calibration function - Python equivalent of Calibration.m
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
            # data = sio.loadmat(data_file)['data']
    
    # Select task
    task_list_exists = 'TaskList' in globals()
    if task_list_exists:
        load_sub = simpledialog.askinteger("Subject Info", "Do you want to use the same subject information? Yes or no (1/0)?")
    else:
        load_sub = 0
    
    if load_sub == 0:
        print("Load subject information ('P_xxx_Info.mat')")
        subject_info_file = filedialog.askopenfilename(
            title="Select subject information file",
            filetypes=[("MAT files", "*.mat"), ("All files", "*.*")]
        )
        
        if subject_info_file:
            # Load subject info (in real implementation, use scipy.io.loadmat)
            print("Note: Subject information loading needs to be implemented with scipy.io.loadmat")
    
    # Display task list and get user selection
    print("Available tasks:")
    # In real implementation, this would come from the loaded data
    print("1: Calibration")
    print("2: Walking")
    print("3: Running")
    
    j = simpledialog.askinteger("Task Selection", "Select Task")
    if j is None:
        return
    
    # Calculate index
    number_sensors = 11  # This would come from loaded data
    ind = number_sensors * (j - 1)
    
    # Create acceleration and rotational velocity variables
    # This section would extract data from the loaded structure
    # For now, we'll use placeholder data
    print("Note: Data extraction from loaded structure needs to be implemented")
    
    # Select calibration approaches
    print("Select which calibration(s) you would like to perform:")
    print("1: Functional Calibration")
    print("2: Static Calibration")
    
    functional_true = simpledialog.askinteger("Functional Calibration", 
                                            "Perform functional Calibration? (1 for yes, 0 for no)")
    static_true = simpledialog.askinteger("Static Calibration", 
                                         "Perform Static Calibration? (1 for yes, 0 for no)")
    
    # Define ranges
    print("Defining ranges of still standing, still sitting, and steady state gait...")
    
    # This would use actual acceleration data
    acc_heel = np.random.randn(1000, 3)  # Placeholder data
    
    range_stand, range_sit = stationary_calibration_ranges(acc_heel)
    
    if functional_true:
        # Zero velocity finder
        acc_net = np.random.randn(1000)  # Placeholder
        rot_net = np.random.randn(1000)  # Placeholder
        
        instances_initial, threshold = zero_velocity_finder(acc_net, rot_net, range_stand, range_sit)
        
        # Angle correction
        rot_heel = np.random.randn(1000, 3)  # Placeholder
        heel, instances = angle_corrected(rot_heel, instances_initial, threshold)
        
        # Define steady state
        range_steady = defining_steady_state(heel, instances)
    
    # Perform functional calibration
    if functional_true:
        print("Performing functional calibration...")
        
        # This would iterate through all sensors and perform axis estimation
        # For now, we'll create a placeholder structure
        zero = {}
        
        # Example for one sensor
        acc_sensor = np.random.randn(1000, 3)  # Placeholder
        rot_sensor = np.random.randn(1000, 3)  # Placeholder
        guess = np.array([0, 1, 0])  # Placeholder
        
        si, ml, ap = axis_estimation(acc_sensor, rot_sensor, range_stand, range_steady[0], guess)
        zero['LowThighAnt'] = np.vstack([si, ml, ap])
        
        # Save functional calibration
        print("Save the functional calibration file")
        save_file = filedialog.asksaveasfilename(
            title="Save functional calibration",
            defaultextension=".mat",
            filetypes=[("MAT files", "*.mat")]
        )
        
        if save_file:
            # In real implementation, use scipy.io.savemat
            print(f"Functional calibration would be saved as {save_file}")
    
    # Perform static calibration
    if static_true:
        print("Performing static calibration...")
        
        # This would iterate through all sensors and perform static axis estimation
        zero = {}
        
        # Example for one sensor
        acc_sensor = np.random.randn(1000, 3)  # Placeholder
        
        si, ml, ap = axis_estimation_stationary(acc_sensor, range_stand, range_sit)
        zero['LowThighAnt'] = np.vstack([si, ml, ap])
        
        # Save static calibration
        print("Save the static calibration file")
        save_file = filedialog.asksaveasfilename(
            title="Save static calibration",
            defaultextension=".mat",
            filetypes=[("MAT files", "*.mat")]
        )
        
        if save_file:
            # In real implementation, use scipy.io.savemat
            print(f"Static calibration would be saved as {save_file}")
    
    return zero


if __name__ == "__main__":
    calibration()
