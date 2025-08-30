# IMU Sensor Placement Optimization - Python Implementation

This is a Python conversion of the original MATLAB codebase for IMU sensor placement optimization, originally developed by W. Niswander, W. Wang, and K. Kontson.

## Overview

This tool implements a complete workflow for analyzing IMU (Inertial Measurement Unit) sensor data to optimize sensor placement for measuring lower limb joint kinematics. The system compares IMU-derived joint angles with optical motion capture reference data to determine optimal sensor configurations.

## Features

- **Data Import**: Import IMU data from Xsens text files
- **Sensor Calibration**: Two calibration methods (functional and static)
- **Joint Angle Calculation**: Euler angle decomposition using YXZ rotation order
- **Comparison Analysis**: Compare IMU results with optical motion capture reference
- **Interactive GUI**: User-friendly interface for data processing
- **Command Line Interface**: Alternative CLI for batch processing

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Required Packages

- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `matplotlib>=3.5.0` - Plotting and visualization
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - Machine learning (for PCA)
- `tkinter-tooltip>=2.0.0` - GUI tooltips

## Usage

### GUI Interface (Recommended)

Start the graphical user interface:

```bash
python main.py
```

### Command Line Interface

For command-line usage:

```bash
python main.py --cli
```

### Help

Display usage information:

```bash
python main.py --help
```

## Workflow

### Step 1: Data Import (`imu_import.py`)

**Purpose**: Import IMU data from Xsens text files into a structured format.

**Process**:
1. Load subject information file (P_xxx_Info.mat)
2. Select folder containing IMU text files
3. Parse and organize data by sensor and task
4. Save imported data for further processing

**Key Functions**:
- `imu_import()`: Core import function
- `data_import()`: Main import workflow

### Step 2: Calibration (`calibration.py`)

**Purpose**: Calibrate IMU sensor axes to anatomical coordinate systems.

**Two Calibration Methods**:

#### Functional Calibration
- Uses walking data to determine medial-lateral axis via PCA
- Superior-inferior axis from gravitational acceleration during standing
- Anterior-posterior axis from cross products

#### Static Calibration
- Uses two static poses (standing and seated)
- Superior-inferior axis from gravitational acceleration
- Medial-lateral axis from cross product of standing and seated vectors
- Anterior-posterior axis from cross products

**Key Functions**:
- `axis_estimation()`: Functional calibration
- `axis_estimation_stationary()`: Static calibration
- `stationary_calibration_ranges()`: Define calibration time ranges
- `zero_velocity_finder()`: Find flat foot instances
- `angle_corrected()`: Drift correction
- `defining_steady_state()`: Define steady state gait regions

### Step 3: Euler Angle Calculations (`euler_angle_calculations.py`)

**Purpose**: Calculate joint angles from calibrated IMU data.

**Process**:
1. Load calibrated sensor data
2. Transform coordinate systems using quaternions
3. Calculate Euler angles using YXZ rotation order
4. Generate plots for different sensor combinations

**Key Functions**:
- `coordinate_transform_quat_pre_specified_axes()`: Quaternion-based coordinate transformation
- `angles_yxz()`: YXZ Euler angle calculation
- `dcm_projection()`: Direction cosine matrix projection
- `quaternion_multiplication()`: Quaternion arithmetic
- `quaternion_conjugation()`: Vector rotation via quaternions

### Step 4: Comparisons Tool (`comparisons_tool.py`)

**Purpose**: Compare IMU-derived angles with optical motion capture reference.

**Process**:
1. Load IMU and optical motion capture data
2. Resample data to common frame rate
3. Calculate correlation coefficients and RMS errors
4. Generate comparison plots

**Key Functions**:
- `resampling()`: Resample data to different sampling frequencies
- `comparisons_tool()`: Main comparison workflow

## File Structure

```
IMU-Sensor-Placement-Optimization/
├── main.py                          # Main application entry point
├── imu_import.py                    # Data import functionality
├── calibration.py                   # Sensor calibration
├── euler_angle_calculations.py      # Joint angle calculations
├── comparisons_tool.py              # Comparison with optical data
├── requirements.txt                 # Python dependencies
├── README_Python.md                 # This file
└── README.md                        # Original MATLAB documentation
```

## Data Format

### Input Data

- **IMU Text Files**: Xsens MTw sensor data in tab-separated format
- **Subject Information**: MATLAB .mat files containing sensor configuration
- **Calibration Data**: MATLAB .mat files with calibration parameters

### Output Data

- **Imported Data**: Structured arrays containing sensor data
- **Calibration Files**: Anatomical axis definitions for each sensor
- **Angle Data**: Joint angle time series
- **Comparison Results**: Correlation coefficients and RMS errors

## Key Algorithms

### Principal Component Analysis (PCA)
Used in functional calibration to identify the primary axis of rotation during gait.

### Quaternion Operations
- Multiplication for combining rotations
- Conjugation for vector rotation
- Coordinate system transformations

### Euler Angle Decomposition
YXZ rotation order for joint angle calculation:
1. Y-axis rotation (flexion/extension)
2. X-axis rotation (abduction/adduction)
3. Z-axis rotation (internal/external rotation)

### Drift Correction
Linear drift correction between flat foot instances during gait analysis.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **File Format Issues**
   - Ensure IMU text files are tab-separated
   - Check that subject information files are valid MATLAB .mat format

3. **GUI Issues**
   - Use command line interface: `python main.py --cli`
   - Check tkinter installation

4. **Data Loading Issues**
   - Verify file paths and permissions
   - Check file format compatibility

### Performance Notes

- Large datasets may require significant memory
- Consider processing data in chunks for very long recordings
- Use command line interface for batch processing

## Differences from MATLAB Version

### Advantages of Python Implementation

1. **Open Source**: No license required
2. **Cross-Platform**: Runs on Windows, macOS, and Linux
3. **Modern Libraries**: Uses scikit-learn for PCA, scipy for signal processing
4. **Better Error Handling**: More robust error messages and recovery
5. **Extensible**: Easy to add new features and algorithms

### Implementation Notes

- Uses `scipy.io` for MATLAB file compatibility
- `pandas` for data import and manipulation
- `matplotlib` for plotting (equivalent to MATLAB plotting)
- `scikit-learn` for PCA (equivalent to MATLAB's pca function)

## Contributing

To extend or modify this implementation:

1. Follow the existing code structure
2. Add type hints for better code documentation
3. Include docstrings for all functions
4. Test with sample data before deployment

## Citation

If you use this Python implementation, please cite the original MATLAB work:

```
Niswander, W., Wang, W., & Kontson, K. (2020). Optimization of IMU Sensor Placement 
for the Measurement of Lower Limb Joint Kinematics. Sensors, 20(24), 7153.
```

## License

This Python implementation is provided as-is for research and educational purposes. Please refer to the original MATLAB codebase for licensing information.

## Support

For issues specific to this Python implementation, please check the troubleshooting section above. For questions about the underlying algorithms, refer to the original MATLAB documentation and research paper.
