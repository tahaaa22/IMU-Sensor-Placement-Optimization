#!/usr/bin/env python3
"""
IMU Sensor Placement Optimization - Python Implementation

Main script for the complete IMU sensor placement optimization workflow.
This is a Python conversion of the original MATLAB codebase.

Author: Python Conversion
Original MATLAB Authors: W. Niswander, W. Wang, and K. Kontson
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox, simpledialog

# Import our modules
from imu_import import data_import
from calibration import calibration
from euler_angle_calculations import euler_angle_calculations
from comparisons_tool import comparisons_tool


def main_menu():
    """
    Main menu interface for the IMU Sensor Placement Optimization tool.
    """
    root = tk.Tk()
    root.title("IMU Sensor Placement Optimization")
    root.geometry("600x400")
    
    # Configure style
    root.configure(bg='#f0f0f0')
    
    # Title
    title_label = tk.Label(root, text="IMU Sensor Placement Optimization", 
                          font=("Arial", 16, "bold"), bg='#f0f0f0')
    title_label.pack(pady=20)
    
    subtitle_label = tk.Label(root, text="Python Implementation", 
                             font=("Arial", 12), bg='#f0f0f0')
    subtitle_label.pack(pady=5)
    
    # Menu frame
    menu_frame = tk.Frame(root, bg='#f0f0f0')
    menu_frame.pack(pady=30)
    
    # Menu buttons
    buttons = [
        ("Step 1: Data Import", "Import IMU data from text files", data_import),
        ("Step 2: Calibration", "Calibrate IMU sensor axes", calibration),
        ("Step 3: Euler Angle Calculations", "Calculate joint angles", euler_angle_calculations),
        ("Step 4: Comparisons Tool", "Compare IMU to optical motion capture", comparisons_tool),
        ("Exit", "Exit the application", root.quit)
    ]
    
    for text, description, command in buttons:
        # Create button frame
        button_frame = tk.Frame(menu_frame, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        # Button
        btn = tk.Button(button_frame, text=text, command=command,
                       font=("Arial", 11), bg='#4CAF50', fg='white',
                       relief='raised', padx=20, pady=10)
        btn.pack()
        
        # Description
        desc_label = tk.Label(button_frame, text=description, 
                             font=("Arial", 9), bg='#f0f0f0', fg='#666666')
        desc_label.pack(pady=2)
    
    # Information frame
    info_frame = tk.Frame(root, bg='#f0f0f0')
    info_frame.pack(pady=20)
    
    info_text = """
    This tool implements the complete IMU sensor placement optimization workflow.
    
    Workflow:
    1. Import IMU data from Xsens text files
    2. Calibrate sensors using functional or static methods
    3. Calculate joint angles using Euler angle decomposition
    4. Compare results to optical motion capture reference
    
    For detailed instructions, see the README.md file.
    """
    
    info_label = tk.Label(info_frame, text=info_text, 
                         font=("Arial", 9), bg='#f0f0f0', 
                         justify='left', wraplength=500)
    info_label.pack()
    
    # Run the main loop
    root.mainloop()


def command_line_interface():
    """
    Command line interface for the tool.
    """
    print("IMU Sensor Placement Optimization - Python Implementation")
    print("=" * 60)
    print()
    
    while True:
        print("Available options:")
        print("1. Data Import - Import IMU data from text files")
        print("2. Calibration - Calibrate IMU sensor axes")
        print("3. Euler Angle Calculations - Calculate joint angles")
        print("4. Comparisons Tool - Compare IMU to optical motion capture")
        print("5. Exit")
        print()
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                print("\nStarting Data Import...")
                data_import()
            elif choice == '2':
                print("\nStarting Calibration...")
                calibration()
            elif choice == '3':
                print("\nStarting Euler Angle Calculations...")
                euler_angle_calculations()
            elif choice == '4':
                print("\nStarting Comparisons Tool...")
                comparisons_tool()
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
            
            print("\n" + "=" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")


def check_dependencies():
    """
    Check if all required dependencies are installed.
    """
    required_packages = [
        'numpy', 'scipy', 'matplotlib', 'pandas', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    return True


def main():
    """
    Main entry point for the application.
    """
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--cli':
            command_line_interface()
        elif sys.argv[1] == '--help':
            print("IMU Sensor Placement Optimization - Python Implementation")
            print()
            print("Usage:")
            print("  python main.py          # Start GUI interface")
            print("  python main.py --cli    # Start command line interface")
            print("  python main.py --help   # Show this help message")
            print()
            print("For more information, see README.md")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Start GUI interface
        try:
            main_menu()
        except Exception as e:
            print(f"Error starting GUI: {e}")
            print("Falling back to command line interface...")
            command_line_interface()


if __name__ == "__main__":
    main()
