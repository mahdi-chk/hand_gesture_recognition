#!/usr/bin/env python3
"""
Quick Start Script for Hand Gesture Recognition
Helps with initial setup and verification
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    print(f"✅ Python {sys.version.split()[0]}")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    required = [
        'tensorflow',
        'opencv-python',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    return len(missing) == 0, missing


def install_dependencies():
    """Install missing dependencies."""
    print("\n" + "="*50)
    print("Installing dependencies...")
    print("="*50)
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Dependencies installed!")


def check_directories():
    """Check if required directories exist."""
    dirs = [
        'data',
        'data/raw',
        'data/processed',
        'src',
        'models',
        'notebooks',
        'logs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path}")


def print_menu():
    """Print main menu."""
    print("\n" + "="*50)
    print("HAND GESTURE RECOGNITION - QUICK START")
    print("="*50)
    print("\n1. Check system requirements")
    print("2. Install/verify dependencies")
    print("3. Create project directories")
    print("4. Start data collection")
    print("5. Run full setup")
    print("6. Launch main program")
    print("0. Exit")
    print("\n" + "-"*50)
    return input("Select option (0-6): ").strip()


def main():
    """Main function."""
    while True:
        choice = print_menu()
        
        if choice == '0':
            print("Goodbye!")
            break
        
        elif choice == '1':
            print("\nChecking system requirements...")
            print("-"*50)
            if check_python_version():
                print("✅ Python version OK")
            else:
                print("❌ Python version not suitable")
        
        elif choice == '2':
            print("\nChecking dependencies...")
            print("-"*50)
            all_ok, missing = check_dependencies()
            
            if not all_ok:
                print(f"\n❌ Missing packages: {', '.join(missing)}")
                response = input("Install missing packages? (y/n): ")
                if response.lower() == 'y':
                    install_dependencies()
            else:
                print("✅ All dependencies installed!")
        
        elif choice == '3':
            print("\nCreating project directories...")
            print("-"*50)
            check_directories()
            print("✅ Directories created!")
        
        elif choice == '4':
            print("\nStarting data collection...")
            print("-"*50)
            try:
                from src.data_collector import main as collector_main
                collector_main()
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '5':
            print("\nRunning full setup...")
            print("-"*50)
            
            print("\n[1/3] Checking dependencies...")
            all_ok, missing = check_dependencies()
            if not all_ok:
                print(f"Installing missing: {', '.join(missing)}")
                install_dependencies()
            
            print("\n[2/3] Creating directories...")
            check_directories()
            
            print("\n[3/3] Verifying setup...")
            import tensorflow as tf
            import cv2
            print(f"TensorFlow: {tf.__version__}")
            print(f"OpenCV: {cv2.__version__}")
            
            print("\n✅ Setup complete!")
        
        elif choice == '6':
            print("\nLaunching main program...")
            try:
                from main import main as main_program
                main_program()
            except Exception as e:
                print(f"Error: {e}")
        
        else:
            print("Invalid option. Try again.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
