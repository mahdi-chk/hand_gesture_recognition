"""
Main entry point for the Hand Gesture Recognition project.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def print_menu():
    """Print main menu."""
    print("\n" + "="*60)
    print("Hand Gesture Recognition - Main Menu")
    print("="*60)
    print("\n1. Collect gesture data (webcam)")
    print("2. Preprocess and prepare dataset")
    print("3. Train models")
    print("4. Evaluate models")
    print("5. Run real-time inference demo")
    print("6. View project information")
    print("0. Exit")
    print("\n" + "-"*60)
    return input("Select option (0-6): ").strip()


def show_info():
    """Show project information."""
    info = """
HAND GESTURE RECOGNITION PROJECT
==================================

Description:
  A comprehensive system for recognizing different hand gestures
  using deep learning and computer vision.

Gestures Supported:
  - Palm (open hand)
  - Fist (closed hand)
  - Victory (two fingers)
  - OK (circle with thumb and index)
  - Thumbs up

Components:
  1. Data Collection: Capture gestures via webcam
  2. Preprocessing: Hand detection, segmentation, augmentation
  3. Models: CNN, CNN+LSTM for static/temporal recognition
  4. Training: Full training pipeline with validation
  5. Inference: Real-time gesture recognition demo

Project Structure:
  data/
    ├── raw/           # Collected raw images
    └── processed/     # Preprocessed and augmented data
  
  src/
    ├── data_collector.py    # Webcam data collection
    ├── preprocessing.py     # Image preprocessing
    ├── models.py            # Neural network models
    ├── train.py             # Training pipeline
    ├── inference.py         # Real-time inference
    └── utils.py             # Utility functions
  
  models/              # Saved trained models
  notebooks/           # Jupyter notebooks for exploration
  logs/                # Training logs and visualizations

Quick Start:
  1. Run: python main.py
  2. Select "1" to collect gesture data
  3. Select "2" to preprocess the data
  4. Select "3" to train models
  5. Select "5" to see real-time predictions

Requirements:
  - Python 3.7+
  - TensorFlow 2.x
  - OpenCV
  - NumPy, Pandas, Scikit-learn
  - Matplotlib, Seaborn

For more information, see the notebooks in the notebooks/ folder.
    """
    print(info)


def main():
    """Main entry point."""
    
    while True:
        choice = print_menu()
        
        if choice == '0':
            print("Goodbye!")
            break
        
        elif choice == '1':
            print("\nStarting data collection...")
            from src.data_collector import main as collector_main
            try:
                collector_main()
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '2':
            print("\nPreparing dataset...")
            from src.preprocessing import HandPreprocessor, save_dataset
            try:
                preprocessor = HandPreprocessor(image_size=(224, 224))
                X_train, X_test, y_train, y_test, class_names, class_to_idx = \
                    preprocessor.load_dataset_from_directory('data/raw', augment=True)
                save_dataset(X_train, X_test, y_train, y_test, class_names, class_to_idx)
                print("\n✓ Dataset preparation completed!")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '3':
            print("\nTraining models...")
            from src.train import main as train_main
            try:
                train_main()
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '4':
            print("\nModel evaluation...")
            print("(See training output or logs/ folder for detailed results)")
        
        elif choice == '5':
            print("\nStarting real-time inference demo...")
            from src.inference import main as inference_main
            try:
                inference_main()
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '6':
            show_info()
        
        else:
            print("Invalid option. Please try again.")


if __name__ == '__main__':
    main()
