"""
Utilities and helper functions for Hand Gesture Recognition.
"""

import cv2
import numpy as np
from pathlib import Path
import json


def download_sign_language_mnist():
    """
    Download Sign Language MNIST dataset.
    This provides pre-labeled hand gesture data.
    
    Note: Run this function to download and prepare the dataset.
    """
    import urllib.request
    import zipfile
    
    url = "https://www.kaggle.com/api/v1/datasets/download/datamosh/sign-language-mnist"
    dataset_dir = Path('data/sign_language_mnist')
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Sign Language MNIST dataset...")
    print("Note: This requires Kaggle API credentials")
    print("See: https://github.com/Kaggle/kaggle-api")
    
    # Alternative: Download using kaggle CLI
    # run: kaggle datasets download -d datamosh/sign-language-mnist
    
    return dataset_dir


def create_config(config_path='config.json'):
    """Create a default configuration file."""
    config = {
        "data": {
            "image_size": [224, 224],
            "augmentation": True,
            "test_split": 0.2,
            "random_seed": 42
        },
        "training": {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "validation_split": 0.2
        },
        "model": {
            "types": ["simple_cnn", "cnn", "efficient", "cnn_lstm"],
            "default": "simple_cnn"
        },
        "inference": {
            "confidence_threshold": 0.6,
            "smooth_predictions": True
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration file created: {config_path}")
    return config


def load_config(config_path='config.json'):
    """Load configuration from file."""
    if not Path(config_path).exists():
        return create_config(config_path)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


class FrameBuffer:
    """Utility class for managing frame sequences."""
    
    def __init__(self, max_length=10):
        self.buffer = []
        self.max_length = max_length
    
    def add_frame(self, frame):
        """Add frame to buffer."""
        self.buffer.append(frame)
        if len(self.buffer) > self.max_length:
            self.buffer.pop(0)
    
    def get_sequence(self):
        """Get buffered sequence as numpy array."""
        if len(self.buffer) < self.max_length:
            return None
        return np.array(self.buffer)
    
    def is_full(self):
        """Check if buffer is full."""
        return len(self.buffer) == self.max_length
    
    def clear(self):
        """Clear buffer."""
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)


def compute_image_statistics(data_dir):
    """Compute mean and std for dataset normalization."""
    from pathlib import Path
    import cv2
    
    print("Computing dataset statistics...")
    
    pixel_values = []
    
    for gesture_dir in Path(data_dir).iterdir():
        if gesture_dir.is_dir():
            for image_path in gesture_dir.glob('*.jpg'):
                image = cv2.imread(str(image_path))
                if image is not None:
                    # Convert to RGB and normalize
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_resized = cv2.resize(image_rgb, (224, 224))
                    pixel_values.append(image_resized.flatten())
    
    if not pixel_values:
        print("No images found!")
        return None, None
    
    pixel_values = np.array(pixel_values)
    
    mean = np.mean(pixel_values, axis=0)
    std = np.std(pixel_values, axis=0)
    
    print(f"Statistics computed for {len(pixel_values)} images")
    print(f"  Mean: {np.mean(mean):.2f}")
    print(f"  Std: {np.mean(std):.2f}")
    
    return mean, std


if __name__ == '__main__':
    # Create default configuration
    create_config()
