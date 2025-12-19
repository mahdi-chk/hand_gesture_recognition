"""
Preprocessing Utilities for Hand Gesture Recognition
Handles image preprocessing, hand detection, and augmentation.
"""

import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt


class HandPreprocessor:
    """Handles preprocessing of hand gesture images."""
    
    def __init__(self, image_size=(224, 224)):
        """
        Initialize the preprocessor.
        
        Args:
            image_size: Target image size (width, height)
        """
        self.image_size = image_size
        
    def load_and_preprocess(self, image_path):
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Convert BGR to RGB for proper color handling
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, self.image_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def hand_detection_and_segmentation(self, image):
        """
        Detect hand region using skin color and morphological operations.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Hand mask and bounding box
        """
        # Convert BGR to HSV for better skin detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Expand the range to catch more skin tones
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        mask = cv2.bitwise_or(mask, mask2)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours and get bounding box
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        bbox = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            bbox = cv2.boundingRect(largest_contour)
        
        return mask, bbox
    
    def extract_hand_roi(self, image, padding=20):
        """
        Extract hand region from image.
        
        Args:
            image: Input image (BGR format)
            padding: Padding around detected hand
            
        Returns:
            Cropped hand image and bounding box
        """
        mask, bbox = self.hand_detection_and_segmentation(image)
        
        if bbox is None:
            return image, None
        
        x, y, w, h = bbox
        
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        roi = image[y:y+h, x:x+w]
        
        return roi, (x, y, w, h)
    
    def apply_data_augmentation(self, image):
        """
        Apply random augmentations to image.
        
        Args:
            image: Input image (0-1 range, RGB format)
            
        Returns:
            Augmented image
        """
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 1)
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random Gaussian blur
        if np.random.random() > 0.7:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        
        return image
    
    def load_dataset_from_directory(self, data_dir, augment=True, test_split=0.2, 
                                   max_samples_per_class=None):
        """
        Load all images from directory organized by gesture class.
        
        Args:
            data_dir: Root directory containing subdirectories for each gesture class
            augment: Whether to apply data augmentation
            test_split: Fraction of data for test set
            max_samples_per_class: Limit samples per class (None for all)
            
        Returns:
            X_train, X_test, y_train, y_test, class_names, class_to_idx
        """
        data_dir = Path(data_dir)
        
        X = []
        y = []
        class_names = []
        class_to_idx = {}
        
        gesture_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        
        print(f"Loading dataset from {data_dir}")
        print(f"Found {len(gesture_dirs)} gesture classes")
        
        for class_idx, gesture_dir in enumerate(gesture_dirs):
            gesture_name = gesture_dir.name
            class_names.append(gesture_name)
            class_to_idx[gesture_name] = class_idx
            
            image_files = list(gesture_dir.glob('*.jpg'))
            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
            print(f"  {gesture_name}: {len(image_files)} images", end="")
            
            for image_path in image_files:
                image = self.load_and_preprocess(image_path)
                if image is not None:
                    X.append(image)
                    y.append(class_idx)
                    
                    # Add augmented versions
                    if augment:
                        for _ in range(2):  # Add 2 augmented versions per image
                            aug_image = self.apply_data_augmentation(image.copy())
                            X.append(aug_image)
                            y.append(class_idx)
            
            print(f" -> {len([i for i in y if i == class_idx])} samples (with augmentation)")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nTotal samples: {len(X)}")
        print(f"Dataset shape: {X.shape}")
        
        # Split into train and test
        # Handle small datasets: ensure test set has at least one sample per class
        n_samples = len(X)
        n_classes = len(class_names)

        if n_samples == 0:
            raise ValueError(f"No images found in {data_dir}")

        # Desired number of test samples (integer)
        desired_test_n = max(int(np.floor(n_samples * test_split)), n_classes)

        # Make sure there is at least one sample left for training
        if desired_test_n >= n_samples:
            # leave at least one sample per class in training if possible
            desired_test_n = max(n_classes, n_samples - n_classes)

        # At minimum one test sample
        desired_test_n = max(1, int(desired_test_n))

        # Convert to fraction for train_test_split when desired
        test_size_param = desired_test_n

        # Check per-class counts to decide whether stratify is safe
        class_counts = [int(np.sum(y == i)) for i in range(n_classes)]
        min_count = min(class_counts) if class_counts else 0

        stratify_param = y
        if min_count < 2:
            # Not enough samples per class to guarantee stratified split
            print("Warning: Some classes have fewer than 2 samples. Falling back to non-stratified split.")
            stratify_param = None

        try:
            # Use integer test_size (absolute number) when possible
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_param, random_state=42, stratify=stratify_param
            )
        except ValueError:
            # Fallback: try using fraction instead
            test_frac = float(desired_test_n) / float(n_samples)
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_frac, random_state=42, stratify=stratify_param
                )
            except Exception as e:
                # As a last resort, perform a plain (non-stratified) split
                print(f"Warning: Stratified split failed ({e}). Using non-stratified split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_frac, random_state=42, stratify=None
                )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test, class_names, class_to_idx
    
    def visualize_preprocessing(self, image_path, save_path=None):
        """
        Visualize preprocessing steps.
        
        Args:
            image_path: Path to the image
            save_path: Optional path to save visualization
        """
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Original
        original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Hand detection
        mask, bbox = self.hand_detection_and_segmentation(image)
        
        # Extracted ROI
        roi, _ = self.extract_hand_roi(image)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Preprocessed
        preprocessed = self.load_and_preprocess(image_path)
        if preprocessed is not None:
            preprocessed = (preprocessed * 255).astype(np.uint8)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('Hand Detection Mask')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(roi_rgb)
        axes[1, 0].set_title('Extracted ROI')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(preprocessed)
        axes[1, 1].set_title('Preprocessed Image')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()


def save_dataset(X_train, X_test, y_train, y_test, class_names, class_to_idx, 
                 output_dir='data/processed'):
    """Save processed dataset to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'class_names': class_names,
        'class_to_idx': class_to_idx
    }
    
    output_file = output_dir / 'dataset.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Dataset saved to {output_file}")


def load_dataset(dataset_file='data/processed/dataset.pkl'):
    """Load processed dataset from disk."""
    with open(dataset_file, 'rb') as f:
        data = pickle.load(f)
    
    return (data['X_train'], data['X_test'], data['y_train'], data['y_test'],
            data['class_names'], data['class_to_idx'])


if __name__ == '__main__':
    # Example usage
    preprocessor = HandPreprocessor(image_size=(224, 224))
    
    # Load dataset from collected images
    X_train, X_test, y_train, y_test, class_names, class_to_idx = \
        preprocessor.load_dataset_from_directory('data/raw', augment=True)
    
    # Save processed dataset
    save_dataset(X_train, X_test, y_train, y_test, class_names, class_to_idx)
    
    print("\nDataset preprocessing completed!")
