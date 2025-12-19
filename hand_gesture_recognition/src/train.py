"""
Training Pipeline for Hand Gesture Recognition
Handles model training, validation, and evaluation.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from src.models import (GestureRecognitionCNN, TemporalGestureRecognitionCNNLSTM,
                        create_simple_cnn_model, create_efficient_model)
from src.preprocessing import load_dataset


class GestureRecognitionTrainer:
    """Handles training and evaluation of gesture recognition models."""
    
    def __init__(self, model_type='simple_cnn', num_classes=5, 
                 model_dir='models', log_dir='logs'):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of model ('simple_cnn', 'cnn', 'cnn_lstm', 'efficient')
            num_classes: Number of gesture classes
            model_dir: Directory to save models
            log_dir: Directory to save logs
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.history = None
        self.class_names = None
        
    def create_model(self):
        """Create model based on specified type."""
        print(f"Creating model: {self.model_type}")
        
        if self.model_type == 'simple_cnn':
            self.model = create_simple_cnn_model(num_classes=self.num_classes)
        elif self.model_type == 'cnn':
            self.model = GestureRecognitionCNN(num_classes=self.num_classes)
        elif self.model_type == 'cnn_lstm':
            self.model = TemporalGestureRecognitionCNNLSTM(num_classes=self.num_classes)
        elif self.model_type == 'efficient':
            self.model = create_efficient_model(num_classes=self.num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"Model created with {self.model.count_params():,} parameters")
        return self.model
    
    def compile_model(self, learning_rate=1e-3):
        """Compile model with optimizer and loss."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss = keras.losses.SparseCategoricalCrossentropy()
        metrics = [
            keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy')
        ]
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print("Model compiled")
    
    def create_callbacks(self, model_name):
        """Create training callbacks."""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                str(self.model_dir / f'{model_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(self.log_dir / model_name),
                histogram_freq=1
            )
        ]
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32,
              learning_rate=1e-3, model_name=None):
        """
        Train the model.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            model_name: Name for saving model
        """
        if model_name is None:
            model_name = f'{self.model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Compile
        self.compile_model(learning_rate=learning_rate)
        
        # Create callbacks
        callbacks = self.create_callbacks(model_name)
        
        # Train
        print(f"\nTraining model: {model_name}")
        print(f"Train set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = self.model_dir / f'{model_name}_final.h5'
        self.model.save(str(final_model_path))
        print(f"Model saved to {final_model_path}")
        
        # Save training history
        self._save_history(model_name)
        
        return self.history
    
    def _save_history(self, model_name):
        """Save training history to JSON."""
        history_data = {
            'epochs': len(self.history.history['loss']),
            'history': {k: [float(v) for v in vals] 
                       for k, vals in self.history.history.items()}
        }
        
        history_file = self.log_dir / f'{model_name}_history.json'
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"Training history saved to {history_file}")
    
    def evaluate(self, X_test, y_test, class_names=None):
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test images
            y_test: Test labels
            class_names: Names of gesture classes
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        self.class_names = class_names
        
        print("\nEvaluating model...")
        
        # Get predictions
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate accuracy
        test_loss, test_accuracy, test_top2 = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Top-2 Accuracy: {test_top2:.4f}")
        
        # Classification report
        if class_names:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=class_names))
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_top2_accuracy': test_top2,
            'y_pred': y_pred,
            'y_pred_probs': y_pred_probs,
            'y_true': y_test
        }
    
    def plot_training_history(self, model_name=None):
        """Plot training history."""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid()
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Train')
        axes[1].plot(self.history.history['val_loss'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid()
        
        plt.tight_layout()
        
        if model_name:
            save_path = self.log_dir / f'{model_name}_history_plot.png'
            plt.savefig(save_path)
            print(f"History plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, model_name=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        if model_name:
            save_path = self.log_dir / f'{model_name}_confusion_matrix.png'
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
    
    def load_model(self, model_path):
        """Load a saved model."""
        self.model = keras.models.load_model(str(model_path))
        print(f"Model loaded from {model_path}")
    
    def predict(self, images):
        """
        Make predictions on new images.
        
        Args:
            images: Array of images or single image
            
        Returns:
            Predictions and confidence scores
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Ensure proper shape
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        
        predictions = self.model.predict(images, verbose=0)
        
        return predictions


def main():
    """Main training script."""
    
    print("="*60)
    print("Hand Gesture Recognition - Training Pipeline")
    print("="*60)
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        X_train, X_test, y_train, y_test, class_names, class_to_idx = \
            load_dataset('data/processed/dataset.pkl')
    except FileNotFoundError:
        print("Dataset not found. Please run preprocessing first.")
        return
    
    # Split validation set
    split_idx = int(len(X_train) * 0.8)
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    print("Dataset loaded")
    print(f"  Classes: {class_names}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train models
    models_to_train = ['simple_cnn', 'cnn', 'efficient']
    
    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training: {model_type}")
        print(f"{'='*60}")
        
        trainer = GestureRecognitionTrainer(model_type=model_type,
                                           num_classes=len(class_names))
        trainer.create_model()
        
        # Train
        trainer.train(X_train, y_train, X_val, y_val,
                     epochs=50, batch_size=32, model_name=model_type)
        
        # Evaluate
        results = trainer.evaluate(X_test, y_test, class_names=class_names)
        
        # Visualize
        trainer.plot_training_history(model_name=model_type)
        trainer.plot_confusion_matrix(results['y_true'], results['y_pred'],
                                     class_names=class_names, model_name=model_type)
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
