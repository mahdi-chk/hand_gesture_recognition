"""
Retrain EfficientNetB0 model with class weights to fix ok vs victory confusion.
This addresses the most common misclassification from the evaluation.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_weighted_efficient_model(num_classes=5, input_shape=(224, 224, 3)):
    """Create EfficientNetB0 model optimized for transfer learning."""
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build custom head
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile with high learning rate for fresh training
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model


def main():
    print("\n" + "="*60)
    print("Retraining EfficientNetB0 with Class Weights")
    print("="*60)
    
    data_dir = Path('data/processed/roi')
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Load dataset
    img_size = (224, 224)
    batch_size = 32
    
    print("\nLoading dataset...")
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        str(data_dir),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    
    class_names = dataset.class_names
    num_classes = len(class_names)
    
    # Convert labels to one-hot encoding
    def convert_to_onehot(images, labels):
        return images, tf.one_hot(labels, num_classes)
    
    dataset = dataset.map(convert_to_onehot, num_parallel_calls=tf.data.AUTOTUNE)
    
    print(f"Classes: {class_names}")
    print(f"Number of batches: {len(dataset)}")
    
    # Count samples per class to calculate class weights
    print("\nCounting samples per class...")
    class_counts = {i: 0 for i in range(num_classes)}
    for images, labels in dataset:
        # labels are already one-hot, so convert back to class index
        class_indices = tf.argmax(labels, axis=1).numpy()
        for class_idx in class_indices:
            class_counts[class_idx] += 1
    
    print("Class distribution:")
    for i, name in enumerate(class_names):
        count = class_counts[i]
        print(f"  {name}: {count} samples")
    
    # Calculate class weights (emphasize minority/confused classes)
    total = sum(class_counts.values())
    class_weights = {}
    for i in range(num_classes):
        # Weight = total / (num_classes * count_i)
        # This makes less frequent classes have higher weight
        class_weights[i] = total / (num_classes * max(class_counts[i], 1))
    
    # Boost weights for confused classes (ok and victory have issues)
    ok_idx = class_names.index('ok') if 'ok' in class_names else None
    victory_idx = class_names.index('victory') if 'victory' in class_names else None
    
    if ok_idx is not None:
        class_weights[ok_idx] *= 1.5  # Boost ok recognition
    if victory_idx is not None:
        class_weights[victory_idx] *= 1.5  # Boost victory recognition
    
    print("\nClass weights:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_weights[i]:.2f}")
    
    # Normalize to average around 1.0
    avg_weight = np.mean(list(class_weights.values()))
    class_weights = {k: v / avg_weight for k, v in class_weights.items()}
    
    print("\nNormalized class weights:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_weights[i]:.2f}")
    
    # Split into train/val
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    
    # Prefetch for performance
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Create model
    print("\nCreating model...")
    model, base_model = create_weighted_efficient_model(num_classes=num_classes)
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/efficient_weighted_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("\nTraining with class weights...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save final model
    model.save('models/efficient_weighted_final.h5')
    print("\nModel saved to models/efficient_weighted_final.h5")
    
    # Evaluate
    print("\nEvaluating on validation set...")
    val_loss, val_acc = model.evaluate(val_dataset, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    print("\n" + "="*60)
    print("Retraining complete!")
    print("="*60)


if __name__ == '__main__':
    main()
