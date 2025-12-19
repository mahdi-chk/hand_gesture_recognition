"""
Fresh fine-tune: build a new EfficientNetB0 model, unfreeze top layers, and train on ROI-processed data.
This avoids weight shape mismatch errors from loading old models.

Usage:
python hand_gesture_recognition/scripts/finetune_roi_fresh.py --data-dir hand_gesture_recognition/data/processed/roi --output hand_gesture_recognition/models/efficient_finetuned_roi.h5 --epochs 8
"""
from __future__ import annotations
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def build_and_finetune_model(num_classes=5, input_shape=(224,224,3)):
    """Build EfficientNetB0 with unfrozen top layers."""
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, 
        input_shape=input_shape, 
        weights='imagenet'
    )
    
    # Unfreeze the last 50 layers for fine-tuning
    for layer in base.layers[:-50]:
        layer.trainable = False
    for layer in base.layers[-50:]:
        layer.trainable = True
    
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=True)  # Important: training=True to use batch norm updates
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="hand_gesture_recognition/data/processed/roi")
    p.add_argument("--output", default="hand_gesture_recognition/models/efficient_finetuned_roi.h5")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    img_size = (args.image_size, args.image_size)
    batch_size = args.batch_size

    # Load train/val datasets
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(data_dir),
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(data_dir),
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Found classes:", class_names)
    print(f"Training samples: {len(train_ds) * batch_size}, Validation samples: {len(val_ds) * batch_size}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Add data augmentation
    def augment(x, y):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.rot90(x, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        x = tf.image.random_brightness(x, 0.2)
        x = tf.image.random_contrast(x, 0.8, 1.2)
        return x, y

    train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)

    # Build fresh model
    print("Building fresh EfficientNetB0 model with unfrozen top layers...")
    model = build_and_finetune_model(num_classes=num_classes, input_shape=(args.image_size, args.image_size, 3))
    
    # Compile with low learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            str(out_path),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print(f"\nTraining for {args.epochs} epochs with low learning rate (2e-5)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nFine-tuning complete!")
    print(f"Best model saved to: {out_path}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

if __name__ == '__main__':
    main()
