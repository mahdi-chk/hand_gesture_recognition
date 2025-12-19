"""
Train a transfer-learning model using EfficientNetB0 on a dataset organized as:
  hand_gesture_recognition/data/raw/<class>/*.jpg

This script uses `tf.keras.preprocessing.image_dataset_from_directory` for simplicity.
It includes data augmentation, checkpointing, early stopping, and saves the final model.

Usage:
python hand_gesture_recognition/scripts/train_efficientnet.py --data-dir hand_gesture_recognition/data/raw --output models/efficient_finetuned.h5 --epochs 10

Note: Training can be slow on CPU. Use GPU for reasonable speed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def build_model(input_shape=(224,224,3), num_classes=5):
    base = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="hand_gesture_recognition/data/raw", help="Gallery-style data directory")
    p.add_argument("--output", default="hand_gesture_recognition/models/efficient_finetuned.h5", help="Output model path")
    p.add_argument("--image-size", type=int, default=224, help="Image height/width")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--val-split", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    img_size = (args.image_size, args.image_size)
    batch_size = args.batch_size

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(data_dir),
        validation_split=args.val_split,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(data_dir),
        validation_split=args.val_split,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Found classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    def augment(x, y):
        return data_augmentation(x, training=True), y

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    model = build_model(input_shape=(args.image_size, args.image_size, 3), num_classes=num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(str(out_path), save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    print(f"Training complete. Best model saved to {out_path}")


if __name__ == "__main__":
    main()
