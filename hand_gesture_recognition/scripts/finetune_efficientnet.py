"""
Finetune EfficientNetB0 on the processed ROI dataset.
If `models/efficient_finetuned.h5` exists, this script will load it and unfreeze the top `--unfreeze-layers` layers of the base model and continue training with a low LR.

Usage:
python hand_gesture_recognition/scripts/finetune_efficientnet.py --data-dir hand_gesture_recognition/data/processed/roi --output hand_gesture_recognition/models/efficient_finetuned.h5 --epochs 6 --unfreeze-layers 40
"""
from __future__ import annotations
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def build_model(input_shape=(224,224,3), num_classes=5):
    base = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="hand_gesture_recognition/data/processed/roi")
    p.add_argument("--output", default="hand_gesture_recognition/models/efficient_finetuned.h5")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--unfreeze-layers", type=int, default=40, help="Number of layers from the end of the base model to unfreeze")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    img_size = (args.image_size, args.image_size)
    batch_size = args.batch_size

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(data_dir),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(data_dir),
        validation_split=0.2,
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

    model, base = build_model(input_shape=(args.image_size,args.image_size,3), num_classes=num_classes)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If a previously trained model exists, try loading weights
    if out_path.exists():
        try:
            print(f"Loading existing model weights from {out_path}")
            model.load_weights(str(out_path))
        except Exception as e:
            print(f"Could not load existing weights: {e}")

    # Unfreeze last N layers of base
    total_layers = len(base.layers)
    unfreeze_from = max(0, total_layers - args.unfreeze_layers)
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= unfreeze_from)
    trainable_count = sum(1 for l in model.trainable_weights)
    print(f"Unfroze layers from {unfreeze_from} -> {total_layers} (total trainable weights: {trainable_count})")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        ModelCheckpoint(str(out_path), save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    print(f"Finetuning complete. Best model saved to {out_path}")

if __name__ == '__main__':
    main()
