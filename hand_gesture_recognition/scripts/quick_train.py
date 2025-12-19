"""Quick training script using the augmented dataset at `data/processed/augmented`.
Trains the `create_simple_cnn_model` for a few epochs and saves the resulting model.
"""
import os
import tensorflow as tf
from src.models import create_simple_cnn_model


def run_train(data_dir='../data/processed/augmented', out_model='models/simple_cnn_finetuned.h5', epochs=3, batch_size=16):
    data_dir = os.path.abspath(data_dir)
    print('Loading dataset from', data_dir)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        image_size=(224,224),
        batch_size=batch_size,
        shuffle=True
    )

    class_names = train_ds.class_names
    print('Classes:', class_names)

    model = create_simple_cnn_model(num_classes=len(class_names), input_shape=(224,224,3))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_ds, epochs=epochs)

    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    model.save(out_model)
    print('Saved model to', out_model)


if __name__ == '__main__':
    run_train(epochs=2)
