"""
Neural Network Models for Hand Gesture Recognition
Includes CNN for static classification and CNN+LSTM for temporal modeling.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import numpy as np


class GestureRecognitionCNN(models.Model):
    """
    Simple CNN for hand gesture classification.
    Designed for static hand gesture images.
    """
    
    def __init__(self, num_classes=5, input_shape=(224, 224, 3), **kwargs):
        """
        Initialize CNN model.
        
        Args:
            num_classes: Number of gesture classes
            input_shape: Input image shape (height, width, channels)
        """
        # Pass through unexpected kwargs (like 'trainable' or 'dtype') to base class
        super(GestureRecognitionCNN, self).__init__(**kwargs)

        self.num_classes = num_classes
        # keep a normalized input_shape value for serialization and building
        self.input_shape_val = tuple(input_shape)
        
        # Block 1
        self.conv1_1 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.batch_norm1_1 = layers.BatchNormalization()
        self.conv1_2 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.batch_norm1_2 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(2)
        self.dropout1 = layers.Dropout(0.25)
        
        # Block 2
        self.conv2_1 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.batch_norm2_1 = layers.BatchNormalization()
        self.conv2_2 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.batch_norm2_2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D(2)
        self.dropout2 = layers.Dropout(0.25)
        
        # Block 3
        self.conv3_1 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.batch_norm3_1 = layers.BatchNormalization()
        self.conv3_2 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.batch_norm3_2 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D(2)
        self.dropout3 = layers.Dropout(0.25)
        
        # Block 4
        self.conv4_1 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.batch_norm4_1 = layers.BatchNormalization()
        self.conv4_2 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.batch_norm4_2 = layers.BatchNormalization()
        self.pool4 = layers.MaxPooling2D(2)
        self.dropout4 = layers.Dropout(0.25)
        
        # Global average pooling
        self.gap = layers.GlobalAveragePooling2D()
        
        # Dense layers
        self.dense1 = layers.Dense(256, activation='relu')
        self.batch_norm_dense = layers.BatchNormalization()
        self.dropout_dense = layers.Dropout(0.5)
        self.dense2 = layers.Dense(num_classes, activation='softmax')
    
    def call(self, x, training=False):
        """Forward pass."""
        # Block 1
        x = self.conv1_1(x)
        x = self.batch_norm1_1(x, training=training)
        x = self.conv1_2(x)
        x = self.batch_norm1_2(x, training=training)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)
        
        # Block 2
        x = self.conv2_1(x)
        x = self.batch_norm2_1(x, training=training)
        x = self.conv2_2(x)
        x = self.batch_norm2_2(x, training=training)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)
        
        # Block 3
        x = self.conv3_1(x)
        x = self.batch_norm3_1(x, training=training)
        x = self.conv3_2(x)
        x = self.batch_norm3_2(x, training=training)
        x = self.pool3(x)
        x = self.dropout3(x, training=training)
        
        # Block 4
        x = self.conv4_1(x)
        x = self.batch_norm4_1(x, training=training)
        x = self.conv4_2(x)
        x = self.batch_norm4_2(x, training=training)
        x = self.pool4(x)
        x = self.dropout4(x, training=training)
        
        # Global pooling and dense layers
        x = self.gap(x)
        x = self.dense1(x)
        x = self.batch_norm_dense(x, training=training)
        x = self.dropout_dense(x, training=training)
        x = self.dense2(x)
        
        return x

    def get_config(self):
        """Return model config for serialization."""
        return {
            'num_classes': self.num_classes,
            'input_shape': list(self.input_shape_val),
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape=None):
        """Ensure layers/weights are created and model has a defined input for Keras utilities.

        This calls the model once on a Keras Input with the configured input shape
        so that `model.input` and layer build states exist for serialization
        and Grad-CAM utilities.
        """
        if input_shape is None:
            input_shape = (None,) + tuple(self.input_shape_val)
        # Create a Keras Input and run a forward pass to build layers
        x = tf.keras.Input(shape=self.input_shape_val)
        try:
            _ = self.call(x, training=False)
        except Exception:
            # Some environments may require numpy arrays; fall back to a dummy numpy pass
            import numpy as _np
            _ = self(_np.zeros((1,) + self.input_shape_val, dtype=_np.float32), training=False)
        super(GestureRecognitionCNN, self).build(input_shape)


class TemporalGestureRecognitionCNNLSTM(models.Model):
    """
    CNN + LSTM model for temporal hand gesture recognition.
    Processes sequences of frames to capture gesture dynamics.
    """
    
    def __init__(self, num_classes=5, sequence_length=10, input_shape=(224, 224, 3), **kwargs):
        """
        Initialize CNN+LSTM model.
        
        Args:
            num_classes: Number of gesture classes
            sequence_length: Number of frames in a sequence
            input_shape: Input frame shape (height, width, channels)
        """
        # allow extra kwargs (e.g., trainable/dtype) to be passed during deserialization
        super(TemporalGestureRecognitionCNNLSTM, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.input_shape_val = tuple(input_shape)
        
        # Shared CNN for feature extraction
        self.cnn_features = self._build_cnn_feature_extractor()
        
        # LSTM for temporal modeling
        self.lstm1 = layers.LSTM(256, return_sequences=True, dropout=0.3)
        self.batch_norm_lstm1 = layers.BatchNormalization()
        self.lstm2 = layers.LSTM(128, return_sequences=False, dropout=0.3)
        self.batch_norm_lstm2 = layers.BatchNormalization()
        
        # Dense layers
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout_dense = layers.Dropout(0.5)
        self.dense2 = layers.Dense(num_classes, activation='softmax')
    
    def _build_cnn_feature_extractor(self):
        """Build CNN for feature extraction without classification head."""
        model = models.Sequential([
            layers.Conv2D(32, 3, padding='same', activation='relu', 
                         input_shape=(224, 224, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            layers.GlobalAveragePooling2D(),
        ])
        return model
    
    def call(self, x, training=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, height, width, channels)
            training: Whether in training mode
        """
        # Process each frame through CNN
        batch_size = tf.shape(x)[0]
        features = []
        
        for t in range(self.sequence_length):
            frame = x[:, t, :, :, :]
            feature = self.cnn_features(frame, training=training)
            features.append(feature)
        
        # Stack features for LSTM (batch_size, sequence_length, feature_dim)
        x = tf.stack(features, axis=1)
        
        # LSTM layers
        x = self.lstm1(x, training=training)
        x = self.batch_norm_lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.batch_norm_lstm2(x, training=training)
        
        # Dense layers
        x = self.dense1(x)
        x = self.dropout_dense(x, training=training)
        x = self.dense2(x)
        
        return x

    def get_config(self):
        return {
            'num_classes': self.num_classes,
            'sequence_length': self.sequence_length,
            'input_shape': list(self.input_shape_val),
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape=None):
        """Build the internal CNN feature extractor and LSTM layers without executing
        the full subclassed `call` on numpy arrays (avoids recursion issues).
        """
        if input_shape is None:
            input_shape = (None, self.sequence_length) + tuple(self.input_shape_val)

        # Build the CNN feature extractor by running a symbolic Input through it
        img_input = tf.keras.Input(shape=tuple(self.input_shape_val))
        features = self.cnn_features(img_input)

        # Determine feature dimension and build LSTM layers by passing a symbolic sequence
        feature_dim = int(features.shape[-1]) if features.shape[-1] is not None else None
        if feature_dim is None:
            # Fallback: try building cnn_features with a numpy dummy to infer shape
            import numpy as _np
            _f = self.cnn_features(_np.zeros((1,) + self.input_shape_val, dtype=_np.float32))
            feature_dim = int(_f.shape[-1])

        seq_input = tf.keras.Input(shape=(self.sequence_length, feature_dim))
        x = self.lstm1(seq_input)
        x = self.batch_norm_lstm1(x)
        x = self.lstm2(x)
        x = self.batch_norm_lstm2(x)

        # Build dense head
        d = tf.keras.Input(shape=(int(x.shape[-1]),))
        _ = self.dense1(d)
        _ = self.dropout_dense(_)
        _ = self.dense2(_)

        super(TemporalGestureRecognitionCNNLSTM, self).build(input_shape)


def create_simple_cnn_model(num_classes=5, input_shape=(224, 224, 3)):
    """
    Create a simple Sequential CNN model.
    Useful for quick experiments.
    """
    model = models.Sequential([
        layers.Conv2D(32, 3, padding='same', activation='relu', 
                     input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_efficient_model(num_classes=5, input_shape=(224, 224, 3)):
    """
    Create an EfficientNetB0-based transfer learning model.
    Good for limited data scenarios.
    """
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model weights
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing model creation...")
    
    # Test CNN model
    print("\n1. Creating Simple CNN...")
    cnn_model = GestureRecognitionCNN(num_classes=5)
    print(f"   Model created. Parameters: {cnn_model.count_params():,}")
    
    # Test with dummy input
    dummy_input = np.random.randn(2, 224, 224, 3).astype(np.float32)
    output = cnn_model(dummy_input, training=False)
    print(f"   Output shape: {output.shape}")
    
    # Test CNN+LSTM model
    print("\n2. Creating CNN+LSTM Model...")
    lstm_model = TemporalGestureRecognitionCNNLSTM(num_classes=5, sequence_length=10)
    print(f"   Model created. Parameters: {lstm_model.count_params():,}")
    
    # Test with dummy input
    dummy_input_seq = np.random.randn(2, 10, 224, 224, 3).astype(np.float32)
    output = lstm_model(dummy_input_seq, training=False)
    print(f"   Output shape: {output.shape}")
    
    # Test simple sequential model
    print("\n3. Creating Simple Sequential CNN...")
    simple_model = create_simple_cnn_model(num_classes=5)
    print(f"   Model created. Parameters: {simple_model.count_params():,}")
    print(simple_model.summary())
    
    print("\nAll models tested successfully!")
