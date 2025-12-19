# Hand Gesture Recognition - Complete Guide
## Reconnaissance des gestes de la main - Guide Complet

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Data Collection](#data-collection)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training](#model-training)
6. [Real-time Inference](#real-time-inference)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

This comprehensive Hand Gesture Recognition system teaches:
- **Data Science**: Collecting and preparing datasets
- **Computer Vision**: OpenCV for image processing and hand detection
- **Deep Learning**: Building and training neural networks
- **Model Evaluation**: Assessing model performance
- **Deployment**: Real-time inference and applications

### Supported Gestures

| Gesture | French | Description |
|---------|--------|-------------|
| Palm | Paume ouverte | Open hand facing camera |
| Fist | Poing fermé | Closed fist |
| Victory | Signe de la victoire | Peace/V sign |
| OK | Signe OK | Circle with thumb and index |
| Thumbs Up | Pouce vers le haut | Thumbs up |

---

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Webcam (for data collection and inference)
- 4GB RAM minimum
- GPU recommended (CUDA 11.0+)

### Installation Steps

```bash
# 1. Navigate to project directory
cd hand_gesture_recognition

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
```

### Verify Setup

```bash
# Test all components
python main.py  # Should show menu
```

---

## Data Collection

### Step 1: Understanding the Collection Process

The `data_collector.py` script captures hand gestures with:
- Real-time preview from webcam
- Region of Interest (ROI) highlighting
- Image cropping and resizing
- Organized storage by gesture class

### Step 2: Collecting Data

#### Method 1: Using Main Menu
```bash
python main.py
# Select option 1
```

#### Method 2: Direct Execution
```bash
python src/data_collector.py
```

### Step 3: Collection Guidelines

**For Each Gesture Class:**

1. **Lighting**: Good natural or artificial lighting
2. **Distance**: 30-100 cm from camera
3. **Background**: Simple, contrasting background
4. **Variations**:
   - Different hand positions
   - Different hand sizes
   - Different angles (rotate hand)
   - Partial hand visibility
   - Different lighting conditions

**Target Counts:**
- Minimum: 100 images per gesture
- Recommended: 200-300 images per gesture
- Excellent: 500+ images per gesture

### Step 4: Collected Data Structure

```
data/raw/
├── palm/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── fist/
├── victory/
├── ok/
└── thumbs_up/
```

### Keyboard Controls During Collection

| Key | Action |
|-----|--------|
| SPACE | Capture current frame |
| 's' | Print batch statistics |
| 'q' | Save batch and quit |

---

## Data Preprocessing

### Overview of Preprocessing Steps

```
Raw Images
    ↓
[1] Load & Convert (BGR → RGB)
    ↓
[2] Hand Detection (Skin color in HSV)
    ↓
[3] Segmentation (Morphological ops)
    ↓
[4] Resize (224×224)
    ↓
[5] Normalize (0-1 range)
    ↓
[6] Data Augmentation
    ├─ Random rotation (±15°)
    ├─ Random brightness
    ├─ Random flip
    └─ Gaussian blur
    ↓
Preprocessed Dataset
```

### Hand Detection Algorithm

**HSV Skin Color Range:**
```python
# Primary range (red hues)
Lower: [0, 20, 70]
Upper: [20, 255, 255]

# Secondary range (wraparound red)
Lower: [170, 20, 70]
Upper: [180, 255, 255]
```

### Running Preprocessing

#### Method 1: Using Main Menu
```bash
python main.py
# Select option 2
```

#### Method 2: Python Script
```python
from src.preprocessing import HandPreprocessor, save_dataset

preprocessor = HandPreprocessor(image_size=(224, 224))
X_train, X_test, y_train, y_test, class_names, class_to_idx = \
    preprocessor.load_dataset_from_directory('data/raw', augment=True)

save_dataset(X_train, X_test, y_train, y_test, class_names, class_to_idx)
```

### Output

```
data/processed/
└── dataset.pkl  # Contains X_train, X_test, y_train, y_test, class_names, class_to_idx
```

### Data Splits

- **Training**: 64% of data
- **Validation**: 16% of data
- **Testing**: 20% of data

---

## Model Training

### Available Model Architectures

#### 1. Simple CNN (Recommended for beginners)
- 4 convolutional blocks
- 32→64→128→256 filters
- Global average pooling
- Dense layers with dropout
- **Parameters**: ~3.5M
- **Speed**: Fast training

#### 2. Advanced CNN (GestureRecognitionCNN)
- More sophisticated architecture
- Better regularization
- Higher accuracy
- **Parameters**: ~5M

#### 3. EfficientNet (Transfer Learning)
- Pre-trained ImageNet weights
- Efficient architecture
- Great for limited data
- **Parameters**: ~4M

#### 4. CNN+LSTM (Temporal)
- CNN for spatial features
- LSTM for temporal sequences
- Better for gesture dynamics
- **Parameters**: ~6M

### Starting Training

#### Method 1: Using Main Menu
```bash
python main.py
# Select option 3
```

#### Method 2: Python Script
```python
from src.train import GestureRecognitionTrainer
from src.preprocessing import load_dataset

# Load data
X_train, X_test, y_train, y_test, class_names, class_to_idx = \
    load_dataset('data/processed/dataset.pkl')

# Split validation
split_idx = int(len(X_train) * 0.8)
X_val = X_train[split_idx:]
y_val = y_train[split_idx:]
X_train = X_train[:split_idx]
y_train = y_train[:split_idx]

# Train
trainer = GestureRecognitionTrainer(
    model_type='simple_cnn',
    num_classes=len(class_names)
)
trainer.create_model()
trainer.train(
    X_train, y_train,
    X_val, y_val,
    epochs=50,
    batch_size=32
)
```

### Training Monitoring

During training, watch for:
- **Accuracy**: Should increase
- **Loss**: Should decrease
- **Validation Accuracy**: Should follow training accuracy
- **Overfitting**: If validation accuracy plateaus while training continues

### Training Artifacts

```
models/
├── simple_cnn_best.h5           # Best model during training
├── simple_cnn_final.h5          # Final model
└── simple_cnn_class_info.json   # Class information

logs/
├── simple_cnn/
│   ├── events.out.tfevents...   # TensorBoard logs
│   ├── simple_cnn_history.json
│   ├── simple_cnn_history_plot.png
│   └── simple_cnn_confusion_matrix.png
```

---

## Model Evaluation

### Metrics Calculated

1. **Accuracy**: Percentage of correct predictions
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1-Score**: Harmonic mean of precision and recall
5. **Confusion Matrix**: Shows misclassification patterns

### Viewing Results

Results automatically displayed after training:
- **Training history plots** (accuracy & loss curves)
- **Confusion matrix** (misclassification analysis)
- **Classification report** (per-class metrics)
- **Per-class accuracy**

### Expected Performance

- **Good**: 85-90% test accuracy
- **Excellent**: 90-95% test accuracy
- **Outstanding**: 95%+ test accuracy

(Depends on data quality and quantity)

---

## Real-time Inference

### Starting Inference

#### Method 1: Using Main Menu
```bash
python main.py
# Select option 5
```

#### Method 2: Direct Execution
```bash
python src/inference.py
```

### Real-time Features

1. **Live Prediction**: Gesture recognized in real-time
2. **Confidence Score**: Prediction confidence (%)
3. **Top-3 Predictions**: Alternative predictions shown
4. **ROI Highlighting**: Region of interest displayed
5. **Prediction Smoothing**: Temporal smoothing for stability

### Keyboard Controls

| Key | Action |
|-----|--------|
| SPACE | Pause/Resume |
| 's' | Save current frame |
| 'r' | Reset smoothing buffer |
| 'q' | Quit |

### Output Location

Saved frames stored in `predictions/` directory

### Customizing Inference

```python
from src.inference import GestureRecognitionDemo
import json

# Load model
with open('models/cnn_class_info.json') as f:
    info = json.load(f)

demo = GestureRecognitionDemo(
    'models/cnn_gesture_model.h5',
    info['class_names'],
    image_size=(224, 224)
)

# Run with custom confidence threshold
demo.run(confidence_threshold=0.7, smooth_predictions=True)
```

---

## Advanced Usage

### 1. Custom Model Architecture

```python
from tensorflow import keras
from tensorflow.keras import layers

def custom_model(num_classes):
    model = keras.Sequential([
        layers.Conv2D(64, 3, padding='same', activation='relu', 
                     input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

### 2. Transfer Learning with Pre-trained Models

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```

### 3. Hyperparameter Tuning

```python
from keras_tuner import HyperModel, RandomSearch

class MyHyperModel(HyperModel):
    def build(self, hp):
        model = keras.Sequential([
            layers.Conv2D(
                hp.Int('conv_1_filters', min_value=32, max_value=128, step=32),
                3, padding='same', activation='relu'
            ),
            # ... more layers with hp parameters
        ])
        return model

tuner = RandomSearch(
    MyHyperModel(),
    objective='val_accuracy',
    max_trials=10
)
tuner.search(X_train, y_train, validation_data=(X_val, y_val))
```

### 4. Batch Inference

```python
import numpy as np

# Inference on multiple images
batch_images = np.random.randn(10, 224, 224, 3)  # 10 images
predictions = model.predict(batch_images)  # (10, num_classes)
predicted_classes = np.argmax(predictions, axis=1)
```

### 5. Model Conversion and Optimization

```python
# Save as TensorFlow Lite for mobile
converter = tf.lite.TFLiteConverter.from_saved_model('models/gesture_model')
tflite_model = converter.convert()

with open('models/gesture_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## Troubleshooting

### Issue: Webcam Not Detected

**Solution:**
```python
import cv2

# Test camera
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    print("Camera works!")
    cap.release()
else:
    print("Camera not found. Try device index 1, 2, etc.")
    # cv2.VideoCapture(1), cv2.VideoCapture(2)
```

### Issue: Low Training Accuracy

**Solutions:**
1. Collect more diverse data (different angles, lighting)
2. Increase augmentation strength
3. Use a more complex model (EfficientNet)
4. Train for more epochs
5. Reduce batch size (16 instead of 32)

```python
# More augmentation
preprocessor.load_dataset_from_directory(
    'data/raw',
    augment=True,  # Enable augmentation
    # Images are 3x duplicated with augmentation
)
```

### Issue: Out of Memory Error

**Solutions:**
```python
# Reduce batch size
history = model.fit(X_train, y_train, batch_size=16)  # Down from 32

# Use smaller model
model = create_simple_cnn_model()  # Instead of EfficientNet

# Process fewer images
X_train = X_train[:1000]  # Limit data
```

### Issue: Model Overfitting

**Solutions:**
1. Add more dropout layers
2. Increase data augmentation
3. Use L1/L2 regularization
4. Train for fewer epochs
5. Use early stopping

```python
# More regularization
model = keras.Sequential([
    layers.Conv2D(64, 3, padding='same', activation='relu',
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),  # Increased dropout
    # ...
])
```

### Issue: Very Slow Training

**Solutions:**
1. Use GPU: Install tensorflow-gpu
2. Reduce image size: (128, 128) instead of (224, 224)
3. Use smaller model
4. Reduce epochs for testing

```python
# Check GPU
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

### Issue: Poor Real-time Performance

**Solutions:**
1. Reduce confidence threshold
2. Enable prediction smoothing
3. Use lighter model (simple_cnn)
4. Skip some frames for processing

```python
# Better smoothing
demo.run(
    confidence_threshold=0.5,  # Lower threshold
    smooth_predictions=True
)
```

---

## Performance Tips

### Optimization Strategies

1. **Data Quality**
   - Collect diverse, balanced data
   - Remove blurry/unclear images
   - Ensure consistent lighting

2. **Model Selection**
   - Use appropriate model size
   - Transfer learning for small datasets
   - Ensemble multiple models

3. **Training**
   - Use appropriate batch size (32 typical)
   - Monitor validation closely
   - Use callbacks for early stopping

4. **Inference**
   - Use prediction smoothing
   - Cache model in memory
   - Batch process when possible

---

## Next Steps

1. **Collect comprehensive data** (200+ images per gesture)
2. **Train and evaluate models** (Compare different architectures)
3. **Fine-tune hyperparameters** (Batch size, learning rate, epochs)
4. **Deploy inference** (Real-time gesture recognition)
5. **Extend functionality**:
   - Add more gesture types
   - Recognize gesture sequences
   - Integrate with applications

---

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org)
- [OpenCV Documentation](https://docs.opencv.org)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Deep Learning Course](https://www.deeplearningbook.org)

---

**Happy Learning! 🖐️**
