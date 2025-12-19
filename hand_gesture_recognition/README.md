# Hand Gesture Recognition System
## Reconnaissance des gestes de la main

A comprehensive machine learning project for recognizing hand gestures using deep learning and computer vision techniques.

## 📋 Project Overview

This project implements a complete system for hand gesture recognition with the following components:

### Supported Gestures
- **Palm** (Paume ouverte) - Open hand
- **Fist** (Poing fermé) - Closed fist
- **Victory** (Signe de la victoire) - Peace/victory sign
- **OK** (Signe OK) - Circle with thumb and index finger
- **Thumbs Up** (Pouce vers le haut) - Thumbs up gesture

### Key Features
- ✅ Real-time hand gesture recognition via webcam
- ✅ Custom dataset collection and annotation
- ✅ OpenCV-based hand detection and preprocessing
- ✅ Multiple neural network architectures:
  - Simple CNN for static gesture classification
  - Advanced CNN with batch normalization and dropout
  - CNN+LSTM for temporal sequence modeling
  - Transfer learning with EfficientNet
- ✅ Comprehensive training pipeline with callbacks
- ✅ Model evaluation with confusion matrices and metrics
- ✅ Interactive Jupyter notebooks for learning

## 📁 Project Structure

```
hand_gesture_recognition/
├── data/
│   ├── raw/              # Collected raw gesture images
│   └── processed/        # Preprocessed and augmented data
├── src/
│   ├── data_collector.py    # Webcam data collection utility
│   ├── preprocessing.py     # Image preprocessing with OpenCV
│   ├── models.py            # Neural network model definitions
│   ├── train.py             # Training pipeline
│   ├── inference.py         # Real-time inference demo
│   └── utils.py             # Utility functions
├── models/               # Saved trained models
├── notebooks/
│   ├── 01_Data_Collection_and_Exploration.ipynb
│   └── 02_Model_Training.ipynb
├── main.py              # Main entry point with menu
├── requirements.txt     # Project dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd hand_gesture_recognition

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Collection

```bash
# Run the interactive data collector
python src/data_collector.py

# Or use the main menu
python main.py
# Select option 1
```

**Instructions:**
- For each gesture class, the collector will prompt you
- Press SPACE to capture frames (aim for 100-200 per gesture)
- Try different angles, distances, and lighting conditions
- Press 'q' to finish and save

### 3. Preprocess Data

```bash
# Prepare collected data for training
python main.py
# Select option 2

# Or directly:
python -c "from src.preprocessing import HandPreprocessor, save_dataset; \
preprocessor = HandPreprocessor(); \
X_train, X_test, y_train, y_test, class_names, class_to_idx = \
preprocessor.load_dataset_from_directory('data/raw', augment=True); \
save_dataset(X_train, X_test, y_train, y_test, class_names, class_to_idx)"
```

### 4. Train Models

```bash
# Train CNN, EfficientNet, and other models
python main.py
# Select option 3

# Or directly:
python src/train.py
```

### 5. Run Real-time Inference

```bash
# Live gesture recognition with webcam
python main.py
# Select option 5

# Or directly:
python src/inference.py
```

**Controls:**
- SPACE: Pause/Resume
- 's': Save current frame
- 'q': Quit

## 📚 Jupyter Notebooks

Interactive learning notebooks included:

### 1. **01_Data_Collection_and_Exploration.ipynb**
- Dataset visualization
- Gesture class statistics
- Preprocessing pipeline demonstration
- Sample image exploration

### 2. **02_Model_Training.ipynb**
- Data loading and preparation
- CNN model architecture
- Training with callbacks
- Model evaluation
- Confusion matrix analysis
- Per-class accuracy metrics

## 🧠 Model Architectures

### Simple CNN
```
Conv2D(32) → BN → Conv2D(32) → MaxPool
    ↓
Conv2D(64) → BN → Conv2D(64) → MaxPool
    ↓
Conv2D(128) → BN → Conv2D(128) → MaxPool
    ↓
Conv2D(256) → BN → Conv2D(256) → MaxPool
    ↓
GlobalAvgPool → Dense(256) → Dense(num_classes)
```

### CNN+LSTM (Temporal Model)
- CNN feature extractor for each frame
- LSTM layer for temporal modeling
- Captures gesture dynamics and sequences

### EfficientNet Transfer Learning
- Pre-trained EfficientNetB0 backbone
- Efficient for limited computational resources
- Fine-tuned dense layers for gesture classification

## 🔧 Configuration

Default configuration in `config.json`:

```json
{
  "data": {
    "image_size": [224, 224],
    "augmentation": true,
    "test_split": 0.2
  },
  "training": {
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001
  },
  "inference": {
    "confidence_threshold": 0.6,
    "smooth_predictions": true
  }
}
```

## 📊 Data Preprocessing Pipeline

1. **Image Loading**: Read and convert BGR to RGB
2. **Resizing**: Resize to 224×224 pixels
3. **Hand Detection**: Skin color segmentation in HSV space
4. **Morphological Operations**: Clean detection with closing/opening
5. **Normalization**: Scale pixel values to [0, 1]
6. **Data Augmentation**:
   - Random rotation (±15°)
   - Random brightness adjustment
   - Random horizontal flip
   - Gaussian blur

## 📈 Training Details

- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy, Top-2 Accuracy
- **Callbacks**:
  - Early Stopping (patience: 15 epochs)
  - Learning Rate Reduction (factor: 0.5)
  - Model Checkpointing (best validation accuracy)
  - TensorBoard logging

## 🎯 Performance Metrics

The system provides:
- **Overall Accuracy**: Percentage of correct predictions
- **Per-Class Accuracy**: Accuracy for each gesture type
- **Confusion Matrix**: Misclassification patterns
- **Precision & Recall**: Classification quality metrics
- **Top-2 Accuracy**: Correct answer within top 2 predictions

## 🔍 Real-time Inference Features

- **Live Webcam Feed**: Process video stream in real-time
- **Confidence Scores**: Display prediction confidence
- **Smoothing**: Optional prediction smoothing for stability
- **ROI Display**: Visual feedback on tracked hand region
- **Top-3 Predictions**: Show alternative predictions

## 🐛 Troubleshooting

### Issue: Webcam not detected
```python
# Check available cameras
import cv2
cap = cv2.VideoCapture(0)
print(cap.isOpened())  # Should be True
```

### Issue: Low accuracy
- Collect more diverse samples (different angles, lighting)
- Increase data augmentation
- Train for more epochs
- Use transfer learning (EfficientNet)

### Issue: Memory error
- Reduce batch size (16 or 8)
- Use a smaller model
- Process fewer images

## 📦 Dependencies

- **Python 3.7+**
- **TensorFlow 2.x** - Deep learning framework
- **OpenCV** - Computer vision library
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning utilities
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter** - Interactive notebooks

Install all with:
```bash
pip install -r requirements.txt
```

## 🎓 Learning Objectives

This project teaches:

1. **Data Collection & Annotation**
   - Capturing data from webcam
   - Organizing and labeling datasets
   - Understanding data quality

2. **Computer Vision with OpenCV**
   - Image preprocessing
   - Hand detection via skin color segmentation
   - Morphological operations

3. **Deep Learning**
   - CNN architecture design
   - Transfer learning
   - Temporal modeling with LSTM

4. **Model Evaluation**
   - Classification metrics
   - Confusion matrices
   - Cross-validation

5. **Production Deployment**
   - Real-time inference
   - Model optimization
   - System integration

## 📝 Example Usage

### Training a custom model

```python
from src.train import GestureRecognitionTrainer
from src.preprocessing import load_dataset

# Load data
X_train, X_test, y_train, y_test, class_names, class_to_idx = \
    load_dataset('data/processed/dataset.pkl')

# Train model
trainer = GestureRecognitionTrainer(model_type='cnn', num_classes=len(class_names))
trainer.create_model()
trainer.train(X_train, y_train, X_test, y_test, epochs=50)
trainer.plot_training_history()
```

### Making predictions

```python
from src.inference import GestureRecognitionDemo
import json

# Load model and class info
with open('models/cnn_class_info.json') as f:
    info = json.load(f)

demo = GestureRecognitionDemo('models/cnn_gesture_model.h5', info['class_names'])
demo.run()
```

## 🤝 Contributing

Improvements and extensions welcome:
- Collect more diverse gesture data
- Implement additional gesture types
- Optimize model architecture
- Add gesture sequences/combos
- Mobile deployment

## 📄 License

This project is provided for educational purposes.

## 🎯 Future Enhancements

- [ ] Multi-hand gesture recognition
- [ ] Gesture sequences (combo moves)
- [ ] 3D hand pose estimation
- [ ] Mobile app deployment
- [ ] Real-time performance optimization
- [ ] Additional gesture classes
- [ ] Cross-platform support

## 📞 Support

For issues or questions:
1. Check the Jupyter notebooks for detailed explanations
2. Review the docstrings in source files
3. Verify all dependencies are installed
4. Check webcam permissions on your system

---

**Happy Gesture Recognition! 🖐️**

Made with ❤️ for learning deep learning and computer vision.
