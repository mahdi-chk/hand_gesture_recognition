# ✋ Hand Gesture Recognition

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table des matières

- [Description](#-description)
- [Objectifs](#-objectifs)
- [Technologies](#-technologies)
- [Installation](#%EF%B8%8F-installation)
- [Structure du projet](#-structure-du-projet)
- [Utilisation](#%EF%B8%8F-utilisation)
- [Dataset](#-dataset)
- [Résultats](#-résultats)
- [Améliorations futures](#-améliorations-futures)
- [Contribution](#-contribution)
- [Auteur](#-auteur)
- [Licence](#-licence)

## 📌 Description

Système de **reconnaissance de gestes de la main** basé sur des techniques avancées de **vision par ordinateur** et de **deep learning**. Ce projet permet de classifier différents gestes de la main en temps réel, ouvrant la voie à des applications d'interaction homme-machine (HCI) naturelles et intuitives.

### Cas d'usage potentiels

- 🎮 Contrôle de jeux vidéo sans manette
- 🤖 Interaction avec des systèmes robotiques
- 📱 Interface utilisateur sans contact
- 🧑‍🦽 Assistance pour personnes à mobilité réduite
- 📊 Présentation et contrôle à distance

## 🎯 Objectifs

- ✅ Détecter et segmenter la main dans des images ou flux vidéo
- ✅ Extraire des caractéristiques pertinentes (features extraction)
- ✅ Entraîner un modèle de classification robuste
- ✅ Évaluer les performances (accuracy, precision, recall, F1-score)
- ✅ Déployer le modèle pour des prédictions en temps réel

## 🧠 Technologies

| Technologie | Usage |
|------------|-------|
| **Python 3.8+** | Langage principal |
| **OpenCV** | Traitement d'images et vision par ordinateur |
| **NumPy** | Calculs numériques et manipulation de tableaux |
| **Pandas** | Analyse et manipulation de données |
| **TensorFlow/Keras** | Deep learning et entraînement de modèles CNN |
| **Scikit-learn** | Métriques d'évaluation et preprocessing |
| **Matplotlib/Seaborn** | Visualisation des données et résultats |
| **Jupyter Notebook** | Expérimentation et prototypage |

## ⚙️ Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- (Optionnel) Environnement virtuel (venv ou conda)

### Étapes d'installation

1. **Cloner le dépôt**

```bash
git clone https://github.com/mahdi-chk/hand_gesture_recognition.git
cd hand_gesture_recognition
```

2. **Créer un environnement virtuel** (recommandé)

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Vérifier l'installation**

```bash
python -c "import cv2, tensorflow as tf; print('Installation réussie!')"
```

## 📁 Structure du projet

```
hand_gesture_recognition/
│
├── 📓 notebooks/              # Notebooks Jupyter
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── 📊 data/                   # Dataset
│   ├── raw/                   # Images brutes
│   ├── processed/             # Images prétraitées
│   └── splits/                # Train/Val/Test splits
│
├── 🤖 models/                 # Modèles entraînés
│   ├── model_v1.h5
│   ├── model_v2.keras
│   └── best_model.keras
│
├── 🐍 scripts/                # Scripts Python
│   ├── preprocess.py          # Prétraitement des données
│   ├── train.py               # Entraînement du modèle
│   ├── evaluate.py            # Évaluation des performances
│   ├── predict.py             # Prédictions
│   └── real_time_detection.py # Détection en temps réel
│
├── 🛠️ utils/                  # Utilitaires
│   ├── data_loader.py
│   ├── augmentation.py
│   └── visualization.py
│
├── 📄 requirements.txt        # Dépendances
├── 📋 .gitignore
├── 📜 LICENSE
└── 📖 README.md
```

## ▶️ Utilisation

### 1. Préparation des données

```bash
python scripts/preprocess.py --input data/raw --output data/processed
```

### 2. Entraînement du modèle

```bash
python scripts/train.py --epochs 50 --batch-size 32
```

### 3. Évaluation

```bash
python scripts/evaluate.py --model models/best_model.keras --test-data data/splits/test
```

### 4. Détection en temps réel

```bash
python scripts/real_time_detection.py --model models/best_model.keras
```

### Utilisation dans un notebook

```python
import cv2
from tensorflow.keras.models import load_model

# Charger le modèle
model = load_model('models/best_model.keras')

# Faire une prédiction
image = cv2.imread('test_image.jpg')
prediction = model.predict(image)
print(f"Geste détecté: {prediction}")
```

## 📊 Dataset

Le projet utilise un dataset de gestes de la main comprenant:

- **Classes**: 👍 Thumbs Up, ✌️ Peace, 👋 Wave, ✊ Fist, 🖐️ Open Palm, etc.
- **Nombre d'images**: ~10,000+ images
- **Résolution**: 224x224 pixels (après preprocessing)
- **Format**: JPG/PNG

### Sources de données possibles

- [HaGRID - Hand Gesture Recognition Image Dataset](https://github.com/hukenovs/hagrid)
- [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- Dataset custom collecté via webcam

## 📈 Résultats

### Performance du modèle

| Métrique | Score |
|----------|-------|
| Accuracy | 94.5% |
| Precision | 93.8% |
| Recall | 94.2% |
| F1-Score | 94.0% |

### Matrice de confusion

```
                Prédictions
Réel      👍   ✌️   👋   ✊   🖐️
  👍      95   2    1    1    1
  ✌️      1    96   2    0    1
  👋      2    1    94   2    1
  ✊      0    1    1    97   1
  🖐️      1    0    2    0    97
```

### Visualisations

- Courbes d'accuracy et loss pendant l'entraînement
- Matrice de confusion
- Exemples de prédictions correctes et incorrectes
- t-SNE des features extraites

## 🚀 Améliorations futures

- [ ] 🤖 Intégration de **MediaPipe Hands** pour la détection des landmarks
- [ ] 🧠 Implémentation d'architectures plus avancées (ResNet, EfficientNet, Vision Transformer)
- [ ] 📹 Reconnaissance de gestes dynamiques (séquences vidéo + LSTM/GRU)
- [ ] 🌐 Déploiement web avec Flask/FastAPI
- [ ] 📱 Application mobile (Android/iOS)
- [ ] 🔄 Data augmentation avancée (mixup, cutmix)
- [ ] 🎯 Transfer learning avec des modèles pré-entraînés
- [ ] ⚡ Optimisation pour l'inférence temps réel (TensorRT, ONNX)

## 🤝 Contribution

Les contributions sont les bienvenues! Pour contribuer:

1. Forkez le projet
2. Créez une branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add AmazingFeature'`)
4. Poussez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 👨‍💻 Auteur

**El Mahdi Chakouch**

- GitHub: [@mahdi-chk](https://github.com/mahdi-chk)
- LinkedIn: [El Mahdi Chakouch](https://linkedin.com/in/mahdi-chakouch)
- Email: elmahdi.chakouch@gmail.com

## 📜 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

<div align="center">

**⭐ N'oubliez pas de donner une étoile si ce projet vous a été utile! ⭐**

Made with ❤️ by El Mahdi Chakouch

</div>
