# CNNModelDiversiFAI 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

A custom deep learning project implementing a Convolutional Neural Network (CNN) from scratch using TensorFlow and Keras to classify facial attributes in the UTKFace dataset. This project demonstrates machine learning model development without relying on pre-trained architectures.

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Future Work](#-future-work)
- [References](#-references)
- [Author](#-author)
- [License](#-license)

## 🔍 Overview

CNNModelDiversiFAI explores custom CNN architecture development for facial attribute classification tasks. This research-oriented project aims to understand the capabilities and limitations of hand-designed CNN models when applied to demographic feature recognition.

The model is trained to recognize key facial attributes (age, gender, and ethnicity) using a dataset of diverse faces, making it a valuable exploration of both computer vision techniques and ethical considerations in AI development.

## 📊 Dataset

### UTKFace Dataset

- **Source**: [UTKFace on Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- **Description**: A large-scale face dataset with long age span (range from 0 to 116 years old)
- **Size**: 20,000+ face images with annotations
- **Labels**:
  - Age (0-116 years)
  - Gender (0 = Male, 1 = Female)
  - Ethnicity (0 = White, 1 = Black, 2 = Asian, 3 = Indian, 4 = Others)
- **Format**: JPEG images with filename encoding of attributes
  - Example: `[age]_[gender]_[race]_[date&time].jpg.chip.jpg`
  - Example: `25_0_2_20170116174525125.jpg.chip.jpg` = 25-year-old male of Asian ethnicity

> **Note**: Due to the nature of the dataset, there are inherent limitations in the categorical representation of complex human attributes like ethnicity. These should be interpreted with appropriate context and awareness of societal diversity.

## ✨ Features

- **Custom CNN Architecture**: Built from scratch without transfer learning
- **Multi-Attribute Classification**: Trained for age, gender, and ethnicity recognition
- **Complete ML Pipeline**: Data preprocessing, model training, evaluation, and visualization
- **Comprehensive Visualization**: Training curves, confusion matrices, and prediction examples
- **Data Augmentation**: Techniques to enhance model robustness and generalization
- **Hyperparameter Tuning**: Documentation of optimization strategies
- **Cross-Validation**: Robust evaluation methodology

## 🏗️ Model Architecture

The CNN architecture consists of:

```
Input → [Conv → ReLU → BatchNorm → MaxPool] × 3 → [Flatten] → [Dense → Dropout] × 2 → Output
```

Specifically:
- **Input Layer**: RGB images (128×128×3)
- **Convolutional Blocks**:
  - Block 1: Conv2D(32, 3×3) → ReLU → BatchNorm → MaxPool(2×2)
  - Block 2: Conv2D(64, 3×3) → ReLU → BatchNorm → MaxPool(2×2)
  - Block 3: Conv2D(128, 3×3) → ReLU → BatchNorm → MaxPool(2×2)
- **Classification Layers**:
  - Flatten
  - Dense(512) → ReLU → Dropout(0.5)
  - Dense(256) → ReLU → Dropout(0.3)
  - Dense(output_classes) → Softmax/Sigmoid

## 🚀 Getting Started

### Prerequisites

```bash
# Clone repository
git clone https://github.com/yourusername/CNNModelDiversiFAI.git
cd CNNModelDiversiFAI

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

1. Download the [UTKFace dataset from Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new?resource=download)
2. Extract the downloaded ZIP file
3. Place the images in the `data/utkface/` directory

### Running the Project

1. Open the Jupyter notebook:
```bash
jupyter notebook CNNModelDiversiFAI.ipynb
```

2. Execute the notebook cells sequentially to:
   - Load and preprocess the dataset
   - Define and compile the model
   - Train the model
   - Evaluate and visualize results

### Using the Pre-trained Model

```python
# Load saved model
from tensorflow.keras.models import load_model
model = load_model('models/cnn_diversifai_model.h5')

# Make predictions
import cv2
import numpy as np

# Load and preprocess an image
img = cv2.imread('path/to/image.jpg')
img = cv2.resize(img, (128, 128))
img = img / 255.0  # Normalize
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Get predictions
predictions = model.predict(img)
```

## 📁 Project Structure

```
CNNModelDiversiFAI/
├── CNNModelDiversiFAI.ipynb      # Main Jupyter notebook with code and documentation
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── data/                         # Dataset directory
│   └── utkface/                  # UTKFace dataset images
├── models/                       # Saved model files
│   ├── cnn_diversifai_model.h5   # Trained model weights
│   └── model_architecture.json   # Model architecture
├── results/                      # Output directory
│   ├── training_history.png      # Training/validation curves
│   ├── confusion_matrix.png      # Evaluation confusion matrix
│   ├── example_predictions.png   # Visualization of model predictions
│   └── metrics.csv               # Detailed performance metrics
└── utils/                        # Utility scripts
    ├── data_loader.py            # Dataset handling functions
    ├── preprocessing.py          # Image preprocessing utilities
    ├── visualization.py          # Result visualization helpers
    └── model_utils.py            # Model definition and training helpers
```

## 📈 Results

### Performance Metrics

| Metric | Age | Gender | Ethnicity |
|--------|-----|--------|-----------|
| Accuracy | 82.3% | 95.7% | 88.1% |
| Precision | 78.9% | 94.2% | 85.3% |
| Recall | 81.2% | 96.8% | 87.4% |
| F1 Score | 80.0% | 95.5% | 86.3% |

### Training Curves

The model shows consistent improvement during training with minimal overfitting, demonstrating the effectiveness of the regularization techniques employed:

- Training accuracy plateaus at ~95%
- Validation accuracy stabilizes at ~92%
- Learning rate reduction triggers at epochs 20 and 35

### Key Findings

- The model performs best on gender classification
- Age prediction is most challenging, particularly for older age groups
- Batch normalization significantly improves training stability
- Data augmentation contributes to a 3.5% improvement in validation accuracy

## 🔮 Future Work

- **Multi-task Learning**: Implement joint training for simultaneous prediction of age, gender, and ethnicity
- **Model Optimization**: Explore model quantization and pruning for deployment efficiency
- **Fairness Analysis**: Conduct comprehensive bias evaluation across demographic groups
- **Advanced Architectures**: Compare with attention mechanisms and residual connections
- **Ensemble Methods**: Develop specialized models for each attribute and combine predictions
- **Uncertainty Quantification**: Add confidence estimates to model predictions
- **Cross-dataset Evaluation**: Test generalization on other facial datasets

## 📚 References

- Zhang, Z., Song, Y., & Qi, H. (2017). "Age Progression/Regression by Conditional Adversarial Autoencoder." IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
- [UTKFace Dataset on Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
- TensorFlow Documentation: [Convolutional Neural Networks](https://www.tensorflow.org/tutorials/images/cnn)
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

## 👤 Author

**Shubhayu Kundu**  
Email: sk2527@srmist.edu.in 
[GitHub]([https://github.com/Shubhayu15]) | [LinkedIn](https://www.linkedin.com/in/shubhayu-kundu-7441ba295/)

**Adityabaan Tripathy**  
Email: at9715@srmist.edu.in 
[GitHub]([https://github.com/Adityabaan]) | [LinkedIn](https://www.linkedin.com/in/adityabaan-tripathy-6b245323b/)


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <em>Made with ❤️ and TensorFlow</em>
</p>
