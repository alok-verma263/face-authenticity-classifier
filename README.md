# Face Authenticity Classifier

> A deep learning Convolutional Neural Network (CNN) built to distinguish between real human faces and AI-generated deepfakes.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-95.20%25-brightgreen)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

[Overview](#-overview) · [Features](#-key-features) · [Architecture](#️-system-architecture) · [Results](#-results) · [Visualizations](#-visualizations) · [Usage](#️-usage)

---

## Table of Contents

- [Overview](#-overview)
- [Objectives](#-objectives)
- [Problem Statement](#-problem-statement)
- [Proposed Solution](#-proposed-solution)
- [Key Features](#-key-features)
- [System Architecture](#️-system-architecture)
- [Project Workflow](#-project-workflow)
- [Technology Stack](#️-technology-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation](#️-installation)
- [Usage](#️-usage)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Limitations](#️-limitations)
- [Future Enhancements](#-future-enhancements)
- [Contributors](#-contributors)
- [References](#-references)

---

## 📌 Overview

With the rapid advancement of generative AI, the ability to generate hyper-realistic fake images has increased dramatically. This project addresses the growing need for reliable image verification by implementing a custom CNN that classifies facial images as either **REAL** or **FAKE**. The system is evaluated not just on overall accuracy, but also on its robustness across predefined splits, varying demographic metadata (Gender, Age Groups), and image-quality difficulty tiers.

## 🎯 Objectives

1. Preprocess and analyze a facial image dataset, validating and filtering down to **5,556 viable images**.
2. Extract meaningful spatial and visual features from images using TensorFlow's `Conv2D` and `MaxPooling2D` layers.
3. Train a high-performing binary classification model using dataset-predefined train (3,883), validation (547), and test (1,126) splits.
4. Prevent mode collapse using `EarlyStopping` and `ModelCheckpoint`, automatically preserving peak validation weights.
5. Evaluate model performance using Accuracy, Precision, Recall, and F1-Score metrics on the hold-out test set.
6. Analyze the impact of demographic (gender, age groups) and image-quality attributes (detection difficulty, image quality) on classification performance.
7. Develop an end-user verification system (`predict.py`) that returns authenticity classification and confidence score percentages.

## 🚨 Problem Statement

The proliferation of AI-generated media poses significant risks to digital identity, security, and information integrity. Existing verification systems often struggle with sophisticated deepfakes or exhibit undetected biases against certain demographic groups, creating a need for transparent, reproducible, and rigorously evaluated classification models.

## 💡 Proposed Solution

A custom deep learning pipeline using TensorFlow/Keras that ingests raw facial images, standardizes pixel data, and trains a CNN to detect microscopic deepfake artifacts. The solution includes a dedicated metadata evaluation script to ensure the model maintains consistent performance across varying difficulty tiers, age categories, and gender demographics, paired with an interactive single-image verification tool.

## ✨ Key Features

- **Custom CNN Architecture** — optimized for binary classification of 128×128 RGB facial images
- **Mode Collapse Prevention** — `EarlyStopping` (`restore_best_weights=True`) & `ModelCheckpoint` monitoring validation accuracy to automatically restore optimal weights
- **Predefined Data Splits** — strict evaluation on 3,883 training, 547 validation, and 1,126 testing samples
- **High Test Accuracy** — achieved **95.20%** accuracy, **97.30%** precision, and **94.92%** F1-score on the holdout test set
- **Bias & Demographic Robustness** — multi-attribute evaluation across Gender, Age Groups, Image Quality, and Difficulty tiers
- **Single-Image Verification CLI** — `predict.py` script evaluating individual images with confidence score percentages
- **Automated Visual Reporting** — automated generation of training curves and confusion matrix plots

## 🏗️ System Architecture

<img width="2492" height="189" alt="architecture" src="https://github.com/user-attachments/assets/18545565-2df9-413e-96f6-a46876c569ae" />

## 🔄 Project Workflow

<img width="1192" height="2000" alt="workflow" src="https://github.com/user-attachments/assets/e034c51f-04e4-4eb1-9f94-2098afe36a08" />

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming | Python |
| Deep Learning | TensorFlow, Keras |
| Data Manipulation | Pandas, NumPy |
| Machine Learning & Metrics | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Environment | Conda (`face_env`) |
| Version Control | Git & GitHub |

## 📂 Project Structure

```text
face-authenticity-classifier/
├── data/
│   ├── processed/         # Cleaned metadata CSVs (cleaned_metadata.csv)
│   └── raw/               # Raw images (5,556 images)
├── docs/                  # Architectural diagrams and generated evaluation plots
│   ├── architecture.png
│   ├── workflow.png
│   ├── training_curves.png
│   └── confusion_matrix.png
├── models/                # Saved trained CNN models (face_authenticity_cnn.keras)
├── reports/
│   └── figures/           # Exported publication-ready charts
├── src/
│   ├── download_images.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   ├── evaluate_metadata.py
│   └── predict.py
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── predict.py             # User verification tool (CLI)
├── requirements.txt
└── README.md
```

## 📊 Dataset

The project utilizes a curated dataset of facial images, filtered from 6,557 initial rows down to **5,556 validated images**. The dataset includes accompanying metadata (`cleaned_metadata.csv`) detailing labels (REAL vs. FAKE), gender demographics, age brackets, image quality indicators, and assigned detection difficulty.

- **Training Split:** 3,883 images (69.9%)
- **Validation Split:** 547 images (9.8%)
- **Testing Split:** 1,126 images (20.3%)

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/alok-verma263/face-authenticity-classifier.git
cd face-authenticity-classifier
```

**2. Create and activate a Conda environment**

```bash
conda create -n face_env python=3.10
conda activate face_env
```

**3. Install the required dependencies**

```bash
pip install -r requirements.txt
```

## ▶️ Usage

Execute the pipeline in the following order using your active environment:

**1. Preprocess the data**

```bash
python src/preprocess_data.py
```

**2. Train the model and generate training curves**

```bash
python src/train_model.py
```
*Outputs: `models/face_authenticity_cnn.keras` and `docs/training_curves.png`.*

**3. Run the metadata and bias evaluation**

```bash
python src/evaluate_metadata.py
```
*Outputs: Detailed demographic metrics and `docs/confusion_matrix.png`.*

**4. Verify authenticity of single images (User Verification System)**

```bash
# Evaluate any custom facial image
python predict.py data/raw/images/image_1.jpg

# Or specify via the image flag
python predict.py --image path/to/face.jpg
```

Example CLI Output:
```text
============================================================
    AI FACE AUTHENTICITY VERIFICATION RESULT
============================================================
Target Image:      data/raw/images/image_1.jpg
Authenticity:      REAL (Authentic Face)
Confidence Score:  94.70%
------------------------------------------------------------
Authentic Prob:    94.70%
Manipulated Prob:  5.30%
============================================================
Status: [PASS] Image verified as an authentic human face.
```

## 📈 Results

### Overall Performance Metrics

Evaluation on the official holdout test set (**1,126 images**) demonstrates strong discrimination capability between authentic and synthetic faces:

| Metric | Holdout Test Set (Predefined) | Validation Set |
|---|---|---|
| **Accuracy** | **95.20%** | 93.24% |
| **Precision** | **97.30%** | 95.13% |
| **Recall** | **92.65%** | 91.37% |
| **F1-Score** | **94.92%** | 93.21% |

### Demographic & Metadata Robustness Breakdown

The model was subjected to slice-based evaluation across four key metadata dimensions:

#### 1. Age Group Demographics
| Age Group | Sample Count | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **18–25** | 309 | 93.85% | 96.21% | 90.07% | 93.04% |
| **26–35** | 260 | 93.46% | 95.56% | 92.14% | 93.82% |
| **36–50** | 297 | 96.97% | 98.59% | 95.24% | 96.89% |
| **50+** | 260 | 96.54% | 99.08% | 93.10% | 96.00% |

#### 2. Image Quality Levels
| Quality | Sample Count | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **High** | 755 | 94.97% | 96.49% | 92.70% | 94.56% |
| **Medium** | 371 | 95.69% | 98.86% | 92.55% | 95.60% |

#### 3. Detection Difficulty Tiers
| Difficulty Tier | Sample Count | Accuracy | F1-Score | Note |
|---|---|---|---|---|
| **Easy** | 582 | 97.59% | Baseline | Predominantly authentic samples |
| **Medium** | 180 | 94.44% | 97.14% | Balanced manipulation detection |
| **Hard** | 364 | 91.76% | 95.70% | High-fidelity synthetic artifacts detected |

#### 4. Gender Demographics
| Gender | Sample Count | Accuracy |
|---|---|---|
| **Female** | 217 | 88.02% |
| **Male** | 205 | 93.66% |
| **Unknown** | 704 | 97.87% |

## 📊 Visualizations

### Training & Validation Curves
The loss and accuracy progressions across training epochs demonstrate stable convergence without extreme overfitting:

![Training Curves](docs/training_curves.png)

### Confusion Matrix (Test Set)
Performance breakdown on the 1,126 holdout testing images:

![Confusion Matrix](docs/confusion_matrix.png)

## ⚠️ Limitations

- High-quality, newer-generation deepfakes (e.g., from updated diffusion models) may evade current feature-extraction layers
- Dataset distribution in gender attributes shows uneven proportion of synthetic samples in specific subgroups

## 🚀 Future Enhancements

- Integrate Vision Transformers (ViT) to benchmark against CNN spatial features
- Develop a web interface (Streamlit or FastAPI) for drag-and-drop face verification
- Implement Grad-CAM visualizations to explain which facial regions trigger manipulated classifications

## 👥 Contributors

**Alok Verma** — MBA Student (Business Analytics) ([@alok-verma263](https://github.com/alok-verma263))

## 📚 References

- [TensorFlow Core Documentation](https://www.tensorflow.org/guide)
- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
