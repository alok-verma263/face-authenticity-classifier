# Face Authenticity Classifier

> A deep learning Convolutional Neural Network (CNN) built to distinguish between real human faces and AI-generated deepfakes.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-97.21%25-brightgreen)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

[Overview](#-overview) · [Features](#-key-features) · [Architecture](#️-system-architecture) · [Results](#-results) · [Usage](#️-usage)

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
- [Screenshots](#-screenshots)
- [Limitations](#️-limitations)
- [Future Enhancements](#-future-enhancements)
- [Contributors](#-contributors)
- [References](#-references)

---

## 📌 Overview

With the rapid advancement of generative AI, the ability to generate hyper-realistic fake images has increased dramatically. This project addresses the growing need for reliable image verification by implementing a custom CNN that classifies facial images as either **REAL** or **FAKE**. The system is evaluated not just on overall accuracy, but also on its robustness across varying demographic metadata and image-quality difficulty tiers.

## 🎯 Objectives

1. Preprocess and analyze a facial image dataset, validating and filtering down to **5,556 viable images**.
2. Extract meaningful spatial and visual features from images using TensorFlow's `Conv2D` and `MaxPooling2D` layers.
3. Train a high-performing binary classification model to identify image authenticity.
4. Evaluate model performance using Accuracy, Precision, Recall, and F1-Score metrics.
5. Analyze the impact of demographic (e.g., gender) and image-quality (e.g., detection difficulty) attributes on classification performance.
6. Develop a modular, reproducible system capable of supporting future deepfake-detection applications.

## 🚨 Problem Statement

The proliferation of AI-generated media poses significant risks to digital identity, security, and information integrity. Existing verification systems often struggle with sophisticated deepfakes or exhibit undetected biases against certain demographic groups, creating a need for transparent and rigorously evaluated classification models.

## 💡 Proposed Solution

A custom deep learning pipeline using TensorFlow/Keras that ingests raw facial images, standardizes pixel data, and trains a CNN to detect microscopic deepfake artifacts. The solution includes a dedicated metadata evaluation script to ensure the model maintains consistent performance across varying difficulty tiers and gender demographics.

## ✨ Key Features

- **Custom CNN Architecture** — optimized for binary classification of 128×128 RGB facial images
- **Automated Data Pipeline** — scripts for downloading, cleaning, and preprocessing image datasets
- **Bias & Metadata Analysis** — built-in evaluation to check performance breakdowns by gender and detection difficulty
- **High Accuracy** — achieved **97.21%** validation accuracy on test data

## 🏗️ System Architecture

![Architecture](<img width="2492" height="189" alt="architecture" src="https://github.com/user-attachments/assets/673d7119-d109-4a9a-8b41-0179cfa2e6f8" />
)
*(Upload your architecture diagram to `docs/architecture.png` — it will render here automatically)*

## 🔄 Project Workflow

![Workflow](![Uploading workflow.png…]()
)
*(Upload your workflow diagram to `docs/workflow.png` — it will render here automatically)*

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming | Python |
| Deep Learning | TensorFlow, Keras |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Environment | Conda (`face_env`) |
| Version Control | Git & GitHub |

## 📂 Project Structure

```text
face-authenticity-classifier/
├── data/
│   ├── processed/         # Cleaned metadata CSVs
│   └── raw/                # Raw images (ignored in Git)
├── src/
│   ├── download_images.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── evaluate_metadata.py
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── models/                 # Saved .keras models (ignored in Git)
├── docs/                   # Images and diagrams
├── requirements.txt
└── README.md
```

## 📊 Dataset

The project utilizes a curated dataset of facial images, filtered from 6,557 initial rows down to **5,556 validated images**. The dataset includes accompanying metadata (`cleaned_metadata.csv`) detailing labels (REAL vs. FAKE), gender demographics, and assigned detection difficulty (Easy, Medium, Hard).

> **Note:** Due to file size constraints, raw images are not hosted in this repository.

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

**2. Train the model**

```bash
python src/train_model.py
```

**3. Run the metadata and bias evaluation**

```bash
python src/evaluate_metadata.py
```

## 📈 Results

The final baseline model (`face_authenticity_cnn.keras`) achieved the following metrics on the test split:

| Metric | Score |
|---|---|
| Validation Accuracy | **97.21%** |
| F1-Score | 97.18% |
| Precision | 97.81% |
| Recall | 96.57% |

**Metadata performance highlights:**

- **Gender** — consistent performance across Male (94.95%) and Female (94.20%) samples
- **Difficulty** — scaled logically, maintaining 95.35% accuracy even on "Hard" detection-difficulty samples, indicating the model learned genuine artifacts rather than superficial cues

## 📸 Screenshots

*(Add terminal outputs, training curves, or confusion-matrix visualizations here)*

## ⚠️ Limitations

- High-quality, newer-generation deepfakes (e.g., from updated diffusion models) may evade current feature-extraction layers
- The model's demographic robustness is currently limited to the labels available in the source dataset

## 🚀 Future Enhancements

- Integrate Vision Transformers (ViT) to compare performance against the baseline CNN
- Develop a Flask or FastAPI web application for real-time image uploads and inference
- Expand metadata analysis to include varying lighting conditions and age demographics

## 👥 Contributors

**Alok Verma** — Data Scientist & Machine Learning Engineer ([@alok-verma263](https://github.com/alok-verma263))

## 📚 References

- [TensorFlow Core Documentation](https://www.tensorflow.org/guide)
- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
