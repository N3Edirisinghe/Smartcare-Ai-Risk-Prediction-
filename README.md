# 🏥 SmartCare AI — Disease Risk Classification System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-green.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end Machine Learning system engineered to classify hospital patients into disease risk categories (**Low**, **Medium**, **High**) using clinical, demographic, and financial data. Designed for clinical decision support with high interpretability and interactive prototype deployment.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Problem Statement](#-problem-statement)
- [Dataset Overview](#-dataset-overview)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Running the Web App](#-running-the-web-app)
- [Project Team](#-project-team)
- [License](#-license)

---

## 🔍 Overview

Healthcare providers handle vast volumes of patient and clinical data. Extracting actionable, real-time risk predictions requires structured, validated machine learning workflows rather than static analysis.

**SmartCare AI** delivers a complete, reproducible ML pipeline — from automated preprocessing and EDA to model training, evaluation, SHAP explainability analysis, and a user-friendly Streamlit web prototype.

---

## ✨ Key Features

- ⚙️ **Automated Data Preprocessing**: Cleaning, scaling, and categorical encoding pipelines.
- 📊 **Exploratory Data Analysis (EDA)**: Correlation matrices, statistical profiling, and feature distribution plots.
- 🤖 **Multi-Model Machine Learning**: Comparative evaluation across Logistic Regression, Random Forest, and XGBoost classifiers.
- 💡 **Explainable AI (SHAP)**: Global and local SHAP explanations for transparent clinical decision support.
- 🖥️ **Interactive Streamlit Prototype**: Real-time risk assessment web application for clinical input.
- 🧱 **Modular Architecture**: Clean, reusable Python modules under `src/` decoupled from notebooks.

---

## 🎯 Problem Statement

Manual patient risk classification in high-volume hospital environments is often subject to human inconsistency and delays. 

SmartCare AI solves this by predicting patient disease risk automatically:
- **Task Type:** Multi-Class Supervised Classification
- **Target Variable:** `disease_risk_level`
- **Output Classes:** `Low` | `Medium` | `High`

---

## 📊 Dataset Overview

The dataset encompasses 1,000 patient records structured across 4 primary domains:

| Domain | Key Attributes |
|---|---|
| 👤&nbsp;**Patient&nbsp;Demographics** | Patient ID, Age, Gender, Blood Group |
| 🩺&nbsp;**Clinical&nbsp;Information** | Diagnosis, Blood Pressure, Blood Sugar, Cholesterol, BMI |
| 🏥&nbsp;**Hospital&nbsp;Operations** | Department, Appointment History, Previous Admissions, Length of Stay, Room Type, Treatment Count, Lab Test Count |
| 💳&nbsp;**Financial&nbsp;Attributes** | Consultation Charges, Lab Charges, Room Charges, Medicine Charges, Total Bill Amount |

*Note: Raw dataset files belong under `data/raw/` (unversioned).*

---

## 🏗️ System Architecture

```text
       ┌────────────────────────┐
       │     Raw Dataset        │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Data Preprocessing &   │ ──► data/processed/*.csv
       │ Feature Engineering    │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Exploratory Analysis   │ ──► reports/figures/eda/*.png
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Model Training &       │ ──► models/*.pkl
       │ Hyperparameter Tuning  │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Model Evaluation &     │ ──► reports/model_comparison_table.csv
       │ Best Model Selection   │     models/best_model.pkl
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Explainable AI (SHAP)  │ ──► reports/figures/shap/*.png
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Streamlit Prototype    │ ──► Interactive Web Dashboard
       └────────────────────────┘
```

---

## 📂 Project Structure

```text
Smartcare-Ai-Risk-Prediction-/
├── app/                        # Interactive Streamlit prediction dashboard
│   ├── app.py                  # Main Streamlit web application entry point
│   └── components/             # Reusable UI components
├── data/
│   ├── raw/                    # Raw dataset & dictionary (local unversioned)
│   └── processed/              # Preprocessed & engineered CSV datasets
├── models/                     # Saved model artifacts (.pkl files)
├── notebooks/                  # Step-by-step Jupyter analysis notebooks
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_model_development.ipynb
│   └── 05_xai_analysis.ipynb
├── presentation/               # Project presentation slides & assets
├── reports/                    # Generated charts, tables, & evaluation reports
├── src/                        # Modular, production-ready Python source code
│   ├── data/                   # Dataset loader & validation scripts
│   │   ├── dataset_info.py
│   │   ├── loader.py
│   │   └── validator.py
│   ├── eda/                    # Visualization & statistical utilities
│   │   ├── correlations.py
│   │   ├── distributions.py
│   │   ├── statistics.py
│   │   └── visualizations.py
│   ├── models/                 # Model building, training, & evaluation
│   │   ├── compare.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   └── train.py
│   ├── prediction/             # Inference engine
│   │   └── predictor.py
│   ├── preprocessing/          # Cleaning, scaling, & encoding pipelines
│   │   ├── cleaning.py
│   │   ├── encoding.py
│   │   ├── pipeline.py
│   │   └── scaling.py
│   └── xai/                    # SHAP feature importance & explainability
│       ├── feature_importance.py
│       └── shap_explainer.py
├── .gitignore                  # Git ignore configuration
├── LICENSE                     # Project license
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Language** | Python 3.10+ |
| **Data Manipulation** | Pandas, NumPy |
| **Data Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-Learn, Joblib |
| **Explainable AI** | SHAP |
| **Web Dashboard** | Streamlit |
| **Environment** | Jupyter Notebook, VS Code |
| **Version Control** | Git, GitHub |

---

## ⚡ Getting Started

### 1️⃣ Prerequisites

Ensure you have **Python 3.10+** and **Git** installed on your system.

### 2️⃣ Clone Repository

```bash
git clone https://github.com/N3Edirisinghe/Smartcare-Ai-Risk-Prediction-.git
cd Smartcare-Ai-Risk-Prediction-
```

### 3️⃣ Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Web App

Launch the interactive Streamlit clinical prediction dashboard:

```bash
streamlit run app/app.py
```

---

## 👥 Project Team

| Student ID | Contributor | GitHub Profile | Primary Responsibility |
|---|---|---|---|
| **CIT-23-02-0021** | Nilupul Thisaranga | [@N3Edirisinghe](https://github.com/N3Edirisinghe) | Model Evaluation & Selection |
| **CIT-23-02-0025** | Siluna Nusal | [@GitGuru29](https://github.com/GitGuru29) | Explainable AI (XAI) & Prototype |
| **CIT-23-02-0042** | Dulani Madubashini | [@cobweb-sudo](https://github.com/cobweb-sudo) | Model Development & Tuning |
| **CIT-23-02-0127** | Kaveesha Dilshan | [@Kaveesha23dil](https://github.com/Kaveesha23dil) | Data Preprocessing & Feature Engineering |
| **CIT-23-02-0359** | Zumra Hassan | [@Zumrahassan222](https://github.com/Zumrahassan222) | Exploratory Data Analysis (EDA) |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
