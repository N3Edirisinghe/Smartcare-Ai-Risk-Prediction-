# 🏥 SmartCare AI — Patient Appointment No-Show Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-008080.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-750014.svg)](LICENSE)
[![Task](https://img.shields.io/badge/Option%20A-No--Show%20Prediction-blue.svg)](#)

An enterprise-grade **Machine Learning System** specifically engineered for **Option A: Patient Appointment No-Show Prediction**. The system classifies hospital appointments into **Show-Up Likely** (`0`) vs. **No-Show Likely** (`1`) using clinical, demographic, operational, and historical attendance metrics to optimize hospital scheduling and reduce non-attendance rates.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Problem Statement](#-problem-statement)
- [Dataset Overview](#-dataset-overview)
- [System Architecture](#-system-architecture)
- [Task 06 — Model Evaluation & Selection](#-task-06--model-evaluation--selection)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Running the Streamlit Prototype](#-running-the-streamlit-prototype)
- [Project Team](#-project-team)
- [License](#-license)

---

## 🔍 Overview

Hospital appointment non-attendance (**No-Show**) is a major operational challenge in healthcare, causing schedule inefficiencies, underutilized medical staff, increased wait times for other patients, and substantial financial revenue loss.

**SmartCare AI (Option A)** provides an end-to-end, automated machine learning pipeline to accurately forecast whether a patient will miss their scheduled appointment. By identifying high-risk no-show cases in advance, hospital administration can take proactive intervention measures — such as automated reminder alerts, standby slot allocations, and intelligent overbooking strategies.

---

## ✨ Key Features

- ⚙️ **Automated Data Preprocessing**: Cleaning, scaling (StandardScaler), handling missing values, and categorical encoding (One-Hot & Ordinal Encoding).
- 📊 **Exploratory Data Analysis (EDA)**: Statistical profiling, feature correlation analysis, attendance patterns across age/waiting days, and class distribution checks.
- 🤖 **Multi-Model ML Pipeline**: Training and tuning Logistic Regression, Decision Tree, and Random Forest classifiers for binary no-show prediction.
- 🏆 **Task 06 Model Evaluation**: Rigorous evaluation using Accuracy, Precision, Recall, F1 Score, and ROC-AUC metrics.
- 💡 **Explainable AI (SHAP)**: Global feature importance and local SHAP force plots to understand key drivers influencing patient no-shows.
- 🖥️ **Interactive Streamlit Prototype**: Intuitive web interface for entering patient parameters and predicting real-time no-show probability.
- 🧱 **Modular Architecture**: Decoupled Python packages under `src/` ensuring clean code reusability across notebooks and web apps.

---

## 🎯 Problem Statement (Option A)

Manual tracking of patient attendance leads to unexpected appointment gaps and clinic downtime.

**SmartCare AI — Option A** solves this through automated binary risk classification:
- **Task Type**: Binary Supervised Classification
- **Target Variable**: `no_show`
- **Output Classes**:
  - `0` — **Low Risk (Show-Up Likely)**: Patient is expected to attend the appointment.
  - `1` — **High Risk (No-Show Likely)**: Patient is at risk of missing the appointment.

---

## 📊 Dataset Overview

The dataset consists of **1,000 patient records** containing demographic, clinical, and historical appointment scheduling features:

| Feature Category | Attributes & Variables |
|---|---|
| 👤 **Patient Demographics** | Age, Gender, Blood Group |
| 🩺 **Clinical Metrics** | Systolic BP, Diastolic BP, BP Category, Blood Sugar (mg/dL), Cholesterol (mg/dL), BMI, Diagnosis |
| 🗓️ **Scheduling & History** | Waiting Days, Appointment Month, Day of Week, Previous Appointments Count, Missed Appointments Count, Previous Missed Rate, Has Previous Appointment |
| 🏥 **Hospital Context** | Department |
| 🎯 **Target Variable** | `no_show` (0 = Attended / Show-Up, 1 = Missed / No-Show) |

*Note: Raw datasets belong under `data/raw/` (git-ignored).*

---

## 🏗️ System Architecture

```text
       ┌────────────────────────┐
       │      Raw Dataset       │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Data Preprocessing &   │ ──► data/processed/smartcare_final_preprocessed_dataset.csv
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
       │ Model Training &       │ ──► models/logistic_regression.pkl
       │ Hyperparameter Tuning  │     models/decision_tree.pkl
       └───────────┬────────────┘     models/random_forest.pkl
                   │
                   ▼
       ┌────────────────────────┐
       │ Model Evaluation &     │ ──► reports/task06_evaluation_results.csv
       │ Best Model Selection   │ ──► models/best_model.pkl
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Explainable AI (SHAP)  │ ──► reports/figures/shap/*.png
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Streamlit Prototype    │ ──► Real-Time No-Show Risk Dashboard
       └────────────────────────┘
```

---


## 📂 Project Structure

```text
Smartcare-Ai-Risk-Prediction-/
├── app/                        # Interactive Streamlit prediction dashboard
│   ├── app.py                  # Streamlit entry point for real-time inference
│   └── components/             # Custom UI widgets and components
├── data/
│   ├── raw/                    # Raw dataset and data dictionary (unversioned)
│   └── processed/              # Preprocessed & engineered dataset files
├── models/                     # Saved model artifacts (.pkl files)
│   ├── best_model.pkl          # Selected best performing model (Logistic Regression)
│   ├── decision_tree.pkl       # Decision Tree model artifact
│   ├── logistic_regression.pkl # Logistic Regression model artifact
│   └── random_forest.pkl       # Random Forest model artifact
├── notebooks/                  # Execution notebooks by project task
│   ├── 01_Data Preprocessing And Feature Engineering.ipynb
│   ├── 02_Exploratory Data Analysis.ipynb
│   ├── 03_Model-Evaluation-&-Selection.ipynb
│   ├── 04_model_development.ipynb
│   └── 05_xai_analysis.ipynb
├── presentation/               # Project slides & presentation documentation
├── reports/                    # Generated evaluation charts, matrices, & tables
│   ├── task06_evaluation_results.csv
│   ├── task06_evaluation_percentage.csv
│   └── task06_hyperparameter_summary.csv
├── src/                        # Modular, production-ready Python source code
│   ├── data/                   # Dataset loader & validation modules
│   │   ├── dataset_info.py
│   │   ├── loader.py
│   │   └── validator.py
│   ├── eda/                    # Visualization & statistical utilities
│   │   ├── correlations.py
│   │   ├── distributions.py
│   │   ├── statistics.py
│   │   └── visualizations.py
│   ├── models/                 # Model training, evaluation, & comparison
│   │   ├── compare.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   └── train.py
│   ├── prediction/             # Inference engine for No-Show risk calculation
│   │   └── predictor.py
│   ├── preprocessing/          # Cleaning, scaling, & encoding pipelines
│   │   ├── cleaning.py
│   │   ├── encoding.py
│   │   ├── pipeline.py
│   │   └── scaling.py
│   └── xai/                    # SHAP feature importance & explainability engines
│       ├── feature_importance.py
│       └── shap_explainer.py
├── .gitignore                  # Git ignore configuration
├── LICENSE                     # Project MIT License
├── README.md                   # Complete project documentation
└── requirements.txt            # Python dependencies manifest
```

---

## 🛠️ Tech Stack

| Category | Technologies & Libraries |
|---|---|
| **Language** | Python 3.10+ |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn, Joblib |
| **Explainable AI** | SHAP |
| **Data Visualization** | Matplotlib, Seaborn |
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

### 3️⃣ Setup Virtual Environment

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

## 🚀 Running the Streamlit Prototype

Launch the interactive patient appointment no-show prediction dashboard:

```bash
streamlit run app/app.py
```

Open your web browser at `http://localhost:8501` to test real-time no-show risk inference.

---

## 👥 Project Team

| Student ID | Contributor | GitHub Profile | Primary Task Responsibility |
|---|---|---|---|
| **CIT-23-02-0021** | **Nilupul Thisaranga** | [@N3Edirisinghe](https://github.com/N3Edirisinghe) | **Model Evaluation & Selection (Task 06)** |
| **CIT-23-02-0025** | **Siluna Nusal** | [@GitGuru29](https://github.com/GitGuru29) | **Explainable AI (XAI) & Prototype** |
| **CIT-23-02-0042** | **Dulani Madubashini** | [@cobweb-sudo](https://github.com/cobweb-sudo) | **Exploratory Data Analysis (EDA)** |
| **CIT-23-02-0127** | **Kaveesha Dilshan** | [@Kaveesha23dil](https://github.com/Kaveesha23dil) | **Data Preprocessing & Feature Engineering** |
| **CIT-23-02-0359** | **Zumra Hassan** | [@Zumrahassan222](https://github.com/Zumrahassan222) | **Model Development & Tuning** |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
