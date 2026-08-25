# Diabetes-prediction


A machine learning project that builds classification models to predict diabetes risk based on patient health indicators. The system includes data preprocessing, feature engineering, model evaluation, and an interactive demo deployed on Hugging Face using Gradio.

Overview

This project applies classical machine learning techniques to the diabetes prediction task. After thorough data preprocessing and feature engineering, multiple classification models are trained and evaluated. An interactive web demo is provided for real-time predictions.

Key Features





End-to-end classification pipeline



Data preprocessing and feature engineering



Model evaluation using:





Accuracy



Precision



Confusion Matrix



Interactive demo deployed with Hugging Face + Gradio

Tech Stack





Machine Learning: Scikit-learn



Data Processing: Pandas, NumPy



Visualization: Matplotlib / Seaborn



Deployment: Hugging Face Spaces + Gradio



Language: Python

Dataset

The project uses a standard diabetes dataset containing health-related features (e.g., glucose level, BMI, age, blood pressure, etc.). Preprocessing steps include handling missing values, scaling, and feature selection/engineering where applicable.

Methodology





Data Preprocessing – Cleaning, handling missing values, normalization/scaling



Feature Engineering – Creating or selecting relevant features



Model Training – Training classification models



Evaluation – Measuring performance with accuracy, precision, and confusion matrix



Deployment – Interactive Gradio interface hosted on Hugging Face

Project Structure

Diabetes-prediction/
├── notebooks/             # Exploration, training, and evaluation
├── models/                # Saved model files
├── app.py                 # Gradio application
├── data/                  # Dataset or data loading scripts
├── requirements.txt
└── README.md

Installation

git clone https://github.com/rabbi2580/Diabetes-prediction.git
cd Diabetes-prediction
pip install -r requirements.txt

Usage

Local Demo

python app.py

Hugging Face

The interactive demo is available on Hugging Face Spaces (link available in the repository).

Evaluation

Models are evaluated using:





Accuracy



Precision



Confusion Matrix analysis

Future Improvements





Compare multiple algorithms (Logistic Regression, Random Forest, XGBoost, etc.)



Hyperparameter tuning



Feature importance analysis



Improved UI/UX for the Gradio demo



Add model explainability (SHAP / LIME)

Author

Tarif Ul Haider Rabbi
GitHub · Portfolio · LinkedIn

License

This project is open source and available under the MIT License.
