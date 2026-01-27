import gradio as gr
import pandas as pd
import numpy as np
import pickle


with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_diabetes(
    Pregnancies, Glucose, BloodPressure,
    SkinThickness, Insulin, BMI,
    DiabetesPedigreeFunction, Age
):
    if not (0 <= Pregnancies <= 20):
        return " Invalid input: Pregnancies must be between 0 and 20."

    if not (50 <= Glucose <= 300):
        return "Invalid input: Glucose must be between 50 and 300."

    if not (40 <= BloodPressure <= 200):
        return "Invalid input: Blood Pressure must be between 40 and 200."

    if not (0 <= SkinThickness <= 100):
        return "Invalid input: Skin Thickness must be between 0 and 100."

    if not (0 <= Insulin <= 300):
        return "Invalid input: Insulin must be between 0 and 300."

    if not (10 <= BMI <= 60):
        return "Invalid input: BMI must be between 10 and 60."
    if not (0 <= DiabetesPedigreeFunction <= 2.5):
        return "Invalid input: Diabetes Pedigree Function out of range."

    if not (0 <= Age <= 120):
        return "Invalid input: Age must be between 0 and 120."

    input_data = pd.DataFrame([[
        Pregnancies, Glucose, BloodPressure,
        SkinThickness, Insulin, BMI,
        DiabetesPedigreeFunction, Age
    ]], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure',
        'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age'
    ])

    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    input_data[zero_cols] = input_data[zero_cols].replace(0, np.nan)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = " Diabetic" if prediction == 1 else "Non-Diabetic"

    return (
        f"Prediction: {result}\n"
        f"Diabetes Probability: {probability * 100:.2f}%"
    )

inputs = [
    gr.Slider(label="Pregnancies", value=0, minimum=0, maximum=20, step=1),
    gr.Number(label="Glucose", value=120, minimum=50, maximum=300),
    gr.Number(label="Blood Pressure", value=80, minimum=40, maximum=200),
    gr.Number(label="Skin Thickness", value=20, minimum=0, maximum=100),
    gr.Number(label="Insulin", value=75, minimum=0, maximum=600),
    gr.Slider(label="BMI", value=25, minimum=10, maximum=60),
    gr.Number(label="Diabetes Pedigree Function", value=0.5, minimum=0, maximum=2.5),
    gr.Slider(label="Age", value=30, minimum=0, maximum=120, step=1),
]

app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs=gr.Text(),
    title="Diabetes Prediction System",
    description="Enter valid medical details to predict \diabetes using a Random Forest model."
)

app.launch(share=True)