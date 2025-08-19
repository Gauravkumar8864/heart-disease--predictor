import gradio as gr
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "stacking_pipeline.joblib"

def predict_heart_disease(Age, Sex, ChestPainType, RestingBP, Cholesterol,
                          FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope):
    model = joblib.load(MODEL_PATH)
    new_data = pd.DataFrame([{
        "Age": Age,
        "Sex": Sex,
        "ChestPainType": ChestPainType,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "RestingECG": RestingECG,
        "MaxHR": MaxHR,
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": Oldpeak,
        "ST_Slope": ST_Slope
    }])
    pred = model.predict(new_data)[0]
    prob = model.predict_proba(new_data)[0][1]
    return f"Prediction: {'Heart Disease' if pred==1 else 'No Heart Disease'} (prob={prob:.2f})"

with gr.Blocks() as demo:
    gr.Markdown("# Heart Disease Predictor")
    with gr.Row():
        Age = gr.Number(label="Age", value=50)
        Sex = gr.Radio(["M","F"], label="Sex", value="M")
        ChestPainType = gr.Radio(["ATA","NAP","ASY","TA"], label="Chest Pain Type", value="ASY")
    with gr.Row():
        RestingBP = gr.Number(label="Resting BP", value=120)
        Cholesterol = gr.Number(label="Cholesterol", value=200)
        FastingBS = gr.Radio([0,1], label="Fasting BS", value=0)
    with gr.Row():
        RestingECG = gr.Radio(["Normal","ST","LVH"], label="Resting ECG", value="Normal")
        MaxHR = gr.Number(label="Max HR", value=150)
        ExerciseAngina = gr.Radio(["Y","N"], label="Exercise Angina", value="N")
    with gr.Row():
        Oldpeak = gr.Number(label="Oldpeak", value=1.0)
        ST_Slope = gr.Radio(["Up","Flat","Down"], label="ST Slope", value="Flat")

    btn = gr.Button("Predict")
    out = gr.Textbox(label="Result")

    btn.click(
        fn=predict_heart_disease,
        inputs=[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope],
        outputs=out
    )

if __name__ == "__main__":
    demo.launch()