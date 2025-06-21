# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("diabetes_prediction_model.pkl")

class InputData(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[data.gender, data.age, data.hypertension, data.heart_disease,
                             data.smoking_history, data.bmi, data.HbA1c_level, data.blood_glucose_level]])
    
    # TODO: Encode 'gender' and 'smoking_history' properly
    # Example assumes already preprocessed before training
    
    prediction = model.predict(input_array)
    return {"prediction": str(prediction[0])}
