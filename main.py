ofrom fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Cargar el pipeline entrenado
model = joblib.load("pipeline.joblib")
app = FastAPI()

class InputData(BaseModel):
    age: float
    trtbps: float
    chol: float
    thalachh: float
    oldpeak: float

class DataPredict(BaseModel):
    data_to_predict: list[list] = [[45, 145, 233, 150, 2.3, 0, 1, 0, 0, 0, 1, 1, 0]]

# Ruta inicial o "home"
@app.get("/")
def home():
    return {"Universidad EIA": "MLOps"}

@app.get("/status")
def status():
    return {"message": "API for Logistic Regression Model is Running!"}

@app.post("/predict_batch")
def predict_batch(data: DataPredict):
    try:
        columns = [
            'age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 
            'caa', 'sex', 'cp', 'fbs', 'restecg', 
            'slp', 'exng', 'thall'
        ]
        input_features = pd.DataFrame(data.data_to_predict, columns=columns)
        predictions = model.predict(input_features)

        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")

@app.post("/predict_batch")
def predict_batch(data: DataPredict):
    try:
        input_features = np.array(data.data_to_predict)
        predictions = model.predict(input_features)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")
