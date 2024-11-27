from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Cargar el pipeline entrenado
model = joblib.load("pipeline.joblib")

# Crear una aplicación FastAPI
app = FastAPI()

# Definir la estructura esperada para solicitudes individuales
class InputData(BaseModel):
    age: float
    trtbps: float
    chol: float
    thalachh: float
    oldpeak: float

# Definir la estructura esperada para predicciones múltiples
class DataPredict(BaseModel):
    data_to_predict: list[list] = [[45, 145, 233, 150, 2.3, 0, 1, 0, 0, 0, 1, 1, 0]]

# Ruta inicial o "home"
@app.get("/")
def home():
    return {"Universidad EIA": "MLOps"}

# Ruta de estado del API
@app.get("/status")
def status():
    return {"message": "API for Logistic Regression Model is Running!"}

# Endpoint para predicciones individuales
@app.post("/predict_batch")
def predict_batch(data: DataPredict):
    try:
        # Convertir los datos en un DataFrame con los nombres de las columnas esperados
        columns = [
            'age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 
            'caa', 'sex', 'cp', 'fbs', 'restecg', 
            'slp', 'exng', 'thall'
        ]
        input_features = pd.DataFrame(data.data_to_predict, columns=columns)

        # Hacer las predicciones usando el pipeline
        predictions = model.predict(input_features)

        # Retornar las predicciones como una lista
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")



# Endpoint para predicciones múltiples
@app.post("/predict_batch")
def predict_batch(data: DataPredict):
    try:
        # Convertir los datos en un arreglo numpy
        input_features = np.array(data.data_to_predict)

        # Hacer las predicciones
        predictions = model.predict(input_features)

        # Retornar las predicciones como respuesta
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")
