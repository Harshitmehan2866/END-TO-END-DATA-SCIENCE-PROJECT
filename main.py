# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the trained model
with open("model/iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Create FastAPI app instance
app = FastAPI(title="Iris Flower Classifier API")

# Define input format using Pydantic
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API"}

# Predict endpoint
@app.post("/predict")
def predict_species(data: IrisInput):
    # Convert input to list for model
    input_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    
    # Get prediction
    prediction = model.predict(input_data)[0]

    # Map prediction to species name
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    species_name = species_map[prediction]

    return {"predicted_class": prediction, "species_name": species_name}
