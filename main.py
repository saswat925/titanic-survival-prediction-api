
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Load the trained ML model
model = joblib.load("models/titanic_survival_model.pkl")

# Create the FastAPI application
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict passenger survival using a trained Machine Learning model",
    version="1.0.0"
)

# Define and validate input data
class PassengerInput(BaseModel):
    pclass: int = Field(ge=1, le=3)
    sex: str
    age: float = Field(ge=0, le=100)
    sibsp: int = Field(ge=0)
    parch: int = Field(ge=0)
    fare: float = Field(ge=0)
    embarked: str

# Create a prediction endpoint
@app.post("/predict")
def predict_survival(passenger: PassengerInput):

    passenger_data = pd.DataFrame([passenger.model_dump()])

    prediction = int(model.predict(passenger_data)[0])
    probability = float(model.predict_proba(passenger_data)[0][1])

    return {
        "prediction": prediction,
        "survival_status": (
            "Likely to survive" if prediction == 1
            else "Likely not to survive"
        ),
        "survival_probability": round(probability, 4)
    }

# Health-check endpoint
@app.get("/")
def home():
    return {
        "message": "Titanic Survival Prediction API is running!"
    }
