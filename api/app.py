from fastapi import FastAPI
from pydantic import BaseModel

from src.utils import load_model
from predict import predict_text
from src.data_loader import load_data

app = FastAPI()

# Load model 
MODEL_PATH = "models/svm_model.pkl"
model, vectorizer, target_names = load_model(MODEL_PATH)

# Load label names
train_data, _ = load_data()
target_names = train_data.target_names


# Define input format
class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "API is running"}


# Main endpoint
@app.post("/predict")
def predict(input: TextInput):
    try:
        results = predict_text(
            input.text,
            model,
            vectorizer,
            target_names
        )
        return {"predictions": results}
    except Exception as e:
        return {"error": str(e)}