from src.data_loader import load_data
from src.preprocess import vectorize
from src.model import train_model, predict
from src.evaluate import evaluate
from predict import predict_text
from src.utils import save_model, load_model

import os

MODEL_PATH="models/svm_model.pkl"

def train_pipeline(train_data, test_data):
    X_train, X_test, vectorizer = vectorize(train_data, test_data)

    model = train_model(X_train, train_data.target)

    save_model(model, vectorizer, MODEL_PATH)
    return model, vectorizer, X_test

def load_pipeline(test_data):
    model, vectorizer = load_model(MODEL_PATH)
    X_test = vectorizer.transform(test_data.data)

    return model, vectorizer, X_test

def main():
    # load data
    train_data, test_data = load_data()

    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        model, vectorizer, X_test = load_pipeline(test_data)
    else:
        print("Training new model...")
        model, vectorizer, X_test = train_pipeline(train_data, test_data)

    # Predict
    y_pred = predict(model, X_test)

    # Evaluate
    evaluate(test_data.target, y_pred, train_data.target_names)

    # Custom test
    print("\n--- Prediction ---")
    text = "The government is discussing new policies in the Middle East"
    results = predict_text(text, model, vectorizer, train_data.target_names)

    print("\nText:", text)
    print("\nTop predictions:")

    for i, res in enumerate(results):
        print(f"{i+1}. {res['label']} ({res['confidence']:.4f})")

if __name__ == "__main__":
    main()
