from src.data_loader import load_data
from src.preprocess import preprocess_pipeline, clean_text
from src.model import train_model, predict, predict_with_proba
from src.evaluate import evaluate
from src.utils import save_model, load_model

import os

MODEL_PATH = "models/svm_model.pkl"


def train_pipeline(train_data, test_data):
    X_train, X_test, vectorizer = preprocess_pipeline(train_data, test_data)

    model = train_model(X_train, train_data.target)

    save_model(model, vectorizer, train_data.target_names, MODEL_PATH)

    return model, vectorizer, train_data.target_names, X_test


def load_pipeline(test_data):
    model, vectorizer, target_names = load_model(MODEL_PATH)

    # clean before transform
    test_texts = [clean_text(doc) for doc in test_data.data]
    X_test = vectorizer.transform(test_texts)

    return model, vectorizer, target_names, X_test


def main():
    # Load data
    train_data, test_data = load_data()

    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        model, vectorizer, target_names, X_test = load_pipeline(test_data)
    else:
        print("Training new model...")
        model, vectorizer, target_names, X_test = train_pipeline(train_data, test_data)

    # Predict label
    y_pred = predict(model, X_test)

    # Evaluate
    evaluate(test_data.target, y_pred, target_names)

    print("\n--- Prediction ---")
    text = "The government is discussing new policies in the Middle East"

    # Clean text before transform
    clean_input = clean_text(text)
    X_input = vectorizer.transform([clean_input])

    results = predict_with_proba(
        model,
        X_input,
        target_names,
        top_k=3
    )

    print("\nText:", text)
    print("\nTop predictions:")

    for i, (label, confidence) in enumerate(results[0]):
        print(f"{i+1}. {label} ({confidence:.2f}%)")


if __name__ == "__main__":
    main()