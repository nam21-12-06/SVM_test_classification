
def predict_text(text, model, vectorizer, target_names, top_k = 3):
    X = vectorizer.transform([text])

    # Predict probabilities
    probs = model.predict_proba(X)[0]

    # Get top-k indices
    top_k_idx = probs.argsort()[-top_k:][::-1]

    result = []
    for idx in top_k_idx:
        result.append({
            "label": target_names[idx],
            "confidence": round(float(probs[idx]), 6),
            "confidence_percent": round(float(probs[idx]) * 100, 2)
        })

    return result

