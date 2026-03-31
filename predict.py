from src.preprocess import clean_text

def predict_text(text, model, vectorizer, target_names, top_k=3):
    clean_input = clean_text(text)
    X = vectorizer.transform([clean_input])

    probs = model.predict_proba(X)[0]

    top_k = min(top_k, len(target_names))
    top_indices = probs.argsort()[::-1][:top_k]

    formatted = []
    for i in top_indices:
        prob = probs[i] * 100
        formatted.append({
            "label": target_names[i],
            "confidence": round(probs[i], 4),
            "confidence_percent": round(prob, 2)
        })

    return formatted