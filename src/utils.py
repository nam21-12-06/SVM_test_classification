import pickle

def save_model(model, vectorizer, filename="svm_model.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump((model, vectorizer), f)

def load_model(filename="svm_model.pkl"):
    with open(filename, 'rb') as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer