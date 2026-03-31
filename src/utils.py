import pickle

def save_model(model, vectorizer, target_names, filename="svm_model.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump((model, vectorizer, target_names), f)

def load_model(filename="svm_model.pkl"):
    with open(filename, 'rb') as f:
        model, vectorizer, target_names = pickle.load(f)
    return model, vectorizer, target_names