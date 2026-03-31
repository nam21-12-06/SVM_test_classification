import re
from src.features import extract_tfidf_features

def clean_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove email
    text = re.sub(r'\S+@\S+', '', text)
    
    # 3. Remove URL
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # 4. Remove all non-alphabet characters
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 5. Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_pipeline(train_data, test_data):
    # APPLY cleaning for both train_data, test_data
    print("--- Cleaning Text ---")
    train_texts = [clean_text(doc) for doc in train_data.data]
    test_texts = [clean_text(doc) for doc in test_data.data]

    # FEATURES EXTRACTION
    X_train, X_test, vectorizer = extract_tfidf_features(train_texts, test_texts)
    
    return X_train, X_test, vectorizer