from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

def extract_tfidf_features(train_texts, test_texts, max_features=15000):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2), 
        max_df=0.7,                                # Remove words appear multiple times
        min_df= 4 ,                                # Remove words rarely appear
        max_features=max_features,
        sublinear_tf= True
    )
    
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)
    
    return X_train_tfidf, X_test_tfidf, vectorizer

