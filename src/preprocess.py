from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize(train_data, test_data):
    vectorizer = TfidfVectorizer()
    #stop_words="english",                               max_df= 0.9 min_df=2,ngram_range=(1,2)
    # Removes common words

    X_train = vectorizer.fit_transform(train_data.data) # Extract raw text documents from sklearn dataset (Bunch object)
    X_test = vectorizer.transform(test_data.data)     # Transform test data using the same vocabulary learned from training data
    
    return X_train, X_test, vectorizer


