from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

def train_model(X_train, y_train):

    params = {
        'C' : [0.01, 0.05, 0.1, 0.5, 1, 5]
    }
    # Initialize the GridSearchCV
    grid = GridSearchCV(svm.LinearSVC(class_weight='balanced', random_state= 42),
                        param_grid=params,
                        cv=3,
                        n_jobs= 3,
                        scoring='accuracy',
                        verbose=1)
    
    # Fit the grid search to the data
    grid.fit(X_train, y_train)
    
    calibrated_model = CalibratedClassifierCV(grid.best_estimator_, method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)

    return calibrated_model

def predict(model, X_test):
    return model.predict(X_test)

def predict_with_proba(model, X_test, labels, top_k=3):
    probs = model.predict_proba(X_test)

    results = []
    for prob in probs:
        top_k_local = min(top_k, len(labels))
        top_indices = np.argsort(prob)[::-1][:top_k_local]

        top_preds = [(labels[i], round(prob[i]*100, 2)) for i in top_indices]
        results.append(top_preds)

    return results