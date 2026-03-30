from sklearn import svm
from sklearn.model_selection import GridSearchCV

def train_model(X_train, y_train):

    params = {
        'C' : [0.1, 1, 10, 100]
    }
    # Initialize the GridSearchCV
    grid = GridSearchCV(svm.SVC(kernel='linear', probability= True),
                        param_grid=params,
                        cv=3,
                        n_jobs= -1,
                        scoring='accuracy',
                        verbose=1)
    
    # Fit the grid search to the data
    grid.fit(X_train, y_train)
    
    print("Best params: ", grid.best_params_)
    print("Best CV score: ", grid.best_score_)

    return grid.best_estimator_

def predict(model, X_test):
    return model.predict(X_test)