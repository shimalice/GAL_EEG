import xgboost as xgb
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

def classifyXGB(X_train, y_train):
    # Choose the type of classifier
    clf = xgb.XGBClassifier(learning_rate=0.05)

    # Choose some parameter combinations to try
    parameters = {'max_depth': [3],
                  'n_estimators': [50, 100]
                 }

    # Type of scoring used to compare parameter combinations
    acc_scorer = make_scorer(accuracy_score)

    # Run the grid search
    grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, y_train)

    # Set the clf to the best combination of parameters
    clf = grid_obj.best_estimator_

    # Fit the best algorithm to the data.
    clf.fit(X_train, y_train)
    return clf
