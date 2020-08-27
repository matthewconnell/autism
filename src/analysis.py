# analysis.py : take train/test csvs and fit a Decision tree model.
# author: Matthew Connell
# date: August 2020

import pandas as pd
import numpy as np

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer

# Models
from sklearn.tree import DecisionTreeClassifier

# Metrics
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, roc_curve

# Model selection
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

# Suppress warnings
import warnings
from sklearn.exceptions import FitFailedWarning

from preprocess import *

def analysis():

    X_train, X_test, X_valid, y_train, y_test, y_valid = preprocess(train_X, test_X, train_y, test_y)

    ## Initialize models
    dt = DecisionTreeClassifier(random_state=1)

    ## Make list for models and a list to store their values
    estimators = [dt]
    best_parameters = []
    best_precision_scores = []

    ## Make list of dictionaries for parameters
    params = [
            {'max_depth': [1, 5, 10, 15, 20, 25, None],
            'max_features': [3, 5, 10, 15, 20, 50, None]}
            ]

    # Run for loop to find best parameters for each model
    # Scoring = precision to reduce false negatives
    for i in range(len(estimators)):
        search = GridSearchCV(estimator=estimators[i], 
                            param_grid=params[i],
                            cv = 10,
                            n_jobs=-1,
                            scoring='precision')
        
        search_object = search.fit(X_train, y_train)
        
        # Store the output on each iteration
        best_parameters.append(search_object.best_params_)
        best_precision_scores.append(search_object.best_score_)

    best_parameters[np.argmax(best_precision_scores)]


    # the best precision score comes from a decision tree classifier with max_depth=15 and max_features=50
    # and precision = 0.46

    dt = DecisionTreeClassifier(max_depth=15, max_features=50)
    dt.fit(X_train, y_train).score(X_train, y_train)

    dt.score(X_valid, y_valid)

    # and ~81% on the validation set

    prelim_matrix = pd.DataFrame(confusion_matrix(y_valid, dt.predict(X_valid)))


    preliminary_matrix = prelim_matrix.rename(columns={0:"Predicted no autism", 1:'Predicted autism'}, 
                index={0:"Does not have autism", 1:'Has autism'})

    # Try all questions:
    dt3 = DecisionTreeClassifier()

    dt3.fit(train_X, y_train)

    conf_matrix = pd.DataFrame(confusion_matrix(y_test, dt.predict(X_test)))

    final_matrix = conf_matrix.rename(columns={0:"Predicted no autism", 1:'Predicted autism'}, 
                index={0:"Does not have autism", 1:'Has autism'})
