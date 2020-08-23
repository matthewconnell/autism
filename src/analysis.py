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

def main(train_X, test_X, train_y, test_y, conf1, conf2, roc_path):

    np.random.RandomState(414)

    warnings.filterwarnings(action='ignore', category=FitFailedWarning)

    # import the already split datasets
    X_train = pd.read_csv(train_X, index_col=0)
    y_train = pd.read_csv(train_y, index_col=0)
    X_test = pd.read_csv(test_X, index_col=0)
    y_test = pd.read_csv(test_y, index_col=0)


    # Test that X_train has more rows the X_test
    try:
        assert(X_train.shape[0] > X_test.shape[0])
    except Exception as bad_size:
        print("X_train should have more rows than X_test.\nDid you put them in the wrong order?")

    # Make validation set 
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=414)

    numeric_features = ["age", 
                        "result"]

    one_hot_features = ["gender", 
                        "ethnicity", 
                        "jaundice", 
                        "country_of_res", 
                        "used_app_before", 
                        "age_desc", 
                        "relation",
                        "Class/ASD"]

    other_columns = list(X_train.columns[0:10])

    preprocessor = ColumnTransformer(sparse_threshold=0,
        transformers=[
            ("scale", 
            StandardScaler(), 
            numeric_features),
            ("one_hot", 
            OneHotEncoder(drop=None, 
                        handle_unknown="ignore"), 
            one_hot_features)
        ])

    X_train_temp = pd.DataFrame(preprocessor.fit_transform(X_train), 
                index = X_train.index,
                columns = (numeric_features + 
                        list(preprocessor
                            .named_transformers_["one_hot"]
                            .get_feature_names(one_hot_features)))
                        )

    X_test_temp = pd.DataFrame(preprocessor.transform(X_test),
                        index = X_test.index,
                        columns = X_train_temp.columns)

    X_valid_temp = pd.DataFrame(preprocessor.transform(X_valid),
                        index = X_valid.index,
                        columns = X_train_temp.columns)

    X_train = X_train_temp.join(X_train[other_columns])
    X_test = X_test_temp.join(X_test[other_columns])
    X_valid = X_valid_temp.join(X_valid[other_columns])

    le = LabelEncoder()

    y_train = le.fit_transform(y_train.to_numpy().ravel())
    y_test = le.transform(y_test.to_numpy().ravel())
    y_valid = le.transform(y_valid.to_numpy().ravel())

    ## Trying Gridsearch on different models to find best

    ## Initialize models
    dt = DecisionTreeClassifier(random_state=1)

    # Make list for models and a list to store their values
    estimators = [dt, rf, svm, knn]
    best_parameters = []
    best_precision_scores = []

    # Make list of dictionaries for parameters
    params = [
            {'max_depth': [1, 5, 10, 15, 20, 25, None],
            'max_features': [3, 5, 10, 15, 20, 50, None]},
            {'min_impurity_decrease': [0, 0.25, 0.5],
            'max_features': [3, 5, 10, 20, 50, 'auto']},
            {'C':[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'gamma':[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
            {'n_neighbors': [2, 5, 10, 15, 20, 50, 100],
            'algorithm': ['auto', 'brute']}
            ]

    # Run for loop to find best parameters for each model
    # Scoring = recall to reduce false positives
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

    preliminary_matrix.to_csv(conf1)

    #print(classification_report(y_test, dt.predict(X_test)))



    # Try all questions:
    dt3 = DecisionTreeClassifier()

    dt3.fit(questions_train_df, y_train)

    conf_matrix = pd.DataFrame(confusion_matrix(y_test, dt.predict(X_test)))

    final_matrix = conf_matrix.rename(columns={0:"Predicted no autism", 1:'Predicted autism'}, 
                index={0:"Does not have autism", 1:'Has autism'})

    final_matrix.to_csv(conf2)

    # ROC curve

    fpr, tpr, _ = roc_curve(y_test, dt.predict_proba(X_test)[:,1])

    roc_df = pd.DataFrame({"fpr":fpr, "tpr":tpr})

    line_df = pd.DataFrame({"start":[0,1], "end":[0,1]})

    roc = alt.Chart(roc_df).mark_line().encode(
        x = alt.X("fpr:Q"),
        y = alt.Y("tpr:Q")
    )
        
    line = alt.Chart(line_df).mark_line(strokeDash=[5,5], color="orange").encode(
        x = alt.X("start:Q", axis=alt.Axis(title="False Positive Rate")),
        y = alt.Y("end:Q", axis=alt.Axis(title="True Positive Rate"))
    )
        
    chart = (roc + line).configure_axis(titleFontSize=20).properties(title="ROC Curve").configure_title(fontSize=20)

    chart

    chart.save(roc_path)

if __name__ == "__main__":
  main()