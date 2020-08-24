import pandas as pd
import numpy as np

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

clean_path = '../data/clean-data/'

train_X = clean_path + 'Xtrain.csv'
train_y = clean_path + 'ytrain.csv'
test_X = clean_path + 'Xtest.csv'
test_y = clean_path + 'ytest.csv'

def main(train_X, test_X, train_y, test_y):

    np.random.RandomState(414)

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

    return X_train, X_test, X_valid, y_train, y_test, y_valid

if __name__ == "__main__":
    
    main(train_X, test_X, train_y, test_y)
