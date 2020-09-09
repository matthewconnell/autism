from preprocess import *
from analysis import *

clean_path = '../data/clean-data/'

train_X = clean_path + 'Xtrain.csv'
train_y = clean_path + 'ytrain.csv'
test_X = clean_path + 'Xtest.csv'
test_y = clean_path + 'ytest.csv'

def main():

    # preprocess(train_X, test_X, train_y, test_y)
    analysis(train_X, test_X, train_y, test_y)

if __name__ == "__main__":
    main()