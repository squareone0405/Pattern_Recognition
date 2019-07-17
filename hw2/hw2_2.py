import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

dataset_path = './breast-cancer-wisconsin.txt'

def load_data(file_path, split_ratio):
    dataset_raw = pd.read_csv(filepath_or_buffer=file_path,
                              sep="\t", header=None, na_values=['?']) # read the '?' as nan
    dataset_raw = dataset_raw.dropna() # drop nan
    dataset_raw = dataset_raw.sample(frac=1.0) # shuffle in sample
    X_data = dataset_raw.iloc[:, 1: -1].values
    Y_data = dataset_raw.iloc[:, -1].values.astype('int8')
    train_len = int(Y_data.size * split_ratio)
    X_train = X_data[0: train_len, :]
    X_test = X_data[train_len:, :]
    Y_train = Y_data[0: train_len]
    Y_test = Y_data[train_len:]
    return X_train, Y_train, X_test, Y_test

class FisherClassifier:
    def __init__(self, X_train, Y_train):
        X_train_posi = X_train[np.where(Y_train == 1)[0]]
        X_train_nege = X_train[np.where(Y_train == 0)[0]]
        num_posi = X_train_posi.shape[0]
        num_nege = X_train_nege.shape[0]
        m1 = np.average(X_train_posi, axis=0)
        m2 = np.average(X_train_nege, axis=0)
        self.m = (num_posi * m1 + num_nege * m2) / (num_posi + num_nege)
        Sw = np.dot((X_train_posi - m1).transpose(), (X_train_posi - m1)) + \
             np.dot((X_train_nege - m2).transpose(), (X_train_nege - m2))
        self.weight = np.dot(np.linalg.inv(Sw), (m1 - m2))

    def predict(self, X_test):
        return ((np.dot(X_test - self.m, self.weight) >= 0) + 0).astype('int8')

if __name__ == '__main__':
    ''' question 2 '''
    X_train, Y_train, X_test, Y_test = load_data(dataset_path, 0.75)

    print('logistic regression: ')
    logistic_clf = LogisticRegression()
    logistic_clf.fit(X_train, Y_train)
    print('accuracy on training data: %.2f%%' %
          (logistic_clf.score(X_train, Y_train) * 100))
    print('accuracy on test data: %.2f%%' %
          (logistic_clf.score(X_test, Y_test) * 100))

    print('fisher classifier: ')
    fisher_clf = FisherClassifier(X_train, Y_train)
    print('accuracy on training data: %.2f%%' %
          (np.sum(fisher_clf.predict(X_train) == Y_train) / len(Y_train) * 100))
    print('accuracy on test data: %.2f%%' %
          (np.sum(fisher_clf.predict(X_test) == Y_test) / len(Y_test) * 100))
