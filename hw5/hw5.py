import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers
import time

def load_data():
    npzfile = np.load('mnist.npz')
    X_train = npzfile['X_train'].astype(np.float32) / 255.0
    X_test = npzfile['X_test'].astype(np.float32) / 255.0
    y_train = npzfile['y_train'].astype(np.int64)
    y_test = npzfile['y_test'].astype(np.int64)

    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    idx4_train = np.where(y_train == 4)[0]
    idx9_train = np.where(y_train == 9)[0]
    y_train[idx4_train] = 0 # 4 -> 0, 9 -> 1
    y_train[idx9_train] = 1  # 4 -> 0, 9 -> 1
    idx49_train = np.hstack((idx4_train, idx9_train))
    np.random.shuffle(idx49_train)
    X_train = X_train[idx49_train]
    y_train = y_train[idx49_train]

    idx4_test = np.where(y_test == 4)[0]
    idx9_test = np.where(y_test == 9)[0]
    y_test[idx4_test] = 0  # 4 -> 0, 9 -> 1
    y_test[idx9_test] = 1  # 4 -> 0, 9 -> 1
    idx49_test = np.hstack((idx4_test, idx9_test))
    np.random.shuffle(idx49_test)
    X_test = X_test[idx49_test]
    y_test = y_test[idx49_test]
    return X_train, y_train, X_test, y_test

class NNClassifier():
    def __init__(self):
        model = Sequential()
        model.add(Dense(units=28 * 28, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', activation='relu', name='fc0'))
        model.add(Dense(units=128, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', activation='relu', name='fc1'))
        model.add(Dense(units=1, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', activation='sigmoid', name='fc2'))
        self.model = model

    def fit(self, X_train, y_train):
        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(x=X_train, y=y_train, epochs=20, batch_size=128, verbose=0)

    def score(self, X_test, y_test):
        preds = self.model.evaluate(X_test, y_test)
        return preds[1]


if __name__ == '__main__':
    tic = time.time()
    X_train, y_train, X_test, y_test = load_data()
    '''X_train = X_train[:2000, :]
    y_train = y_train[:2000]'''

    svm_linear_clf = SVC(C=0.2, kernel='linear', verbose=True)
    svm_poly_clf = SVC(C=10000.0, kernel='poly', degree=3, gamma=0.01, verbose=True)
    svm_rbf_clf = SVC(C=10000.0, kernel='rbf', gamma=0.03, verbose=True)
    svm_sigmoid_clf = SVC(C=30.0, kernel='sigmoid', gamma='auto', verbose=True)

    # svm_clf = svm_linear_clf
    # svm_clf = svm_poly_clf
    svm_clf = svm_rbf_clf
    # svm_clf = svm_sigmoid_clf

    svm_clf.fit(X_train, y_train)
    score = svm_clf.score(X_train, y_train)
    print(score)
    score = svm_clf.score(X_test, y_test)
    print('Accuracy of SVM: %.2f%%' % (score * 100))
    print('SVM time elapsed: %.2fs' % (time.time() - tic))

    logistic_clf = LogisticRegression()
    logistic_clf.fit(X_train, y_train)
    score = logistic_clf.score(X_test, y_test)
    print('Accuracy of Logistic Regression: %.2f%%' % (score * 100))

    nn_clf = NNClassifier()
    nn_clf.fit(X_train, y_train)
    score = nn_clf.score(X_test, y_test)
    print('Accuracy of Neural Network: %.2f%%' % (score * 100))
