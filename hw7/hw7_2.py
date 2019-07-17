import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def load_data(data_num):
    npzfile = np.load('mnist.npz')
    X_train = npzfile['X_train'].astype(np.float32) / 255.0
    y_train = npzfile['y_train'].astype(np.int64)

    X_train = X_train.reshape(-1, 28 * 28)
    y_train = np.argmax(y_train, axis=1)
    idx0_train = np.where(y_train == 0)[0]
    idx8_train = np.where(y_train == 8)[0]

    if idx0_train.shape[0] > int(data_num / 2):
        idx0_train = idx0_train[0: int(data_num / 2)]
    if idx8_train.shape[0] > data_num / 2:
        idx8_train = idx8_train[0: int(data_num / 2)]

    y_train[idx0_train] = 0 # 0 -> 0, 8 -> 1
    y_train[idx8_train] = 1 # 0 -> 0, 8 -> 1

    idx08_train = np.hstack((idx0_train, idx8_train))
    np.random.shuffle(idx08_train)
    X_train = X_train[idx08_train]
    y_train = y_train[idx08_train]

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    return X_train, y_train, X_test, y_test

class DimensionCompresser:
    def __init__(self, method):
        assert method in ['pca', 'tsne', 'lle']
        self.method = method
        self.clf = LogisticRegression()

    def fit_transform(self, X, n_components=2):
        if self.method == 'pca':
            self.compresser = PCA(n_components=n_components)
        elif self.method == 'tsne':
            self.compresser = TSNE(n_components=n_components, verbose=1)
        elif self.method == 'lle':
            self.compresser = LocallyLinearEmbedding(n_components=n_components, n_jobs=4)
        return self.compresser.fit_transform(X)

    def visualize2d(self, X, y):
        X_embedded = self.fit_transform(X)
        zero_idx = y == 0
        eight_idx = ~zero_idx
        plt.figure()
        plt.title('First 2 Components')
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.scatter(X_embedded[zero_idx, 0], X_embedded[zero_idx, 1], label='0')
        plt.scatter(X_embedded[eight_idx, 0], X_embedded[eight_idx, 1], label='8')
        plt.legend(loc='best')
        plt.show()

    def transform(self, X):
        return self.compresser.transform(X)

    def classify_origin(self, X_train, y_train, X_test, y_test):
        self.clf.fit(X_train, y_train)
        return self.clf.score(X_train, y_train), self.clf.score(X_test, y_test)

    def classify_compressed(self, X_train, y_train, X_test, y_test, n_components=2):
        if self.method == 'tsne':
            if n_components > 3: # tsne method do not support n_components larger than 3
                return 0, 0
            X_origin = np.vstack((X_train, X_test))
            X_compressed = self.fit_transform(X_origin, n_components)
            X_train_compressed = X_compressed[:int(X_compressed.shape[0] * 0.8)]
            X_test_compressed = X_compressed[int(X_compressed.shape[0] * 0.8):]
        else:
            X_train_compressed = self.fit_transform(X_train, n_components)
            X_test_compressed = self.transform(X_test)
        self.clf.fit(X_train_compressed, y_train)
        return self.clf.score(X_train_compressed, y_train), self.clf.score(X_test_compressed, y_test)

if __name__ == '__main__':
    data_num = 3000
    X_train, y_train, X_test, y_test = load_data(data_num)

    dc = DimensionCompresser('pca')
    # dc.visualize2d(X_train, y_train)
    print('*' * 20 + 'without compression' + '*' * 20)
    print(dc.classify_origin(X_train, y_train, X_test, y_test))

    dc_methods = ['pca', 'lle', 'tsne']
    dims = [2, 3, 5, 10, 30]

    train_scores = np.zeros((len(dc_methods), len(dims)))
    test_scores = np.zeros((len(dc_methods), len(dims)))

    for method_idx in range(len(dc_methods)):
        for dim_idx in range(len(dims)):
            dc = DimensionCompresser(dc_methods[method_idx])
            train_scores[method_idx, dim_idx], test_scores[method_idx, dim_idx] = \
                dc.classify_compressed(X_train, y_train, X_test, y_test, dims[dim_idx])

    train_scores_df = pd.DataFrame(train_scores, index=dc_methods, columns=dims)
    print('*' * 20 + 'train accuracy with compression' + '*' * 20)
    print(train_scores_df)
    test_scores_df = pd.DataFrame(test_scores, index=dc_methods, columns=dims)
    print('*' * 20 + 'test accuracy with compression' + '*' * 20)
    print(test_scores_df)

    '''train_scores_df.to_csv('train.csv')
    test_scores_df.to_csv('test.csv')'''
