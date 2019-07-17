import numpy as np
import scipy.io as scio
import pandas as pd
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pickle

from models import *
from utils import *
from visualize import *
from feature_select import *

pkl_filename = 'pca_model.pkl'

if __name__ == '__main__':
    train_features, train_labels, test_features = load_data1()
    label_data = pd.DataFrame({'labels': train_labels})
    string_labels = np.unique(train_labels)
    string_idx_map = dict(zip(string_labels, range(string_labels.shape[0])))
    idx_string_map = dict(zip(range(string_labels.shape[0]), string_labels))
    train_labels = label_data['labels'].map(string_idx_map).values
    (bin, count) = np.unique(train_labels, return_counts=True)

    '''X_train, X_test, y_train, y_test = \
        train_test_split(train_features, train_labels, test_size=0.2, shuffle=True)'''

    ''' random '''
    '''random_idx = np.random.choice(train_features.shape[0], 1000)
    X_train_random = X_train[:, random_idx]
    X_test_random = X_test[:, random_idx]

    X_train_mean = np.mean(X_train_random, axis=0)
    X_train_std = np.std(X_train_random, axis=0)
    nonzero = np.where(X_train_mean > 1e-10)[0]
    X_train_random[:, nonzero] = (X_train_random[:, nonzero] - X_train_mean[nonzero]) / X_train_std[nonzero]
    X_test_random[:, nonzero] = (X_test_random[:, nonzero] - X_train_mean[nonzero]) / X_train_std[nonzero]

    model = Model1('nn')
    score = model.fit(X_train_random, y_train)
    plot_confuse_matrix(model.predict(X_test_random), y_test, string_labels)'''

    ''' peak '''
    '''mask = np.array([False] * train_features.shape[1])
    for i in np.arange(12):
        idx = y_train == i
        sum = np.sum(X_train[idx, :], axis=0)
        peak_idx = sum.argsort()[-1000:][::-1]
        mask[peak_idx] = True
        peaks = sum[peak_idx]
    print(np.sum(mask))
    X_train_peak = X_train[:, mask]
    fs = FeatureSelector('mutual_info')
    selected = fs.select_feature(X_train_peak, y_train, 1000)
    X_train_peak = X_train_peak[:, selected]

    X_test_peak = X_test[:, mask]
    X_test_peak = X_test_peak[:, selected]

    X_train_mean = np.mean(X_train_peak, axis=0)
    X_train_std = np.std(X_train_peak, axis=0)
    X_train_peak = (X_train_peak - X_train_mean) / X_train_std
    X_test_peak = (X_test_peak - X_train_mean) / X_train_std

    model = Model1('nn')
    score = model.fit(X_train_peak, y_train)
    plot_confuse_matrix(model.predict(X_test_peak), y_test, string_labels)'''

    ''' pca '''
    '''pca = PCA(n_components=0.999)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    X_train_mean = np.mean(X_train_pca, axis=0)
    X_train_std = np.std(X_train_pca, axis=0)
    X_train_pca = (X_train_pca - X_train_mean) / X_train_std
    X_test_pca = (X_test_pca - X_train_mean) / X_train_std

    visualize_distrib(X_train_pca, y_train, string_labels)
    
    print(X_train_pca.shape)

    model = Model1('adaboost')
    score = model.fit(X_train_pca, y_train)
    plot_confuse_matrix(model.predict(X_test_pca), y_test, string_labels)'''

    pca = PCA(n_components=0.999)
    pca.fit(train_features)
    X_train = pca.transform(train_features)
    X_test = pca.transform(test_features)

    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    model = Model1('adaboost')
    model.fit(X_train, train_labels)
    y_pred = model.predict(X_test)
    write_result(y_pred, idx_string_map)
