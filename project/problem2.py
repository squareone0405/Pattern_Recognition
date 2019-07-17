import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models import *
from utils import *
from visualize import *
from feature_select import *

colors = [i for i in get_cmap('tab20').colors]

if __name__ == '__main__':
    train_features, train_labels, test_features, feature_names = load_data2()
    label_data = pd.DataFrame({'labels': train_labels})
    string_labels = np.unique(train_labels)
    string_idx_map = dict(zip(string_labels, range(string_labels.shape[0])))
    idx_string_map = dict(zip(range(string_labels.shape[0]), string_labels))
    train_labels = label_data['labels'].map(string_idx_map).values
    (bin, count) = np.unique(train_labels, return_counts=True)

    X_train, X_test, y_train, y_test = \
        train_test_split(train_features, train_labels, test_size=0.2, shuffle=True)

    model = Model2('svm')
    scores = model.fit_10_ford(X_train, y_train)
    y_pred = model.predict(X_test)
    plot_confuse_matrix(y_pred, y_test, string_labels)

    y_pred = model.predict(test_features)
    write_result(y_pred, idx_string_map)

    '''fs = FeatureSelector('lasso')
    idx = fs.select_feature(X_train, y_train, 10)
    idx = np.sort(idx)
    print(idx)
    print(feature_names[idx])
    X_train = X_train[:, idx]
    X_test = X_test[:, idx]
    df = pd.DataFrame(idx)
    df.to_csv('idx_lasso.csv')
    df = pd.DataFrame(feature_names[idx])
    df.to_csv('name_lasso.csv')'''
