import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import feature_selection
from sklearn.feature_selection import SelectKBest


def load_data():
    X = np.loadtxt('./feature_selection_X.txt')
    Y = np.loadtxt('./feature_selection_Y.txt')
    return X, Y


class FeatureSelector:
    def __init__(self, separability, classifier):
        assert separability in ['fisher', 'mutual_info']
        assert classifier in ['logistic', 'svm']
        self.separability = separability
        if classifier == 'logistic':
            self.clf = LogisticRegression()
        else:
            self.clf = SVC()

    def select_feature(self, X_train, y_train, feature_num):
        if self.separability == 'fisher':
            return self.select_fisher(X_train, y_train, feature_num)
        else:
            return self.select_mutual_info(X_train, y_train, feature_num)

    def select_mutual_info(self, X_train, y_train, feature_num):
        skb = SelectKBest(feature_selection.mutual_info_classif, k=feature_num)
        skb.fit(X_train, y_train)
        return skb.get_support(indices=True)

    def select_fisher(self, X_train, y_train, feature_num):
        X_train_pos = X_train[y_train == 1]
        X_train_neg = X_train[y_train == 0]
        X_train_pos_mean = np.mean(X_train_pos, axis=0)
        X_train_pos_var = np.var(X_train_pos, axis=0)
        X_train_neg_mean = np.mean(X_train_neg, axis=0)
        X_train_nge_var = np.var(X_train_neg, axis=0)
        Jf = np.square(X_train_pos_mean - X_train_neg_mean) / (X_train_pos_var + X_train_nge_var)
        return Jf.argsort()[-feature_num:][::-1]

    def score_raw(self, X, y):
        data_num = X.shape[0]
        shuffle_idx = np.random.permutation(data_num)
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        val_mask = np.array([False] * data_num)
        val_num = int(0.1 * data_num)
        val_mask[0:val_num] = True
        scores = np.zeros(10)
        for i in range(10):
            train_mask = ~val_mask
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_val = X[val_mask]
            y_val = y[val_mask]
            self.clf.fit(X_train, y_train)
            scores[i] = self.clf.score(X_val, y_val)
            val_mask = np.roll(val_mask, val_num)
        return scores

    def score_selected(self, X, y, feature_num):
        data_num = X.shape[0]
        shuffle_idx = np.random.permutation(data_num)
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        val_mask = np.array([False] * data_num)
        val_num = int(0.1 * data_num)
        val_mask[0:val_num] = True
        features = np.zeros((10, feature_num))
        scores = np.zeros(10)
        for i in range(10):
            train_mask = ~val_mask
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_val = X[val_mask]
            y_val = y[val_mask]
            feature_selected = self.select_feature(X_train, y_train, feature_num)
            features[i, :] = feature_selected
            self.clf.fit(X_train[:, feature_selected], y_train)
            scores[i] = self.clf.score(X_val[:, feature_selected], y_val)
            val_mask = np.roll(val_mask, val_num)
        return features, scores

if __name__ == '__main__':
    X, y = load_data()

    clfs = ['logistic', 'svm']
    seps = ['fisher', 'mutual_info']

    for clf in clfs:
        for sep in seps:
            print('*' * 20 + clf + '\t' + sep + '*' * 20)
            fs = FeatureSelector(sep, clf)

            print('-' * 15 + 'with all features' + '-' * 15)
            scores = fs.score_raw(X, y)
            print('scores:')
            print(scores)
            print('average error rate %.3f%%' % (100 * (1 - np.mean(scores))))

            print('-' * 15 + 'with selected features' + '-' * 15)
            features, scores = fs.score_selected(X, y, feature_num=5)
            print('features:')
            print(features)
            print('scores:')
            print(scores)
            print('average error rate %.3f%%' % (100 * (1 - np.mean(scores))))


