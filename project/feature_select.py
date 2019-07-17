import numpy as np
from sklearn import feature_selection
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Lasso

class FeatureSelector:
    def __init__(self, separability):
        assert separability in ['fisher', 'mutual_info', 'lasso']
        self.separability = separability

    def select_feature(self, X_train, y_train, feature_num):
        if self.separability == 'fisher':
            return self.select_fisher(X_train, y_train, feature_num)
        elif self.separability == 'mutual_info':
            return self.select_mutual_info(X_train, y_train, feature_num)
        else:
            return self.select_lasso(X_train, y_train, feature_num)

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

    def select_lasso(self, X_train, y_train, feature_num):
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train, y_train)
        nonzero = np.where(np.fabs(lasso.coef_) > 1e-10)[0]
        if nonzero.shape[0] < feature_num:
            return nonzero
        else:
            return np.fabs(lasso.coef_).argsort()[-feature_num:][::-1]
