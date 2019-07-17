import numpy as np
import scipy.io as scio
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import time

data_path = 'Sogou_webpage.mat'

def load_data(path):
    data = scio.loadmat(path)
    feature = data['wordMat']
    label = data['doclabel']
    label = label[:, 0]
    return feature, label

def plot_tree(clf, filename):
    import graphviz
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render(filename)

class TreeNode:
    def __init__(self, depth, features, labels, impurity):
        self.depth = depth
        self.features = features
        self.labels = labels
        self.impurity = impurity
        self.best_feature = None
        self.left_child = None
        self.right_child = None

    def have_child(self):
        return (self.left_child) or (self.right_child)

class MyDecisionTree:
    def __init__(self, max_depth, impurity_thres, info_gain_thres, impurity_metric):
        self.classes = None
        self.feature_dem = None
        self.features = None
        self.labels = None
        self.root_node = None
        self.max_depth = max_depth
        self.impurity_thres = impurity_thres
        self.info_gain_thres = info_gain_thres
        if impurity_metric in ['gini', 'entropy']:
            self.impurity_metric = impurity_metric
        else:
            self.impurity_metric = 'gini'

    def GenerateTree(self, X_train, y_train):
        self.classes = np.unique(y_train).shape[0]
        self.feature_dem = X_train.shape[1]
        self.features = X_train
        self.labels = y_train

        self.root_node = TreeNode(0, self.features, self.labels, impurity=np.inf) # entropy is not used for root
        node_stack = [self.root_node] # DFS
        while len(node_stack) > 0:
            top = node_stack.pop()
            left_child, right_child = self.SplitNode(top)
            if left_child:
                node_stack.append(left_child)
            if right_child:
                node_stack.append(right_child)

    def SplitNode(self, node):
        if node.depth >= self.max_depth or node.impurity < self.impurity_thres:
            return None, None
        best_feature, entropy_left, entropy_right, info_gain = self.SelectFeature(node)
        if info_gain < self.info_gain_thres:
            return None, None
        node.best_feature = best_feature
        idx_left = node.features[:, node.best_feature] == 1
        idx_right = ~idx_left
        node.left_child = TreeNode(node.depth + 1, node.features[idx_left],
                                       node.labels[idx_left], entropy_left)
        node.right_child = TreeNode(node.depth + 1, node.features[idx_right],
                                        node.labels[idx_right], entropy_right)
        return node.left_child, node.right_child

    def SelectFeature(self, node):
        count_left = np.zeros((self.feature_dem, self.classes))
        count_right = np.zeros((self.feature_dem, self.classes))
        for idx in range(self.classes):
            feature_rows = np.where(node.labels == idx)[0]
            feature_temp = node.features[feature_rows, :]
            count_left[:, idx] = np.sum(feature_temp, axis=0)
            count_right[:, idx] = feature_rows.shape[0] - count_left[:, idx]
        p_left = np.mean(node.features, axis=0)
        p_right = 1 - p_left
        impurity_left = self.Impurity(count_left)
        impurity_right = self.Impurity(count_right)
        feature_impurity = p_left * impurity_left + p_right * impurity_right
        best_idx = np.argmin(feature_impurity)
        info_gain = node.impurity - feature_impurity[best_idx]
        '''print('\t' * node.depth + 'depth = %d, feature = %d, gini = %f'
              % (node.depth, best_idx, feature_impurity[best_idx]))
        print(count_left[best_idx, :])
        print(count_right[best_idx, :])'''
        return best_idx, impurity_left[best_idx], impurity_right[best_idx], info_gain

    def Impurity(self, count_mat):
        total = np.sum(count_mat, axis=1, keepdims=True)
        prob_mat = np.zeros_like(count_mat)
        if np.any(total == 0):
            prob_mat[np.where(total == 0)[0], :] = 0
            prob_mat[np.where(total != 0)[0], :] = count_mat[np.where(total != 0)[0], :] / \
                                                   total[np.where(total != 0)[0]]
        else:
            prob_mat = count_mat / total
        if self.impurity_metric == 'gini':
            return np.sum(prob_mat * (1 - prob_mat), axis=1) # maximum = 1
        else:
            temp = prob_mat.flatten()
            log_mat = np.zeros_like(temp)
            non_zero = temp > 0
            log_mat[non_zero] = np.log2(temp[non_zero])
            log_mat = log_mat.reshape(prob_mat.shape)
            return -0.5 * np.sum(log_mat * prob_mat, axis=1)

    def Decision(self, X_test):
        y_pred = np.zeros(X_test.shape[0])
        for idx in range(X_test.shape[0]):
            curr_node = self.root_node
            while curr_node.have_child():
                split_feature = curr_node.best_feature
                if X_test[idx, split_feature] == 1:
                    if curr_node.left_child is None:
                        break
                    curr_node = curr_node.left_child
                else:
                    if curr_node.right_child is None:
                        break
                    curr_node = curr_node.right_child
            (bin, counts) = np.unique(curr_node.labels, return_counts=True)
            y_pred[idx] = bin[np.argmax(counts)]
        return y_pred

    ''' sklearn style functions '''
    def fit(self, X_train, y_train):
        self.GenerateTree(X_train, y_train)

    def predict(self, X_test):
        return self.Decision(X_test)

    def score(self, X_test, y_test):
        y_pred = self.Decision(X_test)
        return np.sum(y_pred == y_test) / y_test.shape[0]

def parametersGrip(X_train, y_train, X_val, y_val):
    impurity_metrics = ['gini', 'entropy']
    max_depths = [10, 20, 30, 40, 50, 60, 70, 80]
    threses = [0.1, 0.01, 0.001, 0.0001]
    train_scores = np.zeros((len(impurity_metrics), len(max_depths), len(threses)))
    val_scores = np.zeros((len(impurity_metrics), len(max_depths), len(threses)))
    for idx_metric in range(len(impurity_metrics)):
        for idx_depth in range(len(max_depths)):
            for idx_thres in range(len(threses)):
                clf = MyDecisionTree(max_depths[idx_depth], threses[idx_thres],
                                     threses[idx_thres], impurity_metrics[idx_metric])
                clf.fit(X_train, y_train)
                train_scores[idx_metric, idx_depth, idx_thres] = clf.score(X_train, y_train)
                val_scores[idx_metric, idx_depth, idx_thres] = clf.score(X_val, y_val)
    pd.DataFrame(train_scores[0, :, :], index=max_depths, columns=threses).to_csv('train_gini.csv')
    pd.DataFrame(train_scores[1, :, :], index=max_depths, columns=threses).to_csv('train_entropy.csv')
    pd.DataFrame(val_scores[0, :, :], index=max_depths, columns=threses).to_csv('val_gini.csv')
    pd.DataFrame(val_scores[1, :, :], index=max_depths, columns=threses).to_csv('val_entropy.csv')

def classifyMine(X_train, y_train, X_test, y_test):
    clf = MyDecisionTree(max_depth=60, impurity_thres=0.01,
                         info_gain_thres=0.01, impurity_metric='gini')
    tic = time.time()
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print('-' * 20 + 'Using My DecisionTreeClassifier' + '-' * 20)
    print("time elapsed: %f" % (time.time() - tic))
    print("train score: %.2f%%" % (train_score * 100))
    print("test score: %.2f%%" % (test_score * 100))
    return train_score, test_score

def classifyDTree(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0005,
        min_impurity_split=None,
        class_weight=None,
        presort=False)
    tic = time.time()
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print('-' * 20 + 'Using Sklearn DecisionTreeClassifier' + '-' * 20)
    print("time elapsed: %f" % (time.time() - tic))
    print("train score: %.2f%%" % (train_score * 100))
    print("test score: %.2f%%" % (test_score * 100))
    return train_score, test_score

def classifyRF(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0001,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None)
    tic = time.time()
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print('-' * 20 + 'Using Sklearn RandomForestClassifier' + '-' * 20)
    print("time elapsed: %f" % (time.time() - tic))
    print("train score: %.2f%%" % (train_score * 100))
    print("test score: %.2f%%" % (test_score * 100))
    return train_score, test_score

if __name__ == '__main__':
    feature, label = load_data(data_path)
    label = label - 1 # make label start with zero
    X_train, X_temp, y_train, y_temp = train_test_split(feature, label, test_size=0.40, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, shuffle=True)

    # parametersGrip(X_train, y_train, X_val, y_val)

    epoches = 10
    my_scores = np.zeros((2, epoches))
    dt_scores = np.zeros((2, epoches))
    rf_scores = np.zeros((2, epoches))
    for i in range(epoches):
        X_train, X_temp, y_train, y_temp = train_test_split(feature, label, test_size=0.40, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, shuffle=True)
        my_scores[0, i], my_scores[1, i] = classifyMine(X_train, y_train, X_test, y_test)
        dt_scores[0, i], dt_scores[1, i] = classifyDTree(X_train, y_train, X_test, y_test)
        rf_scores[0, i], rf_scores[1, i] = classifyRF(X_train, y_train, X_test, y_test)
    print('+' * 50)
    print('my train score: %f, test score: %f' % (np.mean(my_scores[0, :]), np.mean(my_scores[1, :])))
    print('dt train score: %f, test score: %f' % (np.mean(dt_scores[0, :]), np.mean(dt_scores[1, :])))
    print('rf train score: %f, test score: %f' % (np.mean(rf_scores[0, :]), np.mean(rf_scores[1, :])))

