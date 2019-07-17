import numpy as np
import scipy.io as scio
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

test_features_path1 = '1/test_features.txt'
train_features_path1 = '1/train_features.txt'
train_labels_path1 = '1/train_labels.txt'

test_features_path2 = '2/test_features.txt'
train_features_path2 = '2/train_features.txt'
train_labels_path2 = '2/train_labels.txt'

def load_data1():
    train_features = scio.mmread(train_features_path1)
    test_features = scio.mmread(test_features_path1)
    train_labels = np.loadtxt(train_labels_path1, dtype=str)
    '''np.save('train_feature_origin.npy', train_features.toarray().astype(np.int16).transpose())
    np.save('test_feature_origin.npy', test_features.toarray().astype(np.int16).transpose())
    np.save('train_labels.npy', train_labels)'''
    return train_features.toarray().transpose(), train_labels, test_features.toarray().transpose()

def load_npy():
    train_features = np.load('train_feature_origin.npy')
    test_features = np.load('test_feature_origin.npy')
    train_labels = np.load('train_labels.npy')
    return train_features, train_labels, test_features

def load_data2():
    train_features = np.loadtxt(train_features_path2, dtype=str)
    test_features = np.loadtxt(test_features_path2, dtype=str)
    train_labels = np.loadtxt(train_labels_path2, dtype=str)
    feature_names = train_features[:, 0]
    train_features = train_features[:, 1:].astype(np.float64).transpose()
    test_features = test_features[:, 1:].astype(np.float64).transpose()
    return train_features, train_labels, test_features, feature_names

def plot_confuse_matrix(pred, label, classes_name):
    cm = confusion_matrix(pred, label)
    cm_plot = np.zeros((cm.shape[0] + 1, cm.shape[1] + 1), dtype=np.float32)
    cm_plot[0: cm.shape[0], 0: cm.shape[1]] = cm
    sum = 0.0
    for i in range(cm.shape[0]):
        if cm_plot[i, i] == 0:
            cm_plot[i, -1] = 0
        else:
            cm_plot[i, -1] = cm_plot[i, i] / cm_plot[i, :].sum()
        sum += cm_plot[i, i]
    for j in range(cm.shape[1]):
        cm_plot[-1, j] = cm_plot[j, j] / cm_plot[:, j].sum()
    cm_plot[-1, -1] = sum / np.sum(np.sum(cm, axis=0))
    cm_plot = cm_plot.transpose()

    plt.figure(figsize=(10.5, 9))
    plt.title('Confuse Matrix')
    plt.xlabel('Preds')
    plt.ylabel('Labels')
    plt.xticks(range(classes_name.shape[0]), classes_name)
    plt.yticks(range(classes_name.shape[0]), classes_name)
    plt.imshow(cm_plot, cmap=plt.get_cmap('Blues'))
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_plot.max() / 2.
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            if i < cm_plot.shape[0] - 1 and j < cm_plot.shape[1] - 1:
                plt.gca().text(j, i, format(int(cm_plot[i, j]), 'd'), ha="center", va="center",
                               color="white" if cm_plot[i, j] > thresh else "black")
            else:
                plt.gca().text(j, i, format(cm_plot[i, j], '.3f'), ha="center", va="center", color="black")
    plt.show()

def write_result(y_pred, idx_string_map):
    with open('result.txt', 'w') as file:
        for i in range(y_pred.shape[0]):
            file.write(idx_string_map[y_pred[i]])
            if i != y_pred.shape[0] - 1:
                file.write('\t')
