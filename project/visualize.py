import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

colors = [i for i in get_cmap('tab20').colors]

class Visualizer:
    def __init__(self, method):
        assert method in ['pca', 'tsne', 'lle']
        self.method = method
        if self.method == 'pca':
            self.compresser = PCA(n_components=2)
        elif self.method == 'tsne':
            self.compresser = TSNE(n_components=2, verbose=1)
        elif self.method == 'lle':
            self.compresser = LocallyLinearEmbedding(n_components=2)

    def visualize2d(self, X, y, labels):
        X_embedded = self.compresser.fit_transform(X)
        plt.figure(figsize=(8, 6.5))
        plt.title('2D Visualization({0})'.format(self.method))
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        for i in range(12):
            idx = y == i
            plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=labels[i], marker='x', c=colors[i])
        plt.legend(loc='best')
        plt.show()

def visulize_2d_all(train_features, train_labels, string_labels):
    visualizer = Visualizer('pca')
    visualizer.visualize2d(train_features, train_labels, string_labels)
    visualizer = Visualizer('lle')
    visualizer.visualize2d(train_features, train_labels, string_labels)
    visualizer = Visualizer('tsne')
    visualizer.visualize2d(train_features, train_labels, string_labels)

def visualize_distrib(train_features, train_labels, string_labels):
    plt.figure()
    plt.title('Feature Distribution')
    for i in np.arange(12):
        idx = train_labels == i
        peaks = np.mean(train_features[idx, :], axis=0)
        plt.scatter(np.arange(train_features.shape[1]), peaks, color=colors[i],
                    marker='|', linewidths=0.1, label=string_labels[i])
    plt.legend()
    plt.show()

def cluster(train_features, train_labels):
    kmeans = KMeans(n_clusters=12)
    kmeans.fit(train_features, train_labels)
    cluster_pred = kmeans.predict(train_features)
    count_same_class = 0
    count_diff_class = 0
    same_same = 0
    diff_same = 0
    for i in range(train_features.shape[0]):
        for j in np.arange(i + 1, train_features.shape[0]):
            if train_labels[i] == train_labels[j]:
                count_same_class += 1
                if cluster_pred[i] == cluster_pred[j]:
                    same_same += 1
            else:
                count_diff_class += 1
                if cluster_pred[i] == cluster_pred[j]:
                    diff_same += 1
    return same_same / count_same_class, diff_same / count_same_class
