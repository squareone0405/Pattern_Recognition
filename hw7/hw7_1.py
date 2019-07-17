import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import manifold
import pandas as pd

plt.rcParams['font.sans-serif']=['SimHei'] # to show chinese label
plt.rcParams['axes.unicode_minus']=False # to show minus

def load_data():
    df = pd.read_excel('./题目1.xlsx', sheet_name='Sheet1')
    return df.values[:, 0].squeeze(), df.values[:, 1:]

if __name__ == '__main__':
    labels, distance_mat = load_data()
    mds = manifold.MDS(dissimilarity='precomputed')
    pos = mds.fit(distance_mat).embedding_

    ''' scatter '''
    plt.figure()
    plt.title(r'$City\ Distribution$')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    for i in range(pos.shape[0]):
        plt.scatter(pos[i, 0], pos[i, 1], label=labels[i], linewidths=3)
    plt.legend(loc='best')

    ''' draw lines '''
    segments = [[pos[i, :], pos[j, :]] for i in range(len(pos)) for j in range(len(pos))]
    lc = LineCollection(segments)
    lc.set_linewidths(np.full(len(segments), 0.2))
    plt.gca().add_collection(lc)
    plt.show()
