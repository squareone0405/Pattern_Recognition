import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.figure()
    plt.scatter([0, 1], [0, 1], marker='o', c='black')
    plt.scatter([0, 1], [1, 0], marker='o', edgecolors='black', c='')
    plt.title('XOR')
    plt.xlim((-0.5, 1.5))
    plt.ylim((-0.5, 1.5))
    plt.show()
