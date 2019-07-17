import numpy as np
import math
import matplotlib.pyplot as plt

def get_px(x, a, h):
    x_nege = x[np.where(x < 0)]
    px_nege = np.zeros(x_nege.size)
    x_0a = x[(x >= 0) & (x <= a)]
    px_0a = np.zeros(x_0a.size)
    if x_0a.size > 0:
        px_0a = 1.0 / a * (1 - np.exp(-x_0a / h))
    x_a = x[np.where(x > a)]
    px_a = np.zeros(x_a.size)
    if x_a.size > 0:
        px_a = 1.0 / a * (math.exp(a / h) - 1.0) * np.exp(-x_a / h)
    return np.concatenate((px_nege, px_0a, px_a), axis=0)

if __name__ == '__main__':
    '''question 2'''
    h_list = [1, 1 / 4, 1 / 16]
    a = 1.0
    x = np.linspace(start=-1, stop=5, num=1000)
    plt.figure()
    plt.title(r'$\bar{P}(x)\sim x$')
    plt.grid()
    for h in h_list:
        plt.plot(x, get_px(x, a, h), label='h=%.4f' % (h))
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\bar{P}(x)$')
        plt.legend(loc='upper right')
    plt.show()

    '''question 4'''
    h_list = [1, 0.1, 0.01, 0.001, 0.0001]
    x = np.linspace(start=0, stop=0.05, num=1000)
    plt.figure()
    plt.title(r'$\bar{P}(x)\sim x$')
    plt.grid()
    for h in h_list:
        plt.plot(x, get_px(x, a, h), label='h=%.4f' % (h))
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\bar{P}(x)$')
        plt.legend(loc='best')
    plt.show()