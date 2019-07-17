import numpy as np
import matplotlib.pyplot as plt
import math

def plot_gaussian(mu, sigma):
    x = np.linspace(start=-4.0, stop=4.0, num=1000)
    y = 1.0 / (math.sqrt(2.0 * math.pi) * sigma) * \
        np.exp(-1.0 * np.multiply(x - mu, x - mu) / (2 * sigma * sigma))
    return x, y

def plot_uniform(start, end):
    x = np.linspace(start=-5.0, stop=5.0, num=20001)
    y = np.zeros((x.size, ), dtype='uint8')
    y[(x > start) & (x < end)] = 1
    return x, y

if __name__== '__main__':
    sample_sizes = [10, 100, 1000]
    round = 3
    '''gaussian'''
    for size in sample_sizes:
        plt.figure()
        plt.title('gaussian with sample size = %d' % (size))
        plt.grid()
        x_plot, y_plot = plot_gaussian(0.0, 1.0)
        plt.plot(x_plot, y_plot, label=r'$N(%.2f,%.2f)$' % (0.0, 1.0))
        for i in range(round):
            samples = np.random.normal(loc=0.0, scale=1.0, size=size)
            mu_eval = np.average(samples)
            sigma_eval = math.sqrt(np.dot((samples - mu_eval).transpose(), (samples - mu_eval)) / size)
            x_plot, y_plot = plot_gaussian(mu_eval, sigma_eval)
            plt.plot(x_plot, y_plot, label=r'$N(%.2f,%.2f)$' % (mu_eval, sigma_eval))
        plt.legend(loc='upper right')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$p(x)$')
        plt.show()
    '''uniform'''
    samples = np.random.uniform(low=0.0, high=1.0, size=100)
    plt.figure()
    plt.title('fit uniform with gaussian')
    plt.grid()
    x_plot, y_plot = plot_uniform(0.0, 1.0)
    plt.plot(x_plot, y_plot, label=r'$U(%.3f,%.3f)$' % (0.0, 1.0))
    mu_eval = np.average(samples)
    sigma_eval = math.sqrt(np.dot((samples - mu_eval).transpose(), (samples - mu_eval)) / samples.size)
    x_plot, y_plot = plot_gaussian(mu_eval, sigma_eval)
    plt.plot(x_plot, y_plot, label=r'$N(%.2f,%.2f)$' % (mu_eval, sigma_eval))
    plt.legend(loc='best')
    plt.xlim((-2.0, 3.0))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$p(x)$')
    plt.show()
