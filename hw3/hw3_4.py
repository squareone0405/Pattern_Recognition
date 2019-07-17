import numpy as np
import math
import matplotlib.pyplot as plt

def generate_dataset(posi_num, nega_num, train_ratio):
    x_posi = np.random.normal(loc=2.5, scale=1.0, size=posi_num)
    y_posi = np.ones((posi_num, ), dtype='uint8')
    x_nega = np.random.normal(loc=-2.5, scale=math.sqrt(2.0), size=nega_num)
    y_nega = np.zeros((nega_num, ), dtype='uint8')
    x = np.concatenate((x_posi, x_nega), axis=0)
    y = np.concatenate((y_posi, y_nega), axis=0)
    permutation = np.random.permutation(posi_num + nega_num)
    x = x[permutation]
    y = y[permutation]
    train_num = int(x.size * train_ratio)
    return x[0: train_num], y[0: train_num], x[train_num:], y[train_num:]

def get_px4plot(x_train, sigma, h):
    sigma = sigma * h
    x = np.linspace(start=-10, stop=10, num=200)
    x_mat = np.repeat(x, repeats=x_train.size, axis=0).reshape((x.size, x_train.size))
    x_train_mat = np.repeat(x_train.reshape(1, -1), repeats=x.size, axis=0).reshape((x.size, x_train.size))
    x_diff = x_mat - x_train_mat
    px = 1.0 / (x_train.size * math.sqrt(2.0 * math.pi) * sigma) * \
        np.sum(np.exp(-1.0 * np.multiply(x_diff, x_diff) / (2 * sigma * sigma)), axis=1)
    return x, px

def predict(x_train, x_test, sigma, h):
    sigma = sigma * h
    x_test_mat = np.repeat(x_test, repeats=x_train.size, axis=0).reshape((x_test.size, x_train.size))
    x_train_mat = np.repeat(x_train.reshape(1, -1), repeats=x_test.size, axis=0).reshape((x_test.size, x_train.size))
    x_diff = x_test_mat - x_train_mat
    px = 1.0 / (x_train.size * math.sqrt(2.0 * math.pi) * sigma) * \
        np.sum(np.exp(-1.0 * np.multiply(x_diff, x_diff) / (2 * sigma * sigma)), axis=1)
    return px

if __name__ == '__main__':
    data_num = [350, 250]
    train_ratio = 0.7
    sigma = 1.0
    h_list = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16]
    round = 3
    for h in h_list:
        error_rate = np.empty((round, 4))
        for i in range(round):
            x_train, y_train, x_test, y_test = generate_dataset(data_num[0], data_num[1], train_ratio)
            x_train_posi = x_train[np.where(y_train == 1)]
            x_train_nega = x_train[np.where(y_train == 0)]
            '''plot'''
            plt.title(r'$\bar{P}(x)\sim x$')
            x_plot_posi, px_posi = get_px4plot(x_train_posi, sigma, h)
            plt.plot(x_plot_posi, px_posi, label='positive' + str(i))
            x_plot_nega, px_nega = get_px4plot(x_train_nega, sigma, h)
            plt.plot(x_plot_nega, px_nega, label='negative' + str(i))
            plt.xlabel(r'$x$')
            plt.ylabel(r'$\bar{P}(x)$')
            plt.legend(loc='upper right')
            plt.grid()
            '''predict least error'''
            px_posi = predict(x_train_posi, x_test, sigma, h)
            px_nega = predict(x_train_nega, x_test, sigma, h)
            p_posi = x_train_posi.size / x_train.size
            p_nega = x_train_nega.size / x_train.size
            y_pred = px_posi * p_posi > px_nega * p_nega
            error_rate[i, 0] = (len(np.where(y_pred != y_test)[0]) / y_test.size * 100.0)
            '''predict least risk'''
            risk_posi = px_nega * p_posi * 1.0
            risk_nega = px_posi * p_nega * 10.0
            y_pred = risk_posi < risk_nega
            error_rate[i, 1] = \
                len(np.where((y_pred == 1) & (y_test == 0))[0]) / y_test[np.where(y_test == 0)].size * 100.0
            error_rate[i, 2] = \
                len(np.where((y_pred == 0) & (y_test == 1))[0]) / y_test[np.where(y_test == 1)].size * 100.0
            error_rate[i, 3] = \
                len(np.where(y_pred != y_test)[0]) / y_test.size * 100.0
        plt.show()
        print('*' * 10 + 'h = %.4f' % (h) + '*' * 10)
        print('-' * 5 + 'least error estimate:' + '-' * 5)
        print('error rate: ' + str(error_rate[:, 0].transpose())
              + '\t Average = %.2f%%' % (np.average(error_rate[:, 0].transpose())))
        print('-' * 5 + 'least risk estimate:' + '-' * 5)
        print('error rate on negative test set: ' + str(error_rate[:, 1].transpose())
              + '\t Average = %.2f%%' % (np.average(error_rate[:, 1].transpose())))
        print('error rate on positive test set: ' + str(error_rate[:, 2].transpose())
              + '\t Average = %.2f%%' % (np.average(error_rate[:, 2].transpose())))
        print('total error rate on test set: ' + str(error_rate[:, 3].transpose())
              + '\t Average = %.2f%%' % (np.average(error_rate[:, 3].transpose())))
        print()

