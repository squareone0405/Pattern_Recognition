import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression

# question 3
test_num = 100
max_order = 3
# question 4
train_path = './prostate_train.txt'
test_path = './prostate_test.txt'

def generate_dataset(sample_num, sigma, order):
    X_train = np.random.normal(loc=0.0, scale=1.0, size=sample_num)
    epsilon_train = np.random.normal(loc=0.0, scale=sigma, size=sample_num)
    Y_train = 3 * X_train + 6 + epsilon_train
    X_train = X_train.reshape(-1, 1)

    X_test = np.random.normal(loc=0.0, scale=1.0, size=test_num)
    epsilon_test = np.random.normal(loc=0.0, scale=sigma, size=test_num)
    Y_test = 3 * X_test + 6 + epsilon_test
    X_test = X_test.reshape(-1, 1)

    assert order > 0
    for i in range(order - 1):
        X_train = np.hstack((X_train, np.multiply(X_train[:, 0], X_train[:, i]).reshape(-1, 1)))
        X_test = np.hstack((X_test, np.multiply(X_test[:, 0], X_test[:, i]).reshape(-1, 1)))
    return X_train, Y_train, X_test, Y_test

def load_data():
    train_set = np.loadtxt(train_path, skiprows=1)
    X_train = train_set[:, 0: -1]
    Y_train = train_set[:, -1]
    test_set = np.loadtxt(test_path, skiprows=1)
    X_test = test_set[:, 0: -1]
    Y_test = test_set[:, -1]
    return X_train, Y_train, X_test, Y_test

def get_cross_term(X):
    feature_num = X.shape[1]
    X_cross = X
    for i in range(feature_num):
        for j in np.arange(i, feature_num):
            X_cross = np.hstack((X_cross, np.multiply(X[:, i], X[:, j]).reshape(-1, 1)))
    return X_cross

if __name__ == '__main__':
    ''' question 3 '''
    sample_num_list = [10, 100]
    sigma_list = [0.5, 2]
    plt.figure(figsize=(12, 9))
    plt.suptitle('Linear Regression')
    gs = gridspec.GridSpec(2, 2)
    for sample_num_idx in range(len(sample_num_list)):
        for sigma_idx in range(len(sigma_list)):
            X_train, Y_train, X_test, Y_test = generate_dataset(sample_num_list[sample_num_idx],
                                                                sigma_list[sigma_idx], max_order)
            ax = plt.subplot(gs[sample_num_idx, sigma_idx])
            ax.set_title(r'$sample = %d, \sigma = %.1f$' % (sample_num_list[sample_num_idx],
                                                                sigma_list[sigma_idx]))
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.scatter(X_train[:, 0], Y_train, linewidths=1)
            ax.grid()
            legend_list = ['y='] * max_order
            RSS_list = np.zeros((3, 2))
            for order in range(max_order):
                model = LinearRegression()
                model.fit(X_train[:, 0:order + 1], Y_train)
                Y_pred = model.predict(X_test[:, 0:order + 1])
                intercept_ = model.intercept_
                coef_ = model.coef_
                RSS_list[order][0] = np.sum(np.power((Y_train - model.predict(X_train[:, 0:order + 1])), 2))
                RSS_list[order][1] = np.sum(np.power((Y_pred - Y_test), 2))
                for i in range(len(coef_)):
                    legend_list[order] += r'$%+.2fx^%d$' % (coef_[-(i+1)], len(coef_) - i)
                legend_list[order] += r'$%+.2f$' % (intercept_)
                X_plot = np.linspace(min(X_train[:, 0]), max(X_train[:, 0]), 100)
                Y_plot = [intercept_] * 100
                for i in range(len(coef_)):
                    Y_plot = Y_plot + coef_[i] * np.power(X_plot, i + 1)
                ax.plot(X_plot, Y_plot)
            ax.legend(legend_list, loc="upper left")
            ax.text(0.45, 0.05, 'RSS=%.2f(train),%.2f(test)\n        '
                                '%.2f(train),%.2f(test)\n        %.2f(train),%.2f(test)'
                    % (RSS_list[0][0], RSS_list[0][1], RSS_list[1][0], RSS_list[1][1],
                       RSS_list[2][0], RSS_list[2][1]), transform=ax.transAxes)
    plt.show()
    ''' question 4 '''
    X_train, Y_train, X_test, Y_test = load_data()
    X_train_cross = get_cross_term(X_train)
    X_test_cross = get_cross_term(X_test)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print('**********without cross term*********')
    print('intercept:' + str(model.intercept_))
    print('coefficients:' + str(model.coef_))
    print('Standard Deviation on train: ' +
          str(math.sqrt(np.sum(np.power((Y_train - model.predict(X_train)), 2)) / len(Y_train))))
    print('Standard Deviation on test: ' +
          str(math.sqrt(np.sum(np.power((Y_pred - Y_test), 2)) / len(Y_test))))

    model.fit(X_train_cross, Y_train)
    Y_pred_cross = model.predict(X_test_cross)
    print('*********with cross term*********')
    print('intercept: ' + str(model.intercept_))
    print('coefficients: ' + str(model.coef_))
    print('Standard Deviation on train: ' +
          str(math.sqrt(np.sum(np.power((Y_train - model.predict(X_train_cross)), 2)) / len(Y_train))))
    print('Standard Deviation on test: ' +
          str(math.sqrt(np.sum(np.power((Y_pred_cross - Y_test), 2)) / len(Y_test))))

