import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

def idx2onehot(idx, num_classes):
    return np.eye(num_classes)[idx]

'''class NNClassifier():
    def __init__(self, model, save_path=None):
        self.model = model
        self.save_path = save_path

    def fit(self, X_train, y_train):
        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        if self.save_path:
            ckpt = ModelCheckpoint(self.save_path, monitor='val_acc', verbose=0, save_best_only=True,
                                   save_weights_only=False, mode='auto', period=1)
            self.model.fit(x=X_train, y=y_train, validation_split=0.1, epochs=30,
                           batch_size=128, verbose=1, callbacks=[ckpt])
        else:
            self.model.fit(x=X_train, y=y_train, validation_split=0.1, epochs=30,
                           batch_size=128, verbose=1)

    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=1)

    def score(self, X_test, y_test):
        preds = self.model.evaluate(X_test, y_test)
        return preds[1]

    def model_from_file(self):
        self.model = load_model(self.save_path)
        print(self.model.summary())'''

class Model1:
    def __init__(self, method):
        assert method in ['nn', 'adaboost']
        self.method = method
        self.clf = None
        self.acc = None

    def fit(self, train_features, train_labels):
        data_num = train_features.shape[0]
        val_num = int(0.1 * data_num)
        self.clf = np.array([None] * 10)
        self.acc = np.zeros(10)

        for i in range(10):
            random_idx = np.random.choice(data_num, val_num, replace=False)
            val_mask = np.array([False] * data_num)
            val_mask[random_idx] = True
            train_mask = ~val_mask
            X_train = train_features[train_mask]
            y_train = train_labels[train_mask]
            X_val = train_features[val_mask]
            y_val = train_labels[val_mask]
            if self.method == 'nn':
                '''sm = SMOTE()
                X_train, y_train = sm.fit_sample(X_train, y_train)'''
                nn_clf = self.get_model()
                y_train_oh = idx2onehot(y_train, 12)
                y_val_oh = idx2onehot(y_val, 12)
                self.nn_fit(nn_clf, X_train, y_train_oh)
                self.nn_score(nn_clf, X_train, y_train_oh)
                self.nn_score(nn_clf, X_val, y_val_oh)
                self.acc[i] = self.nn_score(nn_clf, X_val, y_val_oh)
                self.clf[i] = nn_clf

            else:
                ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=8, criterion='entropy'),
                                             algorithm='SAMME.R', n_estimators=100)
                ada_clf.fit(X_train, y_train)
                print(ada_clf.score(X_train, y_train))
                print(ada_clf.score(X_val, y_val))
                self.acc[i] = ada_clf.score(X_val, y_val)
                self.clf[i] = ada_clf

        print(self.acc)
        print(np.mean(self.acc))
        return self.acc

    def predict(self, X_test):
        vote = np.zeros((12, X_test.shape[0]))
        for i in range(10):
            if self.method == 'nn':
                y_pred = self.nn_predict(self.clf[i], X_test)
                vote += y_pred.transpose() * self.acc[i]
            else:
                y_pred = self.clf[i].predict(X_test)
                vote += idx2onehot(y_pred, 12).transpose() * self.acc[i]
        '''import pandas as pd
        pd.DataFrame(np.max(vote, axis=0)).to_csv('max.csv')'''
        return np.argmax(vote, axis=0)

    def get_model(self):
        model = Sequential()
        model.add(Dense(units=128, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', activation='sigmoid', name='fc0'))
        model.add(Dense(units=128, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', activation='sigmoid', name='fc1'))
        model.add(Dense(units=12, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', activation='softmax', name='fc2'))
        return model

    def nn_fit(self, model, X_train, y_train):
        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        earlyStopping = EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='min')
        model.fit(x=X_train, y=y_train, validation_split=0.1, epochs=30,
                  batch_size=32, verbose=1, callbacks=[earlyStopping])

    def nn_predict(self, model, X_test):
        return model.predict(X_test)

    def nn_score(self, model, X_test, y_test):
        preds = model.evaluate(X_test, y_test)
        return preds[1]


class Model2:
    def __init__(self, method):
        assert method in ['svm', 'nn', 'adaboost']
        self.method = method
        self.clf = None
        self.acc = None

    def fit_10_ford(self, train_features, train_labels):
        data_num = train_features.shape[0]
        shuffle_idx = np.random.permutation(data_num)
        train_features = train_features[shuffle_idx, :]
        train_labels = train_labels[shuffle_idx]
        val_mask = np.array([False] * data_num)
        val_num = int(0.1 * data_num)
        val_mask[0:val_num] = True
        y_pred = np.zeros_like(train_labels)
        self.clf = np.array([None] * 10)
        self.acc = np.zeros(10)

        for i in range(10):
            train_mask = ~val_mask
            X_train = train_features[train_mask]
            y_train = train_labels[train_mask]
            X_val = train_features[val_mask]
            y_val = train_labels[val_mask]

            if self.method == 'svm':
                '''sm = SMOTE()
                X_train, y_train = sm.fit_sample(X_train, y_train)'''
                (bin, count) = np.unique(y_train, return_counts=True)
                svm_clf = SVC(C=50.0, kernel='rbf', degree=3, gamma='auto',
                              coef0=0.0, shrinking=True, probability=False,
                              tol=1e-3, cache_size=200, class_weight=None,
                              verbose=False, max_iter=-1, decision_function_shape='ovr',
                              random_state=None)
                svm_clf.fit(X_train, y_train)
                print(svm_clf.score(X_train, y_train))
                print(svm_clf.score(X_val, y_val))
                y_pred[val_mask] = svm_clf.predict(X_val)
                self.acc[i] = svm_clf.score(X_val, y_val)
                self.clf[i] = svm_clf

            elif self.method == 'nn':
                sm = SMOTE()
                X_train, y_train = sm.fit_sample(X_train, y_train)
                nn_clf = self.get_model()
                y_train_oh = idx2onehot(y_train, 12)
                y_val_oh = idx2onehot(y_val, 12)
                self.nn_fit(nn_clf, X_train, y_train_oh)
                self.nn_score(nn_clf, X_train, y_train_oh)
                self.nn_score(nn_clf, X_val, y_val_oh)
                y_pred[val_mask] = np.argmax(self.nn_predict(nn_clf, X_val), axis=1)
                self.acc[i] = self.nn_score(nn_clf, X_val, y_val_oh)
                self.clf[i] = nn_clf

            else:
                ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=8),
                                             algorithm='SAMME.R', n_estimators=100)
                ada_clf.fit(X_train, y_train)
                print(ada_clf.score(X_train, y_train))
                print(ada_clf.score(X_val, y_val))
                y_pred[val_mask] = ada_clf.predict(X_val)
                self.acc[i] = ada_clf.score(X_val, y_val)
                self.clf[i] = ada_clf

            val_mask = np.roll(val_mask, val_num)
        print(self.acc)
        print(np.mean(self.acc))
        return self.acc

    def predict(self, X_test):
        vote = np.zeros((12, X_test.shape[0]))
        for i in range(10):
            if self.method == 'svm':
                y_pred = self.clf[i].predict(X_test)
                vote += idx2onehot(y_pred, 12).transpose() * self.acc[i]
            elif self.method == 'nn':
                y_pred = self.nn_predict(self.clf[i], X_test)
                vote += y_pred.transpose() * self.acc[i]
            else:
                y_pred = self.clf[i].predict(X_test)
                vote += idx2onehot(y_pred, 12).transpose() * self.acc[i]
        import pandas as pd
        pd.DataFrame(np.max(vote, axis=0)).to_csv('max.csv')
        return np.argmax(vote, axis=0)


    def get_model(self):
        model = Sequential()
        model.add(Dense(units=128, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', activation='sigmoid', name='fc0'))
        model.add(Dense(units=128, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', activation='sigmoid', name='fc1'))
        model.add(Dense(units=12, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', activation='softmax', name='fc2'))
        return model

    def nn_fit(self, model, X_train, y_train):
        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        earlyStopping = EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='min')
        model.fit(x=X_train, y=y_train, validation_split=0.1, epochs=50,
                  batch_size=128, verbose=0, callbacks=[earlyStopping])

    def nn_predict(self, model, X_test):
        return model.predict(X_test)

    def nn_score(self, model, X_test, y_test):
        preds = model.evaluate(X_test, y_test)
        return preds[1]
