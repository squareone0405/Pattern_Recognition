import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import time

class MnistClassifier:
    def __init__(self):
        self.load_mnist()
        self.train_losses = []
        self.validation_losses = []

    def load_mnist(self):
        npzfile = np.load('mnist.npz')
        self.X_train = npzfile['X_train'].astype(np.float32)
        self.X_test = npzfile['X_test'].astype(np.float32)
        self.y_train = npzfile['y_train'].astype(np.int64)
        self.y_test = npzfile['y_test'].astype(np.int64)
        self.input_shape = self.X_train.shape[1] * self.X_train.shape[2]

    def prepare_model(self, hidden_size):
        self.model = nn.Sequential(
            nn.Linear(self.input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.criterion = nn.CrossEntropyLoss()
        print(self.model)

    def onehot2idx(self, onehot_vec):
        return np.argmax(onehot_vec)

    def show_image(self):
        idxes = []
        occred = set()
        idx = 0
        while len(occred) < 10:
            if self.onehot2idx(self.y_train[idx]) not in occred:
                occred.add(self.onehot2idx(self.y_train[idx]))
                idxes.append(idx)
            idx += 1
        plt.figure(figsize=(12, 3))
        gs = gridspec.GridSpec(1, 10)
        for i in range(10):
            ax = plt.subplot(gs[0, i])
            ax.imshow(self.X_train[idxes[i]], cmap='gray')
            ax.axis('off')
        plt.show()

    def pre_process(self):
        self.X_train = (self.X_train.astype(np.float32) / 255.0).reshape(-1, self.input_shape)
        self.X_test = (self.X_test.astype(np.float32) / 255.0).reshape(-1, self.input_shape)
        self.X_train, self.X_test4train, self.y_train, self.y_test4train = \
            train_test_split(self.X_train, self.y_train, test_size=0.3)
        self.X_val4train, self.X_test4train, self.y_val4train, self.y_test4train = \
            train_test_split(self.X_test4train, self.y_test4train, test_size=0.5)
        self.BATCH_SIZE = 128
        torch_dataset = Data.TensorDataset(torch.from_numpy(self.X_train), torch.from_numpy(self.y_train))
        self.data_loader = Data.DataLoader(torch_dataset, batch_size=self.BATCH_SIZE)

    def train(self, epochs):
        start = time.time()
        self.train_losses = []
        self.validation_losses = []
        val_ascend = 0
        self.model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(self.data_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                _, labels = labels.max(dim=1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss = train_loss / len(self.data_loader.dataset)
            self.train_losses.append(train_loss)

            self.model.eval()
            with torch.no_grad():
                inputs = torch.from_numpy(self.X_val4train)
                labels = torch.from_numpy(self.y_val4train)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, labels = labels.max(dim=1)
                validation_loss = self.criterion(outputs, labels)
                if epoch != 0:
                    val_ascend = 0 if validation_loss < self.validation_losses[epoch - 1] else val_ascend + 1
                if val_ascend >= 5: # early stop
                    break
                self.validation_losses.append(validation_loss)
                _, outputs = outputs.max(dim=1)
                total = labels.size(0)
                correct = (outputs == labels).sum().item()
                accuracy = 100 * correct / total
            print('Epoch: {} \tTraining Loss: {:.6f} \tVal Loss: {:.6f} \tVal Accuracy: {:.6f}'.format(
                epoch + 1,
                self.train_losses[epoch],
                self.validation_losses[epoch],
                accuracy
            ))
        print("Training Time: {}".format(time.time() - start))

    def test(self):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(self.X_test)
            labels = torch.from_numpy(self.y_test)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            _, outputs = outputs.max(dim=1)
            _, labels = labels.max(dim=1)
            total = labels.size(0)
            correct = (outputs == labels).sum().item()
            accuracy = 100 * correct / total
            print('Test Accuracy: {:.6f}'.format(accuracy))
            return accuracy, confusion_matrix(labels.to(torch.device("cpu")).numpy(),
                                              outputs.to(torch.device("cpu")).numpy())

    def get_history(self):
        return self.train_losses, self.validation_losses

    def plot_wrong_answer(self, class1, class2):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(self.X_test)
            labels = torch.from_numpy(self.y_test)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            _, outputs = outputs.max(dim=1)
            _, labels = labels.max(dim=1)
            labels, outputs = labels.to(torch.device("cpu")).numpy(), outputs.to(torch.device("cpu")).numpy()

            plt.figure(figsize=(9, 6))
            plt.suptitle('Wrong Answers')
            wrong_indexes = np.where((labels == class1) & (outputs == class2))[0]
            if len(wrong_indexes) > 3:
                choice = np.random.choice(len(wrong_indexes), 3)
                wrong_indexes = wrong_indexes[choice]
            gs = gridspec.GridSpec(2, len(wrong_indexes))
            for idx in range(len(wrong_indexes)):
                ax = plt.subplot(gs[0, idx])
                ax.imshow(self.X_test[wrong_indexes[idx]].reshape(28, 28), cmap='gray')
                ax.axis('off')
                ax.set_title('label = %d, output = %d' % (labels[wrong_indexes[idx]],
                                                          outputs[wrong_indexes[idx]]))
            wrong_indexes = np.where((labels == class2) & (outputs == class1))[0]
            if len(wrong_indexes) > 3:
                choice = np.random.choice(len(wrong_indexes), 3)
                wrong_indexes = wrong_indexes[choice]
            for idx in range(len(wrong_indexes)):
                ax = plt.subplot(gs[1, idx])
                ax.imshow(self.X_test[wrong_indexes[idx]].reshape(28, 28), cmap='gray')
                ax.axis('off')
                ax.set_title('label = %d, output = %d' % (labels[wrong_indexes[idx]],
                                                          outputs[wrong_indexes[idx]]))
            plt.show()


def test_hidden_sizes(hidden_sizes):
    accuracies = []
    col = 3
    plt.figure()
    plt.suptitle('Learning Curves')
    lc_gs = gridspec.GridSpec(int((len(hidden_sizes) + 1) / col), col)
    cm_list = []
    for idx in range(len(hidden_sizes)):
        mnist_clf = MnistClassifier()
        mnist_clf.pre_process()
        mnist_clf.prepare_model(hidden_sizes[idx])
        mnist_clf.train(20)
        train_losses, validation_losses = mnist_clf.get_history()
        lc_ax = plt.subplot(lc_gs[int(idx / col), idx % int(col)])
        lc_ax.plot(range(len(train_losses)), train_losses, label='training losses')
        lc_ax.plot(range(len(validation_losses)), validation_losses, label='validation losses')
        lc_ax.set_title('hidden size = %d' % hidden_sizes[idx])
        lc_ax.set_xlabel('epoch')
        lc_ax.set_ylabel('loss')
        lc_ax.legend(loc='best')
        lc_ax.grid()
        acc, cm = mnist_clf.test()
        cm_list.append(cm)
        accuracies.append(acc)
    plt.figure()
    plt.suptitle('Confusion Matrices')
    cm_gs = gridspec.GridSpec(int((len(hidden_sizes) + 1) / col), col)
    classes = range(10)
    for idx in range(len(cm_list)):
        cm = np.zeros((cm_list[idx].shape[0] + 1, cm_list[idx].shape[1] + 1), dtype=np.float32)
        cm[0: cm_list[idx].shape[0], 0: cm_list[idx].shape[1]] = cm_list[idx]
        sum = 0.0
        for i in range(cm_list[idx].shape[0]):
            cm[i, -1] = cm[i, i] / cm[i, :].sum()
            sum += cm[i, i]
        for j in range(cm_list[idx].shape[1]):
            cm[-1, j] = cm[j, j] / cm[:, j].sum()
        cm[-1, -1] = sum / np.sum(np.sum(cm_list[idx], axis=0))
        cm = cm.transpose()
        cm_ax = plt.subplot(cm_gs[int(idx / col), idx % int(col)])
        cm_ax.imshow(cm, cmap=plt.get_cmap('Blues'))
        cm_ax.set(xticks=np.arange(cm.shape[1]),
                  yticks=np.arange(cm.shape[0]),
                  xticklabels=classes, yticklabels=classes,
                  title='hidden size = %d' % hidden_sizes[idx],
                  ylabel='Outputs',
                  xlabel='Labels')
        plt.setp(cm_ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i < cm.shape[0] - 1 and j < cm.shape[1] - 1:
                    cm_ax.text(j, i, format(int(cm[i, j]), 'd'), ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
                else:
                    cm_ax.text(j, i, format(cm[i, j], '.2f'), ha="center", va="center", color="black")
    print(accuracies)
    plt.show()


def analyze_wrong_answer(class1, class2):
    mnist_clf = MnistClassifier()
    mnist_clf.pre_process()
    mnist_clf.prepare_model(100)
    mnist_clf.train(20)
    mnist_clf.test()
    mnist_clf.plot_wrong_answer(class1, class2)


if __name__ == '__main__':
    mnist_clf = MnistClassifier()
    mnist_clf.show_image()

    hidden_sizes = [5, 10, 20, 50, 100]
    test_hidden_sizes(hidden_sizes)

    analyze_wrong_answer(4, 9)
