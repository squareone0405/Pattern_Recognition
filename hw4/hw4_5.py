import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
import time


class FCNet(nn.Module):
    def __init__(self, hidden_num, activation='relu'):
        super(FCNet, self).__init__()
        self.hidden_num = hidden_num
        self.activation = activation
        self.fc0 = nn.Linear(32 * 32 * 3, 3)
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 3)

    def forward(self, x):
        if self.activation == 'relu':
            if self.hidden_num == 0:
                return self.fc0(x)
            x = F.relu(self.fc1(x))
            for i in range(self.hidden_num - 1):
                x = F.relu(self.fc2(x))
            return self.fc3(x)
        elif self.activation == 'sigmoid':
            if self.hidden_num == 0:
                return self.fc0(x)
            x = F.sigmoid(self.fc1(x))
            for i in range(self.hidden_num - 1):
                x = F.sigmoid(self.fc2(x))
            return self.fc3(x)
        else:
            return None


class ConvNet(nn.Module):
    def __init__(self, activation='relu', dropout=True):
        super(ConvNet, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 15 * 15, 64) # 32*32->30*30->15*15
        self.fc2=  nn.Linear(64, 3)

    def forward(self, x):
        if self.activation == 'relu':
            x = self.pool(F.relu(self.conv(x)))
        elif self.activation == 'sigmoid':
            x = self.pool(F.sigmoid(self.conv(x)))
        if self.dropout:
            x = self.dropout1(x)
        x = x.view(-1, 64 * 15 * 15)
        if self.activation == 'relu':
            x = x = F.relu(self.fc1(x))
        elif self.activation == 'sigmoid':
            x = x = F.sigmoid(self.fc1(x))
        if self.dropout:
            x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CifarClasifier():
    def __init__(self):
        self.load_cifar10()
        self.train_losses = []
        self.validation_losses = []
        self.train_accuracies = []
        self.validation_accuracies = []

    ''' the x already devided by 255 '''
    def load_cifar10(self):
        npzfile = np.load('cifar10.npz')
        self.X_train = npzfile['x_train'].astype(np.float32)
        self.X_test = npzfile['x_test'].astype(np.float32)
        self.y_train = npzfile['y_train'].astype(np.int64)
        self.y_test = npzfile['y_test'].astype(np.int64)
        self.y_train = self.y_train[:, 0:3]
        self.y_test = self.y_test[:, 0:3]
        self.input_shape = self.X_train.shape[1] * self.X_train.shape[2] * self.X_train.shape[3]

    def pre_process(self, flatten=True):
        if flatten:
            self.X_train = self.X_train.reshape(-1, self.input_shape)
            self.X_test = self.X_test.reshape(-1, self.input_shape)
        else: # chanel first
            self.X_train = np.rollaxis(self.X_train, 3, 1)
            self.X_test = np.rollaxis(self.X_test, 3, 1)
        self.X_train, self.X_test4train, self.y_train, self.y_test4train = \
            train_test_split(self.X_train, self.y_train, test_size=0.3)
        self.X_val4train, self.X_test4train, self.y_val4train, self.y_test4train = \
            train_test_split(self.X_test4train, self.y_test4train, test_size=0.5)
        self.BATCH_SIZE = 128
        torch_dataset = Data.TensorDataset(torch.from_numpy(self.X_train), torch.from_numpy(self.y_train))
        self.data_loader = Data.DataLoader(torch_dataset, batch_size=self.BATCH_SIZE)

    def set_model(self, net):
        self.model = net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs):
        start = time.time()
        self.train_losses = []
        self.validation_losses = []
        self.train_accuracies = []
        self.validation_accuracies = []
        val_ascend = 0
        self.model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            train_total = 0
            train_correct = 0
            for batch_idx, (inputs, labels) in enumerate(self.data_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                _, labels = labels.max(dim=1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                _, outputs = outputs.max(dim=1)
                train_total += labels.size(0)
                train_correct += (outputs == labels).sum().item()
            train_loss = train_loss / len(self.data_loader.dataset)
            self.train_losses.append(train_loss)
            self.train_accuracies.append((1.0 * train_correct) / train_total)

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
                accuracy = correct / total
                self.validation_accuracies.append(accuracy)
            print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f} '
                  '\tVal Loss: {:.6f} \tVal Accuracy: {:.6f}'.format(
                epoch + 1,
                self.train_losses[epoch],
                self.train_accuracies[epoch],
                self.validation_losses[epoch],
                self.validation_accuracies[epoch]
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
            return accuracy

    def get_history(self):
        return self.train_losses, self.validation_losses, \
               self.train_accuracies, self.validation_accuracies


def test_fc_num(hidden_nums, activation='relu'):
    col = 2
    train_accuracies_list = []
    validation_accuracies_list = []
    test_accuracy_list = []
    plt.figure(figsize=(12, 9))
    plt.suptitle('Learning Curves')
    lc_gs = gridspec.GridSpec(int((len(hidden_nums) + 1) / col), col)
    for idx in range(len(hidden_nums)):
        cifar_clf = CifarClasifier()
        cifar_clf.pre_process()
        net = FCNet(hidden_nums[idx], activation)
        cifar_clf.set_model(net)
        cifar_clf.train(20)
        test_accuracy_list.append(cifar_clf.test())
        train_losses, validation_losses, train_accuracies, validation_accuracies = cifar_clf.get_history()
        train_accuracies_list.append(train_accuracies)
        validation_accuracies_list.append(validation_accuracies)
        lc_ax = plt.subplot(lc_gs[int(idx / col), idx % int(col)])
        lc_ax.plot(range(len(train_losses)), train_losses, label='training losses')
        lc_ax.plot(range(len(validation_losses)), validation_losses, label='validation losses')
        lc_ax.set_title('fc num = %d' % hidden_nums[idx])
        lc_ax.set_xlabel('epoch')
        lc_ax.set_ylabel('loss')
        lc_ax.legend(loc='best')
        lc_ax.grid()

    plt.figure(figsize=(12, 9))
    plt.suptitle('Accuracy')
    acc_gs = gridspec.GridSpec(int((len(hidden_nums) + 1) / col), col)
    for idx in range(len(hidden_nums)):
        acc_ax = plt.subplot(acc_gs[int(idx / col), idx % int(col)])
        acc_ax.plot(range(len(train_accuracies_list[idx])),
                    train_accuracies_list[idx], label='training accuracy')
        acc_ax.plot(range(len(validation_accuracies_list[idx])),
                    validation_accuracies_list[idx], label='validation accuracy')
        acc_ax.set_title('fc num = %d' % hidden_nums[idx])
        acc_ax.set_xlabel('epoch')
        acc_ax.set_ylabel('accuracy')
        acc_ax.legend(loc='best')
        acc_ax.grid()
    plt.show()
    print(test_accuracy_list)

def test_conv(activation='relu', dropout=True):
    cifar_clf = CifarClasifier()
    cifar_clf.pre_process(flatten=False)
    net = ConvNet(activation=activation, dropout=dropout)
    cifar_clf.set_model(net)
    cifar_clf.train(20)
    cifar_clf.test()
    train_losses, validation_losses, \
    train_accuracies, validation_accuracies = cifar_clf.get_history()
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label='training losses')
    plt.plot(range(len(validation_losses)), validation_losses, label='validation losses')
    plt.title('Learning Curve on ConvNet')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.grid()
    plt.figure()
    plt.plot(range(len(train_accuracies)), train_accuracies, label='training accuracy')
    plt.plot(range(len(validation_accuracies)), validation_accuracies, label='validation accuracy')
    plt.title('Accuracy on ConvNet')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    hidden_nums = [0, 1, 2, 3]
    activation_list = ['relu', 'sigmoid']
    for activation in activation_list:
        test_fc_num(hidden_nums, activation=activation)
        test_conv(activation=activation)
    test_conv(activation='relu', dropout=False)

