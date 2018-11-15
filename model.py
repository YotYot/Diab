from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from ex_1 import load_data, Dataset

init_lr = 1e-5


class ModelP3Q2(nn.Module):

    def __init__(self):
        super(ModelP3Q2, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.linear1 = nn.Linear(9216, 100)
        self.linear2 = nn.Linear(100, 30)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def adjust_lr(optimizer, epoch):
    lr = init_lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class ModelP3Q3(nn.Module):

    def __init__(self):
        super(ModelP3Q3, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.linear1 = nn.Linear(15488, 500)
        self.linear2 = nn.Linear(500, 500)
        self.linear3 = nn.Linear(500, 30)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        self.maxpool = nn.MaxPool2d(stride=2, kernel_size=2)

    def forward(self, x):
        x = x.view((-1, 1, 96, 96))

        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = x.view(-1, 15488)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train(net, dataset):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(net)
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())
    net.to(device)
    batch_size = 16
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=init_lr)
    # optimizer = optim.Adam(net.parameters(), lr=1e-5)
    optimizer.zero_grad()
    num_epochs = 300

    train_steps_in_e = dataset.train_x.shape[0] // batch_size
    val_steps_in_e = dataset.val_x.shape[0] // batch_size

    train_data = dataset.train_x
    train_labels = dataset.train_y
    net.zero_grad()
    train_epochs_loss = []
    val_epochs_loss = []
    last_loss = 1e4
    for e in range(num_epochs):
        train_p_bar = tqdm(range(train_steps_in_e))
        for step in train_p_bar:
            offset = step * batch_size
            x = torch.Tensor(train_data[offset:offset + batch_size, :]).to(device)
            y = torch.Tensor(train_labels[offset:offset + batch_size, :]).to(device)
            out = net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_p_bar.set_description('Train loss: {:3.3}'.format(loss.item()))
        train_epochs_loss += [loss.item()]
        train_data, train_labels = shuffle(train_data, train_labels, random_state=42)

        # Validate
        val_p_bar = tqdm(range(val_steps_in_e))
        for val_step in val_p_bar:
            offset = val_step * batch_size
            x = torch.Tensor(dataset.val_x[offset:offset + batch_size, :]).to(device)
            y = torch.Tensor(dataset.val_y[offset:offset + batch_size, :]).to(device)
            out = net(x)
            loss = criterion(out, y)
            val_p_bar.set_description('Val loss: {:3.3}'.format(loss.item()))
        val_epochs_loss += [loss.item()]
        print("epoch {:3} Train Loss: {:1.5} Val loss: {:1.5}".format(e, train_epochs_loss[-1], val_epochs_loss[-1]))
        if val_epochs_loss[-1] > last_loss:
            adjust_lr(optimizer, e)
        last_loss = val_epochs_loss[-1]

    # plt.title('Cost Functions')
    # plt.xlabel('epoch num'), plt.ylabel('loss')
    # plt.plot(train_epochs_loss, label="Train Loss")
    # plt.plot(val_epochs_loss, label="Val Loss")
    # plt.legend()
    # plt.show()
    # plt.savefig('Q3.png')
    return train_epochs_loss, val_epochs_loss


def _plot_fig(P3Q2_train_loss, P3Q2_val_loss, P3Q3_train_loss, P3Q3_val_loss):
    plt.title('Cost Functions')
    plt.xlabel('epoch num'), plt.ylabel('loss')
    plt.yscale('log')
    plt.plot(P3Q2_train_loss, label="Small net Train Loss")
    plt.plot(P3Q2_val_loss, label="Small net Val Loss")
    plt.plot(P3Q3_train_loss, label="Big net Train Loss")
    plt.plot(P3Q3_val_loss, label="Big net Val Loss")
    plt.legend()
    plt.savefig('Q3.png')
    plt.show()


def main():
    dataset = load_data()

    print('problem2')
    net = ModelP3Q2()
    P3Q2_train_loss, P3Q2_val_loss = train(net, dataset)

    print('problem3')
    net = ModelP3Q3()
    P3Q3_train_loss, P3Q3_val_loss = train(net, dataset)

    _plot_fig(P3Q2_train_loss, P3Q2_val_loss, P3Q3_train_loss, P3Q3_val_loss)
    _plot_test_images(dataset, net)


def _plot_test_images(dataset, net):
    num_test = 16
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    x = torch.Tensor(dataset.test_x[0:16]).to(device)
    lmrks = net.forward(x) * 48 + 48
    print("Done!")
    f, axarr = plt.subplots(4, 4)
    plt.suptitle('test results')
    plt.subplots_adjust(hspace=0, wspace=0)
    for i in range(num_test):
        lmrk = torch.round(lmrks[i].reshape(15, 2)).detach().cpu().numpy()
        lmrk = np.clip(lmrk.astype(np.uint8), 0, 95)
        img = (dataset.test_x[i] * 255).reshape((96, 96)).astype(np.uint8)
        img_rgb = np.concatenate((img[:, :, None], img[:, :, None], img[:, :, None]), axis=2)
        for lmrk_idx in range(15):
            img_rgb[lmrk[lmrk_idx, 1], lmrk[lmrk_idx, 0], :] = [255, 0, 0]
            img_rgb[lmrk[lmrk_idx, 1]+1, lmrk[lmrk_idx, 0], :] = [255, 0, 0]
            img_rgb[lmrk[lmrk_idx, 1]-1, lmrk[lmrk_idx, 0], :] = [255, 0, 0]
            img_rgb[lmrk[lmrk_idx, 1], lmrk[lmrk_idx, 0]+1, :] = [255, 0, 0]
            img_rgb[lmrk[lmrk_idx, 1], lmrk[lmrk_idx, 0]-1, :] = [255, 0, 0]
            axarr[i // 4, i % 4].imshow(img_rgb), axarr[i // 4, i % 4].axis('off')
    # plt.show()
    plt.savefig('Q3_test.jpg')


if __name__ == '__main__':
    main()
