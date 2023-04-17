import os
import warnings

import torch.optim
from matplotlib import pyplot as plt
from torch import nn

from app.model.fit import fit
from app.model.model import Model
from app.model.test_dataloader import test_data_loader
from app.model.train_dataloader import train_data_loader


def train(batch_size, learning_rate, epochs, app, g):
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    torch.manual_seed(2020)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader = train_data_loader(batch_size, os.path.join(app.config['FILE_DIR'], "kddcup.data_train.csv"))
    test_loader = test_data_loader(batch_size, os.path.join(app.config['FILE_DIR'], "kddcup.data_test.csv"))
    model = Model()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 执行操作
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # 训练 迭代epoch次
    for epoch in range(epochs):
        epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc = fit(epoch, model, train_loader, test_loader, loss_fn,
                                                                     optimizer, device)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        test_loss.append(test_epoch_loss)
        test_acc.append(test_epoch_acc)
    # 保存训练好的模型
    torch.save(model.state_dict(), os.path.join(app.config['FILE_DIR'], 'cnn_lstm.pth'))

    # 训练集 平均损失变化图像
    plt.plot(range(epochs), train_loss)
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.title('Average loss function change')
    plt.savefig(os.path.join(app.config['IMG_DIR'], 'avg-loss.png'))

    # 训练集 准确率变化图像
    plt.plot(range(epochs), train_acc)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('training set accuracy')
    plt.savefig(os.path.join(app.config['IMG_DIR'], 'training-set-accuracy.png'))

    # 测试集 准确率变化图像
    plt.plot(range(epochs), test_acc)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('testing set accuracy')
    plt.savefig(os.path.join(app.config['IMG_DIR'], 'testing-set-accuracy.png'))
    g.status = False


if __name__ == '__main__':
    train(128, 0.0005, 3)
