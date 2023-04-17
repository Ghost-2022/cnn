import numpy as np
from torch.utils.data import DataLoader
import torch.optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def train_data_loader(batch_size, path):
    # 训练数据

    train_df = pd.read_csv(path).values
    X_train = train_df[:, 0:41]
    Y_train = train_df[:, 41:42]

    n_samples = len(X_train)
    n_batches = n_samples // batch_size
    X_train = X_train[:n_batches * batch_size]
    Y_train = Y_train[:n_batches * batch_size]

    # 将数据转化为numpy数组
    data_arr = np.array(X_train)
    # 第一列 protocal 3种
    labels = data_arr[:, 1]
    labels = labels.astype(int)
    one_hot_1 = np.zeros((labels.size, 3))
    one_hot_1[np.arange(labels.size), labels] = 1

    # 第二列 service 70种
    labels = data_arr[:, 2]
    labels = labels.astype(int)
    one_hot_2 = np.zeros((labels.size, 70))
    one_hot_2[np.arange(labels.size), labels] = 1

    # 第三列 flag 11种
    labels = data_arr[:, 3]
    labels = labels.astype(int)
    one_hot_3 = np.zeros((labels.size, 11))
    one_hot_3[np.arange(labels.size), labels] = 1

    # 第六列 land 2种
    labels = data_arr[:, 6]
    labels = labels.astype(int)
    one_hot_6 = np.zeros((labels.size, 2))
    one_hot_6[np.arange(labels.size), labels] = 1

    # 第十一列 logged_in 2种
    labels = data_arr[:, 11]
    labels = labels.astype(int)
    one_hot_11 = np.zeros((labels.size, 2))
    one_hot_11[np.arange(labels.size), labels] = 1

    # 第二十列 is_host_login 2种
    labels = data_arr[:, 20]
    labels = labels.astype(int)
    one_hot_20 = np.zeros((labels.size, 2))
    one_hot_20[np.arange(labels.size), labels] = 1

    # 第二十一列 is_guest_login 2种
    labels = data_arr[:, 21]
    labels = labels.astype(int)
    one_hot_21 = np.zeros((labels.size, 2))
    one_hot_21[np.arange(labels.size), labels] = 1

    X_train = np.hstack((data_arr[:, 0:1], one_hot_1, one_hot_2, one_hot_3, data_arr[:, 4:6],
                         one_hot_6, data_arr[:, 7:11], one_hot_11, data_arr[:, 12:20], one_hot_20, one_hot_21,
                         data_arr[:, 22:41]))
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    Y_train = Y_train.flatten()
    X_train, Y_train = torch.FloatTensor(X_train), torch.LongTensor(Y_train)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    return train_loader
