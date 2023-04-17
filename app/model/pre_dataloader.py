import numpy as np
import torch.optim
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

def Pre_Dataloader(batch_size):
    # 预测数据
    pre_df = pd.read_csv("../new_data/new_data_7.csv").values

    X_pre = pre_df[:, 0:41]

    n_samples = len(X_pre)
    n_batches = n_samples // batch_size
    X_pre = X_pre[:n_batches * batch_size]

    # 将数据转化为numpy数组
    data_arr = np.array(X_pre)
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

    X_pre = np.hstack(
        (data_arr[:, 0:1], one_hot_1, one_hot_2, one_hot_3, data_arr[:, 4:6], one_hot_6, data_arr[:, 7:11],
         one_hot_11, data_arr[:, 12:20], one_hot_20, one_hot_21, data_arr[:, 22:41]))

    scaler = MinMaxScaler()
    X_pre = scaler.fit_transform(X_pre)

    X_pre = torch.FloatTensor(X_pre)
    pre_dataset = torch.utils.data.TensorDataset(X_pre)
    pre_loader = DataLoader(dataset=pre_dataset, batch_size=128, shuffle=True)

    return pre_loader