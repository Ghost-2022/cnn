import numpy as np
import torch.optim
from model import Model
import pandas as pd
from pre_dataloader import pre_data_loader

batch_size = 128
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Model()  #这里需要重新模型结构，My_model
model.to(device)
model.load_state_dict(torch.load('cnn_lstm.pth')) #这里根据模型结构，调用存储的模型参数
model.eval()

pre_loader = pre_data_loader(batch_size)
with torch.no_grad():
    non_zero_indices = []
    prediction = []
    for i, x in enumerate(pre_loader):
        x = x[0].to(device)
        y_pred = model(x)
        pred = torch.argmax(y_pred, dim=1)
        prediction = pred.cpu().numpy()
        prediction_error = np.array([x for x in prediction if x != 0])

        non_zero_idx = torch.nonzero(pred)
        non_zero_indices.append(non_zero_idx.cpu().numpy() + i * pre_loader.batch_size)

    # 合并所有预测结果
    non_zero_indices = np.concatenate(non_zero_indices, axis=0)

    # 获取预测值不为0的表格行号
    non_zero_row_indices = non_zero_indices[prediction[non_zero_indices] != 0]


# 读取原始表格
df = pd.read_csv('../new_data/new_data_7.csv')
# 根据下标提取对应的行数据
new_df = df.loc[non_zero_row_indices]
# 将新的表格保存为 CSV 文件
new_df.to_csv('../bad_data/bad.csv', index=False)
# 读取表格
df = pd.read_csv('../bad_data/bad.csv')
# 最后添加一列异常值的预测
df.insert(41, 'class', prediction_error)
# 保存表格
df.to_csv('../bad_data/bad.csv', index=False)

