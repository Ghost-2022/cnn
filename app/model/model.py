from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv1d(1, 32, 3),  # torch.Size([128, 32, 124])
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),  # torch.Size([128, 32, 122])
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # torch.Size([128, 32, 61])
            nn.Conv1d(32, 64, 3),  # torch.Size([128, 64, 59])
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3),  # torch.Size([128, 64, 57])
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # torch.Size([128, 64, 28])

            nn.Flatten(),  # torch.Size([128, 1792])
            # nn.Flatten() 只会将除了batch size以外的维度拉平，不会改变batch size。因此，输入和输出的batch size保持不变。
            nn.Linear(in_features=1792, out_features=256),  # torch.Size([128, 256])
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout()

        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=5, batch_first=True)

        self.model2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=32, out_features=5)
        )

    def forward(self, input):
        input = input.reshape(-1, 1, 126)  # 结果为[128,1,126]

        x = self.model1(input)
        x = x.reshape(128, 1, 256)
        x, (h_n, c_n) = self.lstm(x)
        x = self.model2(x[:, -1, :])
        return x
