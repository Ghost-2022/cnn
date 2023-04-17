import torch.optim

# 训练过程   迭代一次
def fit(epoch, model, train_loader, test_loader, loss_fn, optimizer, device):
    correct = 0
    total = 0
    running_loss = 0
    for x, y in train_loader:
        # 把数据放到GPU上
        x, y = x.to(device), y.to(device)
        y_pred = model(x)  # 共有batchsize个数据，每个batchsize下有5个概率  [128,5]
        loss = loss_fn(y_pred, y)
        # 梯度清零
        optimizer.zero_grad()
        loss.backward()  # backward 反向传播
        optimizer.step()
        # 计算损失过程
        with torch.no_grad():
            pred = torch.argmax(y_pred, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

        # 循环完一次后, 计算损失
        # 计算平均损失
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    # 测试数据的代码
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            # 计算损失
            pred = torch.argmax(y_pred, dim=1)
            test_correct += (pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
    # 计算平均损失
    test_epoch_loss = test_running_loss / test_total
    test_epoch_acc = test_correct / test_total

    # 打印输出
    print('epoch:', epoch,
          'loss:', round(epoch_loss, 6),
          'accuracy:', round(epoch_acc, 6),
          'test_loss:', round(test_epoch_loss, 6),
          'test_accuracy:', round(test_epoch_acc, 6))

    return epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc