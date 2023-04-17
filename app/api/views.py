#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Date: 2023/4/15 10:32
    @Auth: Jacob
    @Desc:
"""
import os
import warnings
from threading import Thread

import torch.optim
from flask import request, current_app, jsonify, g, session
from matplotlib import pyplot as plt
from torch import nn

from app.api import api
from app.common.db import db
from app.model import train_model
from app.model.fit import fit
from app.model.model import Model
from app.model.models import TrainSetModel, TestSetModel
from app.model.test_dataloader import test_data_loader
from app.model.train_dataloader import train_data_loader

status = False

def train_model(batch_size, learning_rate, epochs, app_context):
    with app_context:
        global status
        status = True
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        torch.manual_seed(2020)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_loader = train_data_loader(batch_size, os.path.join(current_app.config['FILE_DIR'], "kddcup.data_train.csv"))
        test_loader = test_data_loader(batch_size, os.path.join(current_app.config['FILE_DIR'], "kddcup.data_test.csv"))
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
        torch.save(model.state_dict(), os.path.join(current_app.config['FILE_DIR'], 'cnn_lstm.pth'))

        # 训练集 平均损失变化图像
        plt.plot(range(epochs), train_loss)
        plt.xlabel('epochs')
        plt.ylabel('cost')
        plt.title('Average loss function change')
        plt.savefig(os.path.join(current_app.config['IMG_DIR'], 'avg-loss.png'))

        # 训练集 准确率变化图像
        plt.plot(range(epochs), train_acc)
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('training set accuracy')
        plt.savefig(os.path.join(current_app.config['IMG_DIR'], 'training-set-accuracy.png'))

        # 测试集 准确率变化图像
        plt.plot(range(epochs), test_acc)
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('testing set accuracy')
        plt.savefig(os.path.join(current_app.config['IMG_DIR'], 'testing-set-accuracy.png'))
        status = False


@api.route('/train', methods=['POST'])
def train():
    rep_data = request.json
    batch_size = rep_data.get('batchSize')
    learning_rate = rep_data.get('learningRate')
    epochs = rep_data.get('epoch')
    if status:
        return jsonify({'code': -1, 'message': 'The model is in training'})
    else:
        try:
            t = Thread(target=train_model, args=(float(batch_size), float(learning_rate),
                                                 float(epochs), current_app.app_context()))
            t.start()
        except Exception as e:
            print(e)
            return jsonify({'code': -1, 'message': 'Error', 'data': status})
        else:
            return jsonify({'code': '0000', 'message': 'Success', 'data': status})


@api.route('/get-train-result')
def get_train_result():
    """

    :return:
    """
    data = {'status': status}
    if status:
        return {'code': '0000', 'message': 'Success', 'data': data}
    else:
        data.update({
            'avgLoss': '/static/img/avg-loss.png',
            'trainPng': '/static/img/testing-set-accuracy.png',
            'testPng': '/static/img/training-set-accuracy.png'
        })
        return {'code': '0000', 'message': 'Success', 'data': data}


@api.route('/get-train-set')
def get_train_set():
    """

    :return:
    """
    page = int(request.args.get('pageIndex', 1))
    count = int(request.args.get('pageSize', 10))
    total = TrainSetModel.query.count()
    query_data = db.session.query(TrainSetModel).limit(count).offset((page - 1) * count).all()

    data = {
        'total': total,
        'list': [item.get_info() for item in query_data]
    }
    return {'code': '0000', 'data': data, 'message': 'Success'}


@api.route('/get-test-set')
def get_test_set():
    page = int(request.args.get('pageIndex', 1))
    count = int(request.args.get('pageSize', 10))
    total = TestSetModel.query.count()
    query_data = db.session.query(TestSetModel).limit(count).offset((page - 1) * count).all()

    data = {
        'total': total,
        'list': [item.get_info() for item in query_data]
    }
    return {'code': '0000', 'data': data, 'message': 'Success'}


@api.route('/identify-file')
def identify_file():
    """

    :return:
    """
    file = request.files.get('file')
    file.save(os.path.join(current_app.config['FILE_DIR'], 'tmp-identify.csv'))
