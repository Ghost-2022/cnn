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
import traceback

import torch.optim
from flask import request, current_app, jsonify, g, session
from matplotlib import pyplot as plt
from torch import nn
import pandas as pd
import numpy as np

from app.api import api
from app.common.db import db
from app.model import train_model
from app.model.fit import fit
from app.model.model import Model
from app.model.models import TrainSetModel, TestSetModel, BadDataset
from app.model.test_dataloader import test_data_loader
from app.model.train_dataloader import train_data_loader
from app.model.pre_dataloader import pre_data_loader

status = False
train_list = []


def train_model(batch_size, learning_rate, epochs, app_context):
    try:
        with app_context:
            global status
            status = True
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            torch.manual_seed(2020)
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            train_loader = train_data_loader(batch_size,
                                             os.path.join(current_app.config['FILE_DIR'], "kddcup.data_train.csv"))
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
                epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc = fit(epoch, model, train_loader, test_loader,
                                                                             loss_fn,
                                                                             optimizer, device)
                train_list.append({'epoch': epoch, 'loss': epoch_loss, 'accuracy': round(epoch_acc, 6),
                                   'test_loss': round(test_epoch_loss, 6), 'test_accuracy': round(test_epoch_acc, 6)})
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            test_loss.append(test_epoch_loss)
            test_acc.append(test_epoch_acc)
            # 保存训练好的模型
            torch.save(model.state_dict(), os.path.join(current_app.config['FILE_DIR'], 'cnn_lstm.pth'))

            fig, ax = plt.subplots()
            # 训练集 平均损失变化图像
            ax.plot(range(epochs), train_loss)
            ax.set_xlabel('epochs')
            ax.set_ylabel('cost')
            ax.set_title('Average loss function change')
            fig.savefig(os.path.join(current_app.config['IMG_DIR'], 'avg-loss.png'))


            fig, ax = plt.subplots()
            # 训练集 准确率变化图像
            ax.plot(range(epochs), train_acc)
            ax.set_xlabel('epochs')
            ax.set_ylabel('accuracy')
            ax.set_title('training set accuracy')
            fig.savefig(os.path.join(current_app.config['IMG_DIR'], 'training-set-accuracy.png'))

            fig, ax = plt.subplots()
            # 测试集 准确率变化图像
            ax.plot(range(epochs), test_acc)
            ax.set_xlabel('epochs')
            ax.set_ylabel('accuracy')
            ax.set_title('testing set accuracy')
            fig.savefig(os.path.join(current_app.config['IMG_DIR'], 'testing-set-accuracy.png'))

    except Exception as e:
        print(traceback.format_exc())
    finally:
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
        global train_list
        train_list = []
        try:
            t = Thread(target=train_model, args=(int(batch_size), float(learning_rate),
                                                 int(epochs), current_app.app_context()), daemon=True)
            t.start()

        except Exception as e:
            print(e)
            return jsonify({'code': -1, 'message': 'Error', 'data': status})
        else:
            return jsonify({'code': '0000', 'message': 'Success', 'data': status})


@api.route('/get-train-status')
def get_train_status():
    return {'code': '0000', 'message': 'Success', 'data': train_list}


@api.route('/get-train-result')
def get_train_result():
    """

    :return:
    """
    data = {'status': status, 'list': train_list}
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


@api.route('/identify-file', methods=['POST'])
def identify_file():
    """

    :return:
    """
    file = request.files.get('file')
    filepath = os.path.join(current_app.config['FILE_DIR'], 'tmp-identify.csv')
    if os.path.exists(filepath):
        os.remove(filepath)
    file.save(filepath)
    data_loader = pre_data_loader(filepath, 128)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model()  # 这里需要重新模型结构，My_model
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(current_app.config['FILE_DIR'], 'cnn_lstm.pth')))
    model.eval()
    with torch.no_grad():
        non_zero_indices = []
        prediction = []
        for i, x in enumerate(data_loader):
            x = x[0].to(device)
            y_pred = model(x)
            pred = torch.argmax(y_pred, dim=1)
            prediction = pred.cpu().numpy()
            prediction_error = np.array([x for x in prediction if x != 0])

            non_zero_idx = torch.nonzero(pred)
            non_zero_indices.append(non_zero_idx.cpu().numpy() + i * data_loader.batch_size)

        # 合并所有预测结果
        non_zero_indices = np.concatenate(non_zero_indices, axis=0)

        # 获取预测值不为0的表格行号
        non_zero_row_indices = non_zero_indices[prediction[non_zero_indices] != 0]

    df = pd.read_csv(filepath)
    new_df = df.loc[non_zero_row_indices]
    for item in prediction_error:
        new_df['class'] = item
    bad_data_path = os.path.join(current_app.config['FILE_DIR'], 'tmp-bad-data.csv')
    if os.path.exists(bad_data_path):
        os.remove(bad_data_path)
    new_df.to_csv(bad_data_path)
    return jsonify({'code': '0000', 'message': 'success'})


@api.route('/get-bad-data')
def get_bad_data():
    page = int(request.args.get('pageIndex', 1))
    count = int(request.args.get('pageSize', 10))
    bad_data_path = os.path.join(current_app.config['FILE_DIR'], 'tmp-bad-data.csv')
    if not os.path.exists(bad_data_path):
        return jsonify({'code': -1, 'message': 'error'})
    df = pd.read_csv(bad_data_path)
    data = {
        'total': df.shape[0],
        'list': [
            dict(duration=item[0],
                 protocol_type=item[1],
                 service=item[2],
                 flag=item[3],
                 src_bytes=item[4],
                 dst_bytes=item[5],
                 land=item[6],
                 wrong_fragment=item[7],
                 urgent=item[8],
                 hot=item[9],
                 num_failed_logins=item[10],
                 logged_in=item[11],
                 num_compromised=item[12],
                 root_shell=item[13],
                 su_attempted=item[14],
                 num_root=item[15],
                 num_file_creations=item[16],
                 num_shells=item[17],
                 num_access_files=item[18],
                 num_outbound_cmds=item[19],
                 is_hot_login=item[20],
                 is_guest_login=item[21],
                 count=item[22],
                 srv_count=item[23],
                 serror_rate=item[24],
                 srv_serror_rate=item[25],
                 rerror_rate=item[26],
                 srv_rerror_rate=item[27],
                 same_srv_rate=item[28],
                 diff_srv_rate=item[29],
                 srv_diff_host_rate=item[30],
                 dst_host_count=item[31],
                 dst_host_srv_count=item[32],
                 dst_host_same_srv_rate=item[33],
                 dst_host_diff_srv_rate=item[34],
                 dst_host_same_src_port_rate=item[35],
                 dst_host_srv_diff_host_rate=item[36],
                 dst_host_serror_rate=item[37],
                 dst_host_srv_serror_rate=item[38],
                 dst_host_rerror_rate=item[39],
                 dst_host_srv_rerror_rate=item[40],
                 bad_class=item[41]) for item in df.values.tolist()[(page - 1) * count:(page * count + 1)]
        ]
    }
    return jsonify({'code': '0000', 'message': 'success', 'data': data})


@api.route('/save-to-database')
def save_to_database():
    bad_data_path = os.path.join(current_app.config['FILE_DIR'], 'tmp-bad-data.csv')
    df = pd.read_csv(bad_data_path)
    data = [
        dict(duration=item[0],
             protocol_type=item[1],
             service=item[2],
             flag=item[3],
             src_bytes=item[4],
             dst_bytes=item[5],
             land=item[6],
             wrong_fragment=item[7],
             urgent=item[8],
             hot=item[9],
             num_failed_logins=item[10],
             logged_in=item[11],
             num_compromised=item[12],
             root_shell=item[13],
             su_attempted=item[14],
             num_root=item[15],
             num_file_creations=item[16],
             num_shells=item[17],
             num_access_files=item[18],
             num_outbound_cmds=item[19],
             is_hot_login=item[20],
             is_guest_login=item[21],
             count=item[22],
             srv_count=item[23],
             serror_rate=item[24],
             srv_serror_rate=item[25],
             rerror_rate=item[26],
             srv_rerror_rate=item[27],
             same_srv_rate=item[28],
             diff_srv_rate=item[29],
             srv_diff_host_rate=item[30],
             dst_host_count=item[31],
             dst_host_srv_count=item[32],
             dst_host_same_srv_rate=item[33],
             dst_host_diff_srv_rate=item[34],
             dst_host_same_src_port_rate=item[35],
             dst_host_srv_diff_host_rate=item[36],
             dst_host_serror_rate=item[37],
             dst_host_srv_serror_rate=item[38],
             dst_host_rerror_rate=item[39],
             dst_host_srv_rerror_rate=item[40],
             bad_class=item[41]) for item in df.values.tolist()
    ]
    try:
        db.session.execute(BadDataset.__table__.insert(), data)
    except Exception as e:
        print(e)
        return jsonify({'code': -1, 'message': 'error'})
    else:
        return jsonify({'code': '0000', 'message': 'success'})
