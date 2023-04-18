#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Date: 2023/4/12 18:57
    @Auth: Jacob
    @Desc:
"""
import os

from flask import Flask
from flask_cors import CORS
from flask_migrate import Migrate
import pymysql
import torch

from app.common.db import db
from app.setting import env_map
from app.api import api
from app.model.model import Model

pymysql.install_as_MySQLdb()

migrate = Migrate()


def create_app():
    app = Flask(__name__)
    env = os.getenv('APP_ENV', 'development')
    app.config.from_object(env_map.get(env))

    db.init_app(app)
    migrate.init_app(app, db)

    CORS(app, resources={r"/api/*": {"origins": "*"}})
    app.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    app.model = Model()  # 这里需要重新模型结构，My_model
    app.model.to(app.device)
    app.model.load_state_dict(torch.load(os.path.join(app.config['FILE_DIR'], 'cnn_lstm.pth')))
    app.model.eval()

    app.register_blueprint(api, url_prefix='/api/')
    return app
