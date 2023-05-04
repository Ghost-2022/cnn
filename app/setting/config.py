#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Date: 2023/4/12 19:22
    @Auth: Jacob
    @Desc:
"""
import os
from pathlib import Path


class BaseConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent

    IMG_DIR = os.path.join(BASE_DIR, 'static', 'img')
    FILE_DIR = os.path.join(BASE_DIR, 'static', 'file')
    AVG_LOSS_PNG = os.path.join(IMG_DIR, 'avg-loss.png')
    TRAINING_PNG = os.path.join(IMG_DIR, 'training-set-accuracy.png')
    TESTING_PNG = os.path.join(IMG_DIR, 'testing-set-accuracy.png')
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True
    SALT = '3iLgwcbINAoUKAZtovmnrsSkDYUEzza0'


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = r'mysql://lichunxu:TT4RVhRjlJUwjEj*@146.56.219.98:3506/cnn?charset=utf8mb4'


class ProductionConfig(BaseConfig):
    DEBUG = False


class TestingConfig(BaseConfig):
    DEBUG = True

