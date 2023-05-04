#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Date: 2023/4/15 13:40
    @Auth: Jacob
    @Desc:
"""
from flask import render_template, request, g
import jwt
from jwt import exceptions

from app import create_app
from app.model.models import UserModel

app = create_app()


@app.route('/')
def index():
    return render_template('index.html')


@app.before_request
def jwt_authentication():
    """
    1.获取请求头Authorization中的token
    2.判断是否以 Bearer开头
    3.使用jwt模块进行校验
    4.判断校验结果,成功就提取token中的载荷信息,赋值给g对象保存
    """
    auth = request.headers.get('Authorization')
    if auth and auth.startswith('Bearer '):
        "提取token 0-6 被Bearer和空格占用 取下标7以后的所有字符"
        token = auth[7:]
        "校验token"
        g.username = None
        try:
            payload = jwt.decode(token, app.config['SALT'], algorithms=['HS256'])
            user = UserModel.query.filter_by(username=payload['username']).first()
            if user is None:
                g.username = 2
            elif user.validate_password(payload['password']):
                g.username = payload.get('username')
            else:
                g.username = 2
        except exceptions.ExpiredSignatureError:  # 'token已失效'
            g.username = 1
        except jwt.DecodeError:  # 'token认证失败'
            g.username = 2
        except jwt.InvalidTokenError:  # '非法的token'
            g.username = 3


if __name__ == '__main__':
    print(app.url_map)
    app.run(host='0.0.0.0')
