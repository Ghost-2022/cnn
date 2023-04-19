# 安装说明
## 1. 配置数据库
位置于 `app/setting/config.py`
SQLALCHEMY_DATABASE_URI
mysql://[username]:[password]@[ip]:[port]/[database]?charset=utf8mb4

## 2. 安装所需环境
执行`pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/`

## 3. 运行
执行`python main.iy`