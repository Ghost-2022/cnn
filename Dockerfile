FROM python:3.9.16-slim-bullseye

COPY requirements.txt /code/cnn/requirements.txt
WORKDIR /code/cnn/
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

CMD [ "bash" ]