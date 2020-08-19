#!/bin/bash
FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime
MAINTAINER mieryu@naver.com

# Set locale in korea
RUN apt-get update && apt-get install -y locales
RUN locale-gen ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8

# Install basic packages
RUN apt-get install -y vim curl git net-tools

# Change pip mirror
RUN mkdir -p ~/.pip/
RUN echo -e "[global]\nindex-url=http://mirror.kakao.com/pypi/simple\ntrusted-host=mirror.kakao.com" > ~/.pip/pip.conf

# Install Python lib
RUN pip install matplotlib && \
    pip install opencv-python && \
    pip install pandas && \
    pip install numpy==1.17 && \
    pip install notebook && \
    pip install tensorboard && \
    pip install tensorboardX

# Set WORKDIR
WORKDIR /workspace

CMD jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --notebookApp.token='mieryu'

