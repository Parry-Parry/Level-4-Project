FROM tensorflow/tensorflow:latest-gpu

RUN touch /etc/inside-container

RUN set -ex && pip3 install --upgrade pip
RUN set -ex && pip3 --no-cache-dir install --upgrade pandas \
    tensorflow-addons==0.11.2 \
    numpy \
    transformers \
    tqdm 

RUN mkdir src

