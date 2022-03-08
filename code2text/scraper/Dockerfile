FROM python:3.7.3

RUN touch /etc/inside-container

RUN set -ex && pip3 install --upgrade pip
RUN set -ex && pip3 --no-cache-dir install --upgrade jupyter \
    tree_sitter \
    requests \
    pyhive \
    tqdm \
    pandas \
    python-arango \
    docopt \
    elasticsearch \
    dpu_utils

