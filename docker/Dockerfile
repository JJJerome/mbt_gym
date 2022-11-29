FROM ubuntu:20.04 AS MBT_GYM
USER root

RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get -y install \
    fish \
    tmux \
    python3-pip

WORKDIR /home
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY launcher.sh launcher.sh
RUN chmod +x launcher.sh
