FROM ubuntu:20.04 AS MBT_GYM
USER root
WORKDIR ../


RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get -y install \
    fish \
    tmux \
    && curl -L https://github.com/oh-my-fish/oh-my-fish/raw/master/bin/install > install \
    && fish install \
    && chsh -s `which fish` \
    && omf install pyenv \
    && pyenv install 3.9.12 \
    && pyenv virtualenv 3.9.12 mbt_gym \
    && pyenv global mbt_gym \
    && pyenv activate mbt_gym \
    && pip install --no-cache-dir -r requirements.txt
