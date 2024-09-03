FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m rashika

RUN chown -R rashika:rashika /home/rashika

COPY --chown=rashika . /home/rashika/MELANOMA-DEEP-LEARNING

USER rashika

RUN cd /home/rashika/MELANOMA-DEEP-LEARNING && pip3 install -r requirements.txt

WORKDIR /home/rashika/MELANOMA-DEEP-LEARNING