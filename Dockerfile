FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install matplotlib pandas

COPY ./scripts /code/scripts
WORKDIR /code/scripts
