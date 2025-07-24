FROM tensorflow/tensorflow:2.16.1-gpu

ARG USER_ID
ARG GROUP_ID

RUN groupadd -r -g $GROUP_ID myuser && useradd -r -u $USER_ID -g myuser -m -d /home/myuser myuser
ENV SHELL /bin/bash

RUN mkdir -p /home/myuser/code && chown -R myuser:myuser /home/myuser/code

WORKDIR /home/myuser/code

RUN apt update
RUN apt install -y jq curl ca-certificates
RUN pip install --upgrade pip
RUN pip install numpy==1.26.4
RUN pip install scikit-learn==1.4.2
RUN pip install aeon==0.11.1
RUN pip install keras==3.6.0
RUN pip install hydra-core==1.3.2
RUN pip install omegaconf==2.3.0
RUN pip install pandas==2.0.3
RUN pip install matplotlib==3.9.0
