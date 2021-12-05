FROM golang:1.17

RUN apt update && \
  apt-get -y install wget && \
  wget "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz" && \
  tar -C /usr -xzvf libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz