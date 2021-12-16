FROM tensorflow/tensorflow:2.6.0-gpu-jupyter

ENV PATH=$PATH:/usr/local/go/bin:/root/go/bin

RUN apt update && \
  apt-get -y install wget && \
  wget "https://go.dev/dl/go1.17.4.linux-amd64.tar.gz" && \
  rm -rf /usr/local/go && \
  tar -C /usr/local -xzvf go1.17.4.linux-amd64.tar.gz && \
  wget "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz" && \
  tar -C /usr -xzvf libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz