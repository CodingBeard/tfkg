FROM tensorflow/tensorflow:devel-gpu@sha256:1452331ddc5c1995b508114ec9bae0812e17ce14342bd67e08244a07c0d5a5cb

ENV PATH=$PATH:/usr/local/go/bin:/root/go/bin

RUN apt update && \
  apt -y install wget && \
  wget "https://go.dev/dl/go1.17.4.linux-amd64.tar.gz" && \
  rm -rf /usr/local/go && \
  tar -C /usr/local -xzvf go1.17.4.linux-amd64.tar.gz

COPY libtensorflow.tar.gz /tf/
RUN tar -C /usr/local -xzvf /tf/libtensorflow.tar.gz

COPY tensorflow-2.6.0-cp38-cp38-linux_x86_64.whl /tf/

RUN pip install /tf/tensorflow-2.6.0-cp38-cp38-linux_x86_64.whl
RUN pip uninstall -y keras && pip install keras==2.6.0

RUN ldconfig