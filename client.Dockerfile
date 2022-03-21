FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

RUN apt update -y
RUN apt install python3-pip -y
RUN apt install git -y
RUN git clone --depth=1 https://github.com/adap/flower.git
RUN pip install flwr==0.17.0
RUN pip install importlib_metadata
WORKDIR flower/src/py/

CMD python -m flwr_example.quickstart_pytorch.client --cid=1 --server_address=127.0.0.1:8080 --nb_clients=2