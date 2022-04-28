ARG BASE_IMAGE_TYPE=cpu
FROM jafermarq/jetsonfederated_$BASE_IMAGE_TYPE:latest

RUN apt update -y
RUN apt install python3-pip -y
RUN apt install git -y
RUN git clone --depth=1 https://github.com/Zigolox/Federated-Learning-On-RPi.git
RUN pip install flwr==0.17.0
RUN pip install importlib_metadata
WORKDIR Federated-Learning-On-RPi/src/

CMD python3 client.py --cid=$CLIENT --server_address=$SERVER_ADRESS --nb_clients=2
