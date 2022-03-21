FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

RUN apt update -y
RUN apt install python3-pip -y
RUN apt install git -y
RUN git clone --depth=1 https://github.com/Zigolox/Federated-Learning-On-RPi.git
RUN pip install flwr==0.17.0
RUN pip install importlib_metadata
WORKDIR Federated-Learning-On-RPi/src/
CMD python server.py 




