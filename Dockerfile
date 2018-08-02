FROM pytorch/pytorch

RUN apt-get update && apt-get install -y python-opencv
RUN pip install matplotlib numpy pillow opencv-python tensorboardX flask seaborn pickle


ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

WORKDIR /opt/code
