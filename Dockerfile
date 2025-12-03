FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev python3-venv \
    git curl wget nano build-essential vim nvtop cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN wget https://repo.anaconda.com/archive/Anaconda3-2025.06-1-Linux-x86_64.sh
RUN chmod +x Anaconda3-2025.06-1-Linux-x86_64.sh
RUN bash Anaconda3-2025.06-1-Linux-x86_64.sh -b -p /root/anaconda3 && rm -f Anaconda3-2025.06-1-Linux-x86_64.sh

ENV PATH="/root/anaconda3/bin:$PATH"
ENV CONDA_ALWAYS_YES=true
RUN conda init bash

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

WORKDIR /workspace
COPY grpo_environment.yaml grpo_environment.yaml
RUN conda env create -f grpo_environment.yaml

COPY scripts/ scripts/
COPY data/ data/
COPY run_grpo.sh run_grpo.sh
COPY open_server.sh open_server.sh
COPY down_model.py down_model.py
COPY down_model.sh down_model.sh

RUN echo "conda activate grpo" >> /root/.bashrc
SHELL ["/bin/bash", "-c"]