FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    WAN_REPO_DIR=/workspace/Wan2.2 \
    HF_HOME=/models/hf \
    WAN_CKPT_DIR=/models/Wan2.2-S2V-14B \
    WAN_OUTPUT_DIR=/outputs \
    
    CUDA_HOME=/usr/local/cuda

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git ffmpeg build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN git clone https://github.com/Wan-Video/Wan2.2.git

WORKDIR /workspace/Wan2.2
RUN pip install -r requirements.txt \
    && pip install -r requirements_s2v.txt

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY handler.py /app/handler.py

ENV RUNPOD_HANDLER_MODULE=handler

CMD ["python", "-m", "runpod"]
