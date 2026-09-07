# YOLOv5-VisionHub Dockerfile
# 支持 CPU 和 GPU 两种构建版本
#
# CPU版本构建: docker build -t yolov5-visionhub:cpu .
# GPU版本构建: docker build --build-arg DEVICE=gpu -t yolov5-visionhub:gpu .
#
# 注意：Docker中运行GUI需要X11转发，详见 README.md 的 Docker 部分

ARG DEVICE=cpu
FROM python:3.10-slim AS base

LABEL maintainer="HAO7919"
LABEL description="YOLOv5-VisionHub - Desktop Object Detection Tool"

# 安装系统依赖（PyQt5和OpenCV需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    libfontconfig1 \
    libx11-6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制requirements，利用Docker缓存
COPY requirements.txt .

# CPU版本：安装CPU版PyTorch（体积小）
FROM base AS cpu
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# GPU版本：安装CUDA版PyTorch（体积大，需要nvidia-docker）
FROM base AS gpu
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir -r requirements.txt

# 最终镜像
FROM ${DEVICE} AS final

# 复制源代码
COPY . .

# 设置环境变量
ENV QT_QPA_PLATFORM=xcb
ENV DISPLAY=:0

# 默认启动命令
CMD ["python", "main.py"]
