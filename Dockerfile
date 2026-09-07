# YOLOv5-VisionHub Dockerfile (优化版)
# 多阶段构建，最小化镜像体积
#
# CPU版: docker build -t yolov5-visionhub:cpu .
# GPU版: docker build --build-arg DEVICE=gpu -t yolov5-visionhub:gpu .

ARG DEVICE=cpu

# ========== 阶段1: 构建依赖 ==========
FROM python:3.10-slim AS builder

WORKDIR /build

# 安装编译依赖（只在构建阶段需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements
COPY requirements.txt .

# 安装Python依赖到单独目录
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 根据设备类型安装PyTorch
FROM builder AS builder-cpu
RUN pip install --no-cache-dir --prefix=/install torch torchvision --index-url https://download.pytorch.org/whl/cpu

FROM builder AS builder-gpu
RUN pip install --no-cache-dir --prefix=/install torch torchvision --index-url https://download.pytorch.org/whl/cu118

FROM builder-${DEVICE} AS builder-final

# 清理不需要的文件（减小体积）
RUN find /install -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /install -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true \
    && find /install -type d -name "test" -exec rm -rf {} + 2>/dev/null || true \
    && find /install -name "*.pyc" -delete 2>/dev/null || true \
    && find /install -name "*.pyo" -delete 2>/dev/null || true \
    && rm -rf /install/lib/python3.10/site-packages/torch/test \
    && rm -rf /install/lib/python3.10/site-packages/torch/include \
    && rm -rf /install/lib/python3.10/site-packages/torch/share \
    || true

# ========== 阶段2: 运行时镜像 ==========
FROM python:3.10-slim AS runtime

LABEL maintainer="HAO7919"
LABEL description="YOLOv5-VisionHub - Desktop Object Detection Tool (optimized)"

# 只安装运行时需要的系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    libfontconfig1 \
    libx11-6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 从构建阶段复制依赖
COPY --from=builder-final /install /usr/local

# 复制源代码（.dockerignore会排除大文件）
COPY . /app

WORKDIR /app

# 再次清理
RUN find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /app -name "*.pyc" -delete 2>/dev/null || true

# 设置环境变量
ENV QT_QPA_PLATFORM=xcb
ENV DISPLAY=:0
ENV PYTHONUNBUFFERED=1

# 默认启动命令
CMD ["python", "main.py"]
