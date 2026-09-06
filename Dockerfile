FROM python:3.10-slim

LABEL maintainer="HAO7919"
LABEL description="YOLOv5-VisionHub - 桌面版目标检测工具"

# 安装系统依赖（PyQt5和OpenCV需要）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY . .

# 设置环境变量
ENV QT_QPA_PLATFORM=xcb
ENV DISPLAY=:0

# 默认启动命令（需要X11转发：docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix）
CMD ["python", "main.py"]
