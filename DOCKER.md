# Docker 使用说明（优化版）

## 体积对比

| 版本 | 优化前 | 优化后 | 说明 |
|------|--------|--------|------|
| CPU版 | 约2.5GB | **约1.5GB** | 不需要NVIDIA显卡，所有电脑都能用 |
| GPU版 | 约5GB | **约4GB** | 需要NVIDIA显卡 + nvidia-docker，检测速度更快 |

> 优化了什么：多阶段构建、删除测试文件、清理缓存、只装运行时依赖、.dockerignore排除大文件。

## 前置要求

1. 安装 Docker Desktop：https://www.docker.com/products/docker-desktop/
2. Windows 用户需要安装 WSL2（Docker Desktop 会自动提示）
3. GPU版本需要 NVIDIA 显卡 + nvidia-docker

## 快速开始

### CPU版本（推荐）

```bash
# 构建镜像
docker build -t yolov5-visionhub:cpu .

# 运行（Windows PowerShell）
docker run -it --rm `
  -e DISPLAY=host.docker.internal:0 `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/output:/app/output `
  yolov5-visionhub:cpu
```

### GPU版本

```bash
# 构建镜像
docker build --build-arg DEVICE=gpu -t yolov5-visionhub:gpu .

# 运行
docker run -it --rm --gpus all `
  -e DISPLAY=host.docker.internal:0 `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/output:/app/output `
  yolov5-visionhub:gpu
```

## 用 docker-compose（更简单）

```bash
# CPU版本
docker-compose up yolov5-cpu

# GPU版本
docker-compose up yolov5-gpu
```

## 进一步压缩体积

如果还想更小，可以试试：

### 1. 用更小的基础镜像

把 Dockerfile 里的 `python:3.10-slim` 换成 `python:3.10-alpine`，能再小几百MB。但Alpine上PyQt5可能有兼容性问题，需要自己测试。

### 2. 只装需要的依赖

编辑 `requirements.txt`，删掉你不用的包。比如不用屏幕检测可以删掉 `mss`，不用导出可以删掉相关依赖。

### 3. 删掉PyTorch的CUDA库（CPU版）

CPU版构建完成后，可以进入容器删掉不需要的CUDA相关文件：
```bash
docker run --rm -it yolov5-visionhub:cpu bash
# 在容器里执行
rm -rf /usr/local/lib/python3.10/site-packages/torch/lib/libcud*
```

### 4. 用 docker-slim 进一步压缩

```bash
# 安装docker-slim后执行
docker-slim build --target yolov5-visionhub:cpu --tag yolov5-visionhub:cpu-slim
```
通常能再压缩30%-50%。

## Windows 用户额外步骤

Docker 在 Windows 上运行 GUI 程序需要配置 X Server：

1. 安装 VcXsrv：https://sourceforge.net/projects/vcxsrv/
2. 启动 VcXsrv，勾选 "Disable access control"
3. 然后再运行上面的 docker 命令

> **更简单的方式：** 直接用 Release 里的 exe 版本，不用折腾 Docker。

## 常见问题

### Q: 报错 "could not connect to display"
A: GUI显示没配置好。Windows用户需要装VcXsrv，Linux用户需要加 `-v /tmp/.X11-unix:/tmp/.X11-unix`。

### Q: 构建很慢？
A: 第一次构建需要下载PyTorch（几百MB），耐心等。后续构建会用缓存，快很多。

### Q: 我就想简单用，用哪个？
A: **直接下载 Release 里的 exe**，解压双击就能用，最省事。Docker 适合开发者或者Linux/Mac用户。

## 自己训练的模型

把 `.pt` 文件放到 `models/` 目录，Docker 会自动挂载进去，软件里就能选到。
