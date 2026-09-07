# Docker 使用说明

## 为什么用 Docker？

- 不用装 Python、PyTorch 等环境
- 跨平台，Windows/Mac/Linux 都能用
- 隔离环境，不会搞乱本机

## 体积说明

| 版本 | 镜像大小 | 说明 |
|------|---------|------|
| CPU版 | 约2GB | 不需要NVIDIA显卡，所有电脑都能用 |
| GPU版 | 约5GB | 需要NVIDIA显卡 + nvidia-docker，检测速度更快 |

> **为什么这么大？** PyTorch + CUDA 本身就很大（CUDA库约3GB），这是深度学习框架的通病，不是软件的问题。如果只需要CPU，用CPU版可以省很多空间。

## 前置要求

1. 安装 Docker Desktop：https://www.docker.com/products/docker-desktop/
2. Windows 用户需要安装 WSL2（Docker Desktop 会自动提示安装）
3. GPU版本需要 NVIDIA 显卡 + nvidia-docker

## 快速开始

### CPU版本（推荐，所有电脑都能用）

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

### GPU版本（有NVIDIA显卡）

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

## Windows 用户额外步骤

Docker 在 Windows 上运行 GUI 程序需要额外配置：

1. 安装 VcXsrv（X Server）：https://sourceforge.net/projects/vcxsrv/
2. 启动 VcXsrv，勾选 "Disable access control"
3. 然后再运行上面的 docker 命令

或者更简单的方式：**直接用 Release 里的 exe 版本**，不用折腾 Docker。

## 常见问题

### Q: 报错 "could not connect to display"
A: GUI显示没配置好。Windows用户需要装VcXsrv，Linux用户需要加 `-v /tmp/.X11-unix:/tmp/.X11-unix`。

### Q: Docker版本比exe版本还大？
A: 是的，Docker镜像包含了完整的系统环境。exe版本只包含运行时依赖，反而更小。

### Q: 我就想简单用，用哪个？
A: **直接下载 Release 里的 exe**，解压双击就能用，最省事。Docker 适合开发者或者Linux/Mac用户。

## 自己训练的模型

把 `.pt` 文件放到 `models/` 目录，Docker 会自动挂载进去，软件里就能选到。
