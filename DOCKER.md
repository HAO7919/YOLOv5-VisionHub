
Docker 使用指南

首先说体积（说实话）

-v ${PWD}/output:/app/output ` 版本 | 大小 | 说明 |
|------|------|------|
| CPU稳定版 | 约1.5GB | 用slim基础镜像，兼容性好 |
| CPU超轻量版 | 约1GB | 用alpine基础镜像，可能有兼容性问题 |
| GPU版 | 约4GB | 带CUDA，想快就得大 |

**为啥还是这么大？** 因为PyTorch本身就占了1GB左右，Dockerfile能优化的都优化了，再小就得换技术栈了。

**想真正小？** 把PyTorch换成ONNX Runtime，镜像能降到500MB以下。但这得改代码，把模型转成onnx格式，推理部分也得重写。有兴趣自己研究，这里就不提供了。

## 超轻量版（想更小的试试这个）

用alpine基础镜像，删了更多东西，体积能到1GB左右。但alpine上PyQt5可能出问题，出问题就换回稳定版。

```bash
# 构建超轻量版
docker build -f Dockerfile.lite -t yolov5-visionhub:lite .

# 运行
`docker run -it --rm `
  -e DISPLAY=host.docker.internal:0 `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/output:/app/output `
yolov5-visionhub:lite
“```”

## 你需要先安装什么

1. Docker Desktop：直接到官网下载并安装即可
2. Windows 用户还需要安装 WSL2（安装 Docker 时会提示你）
3. 如果想用 GPU 加速，还需要安装 nvidia-docker

## 如何运行  
```

### CPU版（大部分人用这个）

```bash
# 构建镜像（第一次会慢点，要下载东西）
docker build -t yolov5-visionhub:cpu .

# 运行（Windows PowerShell里执行）
docker run -it --rm `
  -e DISPLAY=host.docker.internal:0 `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/output:/app/output `
  yolov5-visionhub:cpu
```

### GPU版（有N卡的用这个）

```bash
# 构建
docker build --build-arg DEVICE=gpu -t yolov5-visionhub:gpu .

# 运行
docker run -it --rm --gpus all `
  -e DISPLAY=host.docker.internal:0 `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/output:/app/output `
  yolov5-visionhub:gpu
```

### 用docker-compose（更省事）

```bash
# CPU版
docker-compose up yolov5-cpu

# GPU版
docker-compose up yolov5-gpu
```

## Windows用户注意

Docker在Windows上跑GUI程序比较麻烦，得额外装个VcXsrv：

1. 下载VcXsrv：https://sourceforge.net/projects/vcxsrv/
2. 装好后启动，勾选"Disable access control"
3. 然后再跑上面的docker命令

嫌麻烦就直接用Release里的exe，双击就跑，比Docker省事多了。

## 还想再小点？

### 方法1：换alpine基础镜像

把Dockerfile里的`python:3.10-slim`改成`python:3.10-alpine`，能再小几百MB。但alpine上PyQt5可能出问题，得自己试。

### 方法2：删不用的依赖

打开requirements.txt，把你用不到的包删掉。比如不用屏幕检测就删mss，不用导出就删相关的包。

### 方法3：用docker-slim

```bash
docker-slim build --target yolov5-visionhub:cpu --tag yolov5-visionhub:cpu-slim
```
一般能再压30%-50%。

### 方法4：终极方案——换ONNX Runtime

把PyTorch换成ONNX Runtime，镜像能降到500MB以下。但这得改代码，把模型转成onnx格式，推理部分也得重写。有兴趣可以自己研究。

## 常见问题

**Q: 报错连不上display？**
A: Windows用户没装VcXsrv，或者没启动。Linux用户加个`-v /tmp/.X11-unix:/tmp/.X11-unix`参数。

**Q: 构建好慢？**
A: 第一次要下载PyTorch，几百MB呢，耐心等。后面再构建就快了，有缓存。

**Q: 我就想用，折腾这玩意干啥？**
A: 对，普通用户直接下exe就行。Docker是给开发者或者用Linux/Mac的人准备的。

## 用自己训练的模型

把.pt文件扔到models目录里，Docker会自动挂载进去，软件里就能选到。
