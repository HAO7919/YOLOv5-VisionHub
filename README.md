# YOLOv5-VisionHub

YOLOv5 可视化检测系统 - 桌面版目标检测工具

## 功能特性

- **图片检测** - 打开图片，一键检测，支持导出 JSON/CSV/VOC XML
- **视频检测** - 视频文件逐帧检测，自动保存结果视频
- **摄像头实时检测** - 调用摄像头实时检测，支持截图保存
- **屏幕实时检测** - 实时捕获屏幕内容进行检测，叠加显示检测框
- **多模型支持** - 支持 YOLOv5/v8/v9/v10/v11 模型，可加载自己训练的模型
- **多语言界面** - 支持中文、英语、日语、韩语、法语、德语、西班牙语、俄语、葡萄牙语
- **CPU/GPU 自适应** - 有 NVIDIA 显卡自动用 GPU，没有自动用 CPU
- **参数可调** - 置信度阈值、IOU 阈值、最大检测数、线条粗细
- **历史记录** - 自动记录最近检测，方便回看

## 系统要求

- Windows 10 及以上（64位）
- 内存：4GB 以上（推荐 8GB）
- GPU：NVIDIA 显卡（可选，没有也能用 CPU）

## 快速开始

### 直接使用（推荐）

1. 下载最新的 Release 包
2. 解压到任意目录
3. 双击 `YOLO检测工具.exe` 启动
4. 不需要安装 Python、PyTorch 等环境

### 从源码运行

```bash
# 克隆仓库
git clone https://github.com/你的用户名/YOLOv5-VisionHub.git
cd YOLOv5-VisionHub

# 安装依赖
pip install -r requirements.txt

# 运行
python main.py
```

## 使用自己的模型

把训练好的 `.pt` 文件放到 `models/` 目录，重启软件，在左侧「模型管理」里选择并切换。

## 项目结构

```
YOLOv5-VisionHub/
├── main.py              # 程序入口
├── main_window.py       # 主窗口界面
├── detector.py          # 检测引擎封装
├── screen_capture.py    # 屏幕捕获
├── theme.py             # 主题样式
├── config.py            # 配置管理
├── i18n.py              # 多语言支持
├── launcher.py          # 智能启动器
├── requirements.txt     # Python 依赖
├── models/              # 模型文件目录
├── assets/              # 图标资源
└── torch_cache/         # 旧版模型支持
```

## 打包 exe

```bash
pip install pyinstaller
pyinstaller yolov5_app.spec --clean --noconfirm
```

打包后把 `models/`、`torch_cache/`、`assets/` 复制到 exe 旁边。

## 许可证

MIT License

## 免责声明

本软件仅供学习和研究使用，请勿用于非法用途。
