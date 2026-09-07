# YOLOv5-VisionHub

YOLOv5 Visual Detection System - Desktop Object Detection Tool

> **For regular users: No installation required! Download the Release package, extract it, and double-click `YOLO检测工具.exe` to run.**
>
> The "Run from source" section below is for developers who want to modify the code. Regular users can ignore it.

[中文文档](README.zh-CN.md) | English

## Features

- **Image Detection** - Open an image and detect objects with one click, supports exporting JSON/CSV/VOC XML
- **Video Detection** - Frame-by-frame detection on video files, automatically saves result video
- **Live Camera Detection** - Real-time detection using webcam, supports screenshot capture
- **Screen Capture Detection** - Real-time screen capture detection with overlay bounding boxes
- **Multi-Model Support** - Supports YOLOv5/v8/v9/v10/v11 models, load your own trained models
- **Multi-Language UI** - Supports Chinese, English, Japanese, Korean, French, German, Spanish, Russian, Portuguese
- **CPU/GPU Auto-Switch** - Automatically uses NVIDIA GPU if available, falls back to CPU
- **Adjustable Parameters** - Confidence threshold, IOU threshold, max detections, line thickness
- **History Records** - Automatically records recent detections for easy review

## Screenshots

### Main Interface
![Main Interface](screenshots/main-interface.png)

### Screen Real-time Detection
![Screen Detection](screenshots/screen-detection.png)

## System Requirements

- Windows 10 or later (64-bit)
- RAM: 4GB+ (8GB recommended)
- GPU: NVIDIA graphics card (optional, CPU works without it)

## Quick Start (Regular Users)

1. Click **Releases** on the right side of the page, download the latest zip package
2. Extract to any directory (avoid Chinese characters in the path)
3. Double-click `YOLO检测工具.exe` to launch
4. **No need to install Python, PyTorch, or any environment**

## Using Your Own Model

Put your trained `.pt` file into the `models/` directory, restart the software, and select and switch to it in the "Model Management" panel on the left.

## Docker

Want to run in Docker? See [DOCKER.md](DOCKER.md) for detailed instructions. Supports both CPU and GPU versions.

> **Note:** Docker images are larger than the exe package because they include a full system environment. For regular Windows users, the exe release is simpler and smaller.

## Package Size Explanation

The release package is ~3GB because it bundles:
- **PyTorch + CUDA** (~2GB): Required for GPU acceleration, this is the bulk of the size
- **OpenCV, PyQt5, and other dependencies** (~500MB)
- **4 pre-trained YOLO models** (~200MB)

**If you don't need GPU:** The software automatically falls back to CPU. You can also delete unused model files from the `models/` folder to save space.

---

## Developers: Run from Source

> Only for those who want to modify the code. Regular users should use the Release package above.

```bash
# Clone the repository
git clone https://github.com/HAO7919/YOLOv5-VisionHub.git
cd YOLOv5-VisionHub

# Install PyTorch (must install first, choose based on GPU availability)
# CPU version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# GPU version (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Run
python main.py
```

## Project Structure

```
YOLOv5-VisionHub/
├── main.py              # Entry point (splash screen, single instance lock)
├── main_window.py       # Main window UI (all interaction logic)
├── detector.py          # Detection engine (model loading, inference, drawing)
├── screen_capture.py    # Real-time screen detection
├── theme.py             # Theme styles (light/dark, dialog animations)
├── config.py            # Configuration management (read/write config.json)
├── i18n.py              # Multi-language support (9 languages)
├── launcher.py          # Smart launcher (environment detection)
├── requirements.txt     # Python dependencies
├── run.bat              # Windows one-click launch script
├── start.sh             # Linux/Mac launch script
├── yolov5_app.spec      # PyInstaller build configuration
├── build_exe.bat        # Windows one-click build script
├── Dockerfile           # Docker container support
├── models/              # Model files directory (.gitkeep placeholder)
├── assets/              # Icon resources
├── torch_cache/         # Legacy model compatibility
├── docs/                # Project homepage (GitHub Pages)
├── .github/workflows/   # CI auto-test and Pages deployment
├── .gitignore
├── .gitattributes
├── LICENSE              # MIT
└── README.md
```

## Build EXE

```bash
# Install PyInstaller
pip install pyinstaller

# One-click build (auto-copies models, cleans personal files)
build_exe.bat
```

After building, the `dist/YOLO检测工具/` directory is the complete program. Compress and send to others. They just extract and double-click the exe - no environment installation needed.

## License

MIT License

This project is licensed under MIT. You can use, modify, and distribute it freely, including for commercial purposes. Just keep the original copyright notice.

## Disclaimer

This software is for learning and research purposes only. Do not use for illegal purposes.
