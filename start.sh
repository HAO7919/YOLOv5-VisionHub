#!/bin/bash
# YOLOv5-VisionHub 启动脚本 (Linux/macOS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到 python3，请先安装 Python 3.8+"
    exit 1
fi

# 检查依赖
if ! python3 -c "import PyQt5" 2>/dev/null; then
    echo "[提示] 正在安装依赖..."
    pip3 install -r requirements.txt
fi

# 启动程序
echo "[启动] YOLOv5-VisionHub..."
python3 main.py
