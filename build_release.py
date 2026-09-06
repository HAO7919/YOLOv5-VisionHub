"""
发布包构建脚本
用法：python build_release.py
作用：复制项目到 release 目录，删除用户私人模型和配置，生成干净的发布包
"""
import os
import shutil
import json
from pathlib import Path

# 配置
SRC_DIR = Path(__file__).parent.resolve()
RELEASE_DIR = Path(r"F:\yolov5-desktop-release")

# 官方模型白名单（只有这些会被保留）
OFFICIAL_MODELS = {
    "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt",
    "yolov5su.pt", "yolov5mu.pt", "yolov5lu.pt", "yolov5xu.pt",
    "yolov5n.pt", "yolov5nu.pt",
}

# 需要排除的文件/目录模式
EXCLUDE_DIRS = {
    "__pycache__", ".git", ".idea", ".vscode",
    "test_*.py", "*.pyc",
}

# 需要清理的用户数据
USER_DATA_DIRS = ["output"]
USER_DATA_FILES = ["config.json"]


def is_official_model(name):
    """判断是否是官方模型"""
    return name.lower() in OFFICIAL_MODELS


def copy_project():
    """复制项目到发布目录"""
    print(f"[1/5] 复制项目到 {RELEASE_DIR} ...")
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    
    def ignore_func(dirpath, names):
        ignored = []
        for name in names:
            full = Path(dirpath) / name
            if name in EXCLUDE_DIRS:
                ignored.append(name)
            elif name.endswith(".pyc"):
                ignored.append(name)
            elif name.startswith("test_") and name.endswith(".py"):
                ignored.append(name)
        return ignored
    
    shutil.copytree(SRC_DIR, RELEASE_DIR, ignore=ignore_func)
    print("  复制完成")


def clean_models():
    """清理模型目录，只保留官方模型"""
    print("[2/5] 清理模型目录（只保留官方模型）...")
    models_dir = RELEASE_DIR / "models"
    if not models_dir.exists():
        models_dir.mkdir(parents=True)
        return
    
    removed = []
    for pt_file in models_dir.glob("*.pt"):
        if not is_official_model(pt_file.name):
            pt_file.unlink()
            removed.append(pt_file.name)
    
    if removed:
        print(f"  已删除用户模型: {', '.join(removed)}")
    else:
        print("  没有用户模型需要删除")
    
    # 列出保留的官方模型
    kept = [f.name for f in models_dir.glob("*.pt")]
    print(f"  保留的官方模型: {', '.join(kept)}")


def reset_config():
    """重置配置文件为默认值"""
    print("[3/5] 重置配置文件...")
    config_path = RELEASE_DIR / "config.json"
    
    # 从 config.py 读取默认配置
    default_config = {
        "model": {
            "conf_thres": 0.25,
            "iou_thres": 0.45,
            "line_thickness": 2,
            "current_model": "yolov5su.pt"
        },
        "ui": {
            "theme": "light",
            "window_width": 1280,
            "window_height": 800,
            "video_speed": 1.0,
            "camera_res": "640x480",
            "current_mode": "image"
        },
        "cache": {
            "torch_home": ""
        },
        "first_run": True,
        "history": {
            "records": []
        }
    }
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)
    
    print("  配置已重置（默认模型: yolov5su.pt, 主题: 白色, 首次运行: 是）")


def clean_user_data():
    """清理用户数据"""
    print("[4/5] 清理用户数据...")
    for dir_name in USER_DATA_DIRS:
        dir_path = RELEASE_DIR / dir_name
        if dir_path.exists():
            for f in dir_path.iterdir():
                if f.is_file():
                    f.unlink()
            print(f"  已清理 {dir_name} 目录")
    
    # 清理用户截图等临时文件
    for pattern in ["user_screenshot.png", "*.tmp"]:
        for f in RELEASE_DIR.glob(pattern):
            if f.is_file():
                f.unlink()
                print(f"  已删除临时文件: {f.name}")
    
    print("  用户数据清理完成")


def create_readme():
    """创建发布说明"""
    print("[5/5] 创建发布说明...")
    readme_content = """# YOLO 可视化检测工具 - 使用说明

## 快速开始
1. 确保电脑已安装 Python 3.8 ~ 3.11
2. 双击 `run.bat` 启动
3. 首次启动会自动安装依赖（需要联网，约2-5分钟）
4. 启动后默认加载官方模型 yolov5su.pt

## 加载你自己的模型
1. 点击「选择模型文件」
2. 选择你训练好的 .pt 文件（支持 YOLOv5/v8/v9/v10/v11）
3. 软件会自动复制模型到 models 目录并加载
4. 旧版 YOLOv5 训练的模型也能加载（完全离线）

## 功能
- 图片检测：打开图片，自动检测并标注
- 视频检测：打开视频，逐帧检测并保存结果
- 摄像头检测：实时检测
- 屏幕检测：实时检测屏幕内容
- 参数调节：置信度、IOU、线宽
- 历史记录：最近检测的文件
- 主题切换：白色/深色/跟随系统

## 系统要求
- Windows 10 及以上
- Python 3.8 ~ 3.11
- 推荐 NVIDIA GPU（CPU 也能用，慢一些）

## 常见问题
- 启动慢：首次加载模型需要编译，耐心等待
- 模型加载失败：确认是 .pt 格式，且是 YOLO 系列模型
- 检测慢：CPU 检测会慢一些，建议用 GPU
"""
    readme_path = RELEASE_DIR / "使用说明.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("  使用说明已创建")


def main():
    print("=" * 50)
    print("  YOLO 可视化检测工具 - 发布包构建")
    print("=" * 50)
    
    copy_project()
    clean_models()
    reset_config()
    clean_user_data()
    create_readme()
    
    # 统计发布包大小
    total_size = sum(f.stat().st_size for f in RELEASE_DIR.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    print("\n" + "=" * 50)
    print(f"  发布包构建完成！")
    print(f"  位置: {RELEASE_DIR}")
    print(f"  大小: {size_mb:.1f} MB")
    print(f"  默认模型: yolov5su.pt（官方）")
    print(f"  已排除: 用户私人模型、配置、检测结果")
    print("=" * 50)
    print("\n把这个文件夹给别人即可，别人双击 run.bat 启动。")


if __name__ == "__main__":
    main()
