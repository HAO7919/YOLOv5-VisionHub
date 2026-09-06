# -*- coding: utf-8 -*-
"""
智能启动器 v2.0
负责：环境检查、GPU检测、依赖检查、配置初始化、启动主程序、友好错误提示
支持：python_path.txt 指定路径 / 系统Python / Conda环境 / 自动创建虚拟环境
兼容：Windows 10 / Windows 11 / Windows Server 2019+
所有错误都会以人话告诉用户，不会闪退
"""
import sys
import os
import platform
import subprocess
from pathlib import Path


def print_info(msg):
    print(f"[信息] {msg}")


def print_warn(msg):
    print(f"[警告] {msg}")


def print_error(msg):
    print(f"[错误] {msg}")


def print_step(step, total, msg):
    print(f"\n[{step}/{total}] {msg}")


def print_separator():
    print("=" * 56)


def detect_gpu():
    """检测是否有 NVIDIA GPU，返回显卡信息或 None"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({
                        "name": parts[0],
                        "memory_mb": int(parts[1]) if parts[1].isdigit() else 0,
                        "driver": parts[2]
                    })
            return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None


def detect_system_info():
    """检测系统信息"""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "cpu": platform.processor() or "未知",
        "gpu": None,
        "is_windows": platform.system() == "Windows",
    }
    # 检测内存（仅 Windows）
    if info["is_windows"]:
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            info["memory_gb"] = round(mem.ullTotalPhys / (1024**3), 1)
        except Exception:
            info["memory_gb"] = "未知"
    else:
        info["memory_gb"] = "未知"

    # 检测 GPU
    gpus = detect_gpu()
    if gpus:
        info["gpu"] = gpus
    return info


def check_python_version(python_path=None):
    """检查 Python 版本是否兼容，返回 (是否兼容, 版本字符串)"""
    if python_path:
        try:
            result = subprocess.run(
                [python_path, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return False, "无法获取版本"
            version_str = result.stdout.strip()
            parts = version_str.split(".")
            major, minor = int(parts[0]), int(parts[1])
        except Exception:
            return False, "无法执行"
    else:
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        major, minor = version.major, version.minor

    if major != 3:
        return False, f"{version_str} (需要 Python 3.x)"

    if minor < 8:
        return False, f"{version_str} (版本过低，需要 3.8 或更高)"

    if minor > 12:
        return False, f"{version_str} (版本过高，PyTorch 可能不兼容，建议 3.8-3.11)"

    return True, version_str


def find_python_path_txt():
    """查找 python_path.txt 指定的 Python 路径"""
    txt_path = Path(__file__).parent / "python_path.txt"
    if txt_path.exists():
        try:
            content = txt_path.read_text(encoding="utf-8").strip()
            if content and Path(content).exists():
                return content
        except Exception:
            pass
    return None


def find_conda_python():
    """查找 Conda 环境中的 Python"""
    # 常见 conda 安装路径
    conda_paths = [
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
        Path("C:/ProgramData/miniconda3"),
        Path("C:/ProgramData/anaconda3"),
        Path("D:/miniconda3"),
        Path("E:/miniconda3"),
        Path("F:/miniconda3"),
    ]
    for conda_root in conda_paths:
        if not conda_root.exists():
            continue
        # 查找 envs 目录下的环境
        envs_dir = conda_root / "envs"
        if envs_dir.exists():
            for env_dir in envs_dir.iterdir():
                python_exe = env_dir / "python.exe"
                if python_exe.exists():
                    ok, ver = check_python_version(str(python_exe))
                    if ok:
                        return str(python_exe), env_dir.name
        # base 环境
        base_python = conda_root / "python.exe"
        if base_python.exists():
            ok, ver = check_python_version(str(base_python))
            if ok:
                return str(base_python), "base"
    return None, None


def find_system_python():
    """查找系统 Python"""
    try:
        result = subprocess.run(
            ["where", "python"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line and Path(line).exists():
                    ok, ver = check_python_version(line)
                    if ok:
                        return line
    except Exception:
        pass
    return None


def find_venv_python():
    """查找项目虚拟环境中的 Python"""
    venv_python = Path(__file__).parent / "venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return None


def create_venv(base_python):
    """创建虚拟环境"""
    print_info("正在创建虚拟环境...")
    try:
        subprocess.check_call([base_python, "-m", "venv", "venv"])
        print_info("虚拟环境创建成功")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"虚拟环境创建失败: {e}")
        return False


def check_and_install_deps(python_path, has_gpu):
    """检查并安装依赖"""
    print_info("检查依赖包...")

    core_modules = [
        ("PyQt5", "PyQt5"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("ultralytics", "ultralytics"),
    ]

    missing = []
    for module, package in core_modules:
        try:
            subprocess.check_call([python_path, "-c", f"import {module}"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            missing.append(package)

    if not missing:
        print_info("所有核心依赖已安装")
        # 检查是否是 CPU 版 PyTorch 但有 GPU
        try:
            result = subprocess.run(
                [python_path, "-c",
                 "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"],
                capture_output=True, text=True, timeout=15
            )
            device = result.stdout.strip()
            if device == "cpu" and has_gpu:
                print_warn("检测到 NVIDIA 显卡，但当前使用的是 CPU 版 PyTorch")
                print_warn("安装 GPU 版 PyTorch 可提升检测速度 5-20 倍")
            elif device == "cuda":
                print_info("GPU 加速已启用")
        except Exception:
            pass
        return True

    print_warn(f"缺少依赖: {', '.join(missing)}")
    print_info("正在安装依赖（首次运行需要 5-15 分钟，请耐心等待）...")

    try:
        # 先安装基础依赖
        basic_packages = ["PyQt5", "opencv-python", "Pillow", "numpy==1.26.4",
                          "pandas", "pyyaml", "tqdm", "ultralytics"]
        print_info("安装基础依赖...")
        subprocess.check_call([
            python_path, "-m", "pip", "install",
            "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
        ] + basic_packages)

        # 安装 PyTorch
        if has_gpu:
            print_info("检测到 NVIDIA 显卡，尝试安装 GPU 版 PyTorch...")
            try:
                subprocess.check_call([
                    python_path, "-m", "pip", "install",
                    "torch==2.3.1", "torchvision==0.18.1",
                    "--index-url", "https://download.pytorch.org/whl/cu118"
                ], timeout=600)
                print_info("GPU 版 PyTorch 安装成功！")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                print_warn("GPU 版安装失败，回退到 CPU 版...")
                subprocess.check_call([
                    python_path, "-m", "pip", "install",
                    "torch==2.3.1", "torchvision==0.18.1",
                    "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
                ])
        else:
            print_info("未检测到 NVIDIA 显卡，安装 CPU 版 PyTorch...")
            subprocess.check_call([
                python_path, "-m", "pip", "install",
                "torch==2.3.1", "torchvision==0.18.1",
                "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
            ])

        # 锁定 numpy 版本
        subprocess.check_call([
            python_path, "-m", "pip", "install", "numpy==1.26.4",
            "--force-reinstall", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print_info("依赖安装完成！")
        return True

    except subprocess.CalledProcessError as e:
        print_error(f"依赖安装失败: {e}")
        print_error("请检查网络连接后重试")
        return False


def check_model_files():
    """检查模型文件"""
    models_dir = Path(__file__).parent / "models"
    if not models_dir.exists():
        models_dir.mkdir(exist_ok=True)
        return False
    pt_files = list(models_dir.glob("*.pt"))
    return len(pt_files) > 0


def launch_main(python_path):
    """启动主程序"""
    print_info("")
    print_separator()
    print_info("  正在启动 YOLOv5 可视化检测系统...")
    print_info("  首次启动需要加载模型，请稍候")
    print_separator()
    print_info("")

    try:
        result = subprocess.run([python_path, "main.py"],
                               cwd=str(Path(__file__).parent))
        if result.returncode != 0:
            print_error(f"\n程序异常退出，退出码: {result.returncode}")
            print_error("可能的原因:")
            print_error("  1. 模型加载失败 - 检查 models 目录下是否有 .pt 文件")
            print_error("  2. 依赖版本冲突 - 删除 venv 文件夹后重新运行")
            print_error("  3. 内存不足 - 关闭其他程序后重试")
            return result.returncode
        return 0
    except Exception as e:
        print_error(f"启动失败: {e}")
        return 1


def main():
    print_separator()
    print("  YOLOv5 可视化检测系统 - 智能启动器 v2.0")
    print_separator()

    os.chdir(Path(__file__).parent)

    # 第0步：系统信息
    print_step(0, 5, "系统信息检测")
    sys_info = detect_system_info()
    print_info(f"操作系统: {sys_info['os']}")
    print_info(f"CPU: {sys_info['cpu']}")
    print_info(f"内存: {sys_info['memory_gb']} GB")
    if sys_info['gpu']:
        for gpu in sys_info['gpu']:
            mem_gb = round(gpu['memory_mb'] / 1024, 1) if gpu['memory_mb'] > 0 else "未知"
            print_info(f"GPU: {gpu['name']} ({mem_gb} GB, 驱动 {gpu['driver']})")
        has_gpu = True
    else:
        print_info("GPU: 未检测到 NVIDIA 显卡（将使用 CPU 检测）")
        has_gpu = False

    # 第1步：查找 Python 环境
    print_step(1, 5, "查找 Python 环境")
    python_path = None
    python_source = ""

    # 1.1 优先读 python_path.txt
    txt_python = find_python_path_txt()
    if txt_python:
        ok, ver = check_python_version(txt_python)
        if ok:
            python_path = txt_python
            python_source = "python_path.txt 指定"
            print_info(f"使用 python_path.txt 指定的 Python: {ver}")
        else:
            print_warn(f"python_path.txt 中的 Python 不可用: {ver}")

    # 1.2 查找项目虚拟环境
    if not python_path:
        venv_python = find_venv_python()
        if venv_python:
            ok, ver = check_python_version(venv_python)
            if ok:
                python_path = venv_python
                python_source = "项目虚拟环境"
                print_info(f"使用项目虚拟环境 Python: {ver}")

    # 1.3 查找 Conda 环境
    if not python_path:
        conda_python, conda_env = find_conda_python()
        if conda_python:
            ok, ver = check_python_version(conda_python)
            if ok:
                python_path = conda_python
                python_source = f"Conda环境({conda_env})"
                print_info(f"使用 Conda 环境 Python: {ver} ({conda_env})")

    # 1.4 查找系统 Python
    if not python_path:
        sys_python = find_system_python()
        if sys_python:
            ok, ver = check_python_version(sys_python)
            if ok:
                python_path = sys_python
                python_source = "系统 Python"
                print_info(f"使用系统 Python: {ver}")

    # 1.5 都没找到，提示安装
    if not python_path:
        print_error("未找到可用的 Python 环境")
        print_error("")
        print_error("请安装以下任一环境后重试:")
        print_error("  方案1（推荐）: 安装 Miniconda")
        print_error("    下载地址: https://docs.conda.io/en/latest/miniconda.html")
        print_error("    安装后创建环境: conda create -n yolov5 python=3.10")
        print_error("    激活环境: conda activate yolov5")
        print_error("")
        print_error("  方案2: 安装 Python 3.10")
        print_error("    下载地址: https://www.python.org/downloads/release/python-31011/")
        print_error("    安装时勾选 'Add Python to PATH'")
        print_error("")
        print_error("安装完成后重新双击 run.bat 启动")
        input("\n按回车键退出...")
        return 1

    # 第2步：如果用的是系统 Python 或 Conda，检查是否需要创建虚拟环境
    print_step(2, 5, "检查运行环境")
    if python_source in ["系统 Python", "Conda环境(base)"]:
        # 建议创建项目虚拟环境，避免污染系统环境
        venv_python = find_venv_python()
        if not venv_python:
            print_info("为避免影响系统环境，建议创建项目虚拟环境")
            print_info("正在创建虚拟环境（仅首次运行需要）...")
            if create_venv(python_path):
                venv_python = find_venv_python()
                if venv_python:
                    python_path = venv_python
                    python_source = "项目虚拟环境"
                    print_info("已切换到项目虚拟环境")
            else:
                print_warn("虚拟环境创建失败，将直接使用当前 Python")
    print_info(f"Python 来源: {python_source}")

    # 第3步：检查/安装依赖
    print_step(3, 5, "检查依赖包")
    if not check_and_install_deps(python_path, has_gpu):
        input("\n按回车键退出...")
        return 1

    # 第4步：检查模型文件
    print_step(4, 5, "检查模型文件")
    has_model = check_model_files()
    if has_model:
        print_info("模型文件已就绪")
    else:
        print_warn("models 目录下没有找到 .pt 模型文件")
        print_warn("软件启动后将尝试联网下载 yolov5s 默认模型")
        print_warn("如果网络不通，启动后请点击'选择模型文件'加载你自己的模型")

    # 第5步：启动程序
    print_step(5, 5, "启动程序")
    if not has_gpu:
        print_info("")
        print_warn("性能提示: 当前使用 CPU 检测，速度较慢")
        print_warn("如果有 NVIDIA 显卡，安装 GPU 版 PyTorch 可提速 5-20 倍")

    exit_code = launch_main(python_path)

    if exit_code != 0:
        input("\n按回车键退出...")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
