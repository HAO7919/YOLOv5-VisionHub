@echo off
chcp 65001 >nul
title 打包 YOLOv5 可视化检测系统为 EXE

echo ============================================================
echo    打包 YOLOv5 可视化检测系统为独立 EXE 程序
echo ============================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [错误] 请先运行 run.bat 安装依赖后再打包
    pause
    exit /b 1
)

set VENV_PYTHON=venv\Scripts\python.exe
set VENV_PIP=venv\Scripts\pip.exe

:: 安装 PyInstaller
"%VENV_PYTHON%" -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [信息] 正在安装 PyInstaller...
    "%VENV_PIP%" install pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo.
echo [信息] 开始打包...
echo [信息] 打包过程需要 3-10 分钟，请耐心等待
echo.

:: 清理旧的打包文件
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist "YOLOv5检测系统.spec" del /q "YOLOv5检测系统.spec"

:: 打包
"%VENV_PYTHON%" -m PyInstaller ^
    --onefile ^
    --windowed ^
    --name "YOLOv5检测系统" ^
    --hidden-import=PyQt5 ^
    --hidden-import=cv2 ^
    --hidden-import=torch ^
    --hidden-import=torchvision ^
    --add-data "detector.py;." ^
    --add-data "main_window.py;." ^
    main.py

if errorlevel 1 (
    echo.
    echo [错误] 打包失败
    pause
    exit /b 1
)

echo.
echo ============================================================
echo    打包成功！
echo    程序位置: dist\YOLOv5检测系统.exe
echo.
echo    注意:
echo    1. 运行时需要将 models 文件夹放在 exe 同目录下
echo    2. 首次运行需要联网下载 YOLOv5 模型
echo    3. 可以将 exe 和 models 文件夹一起打包分发
echo ============================================================
echo.

:: 复制 models 目录到 dist
if not exist "dist\models" mkdir "dist\models"
xcopy /e /i /y models dist\models >nul 2>&1
if not exist "dist\output" mkdir "dist\output"

echo [信息] 已将 models 和 output 目录复制到 dist 文件夹
echo.
pause
