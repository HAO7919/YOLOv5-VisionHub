@echo off
chcp 65001 >nul
echo ========================================
echo   YOLO检测工具 - 一键打包脚本
echo ========================================
echo.

echo [1/5] 正在打包...
python -m PyInstaller yolov5_app.spec --clean --noconfirm
if errorlevel 1 (
    echo 打包失败！
    pause
    exit /b 1
)

echo.
echo [2/5] 复制官方模型和数据文件...
set DIST=dist\YOLO检测工具
set SRC=%~dp0

:: 复制官方模型（只复制yolov5开头的，排除best.pt/last.pt等私人模型）
if not exist "%DIST%\models" mkdir "%DIST%\models"
for %%f in ("%SRC%models\yolov5*.pt") do (
    copy /Y "%%f" "%DIST%\models\" >nul
)

:: 复制torch_cache和assets（从项目目录复制，确保完整）
if exist "%DIST%\torch_cache" rmdir /S /Q "%DIST%\torch_cache"
xcopy /E /I /Y /Q "%SRC%torch_cache" "%DIST%\torch_cache" >nul
if exist "%DIST%\assets" rmdir /S /Q "%DIST%\assets"
xcopy /E /I /Y /Q "%SRC%assets" "%DIST%\assets" >nul

echo.
echo [3/5] 排除私人模型和用户数据...
:: 删除私人模型（双重保险）
del /Q "%DIST%\models\best.pt" 2>nul
del /Q "%DIST%\models\last.pt" 2>nul
del /Q "%DIST%\_internal\models\best.pt" 2>nul
del /Q "%DIST%\_internal\models\last.pt" 2>nul

:: 删除日志文件
del /Q "%DIST%\*.log" 2>nul
del /Q "%DIST%\_internal\*.log" 2>nul

:: 清理output目录
if exist "%DIST%\output" rmdir /S /Q "%DIST%\output"
mkdir "%DIST%\output"

:: 删除config.json（首次运行自动创建默认配置）
del /Q "%DIST%\config.json" 2>nul

:: 删除测试图片
del /Q "%DIST%\test_*.jpg" 2>nul
del /Q "%DIST%\test_*.png" 2>nul

echo.
echo [4/5] 写入版本信息...
echo YOLO检测工具 v1.0 > "%DIST%\版本.txt"
echo 打包时间: %date% %time% >> "%DIST%\版本.txt"
echo 包含模型: >> "%DIST%\版本.txt"
dir /B "%DIST%\models\*.pt" >> "%DIST%\版本.txt"

echo.
echo [5/5] 完成！
echo.
echo 打包位置: %DIST%
echo 官方模型:
dir /B "%DIST%\models\*.pt"
echo.
echo 别人双击 "YOLO检测工具.exe" 就能用，不需要装Python
echo 把整个 "YOLO检测工具" 文件夹压缩发给别人即可
echo.
pause
