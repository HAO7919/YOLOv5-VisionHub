# -*- coding: utf-8 -*-
"""
YOLOv5 可视化检测系统 - 程序入口
双击运行此文件即可启动桌面程序
"""
import sys
import os
import math
import warnings
from pathlib import Path

# 抑制 PyTorch 的 FutureWarning 和其他无关警告，让输出更干净
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

# 确保工作目录为程序所在目录（开发时是脚本目录，打包后是exe所在目录）
if getattr(sys, 'frozen', False):
    os.chdir(Path(sys.executable).parent)
else:
    os.chdir(Path(__file__).parent)

# 启动依赖自检：关键模块缺失时明确提示，避免用户以为"检测不到东西"
def _check_dependencies():
    required = [
        ("torch", "PyTorch（深度学习框架）"),
        ("torchvision", "TorchVision（图像处理，检测必需）"),
        ("ultralytics", "Ultralytics（YOLO模型加载，检测必需）"),
        ("cv2", "OpenCV（视频/图片处理）"),
        ("sympy", "SymPy（数学计算，torchvision依赖）"),
        ("numpy", "NumPy（数值计算）"),
        ("PyQt5", "PyQt5（图形界面）"),
    ]
    missing = []
    for mod, desc in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(f"  - {mod}（{desc}）")
    if missing:
        msg = "以下关键模块缺失，软件可能无法正常检测：\n\n" + "\n".join(missing) + \
              "\n\n请重新安装软件，或检查是否被杀毒软件误删。"
        try:
            from PyQt5.QtWidgets import QMessageBox, QApplication
            app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(None, "依赖缺失", msg)
        except Exception:
            print(msg)
        return False
    return True

_deps_ok = _check_dependencies()

from PyQt5.QtWidgets import QApplication, QSplashScreen, QLabel, QVBoxLayout, QWidget, QProgressBar
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QColor, QPainter, QLinearGradient, QPen, QBrush


class ModernSplash(QSplashScreen):
    """现代化启动画面：渐变背景 + 进度条 + 加载动画"""

    def __init__(self):
        super().__init__()
        self.setFixedSize(480, 280)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._progress = 0
        self._status = "正在初始化..."
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self._timer.start(45)  # 约22fps，低配电脑也流畅

    def set_status(self, text):
        self._status = text
        self.update()

    def set_progress(self, value):
        self._progress = max(0, min(100, value))
        self.update()

    def _rotate(self):
        self._angle = (self._angle + 8) % 360
        self.update()

    def drawContents(self, painter):
        painter.setRenderHint(QPainter.Antialiasing)
        # 绘制圆角背景
        rect = self.rect().adjusted(1, 1, -1, -1)
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, QColor("#1a1d23"))
        gradient.setColorAt(1, QColor("#2d3139"))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 16, 16)

        # 顶部装饰线
        line_gradient = QLinearGradient(rect.left() + 40, 0, rect.right() - 40, 0)
        line_gradient.setColorAt(0, QColor("#4fc3f7"))
        line_gradient.setColorAt(0.5, QColor("#7c4dff"))
        line_gradient.setColorAt(1, QColor("#4fc3f7"))
        painter.setBrush(QBrush(line_gradient))
        painter.drawRoundedRect(rect.left() + 40, 70, rect.width() - 80, 3, 2, 2)

        # 标题
        title_font = QFont("Microsoft YaHei", 22, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(rect.adjusted(0, 85, 0, 0), Qt.AlignHCenter | Qt.AlignTop, "YOLOv5 检测系统")

        # 副标题
        sub_font = QFont("Microsoft YaHei", 11)
        painter.setFont(sub_font)
        painter.setPen(QColor("#888888"))
        painter.drawText(rect.adjusted(0, 125, 0, 0), Qt.AlignHCenter | Qt.AlignTop, "智能目标检测可视化工具")

        # 旋转加载圈
        center_x = rect.center().x()
        center_y = 185
        radius = 18
        for i in range(12):
            angle = (self._angle + i * 30) * math.pi / 180
            alpha = int(255 * (i / 12))
            r = radius * 1.2 * (1 - i / 12)
            painter.setPen(QPen(QColor(79, 195, 247, alpha), 3))
            painter.drawPoint(
                int(center_x + r * math.cos(angle)),
                int(center_y + r * math.sin(angle))
            )

        # 状态文字
        status_font = QFont("Microsoft YaHei", 10)
        painter.setFont(status_font)
        painter.setPen(QColor("#aaaaaa"))
        painter.drawText(rect.adjusted(0, 215, 0, 0), Qt.AlignHCenter | Qt.AlignTop, self._status)

        # 进度条背景
        bar_rect = rect.adjusted(60, 245, -60, -20)
        painter.setBrush(QColor("#3a3e47"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(bar_rect, 4, 4)

        # 进度条填充
        if self._progress > 0:
            fill_width = int(bar_rect.width() * self._progress / 100)
            fill_rect = bar_rect.adjusted(0, 0, fill_width - bar_rect.width(), 0)
            fill_gradient = QLinearGradient(fill_rect.topLeft(), fill_rect.topRight())
            fill_gradient.setColorAt(0, QColor("#4fc3f7"))
            fill_gradient.setColorAt(1, QColor("#7c4dff"))
            painter.setBrush(QBrush(fill_gradient))
            painter.drawRoundedRect(fill_rect, 4, 4)

        # 进度百分比
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.setPen(QColor("#888888"))
        painter.drawText(bar_rect, Qt.AlignRight | Qt.AlignVCenter, f" {self._progress}%")


def main():
    # 单实例锁：防止多次双击启动多个程序
    from PyQt5.QtCore import QSharedMemory
    shared_memory = QSharedMemory("YOLOv5_VisionHub_SingleInstance")
    if not shared_memory.create(1):
        # 已有实例在运行，直接退出
        print("[Main] 程序已在运行中，请勿重复启动")
        sys.exit(0)

    # 高 DPI 支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("YOLOv5 可视化检测系统")
    app.setOrganizationName("YOLOv5 Vision")
    # 设置应用图标（任务栏、窗口标题栏）
    icon_path = str(Path(__file__).parent / "assets" / "app.ico")
    if Path(icon_path).exists():
        from PyQt5.QtGui import QIcon
        app.setWindowIcon(QIcon(icon_path))

    # 加载界面语言（空=跟随系统）
    try:
        from config import get_config
        from i18n import set_language, detect_system_language
        cfg = get_config()
        lang = cfg.get("ui.language", "")
        if not lang:
            lang = detect_system_language()
        set_language(lang)
    except Exception as e:
        print(f"[Main] 语言加载失败，使用默认: {e}")

    # 显示现代化启动画面
    splash = ModernSplash()
    splash.show()
    # 启动画面居中显示在主屏幕
    screen = app.primaryScreen().geometry()
    splash.move(
        screen.center().x() - splash.width() // 2,
        screen.center().y() - splash.height() // 2
    )
    app.processEvents()

    # 模拟加载进度
    progress_value = [0]
    def update_progress():
        progress_value[0] += 2
        if progress_value[0] > 90:
            progress_value[0] = 90
        splash.set_progress(progress_value[0])
        if progress_value[0] < 30:
            splash.set_status("正在加载运行环境...")
        elif progress_value[0] < 60:
            splash.set_status("正在初始化检测引擎...")
        elif progress_value[0] < 90:
            splash.set_status("正在加载 AI 模型...")

    progress_timer = QTimer()
    progress_timer.timeout.connect(update_progress)
    progress_timer.start(80)

    # 延迟创建主窗口
    def create_window():
        from main_window import MainWindow
        splash.set_status("正在初始化界面...")
        window = MainWindow()
        window.show()
        splash.set_progress(95)
        splash.set_status("即将完成...")

        # 模型加载完成后关闭启动画面
        def close_splash():
            progress_timer.stop()
            splash.set_progress(100)
            splash.set_status("加载完成")
            QTimer.singleShot(300, lambda: splash.finish(window))

        # 检查模型是否加载完成
        def check_model():
            if hasattr(window, 'detector') and window.detector and window.detector.is_ready():
                close_splash()
            else:
                QTimer.singleShot(300, check_model)
        QTimer.singleShot(300, check_model)
        # 最多显示 60 秒后强制关闭
        QTimer.singleShot(60000, close_splash)

    QTimer.singleShot(200, create_window)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
