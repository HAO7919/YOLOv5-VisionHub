# -*- coding: utf-8 -*-
"""
YOLOv5 可视化检测系统 - PyQt5 主窗口
真正的桌面应用程序，独立窗口运行
"""
import sys
import os
import gc
import cv2
import time
import logging
import shutil
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# 国际化多语言支持
from i18n import tr, set_language, get_language, LANGUAGES, detect_system_language

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QStatusBar, QScrollArea, QStackedWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar, QFrame,
    QSizePolicy, QGroupBox, QGridLayout, QSplashScreen, QLineEdit, QTextEdit, QShortcut, QListWidget, QCheckBox
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QRect, QPoint, QPropertyAnimation,
    QEasingCurve, QEvent
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette, QIcon, QPainter, QPen, QBrush,
    QLinearGradient, QMovie, QKeySequence
)

from detector import get_detector, YOLOv5Detector
from config import get_config
from theme import get_theme_qss, is_system_dark, AnimatedDialog, NoticeDialog
from screen_capture import ScreenDetectThread, OverlayWindow


# 程序根目录（开发时是脚本目录，打包后是exe所在目录）
def _get_app_dir():
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(__file__).parent

APP_DIR = _get_app_dir()


# ==================== 主题常量 ====================
THEME_DARK = "dark"
THEME_LIGHT = "light"
THEME_SYSTEM = "system"


# ==================== 加载遮罩组件 ====================
class LoadingOverlay(QWidget):
    """半透明加载遮罩，带旋转动画和文字提示，支持主题适配"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._angle = 0
        self._text = tr("加载中...")
        self._theme = "light"  # 默认白色主题
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self.hide()

    def set_theme(self, theme):
        """设置主题颜色"""
        self._theme = theme
        self.update()

    def show_loading(self, text=tr("加载中...")):
        self._text = text
        self._timer.start(45)  # 约22fps，低配电脑也流畅
        self.show()
        self.raise_()

    def hide_loading(self):
        self._timer.stop()
        self.hide()

    def _rotate(self):
        self._angle = (self._angle + 12) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # 根据主题选择颜色
        if self._theme == "dark":
            bg_color = QColor(18, 20, 23, 210)
            ring_bg = QColor(45, 49, 57)
            ring_fg = QColor(79, 195, 247)
            text_color = QColor(232, 234, 237)
        else:  # light
            bg_color = QColor(255, 255, 255, 220)
            ring_bg = QColor(220, 224, 230)
            ring_fg = QColor(66, 133, 244)
            text_color = QColor(60, 64, 67)
        # 半透明背景
        painter.fillRect(self.rect(), bg_color)
        # 旋转圆环
        center_x = self.width() // 2
        center_y = self.height() // 2 - 40
        radius = 36
        rect = QRect(center_x - radius, center_y - radius, radius * 2, radius * 2)
        # 背景圆环
        painter.setPen(QPen(ring_bg, 4))
        painter.drawArc(rect, 0, 360 * 16)
        # 前景圆弧（旋转）
        painter.setPen(QPen(ring_fg, 4))
        painter.drawArc(rect, self._angle * 16, 270 * 16)
        # 多行文字
        painter.setPen(text_color)
        painter.setFont(QFont("Segoe UI", 13))
        lines = self._text.split("\n")
        line_height = 24
        start_y = center_y + radius + 20
        for i, line in enumerate(lines):
            text_rect = QRect(0, start_y + i * line_height, self.width(), line_height)
            painter.drawText(text_rect, Qt.AlignCenter, line)


# ==================== 后台检测线程 ====================
class DetectThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, detector, image_path, output_path):
        super().__init__()
        self.detector = detector
        self.image_path = image_path
        self.output_path = output_path

    def run(self):
        try:
            result = self.detector.detect_image_file(self.image_path, self.output_path)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class VideoDetectThread(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, detector, input_path, output_path, skip_frames=0):
        super().__init__()
        self.detector = detector
        self.input_path = input_path
        self.output_path = output_path
        self.skip_frames = skip_frames
        self._is_running = True

    def run(self):
        try:
            def cb(cur, total):
                if not self._is_running:
                    raise RuntimeError("用户中止了视频处理")
                self.progress.emit(cur, total)
            result = self.detector.detect_video_file(
                self.input_path, self.output_path, cb, skip_frames=self.skip_frames)
            if self._is_running:
                self.finished.emit(result)
            else:
                self.stopped.emit()
        except RuntimeError as e:
            if "中止" in str(e):
                self.stopped.emit()
            else:
                self.error.emit(str(e))
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        """请求停止视频处理（异步，线程会在处理完当前帧后退出）"""
        self._is_running = False


class ModelLoadThread(QThread):
    finished = pyqtSignal(bool, str, str)

    def __init__(self, detector, model_name):
        super().__init__()
        self.detector = detector
        self.model_name = model_name

    def run(self):
        success, error_msg = self.detector.load_model_by_name(self.model_name)
        self.finished.emit(success, self.model_name, error_msg)


# ==================== 摄像头检测线程（独立线程，不阻塞UI）====================
class CameraDetectThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, list, np.ndarray)  # 原始帧, 检测结果, 标注后帧
    fps_updated = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    thread_finished = pyqtSignal(str)  # 线程正常结束时通知，带原因

    def __init__(self, detector, camera_index=0, resolution=(640, 480), skip_frames=0):
        super().__init__()
        self.detector = detector
        self.camera_index = camera_index
        self.resolution = resolution
        self.skip_frames = skip_frames
        self._running = False
        self._last_detections = []
        self._last_annotated = None
        self._logger = logging.getLogger("yolov5_app")

    def stop(self):
        self._running = False

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  # 用DirectShow，Windows下更稳定
            if not cap.isOpened():
                # 退回到默认后端再试一次
                cap.release()
                cap = cv2.VideoCapture(self.camera_index)
                if not cap.isOpened():
                    self._logger.error("摄像头打开失败: index=%d", self.camera_index)
                    self.error_occurred.emit("无法打开摄像头，请检查摄像头是否已连接、是否被其他软件占用（如微信、QQ视频）")
                    return
            # 设置分辨率（部分摄像头不支持任意分辨率，设置后读取实际值）
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._logger.info("摄像头已开启: 期望%d x %d, 实际%d x %d",
                              self.resolution[0], self.resolution[1], actual_w, actual_h)
            self._running = True
            frame_count = 0
            fps_frames = 0
            fps_time = time.time()
            read_fail_count = 0
            detect_fail_count = 0
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    read_fail_count += 1
                    self._logger.warning("摄像头读取帧失败: 第%d次", read_fail_count)
                    if read_fail_count >= 10:
                        self.thread_finished.emit("摄像头读取连续失败，可能已断开")
                        break
                    time.sleep(0.05)
                    continue
                read_fail_count = 0
                frame_count += 1
                # 跳帧检测，降低CPU/GPU占用
                need_detect = (frame_count - 1) % (self.skip_frames + 1) == 0
                if need_detect or self._last_annotated is None:
                    try:
                        annotated, detections = self.detector.detect(frame)
                        self._last_detections = detections
                        self._last_annotated = annotated
                        detect_fail_count = 0
                    except Exception as det_err:
                        detect_fail_count += 1
                        self._logger.error("检测出错: %s (第%d次)", str(det_err), detect_fail_count)
                        if detect_fail_count >= 5:
                            self.error_occurred.emit(f"检测连续失败: {str(det_err)[:50]}")
                            break
                        # 检测失败时用原始帧继续显示，不中断
                        annotated = frame
                        detections = self._last_detections
                else:
                    # 非检测帧用上一帧标注结果（框轻微滞后但保证流畅）
                    annotated = self._last_annotated
                    detections = self._last_detections
                self.frame_ready.emit(frame, detections, annotated)
                fps_frames += 1
                now = time.time()
                if now - fps_time >= 1.0:
                    self.fps_updated.emit(int(fps_frames / (now - fps_time)))
                    fps_frames = 0
                    fps_time = now
            if cap:
                cap.release()
            self._logger.info("摄像头线程结束, 共处理%d帧", frame_count)
        except Exception as e:
            self._logger.error("摄像头线程异常: %s", str(e), exc_info=True)
            if cap:
                cap.release()
            self.error_occurred.emit(f"摄像头异常: {str(e)[:80]}")


# ==================== 工具函数 ====================
def cvimg_to_qpixmap(img: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ==================== 放大查看窗口 ====================
class ImageViewer(QMainWindow):
    """图片/视频帧放大查看窗口：滚轮缩放、拖拽移动、双击重置、ESC关闭"""
    def __init__(self, pixmap, title="放大查看", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)
        self.resize(900, 650)
        # 居中显示
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.center().x() - 450, screen.center().y() - 325)
        # 中心部件
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        # 图片标签
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: #1a1a1a;")
        self.label.setMinimumSize(1, 1)
        # 滚动区域（支持大图片）
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.label)
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setStyleSheet("QScrollArea { border: none; background-color: #1a1a1a; }")
        layout.addWidget(self.scroll)
        # 状态栏显示缩放比例
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("滚轮缩放 | 拖拽移动 | 双击重置 | ESC关闭")
        # 数据
        self.original_pixmap = pixmap
        self.scale_factor = 1.0
        self._drag_pos = None
        self._update_image()

    def _update_image(self):
        """按当前缩放比例更新图片"""
        if self.original_pixmap.isNull():
            return
        scaled = self.original_pixmap.scaled(
            self.original_pixmap.size() * self.scale_factor,
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled)
        self.label.resize(scaled.size())
        self.status.showMessage(f"缩放: {int(self.scale_factor * 100)}%  |  滚轮缩放 | 拖拽移动 | 双击重置 | ESC关闭")

    def wheelEvent(self, event):
        """滚轮缩放（以鼠标位置为中心）"""
        try:
            delta = event.angleDelta().y()
        except Exception:
            delta = event.delta()
        if delta > 0:
            factor = 1.15
        else:
            factor = 1 / 1.15
        new_scale = self.scale_factor * factor
        if 0.1 <= new_scale <= 20:
            self.scale_factor = new_scale
            self._update_image()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None:
            delta = event.pos() - self._drag_pos
            self.scroll.horizontalScrollBar().setValue(
                self.scroll.horizontalScrollBar().value() - delta.x())
            self.scroll.verticalScrollBar().setValue(
                self.scroll.verticalScrollBar().value() - delta.y())
            self._drag_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = None
            self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        """双击重置缩放"""
        self.scale_factor = 1.0
        self._update_image()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.detector = None
        self.current_image = None
        self.current_image_path = None  # 当前检测的图片路径（用于调参后重新检测）
        self.current_detections = []
        self.current_image_info = {}
        self.video_capture = None
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self._update_video_frame)
        self.video_output_path = None
        self.video_is_playing = False
        self.camera_capture = None
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self._update_camera_frame)
        self.camera_running = False
        self.camera_fps_frames = 0
        self.camera_fps_time = 0
        self.camera_frame_count = 0
        self.camera_last_detections = []
        self.camera_last_annotated = None
        self.skip_frames = 0  # 默认高质量，逐帧检测
        self.camera_resolution = (640, 480)
        self.detect_thread = None
        self.video_thread = None
        self.model_thread = None
        self.screen_thread = None
        self.overlay = None
        self.detect_history = []  # 检测历史记录
        self.current_theme = self.config.get("ui.theme", THEME_LIGHT)
        # 强制默认白色：如果配置中没有主题或主题无效，强制设为 light
        if self.current_theme not in [THEME_DARK, THEME_LIGHT, THEME_SYSTEM]:
            self.current_theme = THEME_LIGHT
        # 在 UI 初始化前就应用主题，确保所有控件创建时就是正确主题
        self.setStyleSheet(get_theme_qss(self.current_theme))
        # 启用拖拽支持
        self.setAcceptDrops(True)

        self._init_ui()
        self._apply_theme(self.current_theme)
        self._load_config()
        self._init_shortcuts()  # 初始化快捷键
        self._init_detector_async()

    def _init_ui(self):
        self.setWindowTitle(tr("YOLOv5 可视化检测系统"))
        # 设置窗口图标
        icon_path = APP_DIR / "assets" / "app.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.setMinimumSize(1100, 700)
        w = self.config.get("ui.window_width", 1400)
        h = self.config.get("ui.window_height", 850)
        self.resize(w, h)
        x = self.config.get("ui.window_x")
        y = self.config.get("ui.window_y")
        if x is not None and y is not None and x >= 0 and y >= 0:
            self.move(x, y)
        else:
            # 第一次启动：窗口居中
            self._center_on_screen()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 顶部标题栏
        title_bar = QFrame()
        title_bar.setObjectName("titleBar")
        title_bar.setFixedHeight(52)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(16, 0, 16, 0)
        title_label = QLabel(tr("YOLOv5 可视化检测系统"))
        title_label.setObjectName("titleLabel")
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        # 主题选择
        theme_label = QLabel("主题:")
        theme_label.setObjectName("themeLabel")
        title_layout.addWidget(theme_label)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems([tr("夜间"), tr("白天"), tr("跟随系统")])
        self.theme_combo.setFixedWidth(115)
        self.theme_combo.setObjectName("themeCombo")
        theme_map = {THEME_DARK: 0, THEME_LIGHT: 1, THEME_SYSTEM: 2}
        self.theme_combo.setCurrentIndex(theme_map.get(self.current_theme, 1))
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        title_layout.addWidget(self.theme_combo)
        title_layout.addSpacing(12)
        # 语言选择
        lang_label = QLabel("语言:")
        lang_label.setObjectName("themeLabel")
        title_layout.addWidget(lang_label)
        self.lang_combo = QComboBox()
        lang_names = [LANGUAGES[k] for k in LANGUAGES]
        lang_names.insert(0, tr("跟随系统"))
        self.lang_combo.addItems(lang_names)
        self.lang_combo.setFixedWidth(110)
        self.lang_combo.setObjectName("themeCombo")
        # 设置当前语言
        current_lang = get_language()
        lang_keys = list(LANGUAGES.keys())
        if current_lang in lang_keys:
            self.lang_combo.setCurrentIndex(lang_keys.index(current_lang) + 1)
        else:
            self.lang_combo.setCurrentIndex(0)
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        title_layout.addWidget(self.lang_combo)
        title_layout.addSpacing(16)

        self.status_model_label = QLabel(tr("模型: 未加载"))
        self.status_model_label.setObjectName("statusModelLabel")
        title_layout.addWidget(self.status_model_label)
        self.status_device_label = QLabel(tr("设备: -"))
        self.status_device_label.setObjectName("statusDeviceLabel")
        title_layout.addWidget(self.status_device_label)
        main_layout.addWidget(title_bar)

        # 主体三栏
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(12, 12, 12, 12)
        body_layout.setSpacing(12)
        left_panel = self._create_left_panel()
        body_layout.addWidget(left_panel, 0)
        center_panel = self._create_center_panel()
        body_layout.addWidget(center_panel, 1)
        # 给图片/视频显示标签安装事件过滤器（双击放大查看）
        for lbl in [self.original_label, self.result_label, self.video_label]:
            lbl.installEventFilter(self)
        right_panel = self._create_right_panel()
        body_layout.addWidget(right_panel, 0)
        main_layout.addWidget(body, 1)

        # 状态栏
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("就绪")
        self.setStatusBar(self.status_bar)

        # 加载遮罩
        self.loading_overlay = LoadingOverlay(self)

        # 统一给所有下拉框/数字框/滑块安装事件过滤器，鼠标悬停时滚轮不改变值（必须点击获得焦点后滚轮才生效）
        wheel_widgets = []
        wheel_widgets += self.findChildren(QComboBox)
        wheel_widgets += self.findChildren(QSpinBox)
        wheel_widgets += self.findChildren(QDoubleSpinBox)
        wheel_widgets += self.findChildren(QSlider)
        for w in wheel_widgets:
            w.installEventFilter(self)
        # 统一设置所有按钮最小高度，防止文字被砍
        for btn in self.findChildren(QPushButton):
            btn.setMinimumHeight(36)

    def _center_on_screen(self):
        """把窗口居中到主屏幕"""
        screen = self.screen().geometry() if hasattr(self, 'screen') else QApplication.primaryScreen().geometry()
        x = screen.center().x() - self.width() // 2
        y = screen.center().y() - self.height() // 2
        self.move(max(0, x), max(0, y))

    def changeEvent(self, event):
        """窗口状态变化：最大化还原时自动居中"""
        if event.type() == QEvent.WindowStateChange:
            old_state = event.oldState()
            new_state = self.windowState()
            # 从最大化/全屏还原为普通窗口时，窗口居中
            was_max = bool(old_state & Qt.WindowMaximized) or bool(old_state & Qt.WindowFullScreen)
            now_normal = not (new_state & Qt.WindowMaximized) and not (new_state & Qt.WindowFullScreen)
            if was_max and now_normal:
                QTimer.singleShot(0, self._center_on_screen)
        super().changeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.setGeometry(self.rect())
        # 防抖：窗口拖动过程中不立即缩放，停止200ms后再缩放，低配电脑不卡
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._do_resize_images)
        self._resize_timer.start(200)

    def _do_resize_images(self):
        """窗口大小稳定后重新缩放图片"""
        for label in [self.original_label, self.result_label, self.video_label, self.camera_label]:
            pixmap = label.pixmap()
            if pixmap and not pixmap.isNull():
                label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def eventFilter(self, obj, event):
        """拦截滚轮事件：下拉框/数字框/滑块完全禁用滚轮，只能用点击操作"""
        if event.type() == QEvent.Wheel:
            if isinstance(obj, (QComboBox, QSpinBox, QDoubleSpinBox, QSlider)):
                return True  # 直接忽略滚轮事件，只能点击操作
        # 双击图片/视频标签 → 放大查看
        if event.type() == QEvent.MouseButtonDblClick:
            if obj in [getattr(self, 'original_label', None),
                       getattr(self, 'result_label', None),
                       getattr(self, 'video_label', None)]:
                pixmap = obj.pixmap()
                if pixmap and not pixmap.isNull():
                    title_map = {
                        self.original_label: "原图 - 放大查看",
                        self.result_label: "检测结果 - 放大查看",
                        self.video_label: "视频画面 - 放大查看",
                    }
                    viewer = ImageViewer(pixmap, title_map.get(obj, "放大查看"), self)
                    viewer.show()
                    return True
        return super().eventFilter(obj, event)

    def _create_left_panel(self):
        panel = QWidget()
        panel.setFixedWidth(310)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        # 滚动条样式优化
        scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { width: 8px; background: transparent; margin: 0; }
            QScrollBar::handle:vertical { background: rgba(128,128,128,0.3); border-radius: 4px; min-height: 30px; }
            QScrollBar::handle:vertical:hover { background: rgba(128,128,128,0.5); }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        """)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(14, 14, 8, 14)
        content_layout.setSpacing(14)

        # 检测模式
        mode_group = QGroupBox(tr("检测模式"))
        mode_layout = QVBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([tr("图片检测"), tr("视频检测"), tr("摄像头实时检测"), tr("屏幕实时检测")])
        self.mode_combo.currentIndexChanged.connect(self._switch_mode)
        mode_layout.addWidget(self.mode_combo)
        mode_group.setLayout(mode_layout)
        content_layout.addWidget(mode_group)

        # 模型管理
        model_group = QGroupBox(tr("模型管理"))
        model_layout = QVBoxLayout()
        # 当前模型显示
        self.current_model_label = QLabel("当前模型: 未选择")
        self.current_model_label.setStyleSheet("color: #4fc3f7; font-weight: 600; font-size: 12px;")
        self.current_model_label.setWordWrap(True)
        model_layout.addWidget(self.current_model_label)
        # 模型下拉框
        self.model_combo = QComboBox()
        self.model_combo.addItem("（选择模型）")
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        model_layout.addWidget(self.model_combo)
        # 按钮行
        btn_row1 = QHBoxLayout()
        self.select_model_btn = QPushButton("选择模型文件")
        self.select_model_btn.setObjectName("primaryBtn")
        self.select_model_btn.clicked.connect(self._select_model_file)
        btn_row1.addWidget(self.select_model_btn)
        self.refresh_model_btn = QPushButton(tr("刷新列表"))
        self.refresh_model_btn.setMinimumWidth(80)
        self.refresh_model_btn.clicked.connect(self._refresh_model_list)
        btn_row1.addWidget(self.refresh_model_btn)
        model_layout.addLayout(btn_row1)
        # 切换模型按钮
        self.load_model_btn = QPushButton(tr("切换到选中模型"))
        self.load_model_btn.clicked.connect(self._switch_model)
        self.load_model_btn.setEnabled(False)
        model_layout.addWidget(self.load_model_btn)
        model_group.setLayout(model_layout)
        content_layout.addWidget(model_group)

        # 参数调节
        param_group = QGroupBox(tr("检测参数"))
        param_layout = QVBoxLayout()
        # 置信度
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel(tr("置信度")))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setFixedWidth(70)
        conf_row.addWidget(self.conf_spin)
        param_layout.addLayout(conf_row)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(lambda v: self._sync_spin_from_slider(self.conf_spin, v / 100))
        self.conf_spin.valueChanged.connect(lambda v: self._sync_slider_from_spin(self.conf_slider, int(v * 100)))
        param_layout.addWidget(self.conf_slider)
        # IOU
        iou_row = QHBoxLayout()
        iou_row.addWidget(QLabel("IOU 阈值"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.1, 0.95)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setFixedWidth(70)
        iou_row.addWidget(self.iou_spin)
        param_layout.addLayout(iou_row)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(10, 95)
        self.iou_slider.setValue(45)
        self.iou_slider.valueChanged.connect(lambda v: self._sync_spin_from_slider(self.iou_spin, v / 100))
        self.iou_spin.valueChanged.connect(lambda v: self._sync_slider_from_spin(self.iou_slider, int(v * 100)))
        param_layout.addWidget(self.iou_slider)
        # 线条粗细
        line_row = QHBoxLayout()
        line_row.addWidget(QLabel(tr("线条粗细")))
        self.line_spin = QSpinBox()
        self.line_spin.setRange(1, 6)
        self.line_spin.setValue(2)
        self.line_spin.setFixedWidth(70)
        line_row.addWidget(self.line_spin)
        param_layout.addLayout(line_row)
        # 应用参数
        self.apply_param_btn = QPushButton(tr("应用参数"))
        self.apply_param_btn.setObjectName("primaryBtn")
        self.apply_param_btn.clicked.connect(self._apply_params)
        param_layout.addWidget(self.apply_param_btn)
        param_group.setLayout(param_layout)
        content_layout.addWidget(param_group)

        # 视频与摄像头设置
        perf_group = QGroupBox(tr("视频与摄像头"))
        perf_layout = QVBoxLayout()
        # 视频处理速度（替代跳帧，更易懂）
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("视频处理速度"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["高质量 (逐帧)", "标准 (平衡)", "快速 (省时间)"])
        self.speed_combo.setCurrentIndex(0)  # 默认高质量，逐帧检测
        speed_row.addWidget(self.speed_combo)
        perf_layout.addLayout(speed_row)
        speed_hint = QLabel("高质量=每一帧都检测，最清晰；快速=跳过部分帧，处理更快")
        speed_hint.setStyleSheet("color: #5f6368; font-size: 11px;")
        speed_hint.setWordWrap(True)
        perf_layout.addWidget(speed_hint)
        # 摄像头分辨率
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel(tr("摄像头分辨率")))
        self.res_combo = QComboBox()
        self.res_combo.addItems([tr("640x480 (流畅)"), tr("800x600 (均衡)"), tr("1280x720 (清晰)"), tr("1920x1080 (高清)")])
        self.res_combo.setCurrentIndex(0)
        res_row.addWidget(self.res_combo)
        perf_layout.addLayout(res_row)
        # 自动应用
        self.speed_combo.currentIndexChanged.connect(self._on_perf_changed)
        self.res_combo.currentIndexChanged.connect(self._on_perf_changed)
        perf_group.setLayout(perf_layout)
        content_layout.addWidget(perf_group)
        # 缓存管理
        cache_group = QGroupBox(tr("缓存管理"))
        cache_layout = QVBoxLayout()
        self.cache_path_label = QLabel("缓存位置: 项目目录/torch_cache")
        self.cache_path_label.setWordWrap(True)
        self.cache_path_label.setStyleSheet("font-size: 11px; color: #888;")
        cache_layout.addWidget(self.cache_path_label)
        self.cache_size_label = QLabel("缓存大小: 计算中...")
        self.cache_size_label.setStyleSheet("font-size: 11px;")
        cache_layout.addWidget(self.cache_size_label)
        cache_btn_row = QHBoxLayout()
        self.change_cache_btn = QPushButton(tr("更改位置"))
        self.change_cache_btn.setMinimumWidth(80)
        self.change_cache_btn.clicked.connect(self._change_cache_dir)
        self.clean_cache_btn = QPushButton(tr("清理缓存"))
        self.clean_cache_btn.setMinimumWidth(80)
        self.clean_cache_btn.clicked.connect(self._clean_cache)
        cache_btn_row.addWidget(self.change_cache_btn)
        cache_btn_row.addWidget(self.clean_cache_btn)
        cache_layout.addLayout(cache_btn_row)
        cache_group.setLayout(cache_layout)
        content_layout.addWidget(cache_group)
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        return panel

    def _create_center_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.stack = QStackedWidget()
        self.image_page = self._create_image_page()
        self.stack.addWidget(self.image_page)
        self.video_page = self._create_video_page()
        self.stack.addWidget(self.video_page)
        self.camera_page = self._create_camera_page()
        self.stack.addWidget(self.camera_page)
        self.screen_page = self._create_screen_page()
        self.stack.addWidget(self.screen_page)
        layout.addWidget(self.stack)
        return panel

    def _create_image_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        btn_bar = QHBoxLayout()
        self.open_image_btn = QPushButton(tr("打开图片"))
        self.open_image_btn.setObjectName("primaryBtn")
        self.open_image_btn.clicked.connect(self._open_image)
        btn_bar.addWidget(self.open_image_btn)
        self.redetect_image_btn = QPushButton("重新检测")
        self.redetect_image_btn.clicked.connect(self._redetect_current_image)
        self.redetect_image_btn.setEnabled(False)
        btn_bar.addWidget(self.redetect_image_btn)
        self.save_image_btn = QPushButton(tr("保存结果图"))
        self.save_image_btn.clicked.connect(self._save_result_image)
        self.save_image_btn.setEnabled(False)
        btn_bar.addWidget(self.save_image_btn)
        self.export_json_btn = QPushButton(tr("导出JSON"))
        self.export_json_btn.clicked.connect(lambda: self._export_result("json"))
        self.export_json_btn.setEnabled(False)
        btn_bar.addWidget(self.export_json_btn)
        self.export_csv_btn = QPushButton(tr("导出CSV"))
        self.export_csv_btn.clicked.connect(lambda: self._export_result("csv"))
        self.export_csv_btn.setEnabled(False)
        btn_bar.addWidget(self.export_csv_btn)
        self.export_voc_btn = QPushButton("导出VOC")
        self.export_voc_btn.clicked.connect(lambda: self._export_result("voc"))
        self.export_voc_btn.setEnabled(False)
        btn_bar.addWidget(self.export_voc_btn)
        self.close_image_btn = QPushButton("关闭")
        self.close_image_btn.clicked.connect(self._close_image)
        self.close_image_btn.setEnabled(False)
        btn_bar.addWidget(self.close_image_btn)
        btn_bar.addStretch()
        layout.addLayout(btn_bar)
        # 图片对比
        display_row = QHBoxLayout()
        display_row.setSpacing(10)
        original_box = QVBoxLayout()
        original_title = QLabel("原图")
        original_title.setAlignment(Qt.AlignCenter)
        original_title.setObjectName("sectionTitle")
        original_box.addWidget(original_title)
        self.original_label = QLabel("请打开图片")
        self.original_label.setObjectName("imageDisplay")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumHeight(350)
        original_box.addWidget(self.original_label, 1)
        display_row.addLayout(original_box, 1)
        result_box = QVBoxLayout()
        result_title = QLabel(tr("检测结果"))
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setObjectName("sectionTitle")
        result_box.addWidget(result_title)
        self.result_label = QLabel("等待检测")
        self.result_label.setObjectName("imageDisplay")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumHeight(350)
        result_box.addWidget(self.result_label, 1)
        display_row.addLayout(result_box, 1)
        layout.addLayout(display_row, 1)
        return page

    def _create_video_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        btn_bar = QHBoxLayout()
        self.open_video_btn = QPushButton(tr("打开视频"))
        self.open_video_btn.setObjectName("primaryBtn")
        self.open_video_btn.clicked.connect(self._open_video)
        btn_bar.addWidget(self.open_video_btn)
        self.play_video_btn = QPushButton("播放")
        self.play_video_btn.clicked.connect(self._toggle_video_play)
        self.play_video_btn.setEnabled(False)
        btn_bar.addWidget(self.play_video_btn)
        self.save_video_btn = QPushButton("保存结果视频")
        self.save_video_btn.clicked.connect(self._save_result_video)
        self.save_video_btn.setEnabled(False)
        btn_bar.addWidget(self.save_video_btn)
        self.close_video_btn = QPushButton("关闭")
        self.close_video_btn.clicked.connect(self._close_video)
        self.close_video_btn.setEnabled(False)
        btn_bar.addWidget(self.close_video_btn)
        self.stop_video_btn = QPushButton("停止处理")
        self.stop_video_btn.clicked.connect(self._stop_video_processing)
        self.stop_video_btn.setVisible(False)
        self.stop_video_btn.setStyleSheet("background-color: #e53935; color: white;")
        btn_bar.addWidget(self.stop_video_btn)
        btn_bar.addStretch()
        layout.addLayout(btn_bar)
        self.video_label = QLabel("请打开视频文件")
        self.video_label.setObjectName("imageDisplay")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(350)
        layout.addWidget(self.video_label, 1)
        self.video_progress = QProgressBar()
        self.video_progress.setValue(0)
        self.video_progress.setFormat("%p%")
        layout.addWidget(self.video_progress)
        return page

    def _create_camera_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        btn_bar = QHBoxLayout()
        self.start_camera_btn = QPushButton(tr("开启摄像头"))
        self.start_camera_btn.setObjectName("primaryBtn")
        self.start_camera_btn.clicked.connect(self._start_camera)
        btn_bar.addWidget(self.start_camera_btn)
        self.stop_camera_btn = QPushButton(tr("停止"))
        self.stop_camera_btn.setObjectName("dangerBtn")
        self.stop_camera_btn.clicked.connect(self._stop_camera)
        self.stop_camera_btn.setEnabled(False)
        btn_bar.addWidget(self.stop_camera_btn)
        self.camera_capture_btn = QPushButton(tr("截图保存"))
        self.camera_capture_btn.clicked.connect(self._capture_camera_frame)
        self.camera_capture_btn.setEnabled(False)
        btn_bar.addWidget(self.camera_capture_btn)
        self.camera_fps_label = QLabel(tr("FPS: 0"))
        self.camera_fps_label.setStyleSheet("color: #4fc3f7; font-weight: 600;")
        btn_bar.addWidget(self.camera_fps_label)
        self.camera_count_label = QLabel(tr("检测目标: 0"))
        btn_bar.addWidget(self.camera_count_label)
        btn_bar.addStretch()
        layout.addLayout(btn_bar)
        self.camera_label = QLabel(tr('点击"开启摄像头"开始实时检测'))
        self.camera_label.setObjectName("imageDisplay")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumHeight(400)
        layout.addWidget(self.camera_label, 1)
        return page

    def _create_screen_page(self):
        """屏幕实时检测页面（简洁布局：常用设置直接显示，高级设置折叠）"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        # 按钮栏
        btn_bar = QHBoxLayout()
        self.start_screen_btn = QPushButton("开始屏幕检测")
        self.start_screen_btn.setObjectName("primaryBtn")
        self.start_screen_btn.clicked.connect(self._start_screen_detect)
        btn_bar.addWidget(self.start_screen_btn)
        self.stop_screen_btn = QPushButton(tr("停止"))
        self.stop_screen_btn.setObjectName("dangerBtn")
        self.stop_screen_btn.clicked.connect(self._stop_screen_detect)
        self.stop_screen_btn.setEnabled(False)
        btn_bar.addWidget(self.stop_screen_btn)
        self.screen_capture_btn = QPushButton(tr("截图保存"))
        self.screen_capture_btn.clicked.connect(self._capture_screen_frame)
        self.screen_capture_btn.setEnabled(False)
        btn_bar.addWidget(self.screen_capture_btn)
        self.screen_fps_label = QLabel(tr("FPS: 0"))
        self.screen_fps_label.setStyleSheet("color: #4fc3f7; font-weight: 600;")
        btn_bar.addWidget(self.screen_fps_label)
        self.screen_count_label = QLabel(tr("检测目标: 0"))
        btn_bar.addWidget(self.screen_count_label)
        self.screen_device_hint = QLabel("")
        self.screen_device_hint.setStyleSheet("color: #ff9800; font-size: 11px;")
        btn_bar.addWidget(self.screen_device_hint)
        btn_bar.addStretch()
        layout.addLayout(btn_bar)

        # 常用设置（一行排开，简洁）
        common_group = QGroupBox("检测设置")
        common_layout = QVBoxLayout()
        common_layout.setSpacing(8)
        # 性能预设
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("性能:"))
        self.screen_perf_combo = QComboBox()
        self.screen_perf_combo.addItems(["流畅 (CPU推荐)", tr("标准"), "高质量 (GPU推荐)", "自定义"])
        self.screen_perf_combo.setCurrentIndex(0)
        self.screen_perf_combo.currentIndexChanged.connect(self._on_screen_preset_changed)
        preset_row.addWidget(self.screen_perf_combo, 1)
        preset_row.addStretch()
        common_layout.addLayout(preset_row)
        # 标签开关
        check_row = QHBoxLayout()
        self.screen_show_label_check = QCheckBox(tr("显示标签"))
        self.screen_show_label_check.setChecked(True)
        self.screen_show_label_check.stateChanged.connect(self._on_screen_label_changed)
        check_row.addWidget(self.screen_show_label_check)
        self.screen_show_conf_check = QCheckBox("显示置信度")
        self.screen_show_conf_check.setChecked(True)
        self.screen_show_conf_check.stateChanged.connect(self._on_screen_conf_display_changed)
        check_row.addWidget(self.screen_show_conf_check)
        check_row.addStretch()
        common_layout.addLayout(check_row)
        common_group.setLayout(common_layout)
        layout.addWidget(common_group)

        # 高级设置（可折叠）
        self.screen_advanced_btn = QPushButton("▼ 高级设置")
        self.screen_advanced_btn.setFlat(True)
        self.screen_advanced_btn.setStyleSheet("text-align: left; color: #4fc3f7; padding: 4px;")
        self.screen_advanced_btn.clicked.connect(self._toggle_screen_advanced)
        layout.addWidget(self.screen_advanced_btn)

        self.screen_advanced_group = QGroupBox()
        adv_layout = QVBoxLayout()
        adv_layout.setSpacing(8)
        # 检测分辨率
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("检测分辨率:"))
        self.screen_res_combo = QComboBox()
        self.screen_res_combo.addItems(["320px (最快)", "480px (流畅)", "640px (标准)", "800px (清晰)"])
        self.screen_res_combo.setCurrentIndex(1)
        res_row.addWidget(self.screen_res_combo, 1)
        adv_layout.addLayout(res_row)
        # 最大帧率
        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("最大帧率:"))
        self.screen_fps_combo = QComboBox()
        self.screen_fps_combo.addItems(["10 FPS", "15 FPS", "20 FPS", "30 FPS", "60 FPS"])
        self.screen_fps_combo.setCurrentIndex(3)
        fps_row.addWidget(self.screen_fps_combo, 1)
        adv_layout.addLayout(fps_row)
        # 框线透明度
        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("框线透明度:"))
        self.screen_alpha_slider = QSlider(Qt.Horizontal)
        self.screen_alpha_slider.setRange(30, 255)
        self.screen_alpha_slider.setValue(220)
        self.screen_alpha_slider.valueChanged.connect(self._on_screen_alpha_changed)
        alpha_row.addWidget(self.screen_alpha_slider, 1)
        self.screen_alpha_label = QLabel("220")
        self.screen_alpha_label.setFixedWidth(35)
        alpha_row.addWidget(self.screen_alpha_label)
        adv_layout.addLayout(alpha_row)
        self.screen_advanced_group.setLayout(adv_layout)
        self.screen_advanced_group.setVisible(False)  # 默认折叠
        layout.addWidget(self.screen_advanced_group)

        # 简短提示
        tip_label = QLabel("提示：检测框叠加在屏幕上，鼠标可穿透正常操作；按 F6 快速开始/停止。CPU 电脑建议流畅模式。")
        tip_label.setStyleSheet("color: #888; font-size: 11px; padding: 4px;")
        tip_label.setWordWrap(True)
        layout.addWidget(tip_label)
        layout.addStretch()
        return page

    def _toggle_screen_advanced(self):
        """折叠/展开屏幕检测高级设置"""
        visible = not self.screen_advanced_group.isVisible()
        self.screen_advanced_group.setVisible(visible)
        self.screen_advanced_btn.setText("▲ 高级设置" if visible else "▼ 高级设置")

    def _create_right_panel(self):
        panel = QWidget()
        panel.setFixedWidth(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)
        # 概览
        overview_group = QGroupBox("检测概览")
        overview_layout = QHBoxLayout()
        det_box = QVBoxLayout()
        self.total_count_label = QLabel("0")
        self.total_count_label.setObjectName("statNumber")
        self.total_count_label.setAlignment(Qt.AlignCenter)
        det_box.addWidget(self.total_count_label)
        det_label = QLabel("检测目标数")
        det_label.setObjectName("statLabel")
        det_label.setAlignment(Qt.AlignCenter)
        det_box.addWidget(det_label)
        overview_layout.addLayout(det_box)
        cls_box = QVBoxLayout()
        self.class_count_label = QLabel("0")
        self.class_count_label.setObjectName("statNumber")
        self.class_count_label.setAlignment(Qt.AlignCenter)
        cls_box.addWidget(self.class_count_label)
        cls_label = QLabel("涉及类别数")
        cls_label.setObjectName("statLabel")
        cls_label.setAlignment(Qt.AlignCenter)
        cls_box.addWidget(cls_label)
        overview_layout.addLayout(cls_box)
        overview_group.setLayout(overview_layout)
        content_layout.addWidget(overview_group)
        # 类别统计
        stats_group = QGroupBox("类别统计")
        stats_layout = QVBoxLayout()
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        content_layout.addWidget(stats_group)
        # 检测明细
        detail_group = QGroupBox("检测明细")
        detail_layout = QVBoxLayout()
        self.detail_table = QTableWidget()
        self.detail_table.setColumnCount(5)
        self.detail_table.setHorizontalHeaderLabels(["序号", tr("类别"), tr("置信度"), "位置", "尺寸"])
        self.detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.detail_table.verticalHeader().setVisible(False)
        self.detail_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.detail_table.setSelectionBehavior(QTableWidget.SelectRows)
        detail_layout.addWidget(self.detail_table)
        detail_group.setLayout(detail_layout)
        content_layout.addWidget(detail_group, 1)
        # 最近检测历史
        history_group = QGroupBox(tr("最近检测"))
        history_layout = QVBoxLayout()
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(120)
        self.history_list.setStyleSheet("QListWidget { font-size: 11px; }")
        self.history_list.itemDoubleClicked.connect(self._on_history_double_clicked)
        history_layout.addWidget(self.history_list)
        self.open_output_btn = QPushButton("打开结果保存文件夹")
        self.open_output_btn.setMinimumHeight(32)
        self.open_output_btn.clicked.connect(self._open_output_folder)
        history_layout.addWidget(self.open_output_btn)
        history_group.setLayout(history_layout)
        content_layout.addWidget(history_group)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        return panel

    # ==================== 配置加载/保存 ====================
    def _load_config(self):
        """从配置文件加载用户设置"""
        try:
            self.conf_spin.setValue(self.config.get("model.conf_thres", 0.25))
            self.iou_spin.setValue(self.config.get("model.iou_thres", 0.45))
            self.line_spin.setValue(self.config.get("model.line_thickness", 2))
            mode = self.config.get("ui.current_mode", 0)
            if 0 <= mode <= 3:
                self.mode_combo.setCurrentIndex(mode)
            # 加载性能设置
            speed = self.config.get("ui.video_speed", 0)
            if 0 <= speed <= 2:
                self.speed_combo.setCurrentIndex(speed)
            res = self.config.get("ui.camera_res", 0)
            if 0 <= res <= 2:
                self.res_combo.setCurrentIndex(res)
            # 加载屏幕检测设置
            screen_perf = self.config.get("ui.screen_perf", 0)
            if 0 <= screen_perf <= 3:
                self.screen_perf_combo.setCurrentIndex(screen_perf)
            screen_label = self.config.get("ui.screen_show_label", True)
            self.screen_show_label_check.setChecked(screen_label)
            screen_conf = self.config.get("ui.screen_show_conf", True)
            self.screen_show_conf_check.setChecked(screen_conf)
            screen_alpha = self.config.get("ui.screen_alpha", 220)
            if 30 <= screen_alpha <= 255:
                self.screen_alpha_slider.setValue(screen_alpha)
            screen_res = self.config.get("ui.screen_res", 1)
            if 0 <= screen_res <= 3:
                self.screen_res_combo.setCurrentIndex(screen_res)
            screen_fps = self.config.get("ui.screen_fps", 3)
            if 0 <= screen_fps <= 4:
                self.screen_fps_combo.setCurrentIndex(screen_fps)
            # 应用性能设置
            self._on_perf_changed()
            # 恢复历史记录
            saved_history = self.config.get("history.records", [])
            if saved_history and isinstance(saved_history, list):
                self.detect_history = saved_history[:20]
                self._update_history_panel()
            # 恢复窗口大小和位置（用户上次调整后的状态）
            win_w = self.config.get("ui.window_width", 0)
            win_h = self.config.get("ui.window_height", 0)
            win_x = self.config.get("ui.window_x", -1)
            win_y = self.config.get("ui.window_y", -1)
            if win_w > 800 and win_h > 600:
                self.resize(win_w, win_h)
            if win_x >= 0 and win_y >= 0:
                self.move(win_x, win_y)
        except Exception as e:
            print(f"[Config] 加载配置失败: {e}")

    def _apply_theme(self, theme):
        """应用主题"""
        self.current_theme = theme
        qss = get_theme_qss(theme)
        self.setStyleSheet(qss)
        # 同步更新加载遮罩主题
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.set_theme(theme)
        # 更新主题下拉框选中状态（避免循环触发）
        theme_map = {THEME_DARK: 0, THEME_LIGHT: 1, THEME_SYSTEM: 2}
        if self.theme_combo.currentIndex() != theme_map.get(theme, 1):
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentIndex(theme_map.get(theme, 1))
            self.theme_combo.blockSignals(False)
        # 保存到配置
        self.config.set("ui.theme", theme, auto_save=True)

    def _on_theme_changed(self, index):
        """主题切换事件"""
        themes = [THEME_DARK, THEME_LIGHT, THEME_SYSTEM]
        theme = themes[index] if index < len(themes) else THEME_DARK
        self._apply_theme(theme)
        # 保存主题设置
        self.config.set("ui.theme", theme)
        # 如果是跟随系统，显示提示
        if theme == THEME_SYSTEM:
            system_dark = is_system_dark()
            actual = tr("夜间") if system_dark else tr("白天")
            self.status_bar.showMessage(f"跟随系统主题（当前: {actual}）")

    def _on_language_changed(self, index):
        """语言切换事件"""
        lang_keys = list(LANGUAGES.keys())
        if index == 0:
            # 跟随系统
            lang = ""
            actual = detect_system_language()
            set_language(actual)
            lang_name = f"跟随系统（当前: {LANGUAGES.get(actual, 'English')}）"
        else:
            lang = lang_keys[index - 1]
            set_language(lang)
            lang_name = LANGUAGES[lang]
        # 保存到配置
        self.config.set("ui.language", lang, auto_save=True)
        # 弹出重启确认对话框
        buttons = [
            ("稍后重启", False, "normal"),
            ("立即重启", True, "primary"),
        ]
        dlg = AnimatedDialog(
            self, "语言已切换",
            f"语言已切换为：{lang_name}\n设置已保存，需要重启软件才能完全生效。\n是否立即重启？",
            AnimatedDialog.QUESTION, buttons
        )
        dlg.exec_()
        if dlg.result_button:
            # 立即重启
            self._restart_app()

    def _restart_app(self):
        """重启当前程序"""
        import sys
        import subprocess
        # 先保存配置
        try:
            self._save_config()
        except Exception:
            pass
        # 停止所有线程
        try:
            if hasattr(self, 'camera_thread') and self.camera_thread:
                self.camera_thread.stop()
            if hasattr(self, 'screen_thread') and self.screen_thread and self.screen_thread.running:
                self._stop_screen_detect()
        except Exception:
            pass
        # 启动新实例
        exe_path = sys.executable if getattr(sys, 'frozen', False) else sys.argv[0]
        try:
            if getattr(sys, 'frozen', False):
                subprocess.Popen([exe_path])
            else:
                subprocess.Popen([sys.executable, exe_path])
        except Exception as e:
            self.status_bar.showMessage(f"重启失败: {e}")
            return
        # 关闭当前实例
        QApplication.quit()

    def _save_config(self):
        """保存用户设置到配置文件"""
        try:
            self.config.set("model.conf_thres", self.conf_spin.value(), auto_save=False)
            self.config.set("model.iou_thres", self.iou_spin.value(), auto_save=False)
            self.config.set("model.line_thickness", self.line_spin.value(), auto_save=False)
            self.config.set("ui.current_mode", self.mode_combo.currentIndex(), auto_save=False)
            self.config.set("ui.theme", self.current_theme, auto_save=False)
            # 语言设置
            if hasattr(self, 'lang_combo'):
                lang_idx = self.lang_combo.currentIndex()
                lang_keys = list(LANGUAGES.keys())
                if lang_idx == 0:
                    self.config.set("ui.language", "", auto_save=False)
                else:
                    self.config.set("ui.language", lang_keys[lang_idx - 1], auto_save=False)
            self.config.set("ui.window_width", self.width(), auto_save=False)
            self.config.set("ui.window_height", self.height(), auto_save=False)
            self.config.set("ui.window_x", self.x(), auto_save=False)
            self.config.set("ui.window_y", self.y(), auto_save=False)
            self.config.set("ui.video_speed", self.speed_combo.currentIndex(), auto_save=False)
            self.config.set("ui.camera_res", self.res_combo.currentIndex(), auto_save=False)
            self.config.set("ui.screen_perf", self.screen_perf_combo.currentIndex(), auto_save=False)
            self.config.set("ui.screen_show_label", self.screen_show_label_check.isChecked(), auto_save=False)
            self.config.set("ui.screen_show_conf", self.screen_show_conf_check.isChecked(), auto_save=False)
            self.config.set("ui.screen_alpha", self.screen_alpha_slider.value(), auto_save=False)
            self.config.set("ui.screen_res", self.screen_res_combo.currentIndex(), auto_save=False)
            self.config.set("ui.screen_fps", self.screen_fps_combo.currentIndex(), auto_save=False)
            if self.detector and self.detector.is_ready():
                self.config.set("model.current_model", self.detector.model_name, auto_save=False)
            self.config.set("first_run", False, auto_save=False)
            # 持久化历史记录（最多保留20条）
            if hasattr(self, 'detect_history') and self.detect_history:
                self.config.set("history.records", self.detect_history[:20], auto_save=False)
            self.config.save()
        except Exception as e:
            print(f"[Config] 保存配置失败: {e}")

    # ==================== 模型管理 ====================
    def _init_detector_async(self):
        """异步初始化检测器（不阻塞界面，状态栏显示进度）"""
        self.status_bar.showMessage("正在后台加载模型，请稍候...（窗口可正常操作）")
        self.status_model_label.setText("模型: 加载中...")
        # 从配置读取上次使用的模型
        saved_model = self.config.get("model.current_model", "")
        self._init_thread = _ModelInitThread(model_name=saved_model)
        self._init_thread.finished.connect(self._on_model_init_finished)
        self._init_thread.start()
        # 30秒后如果还在加载，提示用户首次加载需要时间
        self._load_timeout_timer = QTimer(self)
        self._load_timeout_timer.setSingleShot(True)
        self._load_timeout_timer.timeout.connect(self._on_load_timeout)
        self._load_timeout_timer.start(30000)

    def _on_load_timeout(self):
        """加载超时提示"""
        if hasattr(self, '_init_thread') and self._init_thread.isRunning():
            self.status_bar.showMessage("首次加载需要编译 CUDA 内核，可能需要 1-3 分钟，请耐心等待...")
            self.status_model_label.setText("模型: 首次加载中...")

    def _on_model_init_finished(self, detector, success):
        # 停止超时计时器
        if hasattr(self, '_load_timeout_timer'):
            self._load_timeout_timer.stop()
        if success and detector:
            self.detector = detector
            self.status_model_label.setText(f"模型: {detector.model_name}")
            self.status_device_label.setText(f"设备: {detector.device}")
            self.status_bar.showMessage("模型加载完成，就绪")
            self.current_model_label.setText(f"当前模型: {detector.model_name}")
            # 更新缓存信息和屏幕设备提示
            self._update_cache_info()
            self._update_screen_device_hint()
            # 延迟执行耗时操作，避免阻塞主线程导致界面卡死
            QTimer.singleShot(100, self._after_model_loaded)
        else:
            self.status_model_label.setText("模型: 加载失败")
            self.status_bar.showMessage("保存的模型加载失败，正在回退默认模型...")
            # 自动回退到默认模型yolov5s.pt，避免用户每次启动都看到"加载失败"
            saved_model = self.config.get("model.current_model", "")
            if saved_model and saved_model != "yolov5s.pt":
                self.config.set("model.current_model", "yolov5s.pt")
                QTimer.singleShot(500, lambda: self._fallback_to_default_model())
            else:
                QTimer.singleShot(100, self._show_model_load_error)

    def _fallback_to_default_model(self):
        """回退到默认模型"""
        try:
            self.detector = YOLOv5Detector("yolov5s.pt")
            self.status_model_label.setText(f"模型: {self.detector.model_name}")
            self.status_device_label.setText(f"设备: {self.detector.device}")
            self.status_bar.showMessage("已回退到默认模型 yolov5s.pt")
            self.current_model_label.setText(f"当前模型: {self.detector.model_name}")
            self._update_cache_info()
            self._update_screen_device_hint()
            QTimer.singleShot(100, self._after_model_loaded)
        except Exception as e:
            self.status_bar.showMessage("默认模型也加载失败")
            self._show_model_load_error()

    def _after_model_loaded(self):
        """模型加载完成后的后续操作（延迟执行，避免阻塞）"""
        try:
            self._refresh_model_list()
        except Exception as e:
            print(f"[UI] 刷新模型列表失败: {e}")
        try:
            self._apply_params(silent=True)
        except Exception as e:
            print(f"[UI] 应用参数失败: {e}")
        # GPU 性能提示
        try:
            is_gpu = "cuda" in str(self.detector.device).lower() or self.detector.device == "0"
            has_nvidia = self._check_nvidia_gpu()
            if not is_gpu and has_nvidia:
                QTimer.singleShot(800, self._show_gpu_tip)
        except Exception as e:
            print(f"[UI] GPU检查失败: {e}")
        # 首次运行引导
        try:
            if self.config.get("first_run", True):
                QTimer.singleShot(1500, self._show_first_run_tip)
        except Exception as e:
            print(f"[UI] 首次运行引导失败: {e}")

    def _check_nvidia_gpu(self):
        """检查是否有 NVIDIA 显卡"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0 and result.stdout.strip()
        except Exception:
            return False

    def _show_gpu_tip(self):
        """显示 GPU 加速提示"""
        AnimatedDialog.information(self, "性能提示",
            "检测到你的电脑有 NVIDIA 显卡，但当前使用的是 CPU 版 PyTorch。\n\n"
            "安装 GPU 版 PyTorch 可提升检测速度 5-20 倍！\n\n"
            "启用方法：\n"
            "1. 关闭本软件\n"
            "2. 删除项目目录下的 venv 文件夹\n"
            "3. 重新双击 run.bat 启动\n"
            "4. 启动器会自动检测并尝试安装 GPU 版 PyTorch\n\n"
            "如果安装失败，会自动回退到 CPU 版，不影响使用。")

    def _show_first_run_tip(self):
        """首次运行公告：带倒计时，3秒后才能关闭，只弹一次"""
        message = (
            "这个工具是用来学习和验证 YOLOv5 目标检测的。\n\n"
            "你可以用它检测图片、视频、摄像头画面，甚至实时检测屏幕内容。"
            "训练好自己的模型后，点左侧「选择模型文件」导入就能用。\n\n"
            "用的时候请注意：\n"
            "• 不要偷拍偷录，不要侵犯他人隐私\n"
            "• 未经允许不要采集人脸、车牌等个人信息\n"
            "• 检测数据请妥善保管，遵守相关法律法规\n\n"
            "祝你用得顺手。"
        )
        dlg = NoticeDialog(self, title="使用须知", message=message, countdown=3)
        dlg.exec_()
        # 用户确认后写入配置，下次不再弹
        self.config.set("first_run", False)
        self.config.save()

    def _show_model_load_error(self):
        """模型加载失败提示"""
        message = (
            "模型没有加载成功。\n\n"
            "可能的原因：\n"
            "1. models 文件夹里没有 .pt 模型文件\n"
            "2. 模型文件损坏或版本不兼容\n"
            "3. 缺少 ultralytics 依赖（用 run.bat 启动会自动装）\n\n"
            "点\"选择模型\"可以手动加载你自己的模型文件（支持 v5/v8/v9/v10/v11）。"
        )
        buttons = [
            ("退出程序", False, "normal"),
            ("选择模型", True, "primary"),
        ]
        dlg = AnimatedDialog(self, "模型加载失败", message, AnimatedDialog.ERROR, buttons)
        dlg.exec_()
        if dlg.result_button:
            self._select_model_file()
        else:
            self.close()

    def _select_model_file(self):
        """让用户选择模型文件"""
        # 防止重复点击
        if hasattr(self, '_load_model_thread') and self._load_model_thread and self._load_model_thread.isRunning():
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件（支持 YOLOv5/v8/v9/v10/v11）", "",
            "YOLO 模型 (*.pt);;所有文件 (*)"
        )
        if not file_path:
            return
        try:
            # 复制到 models 目录
            models_dir = APP_DIR / "models"
            models_dir.mkdir(exist_ok=True)
            file_name = Path(file_path).name
            dest_path = models_dir / file_name
            if Path(file_path) != dest_path:
                shutil.copy2(file_path, str(dest_path))
            self.select_model_btn.setEnabled(False)
            self.loading_overlay.show_loading(f"正在加载模型: {file_name}")
            self.status_bar.showMessage(f"正在加载模型: {file_name}")
            # 用线程加载（直接在现有detector上加载，不重复创建）
            self._load_model_thread = _LoadModelFileThread(self.detector, str(dest_path))
            self._load_model_thread.finished.connect(self._on_model_file_loaded)
            self._load_model_thread.start()
        except Exception as e:
            self.select_model_btn.setEnabled(True)
            self.loading_overlay.hide_loading()
            AnimatedDialog.critical(self, "错误", f"模型文件复制失败: {str(e)}")

    def _on_model_file_loaded(self, success, model_name, error_msg):
        self.loading_overlay.hide_loading()
        self.select_model_btn.setEnabled(True)
        if success:
            self.status_model_label.setText(f"模型: {model_name}")
            self.status_device_label.setText(f"设备: {self.detector.device}")
            self.current_model_label.setText(f"当前模型: {model_name}")
            self.status_bar.showMessage(f"模型加载成功: {model_name}, {len(self.detector.classes)}个类别, 设备: {self.detector.device}")
            # 保存当前模型到配置，退出重进后恢复
            self.config.set("model.current_model", model_name)
            self._refresh_model_list()
            self._apply_params(silent=True)
            self._clear_results()
            # 成功不弹窗，状态栏已显示详细信息
        else:
            AnimatedDialog.critical(self, "加载失败",
                f"模型加载失败: {model_name}\n\n{error_msg}\n\n"
                "常见原因：\n"
                "1. 模型不是 YOLOv5/v8/v9/v10/v11 的 .pt 文件\n"
                "2. 模型文件已损坏或没导出完整\n"
                "3. 缺少 ultralytics 依赖（用 run.bat 启动会自动安装）")

    def _refresh_model_list(self):
        if not self.detector:
            return
        # 记录当前下拉框选中的模型（用户可能选了但还没加载）
        current_data = self.model_combo.currentData()
        self.model_combo.clear()
        self.model_combo.addItem("（选择模型）")
        models = self.detector.list_models()
        has_current = False
        restored_index = 0
        for m in models:
            label = m["name"]
            if m["size_mb"] > 0:
                label += f" ({m['size_mb']}MB)"
            self.model_combo.addItem(label, m["name"])
            if m["is_current"]:
                self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
                has_current = True
            # 如果用户之前选了这个模型但没加载，保持选中
            elif current_data == m["name"] and not has_current:
                restored_index = self.model_combo.count() - 1
        # 如果没有当前加载的模型，恢复用户之前的选择
        if not has_current and restored_index > 0:
            self.model_combo.setCurrentIndex(restored_index)
        self.load_model_btn.setEnabled(self.model_combo.count() > 1 and not has_current)
        # 刷新反馈：告诉用户扫到几个模型
        model_count = self.model_combo.count() - 1
        self.status_bar.showMessage(f"已扫描 models 文件夹，找到 {model_count} 个模型")
        self._flash_button(self.refresh_model_btn, "已刷新")

    def _on_model_combo_changed(self, index):
        """模型下拉框选择变化时，启用切换按钮"""
        if index <= 0:
            self.load_model_btn.setEnabled(False)
            return
        selected_data = self.model_combo.currentData()
        # 如果选择的是当前已加载的模型，不启用切换按钮
        if self.detector and self.detector.is_ready():
            current_name = self.detector.model_name
            if selected_data == current_name:
                self.load_model_btn.setEnabled(False)
                return
        self.load_model_btn.setEnabled(True)

    def _switch_model(self):
        if not self.detector or not self.detector.is_ready():
            AnimatedDialog.warning(self, "提示", "模型尚未加载完成")
            return
        # 防止重复点击：模型加载中不允许再次切换
        if hasattr(self, '_model_loading') and self._model_loading:
            AnimatedDialog.information(self, "提示", "模型正在加载中，请稍候...")
            return
        model_name = self.model_combo.currentData()
        if not model_name or model_name == "":
            AnimatedDialog.information(self, "提示", "请先从下拉框选择一个模型")
            return
        current = self.detector.model_name
        if current == model_name:
            AnimatedDialog.information(self, "提示", "当前已经是该模型")
            return
        reply = AnimatedDialog.question(self, "确认切换",
            f"确定要切换到模型 \"{model_name}\" 吗？\n切换需要几秒钟。")
        if not reply:
            return
        self._model_loading = True
        self.loading_overlay.show_loading(f"正在切换模型: {model_name}")
        self.load_model_btn.setEnabled(False)
        self.select_model_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.model_thread = ModelLoadThread(self.detector, model_name)
        self.model_thread.finished.connect(self._on_model_switch_finished)
        self.model_thread.start()

    def _on_model_switch_finished(self, success, model_name, error_msg=""):
        self._model_loading = False
        self.loading_overlay.hide_loading()
        self.load_model_btn.setEnabled(True)
        self.select_model_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        if success:
            self.status_model_label.setText(f"模型: {self.detector.model_name}")
            self.current_model_label.setText(f"当前模型: {self.detector.model_name}")
            self.status_bar.showMessage(f"模型已切换: {self.detector.model_name}, {len(self.detector.classes)}个类别")
            # 保存当前模型到配置，退出重进后恢复
            self.config.set("model.current_model", self.detector.model_name)
            self._refresh_model_list()
            self._clear_results()
            self._flash_button(self.load_model_btn, "已切换")
            # 成功不弹窗，状态栏已显示
        else:
            self.status_bar.showMessage("模型切换失败")
            AnimatedDialog.critical(self, "切换失败",
                f"模型切换失败: {model_name}\n\n{error_msg}")

    # ==================== 参数 ====================
    def _sync_spin_from_slider(self, spin, value):
        """从 slider 同步到 spin，阻塞信号避免循环"""
        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)

    def _sync_slider_from_spin(self, slider, value):
        """从 spin 同步到 slider，阻塞信号避免循环"""
        slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(False)

    def _apply_params(self, silent=False):
        if not self.detector:
            if not silent:
                AnimatedDialog.warning(self, "提示", "模型尚未加载完成")
            return
        try:
            self.detector.set_params(
                conf_thres=self.conf_spin.value(),
                iou_thres=self.iou_spin.value(),
                line_thickness=self.line_spin.value(),
            )
            self.status_bar.showMessage(f"参数已应用: 置信度={self.conf_spin.value():.2f}, IOU={self.iou_spin.value():.2f}, 线宽={self.line_spin.value()}")
            # 按钮视觉反馈：短暂显示"已应用"
            self._flash_button(self.apply_param_btn, "已应用")
            # 保存参数到配置
            self.config.set("model.conf_thres", self.conf_spin.value(), auto_save=False)
            self.config.set("model.iou_thres", self.iou_spin.value(), auto_save=False)
            self.config.set("model.line_thickness", self.line_spin.value(), auto_save=False)
            self.config.save()
            # 非静默调用时：若当前在图片模式且已打开图片，自动用新参数重新检测，调参立即看效果
            # 加300ms防抖：用户连续调参时只在停止后检测一次，避免多次重复检测卡顿
            if not silent and self.mode_combo.currentIndex() == 0 and self.current_image_path:
                if Path(self.current_image_path).exists():
                    if not hasattr(self, '_redetect_timer'):
                        self._redetect_timer = QTimer(self)
                        self._redetect_timer.setSingleShot(True)
                        self._redetect_timer.timeout.connect(self._redetect_current_image)
                    self._redetect_timer.start(300)
        except Exception as e:
            if not silent:
                AnimatedDialog.critical(self, "错误", f"参数应用失败: {str(e)}")

    def _flash_button(self, button, text="已完成", duration=1200):
        """按钮短暂显示反馈文字，然后恢复（防重复：正在闪烁时不重复触发）"""
        if hasattr(button, '_flashing') and button._flashing:
            return
        button._flashing = True
        original_text = button.text()
        original_style = button.styleSheet()
        button.setText(text)
        button.setStyleSheet(original_style + " background-color: #4caf50; color: white; border-color: #4caf50;")
        def restore():
            try:
                self._restore_button(button, original_text, original_style)
            finally:
                button._flashing = False
        QTimer.singleShot(duration, restore)

    def _restore_button(self, button, original_text, original_style):
        """恢复按钮原始状态"""
        try:
            button.setText(original_text)
            button.setStyleSheet(original_style)
        except Exception:
            pass

    def _on_perf_changed(self):
        """性能设置变化时更新参数"""
        # 处理速度映射：高质量=0跳帧，标准=1跳帧，快速=2跳帧
        speed_map = [0, 1, 2]
        speed_names = [tr("逐帧检测"), tr("平衡模式"), tr("快速模式")]
        speed_idx = self.speed_combo.currentIndex()
        self.skip_frames = speed_map[speed_idx] if speed_idx < len(speed_map) else 0
        res_map = [(640, 480), (800, 600), (1280, 720), (1920, 1080)]
        self.camera_resolution = res_map[self.res_combo.currentIndex()]
        # 如果摄像头正在运行，重启线程应用新设置
        if self.camera_running and hasattr(self, 'camera_thread') and self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait(2000)
            self.camera_thread = CameraDetectThread(
                detector=self.detector,
                camera_index=0,
                resolution=self.camera_resolution,
                skip_frames=self.skip_frames
            )
            self.camera_thread.frame_ready.connect(self._on_camera_frame)
            self.camera_thread.fps_updated.connect(self._on_camera_fps)
            self.camera_thread.error_occurred.connect(self._on_camera_error)
            self.camera_thread.thread_finished.connect(self._on_camera_thread_finished)
            self.camera_thread.start()
        self.status_bar.showMessage(f"性能设置: {speed_names[speed_idx]}, 摄像头分辨率 {self.camera_resolution[0]}x{self.camera_resolution[1]}")

    # ==================== 缓存管理 ====================
    def _get_cache_dir(self):
        """获取缓存目录"""
        custom = self.config.get("cache.torch_home", "")
        if custom:
            return Path(custom)
        return APP_DIR / "torch_cache"

    def _update_cache_info(self):
        """更新缓存信息显示"""
        cache_dir = self._get_cache_dir()
        self.cache_path_label.setText(f"缓存位置: {cache_dir}")
        # 计算缓存大小
        try:
            total_size = 0
            if cache_dir.exists():
                for f in cache_dir.rglob("*"):
                    if f.is_file():
                        total_size += f.stat().st_size
            if total_size > 1024 * 1024 * 1024:
                size_str = f"{total_size / (1024**3):.2f} GB"
            elif total_size > 1024 * 1024:
                size_str = f"{total_size / (1024**2):.1f} MB"
            else:
                size_str = f"{total_size / 1024:.0f} KB"
            self.cache_size_label.setText(f"缓存大小: {size_str}")
        except Exception:
            self.cache_size_label.setText("缓存大小: 未知")

    def _change_cache_dir(self):
        """更改缓存目录"""
        current = str(self._get_cache_dir())
        new_dir = QFileDialog.getExistingDirectory(self, "选择缓存目录", current)
        if new_dir:
            self.config.set("cache.torch_home", new_dir)
            os.environ["TORCH_HOME"] = new_dir
            self._update_cache_info()
            self.status_bar.showMessage(f"缓存目录已更改: {new_dir}")
            AnimatedDialog.information(self, "提示", "缓存目录已更改，重启软件后生效。")

    def _clean_cache(self):
        """清理缓存（后台线程执行，避免卡死）"""
        # 模型加载中不允许清缓存，避免加载失败
        if hasattr(self, '_model_loading') and self._model_loading:
            AnimatedDialog.warning(self, "提示", "模型正在加载中，请稍候再清理缓存")
            return
        cache_dir = self._get_cache_dir()
        if not cache_dir.exists():
            self.status_bar.showMessage("缓存目录不存在，无需清理")
            return
        reply = AnimatedDialog.question(self, "确认清理",
            f"确定要清理缓存吗？\n\n缓存目录: {cache_dir}\n\n清理后下次启动需要重新下载模型代码。")
        if not reply:
            return
        # 禁用按钮，防止重复点击
        self.clean_cache_btn.setEnabled(False)
        self.clean_cache_btn.setText("清理中...")
        self.status_bar.showMessage("正在清理缓存...")
        # 后台线程执行清理
        self._clean_thread = _CleanCacheThread(cache_dir)
        self._clean_thread.finished.connect(self._on_cache_cleaned)
        self._clean_thread.start()

    def _on_cache_cleaned(self, success, message):
        """缓存清理完成回调"""
        self.clean_cache_btn.setEnabled(True)
        self.clean_cache_btn.setText(tr("清理缓存"))
        if success:
            self._update_cache_info()
            self.status_bar.showMessage("缓存已清理")
        else:
            AnimatedDialog.critical(self, "清理失败", f"清理缓存失败: {message}")

    # ==================== 快捷键 ====================
    def _init_shortcuts(self):
        """初始化快捷键"""
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self._open_image)
        QShortcut(QKeySequence("Ctrl+Shift+O"), self, activated=self._open_video)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_result_image)
        QShortcut(QKeySequence("Space"), self, activated=self._toggle_video_play)
        QShortcut(QKeySequence("Esc"), self, activated=self._stop_camera)
        QShortcut(QKeySequence("F6"), self, activated=self._toggle_screen_detect)
        QShortcut(QKeySequence("F1"), self, activated=self._show_about)

    def _show_about(self):
        """显示关于对话框"""
        device_info = "GPU" if (self.detector and self.detector.device != "cpu") else "CPU"
        model_info = self.detector.model_name if self.detector else "未加载"
        AnimatedDialog.information(self, "关于",
            "YOLOv5 可视化检测系统 v1.0\n\n"
            "功能：图片/视频/摄像头/屏幕实时检测\n"
            f"当前模型：{model_info}\n"
            f"当前设备：{device_info}\n\n"
            "快捷键：\n"
            "  Ctrl+O  打开图片\n"
            "  Ctrl+Shift+O  打开视频\n"
            "  Ctrl+S  保存结果图\n"
            "  Space  播放/暂停视频\n"
            "  F6  开关屏幕检测\n"
            "  F1  显示本帮助\n\n"
            "支持 YOLOv5/v8/v9/v10/v11 模型")

    # ==================== 拖拽支持 ====================
    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """拖拽释放事件"""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if not file_path:
                continue
            ext = Path(file_path).suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                # 切换到图片模式并打开
                self.mode_combo.setCurrentIndex(0)
                QTimer.singleShot(100, lambda: self._open_dropped_file(file_path, 'image'))
                break
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
                # 切换到视频模式并打开
                self.mode_combo.setCurrentIndex(1)
                QTimer.singleShot(100, lambda: self._open_dropped_file(file_path, 'video'))
                break

    def _open_dropped_file(self, file_path, file_type):
        """打开拖拽的文件"""
        if not self._check_model_ready():
            return
        if file_type == 'image':
            self._process_image_file(file_path)
        elif file_type == 'video':
            self._process_video_file(file_path)

    def _redetect_current_image(self):
        """用当前参数重新检测当前图片（调参后快速看效果）"""
        if not self.current_image_path:
            self.status_bar.showMessage("还没有打开图片")
            return
        if not Path(self.current_image_path).exists():
            AnimatedDialog.warning(self, "提示", "原图片文件已移动或删除，请重新打开")
            return
        if not self._check_model_ready():
            return
        self._is_redetecting = True
        self._process_image_file(self.current_image_path)

    def _process_image_file(self, file_path):
        """处理图片文件（供拖拽和打开共用）"""
        # 防止重复：上一次检测还在运行时不允许新检测
        if hasattr(self, 'detect_thread') and self.detect_thread and self.detect_thread.isRunning():
            self.status_bar.showMessage("正在检测中，请稍候...")
            return
        try:
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                AnimatedDialog.critical(self, "错误", "无法读取图片文件，请检查文件格式")
                return
            # 大图片内存保护：超过5000万像素提示用户（避免内存耗尽崩溃）
            img_w, img_h = pixmap.width(), pixmap.height()
            if img_w * img_h > 50000000:
                reply = AnimatedDialog.question(self, "图片过大",
                    f"这张图片尺寸为 {img_w}x{img_h}（{img_w*img_h//1000000}百万像素），\n"
                    f"处理大图可能占用大量内存并导致卡顿。\n\n"
                    f"是否继续处理？建议先用图片软件缩小到4K以内。")
                if not reply:
                    return
            self.current_image_path = file_path
            self.original_label.setPixmap(pixmap.scaled(
                self.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.result_label.setText("检测中...")
            self.open_image_btn.setEnabled(False)
            self.loading_overlay.show_loading("正在检测图片...")
            self.status_bar.showMessage(f"正在检测: {Path(file_path).name}")
            output_path = str(Path("output") / f"{Path(file_path).stem}_result.jpg")
            Path("output").mkdir(exist_ok=True)
            self.detect_thread = DetectThread(self.detector, file_path, output_path)
            self.detect_thread.finished.connect(lambda r: self._on_image_detected(r, file_path))
            self.detect_thread.error.connect(self._on_detect_error)
            self.detect_thread.start()
        except Exception as e:
            self.open_image_btn.setEnabled(True)
            self.loading_overlay.hide_loading()
            AnimatedDialog.critical(self, "错误", f"打开图片失败: {str(e)}")

    def _process_video_file(self, file_path):
        """处理视频文件（供拖拽和打开共用）"""
        if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.isRunning():
            return
        # 释放旧的视频播放资源，避免文件句柄泄漏
        self.video_is_playing = False
        if self.video_timer:
            self.video_timer.stop()
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.play_video_btn.setText("播放")
        self.config.set("recent.last_video", str(Path(file_path).parent))
        self.open_video_btn.setEnabled(False)
        self.play_video_btn.setEnabled(False)
        # 不显示全屏遮罩，让进度条可见
        self.video_progress.setValue(0)
        self.video_progress.setVisible(True)
        self.video_label.setText(f"正在处理视频: {Path(file_path).name}\n\n请稍候，进度条显示处理进度...")
        self.video_label.setAlignment(Qt.AlignCenter)
        output_path = str(Path("output") / f"{Path(file_path).stem}_result.mp4")
        Path("output").mkdir(exist_ok=True)
        self.video_output_path = output_path
        self.current_video_input_path = file_path  # 保存原始视频路径，历史记录用
        self.video_thread = VideoDetectThread(self.detector, file_path, output_path, skip_frames=self.skip_frames)
        self.video_thread.progress.connect(self._on_video_progress)
        self.video_thread.finished.connect(self._on_video_finished)
        self.video_thread.error.connect(self._on_video_error)
        self.video_thread.stopped.connect(self._on_video_stopped)
        self.stop_video_btn.setVisible(True)
        self.video_thread.start()

    # ==================== 检测历史记录 ====================
    def _add_to_history(self, file_type, file_name, detection_count, output_path="", file_path="", update_only=False):
        """添加检测记录（同一文件去重：已存在则更新并移到最前，不重复堆叠）"""
        from datetime import datetime
        # 移除同一文件的旧记录，避免重复堆叠
        if file_path:
            self.detect_history = [r for r in self.detect_history
                                   if not (r.get("path") == file_path and r.get("type") == file_type)]
        record = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": file_type,
            "name": file_name,
            "count": detection_count,
            "output": output_path,
            "path": file_path,  # 完整文件路径
        }
        self.detect_history.insert(0, record)
        # 最多保留 50 条
        if len(self.detect_history) > 50:
            self.detect_history = self.detect_history[:50]
        self._update_history_panel()

    def _update_history_panel(self):
        """更新历史记录面板"""
        if not hasattr(self, 'history_list'):
            return
        self.history_list.clear()
        for i, record in enumerate(self.detect_history[:20]):
            type_icon = "图片" if record["type"] == "image" else "视频"
            item_text = f"[{record['time']}] {type_icon}: {record['name']} ({record['count']}个目标)"
            self.history_list.addItem(item_text)

    def _on_history_double_clicked(self, item):
        """双击历史记录：直接打开已有的检测结果，不重新处理"""
        row = self.history_list.row(item)
        if not (0 <= row < len(self.detect_history)):
            return
        record = self.detect_history[row]
        record_type = record.get("type", "image")
        output_path = record.get("output", "")
        original_path = record.get("path", "")

        # 优先用已有的结果文件；结果文件不存在时才回退到重新处理
        if output_path and Path(output_path).exists():
            if record_type == "video":
                self._open_history_video(output_path, record)
            else:
                self._open_history_image(output_path, record)
        elif original_path and Path(original_path).exists():
            # 结果文件丢了但原文件还在，重新处理
            self.status_bar.showMessage("结果文件已不存在，重新处理中...")
            if record_type == "image":
                self.mode_combo.setCurrentIndex(0)
                QTimer.singleShot(100, lambda: self._process_image_file(original_path))
            else:
                self.mode_combo.setCurrentIndex(1)
                QTimer.singleShot(100, lambda: self._process_video_file(original_path))
        else:
            self.status_bar.showMessage(f"文件不存在: {record.get('name', '')}")

    def _open_history_video(self, video_path, record):
        """从历史记录直接播放结果视频"""
        self.mode_combo.setCurrentIndex(1)
        # 停止当前播放
        self.video_is_playing = False
        if self.video_timer:
            self.video_timer.stop()
        if self.video_capture:
            self.video_capture.release()
        self.play_video_btn.setText("播放")
        # 打开结果视频
        self.video_output_path = video_path
        self.video_capture = cv2.VideoCapture(video_path)
        if self.video_capture.isOpened():
            self._update_video_frame()
            self.play_video_btn.setEnabled(True)
            self.save_video_btn.setEnabled(True)
            self.status_bar.showMessage(f"已打开历史结果: {record.get('name', '')} ({record.get('count', 0)}个目标)")
        else:
            self.video_label.setText("无法打开视频")
            self.status_bar.showMessage("无法打开历史结果视频")

    def _open_history_image(self, image_path, record):
        """从历史记录直接显示结果图片"""
        self.mode_combo.setCurrentIndex(0)
        img = cv2.imread(image_path)
        if img is None:
            self.status_bar.showMessage("无法打开历史结果图片")
            return
        pixmap = cvimg_to_qpixmap(img)
        self.result_label.setPixmap(pixmap.scaled(
            self.result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # 保存原始图片路径（重新检测时用原图，不用结果图）
        original_path = record.get("path", "")
        self.current_image_path = original_path if original_path and Path(original_path).exists() else image_path
        self.save_image_btn.setEnabled(True)
        self.redetect_image_btn.setEnabled(True)
        self.export_json_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)
        self.export_voc_btn.setEnabled(False)
        self.close_image_btn.setEnabled(True)
        self.status_bar.showMessage(f"已打开历史结果: {record.get('name', '')} ({record.get('count', 0)}个目标)")
        # 历史记录图片没有原始检测数据，清空统计面板
        self._update_result_panel([], {})

    def _open_output_folder(self):
        """打开检测结果保存文件夹"""
        try:
            output_dir = Path("output").resolve()
            output_dir.mkdir(exist_ok=True)
            import subprocess
            import sys as _sys
            if _sys.platform == "win32":
                import os
                os.startfile(str(output_dir))
            elif _sys.platform == "darwin":
                subprocess.Popen(["open", str(output_dir)])
            else:
                subprocess.Popen(["xdg-open", str(output_dir)])
            self.status_bar.showMessage(f"已打开: {output_dir}")
        except Exception as e:
            AnimatedDialog.warning(self, "提示", f"打开文件夹失败: {str(e)}")

    # ==================== 模式切换 ====================
    def _switch_mode(self, index):
        # 切换模式前停止所有正在运行的检测线程，避免线程完成后访问已隐藏的UI
        if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait(3000)
        if hasattr(self, 'detect_thread') and self.detect_thread and self.detect_thread.isRunning():
            self.detect_thread.wait(3000)
        self.stack.setCurrentIndex(index)
        if index != 2 and self.camera_running:
            self._stop_camera()
        if index != 1 and self.video_is_playing:
            self._toggle_video_play()
        # 离开屏幕检测模式时停止
        if index != 3 and hasattr(self, 'screen_thread') and self.screen_thread and self.screen_thread.running:
            self._stop_screen_detect()

    # ==================== 图片检测 ====================
    def _open_image(self):
        if not self._check_model_ready():
            return
        # 防止重复点击
        if hasattr(self, 'detect_thread') and self.detect_thread and self.detect_thread.isRunning():
            return
        last_dir = self.config.get("recent.last_image", "")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", last_dir,
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)"
        )
        if not file_path:
            return
        self.config.set("recent.last_image", str(Path(file_path).parent))
        self._process_image_file(file_path)

    def _check_model_ready(self):
        if not self.detector or not self.detector.is_ready():
            message = (
                "模型尚未加载完成，无法进行检测。\n\n"
                "你可以等待模型加载完成，或点击\"选择模型\"加载你自己的模型文件。\n\n"
                "是否现在选择模型文件？"
            )
            buttons = [
                ("取消", False, "normal"),
                ("选择模型", True, "primary"),
            ]
            dlg = AnimatedDialog(self, "模型未就绪", message, AnimatedDialog.WARNING, buttons)
            dlg.exec_()
            if dlg.result_button:
                self._select_model_file()
            return False
        return True

    def _on_image_detected(self, result, original_path):
        self.loading_overlay.hide_loading()
        self.open_image_btn.setEnabled(True)
        self.redetect_image_btn.setEnabled(True)
        self.current_detections = result["detections"]
        self.current_image_info = {
            "filename": Path(original_path).name,
            "width": result["image_width"],
            "height": result["image_height"],
            "folder": "images",
            "path": original_path,
        }
        annotated = result["annotated"]
        pixmap = cvimg_to_qpixmap(annotated)
        self.result_label.setPixmap(pixmap.scaled(
            self.result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.save_image_btn.setEnabled(True)
        self.export_json_btn.setEnabled(True)
        self.export_csv_btn.setEnabled(True)
        self.export_voc_btn.setEnabled(True)
        self.close_image_btn.setEnabled(True)
        self._update_result_panel(result["detections"], result["statistics"])
        self.status_bar.showMessage(f"检测完成，发现 {result['detection_count']} 个目标")
        # 检测为空时给用户明确提示（区分推理错误和真的没检测到）
        if result['detection_count'] == 0:
            last_err = self.detector.get_last_error() if self.detector else ""
            if last_err:
                AnimatedDialog.warning(self, "检测异常",
                    f"检测时发生错误，未返回结果：\n\n{last_err}\n\n"
                    f"常见原因：缺少依赖模块、模型文件损坏。\n"
                    f"日志文件位置：{Path.home()}/.yolov5_desktop/yolov5_app.log")
            else:
                # 真的没检测到，给调参建议，不弹窗（用状态栏），避免打扰
                self.status_bar.showMessage(
                    f"未检测到目标。当前置信度阈值={self.conf_spin.value():.2f}，"
                    f"可尝试降低置信度或换用更大的模型（如yolov5m）")
        # 自动保存标注后的图片到 output 目录，方便历史记录直接打开
        output_img_path = ""
        try:
            Path("output").mkdir(exist_ok=True)
            output_img_path = str(Path("output") / f"{Path(original_path).stem}_result.jpg")
            cv2.imwrite(output_img_path, annotated)
        except Exception as e:
            print(f"[History] 自动保存结果图片失败: {e}")
            output_img_path = ""
        # 添加到历史记录（重新检测同一图片时更新而非重复添加）
        is_redetect = getattr(self, '_is_redetecting', False)
        self._add_to_history("image", Path(original_path).name, result['detection_count'],
                             output_img_path, original_path, update_only=is_redetect)
        self._is_redetecting = False
        # 释放大图片内存（不主动gc.collect，避免卡顿，让Python自动回收）
        del annotated
        del pixmap

    def _on_detect_error(self, error_msg):
        self.loading_overlay.hide_loading()
        self.open_image_btn.setEnabled(True)
        self.result_label.setText("检测失败")
        self.status_bar.showMessage(f"检测失败: {error_msg}")
        AnimatedDialog.critical(self, "检测失败", f"图片检测失败:\n{error_msg}")

    def _save_result_image(self):
        if not self.current_detections:
            return
        default_name = f"{Path(self.current_image_info.get('filename', 'result')).stem}_result.jpg"
        file_path, _ = QFileDialog.getSaveFileName(self, "保存结果图片", default_name, "JPEG图片 (*.jpg);;PNG图片 (*.png)")
        if not file_path:
            return
        try:
            pixmap = self.result_label.pixmap()
            if pixmap and not pixmap.isNull():
                pixmap.save(file_path)
                self.status_bar.showMessage(f"图片已保存: {file_path}")
                self._flash_button(self.save_image_btn, "已保存")
        except Exception as e:
            AnimatedDialog.critical(self, "错误", f"保存失败: {str(e)}")

    def _close_image(self):
        """关闭当前图片，清空显示和结果"""
        # 检测还在运行时不允许关闭
        if hasattr(self, 'detect_thread') and self.detect_thread and self.detect_thread.isRunning():
            AnimatedDialog.information(self, "提示", "正在检测中，请稍候再关闭")
            return
        self.original_label.clear()
        self.original_label.setText("请打开图片")
        self.result_label.clear()
        self.result_label.setText("等待检测")
        self.current_image_path = None
        self.current_detections = []
        self.current_image_info = {}
        # 禁用相关按钮
        self.redetect_image_btn.setEnabled(False)
        self.save_image_btn.setEnabled(False)
        self.export_json_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)
        self.export_voc_btn.setEnabled(False)
        self.close_image_btn.setEnabled(False)
        # 清空统计面板
        self._update_result_panel([], {})
        self.status_bar.showMessage("已关闭图片")

    def _close_video(self):
        """关闭当前视频，停止播放并清空"""
        self.video_is_playing = False
        if self.video_timer:
            self.video_timer.stop()
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.video_label.clear()
        self.video_label.setText("请打开视频文件")
        self.video_output_path = None
        self.play_video_btn.setText("播放")
        self.play_video_btn.setEnabled(False)
        self.save_video_btn.setEnabled(False)
        self.close_video_btn.setEnabled(False)
        self.video_progress.setValue(0)
        self._update_result_panel([], {})
        self.status_bar.showMessage("已关闭视频")

    def _export_result(self, fmt):
        if not self.current_detections:
            AnimatedDialog.warning(self, "提示", "没有可导出的检测结果")
            return
        ext = "xml" if fmt == "voc" else fmt
        default_name = f"{Path(self.current_image_info.get('filename', 'result')).stem}_result.{ext}"
        file_path, _ = QFileDialog.getSaveFileName(self, f"导出{fmt.upper()}", default_name, f"{fmt.upper()}文件 (*.{ext})")
        if not file_path:
            return
        try:
            if fmt == "json":
                self.detector.export_to_json(self.current_detections, self.current_image_info, file_path)
                self._flash_button(self.export_json_btn, "已导出")
            elif fmt == "csv":
                self.detector.export_to_csv(self.current_detections, self.current_image_info, file_path)
                self._flash_button(self.export_csv_btn, "已导出")
            elif fmt == "voc":
                self.detector.export_to_voc_xml(self.current_detections, self.current_image_info, file_path)
                self._flash_button(self.export_voc_btn, "已导出")
            self.status_bar.showMessage(f"已导出{fmt.upper()}: {file_path}")
        except Exception as e:
            AnimatedDialog.critical(self, "导出失败", f"导出失败: {str(e)}")

    # ==================== 视频检测 ====================
    def _open_video(self):
        if not self._check_model_ready():
            return
        # 防止重复点击
        if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.isRunning():
            return
        last_dir = self.config.get("recent.last_video", "")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", last_dir,
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*)"
        )
        if not file_path:
            return
        self.config.set("recent.last_video", str(Path(file_path).parent))
        self._process_video_file(file_path)

    def _on_video_progress(self, current, total):
        if total > 0:
            percent = int(current / total * 100)
            # 平滑动画过渡
            if hasattr(self, '_progress_anim') and self._progress_anim:
                self._progress_anim.stop()
            from PyQt5.QtCore import QPropertyAnimation
            self._progress_anim = QPropertyAnimation(self.video_progress, b"value")
            self._progress_anim.setDuration(300)
            self._progress_anim.setStartValue(self.video_progress.value())
            self._progress_anim.setEndValue(percent)
            self._progress_anim.setEasingCurve(QEasingCurve.OutCubic)
            self._progress_anim.start()
            self.status_bar.showMessage(f"正在处理视频... {percent}% ({current}/{total}帧)")

    def _on_video_finished(self, result):
        self.open_video_btn.setEnabled(True)
        self.video_progress.setValue(100)
        self.save_video_btn.setEnabled(True)
        self.play_video_btn.setEnabled(True)
        self.close_video_btn.setEnabled(True)
        self.stop_video_btn.setVisible(False)
        # 用实际输出路径更新（编码器可能改了扩展名，如 .mp4→.avi）
        actual_output = result.get("output_path", self.video_output_path)
        self.video_output_path = actual_output
        encoder = result.get("encoder", "")
        self.status_bar.showMessage(f"视频处理完成: {result['total_frames']}帧, 检测到{result['total_detections']}个目标, 编码器: {encoder}")
        # 添加到历史记录（output是结果视频，path是原始视频）
        original_input = getattr(self, 'current_video_input_path', actual_output)
        self._add_to_history("video", Path(actual_output).name, result['total_detections'],
                             actual_output, original_input)
        if self.video_capture:
            self.video_capture.release()
        self.video_capture = cv2.VideoCapture(actual_output)
        if self.video_capture.isOpened():
            self._update_video_frame()
        self._update_result_panel_from_stats(result["statistics"], result["total_detections"])
        # 成功时不弹窗，只用状态栏提示，避免干扰用户

    def _on_video_error(self, error_msg):
        self.open_video_btn.setEnabled(True)
        self.play_video_btn.setEnabled(True)
        self.stop_video_btn.setVisible(False)
        self.video_label.setText("处理失败")
        self.status_bar.showMessage(f"视频处理失败: {error_msg}")
        AnimatedDialog.critical(self, "处理失败", f"视频处理失败:\n{error_msg}")

    def _stop_video_processing(self):
        """用户点击停止处理按钮"""
        if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.status_bar.showMessage("正在停止视频处理...")
            self.stop_video_btn.setEnabled(False)
            self.stop_video_btn.setText("停止中...")

    def _on_video_stopped(self):
        """用户中止视频处理后的回调"""
        self.open_video_btn.setEnabled(True)
        self.play_video_btn.setEnabled(False)
        self.stop_video_btn.setVisible(False)
        self.stop_video_btn.setEnabled(True)
        self.stop_video_btn.setText("停止处理")
        self.video_progress.setVisible(False)
        self.video_label.setText("视频处理已中止")
        self.status_bar.showMessage("视频处理已中止，部分结果可能未保存")

    def _toggle_video_play(self):
        if not self.video_capture or not self.video_capture.isOpened():
            AnimatedDialog.warning(self, "提示", "没有可播放的视频")
            return
        if self.video_is_playing:
            self.video_timer.stop()
            self.video_is_playing = False
            self.play_video_btn.setText("播放")
        else:
            self.video_timer.start(33)
            self.video_is_playing = True
            self.play_video_btn.setText("暂停")

    def _update_video_frame(self):
        if not self.video_capture:
            return
        ret, frame = self.video_capture.read()
        if not ret:
            # 视频播放结束，自动停止（不循环）
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._toggle_video_play()
            self.status_bar.showMessage("视频播放结束")
            return
        pixmap = cvimg_to_qpixmap(frame)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _save_result_video(self):
        if not self.video_output_path or not Path(self.video_output_path).exists():
            AnimatedDialog.warning(self, "提示", "没有可保存的结果视频")
            return
        default_name = Path(self.video_output_path).name
        current_ext = Path(self.video_output_path).suffix.lower()
        if current_ext == ".avi":
            file_filter = "AVI视频 (*.avi);;MP4视频 (*.mp4)"
        else:
            file_filter = "MP4视频 (*.mp4);;AVI视频 (*.avi)"
        file_path, _ = QFileDialog.getSaveFileName(self, "保存结果视频", default_name, file_filter)
        if not file_path:
            return
        # 确保扩展名正确
        if not file_path.lower().endswith((".mp4", ".avi")):
            file_path += current_ext
        try:
            shutil.copy2(self.video_output_path, file_path)
            self.status_bar.showMessage(f"视频已保存: {file_path}")
            self._flash_button(self.save_video_btn, "已保存")
        except Exception as e:
            AnimatedDialog.critical(self, "错误", f"保存失败: {str(e)}")

    # ==================== 摄像头实时检测 ====================
    def _start_camera(self):
        if not self._check_model_ready():
            return
        # 互斥：屏幕检测运行时不允许开启摄像头
        if hasattr(self, 'screen_thread') and self.screen_thread and self.screen_thread.running:
            AnimatedDialog.warning(self, "提示", "屏幕检测正在运行，请先停止屏幕检测再开启摄像头")
            return
        try:
            self.camera_running = True
            self.camera_last_detections = []
            self.camera_last_annotated = None
            self.start_camera_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(True)
            self.camera_capture_btn.setEnabled(True)
            # 创建摄像头检测线程（独立线程，不阻塞UI）
            self.camera_thread = CameraDetectThread(
                detector=self.detector,
                camera_index=0,
                resolution=self.camera_resolution,
                skip_frames=self.skip_frames
            )
            self.camera_thread.frame_ready.connect(self._on_camera_frame)
            self.camera_thread.fps_updated.connect(self._on_camera_fps)
            self.camera_thread.error_occurred.connect(self._on_camera_error)
            self.camera_thread.thread_finished.connect(self._on_camera_thread_finished)
            self.camera_thread.start()
            speed_names = [tr("高质量"), tr("标准"), tr("快速")]
            speed_text = speed_names[self.speed_combo.currentIndex()] if self.speed_combo.currentIndex() < 3 else tr("标准")
            self.status_bar.showMessage(f"摄像头已开启 ({self.camera_resolution[0]}x{self.camera_resolution[1]}, {speed_text}模式)")
        except Exception as e:
            AnimatedDialog.critical(self, "错误", f"开启摄像头失败: {str(e)}")
            self.camera_running = False
            self.start_camera_btn.setEnabled(True)
            self.stop_camera_btn.setEnabled(False)
            self.camera_capture_btn.setEnabled(False)

    def _on_camera_frame(self, frame, detections, annotated):
        """摄像头线程返回一帧，更新UI"""
        if not self.camera_running:
            return
        self.camera_last_annotated = annotated
        self.camera_last_detections = detections
        pixmap = cvimg_to_qpixmap(annotated)
        # 实时显示用快速缩放，保证流畅度
        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        self.camera_count_label.setText(f"检测目标: {len(detections)}")

    def _on_camera_fps(self, fps):
        self.camera_fps_label.setText(f"FPS: {fps}")

    def _on_camera_error(self, error_msg):
        import logging
        logging.getLogger("yolov5_app").error(f"摄像头错误: {error_msg}")
        self.status_bar.showMessage(f"摄像头错误: {error_msg}")
        # 延迟调用避免在线程信号里wait导致阻塞
        QTimer.singleShot(0, self._stop_camera)

    def _on_camera_thread_finished(self, reason):
        import logging
        logging.getLogger("yolov5_app").info(f"摄像头线程结束: {reason}")
        if self.camera_running:
            self.status_bar.showMessage(f"摄像头已停止: {reason}")
            QTimer.singleShot(0, self._stop_camera)

    def _stop_camera(self):
        import logging
        logging.getLogger("yolov5_app").info("停止摄像头")
        self.camera_running = False
        # 停止摄像头线程
        if hasattr(self, 'camera_thread') and self.camera_thread:
            self.camera_thread.stop()
            if self.camera_thread.isRunning():
                self.camera_thread.wait(3000)
            self.camera_thread = None
        self.start_camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)
        self.camera_capture_btn.setEnabled(False)
        self.camera_label.setText(tr('点击"开启摄像头"开始实时检测'))
        self.camera_fps_label.setText(tr("FPS: 0"))
        self.camera_count_label.setText(tr("检测目标: 0"))
        self.status_bar.showMessage("摄像头已停止")

    def _capture_camera_frame(self):
        """摄像头截图保存当前检测画面"""
        if self.camera_last_annotated is None:
            self.status_bar.showMessage("暂无可保存的画面")
            return
        try:
            from datetime import datetime
            Path("output").mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"camera_{ts}.jpg"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存截图", str(Path("output") / default_name), "JPEG图片 (*.jpg);;PNG图片 (*.png)")
            if not file_path:
                return
            cv2.imwrite(file_path, self.camera_last_annotated)
            self.status_bar.showMessage(f"截图已保存: {file_path}")
            self._flash_button(self.camera_capture_btn, "已截图")
        except Exception as e:
            AnimatedDialog.critical(self, "错误", f"截图失败: {str(e)}")

    def _update_camera_frame(self):
        if not self.camera_capture or not self.camera_running:
            return
        ret, frame = self.camera_capture.read()
        if not ret:
            return
        try:
            self.camera_frame_count += 1
            # 跳帧逻辑：每隔 skip_frames+1 帧检测一次，中间帧复用上次结果
            need_detect = (self.camera_frame_count - 1) % (self.skip_frames + 1) == 0
            if need_detect or self.camera_last_annotated is None:
                annotated, detections = self.detector.detect(frame)
                self.camera_last_detections = detections
                self.camera_last_annotated = annotated
            else:
                # 复用上次检测结果，在当前帧上绘制
                annotated = frame.copy()
                for d in self.camera_last_detections:
                    bbox = d["bbox"]
                    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), d["color"], 2)
                    label = f"{d['class_name']} {d['confidence']:.2f}"
                    cv2.putText(annotated, label, (x1, max(y1 - 8, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, d["color"], 1)
                detections = self.camera_last_detections

            pixmap = cvimg_to_qpixmap(annotated)
            # 实时显示用快速缩放，保证流畅度（摄像头不需要高质量缩放）
            self.camera_label.setPixmap(pixmap.scaled(
                self.camera_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
            self.camera_fps_frames += 1
            now = time.time()
            if now - self.camera_fps_time >= 1.0:
                fps = int(self.camera_fps_frames / (now - self.camera_fps_time))
                self.camera_fps_label.setText(f"FPS: {fps}")
                self.camera_fps_frames = 0
                self.camera_fps_time = now
            self.camera_count_label.setText(f"检测目标: {len(detections)}")
            if self.camera_frame_count % 15 == 0:
                stats = {}
                for d in detections:
                    stats[d["class_name"]] = stats.get(d["class_name"], 0) + 1
                self._update_result_panel(detections, stats)
        except Exception as e:
            print(f"[Camera] 检测帧失败: {e}")

    # ==================== 结果面板 ====================
    # ==================== 屏幕实时检测 ====================
    def _start_screen_detect(self):
        """开始屏幕实时检测"""
        if not self._check_model_ready():
            return
        if hasattr(self, 'screen_thread') and self.screen_thread and self.screen_thread.running:
            return
        # 互斥：摄像头运行时不允许开启屏幕检测
        if self.camera_running:
            AnimatedDialog.warning(self, "提示", "摄像头检测正在运行，请先停止摄像头再开启屏幕检测")
            return
        # 根据性能预设设置参数
        preset_idx = self.screen_perf_combo.currentIndex()
        if preset_idx == 0:  # 流畅
            skip_frames, target_width, max_fps = 2, 480, 20
        elif preset_idx == 1:  # 标准
            skip_frames, target_width, max_fps = 1, 640, 30
        elif preset_idx == 2:  # 高质量
            skip_frames, target_width, max_fps = 0, 640, 30
        else:  # 自定义：读取用户设置
            skip_map = [2, 1, 0, 0]
            res_map = [320, 480, 640, 800]
            fps_map = [10, 15, 20, 30, 60]
            skip_frames = skip_map[self.screen_res_combo.currentIndex()] if self.screen_res_combo.currentIndex() <= 2 else 0
            target_width = res_map[self.screen_res_combo.currentIndex()]
            max_fps = fps_map[self.screen_fps_combo.currentIndex()]
        # 置信度和线宽用左侧全局检测参数（detector已过滤，min_conf传0不二次过滤）
        min_conf = 0.0
        # 创建检测线程
        self.screen_thread = ScreenDetectThread(
            self.detector, skip_frames=skip_frames, target_width=target_width,
            max_fps=max_fps, min_conf=min_conf
        )
        self.screen_thread.detections_ready.connect(self._on_screen_detections)
        self.screen_thread.fps_update.connect(self._on_screen_fps)
        self.screen_thread.error_occurred.connect(self._on_screen_error)
        # 创建叠加窗口（线宽用全局检测参数）
        self.overlay = OverlayWindow()
        self.overlay.set_line_thickness(self.detector.line_thickness)
        self.overlay.set_show_labels(self.screen_show_label_check.isChecked())
        self.overlay.set_show_confidence(self.screen_show_conf_check.isChecked())
        self.overlay.set_box_alpha(self.screen_alpha_slider.value())
        self.overlay.show()
        self.screen_thread.start()
        self.start_screen_btn.setEnabled(False)
        self.stop_screen_btn.setEnabled(True)
        self.screen_capture_btn.setEnabled(True)
        device = self.detector.device if self.detector else "?"
        self.status_bar.showMessage(f"屏幕检测已开始（设备: {device}, 分辨率: {target_width}px, 帧率上限: {max_fps}）")

    def _stop_screen_detect(self):
        """停止屏幕实时检测"""
        if hasattr(self, 'screen_thread') and self.screen_thread:
            self.screen_thread.stop()
            self.screen_thread = None
        if hasattr(self, 'overlay') and self.overlay:
            self.overlay.close()
            self.overlay = None
        self.start_screen_btn.setEnabled(True)
        self.stop_screen_btn.setEnabled(False)
        self.screen_capture_btn.setEnabled(False)
        self.screen_fps_label.setText(tr("FPS: 0"))
        self.screen_count_label.setText(tr("检测目标: 0"))
        self.status_bar.showMessage("屏幕检测已停止")

    def _capture_screen_frame(self):
        """屏幕检测时截取当前屏幕（含检测框）保存"""
        try:
            import mss
            import numpy as np
            from datetime import datetime
            # 先隐藏叠加窗口，避免截到它自己
            if hasattr(self, 'overlay') and self.overlay:
                self.overlay.hide()
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                shot = np.array(sct.grab(monitor))
                shot = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
            # 重新显示叠加窗口
            if hasattr(self, 'overlay') and self.overlay:
                self.overlay.show()
            # 把最近的检测框画到截图上
            if hasattr(self, '_screen_last_dets') and self._screen_last_dets:
                for det in self._screen_last_dets:
                    x1, y1, x2, y2 = det["bbox"]
                    color = det.get("color", (0, 255, 0))
                    thickness = self.detector.line_thickness
                    cv2.rectangle(shot, (x1, y1), (x2, y2), color, thickness)
                    if self.screen_show_label_check.isChecked():
                        label = det["class_name"]
                        if self.screen_show_conf_check.isChecked():
                            label += f" {det['confidence']:.2f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale = 0.6
                        (tw, th), _ = cv2.getTextSize(label, font, scale, 2)
                        cv2.rectangle(shot, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                        cv2.putText(shot, label, (x1, y1 - 4), font, scale, (255, 255, 255), 2)
            Path("output").mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"screen_{ts}.jpg"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存截图", str(Path("output") / default_name), "JPEG图片 (*.jpg);;PNG图片 (*.png)")
            if not file_path:
                return
            cv2.imwrite(file_path, shot)
            self.status_bar.showMessage(f"截图已保存: {file_path}")
            self._flash_button(self.screen_capture_btn, "已截图")
        except Exception as e:
            if hasattr(self, 'overlay') and self.overlay:
                self.overlay.show()
            AnimatedDialog.critical(self, "错误", f"截图失败: {str(e)}")

    def _on_screen_detections(self, detections, screen_w, screen_h):
        """收到屏幕检测结果"""
        self._screen_last_dets = detections
        if hasattr(self, 'overlay') and self.overlay:
            self.overlay.update_detections(detections, screen_w, screen_h)
        self.screen_count_label.setText(f"检测目标: {len(detections)}")
        # 节流更新右侧统计面板（每0.5秒一次，避免高频刷新卡顿）
        now = time.time()
        if not hasattr(self, '_screen_last_stat_time'):
            self._screen_last_stat_time = 0
        if now - self._screen_last_stat_time >= 0.5:
            self._screen_last_stat_time = now
            stats = {}
            for d in detections:
                stats[d["class_name"]] = stats.get(d["class_name"], 0) + 1
            self.total_count_label.setText(str(len(detections)))
            self.class_count_label.setText(str(len(stats)))
            stats_text = ""
            for name, count in sorted(stats.items(), key=lambda x: -x[1]):
                stats_text += f"  {name}: {count}\n"
            self.stats_text.setPlainText(stats_text if stats_text else "  无检测目标")

    def _on_screen_fps(self, fps):
        """收到 FPS 更新"""
        self.screen_fps_label.setText(f"FPS: {fps:.1f}")

    def _on_screen_error(self, error_msg):
        """屏幕检测出错"""
        self._stop_screen_detect()
        AnimatedDialog.critical(self, "屏幕检测失败", f"屏幕检测出错:\n{error_msg}")

    def _on_screen_label_changed(self, state):
        """显示标签变化"""
        if hasattr(self, 'overlay') and self.overlay:
            self.overlay.set_show_labels(state == Qt.Checked)

    def _on_screen_preset_changed(self, index):
        """性能预设切换：自动设置分辨率和帧率"""
        # 自定义模式不自动设置
        if index == 3:
            return
        # 流畅=480px+20fps, 标准=640px+30fps, 高质量=640px+30fps
        res_idx = [1, 2, 2]  # 480, 640, 640
        fps_idx = [2, 3, 3]  # 20, 30, 30
        self.screen_res_combo.blockSignals(True)
        self.screen_fps_combo.blockSignals(True)
        self.screen_res_combo.setCurrentIndex(res_idx[index])
        self.screen_fps_combo.setCurrentIndex(fps_idx[index])
        self.screen_res_combo.blockSignals(False)
        self.screen_fps_combo.blockSignals(False)

    def _on_screen_alpha_changed(self, value):
        """框线透明度变化"""
        self.screen_alpha_label.setText(str(value))
        if hasattr(self, 'overlay') and self.overlay:
            self.overlay.set_box_alpha(value)

    def _on_screen_conf_display_changed(self, state):
        """置信度显示切换"""
        if hasattr(self, 'overlay') and self.overlay:
            self.overlay.set_show_confidence(state == Qt.Checked)

    def _update_screen_device_hint(self):
        """根据设备更新屏幕检测的设备提示和推荐预设"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.screen_device_hint.setText(f"GPU: {gpu_name}")
                self.screen_device_hint.setStyleSheet("color: #4caf50; font-size: 11px;")
            else:
                self.screen_device_hint.setText("CPU模式（建议流畅模式）")
                self.screen_device_hint.setStyleSheet("color: #ff9800; font-size: 11px;")
        except Exception:
            self.screen_device_hint.setText("")

    def _toggle_screen_detect(self):
        """切换屏幕检测（快捷键用）"""
        if hasattr(self, 'screen_thread') and self.screen_thread and self.screen_thread.running:
            self._stop_screen_detect()
        else:
            self.mode_combo.setCurrentIndex(3)
            QTimer.singleShot(100, self._start_screen_detect)

    def _update_result_panel(self, detections, statistics):
        self.total_count_label.setText(str(len(detections)))
        self.class_count_label.setText(str(len(statistics)))
        # 计算每个类别的平均置信度
        class_confs = {}
        for d in detections:
            name = d["class_name"]
            if name not in class_confs:
                class_confs[name] = []
            class_confs[name].append(d["confidence"])
        stats_text = ""
        for name, count in sorted(statistics.items(), key=lambda x: -x[1]):
            avg_conf = sum(class_confs.get(name, [0])) / max(1, len(class_confs.get(name, [0])))
            stats_text += f"  {name}: {count}个 (平均{avg_conf:.2f})\n"
        self.stats_text.setPlainText(stats_text if stats_text else "  无检测目标")
        self.detail_table.setRowCount(len(detections))
        for i, d in enumerate(detections):
            bbox = d["bbox"]
            self.detail_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.detail_table.setItem(i, 1, QTableWidgetItem(d["class_name"]))
            self.detail_table.setItem(i, 2, QTableWidgetItem(f"{d['confidence']:.2f}"))
            self.detail_table.setItem(i, 3, QTableWidgetItem(f"({bbox['x1']},{bbox['y1']})"))
            self.detail_table.setItem(i, 4, QTableWidgetItem(f"{d['width']}×{d['height']}"))

    def _update_result_panel_from_stats(self, statistics, total_detections):
        self.total_count_label.setText(str(total_detections))
        self.class_count_label.setText(str(len(statistics)))
        stats_text = ""
        for name, count in sorted(statistics.items(), key=lambda x: -x[1]):
            stats_text += f"  {name}: {count}\n"
        self.stats_text.setPlainText(stats_text if stats_text else "  无检测目标")
        self.detail_table.setRowCount(0)

    def _clear_results(self):
        self.total_count_label.setText("0")
        self.class_count_label.setText("0")
        self.stats_text.clear()
        self.detail_table.setRowCount(0)
        self.current_detections = []

    # ==================== 窗口事件 ====================
    def closeEvent(self, event):
        # 关闭确认，防止误触叉叉直接退出
        try:
            is_processing = (hasattr(self, 'video_thread') and self.video_thread and self.video_thread.isRunning())
            if is_processing:
                msg = "正在处理视频，退出会中断当前处理。\n确定要退出吗？"
            else:
                msg = "确定要退出吗？"
            buttons = [
                ("取消", False, "normal"),
                ("退出", True, "primary"),
            ]
            dlg = AnimatedDialog(self, "确认退出", msg, AnimatedDialog.QUESTION, buttons)
            dlg.exec_()
            if not dlg.result_button:
                event.ignore()
                return
        except Exception as e:
            print(f"[Close] 关闭确认出错: {e}")
        try:
            self._save_config()
            self._stop_camera()
            # 停止屏幕检测
            if hasattr(self, 'screen_thread') and self.screen_thread and self.screen_thread.running:
                self._stop_screen_detect()
            # 停止视频播放
            self.video_is_playing = False
            if self.video_timer:
                self.video_timer.stop()
            if self.video_capture:
                self.video_capture.release()
            # 停止加载遮罩定时器
            if hasattr(self, 'loading_overlay') and self.loading_overlay:
                self.loading_overlay.hide_loading()
            # 停止超时计时器
            if hasattr(self, '_load_timeout_timer'):
                self._load_timeout_timer.stop()
            # 停止视频处理线程（避免输出文件损坏）
            if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.isRunning():
                self.video_thread.stop()
            # 等待所有线程完成（最多等2秒）
            threads = []
            if hasattr(self, '_init_thread') and self._init_thread:
                threads.append(self._init_thread)
            if hasattr(self, 'detect_thread') and self.detect_thread:
                threads.append(self.detect_thread)
            if hasattr(self, 'video_thread') and self.video_thread:
                threads.append(self.video_thread)
            if hasattr(self, 'model_thread') and self.model_thread:
                threads.append(self.model_thread)
            if hasattr(self, '_load_model_thread') and self._load_model_thread:
                threads.append(self._load_model_thread)
            for t in threads:
                if t.isRunning():
                    t.wait(2000)
        except Exception as e:
            print(f"[Close] 清理资源时出错: {e}")
        event.accept()


# ==================== 缓存清理线程 ====================
class _CleanCacheThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, cache_dir):
        super().__init__()
        self.cache_dir = cache_dir

    def run(self):
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, str(e))


# ==================== 模型初始化线程 ====================
class _ModelInitThread(QThread):
    finished = pyqtSignal(object, bool)

    def __init__(self, model_name=""):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            detector = get_detector(model_dir="models", device="auto")
            # 如果配置中保存了模型，尝试加载
            if self.model_name and self.model_name != "yolov5s (默认)":
                try:
                    success, error_msg = detector.load_model_by_name(self.model_name)
                    if not success:
                        print(f"加载保存的模型失败，使用默认模型: {error_msg}")
                except Exception as e:
                    print(f"加载保存的模型失败，使用默认模型: {e}")
            self.finished.emit(detector, True)
        except Exception as e:
            print(f"模型初始化失败: {e}")
            self.finished.emit(None, False)


class _LoadModelFileThread(QThread):
    """从文件加载模型的线程（直接在现有detector上加载，不重复创建）"""
    finished = pyqtSignal(bool, str, str)

    def __init__(self, detector, model_path):
        super().__init__()
        self.detector = detector
        self.model_path = model_path

    def run(self):
        try:
            model_name = Path(self.model_path).name
            success, error_msg = self.detector.load_model_by_name(model_name)
            self.finished.emit(success, model_name, error_msg)
        except Exception as e:
            self.finished.emit(False, "", str(e))
