# -*- coding: utf-8 -*-
"""
屏幕实时检测模块
- ScreenDetectThread: 屏幕捕获 + YOLOv5 检测线程
- OverlayWindow: 全屏透明叠加窗口，绘制检测框，鼠标穿透
"""
import numpy as np
import cv2
import time
from pathlib import Path
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QImage, QPixmap


class ScreenDetectThread(QThread):
    """屏幕捕获 + 检测线程"""
    frame_ready = pyqtSignal(np.ndarray)        # 原始屏幕帧（用于显示）
    detections_ready = pyqtSignal(list, int, int)  # 检测结果, 屏幕宽, 屏幕高
    fps_update = pyqtSignal(float)              # FPS 更新
    error_occurred = pyqtSignal(str)            # 错误

    def __init__(self, detector, skip_frames=2, target_width=480,
                 max_fps=30, min_conf=0.0):
        super().__init__()
        self.detector = detector
        self.skip_frames = skip_frames      # 跳帧数（0=逐帧）
        self.target_width = target_width    # 检测分辨率宽度
        self.max_fps = max_fps              # 最大帧率
        self.min_conf = min_conf            # 置信度过滤（0=不过滤）
        self.running = False
        self._frame_count = 0
        self._last_detections = []
        self._last_time = 0
        self._fps_frames = 0
        self._fps_start = 0

    def run(self):
        try:
            import mss
            self.running = True
            self._frame_count = 0
            self._fps_frames = 0
            self._fps_start = time.time()
            frame_interval = 1.0 / max(1, self.max_fps)

            with mss.mss() as sct:
                # 获取主显示器
                monitor = sct.monitors[1]  # 1=主显示器
                screen_w = monitor["width"]
                screen_h = monitor["height"]

                while self.running:
                    loop_start = time.time()

                    # 捕获屏幕
                    img = sct.grab(monitor)
                    frame = np.array(img)
                    # BGRA -> BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    self._frame_count += 1
                    self._fps_frames += 1

                    # FPS 计算
                    now = time.time()
                    if now - self._fps_start >= 1.0:
                        fps = self._fps_frames / (now - self._fps_start)
                        self.fps_update.emit(fps)
                        self._fps_frames = 0
                        self._fps_start = now

                    # 跳帧检测：skip_frames=0时逐帧，否则每隔skip_frames+1帧检测一次
                    need_detect = (self._frame_count - 1) % (self.skip_frames + 1) == 0
                    if need_detect or not self._last_detections:
                        # 缩放到检测分辨率
                        h, w = frame.shape[:2]
                        scale = self.target_width / w
                        small_frame = cv2.resize(frame, (self.target_width, int(h * scale)))
                        # 检测
                        _, detections = self.detector.detect(small_frame)
                        # 置信度过滤
                        if self.min_conf > 0:
                            detections = [d for d in detections if d["confidence"] >= self.min_conf]
                        # 坐标映射回原始屏幕分辨率
                        scale_back = w / self.target_width
                        for d in detections:
                            d["bbox"]["x1"] = int(d["bbox"]["x1"] * scale_back)
                            d["bbox"]["y1"] = int(d["bbox"]["y1"] * scale_back)
                            d["bbox"]["x2"] = int(d["bbox"]["x2"] * scale_back)
                            d["bbox"]["y2"] = int(d["bbox"]["y2"] * scale_back)
                        self._last_detections = detections

                    # 发送检测结果（复用上次结果）
                    self.detections_ready.emit(self._last_detections, screen_w, screen_h)

                    # FPS 限制
                    elapsed = time.time() - loop_start
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        self.msleep(int(sleep_time * 1000))

        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self.running = False
        self.wait(3000)


class OverlayWindow(QWidget):
    """全屏透明叠加窗口，绘制检测框，鼠标穿透"""

    def __init__(self):
        super().__init__()
        self.detections = []
        self.screen_w = 1920
        self.screen_h = 1080
        self.line_thickness = 2
        self.show_labels = True
        self.box_alpha = 220          # 框线透明度 0-255
        self.label_bg_alpha = 180     # 标签背景透明度
        self.show_confidence = True   # 显示置信度
        self._init_ui()

    def _init_ui(self):
        # 无边框 + 始终置顶 + 工具窗口（不在任务栏显示）
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        # 透明背景
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 鼠标穿透
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        # 全屏
        self.showFullScreen()

    def update_detections(self, detections, screen_w, screen_h):
        """更新检测结果并重绘"""
        self.detections = detections
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.update()

    def set_line_thickness(self, thickness):
        self.line_thickness = thickness

    def set_show_labels(self, show):
        self.show_labels = show

    def set_box_alpha(self, alpha):
        self.box_alpha = max(0, min(255, alpha))

    def set_label_bg_alpha(self, alpha):
        self.label_bg_alpha = max(0, min(255, alpha))

    def set_show_confidence(self, show):
        self.show_confidence = show

    def paintEvent(self, event):
        if not self.detections:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        for d in self.detections:
            bbox = d["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            color = d.get("color", (0, 255, 0))
            qcolor = QColor(color[0], color[1], color[2], self.box_alpha)

            # 画框
            pen = QPen(qcolor, self.line_thickness)
            painter.setPen(pen)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

            # 画标签
            if self.show_labels:
                if self.show_confidence:
                    label = f"{d['class_name']} {d['confidence']:.2f}"
                else:
                    label = d["class_name"]
                font = QFont("Microsoft YaHei", 9, QFont.Bold)
                painter.setFont(font)
                # 标签背景
                font_metrics = painter.fontMetrics()
                text_w = font_metrics.horizontalAdvance(label)
                text_h = font_metrics.height()
                bg_color = QColor(color[0], color[1], color[2], self.label_bg_alpha)
                painter.fillRect(x1, y1 - text_h - 4, text_w + 8, text_h + 4, bg_color)
                # 标签文字
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(x1 + 4, y1 - 6, label)

        painter.end()
