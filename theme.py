# -*- coding: utf-8 -*-
"""
主题与弹窗模块
- 深色/浅色主题切换
- 跟随系统主题
- 带动画的自定义弹窗
- 文字层级优化
"""
import sys
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGraphicsOpacityEffect, QSizePolicy
)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer, QSize, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QIcon, QPixmap, QPainter, QPen, QBrush


# ==================== 深色主题（优化文字层级）====================
DARK_QSS = """
QMainWindow, QWidget {
    background-color: #1a1d23;
    color: #e8eaed;
    font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
    font-size: 13px;
}
QFrame#titleBar {
    background-color: #121417;
    border-bottom: 1px solid #2d3139;
}
QLabel#titleLabel {
    font-size: 17px;
    font-weight: 700;
    color: #4fc3f7;
    letter-spacing: 0.5px;
}
QLabel#statusModelLabel, QLabel#statusDeviceLabel {
    color: #9aa0a6;
    font-size: 12px;
}
QPushButton {
    background-color: #252830;
    color: #e8eaed;
    border: 1px solid #3a3e47;
    border-radius: 8px;
    padding: 9px 18px;
    font-size: 13px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #2d3139;
    border-color: #4fc3f7;
}
QPushButton:pressed {
    background-color: #1e2128;
}
QPushButton#primaryBtn {
    background-color: #4fc3f7;
    color: #0a0a0a;
    border: none;
    font-weight: 600;
}
QPushButton#primaryBtn:hover {
    background-color: #29b6f6;
}
QPushButton#primaryBtn:disabled {
    background-color: #3a3e47;
    color: #7a7a7a;
}
QPushButton#dangerBtn {
    background-color: #ef5350;
    color: #fff;
    border: none;
    font-weight: 600;
}
QPushButton#dangerBtn:hover {
    background-color: #e53935;
}
QPushButton:disabled {
    background-color: #1e2128;
    color: #5f6368;
    border-color: #2d3139;
}
QComboBox {
    background-color: #16181d;
    color: #e8eaed;
    border: 1px solid #3a3e47;
    border-radius: 8px;
    padding: 7px 12px;
    min-height: 26px;
    font-size: 13px;
}
QComboBox:hover { border-color: #4fc3f7; }
QComboBox::drop-down { border: none; width: 28px; }
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #9aa0a6;
    margin-right: 10px;
}
QComboBox QAbstractItemView {
    background-color: #1e2128;
    color: #e8eaed;
    selection-background-color: #4fc3f7;
    selection-color: #0a0a0a;
    border: 1px solid #3a3e47;
    border-radius: 6px;
    padding: 4px;
    outline: none;
}
QComboBox QAbstractItemView::item {
    padding: 8px 12px;
    border-radius: 4px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #16181d;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #4fc3f7;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 18px;
    height: 18px;
    background: #4fc3f7;
    border-radius: 9px;
    margin: -6px 0;
}
QSlider::handle:horizontal:hover {
    background: #29b6f6;
}
QGroupBox {
    border: 1px solid #2d3139;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 18px;
    font-weight: 600;
    color: #b0b6bd;
    font-size: 13px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    background-color: #1a1d23;
}
QTableWidget {
    background-color: #16181d;
    border: 1px solid #2d3139;
    border-radius: 8px;
    gridline-color: #2d3139;
    color: #e8eaed;
    font-size: 12px;
    outline: none;
}
QTableWidget::item { padding: 6px 10px; }
QTableWidget::item:selected { background-color: rgba(79,195,247,0.2); color: #4fc3f7; }
QHeaderView::section {
    background-color: #1e2128;
    color: #9aa0a6;
    border: none;
    border-bottom: 1px solid #2d3139;
    padding: 8px 10px;
    font-weight: 600;
    font-size: 12px;
}
QProgressBar {
    background-color: #16181d;
    border: 1px solid #2d3139;
    border-radius: 6px;
    height: 18px;
    text-align: center;
    color: #e8eaed;
    font-size: 12px;
}
QProgressBar::chunk {
    background-color: #4fc3f7;
    border-radius: 5px;
}
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 4px;
}
QScrollBar::handle:vertical {
    background: #3a3e47;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #5f6368; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 4px;
}
QScrollBar::handle:horizontal {
    background: #3a3e47;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover { background: #5f6368; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QLabel#imageDisplay {
    background-color: #0d0f12;
    border: 1px solid #2d3139;
    border-radius: 10px;
}
QLabel#sectionTitle {
    font-size: 12px;
    font-weight: 600;
    color: #9aa0a6;
    letter-spacing: 1px;
}
QLabel#statNumber {
    font-size: 26px;
    font-weight: 700;
    color: #4fc3f7;
    font-family: "Consolas", "Monaco", monospace;
}
QLabel#statLabel {
    font-size: 12px;
    color: #7a7a7a;
}
QStatusBar {
    background-color: #121417;
    border-top: 1px solid #2d3139;
    color: #9aa0a6;
    font-size: 12px;
}
QTextEdit {
    background-color: #16181d;
    border: 1px solid #2d3139;
    border-radius: 8px;
    color: #e8eaed;
    font-size: 12px;
    padding: 8px;
}
QSpinBox, QDoubleSpinBox {
    background-color: #16181d;
    color: #e8eaed;
    border: 1px solid #3a3e47;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 13px;
}
QSpinBox:focus, QDoubleSpinBox:focus { border-color: #4fc3f7; }
"""


# ==================== 浅色主题 ====================
LIGHT_QSS = """
QMainWindow, QWidget {
    background-color: #f8fafc;
    color: #1e293b;
    font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
    font-size: 13px;
}
QFrame#titleBar {
    background-color: #ffffff;
    border-bottom: 1px solid #e2e8f0;
}
QLabel#titleLabel {
    font-size: 17px;
    font-weight: 700;
    color: #0284c7;
    letter-spacing: 0.5px;
}
QLabel#statusModelLabel, QLabel#statusDeviceLabel {
    color: #64748b;
    font-size: 12px;
}
QPushButton {
    background-color: #ffffff;
    color: #1e293b;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 9px 18px;
    font-size: 13px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #f1f5f9;
    border-color: #0ea5e9;
}
QPushButton:pressed {
    background-color: #e2e8f0;
    border-color: #0284c7;
}
QPushButton#primaryBtn {
    background-color: #0ea5e9;
    color: #ffffff;
    border: none;
    font-weight: 600;
}
QPushButton#primaryBtn:hover {
    background-color: #0284c7;
}
QPushButton#primaryBtn:pressed {
    background-color: #0369a1;
}
QPushButton#primaryBtn:disabled {
    background-color: #cbd5e1;
    color: #f1f5f9;
}
QPushButton#dangerBtn {
    background-color: #ef4444;
    color: #fff;
    border: none;
    font-weight: 600;
}
QPushButton#dangerBtn:hover {
    background-color: #dc2626;
}
QPushButton:disabled {
    background-color: #f1f5f9;
    color: #94a3b8;
    border-color: #e2e8f0;
}
QComboBox {
    background-color: #ffffff;
    color: #1e293b;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 7px 12px;
    min-height: 26px;
    font-size: 13px;
}
QComboBox:hover { border-color: #0ea5e9; }
QComboBox:focus { border-color: #0ea5e9; }
QComboBox::drop-down { border: none; width: 28px; }
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #64748b;
    margin-right: 10px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #1e293b;
    selection-background-color: #0ea5e9;
    selection-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 6px;
    outline: none;
}
QComboBox QAbstractItemView::item {
    padding: 8px 12px;
    border-radius: 6px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #e2e8f0;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #0ea5e9;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 18px;
    height: 18px;
    background: #ffffff;
    border: 2px solid #0ea5e9;
    border-radius: 9px;
    margin: -6px 0;
}
QSlider::handle:horizontal:hover {
    background: #0ea5e9;
}
QGroupBox {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    margin-top: 14px;
    padding-top: 18px;
    font-weight: 600;
    color: #475569;
    font-size: 13px;
    background-color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    background-color: #ffffff;
    color: #0284c7;
}
QTableWidget {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    gridline-color: #f1f5f9;
    color: #1e293b;
    font-size: 12px;
    outline: none;
}
QTableWidget::item { padding: 6px 10px; }
QTableWidget::item:selected { background-color: rgba(14,165,233,0.1); color: #0284c7; }
QHeaderView::section {
    background-color: #f8fafc;
    color: #475569;
    border: none;
    border-bottom: 1px solid #e2e8f0;
    padding: 8px 10px;
    font-weight: 600;
    font-size: 12px;
}
QProgressBar {
    background-color: #e2e8f0;
    border: none;
    border-radius: 8px;
    height: 16px;
    text-align: center;
    color: #475569;
    font-size: 11px;
    font-weight: 500;
}
QProgressBar::chunk {
    background-color: #0ea5e9;
    border-radius: 8px;
}
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 4px;
}
QScrollBar::handle:vertical {
    background: #cbd5e1;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #94a3b8; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 4px;
}
QScrollBar::handle:horizontal {
    background: #cbd5e1;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover { background: #94a3b8; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QLabel#imageDisplay {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}
QLabel#sectionTitle {
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
    letter-spacing: 1px;
}
QLabel#statNumber {
    font-size: 26px;
    font-weight: 700;
    color: #0284c7;
    font-family: "Consolas", "Monaco", monospace;
}
QLabel#statLabel {
    font-size: 12px;
    color: #64748b;
}
QStatusBar {
    background-color: #ffffff;
    border-top: 1px solid #e2e8f0;
    color: #64748b;
    font-size: 12px;
}
QTextEdit {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    color: #1e293b;
    font-size: 12px;
    padding: 8px;
}
QSpinBox, QDoubleSpinBox {
    background-color: #ffffff;
    color: #1e293b;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 4px 8px;
    font-size: 13px;
}
QSpinBox:focus, QDoubleSpinBox:focus { border-color: #0ea5e9; }
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    width: 16px;
    border: none;
    background: transparent;
}
"""


# ==================== 系统主题检测 ====================
def is_system_dark() -> bool:
    """检测 Windows 系统是否为深色主题"""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        )
        value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        winreg.CloseKey(key)
        return value == 0
    except Exception:
        return False  # 默认深色


def get_theme_qss(theme: str) -> str:
    """获取主题样式表
    theme: 'dark', 'light', 'system'
    """
    if theme == "light":
        return LIGHT_QSS
    elif theme == "system":
        return DARK_QSS if is_system_dark() else LIGHT_QSS
    else:  # dark
        return DARK_QSS


# ==================== 带动画的自定义弹窗 ====================
class AnimatedDialog(QDialog):
    """带动画的自定义弹窗
    支持淡入 + 缩放动画
    """
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    QUESTION = "question"

    def __init__(self, parent=None, title="提示", message="", dialog_type=INFO, buttons=None):
        super().__init__(parent)
        self.dialog_type = dialog_type
        self.result_button = None
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)
        self.setMinimumWidth(380)
        self.setMaximumWidth(520)

        # 主容器（带圆角和阴影）
        container = QFrame(self)
        container.setObjectName("dialogContainer")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(16)

        # 标题
        title_label = QLabel(title)
        title_label.setObjectName("dialogTitle")
        title_label.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        title_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(title_label)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setObjectName("dialogLine")
        layout.addWidget(line)

        # 消息内容
        msg_label = QLabel(message)
        msg_label.setObjectName("dialogMessage")
        msg_label.setFont(QFont("Microsoft YaHei", 12))
        msg_label.setWordWrap(True)
        msg_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        msg_label.setMinimumHeight(60)
        layout.addWidget(msg_label)

        # 按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.setSpacing(10)

        if buttons is None:
            buttons = [("确定", True, "primary")]

        for text, value, style in buttons:
            btn = QPushButton(text)
            btn.setObjectName(f"dialogBtn_{style}")
            btn.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
            btn.setCursor(Qt.PointingHandCursor)
            btn.setMinimumWidth(88)
            btn.setMinimumHeight(36)
            btn.clicked.connect(lambda checked, v=value: self._on_button_clicked(v))
            btn_layout.addWidget(btn)

        layout.addLayout(btn_layout)

        # 设置主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.addWidget(container)

        # 应用样式
        self._apply_style()

        # 调整大小
        self.adjustSize()

    def _apply_style(self):
        """根据主题和类型应用样式"""
        is_dark = True
        if self.parent():
            # 从父窗口判断主题
            bg = self.parent().palette().color(self.parent().backgroundRole())
            is_dark = bg.lightness() < 128

        if is_dark:
            container_bg = "#1e2128"
            border_color = "#3a3e47"
            title_color = "#e8eaed"
            msg_color = "#b0b6bd"
            line_color = "#2d3139"
        else:
            container_bg = "#ffffff"
            border_color = "#e0e4e8"
            title_color = "#1a1d23"
            msg_color = "#5f6368"
            line_color = "#f0f4f8"

        # 类型颜色
        type_colors = {
            self.INFO: "#4fc3f7",
            self.SUCCESS: "#66bb6a",
            self.WARNING: "#ffa726",
            self.ERROR: "#ef5350",
            self.QUESTION: "#ab47bc",
        }
        accent = type_colors.get(self.dialog_type, "#4fc3f7")

        self.setStyleSheet(f"""
            QFrame#dialogContainer {{
                background-color: {container_bg};
                border: 1px solid {border_color};
                border-radius: 14px;
            }}
            QLabel#dialogTitle {{
                color: {title_color};
            }}
            QFrame#dialogLine {{
                color: {line_color};
                background-color: {line_color};
                max-height: 1px;
            }}
            QLabel#dialogMessage {{
                color: {msg_color};
                line-height: 1.6;
            }}
            QPushButton#dialogBtn_primary {{
                background-color: {accent};
                color: {'#0a0a0a' if is_dark else '#ffffff'};
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 13px;
            }}
            QPushButton#dialogBtn_primary:hover {{
                background-color: {accent};
                opacity: 0.85;
            }}
            QPushButton#dialogBtn_normal {{
                background-color: {'#252830' if is_dark else '#f5f7fa'};
                color: {title_color};
                border: 1px solid {border_color};
                border-radius: 8px;
                font-size: 13px;
            }}
            QPushButton#dialogBtn_normal:hover {{
                border-color: {accent};
            }}
        """)

    def _on_button_clicked(self, value):
        self.result_button = value
        # 淡出动画：150ms淡出后再关闭，体验更丝滑
        self._fade_out_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_out_anim.setDuration(150)
        self._fade_out_anim.setStartValue(1.0)
        self._fade_out_anim.setEndValue(0.0)
        self._fade_out_anim.setEasingCurve(QEasingCurve.InCubic)
        self._fade_out_anim.finished.connect(self.accept)
        self._fade_out_anim.start()

    def _animate_in(self):
        """入场动画：淡入（只用 windowOpacity，不用 GraphicsEffect，避免兼容性问题）"""
        self.setWindowOpacity(0.0)
        self._opacity_anim = QPropertyAnimation(self, b"windowOpacity")
        self._opacity_anim.setDuration(180)
        self._opacity_anim.setStartValue(0.0)
        self._opacity_anim.setEndValue(1.0)
        self._opacity_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._opacity_anim.start()

    def showEvent(self, event):
        super().showEvent(event)
        # 居中显示
        if self.parent():
            parent_rect = self.parent().geometry()
            x = parent_rect.center().x() - self.width() // 2
            y = parent_rect.center().y() - self.height() // 2
            self.move(x, y)
        self._animate_in()

    @staticmethod
    def information(parent, title, message):
        dlg = AnimatedDialog(parent, title, message, AnimatedDialog.INFO)
        dlg.exec_()
        return dlg.result_button

    @staticmethod
    def success(parent, title, message):
        dlg = AnimatedDialog(parent, title, message, AnimatedDialog.SUCCESS)
        dlg.exec_()
        return dlg.result_button

    @staticmethod
    def warning(parent, title, message):
        dlg = AnimatedDialog(parent, title, message, AnimatedDialog.WARNING)
        dlg.exec_()
        return dlg.result_button

    @staticmethod
    def critical(parent, title, message):
        dlg = AnimatedDialog(parent, title, message, AnimatedDialog.ERROR)
        dlg.exec_()
        return dlg.result_button

    @staticmethod
    def question(parent, title, message):
        buttons = [
            ("取消", False, "normal"),
            ("确定", True, "primary"),
        ]
        dlg = AnimatedDialog(parent, title, message, AnimatedDialog.QUESTION, buttons)
        dlg.exec_()
        return dlg.result_button


class NoticeDialog(QDialog):
    """首次启动公告弹窗：带倒计时，3秒后才能关闭，只弹一次。跟随主题，中文字体，淡入动画。"""

    def __init__(self, parent=None, title="使用须知", message="", countdown=3):
        super().__init__(parent)
        self.countdown = countdown
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setMaximumWidth(580)

        # 判断主题（从父窗口背景色推断）
        self.is_dark = True
        if self.parent():
            try:
                bg = self.parent().palette().color(self.parent().backgroundRole())
                self.is_dark = bg.lightness() < 128
            except Exception:
                pass

        if self.is_dark:
            self._colors = {
                "bg": "#1e2128", "border": "#3a3e47", "title": "#e8eaed",
                "msg": "#b0b6bd", "line": "#2d3139", "accent": "#4fc3f7",
                "btn_text": "#0a0a0a", "btn_disabled_bg": "#3a3e47", "btn_disabled_text": "#7a7a7a",
            }
        else:
            self._colors = {
                "bg": "#ffffff", "border": "#e0e4e8", "title": "#1a1d23",
                "msg": "#3c4043", "line": "#f0f4f8", "accent": "#0288d1",
                "btn_text": "#ffffff", "btn_disabled_bg": "#e0e4e8", "btn_disabled_text": "#9aa0a6",
            }
        c = self._colors

        # 主容器
        container = QFrame(self)
        container.setObjectName("dialogContainer")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(14)

        # 标题
        title_label = QLabel(title)
        title_label.setObjectName("dialogTitle")
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title_label.setStyleSheet(f"color: {c['title']};")
        layout.addWidget(title_label)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background-color: {c['line']}; max-height: 1px; border: none;")
        layout.addWidget(line)

        # 消息内容
        msg_label = QLabel(message)
        msg_label.setObjectName("dialogMessage")
        msg_label.setFont(QFont("Microsoft YaHei", 12))
        msg_label.setWordWrap(True)
        msg_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        msg_label.setStyleSheet(f"color: {c['msg']}; line-height: 1.7;")
        msg_label.setMinimumHeight(160)
        layout.addWidget(msg_label)

        # 按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.confirm_btn = QPushButton(f"我知道了 ({countdown}s)")
        self.confirm_btn.setObjectName("dialogBtn_primary")
        self.confirm_btn.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.confirm_btn.setMinimumWidth(150)
        self.confirm_btn.setMinimumHeight(40)
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.setCursor(Qt.ForbiddenCursor)
        self.confirm_btn.clicked.connect(self._on_confirm)
        btn_layout.addWidget(self.confirm_btn)
        layout.addLayout(btn_layout)

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.addWidget(container)

        # 样式
        container.setStyleSheet(f"""
            QFrame#dialogContainer {{
                background-color: {c['bg']};
                border: 1px solid {c['border']};
                border-radius: 14px;
            }}
            QPushButton#dialogBtn_primary {{
                background-color: {c['accent']};
                color: {c['btn_text']};
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 13px;
            }}
            QPushButton#dialogBtn_primary:hover {{
                background-color: {c['accent']};
            }}
            QPushButton#dialogBtn_primary:disabled {{
                background-color: {c['btn_disabled_bg']};
                color: {c['btn_disabled_text']};
            }}
        """)

        self.adjustSize()

        # 倒计时
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)

        # 入场动画
        self._animate_in()

    def _animate_in(self):
        """淡入动画"""
        self.setWindowOpacity(0.0)
        self._anim = QPropertyAnimation(self, b"windowOpacity", self)
        self._anim.setDuration(200)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.start()

    def _tick(self):
        self.countdown -= 1
        if self.countdown > 0:
            self.confirm_btn.setText(f"我知道了 ({self.countdown}s)")
        else:
            self._timer.stop()
            self.confirm_btn.setText("我知道了")
            self.confirm_btn.setEnabled(True)
            self.confirm_btn.setCursor(Qt.PointingHandCursor)

    def _on_confirm(self):
        self.accept()

    def showEvent(self, event):
        super().showEvent(event)
        # 居中
        if self.parent():
            parent_rect = self.parent().geometry()
            x = parent_rect.center().x() - self.width() // 2
            y = parent_rect.center().y() - self.height() // 2
            self.move(x, y)
        # 淡入动画
        self.setWindowOpacity(0.0)
        self._opacity_anim = QPropertyAnimation(self, b"windowOpacity")
        self._opacity_anim.setDuration(200)
        self._opacity_anim.setStartValue(0.0)
        self._opacity_anim.setEndValue(1.0)
        self._opacity_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._opacity_anim.start()

    def keyPressEvent(self, event):
        # 倒计时期间禁止按回车/ESC关闭
        if self.countdown > 0:
            return
        super().keyPressEvent(event)
