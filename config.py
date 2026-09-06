# -*- coding: utf-8 -*-
"""
配置管理模块
自动保存和加载用户配置：模型选择、检测参数、窗口状态、最近文件等
配置文件位于程序目录下的 config.json
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _get_app_dir():
    """获取程序根目录（开发时是脚本目录，打包后是exe所在目录）"""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(__file__).parent


class Config:
    """用户配置管理"""

    # 默认配置
    DEFAULT_CONFIG = {
        "version": "1.0.0",
        "model": {
            "current_model": "",           # 当前使用的模型文件名
            "model_dir": "",               # 模型目录（空则用程序目录下的 models）
            "device": "auto",              # 计算设备 auto/cpu/cuda:0
            "conf_thres": 0.25,            # 置信度阈值
            "iou_thres": 0.45,             # IOU 阈值
            "line_thickness": 2,           # 边界框线条粗细
        },
        "ui": {
            "window_width": 1400,
            "window_height": 850,
            "window_x": None,              # 窗口位置 x
            "window_y": None,              # 窗口位置 y
            "current_mode": 0,             # 当前检测模式索引 0=图片 1=视频 2=摄像头
            "theme": "light",
            "language": "",                # 界面语言（空=跟随系统）
            "video_speed": 0,              # 视频处理速度 0=高质量 1=标准 2=快速
            "camera_res": 0,               # 摄像头分辨率 0=640x480 1=800x600 2=1280x720
        },
        "recent": {
            "last_image": "",              # 最近打开的图片
            "last_video": "",              # 最近打开的视频
            "output_dir": "",              # 输出目录
        },
        "cache": {
            "torch_home": "",              # PyTorch 缓存目录（空=项目目录下的 torch_cache）
            "auto_clean": False,           # 退出时自动清理缓存
        },
        "first_run": True,                 # 是否首次运行
    }

    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = _get_app_dir() / "config.json"
        self._data: Dict[str, Any] = {}
        self.load()

    def load(self):
        """加载配置文件，如果不存在则创建默认配置"""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                # 合并默认配置（确保新增字段有默认值）
                self._data = self._deep_merge(self.DEFAULT_CONFIG, self._data)
            except Exception as e:
                print(f"[Config] 配置文件读取失败，使用默认配置: {e}")
                self._data = self._deep_copy(self.DEFAULT_CONFIG)
        else:
            self._data = self._deep_copy(self.DEFAULT_CONFIG)
            self.save()

    def save(self):
        """保存配置到文件（原子性写入：先写临时文件，再替换，避免中断导致文件损坏）"""
        try:
            import os
            tmp_path = str(self.config_path) + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.config_path)
        except Exception as e:
            print(f"[Config] 配置保存失败: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点分隔的嵌套键
        例如: config.get("model.conf_thres")
        """
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any, auto_save: bool = True):
        """
        设置配置值，支持点分隔的嵌套键
        例如: config.set("model.conf_thres", 0.5)
        """
        keys = key.split(".")
        data = self._data
        for k in keys[:-1]:
            if k not in data or not isinstance(data[k], dict):
                data[k] = {}
            data = data[k]
        data[keys[-1]] = value
        if auto_save:
            self.save()

    def reset(self):
        """重置为默认配置"""
        self._data = self._deep_copy(self.DEFAULT_CONFIG)
        self.save()

    @property
    def data(self) -> Dict:
        return self._data

    @staticmethod
    def _deep_merge(default: Dict, override: Dict) -> Dict:
        """深度合并两个字典，override 覆盖 default"""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _deep_copy(data: Dict) -> Dict:
        """深拷贝字典"""
        return json.loads(json.dumps(data))


# 全局配置单例
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置单例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
