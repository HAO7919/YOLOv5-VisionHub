# -*- coding: utf-8 -*-
"""
YOLO 目标检测封装模块（基于 ultralytics，统一支持 YOLOv5/v8/v9/v10/v11 等版本）
支持图片、视频、实时流检测，可动态调整置信度、IOU 阈值等参数
支持模型切换、结果导出（JSON/CSV/VOC XML）
模型全部本地加载，不依赖联网下载框架代码，换电脑也能用
"""
import os
import sys
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from xml.etree.ElementTree import Element, SubElement, ElementTree
from xml.dom import minidom

# 获取程序根目录（开发时是脚本目录，打包后是exe所在目录）
def _get_app_dir():
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(__file__).parent

APP_DIR = _get_app_dir()

# 日志文件：打包后也能看到错误信息（放在用户目录，避免Program Files权限问题）
import logging
from logging.handlers import RotatingFileHandler
_log_dir = Path.home() / ".yolov5_desktop"
_log_dir.mkdir(parents=True, exist_ok=True)
_log_file = _log_dir / "yolov5_app.log"
_logger = logging.getLogger("yolov5")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _fh = RotatingFileHandler(str(_log_file), maxBytes=2*1024*1024, backupCount=3, encoding='utf-8')
    _fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    _logger.addHandler(_fh)
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    _logger.addHandler(_sh)

# 保存最后一次推理错误，让界面可以显示
_last_detect_error = ""

# 设置 PyTorch 缓存目录（默认放在程序目录下，避免占用 C 盘）
# 用户可通过 config.json 中的 cache.torch_home 自定义
_DEFAULT_CACHE_DIR = str(APP_DIR / "torch_cache")
os.environ.setdefault("TORCH_HOME", _DEFAULT_CACHE_DIR)
# 关闭 ultralytics 多余的联网检查和提示，让离线环境更干净，避免加载时卡住
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("ULTRALYTICS_DISABLE_UPDATE_CHECK", "True")
os.environ.setdefault("ULTRALYTICS_NO_CHECKS", "True")

import torch
from ultralytics import YOLO

# 禁用 ultralytics 全局的更新检查和联网提示
try:
    from ultralytics import settings as ultralytics_settings
    ultralytics_settings.update({"verbose": False})
except Exception:
    pass


class YOLOv5Detector:
    """YOLOv5 检测器封装"""

    DEFAULT_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
        (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
        (0, 64, 128), (128, 64, 128), (0, 192, 128), (128, 192, 128),
        (64, 64, 0), (192, 64, 0), (64, 192, 0), (192, 192, 0),
        (64, 64, 128), (192, 64, 128), (64, 192, 128), (192, 192, 128),
        (0, 0, 64), (128, 0, 64), (0, 128, 64), (128, 128, 64),
        (0, 0, 192), (128, 0, 192), (0, 128, 192), (128, 128, 192),
        (64, 0, 64), (192, 0, 64), (64, 128, 64), (192, 128, 64),
        (64, 0, 192), (192, 0, 192), (64, 128, 192), (192, 128, 192),
        (0, 64, 64), (128, 64, 64), (0, 192, 64), (128, 192, 64),
        (0, 64, 192), (128, 64, 192), (0, 192, 192), (128, 192, 192),
        (64, 64, 64), (192, 64, 64), (64, 192, 64), (192, 192, 64),
        (64, 64, 192), (192, 64, 192), (64, 192, 192), (192, 192, 192),
        (32, 0, 0), (160, 0, 0), (32, 128, 0), (160, 128, 0),
        (32, 0, 128), (160, 0, 128), (32, 128, 128), (160, 128, 128),
    ]

    def __init__(self, model_dir: str = "models", device: str = "auto"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._select_device(device)
        self.model = None
        self.model_type = "ultralytics"  # "ultralytics" 或 "torchhub"（旧版yolov5兼容）
        self.model_name = ""
        self.model_path = ""  # 当前模型文件路径
        self.classes = []
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.line_thickness = 2
        self._load_model()

    def _select_device(self, device: str) -> str:
        """返回 ultralytics 兼容的设备字符串：'0'（GPU）或 'cpu'"""
        if device == "auto":
            if torch.cuda.is_available():
                return "0"
            return "cpu"
        device_str = str(device).lower()
        if "cuda" in device_str:
            return "0"
        return device_str

    def _infer_device(self):
        """推理时传给 ultralytics 的 device：GPU 用 int 0，CPU 用 'cpu'"""
        try:
            return int(self.device)
        except (ValueError, TypeError):
            return self.device

    def _find_custom_weights(self) -> Optional[Path]:
        pt_files = list(self.model_dir.glob("*.pt"))
        if pt_files:
            # 优先选择 yolov5s/yolov8n 等小模型（加载快），其次 best，最后最小的
            small = [f for f in pt_files if any(k in f.name.lower() for k in
                    ["yolov5s", "yolov8n", "yolo11n", "yolov5su", "yolov8n.pt"])]
            if small:
                return small[0]
            best = [f for f in pt_files if "best" in f.name.lower()]
            if best:
                return best[0]
            return min(pt_files, key=lambda f: f.stat().st_size)
        return None

    def _extract_classes(self, model):
        """从模型中提取类别名称列表（兼容 dict/list 两种格式）"""
        names = getattr(model, "names", None)
        if names is None:
            return []
        if isinstance(names, dict):
            try:
                return [names[i] for i in range(len(names))]
            except Exception:
                return list(names.values())
        return list(names)

    def _load_model(self):
        """初始加载：优先本地模型文件，全部本地加载不依赖联网下载框架代码"""
        custom_weights = self._find_custom_weights()
        try:
            if custom_weights:
                self.model = YOLO(str(custom_weights))
                self.model_path = str(custom_weights)
                self.model_name = custom_weights.name
            else:
                # models 目录没有模型，尝试内置默认模型（首次需联网下载，离线则提示）
                self.model = YOLO("yolov5su.pt")
                self.model_name = "yolov5su.pt (默认)"
            self.classes = self._extract_classes(self.model)
            _logger.info(f"模型加载成功: {self.model_name}, 设备: {self.device}, 类别数: {len(self.classes)}, 类型: {self.model_type}")
            print(f"[Detector] 模型加载成功: {self.model_name}, 设备: {self.device}, 类别数: {len(self.classes)}")
        except Exception as e:
            _logger.error(f"模型加载失败: {e}", exc_info=True)
            print(f"[Detector] 模型加载失败: {e}")
            self.model = None
            self.model_name = "加载失败"

    def is_ready(self) -> bool:
        return self.model is not None

    def set_params(self, conf_thres=None, iou_thres=None, max_det=None, line_thickness=None):
        # ultralytics 在推理时传参，这里只保存数值，不直接改模型
        if conf_thres is not None:
            self.conf_thres = max(0.0, min(1.0, conf_thres))
        if iou_thres is not None:
            self.iou_thres = max(0.0, min(1.0, iou_thres))
        if max_det is not None:
            self.max_det = max(1, max_det)
        if line_thickness is not None:
            self.line_thickness = max(1, line_thickness)

    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        return self.DEFAULT_COLORS[class_id % len(self.DEFAULT_COLORS)]

    def list_models(self) -> List[Dict]:
        models = []
        for f in sorted(self.model_dir.glob("*.pt")):
            # 精确匹配当前模型文件名
            is_current = (self.model_name == f.name) or (self.model_path == str(f.resolve()))
            models.append({
                "name": f.name,
                "path": str(f),
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                "is_current": is_current,
            })
        if not models:
            models.append({
                "name": "yolov5su.pt (默认)",
                "path": "",
                "size_mb": 0,
                "is_current": ("yolov5" in self.model_name),
            })
        return models

    def load_model_by_name(self, model_name: str) -> Tuple[bool, str]:
        """加载模型（本地加载，兼容 YOLOv5/v8/v9/v10/v11），返回 (是否成功, 错误信息)"""
        try:
            # 默认模型的情况
            if model_name.lower() in ("yolov5s", "default", "默认", "yolov5s (默认)",
                                      "yolov5su.pt (默认)", "yolov5su.pt"):
                local = self._find_custom_weights()
                if local:
                    new_model = YOLO(str(local))
                    self.model_path = str(local)
                    self.model_name = local.name
                else:
                    # 本地没有，尝试联网获取默认模型
                    try:
                        new_model = YOLO("yolov5su.pt")
                        self.model_name = "yolov5su.pt (默认)"
                    except Exception:
                        return False, ("本地 models 文件夹没有模型，且无法联网下载默认模型。\n\n"
                                       "请把你自己的 .pt 模型文件通过「选择模型文件」导入，"
                                       "或在有网络时启动一次让软件自动下载默认模型。")
            else:
                model_path = self.model_dir / model_name
                if not model_path.exists():
                    return False, f"模型文件不存在: {model_path}"
                # 验证文件大小
                file_size = model_path.stat().st_size
                if file_size < 1000:
                    return False, f"模型文件太小（{file_size}字节），可能已损坏或没下载完"
                # 本地加载（ultralytics 兼容 v5/v8/v9/v10/v11，不需要联网下载框架代码）
                new_model = None
                try:
                    new_model = YOLO(str(model_path))
                    self.model_type = "ultralytics"
                except Exception as load_err:
                    err = str(load_err)
                    # 旧版 yolov5 模型兼容性错误：ultralytics 不支持，fallback 到 torch.hub.load
                    if "forwards compatible" in err.lower() or "not forwards" in err.lower():
                        print(f"[Detector] ultralytics 不兼容旧版模型，尝试直接加载: {err[:80]}")
                        # 方案1：torch.load 直接加载 checkpoint 里的模型对象（完全离线）
                        # 旧版 yolov5 的 checkpoint 引用了 models.yolo 模块，需要把 yolov5 代码目录加入 sys.path
                        try:
                            import sys as _sys
                            yolov5_code_dir = None
                            # 优先在程序目录的 torch_cache 里找（打包后也能用）
                            for candidate in [
                                APP_DIR / "torch_cache" / "hub" / "ultralytics_yolov5_master",
                                Path(_DEFAULT_CACHE_DIR) / "hub" / "ultralytics_yolov5_master",
                                Path.home() / ".cache" / "torch" / "hub" / "ultralytics_yolov5_master",
                            ]:
                                if (candidate / "models" / "yolo.py").exists():
                                    yolov5_code_dir = str(candidate)
                                    break
                            if yolov5_code_dir and yolov5_code_dir not in _sys.path:
                                _sys.path.insert(0, yolov5_code_dir)
                                # 清除 ultralytics 的 models 模块缓存，避免和旧版 yolov5 的 models 命名冲突
                                for _key in list(_sys.modules.keys()):
                                    if _key == "models" or _key.startswith("models."):
                                        del _sys.modules[_key]
                                print(f"[Detector] 旧版 yolov5 代码目录: {yolov5_code_dir}")
                            ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
                            legacy_model = ckpt.get("model", None)
                            if legacy_model is None:
                                raise ValueError("checkpoint 里没有 model 字段")
                            legacy_model.eval()
                            legacy_model = legacy_model.float()
                            # 加上 AutoShape 包装，让模型能接受 numpy 输入（和 torch.hub.load 返回的接口一致）
                            try:
                                from models.common import AutoShape
                                legacy_model = AutoShape(legacy_model)
                            except Exception:
                                pass
                            # 移到目标设备
                            dev = "cuda:0" if self.device == "0" else "cpu"
                            legacy_model = legacy_model.to(dev)
                            new_model = legacy_model
                            self.model_type = "legacy"
                            print("[Detector] torch.load 直接加载旧版模型成功（完全离线）")
                        except Exception as direct_err:
                            print(f"[Detector] 直接加载失败，尝试 torch.hub.load: {str(direct_err)[:80]}")
                            # 方案2：torch.hub.load（需要联网或有缓存）
                            try:
                                new_model = torch.hub.load(
                                    "ultralytics/yolov5", "custom",
                                    path=str(model_path), device=self.device, trust_repo=True,
                                )
                                self.model_type = "legacy"
                                print("[Detector] torch.hub.load 加载旧版模型成功")
                            except Exception as hub_err:
                                return False, (
                                    f"这是旧版 YOLOv5 训练的模型，新版 ultralytics 不兼容。\n\n"
                                    f"直接加载和旧版方式都失败了。\n"
                                    f"直接加载错误: {str(direct_err)[:80]}\n"
                                    f"旧版加载错误: {str(hub_err)[:80]}\n\n"
                                    f"解决方案：\n"
                                    f"1. 用新版 ultralytics 重新训练模型（推荐，一劳永逸）\n"
                                    f"2. 或在旧版 yolov5 里用 'python export.py --weights best.pt --include torchscript' 转换模型\n"
                                    f"3. 或确保有网络连接让软件下载旧版 YOLOv5 代码"
                                )
                    elif "unpickle" in err.lower() or "pickle" in err.lower():
                        return False, ("模型文件读取失败，可能已损坏，或不是 ultralytics/PyTorch 模型。\n\n"
                                       "请确认是用 YOLOv5/YOLOv8 正常训练或导出的 .pt 文件。")
                    elif "cuda" in err.lower() or "out of memory" in err.lower():
                        # GPU 加载失败，自动回退 CPU 试一次
                        try:
                            self.device = "cpu"
                            new_model = YOLO(str(model_path))
                            self.model_type = "ultralytics"
                        except Exception as e2:
                            return False, f"GPU 和 CPU 都无法加载模型: {e2}"
                    else:
                        return False, (f"模型加载失败: {err}\n\n"
                                       "请确认这是 YOLOv5/YOLOv8 等 ultralytics 系列的 .pt 模型文件。")
                self.model_path = str(model_path)
                self.model_name = model_name
            # 加载成功，替换旧模型并提取类别
            old = self.model
            self.model = new_model
            self.classes = self._extract_classes(new_model)
            if not self.classes:
                # 类别为空也要提示，但不阻断
                print("[Detector] 警告: 模型未包含类别名称")
            del old  # 释放旧模型显存/内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[Detector] 模型切换成功: {self.model_name}, 类别数: {len(self.classes)}")
            return True, ""
        except Exception as e:
            error_msg = str(e)
            print(f"[Detector] 模型切换失败: {error_msg}")
            if "cuda" in error_msg.lower():
                return False, f"GPU 加载失败，可尝试用 CPU: {error_msg}"
            if "no such file" in error_msg.lower() or "not found" in error_msg.lower():
                return False, f"模型文件路径有问题: {error_msg}"
            return False, f"模型加载失败: {error_msg}"

    def _draw_detections(self, img, raw_detections):
        """把原始检测结果画到图上，返回标注图和统一格式的detections列表
        raw_detections: list of (x1, y1, x2, y2, conf, cls_id)
        """
        annotated = img.copy()
        detections = []
        for x1, y1, x2, y2, conf_val, cls_id in raw_detections:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_name = self.classes[int(cls_id)] if int(cls_id) < len(self.classes) else str(int(cls_id))
            color = self.get_color(int(cls_id))
            thickness = max(1, self.line_thickness)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            label = f"{class_name} {float(conf_val):.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            detections.append({
                "class_id": int(cls_id),
                "class_name": class_name,
                "confidence": round(float(conf_val), 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "width": x2 - x1,
                "height": y2 - y1,
                "color": color,
            })
        return annotated, detections

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        global _last_detect_error
        if self.model is None:
            return img, []
        try:
            return self._detect_internal(img)
        except Exception as e:
            err_str = str(e).lower()
            # GPU显存不足（CUDA OOM）时，自动回退CPU重试一次
            if ("out of memory" in err_str or "cuda" in err_str) and self.device != "cpu":
                _logger.warning(f"GPU显存不足，自动回退CPU重试: {e}")
                print(f"[Detector] GPU显存不足，自动回退CPU重试")
                original_device = self.device
                self.device = "cpu"
                try:
                    result = self._detect_internal(img)
                    return result
                except Exception as e2:
                    _last_detect_error = str(e2)
                    _logger.error(f"CPU回退后推理仍失败: {e2}", exc_info=True)
                    print(f"[Detector] CPU回退后推理仍失败: {e2}")
                    return img, []
                finally:
                    self.device = original_device
            _last_detect_error = str(e)
            _logger.error(f"单帧推理异常: {e}", exc_info=True)
            print(f"[Detector] 单帧推理异常: {e}")
            return img, []

    def _detect_internal(self, img: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """实际推理逻辑（被detect包装，支持GPU OOM自动回退）"""
        raw_detections = []
        if self.model_type == "ultralytics":
            # ultralytics 接口：直接接受 BGR 图
            results = self.model(
                img, conf=self.conf_thres, iou=self.iou_thres,
                max_det=self.max_det, device=self._infer_device(), verbose=False,
            )
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy_list = boxes.xyxy.detach().cpu().numpy()
                conf_list = boxes.conf.detach().cpu().numpy()
                cls_list = boxes.cls.detach().cpu().numpy().astype(int)
                for (x1, y1, x2, y2), conf_val, cls_id in zip(xyxy_list, conf_list, cls_list):
                    raw_detections.append((x1, y1, x2, y2, float(conf_val), int(cls_id)))
        else:
            # 旧版 yolov5 接口（torch.hub.load 或 torch.load 直接加载）：需要 RGB 图，参数在模型上设置
            self.model.conf = self.conf_thres
            self.model.iou = self.iou_thres
            self.model.max_det = self.max_det
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.model(img_rgb)
            if results.pred and len(results.pred) > 0:
                for *xyxy, conf, cls in results.pred[0]:
                    raw_detections.append((float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]),
                                           float(conf), int(cls)))
        return self._draw_detections(img, raw_detections)

    def get_last_error(self) -> str:
        """获取最后一次推理错误信息"""
        return _last_detect_error

    def detect_image_file(self, file_path: str, output_path: Optional[str] = None) -> Dict:
        img = cv2.imread(file_path)
        if img is None:
            _logger.error(f"无法读取图片: {file_path}")
            raise ValueError(f"无法读取图片: {file_path}")
        _logger.info(f"开始检测图片: {file_path}, 尺寸: {img.shape}, 模型: {self.model_name}, 设备: {self.device}, 置信度: {self.conf_thres}")
        annotated, detections = self.detect(img)
        _logger.info(f"检测完成: {len(detections)} 个目标")
        for d in detections[:10]:
            _logger.info(f"  - {d['class_name']} {d['confidence']:.2f}")
        if output_path:
            cv2.imwrite(output_path, annotated)
        stats = {}
        for d in detections:
            stats[d["class_name"]] = stats.get(d["class_name"], 0) + 1
        return {
            "image_width": img.shape[1],
            "image_height": img.shape[0],
            "detection_count": len(detections),
            "detections": detections,
            "statistics": stats,
            "annotated": annotated,
        }

    @staticmethod
    def _test_encoder(fourcc_str: str, ext: str, fps: float, size: tuple, test_dir: Path) -> bool:
        """测试编码器能否正常写出并读回（写多帧测试文件，严格验证文件大小和可读帧数）"""
        test_path = test_dir / f"_encoder_test{ext}"
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(str(test_path), fourcc, fps, size)
            if not writer.isOpened():
                return False
            # 写5帧测试（帧太少可能文件头正常但实际没数据）
            test_frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            cv2.rectangle(test_frame, (10, 10), (size[0]-10, size[1]-10), (0, 255, 0), 2)
            cv2.putText(test_frame, "TEST", (size[0]//4, size[1]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            for _ in range(5):
                writer.write(test_frame)
            writer.release()
            # 严格检查1：文件存在且大小合理（5帧至少应 > 1KB，太小说明写帧静默失败）
            if not test_path.exists() or test_path.stat().st_size < 1024:
                return False
            # 严格检查2：能打开并读回至少3帧（5帧里至少能读到3帧才算正常）
            cap = cv2.VideoCapture(str(test_path))
            if not cap.isOpened():
                return False
            read_count = 0
            for _ in range(10):
                ret, _ = cap.read()
                if not ret:
                    break
                read_count += 1
            cap.release()
            return read_count >= 3
        except Exception:
            return False
        finally:
            try:
                if test_path.exists():
                    test_path.unlink()
            except Exception:
                pass

    def detect_video_file(self, input_path: str, output_path: str, progress_callback=None, skip_frames: int = 0) -> Dict:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件，请检查格式是否支持（支持 mp4/avi/mov/mkv/flv）: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 240:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError(f"无法获取视频尺寸，文件可能已损坏: {input_path}")

        # 确保输出目录存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 编码器候选：(fourcc, 扩展名)，按通用性排序
        # avc1=H.264(最通用，B站/手机/所有播放器都支持)，XVID=.avi(兼容性第二)，mp4v=fallback
        encoder_candidates = [
            ("avc1", ".mp4"),
            ("XVID", ".avi"),
            ("mp4v", ".mp4"),
        ]

        # 逐个测试编码器，选第一个能正常写出并读回的
        selected_fourcc = None
        selected_ext = None
        for fourcc_str, ext in encoder_candidates:
            if self._test_encoder(fourcc_str, ext, fps, (width, height), output_dir):
                selected_fourcc = fourcc_str
                selected_ext = ext
                break

        if selected_fourcc is None:
            cap.release()
            raise RuntimeError("所有视频编码器都不可用，无法输出视频。请检查系统是否安装了视频编码器。")

        # 根据选定编码器调整输出扩展名
        output_path = str(Path(output_path).with_suffix(selected_ext))
        print(f"[Detector] 视频编码器: {selected_fourcc}, 输出: {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*selected_fourcc)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"无法创建视频输出文件: {output_path}")

        all_stats = {}
        frame_count = 0
        total_detections = 0
        last_detections = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is None:
                    break
                frame_count += 1
                # 跳帧逻辑：skip_frames=0时逐帧，否则每隔skip_frames+1帧检测一次
                need_detect = (frame_count - 1) % (skip_frames + 1) == 0
                if need_detect or not last_detections:
                    annotated, detections = self.detect(frame)
                    last_detections = detections
                else:
                    # 复用上次检测结果，在当前帧上绘制
                    annotated = frame.copy()
                    for d in last_detections:
                        bbox = d["bbox"]
                        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), d["color"], 2)
                        label = f"{d['class_name']} {d['confidence']:.2f}"
                        cv2.putText(annotated, label, (x1, max(y1 - 8, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, d["color"], 1)
                    detections = last_detections
                out.write(annotated)
                for d in detections:
                    all_stats[d["class_name"]] = all_stats.get(d["class_name"], 0) + 1
                total_detections += len(detections)
                if progress_callback and frame_count % 5 == 0:
                    progress_callback(frame_count, total_frames)
        except Exception as e:
            cap.release()
            out.release()
            raise RuntimeError(f"视频处理第 {frame_count} 帧时出错: {str(e)}")

        cap.release()
        out.release()

        # 最终验证：输出文件能被正常打开读取
        verify_cap = cv2.VideoCapture(output_path)
        verify_ok = verify_cap.isOpened()
        if verify_ok:
            ret, _ = verify_cap.read()
            verify_ok = ret
        verify_cap.release()
        if not verify_ok:
            # 验证失败但文件存在，不抛异常（可能是个别编码器的边缘情况），给警告
            print(f"[Detector] 警告: 输出视频验证读取失败，文件可能有问题: {output_path}")

        if progress_callback:
            progress_callback(frame_count, total_frames)
        if frame_count == 0:
            raise ValueError("视频没有读取到任何帧，文件可能已损坏或格式不支持")
        return {
            "fps": fps, "width": width, "height": height,
            "total_frames": frame_count, "total_detections": total_detections,
            "statistics": all_stats, "skip_frames": skip_frames,
            "output_path": output_path,
            "encoder": selected_fourcc,
        }

    # ==================== 结果导出 ====================
    @staticmethod
    def export_to_json(detections, image_info, output_path):
        data = {"image": image_info, "detection_count": len(detections), "detections": detections}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def export_to_csv(detections, image_info, output_path):
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["图像文件", "宽度", "高度", "序号", "类别ID", "类别名称",
                             "置信度", "x1", "y1", "x2", "y2", "宽度", "高度"])
            for i, d in enumerate(detections, 1):
                bbox = d["bbox"]
                writer.writerow([
                    image_info.get("filename", ""), image_info.get("width", ""),
                    image_info.get("height", ""), i, d["class_id"], d["class_name"],
                    f"{d['confidence']:.4f}", bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"],
                    d["width"], d["height"],
                ])

    @staticmethod
    def export_to_voc_xml(detections, image_info, output_path):
        annotation = Element("annotation")
        folder = SubElement(annotation, "folder")
        folder.text = image_info.get("folder", "images")
        filename = SubElement(annotation, "filename")
        filename.text = image_info.get("filename", "image.jpg")
        path = SubElement(annotation, "path")
        path.text = image_info.get("path", "")
        source = SubElement(annotation, "source")
        database = SubElement(source, "database")
        database.text = "YOLOv5 Detection"
        size = SubElement(annotation, "size")
        w = SubElement(size, "width"); w.text = str(image_info.get("width", 0))
        h = SubElement(size, "height"); h.text = str(image_info.get("height", 0))
        d = SubElement(size, "depth"); d.text = "3"
        seg = SubElement(annotation, "segmented"); seg.text = "0"
        for det in detections:
            obj = SubElement(annotation, "object")
            name = SubElement(obj, "name"); name.text = det["class_name"]
            pose = SubElement(obj, "pose"); pose.text = "Unspecified"
            tr = SubElement(obj, "truncated"); tr.text = "0"
            diff = SubElement(obj, "difficult"); diff.text = "0"
            bndbox = SubElement(obj, "bndbox")
            bbox = det["bbox"]
            SubElement(bndbox, "xmin").text = str(bbox["x1"])
            SubElement(bndbox, "ymin").text = str(bbox["y1"])
            SubElement(bndbox, "xmax").text = str(bbox["x2"])
            SubElement(bndbox, "ymax").text = str(bbox["y2"])
        rough = ElementTree(annotation)
        rough.write(output_path, encoding="utf-8", xml_declaration=True)
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            dom = minidom.parseString(content)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(dom.toprettyxml(indent="  "))
        except Exception:
            pass


# 全局单例
_detector_instance = None

def get_detector(model_dir=None, device="auto") -> YOLOv5Detector:
    global _detector_instance
    if _detector_instance is None:
        if model_dir is None:
            model_dir = str(APP_DIR / "models")
        _detector_instance = YOLOv5Detector(model_dir=model_dir, device=device)
    return _detector_instance
