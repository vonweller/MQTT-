"""
YOLOv8/v11推理模块
提供目标检测、分割、关键点检测、分类等推理功能
"""

import os
import cv2
import numpy as np
import base64
import time
import json
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

try:
    import torch
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch未安装，推理功能将不可用")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics未安装，YOLOv8/v11推理功能将不可用")


@dataclass
class DetectionResult:
    """检测结果"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'bbox': self.bbox,
            'confidence': round(self.confidence, 4),
            'class_id': self.class_id,
            'class_name': self.class_name
        }


@dataclass
class SegmentationResult(DetectionResult):
    """分割结果"""
    mask: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result['mask_shape'] = self.mask.shape if len(self.mask) > 0 else None
        return result


@dataclass
class KeypointResult:
    """关键点检测结果"""
    keypoints: List[Tuple[float, float, float]]  # [(x, y, confidence), ...]
    bbox: List[float]
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'keypoints': self.keypoints,
            'bbox': self.bbox,
            'confidence': round(self.confidence, 4)
        }


@dataclass
class ClassificationResult:
    """分类结果"""
    class_id: int
    class_name: str
    confidence: float
    top5: List[Tuple[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': round(self.confidence, 4),
            'top5': [(name, round(conf, 4)) for name, conf in self.top5]
        }


@dataclass
class InferenceResult:
    """推理结果"""
    detections: List[DetectionResult] = field(default_factory=list)
    segmentations: List[SegmentationResult] = field(default_factory=list)
    keypoints: List[KeypointResult] = field(default_factory=list)
    classifications: List[ClassificationResult] = field(default_factory=list)
    inference_time: float = 0.0
    fps: float = 0.0
    model_type: str = "detection"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'detections': [d.to_dict() for d in self.detections],
            'segmentations': [s.to_dict() for s in self.segmentations],
            'keypoints': [k.to_dict() for k in self.keypoints],
            'classifications': [c.to_dict() for c in self.classifications],
            'inference_time': round(self.inference_time, 4),
            'fps': round(self.fps, 2),
            'model_type': self.model_type
        }


class YOLOInference:
    """YOLO26推理器"""
    
    # COCO类别名称（英文）
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # COCO类别名称（中文）
    COCO_CLASSES_CN = [
        '人', '自行车', '汽车', '摩托车', '飞机', '公交车', '火车', '卡车', '船',
        '红绿灯', '消防栓', '停车标志', '停车计时器', '长椅', '鸟', '猫',
        '狗', '马', '羊', '牛', '大象', '熊', '斑马', '长颈鹿', '背包',
        '雨伞', '手提包', '领带', '行李箱', '飞盘', '滑雪板', '单板滑雪', '运动球',
        '风筝', '棒球棒', '棒球手套', '滑板', '冲浪板', '网球拍',
        '瓶子', '酒杯', '杯子', '叉子', '刀', '勺子', '碗', '香蕉', '苹果',
        '三明治', '橙子', '西兰花', '胡萝卜', '热狗', '披萨', '甜甜圈', '蛋糕', '椅子',
        '沙发', '盆栽植物', '床', '餐桌', '马桶', '电视', '笔记本电脑', '鼠标',
        '遥控器', '键盘', '手机', '微波炉', '烤箱', '烤面包机', '水槽', '冰箱',
        '书', '时钟', '花瓶', '剪刀', '泰迪熊', '吹风机', '牙刷'
    ]
    
    # 模型类型映射
    MODEL_TYPES = {
        'detection': '目标检测',
        'segmentation': '实例分割',
        'pose': '关键点检测',
        'classification': '图像分类'
    }
    
    # 中文字体路径列表（按优先级）
    CHINESE_FONT_PATHS = [
        # Windows 系统字体
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑粗体
        "C:/Windows/Fonts/simkai.ttf",  # 楷体
        "C:/Windows/Fonts/simfang.ttf", # 仿宋
        # Linux 系统字体
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        # macOS 系统字体
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    
    def __init__(self,
                 model_path: str = "",
                 model_type: str = "detection",
                 device: str = "auto",
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 img_size: int = 640,
                 half_precision: bool = False):
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，请运行: pip install torch torchvision")
        
        self.model_path = model_path
        self.model_type = model_type
        self.device = self._get_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.half_precision = half_precision and self.device.type == 'cuda'
        
        # 模型
        self.model = None
        self.class_names = self.COCO_CLASSES
        self.use_chinese_labels = True  # 默认使用中文标签
        
        # 中文字体
        self.chinese_font = None
        self._load_chinese_font()
        
        # 统计
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        # 回调
        self.on_inference_callbacks: List[Callable[[InferenceResult], None]] = []
        
        logger.info(f"YOLO推理器初始化: 类型={model_type}, 设备={self.device}")
    
    def _load_chinese_font(self):
        """加载中文字体"""
        for font_path in self.CHINESE_FONT_PATHS:
            if os.path.exists(font_path):
                try:
                    # 使用PIL加载字体
                    from PIL import ImageFont
                    self.chinese_font = ImageFont.truetype(font_path, 20)
                    logger.info(f"加载中文字体: {font_path}")
                    return
                except Exception as e:
                    logger.warning(f"加载字体失败 {font_path}: {e}")
        
        logger.warning("未找到中文字体，将使用默认字体")
    
    def get_class_name(self, class_id: int) -> str:
        """获取类别名称（支持中文）"""
        if self.use_chinese_labels and class_id < len(self.COCO_CLASSES_CN):
            return self.COCO_CLASSES_CN[class_id]
        elif class_id < len(self.class_names):
            return self.class_names[class_id]
        else:
            return f"类别{class_id}"
        
        # 如果提供了模型路径，自动加载
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _get_device(self, device: str) -> torch.device:
        """获取计算设备"""
        if device == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def load_model(self, model_path: str) -> bool:
        """加载模型"""
        try:
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return False
            
            logger.info(f"正在加载模型: {model_path}")
            
            # 使用Ultralytics加载YOLO模型
            if ULTRALYTICS_AVAILABLE:
                self.model = YOLO(model_path)
                # 设置推理参数
                self.model.to(self.device)
                logger.info(f"使用Ultralytics加载模型成功")
            else:
                logger.error("Ultralytics未安装，无法加载YOLOv8/v11模型")
                return False
            
            self.model_path = model_path
            
            # 尝试加载类别名称
            self._load_class_names(model_path)
            
            logger.info(f"模型加载成功: {model_path}")
            return True
            
        except Exception as e:
            logger.exception(f"模型加载失败: {e}")
            return False
    
    def _load_class_names(self, model_path: str):
        """加载类别名称"""
        # 尝试从同目录的yaml或txt文件加载
        model_dir = Path(model_path).parent
        model_name = Path(model_path).stem
        
        # 尝试不同的文件名
        possible_names = [
            model_dir / f"{model_name}.yaml",
            model_dir / "classes.txt",
            model_dir / "coco.yaml",
            model_dir / "data.yaml"
        ]
        
        for name_file in possible_names:
            if name_file.exists():
                try:
                    if name_file.suffix == '.yaml':
                        import yaml
                        with open(name_file, 'r') as f:
                            data = yaml.safe_load(f)
                            if 'names' in data:
                                self.class_names = data['names']
                                logger.info(f"从 {name_file} 加载类别名称")
                                return
                    else:
                        with open(name_file, 'r') as f:
                            self.class_names = [line.strip() for line in f if line.strip()]
                            logger.info(f"从 {name_file} 加载类别名称")
                            return
                except Exception as e:
                    logger.warning(f"加载类别名称失败: {e}")
        
        # 使用默认COCO类别
        self.class_names = self.COCO_CLASSES
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 调整图像大小
        h, w = image.shape[:2]
        
        # 计算缩放比例
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 调整大小
        resized = cv2.resize(image, (new_w, new_h))
        
        # 创建填充图像
        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # BGR to RGB
        padded = padded[:, :, ::-1]
        
        # 归一化
        padded = padded.astype(np.float32) / 255.0
        
        # HWC to CHW
        padded = np.transpose(padded, (2, 0, 1))
        
        # 添加batch维度
        tensor = torch.from_numpy(padded).unsqueeze(0)
        
        # 移动到设备
        tensor = tensor.to(self.device)
        
        # 确保输入类型与模型权重类型一致
        if self.model is not None:
            # 获取模型的权重类型
            model_dtype = next(self.model.parameters()).dtype
            if model_dtype != tensor.dtype:
                tensor = tensor.to(model_dtype)
                logger.debug(f"输入类型转换为: {model_dtype}")
        
        return tensor, scale, (new_w, new_h)
    
    def postprocess(self, outputs, scale: float, orig_size: Tuple[int, int]) -> List[DetectionResult]:
        """后处理检测结果"""
        results = []
        
        if outputs is None or len(outputs) == 0:
            return results
        
        # 解析输出（根据YOLOv6的输出格式调整）
        # 假设输出格式为: [batch, num_detections, 6] (x1, y1, x2, y2, conf, class)
        
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().numpy()
        
        orig_w, orig_h = orig_size
        
        for det in outputs[0]:  # batch=1
            if len(det) < 6:
                continue
            
            x1, y1, x2, y2, conf, cls_id = det[:6]
            
            if conf < self.conf_threshold:
                continue
            
            # 缩放到原始图像尺寸
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)
            
            # 限制在图像范围内
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            class_id = int(cls_id)
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            
            results.append(DetectionResult(
                bbox=[x1, y1, x2, y2],
                confidence=float(conf),
                class_id=class_id,
                class_name=class_name
            ))
        
        return results
    
    def infer(self, image: Union[np.ndarray, str, bytes]) -> InferenceResult:
        """执行推理 - 使用Ultralytics API，支持所有任务类型"""
        start_time = time.time()
        
        try:
            # 检查模型
            if self.model is None:
                logger.error("模型未加载")
                return InferenceResult(model_type=self.model_type)
            
            # 使用Ultralytics进行推理
            results = self.model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                verbose=False
            )
            
            # 初始化结果列表
            detections = []
            segmentations = []
            keypoints_list = []
            classifications = []
            
            # 解析结果 - 根据模型类型处理不同任务
            for result in results:
                # 定向检测 (OBB) - 优先检查，因为OBB也有boxes但格式不同
                if hasattr(result, 'obb') and result.obb is not None:
                    obb_boxes = result.obb
                    for i in range(len(obb_boxes)):
                        # 使用多边形格式 xyxyxyxy (4个点)
                        poly = obb_boxes.xyxyxyxy[i].cpu().numpy().flatten()
                        conf = float(obb_boxes.conf[i].cpu().numpy())
                        cls_id = int(obb_boxes.cls[i].cpu().numpy())
                        cls_name = self.get_class_name(cls_id)
                        
                        # 将多边形转换为bbox (x1, y1, x2, y2)
                        x_coords = poly[0::2]
                        y_coords = poly[1::2]
                        bbox = [float(min(x_coords)), float(min(y_coords)), 
                                float(max(x_coords)), float(max(y_coords))]
                        
                        detections.append(DetectionResult(
                            bbox=bbox,
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name
                        ))
                
                # 目标检测 (标准检测)
                elif result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        bbox = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        cls_name = self.get_class_name(cls_id)
                        
                        detections.append(DetectionResult(
                            bbox=bbox.tolist(),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name
                        ))
                
                # 实例分割 - 使用 masks.data (num_objects x H x W)
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks
                    # masks.data 是矩阵格式 (num_objects x H x W)
                    mask_data = masks.data.cpu().numpy()
                    
                    for i in range(len(mask_data)):
                        mask = mask_data[i]
                        # 获取对应的目标检测信息
                        if result.boxes is not None and i < len(result.boxes):
                            box = result.boxes[i]
                            bbox = box.xyxy[0].cpu().numpy().tolist()
                            conf = float(box.conf.cpu().numpy())
                            cls_id = int(box.cls.cpu().numpy())
                            cls_name = self.get_class_name(cls_id)
                        else:
                            bbox = [0, 0, 0, 0]
                            conf = 0.0
                            cls_id = 0
                            cls_name = "unknown"
                        
                        segmentations.append(SegmentationResult(
                            bbox=bbox,
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name,
                            mask=mask
                        ))
                
                # 姿态估计 / 关键点检测 - 使用 keypoints.data (x, y, visibility)
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    kpts = result.keypoints
                    # keypoints.data 包含 x, y, visibility
                    kpts_data = kpts.data.cpu().numpy()
                    
                    for i in range(len(kpts_data)):
                        keypoints = kpts_data[i]  # [num_keypoints, 3] (x, y, conf/visibility)
                        # 转换为列表格式
                        kpt_list = [(float(kp[0]), float(kp[1]), float(kp[2])) for kp in keypoints]
                        
                        # 获取对应的目标框
                        if result.boxes is not None and i < len(result.boxes):
                            box = result.boxes[i]
                            bbox = box.xyxy[0].cpu().numpy().tolist()
                            conf = float(box.conf.cpu().numpy())
                        else:
                            bbox = [0, 0, 0, 0]
                            conf = 0.0
                        
                        keypoints_list.append(KeypointResult(
                            keypoints=kpt_list,
                            bbox=bbox,
                            confidence=conf
                        ))
                
                # 图像分类 - 使用 probs
                if hasattr(result, 'probs') and result.probs is not None:
                    probs = result.probs
                    top5_indices = probs.top5
                    top5_conf = probs.top5conf.cpu().numpy()
                    
                    # 构建top5列表
                    top5_list = []
                    for idx, conf in zip(top5_indices, top5_conf):
                        cls_name = result.names[idx] if idx in result.names else f"类别{idx}"
                        top5_list.append((cls_name, float(conf)))
                    
                    # 获取top1
                    top1_idx = top5_indices[0]
                    top1_conf = float(top5_conf[0])
                    top1_name = result.names[top1_idx] if top1_idx in result.names else f"类别{top1_idx}"
                    
                    classifications.append(ClassificationResult(
                        class_id=int(top1_idx),
                        class_name=top1_name,
                        confidence=top1_conf,
                        top5=top5_list
                    ))
            
            # 计算FPS
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0
            
            logger.info(f"推理完成: {len(detections)}检测/{len(segmentations)}分割/{len(keypoints_list)}关键点/{len(classifications)}分类, "
                       f"推理时间: {inference_time*1000:.1f}ms")
            
            # 更新统计
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            result = InferenceResult(
                detections=detections,
                segmentations=segmentations,
                keypoints=keypoints_list,
                classifications=classifications,
                inference_time=inference_time,
                fps=fps,
                model_type=self.model_type
            )
            
            # 触发回调
            for callback in self.on_inference_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"推理回调错误: {e}")
            
            return result
            
        except Exception as e:
            logger.exception(f"推理过程出错: {e}")
            # 返回空结果
            return InferenceResult(
                detections=[],
                segmentations=[],
                keypoints=[],
                classifications=[],
                inference_time=time.time() - start_time,
                fps=0,
                model_type=self.model_type
            )
    
    def infer_batch(self, images: List[Union[np.ndarray, str]]) -> List[InferenceResult]:
        """批量推理"""
        results = []
        for image in images:
            result = self.infer(image)
            results.append(result)
        return results
    
    def _decode_base64_image(self, base64_string: str) -> np.ndarray:
        """解码Base64图像"""
        try:
            # 移除data URL前缀
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # 解码
            image_bytes = base64.b64decode(base64_string)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            logger.error(f"Base64解码失败: {e}")
            return None
    
    def draw_results(self, image: np.ndarray, result: InferenceResult) -> np.ndarray:
        """在图像上绘制检测结果 - 支持中文标签"""
        img = image.copy()
        
        # 如果有中文字体，使用PIL绘制中文标签
        if self.chinese_font is not None and self.use_chinese_labels:
            return self._draw_results_with_chinese(img, result)
        else:
            return self._draw_results_with_english(img, result)
    
    def _draw_results_with_english(self, img: np.ndarray, result: InferenceResult) -> np.ndarray:
        """使用英文标签绘制（OpenCV默认）- 支持所有任务类型"""
        # 绘制分类结果（分类任务优先，不绘制检测框）
        if result.classifications:
            cls = result.classifications[0]
            # 显示分类结果在图像中央
            cls_text = f"Class: {cls.class_name} ({cls.confidence:.2f})"
            cv2.putText(img, cls_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            # 显示Top5
            y_offset = 60
            for i, (name, conf) in enumerate(cls.top5[:5]):
                top_text = f"#{i+1}: {name} ({conf:.2f})"
                cv2.putText(img, top_text, (10, y_offset + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            return img
        
        # 绘制实例分割掩膜
        for seg in result.segmentations:
            if len(seg.mask) > 0:
                color = self._get_color(seg.class_id)
                mask = seg.mask
                if len(mask.shape) > 2:
                    mask = mask.squeeze()
                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask.astype(np.float32), (img.shape[1], img.shape[0]))
                mask_binary = (mask > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(img, contours, -1, color, 2)
                mask_bool = mask > 0.5
                overlay = img.copy()
                img[mask_bool] = (img[mask_bool] * 0.6 + np.array(color) * 0.4).astype(np.uint8)
        
        # 绘制检测框
        for det in result.detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            color = self._get_color(det.class_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 绘制姿态估计关键点
        for kp_result in result.keypoints:
            keypoints = kp_result.keypoints
            skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
            for connection in skeleton:
                idx1, idx2 = connection[0] - 1, connection[1] - 1
                if idx1 < len(keypoints) and idx2 < len(keypoints):
                    x1, y1, conf1 = keypoints[idx1]
                    x2, y2, conf2 = keypoints[idx2]
                    if conf1 > 0.5 and conf2 > 0.5:
                        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.5:
                    cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
        
        # 绘制FPS和检测数量
        fps_text = f"FPS: {result.fps:.1f}"
        cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        count_text = f"D: {len(result.detections)} S: {len(result.segmentations)} K: {len(result.keypoints)}"
        cv2.putText(img, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return img
    
    def _draw_results_with_chinese(self, img: np.ndarray, result: InferenceResult) -> np.ndarray:
        """使用中文标签绘制（PIL）- 支持所有任务类型"""
        from PIL import Image, ImageDraw, ImageFont
        
        # 转换OpenCV图像为PIL图像
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # 绘制实例分割掩膜 - 使用轮廓线方式
        for seg in result.segmentations:
            if len(seg.mask) > 0:
                color = self._get_color(seg.class_id)
                mask = seg.mask
                
                # 确保掩膜是二维的
                if len(mask.shape) > 2:
                    mask = mask.squeeze()
                
                # 调整掩膜大小到图像尺寸
                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask.astype(np.float32), (img.shape[1], img.shape[0]))
                
                # 二值化掩膜
                mask_binary = (mask > 0.5).astype(np.uint8) * 255
                
                # 查找轮廓
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 绘制轮廓线
                if contours:
                    # 转换PIL图像回OpenCV格式进行轮廓绘制
                    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    cv2.drawContours(img_cv, contours, -1, color, 2)
                    # 转换回PIL
                    pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                
                # 半透明填充
                mask_bool = mask > 0.5
                overlay_color = np.array(color, dtype=np.uint8)
                img_rgb[mask_bool] = (img_rgb[mask_bool] * 0.6 + overlay_color * 0.4).astype(np.uint8)
        
        # 绘制检测框和中文标签
        for det in result.detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            color = self._get_color(det.class_id)
            
            # 绘制矩形框
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            
            # 获取中文类别名称
            class_name = self.get_class_name(det.class_id)
            label = f"{class_name}: {det.confidence:.2f}"
            
            # 计算文本大小
            bbox = draw.textbbox((0, 0), label, font=self.chinese_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 绘制标签背景
            draw.rectangle(
                [(x1, y1 - text_height - 4), (x1 + text_width, y1)],
                fill=color
            )
            
            # 绘制中文标签
            draw.text((x1, y1 - text_height - 2), label, font=self.chinese_font, fill=(255, 255, 255))
        
        # 绘制姿态估计关键点
        for kp_result in result.keypoints:
            keypoints = kp_result.keypoints
            # COCO关键点连接定义 (17个关键点)
            skeleton = [
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # 躯干
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],          # 上半身
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],          # 四肢
                [2, 4], [3, 5], [4, 6], [5, 7]                      # 腿部
            ]
            
            # 绘制骨架连接线
            for connection in skeleton:
                idx1, idx2 = connection[0] - 1, connection[1] - 1  # 转换为0-based索引
                if idx1 < len(keypoints) and idx2 < len(keypoints):
                    x1, y1, conf1 = keypoints[idx1]
                    x2, y2, conf2 = keypoints[idx2]
                    if conf1 > 0.5 and conf2 > 0.5:  # 两个点都有效才连线
                        draw.line([(int(x1), int(y1)), (int(x2), int(y2))], fill=(0, 255, 255), width=2)
            
            # 绘制关键点
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.5:  # 只绘制置信度高的关键点
                    x, y = int(x), int(y)
                    # 不同部位用不同颜色
                    if i < 5:  # 头部
                        color = (255, 0, 0)
                    elif i < 11:  # 手臂
                        color = (0, 255, 0)
                    else:  # 腿部
                        color = (0, 0, 255)
                    draw.ellipse([(x-4, y-4), (x+4, y+4)], fill=color)
        
        # 绘制FPS（中文）
        fps_text = f"帧率: {result.fps:.1f}"
        draw.text((10, 10), fps_text, font=self.chinese_font, fill=(0, 255, 0))
        
        # 绘制检测数量（中文）
        count_text = f"检测: {len(result.detections)} 分割: {len(result.segmentations)} 关键点: {len(result.keypoints)}"
        draw.text((10, 35), count_text, font=self.chinese_font, fill=(0, 255, 0))
        
        # 绘制分类结果
        if result.classifications:
            cls = result.classifications[0]
            cls_text = f"分类: {cls.class_name} ({cls.confidence:.2f})"
            draw.text((10, 60), cls_text, font=self.chinese_font, fill=(255, 255, 0))
        
        # 转换回OpenCV图像
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img_bgr
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """获取类别颜色"""
        # 使用固定的颜色映射
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        return colors[class_id % len(colors)]
    
    def add_inference_callback(self, callback: Callable[[InferenceResult], None]):
        """添加推理回调"""
        self.on_inference_callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_time = self.total_inference_time / self.inference_count if self.inference_count > 0 else 0
        return {
            'inference_count': self.inference_count,
            'total_inference_time': round(self.total_inference_time, 4),
            'average_inference_time': round(avg_time, 4),
            'model_loaded': self.model is not None,
            'model_type': self.model_type,
            'device': str(self.device)
        }
    
    def set_conf_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.conf_threshold = max(0.0, min(1.0, threshold))
    
    def set_iou_threshold(self, threshold: float):
        """设置IOU阈值"""
        self.iou_threshold = max(0.0, min(1.0, threshold))
    
    @staticmethod
    def get_available_models() -> List[str]:
        """获取可用模型类型"""
        return list(YOLOInference.MODEL_TYPES.keys())
    
    @staticmethod
    def get_model_type_name(model_type: str) -> str:
        """获取模型类型名称"""
        return YOLOInference.MODEL_TYPES.get(model_type, "未知")
