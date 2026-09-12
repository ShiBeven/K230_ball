"""
KPU 推理模块 — 模型加载 + 推理 + 后处理完整 Pipeline。

K230 KPU 使用 nncase 编译后的 .kmodel 格式。
典型 Pipeline: 加载模型 → 图像预处理 → KPU forward → 输出解析 → NMS。

依赖: model_utils.py, image_utils.py
"""

import kpu
import image
import gc

from utils.config_loader import load_config
from model_utils import nms, bbox_center, parse_yolo_output, parse_classify_output


class KPUModel:
    """
    KPU 神经网络模型封装。

    生命周期: load → (preprocess → forward → postprocess) × N → deinit
    """

    def __init__(self, model_path, labels=None, threshold=0.7, nms_threshold=0.45,
                 anchors=None, anchor_num=5, postprocess_fn=None):
        """
        初始化并加载 KPU 模型。

        Args:
            model_path: .kmodel 文件路径
            labels: 类别标签列表 (e.g. ["target", "obstacle", ...])
            threshold: 检测置信度阈值
            nms_threshold: NMS IoU 阈值
            anchors: 锚框参数 (YOLO 模型需要)
            anchor_num: 每个 grid cell 的锚框数
            postprocess_fn: 可选的自定义后处理**函数** (不必写类继承!)。
                            签名: (outputs, orig_w, orig_h, model) -> [[cls,conf,x,y,w,h],...]
                            其中 model 就是本 KPUModel 实例 (可取 _input_size/_labels/_threshold)。
                            传了它, detect() 就调它而不是默认 postprocess()。
                            —— 给不会写类继承的选手准备的"函数版"自定义后处理入口。
        """
        self._model_path = model_path
        self._labels = labels if labels else []
        self._threshold = threshold
        self._nms_threshold = nms_threshold
        self._anchors = anchors
        self._anchor_num = anchor_num
        self._postprocess_fn = postprocess_fn

        # 加载模型
        self._kpu = kpu.load(model_path)
        if self._kpu is None:
            raise RuntimeError("Failed to load KPU model: {}".format(model_path))

        # 获取模型输入尺寸
        self._input_size = kpu.get_input_size(self._kpu)

        print("[KPU] Model loaded: {}".format(model_path))
        print("[KPU] Input size: {}x{}".format(
            self._input_size[0], self._input_size[1]))

        gc.collect()

    @classmethod
    def from_config(cls, config=None, config_path="config.json", postprocess_fn=None):
        """
        从配置文件创建 KPUModel 实例。

        Args:
            postprocess_fn: 可选自定义后处理函数, 见 __init__ 说明。
        """
        if config is None:
            config = load_config(config_path)
        kpu_cfg = config.get("kpu", {})

        if not kpu_cfg.get("enabled", False):
            return None

        return cls(
            model_path=kpu_cfg.get("model_path", "models/model.kmodel"),
            labels=kpu_cfg.get("labels", []),
            threshold=kpu_cfg.get("threshold", 0.7),
            nms_threshold=kpu_cfg.get("nms_threshold", 0.45),
            anchors=kpu_cfg.get("anchors"),
            anchor_num=kpu_cfg.get("anchor_num", 5),
            postprocess_fn=postprocess_fn,
        )

    def preprocess(self, img):
        """
        图像预处理 — 缩放并转换为 KPU 输入格式。

        子类可覆盖此方法以适配不同模型的预处理需求。

        Args:
            img: K230 image 对象

        Returns:
            K230 image 对象 (尺寸与模型输入匹配)
        """
        w, h = self._input_size
        return img.resize(w, h)

    def forward(self, img=None):
        """
        KPU 前向推理。

        Args:
            img: 预处理后的 K230 image 对象 (若为 None 则使用上次 forward 的图像)

        Returns:
            模型原始输出 list
        """
        if img is not None:
            kpu.set_inputs(self._kpu, img)

        try:
            outputs = kpu.forward(self._kpu)
            return outputs
        except Exception as e:
            print("[KPU] forward() failed: {}".format(e))
            return None

    def postprocess(self, outputs, orig_w, orig_h):
        """
        后处理 — 解析 KPU 原始输出为检测结果列表。

        ⚠️ 这是**参考实现**, 不是万能解析器。
        K230 KPU 的 YOLO 输出格式取决于 nncase 编译参数, 各模型可能不同。
        本实现覆盖「最常见的 YOLO 单输出 + 锚框」格式:
          每个 grid cell 每个 anchor 产生 [tx, ty, tw, th, obj_conf, cls_conf...]。
        若你的模型输出不是这种布局, 请继承 KPUModel 并覆盖本方法
        

        Args:
            outputs: KPU forward() 返回的原始输出
            orig_w: 原始图像宽度 (用于坐标缩放)
            orig_h: 原始图像高度

        Returns:
            [[class_name, confidence, x, y, w, h], ...] (已做 NMS)
        """
        if not outputs:
            return []

        # 委托给 model_utils.parse_yolo_output 做通用 YOLO 解析。
        # parse_yolo_output 返回 [[class_id, conf, x, y, w, h], ...] (输入尺度、未 NMS)。
        num_classes = len(self._labels) if self._labels else 1
        raw = parse_yolo_output(
            outputs, self._anchors, num_classes,
            input_w=self._input_size[0], input_h=self._input_size[1],
            conf_threshold=self._threshold,
        )
        if not raw:
            return []

        # 坐标从模型输入尺度缩放回原图尺度
        scale_x = orig_w / self._input_size[0]
        scale_y = orig_h / self._input_size[1]

        detections = []
        for det in raw:
            class_id, conf, x, y, w, h = det
            # class_id -> 标签名 (labels 缺省时回退为字符串 id)
            if self._labels and 0 <= class_id < len(self._labels):
                name = self._labels[class_id]
            else:
                name = str(class_id)
            detections.append([
                name, conf,
                x * scale_x, y * scale_y, w * scale_x, h * scale_y,
            ])

        # NMS
        if detections:
            boxes = [d[2:6] for d in detections]
            scores = [d[1] for d in detections]
            keep = nms(boxes, scores, self._nms_threshold)
            detections = [detections[i] for i in keep]

        return detections

    def classify(self, img):
        """
        一键图像分类 pipeline。

        与 detect() 并列, 用于**分类模型** (单一 softmax 概率输出),
        而不是检测模型。preprocess → forward → parse_classify_output。

        Args:
            img: K230 image 对象 (原始尺寸)

        Returns:
            (class_name, confidence)
            —— labels 缺省或越界时 class_name 回退为字符串 id;
               无输出时返回 (None, 0.0)。
        """
        preprocessed = self.preprocess(img)
        outputs = self.forward(preprocessed)
        if not outputs:
            return (None, 0.0)

        # 分类模型通常单输出: 一维概率数组
        output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        class_id, conf = parse_classify_output(output)

        if class_id < 0:
            return (None, 0.0)
        if self._labels and 0 <= class_id < len(self._labels):
            return (self._labels[class_id], conf)
        return (str(class_id), conf)

    def detect(self, img):
        """
        一键目标检测 pipeline。

        preprocess → forward → postprocess → 返回检测结果。

        Args:
            img: K230 image 对象 (原始尺寸)

        Returns:
            [[class_name, confidence, x, y, w, h], ...]
        """
        orig_w = img.width()
        orig_h = img.height()

        # 预处理
        preprocessed = self.preprocess(img)

        # 推理
        outputs = self.forward(preprocessed)

        # 后处理: 优先用传入的自定义函数 (无需继承), 否则用默认 postprocess()
        if self._postprocess_fn is not None:
            detections = self._postprocess_fn(outputs, orig_w, orig_h, self)
        else:
            detections = self.postprocess(outputs, orig_w, orig_h)

        return detections

    def set_threshold(self, threshold):
        """更新置信度阈值"""
        self._threshold = threshold

    def set_nms_threshold(self, nms_threshold):
        """更新 NMS 阈值"""
        self._nms_threshold = nms_threshold

    def labels(self):
        """返回类别标签列表"""
        return self._labels

    def deinit(self):
        """释放 KPU 资源"""
        if self._kpu:
            kpu.deinit(self._kpu)
            self._kpu = None
            gc.collect()
            print("[KPU] Model deinitialized")
