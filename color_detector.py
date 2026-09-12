"""
纯视觉检测器 —— 基于 image.find_blobs 的 LAB 阈值检测, 与 AI 路线可互换。

输出与 KPUModel.detect() 完全一致: [[class_name, confidence, x, y, w, h], ...]
(x, y = 框左上角), 因此下游 recipe / display / image_utils 无需改动即可切换路线。

依赖: K230 image 模块。
"""


class ColorDetector:
    """
    基于颜色阈值 (LAB 色彩空间) 的目标检测器。

    与 KPUModel 鸭子类型兼容 (duck-typing): 提供同名的 detect(img) 方法,
    返回同样的六元组列表。可直接替换 KPUModel 使用。

    原理: img.find_blobs(阈值) 找出符合颜色的连通区域 (blob),
          每个 blob 转成一条 [class_name, confidence, x, y, w, h] 检测结果。
    """

    def __init__(self, thresholds, labels=None,
                 pixels_threshold=100, area_threshold=100,
                 merge=True, max_blobs=10, invert=False):
        """
        Args:
            thresholds: LAB 阈值列表, 每个是 6 元组
                        (L_min, L_max, A_min, A_max, B_min, B_max)。
                        每个阈值对应 labels 里的一个类别 (按顺序)。
            labels: 类别名列表, 和 thresholds 一一对应。
                    缺省时用 "color_0", "color_1", ...
            pixels_threshold: 色块最少像素数 (过滤噪点)
            area_threshold: 色块最小外接矩形面积
            merge: 是否合并重叠色块 (True 更稳)
            max_blobs: 每类最多返回多少个色块 (按面积从大到小)
            invert: 传给 find_blobs 的 invert 参数 (阈值取反)
        """
        self._thresholds = thresholds if thresholds else []
        if labels:
            self._labels = labels
        else:
            self._labels = ["color_{}".format(i) for i in range(len(self._thresholds))]
        self._pixels_threshold = pixels_threshold
        self._area_threshold = area_threshold
        self._merge = merge
        self._max_blobs = max_blobs
        self._invert = invert

    @classmethod
    def from_config(cls, config=None, config_path="config.json"):
        """
        从 config.json 的 "vision" 段创建。

        config["vision"] 结构示例:
            "vision": {
                "enabled": true,
                "thresholds": [[0, 80, 20, 80, -80, -20]],
                "labels": ["target"],
                "pixels_threshold": 100,
                "area_threshold": 100,
                "merge": true,
                "max_blobs": 5
            }

        vision.enabled=false 时返回 None。
        """
        if config is None:
            from utils.config_loader import load_config
            config = load_config(config_path)
        vcfg = config.get("vision", {})

        if not vcfg.get("enabled", False):
            return None

        # thresholds 里的元素可能是 list, 转成 tuple 供 find_blobs 用
        raw = vcfg.get("thresholds", [])
        thresholds = [tuple(t) for t in raw]

        return cls(
            thresholds=thresholds,
            labels=vcfg.get("labels"),
            pixels_threshold=vcfg.get("pixels_threshold", 100),
            area_threshold=vcfg.get("area_threshold", 100),
            merge=vcfg.get("merge", True),
            max_blobs=vcfg.get("max_blobs", 10),
            invert=vcfg.get("invert", False),
        )

    def detect(self, img):
        """
        检测 —— 和 KPUModel.detect() 签名/返回完全一致。

        Args:
            img: K230 image 对象

        Returns:
            [[class_name, confidence, x, y, w, h], ...]
              x, y = 框左上角;  w, h = 框宽高。
              confidence: 用色块面积占比作为伪置信度 (0~1), 面积越大越"确信"。
        """
        results = []
        img_area = float(img.width() * img.height())
        if img_area <= 0:
            return results

        for idx, thr in enumerate(self._thresholds):
            label = self._labels[idx] if idx < len(self._labels) else "color_{}".format(idx)

            blobs = img.find_blobs(
                [thr],
                pixels_threshold=self._pixels_threshold,
                area_threshold=self._area_threshold,
                merge=self._merge,
                invert=self._invert,
            )
            if not blobs:
                continue

            # 面积从大到小, 取前 max_blobs 个
            blobs = sorted(blobs, key=lambda b: b.area(), reverse=True)
            for b in blobs[:self._max_blobs]:
                x, y, w, h = b.x(), b.y(), b.w(), b.h()
                # 伪置信度: 外接矩形面积占全图比例, 截断到 [0, 1]
                conf = min(1.0, (w * h) / img_area * 4.0)
                results.append([label, conf, float(x), float(y),
                                float(w), float(h)])

        return results

    def classify(self, img):
        """
        与 KPUModel.classify() 兼容: 返回面积最大色块的类别。

        Returns:
            (class_name, confidence);  没找到返回 (None, 0.0)。
        """
        dets = self.detect(img)
        if not dets:
            return (None, 0.0)
        best = max(dets, key=lambda d: d[1])
        return (best[0], best[1])

    def labels(self):
        """返回类别标签列表 (与 KPUModel.labels() 兼容)"""
        return self._labels

    def deinit(self):
        """无资源需释放 (与 KPUModel.deinit() 兼容, 空实现)"""
        pass


def make_detector(config):
    """
    统一检测器工厂 —— **一键在 KPU 与纯视觉之间无障碍切换**。

    决策顺序:
      1. 若 kpu.enabled=true → 尝试加载 KPUModel;
         加载成功就用它 (神经网络方案)。
      2. KPU 未启用 / 加载失败 / 抛异常 → 自动回退:
         若 vision.enabled=true → 用 ColorDetector (纯视觉方案)。
      3. 两者都不可用 → 返回 None。

    返回的对象 (KPUModel 或 ColorDetector) **接口完全一致**:
    都有 detect(img) / classify(img) / labels() / deinit()。
    所以调用方拿到后照常用, 不需要知道底层是哪种。

    典型用法:
      - 有可用的 .kmodel → config 里 kpu.enabled=true, 走神经网络;
      - 模型不可用 → 设 kpu.enabled=false、vision.enabled=true,
        代码一行不动, 切换为颜色识别。

    Args:
        config: 完整配置 dict

    Returns:
        KPUModel | ColorDetector | None
    """
    kpu_cfg = config.get("kpu", {})
    if kpu_cfg.get("enabled", False):
        try:
            from kpu_tools import KPUModel
            model = KPUModel.from_config(config)
            if model is not None:
                print("[Detector] 使用 KPU 神经网络方案")
                return model
        except Exception as e:
            print("[Detector] KPU 加载失败, 回退纯视觉: {}".format(e))

    # 回退到纯视觉
    detector = ColorDetector.from_config(config)
    if detector is not None:
        print("[Detector] 使用纯视觉 (颜色识别) 方案")
        return detector

    print("[Detector] 警告: KPU 与纯视觉都未启用, 无检测器")
    return None
