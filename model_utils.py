"""
模型后处理工具 — 纯函数, 不依赖 KPU 硬件。

提供: NMS, YOLO 输出解析, 分类结果解析, IoU 计算, bbox 中心。
可用于任何检测结果的后处理。
"""


def iou(box_a, box_b):
    """
    计算两个边界框的 IoU (交并比)。

    Args:
        box_a, box_b: [x, y, w, h]

    Returns:
        IoU 值 [0, 1]
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    y2 = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union_area = area_a + area_b - inter_area

    if union_area < 1e-9:
        return 0.0
    return inter_area / union_area


def nms(boxes, scores, iou_threshold=0.45):
    """
    非极大值抑制 (NMS)。

    Args:
        boxes: [[x, y, w, h], ...] 列表
        scores: [score, ...] 置信度列表
        iou_threshold: IoU 阈值

    Returns:
        保留的索引列表
    """
    if not boxes:
        return []

    # 按分数降序排列
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)

    keep = []
    while indexed:
        idx, _ = indexed.pop(0)
        keep.append(idx)

        # 移除与当前框 IoU > 阈值的框
        remaining = []
        for j, _ in indexed:
            if iou(boxes[idx], boxes[j]) <= iou_threshold:
                remaining.append((j, scores[j]))
        indexed = remaining

    return keep


def bbox_center(bbox):
    """
    计算边界框中心坐标。

    Args:
        bbox: [x, y, w, h]

    Returns:
        (cx, cy)
    """
    return (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)


def _sigmoid(x):
    """数值稳定的 sigmoid, 纯 Python (无 numpy 依赖)"""
    import math
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def parse_yolo_output(outputs, anchors, num_classes,
                      input_w=320, input_h=320,
                      conf_threshold=0.5):
    """
    解析 YOLO 模型原始输出 (参考实现)。

    ⚠️ K230 nncase 编译的 YOLO 输出布局因模型/编译参数而异。
    本函数覆盖**最常见**的一种: 单个输出张量, 逐 grid cell、逐 anchor 排列,
    每个预测为 [tx, ty, tw, th, obj_conf, cls_0, cls_1, ...] (原始 logits)。
    若你的模型不是这种布局, 请在 KPUModel 子类里覆盖 postprocess() 而不是改这里。

    解析逻辑:
      1. 把扁平输出按 (grid_h, grid_w, num_anchors, 5+num_classes) 还原;
      2. obj_conf、cls_conf 过 sigmoid, 取 obj*max(cls) 为置信度;
      3. 低于 conf_threshold 的丢弃;
      4. tx/ty 过 sigmoid + grid 偏移, tw/th 过 exp*anchor, 换算到输入图像像素;
      5. 输出左上角 (x, y) + 宽高 (w, h)。

    Args:
        outputs: KPU forward() 返回的原始输出 list (取 outputs[0])
        anchors: [(aw, ah), ...] 锚框 (输入图像像素单位); None 时退化为 grid 尺寸
        num_classes: 类别数
        input_w, input_h: 模型输入尺寸
        conf_threshold: 置信度阈值

    Returns:
        [[class_id, confidence, x, y, w, h], ...] (输入尺度, 未做 NMS)
        —— 无法解析或输出为空时返回 []。
    """
    if not outputs:
        return []

    output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

    # 尝试把输出摊平成一维 float 序列
    try:
        flat = _flatten(output)
    except Exception:
        return []
    if not flat:
        return []

    num_anchors = len(anchors) if anchors else 1
    stride = 5 + num_classes
    per_cell = num_anchors * stride
    if per_cell <= 0 or len(flat) % per_cell != 0:
        # 布局与假设不符, 交给用户覆盖 postprocess
        return []

    num_cells = len(flat) // per_cell
    # 假设正方形 grid; 非正方形需用户覆盖
    import math
    grid = int(round(math.sqrt(num_cells)))
    if grid * grid != num_cells:
        return []
    grid_w = grid_h = grid

    cell_w = input_w / grid_w
    cell_h = input_h / grid_h

    detections = []
    idx = 0
    for gy in range(grid_h):
        for gx in range(grid_w):
            for a in range(num_anchors):
                base = idx
                tx = flat[base + 0]
                ty = flat[base + 1]
                tw = flat[base + 2]
                th = flat[base + 3]
                obj = _sigmoid(flat[base + 4])

                # 最大类别概率
                best_cls = 0
                best_p = _sigmoid(flat[base + 5]) if num_classes >= 1 else 1.0
                for c in range(1, num_classes):
                    p = _sigmoid(flat[base + 5 + c])
                    if p > best_p:
                        best_p = p
                        best_cls = c

                conf = obj * best_p
                idx += stride

                if conf < conf_threshold:
                    continue

                # 中心坐标 (输入图像像素)
                cx = (_sigmoid(tx) + gx) * cell_w
                cy = (_sigmoid(ty) + gy) * cell_h
                if anchors:
                    aw, ah = anchors[a]
                    bw = math.exp(tw) * aw
                    bh = math.exp(th) * ah
                else:
                    bw = math.exp(tw) * cell_w
                    bh = math.exp(th) * cell_h

                x = cx - bw / 2.0
                y = cy - bh / 2.0
                detections.append([best_cls, conf, x, y, bw, bh])

    return detections


def _flatten(x):
    """把嵌套 list / tuple / ulab 数组摊平成一维 Python float list"""
    out = []
    stack = [x]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            # 逆序压栈以保持原顺序
            for sub in reversed(item):
                stack.append(sub)
        elif hasattr(item, "flatten"):  # ulab ndarray
            for v in item.flatten():
                out.append(float(v))
        elif hasattr(item, "__iter__") and not isinstance(item, (str, bytes)):
            for sub in reversed(list(item)):
                stack.append(sub)
        else:
            out.append(float(item))
    return out


def parse_classify_output(output):
    """
    解析分类模型输出。

    Args:
        output: KPU forward() 返回的原始输出 (通常是 softmax 后的概率数组)

    Returns:
        (class_id, confidence)
    """
    if not output:
        return (-1, 0.0)

    # 取概率最大的类别
    max_idx = 0
    max_val = output[0]
    for i, val in enumerate(output):
        if val > max_val:
            max_val = val
            max_idx = i

    return (max_idx, max_val)
