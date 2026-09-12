"""
图像处理工具 —— 基于 K230 image 模块 API 的绘制/裁剪/亚像素定位等操作。
"""


def draw_detections(img, detections, color=(255, 0, 0), thickness=2):
    """
    在图像上绘制检测框和标签。

    Args:
        img: K230 image 对象
        detections: [[class_name, confidence, x, y, w, h], ...]
        color: (R, G, B) 框颜色
        thickness: 框线宽
    """
    for det in detections:
        label = det[0]
        conf = det[1]
        x, y, w, h = int(det[2]), int(det[3]), int(det[4]), int(det[5])

        # 画检测框
        img.draw_rectangle(x, y, w, h, color=color, thickness=thickness)

        # 画标签文字 (新版 API: draw_string_advanced(x, y, size, str, color=))
        text = "{} {:.2f}".format(label, conf)
        img.draw_string_advanced(x, y - 20, 20, text, color=color)


def draw_text(img, text, x, y, color=(255, 255, 0), size=20):
    """
    在图像上绘制文字。

    """
    img.draw_string_advanced(int(x), int(y), size, text, color=color)


def draw_crosshair(img, x, y, size=10, color=(0, 255, 0), thickness=2):
    """
    画十字准星 (用于云台瞄准反馈)。

    Args:
        img: K230 image 对象
        x, y: 准星中心坐标
        size: 十字线半长
    """
    x, y = int(x), int(y)
    img.draw_line(x - size, y, x + size, y, color=color, thickness=thickness)
    img.draw_line(x, y - size, x, y + size, color=color, thickness=thickness)


def crop_roi(img, x, y, w, h):
    """
    从图像中裁剪 ROI 区域。

    返回: K230 image 对象 (裁剪后)
    """
    x, y, w, h = int(x), int(y), int(w), int(h)
    return img.copy(roi=(x, y, w, h))


def resize(img, w, h):
    """
    缩放图像到指定尺寸。

    返回: K230 image 对象 (缩放后)
    """
    w, h = int(w), int(h)
    return img.resize(w, h)


def subpixel_centroid(img, bbox, threshold=0):
    """
    灰度质心法: 在检测框内计算亚像素精度中心。

    用于高精度瞄准场景: 像素级中心定位精度不足时, 可用灰度质心获得亚像素中心。

    Args:
        img: K230 image 对象
        bbox: (x, y, w, h) 粗略检测框
        threshold: 灰度阈值 (只统计亮度>threshold的像素)

    Returns:
        (float_x, float_y) 亚像素中心坐标
    """
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # 边界保护
    img_w = img.width()
    img_h = img.height()
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    total_weight = 0.0
    sum_x = 0.0
    sum_y = 0.0

    for dy in range(h):
        for dx in range(dy % 2, w, 2):  # 隔列采样, 平衡精度和速度
            px = x + dx
            py = y + dy
            pixel = img.get_pixel(px, py)
            # K230 image.get_pixel 返回 (R,G,B) 或灰度值
            if isinstance(pixel, tuple):
                gray = (pixel[0] + pixel[1] + pixel[2]) // 3
            else:
                gray = pixel

            weight = gray - threshold
            if weight > 0:
                sum_x += px * weight
                sum_y += py * weight
                total_weight += weight

    if total_weight < 1e-9:
        # 回退到几何中心
        return (x + w / 2.0, y + h / 2.0)

    return (sum_x / total_weight, sum_y / total_weight)
