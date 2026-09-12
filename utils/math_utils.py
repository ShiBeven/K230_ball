"""
数学工具函数 —— 角度归一化、坐标转换、测距、像素-角度映射等。
"""

import math

# MicroPython 兼容: 优先使用 ulab 做矩阵运算, 否则用嵌套 list + 手写线性代数
try:
    from ulab import numpy as np  # K230 MicroPython 自带
    _HAS_ULAB = True
except ImportError:
    _HAS_ULAB = False


def limit_rad(angle):
    """
    将弧度归一化到 (-pi, pi]。
    """
    pi = math.pi
    while angle > pi:
        angle -= 2 * pi
    while angle <= -pi:
        angle += 2 * pi
    return angle


def square(a):
    """
    返回 a²。
    """
    return a * a


def limit_min_max(val, min_val, max_val):
    """
    限幅裁剪。
    """
    if val > max_val:
        return max_val
    if val < min_val:
        return min_val
    return val


def get_abs_angle(v1, v2):
    """
    两向量夹角 [0, pi]。
    """
    norm1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1])
    norm2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cos_val = dot / (norm1 * norm2)
    # 浮点误差保护
    if cos_val > 1.0:
        cos_val = 1.0
    if cos_val < -1.0:
        cos_val = -1.0
    return math.acos(cos_val)


# ---- 3D 坐标转换 ----

def xyz2ypd(xyz):
    """
    直角坐标系 → 球坐标系 (yaw, pitch, distance)。
    yaw:   从 x 轴向 y 轴的转角 (水平)
    pitch: 从 xy 平面向 z 轴的仰角 (垂直)
    """
    x, y, z = xyz[0], xyz[1], xyz[2]
    yaw = math.atan2(y, x)
    pitch = math.atan2(z, math.sqrt(x * x + y * y))
    distance = math.sqrt(x * x + y * y + z * z)
    return [yaw, pitch, distance]


def _mat_inv33(m):
    """
    3x3 矩阵求逆 (手写, 不依赖 numpy)。

    m 为 3x3 扁平 list [m00, m01, m02, m10, m11, m12, m20, m21, m22]
    """
    a, b, c = m[0], m[1], m[2]
    d, e, f = m[3], m[4], m[5]
    g, h, i = m[6], m[7], m[8]

    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-15:
        return [0.0] * 9

    inv_det = 1.0 / det
    return [
        (e * i - f * h) * inv_det,
        (c * h - b * i) * inv_det,
        (b * f - c * e) * inv_det,
        (f * g - d * i) * inv_det,
        (a * i - c * g) * inv_det,
        (c * d - a * f) * inv_det,
        (d * h - e * g) * inv_det,
        (b * g - a * h) * inv_det,
        (a * e - b * d) * inv_det,
    ]


def _mat_mul33(a, b):
    """3x3 矩阵乘法 a * b, 均为扁平 9 元素 list"""
    return [
        a[0]*b[0] + a[1]*b[3] + a[2]*b[6],
        a[0]*b[1] + a[1]*b[4] + a[2]*b[7],
        a[0]*b[2] + a[1]*b[5] + a[2]*b[8],
        a[3]*b[0] + a[4]*b[3] + a[5]*b[6],
        a[3]*b[1] + a[4]*b[4] + a[5]*b[7],
        a[3]*b[2] + a[4]*b[5] + a[5]*b[8],
        a[6]*b[0] + a[7]*b[3] + a[8]*b[6],
        a[6]*b[1] + a[7]*b[4] + a[8]*b[7],
        a[6]*b[2] + a[7]*b[5] + a[8]*b[8],
    ]


def xyz2ypd_jacobian(xyz):
    """
    直角→球坐标雅可比矩阵 J = d(ypd)/d(xyz)。

    返回 3x3 扁平 list [J00, J01, J02, J10, J11, J12, J20, J21, J22]
    """
    x, y, z = xyz[0], xyz[1], xyz[2]
    r2 = x * x + y * y
    r = math.sqrt(r2)
    r3 = r2 * r  # (x²+y²)^(3/2)
    dist = math.sqrt(r2 + z * z)

    eps = 1e-12

    dyaw_dx = -y / (r2 + eps)
    dyaw_dy = x / (r2 + eps)
    dyaw_dz = 0.0

    denom_pitch = (z * z / (r2 + eps) + 1) * r3
    if abs(denom_pitch) < eps:
        dpitch_dx = 0.0
        dpitch_dy = 0.0
    else:
        dpitch_dx = -(x * z) / denom_pitch
        dpitch_dy = -(y * z) / denom_pitch

    denom_pitch_z = (z * z / (r2 + eps) + 1) * r
    if abs(denom_pitch_z) < eps:
        dpitch_dz = 0.0
    else:
        dpitch_dz = 1.0 / denom_pitch_z

    ddistance_dx = x / (dist + eps)
    ddistance_dy = y / (dist + eps)
    ddistance_dz = z / (dist + eps)

    return [
        dyaw_dx, dyaw_dy, dyaw_dz,
        dpitch_dx, dpitch_dy, dpitch_dz,
        ddistance_dx, ddistance_dy, ddistance_dz,
    ]


def ypd2xyz(ypd):
    """
    球坐标系 → 直角坐标系。
    """
    yaw, pitch, distance = ypd[0], ypd[1], ypd[2]
    cos_pitch = math.cos(pitch)
    x = distance * cos_pitch * math.cos(yaw)
    y = distance * cos_pitch * math.sin(yaw)
    z = distance * math.sin(pitch)
    return [x, y, z]


def ypd2xyz_jacobian(ypd):
    """
    球→直角坐标雅可比矩阵 J = d(xyz)/d(ypd)。

    返回 3x3 扁平 list [J00, J01, J02, J10, J11, J12, J20, J21, J22]
    """
    yaw, pitch, distance = ypd[0], ypd[1], ypd[2]
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)

    dx_dyaw = distance * cos_pitch * -sin_yaw
    dy_dyaw = distance * cos_pitch * cos_yaw
    dz_dyaw = 0.0

    dx_dpitch = distance * -sin_pitch * cos_yaw
    dy_dpitch = distance * -sin_pitch * sin_yaw
    dz_dpitch = distance * cos_pitch

    dx_ddistance = cos_pitch * cos_yaw
    dy_ddistance = cos_pitch * sin_yaw
    dz_ddistance = sin_pitch

    return [
        dx_dyaw, dx_dpitch, dx_ddistance,
        dy_dyaw, dy_dpitch, dy_ddistance,
        dz_dyaw, dz_dpitch, dz_ddistance,
    ]


# ---- 视觉测量专用函数 ----

def pixel_to_angle(px, py, camera_matrix):
    """
    针孔模型: 像素坐标 → 光轴偏角。

    camera_matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]

    返回 (yaw_rad, pitch_rad)
        yaw   = atan2(px - cx, fx)    ← 水平偏角
        pitch = atan2(py - cy, fy)    ← 垂直偏角
    """
    fx = camera_matrix[0]
    fy = camera_matrix[4]
    cx = camera_matrix[2]
    cy = camera_matrix[5]

    yaw = math.atan2(px - cx, fx)
    pitch = math.atan2(py - cy, fy)
    return (yaw, pitch)


def estimate_distance(known_real_size_mm, pixel_size, camera_matrix):
    """
    单目测距 (针孔模型)。

    公式: distance = (fx * known_real_size_mm) / (pixel_size * 1000)

    Args:
        known_real_size_mm: 目标已知物理尺寸 (毫米)
        pixel_size: 目标在图像中的像素尺寸
        camera_matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]

    Returns:
        距离 (米)
    """
    fx = camera_matrix[0]
    if pixel_size < 1e-6:
        return float('inf')
    return (fx * known_real_size_mm) / (pixel_size * 1000.0)
