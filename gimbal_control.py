"""
云台控制 —— 像素坐标 → 角度映射 + 串口输出, 帧格式可自定义(默认文本协议)。

依赖: serial_comm.py, utils/math_utils.py
"""

import math
from utils.math_utils import pixel_to_angle


def _default_frame_builder(yaw_rad, pitch_rad):
    """
    默认帧格式: 文本协议。

    "Y:0.1234,P:-0.0567\n"
    """
    return "Y:{:.4f},P:{:.4f}\n".format(yaw_rad, pitch_rad).encode()


def _binary_frame_builder(yaw_rad, pitch_rad):
    """
    备选帧格式: 二进制 float 对 (8 字节)。

    小端序: [yaw_float(4B), pitch_float(4B)]
    """
    import struct
    return struct.pack("<ff", yaw_rad, pitch_rad)


# 协议名 -> 帧构建器映射。config.json 的 gimbal.protocol 由此生效。
_FRAME_BUILDERS = {
    "text": _default_frame_builder,
    "binary": _binary_frame_builder,
}


def frame_builder_from_config(config):
    """
    根据 config["gimbal"]["protocol"] 返回对应的帧构建器。

    让 config.json 的 gimbal.protocol 字段真正生效 ("text" / "binary")。
    未知协议名回退为文本协议。

    Args:
        config: 完整配置 dict

    Returns:
        (yaw_rad, pitch_rad) -> bytes 的帧构建器
    """
    protocol = config.get("gimbal", {}).get("protocol", "text")
    builder = _FRAME_BUILDERS.get(protocol)
    if builder is None:
        print("[Gimbal] Unknown protocol '{}', fallback to text".format(protocol))
        builder = _default_frame_builder
    return builder


class GimbalControl:
    """
    云台控制器。

    核心功能:
      1. 像素坐标 → 光轴偏角 (针孔模型)
      2. 偏角 → 串口发送给 MCU
    """

    def __init__(self, serial, camera_matrix, frame_builder=None):
        """
        初始化云台控制器。

        Args:
            serial: SerialComm 实例
            camera_matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            frame_builder: (yaw_rad, pitch_rad) -> bytes
                           默认: 文本协议 "Y:xxx,P:xxx\\n"
                           可选: _binary_frame_builder
        """
        self._serial = serial
        self._camera_matrix = camera_matrix
        self._frame_builder = frame_builder if frame_builder else _default_frame_builder

        self._last_yaw = 0.0
        self._last_pitch = 0.0

    def pixel_to_angle(self, px, py):
        """
        针孔模型: 像素坐标 → 光轴偏角。

        yaw   = atan2(px - cx, fx)    ← 水平偏角
        pitch = atan2(py - cy, fy)    ← 垂直偏角

        Args:
            px, py: 目标像素坐标 (float, 支持亚像素)

        Returns:
            (yaw_rad, pitch_rad) 弧度
        """
        return pixel_to_angle(px, py, self._camera_matrix)

    def aim_at(self, px, py):
        """
        一键瞄准: 像素坐标 → 角度 → 串口发送。

        Args:
            px, py: 目标像素坐标

        Returns:
            (yaw, pitch) 弧度值
        """
        yaw, pitch = self.pixel_to_angle(px, py)
        self._last_yaw = yaw
        self._last_pitch = pitch
        self._send_angle(yaw, pitch)
        return (yaw, pitch)

    def send_angle(self, yaw, pitch):
        """
        直接发送角度 (绕过像素映射)。

        Args:
            yaw, pitch: 目标角度 (弧度)
        """
        self._last_yaw = yaw
        self._last_pitch = pitch
        self._send_angle(yaw, pitch)

    def _send_angle(self, yaw, pitch):
        """内部: 构建帧并发送"""
        frame = self._frame_builder(yaw, pitch)
        self._serial.send(frame)

    def last_angle(self):
        """
        返回上次发送的角度。

        Returns:
            (yaw_rad, pitch_rad)
        """
        return (self._last_yaw, self._last_pitch)

    def close(self):
        """释放资源"""
        self._serial.close()
