"""
串口通信模块 —— machine.UART 封装, 提供裸字节 / 8 字节定长帧 / CRC16 帧 / 文本行四种模式。

本项目以单向发送(视觉 → MCU)为主; recv 系列保留供诊断使用。
引脚复用必须先经 FPIOA 配置; 构造只接受 UART(id, baudrate)。
依赖: utils/crc.py
"""

import time
from machine import UART
from utils.crc import get_crc16, check_crc16

# ===================================================================
#  track8 定长跟随包 —— 通用目标跟随, 供 recipes/detect.py 使用
# ===================================================================
#     [0]    帧头
#     [1][2] cx   大端, 值 = dx + 32768   (相对画面中心横向偏移 px, 左负右正)
#     [3][4] cy   大端, 值 = dy + 32768   (相对画面中心纵向偏移 px, 上负下正)
#     [5][6] area 大端, 值 = 色块像素数 / 10
#     [7]    帧尾
#  帧头帧尾由 config.serial.head / end 配置(支持 "0xA5" 或 165 两种写法)。
# ===================================================================
TRACK8_HEAD_DEFAULT = 0xAA
TRACK8_END_DEFAULT = 0x55
TRACK8_LEN = 8

# ===================================================================
#  ball8 目标位置包 (视觉 → MCU, 单向)
# ===================================================================
#  与 track8 同构(8 字节定长、同帧头帧尾、同 +32768 偏移码), 仅字段含义不同,
#  电控侧 DMA 收包与头尾校验代码可直接复用。
#
#     [0]    帧头
#     [1][2] xpos 大端, 值 = err_0.01cm + 32768
#            err = 目标 X 坐标 − 当前位置, 单位 0.01cm, 负值表示目标在左侧
#            注意发送的是误差而非绝对位置, MCU 侧 PID 目标值恒为 0
#     [3][4] 保留 大端, 恒 = 32768 (还原后 0), 预留球速等扩展字段
#     [5][6] 质量 大端, 值 = 球色块像素数 / 10, 作粗略置信度
#     [7]    帧尾
#
#  单位取 0.01cm: 量化误差 0.005cm, 相对测量噪声(0.04~0.16cm)可忽略。
#
#  发送原则: 每帧最多一包; 无可信读数时不发送任何字节, 由 MCU 侧超时判丢。
# ===================================================================
BALL8_RESERVED = 32768          # [3][4] 的占位值, 还原后 = 0


def build_ball_packet(x_cm, pixels, head=TRACK8_HEAD_DEFAULT,
                      end=TRACK8_END_DEFAULT):
    """
    组一包 ball8 帧。纯函数, 不依赖串口。

    Args:
        x_cm: 误差 = 当前位置 − 目标位置, 单位 cm, 左负右正
        pixels: 目标色块像素数, 作粗略置信度
    Returns:
        bytes, 长度恒为 8
    """
    # 手写四舍五入: MicroPython 的 round() 是银行家舍入, 负值对称性不如此写法
    v = x_cm * 100.0
    xi = int(v + 0.5) if v >= 0 else int(v - 0.5)

    xp = xi + 32768             # 偏移码, 让负数塞进无符号 16 位
    if xp < 0:
        xp = 0
    elif xp > 65535:
        xp = 65535

    q = int(pixels) // 10
    if q < 0:
        q = 0
    elif q > 65535:
        q = 65535

    return bytes((
        head & 0xFF,
        (xp >> 8) & 0xFF, xp & 0xFF,
        (BALL8_RESERVED >> 8) & 0xFF, BALL8_RESERVED & 0xFF,
        (q >> 8) & 0xFF, q & 0xFF,
        end & 0xFF,
    ))


def _parse_byte(v, fallback):
    """宽容解析一个字节值: 支持 170 / "0xAA" / "0XAA" / "AA" 三种写法。"""
    if v is None:
        return fallback
    if isinstance(v, int):
        return v & 0xFF
    try:
        s = str(v).strip()
        if s.lower().startswith("0x"):
            return int(s, 16) & 0xFF
        return int(s, 0) & 0xFF
    except Exception:
        try:
            return int(str(v).strip(), 16) & 0xFF
        except Exception:
            return fallback


def build_track_packet(dx, dy, pixels, head=TRACK8_HEAD_DEFAULT,
                       end=TRACK8_END_DEFAULT):
    """
    组一包 track8 定长跟随帧 (纯函数, 不依赖串口, PC 上也能单测)。

    Args:
        dx: 相对画面中心横向偏移(px), 左负右正, 会被夹到 int16 范围
        dy: 相对画面中心纵向偏移(px), 上负下正
        pixels: 色块像素个数 (blob.pixels(), 不是 w*h)
    Returns:
        bytes, 长度恒为 8
    """
    # 偏移码 +32768: 让负数也能塞进无符号 16 位。电控那边 -32768 还原
    cx = int(dx) + 32768
    cy = int(dy) + 32768
    if cx < 0:
        cx = 0
    elif cx > 65535:
        cx = 65535
    if cy < 0:
        cy = 0
    elif cy > 65535:
        cy = 65535

    ar = int(pixels) // 10          # 面积压缩 10 倍才塞得进 16 位
    if ar < 0:
        ar = 0
    elif ar > 65535:
        ar = 65535

    return bytes((
        head & 0xFF,
        (cx >> 8) & 0xFF, cx & 0xFF,
        (cy >> 8) & 0xFF, cy & 0xFF,
        (ar >> 8) & 0xFF, ar & 0xFF,
        end & 0xFF,
    ))


class SerialComm:
    """
    通用串口收发封装, 提供裸字节 / 8 字节定长帧 / CRC16 帧 / 文本行四种模式。
    """

    def __init__(self, uart_id=2, baudrate=115200, tx_pin=5, rx_pin=6,
                 head=None, end=None):
        """
        初始化串口。

        Args:
            uart_id: UART 编号 (1/2/3/4; 0 为 REPL 控制台, 禁用)
            baudrate: 波特率
            tx_pin: 发送引脚 IO 编号 (UART2 = IO5, 对应排针 17 号针)
            rx_pin: 接收引脚 IO 编号 (UART2 = IO6, 对应排针 20 号针)
            head/end: 帧头帧尾, 支持 165 / "0xA5" 两种写法
        """
        self._uart_id = uart_id
        self._baudrate = baudrate
        self._tx_pin = tx_pin
        self._rx_pin = rx_pin
        self._head = _parse_byte(head, TRACK8_HEAD_DEFAULT)
        self._end = _parse_byte(end, TRACK8_END_DEFAULT)
        self._uart = None

        self._open()

    def _open(self):
        """
        打开串口。

        新 CanMV API 分两步(不要改回旧写法):
          1. FPIOA().set_function(pin, FPIOA.UART{n}_TXD/RXD) 配引脚复用
          2. UART(id, baudrate) —— 不吃 tx_pin/rx_pin 关键字
        """
        if self._uart_id == 0:
            print("[Serial] REFUSED: UART0 (pin38/39) is the REPL console")
            self._uart = None
            return

        # ---- 第一步: 引脚复用 ----
        try:
            from machine import FPIOA
            fpioa = FPIOA()
            tx_name = "UART{}_TXD".format(self._uart_id)
            rx_name = "UART{}_RXD".format(self._uart_id)
            fpioa.set_function(self._tx_pin, getattr(FPIOA, tx_name))
            fpioa.set_function(self._rx_pin, getattr(FPIOA, rx_name))
            print("[Serial] FPIOA: pin{}={} pin{}={}".format(
                self._tx_pin, tx_name, self._rx_pin, rx_name))
        except Exception as e:
            # 老固件没有 FPIOA, 直接往下走让 UART() 自己碰运气
            print("[Serial] FPIOA setup skipped/failed: {}".format(e))

        # ---- 第二步: 开串口 ----
        try:
            self._uart = UART(self._uart_id, self._baudrate)
            print("[Serial] Opened UART{} at {} baud (head=0x{:02X} end=0x{:02X})".format(
                self._uart_id, self._baudrate, self._head, self._end))
        except Exception as e:
            print("[Serial] Failed to open UART{}: {}".format(
                self._uart_id, e))
            self._uart = None

    def is_open(self):
        """检查串口是否已打开"""
        return self._uart is not None

    # ---- 基础收发 ----

    def send(self, data):
        """
        发送字节数据。

        Args:
            data: bytes
        """
        if self._uart is None:
            print("[Serial] Cannot send: UART not open")
            return False
        try:
            self._uart.write(data)
            return True
        except Exception as e:
            print("[Serial] Write failed: {}".format(e))
            return False

    def recv(self, nbytes, timeout_ms=100):
        """
        接收指定字节数。

        Args:
            nbytes: 预期接收字节数
            timeout_ms: 超时 (毫秒)

        Returns:
            bytes, 超时或失败返回 b""
        """
        if self._uart is None:
            return b""

        deadline = time.ticks_add(time.ticks_ms(), timeout_ms)
        buf = bytearray()

        while len(buf) < nbytes:
            remaining = time.ticks_diff(deadline, time.ticks_ms())
            if remaining <= 0:
                break

            if self._uart.any():
                chunk = self._uart.read(min(nbytes - len(buf),
                                            self._uart.any()))
                if chunk:
                    buf.extend(chunk)
            else:
                time.sleep_ms(1)

        return bytes(buf)

    # ---- track8 定长跟随帧 (电控 MSPM0 现用协议) ----

    def send_track(self, dx, dy, pixels):
        """
        发一包 8 字节跟随帧给电控。

        一帧只发一次, 帧与帧之间天然有间隔 —— 这很重要:
        对方 DMA 是"凑够 8 字节算一包", 万一字节流错位, 只能靠线上静默
        触发它的 RX_TIMEOUT 分支重新对齐。所以绝不要把多包塞进一次 write 连灌。

        目标丢失时【不要调用本方法】—— 一个字节都不发, 电控靠超时判丢。
        这是负荷最小的做法, 也避免了协议里没有有效位的问题。

        Args:
            dx: 相对画面中心横向偏移(px), 左负右正
            dy: 相对画面中心纵向偏移(px), 上负下正
            pixels: 色块像素个数 (blob.pixels())
        Returns:
            True 发送成功
        """
        return self.send(build_track_packet(dx, dy, pixels,
                                           self._head, self._end))

    # ---- ball8 目标位置帧 ----

    def send_ball(self, x_cm, pixels):
        """
        发送一包 8 字节目标位置帧。

        Args:
            x_cm: 目标相对当前位置的偏移(cm), 左负右正
            pixels: 球色块像素个数, 作粗略置信度
        Returns:
            True 发送成功

        注意: 每帧最多一包; 无可信读数时不发送任何字节。
        MCU 侧按定长收包, 字节流错位后只能靠线上静默超时重新对齐,
        因此绝不连续灌包, 宁可让对端超时也不能发送错误位置。
        """
        return self.send(build_ball_packet(x_cm, pixels,
                                          self._head, self._end))

    # ---- CRC16 帧收发 ----

    def send_frame(self, data):
        """
        发送带 CRC16 校验的帧。
            crc16 = get_crc16(data, len)
            serial_.write(data + crc16_bytes)

        Args:
            data: 帧载荷 bytes (不含 CRC)
        """
        crc = get_crc16(data)
        frame = bytearray(data)
        frame.append(crc & 0xFF)         # CRC16 低字节
        frame.append((crc >> 8) & 0xFF)  # CRC16 高字节
        return self.send(bytes(frame))

    def recv_frame(self, timeout_ms=100):
        """
        接收带 CRC16 校验的帧。
            读完整帧 → check_crc16() → 通过则处理

        Returns:
            bytes (不含 CRC), 校验失败或超时返回 None
        """
        if self._uart is None:
            return None

        # 先读两字节确认长度 (简单的帧头检测)
        # 实际使用时用户需根据协议定制
        deadline = time.ticks_add(time.ticks_ms(), timeout_ms)
        buf = bytearray()

        # 等待至少 2 字节 (小帧至少需要载荷+CRC)
        while len(buf) < 2:
            remaining = time.ticks_diff(deadline, time.ticks_ms())
            if remaining <= 0:
                return None
            if self._uart.any():
                buf.extend(self._uart.read(self._uart.any()))
            else:
                time.sleep_ms(1)

        # 继续接收直到超时
        while True:
            remaining = time.ticks_diff(deadline, time.ticks_ms())
            if remaining <= 0:
                break
            if self._uart.any():
                buf.extend(self._uart.read(self._uart.any()))
            else:
                time.sleep_ms(5)

        if len(buf) < 4:  # 至少 2 字节载荷 + 2 字节 CRC
            return None

        if check_crc16(bytes(buf)):
            return bytes(buf[:-2])  # 去掉 CRC 返回载荷
        return None

    # ---- 文本行收发 ----

    def send_line(self, s):
        """
        发送一行文本 (以 \\n 结尾)。

        Args:
            s: 字符串
        """
        if not s.endswith("\n"):
            s += "\n"
        return self.send(s.encode())

    def recv_line(self, timeout_ms=100):
        """
        接收一行文本 (以 \\n 为界)。

        Returns:
            str (不含换行符), 超时返回 None
        """
        if self._uart is None:
            return None

        deadline = time.ticks_add(time.ticks_ms(), timeout_ms)
        buf = bytearray()

        while True:
            remaining = time.ticks_diff(deadline, time.ticks_ms())
            if remaining <= 0:
                return None
            if self._uart.any():
                ch = self._uart.read(1)
                if ch:
                    if ch == b"\n":
                        return bytes(buf).decode("utf-8", "ignore").strip()
                    buf.extend(ch)
            else:
                time.sleep_ms(1)

    # ---- 断线重连 ----

    def reconnect(self):
        """
        断线重连。

        尝试重新打开串口, 最多 10 次。
        """
        max_retry = 10
        for i in range(max_retry):
            if self._uart is not None:
                try:
                    self._uart.deinit()
                except Exception:
                    pass
                self._uart = None

            print("[Serial] Reconnecting... (attempt {}/{})".format(
                i + 1, max_retry))

            self._open()
            if self._uart is not None:
                return True

            time.sleep(1)

        return False

    def close(self):
        """关闭串口"""
        if self._uart is not None:
            try:
                self._uart.deinit()
            except Exception:
                pass
            self._uart = None
