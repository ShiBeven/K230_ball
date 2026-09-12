"""
摄像头模块 —— K230 media.sensor 封装。

初始化顺序不可调换(官方约定):
    Sensor(w,h) → reset() → set_framesize() → set_pixformat()
    → Display.init()(由 display.py 负责) → MediaManager.init() → run()

MediaManager.init() 全局只调一次, 本模块延迟到首次 snapshot() 时执行 ——
因为它必须晚于 Display.init(), 而框架的构造顺序是 Camera 先于 Display。
"""

import time
import os
from media.sensor import Sensor
from media.media import MediaManager

from utils.config_loader import load_config


class Camera:
    """
    K230 摄像头封装 (新版 media.sensor API)。
    对外方法 (snapshot/close/get_fps/set_fps 等) 与旧版保持一致,
    上层代码 (main.py / recipes / detector) 无需改动。
    """

    # 分辨率预设名 -> (宽, 高)
    _FRAMESIZE = {
        "QVGA": (320, 240),
        "VGA": (640, 480),
        "SVGA": (800, 600),
        "HD": (1280, 720),
        "FHD": (1920, 1080),
    }

    def __init__(self, config=None, config_path="config.json"):
        """
        初始化摄像头。

        Args:
            config: 配置 dict (优先), 为 None 时从 config_path 加载
            config_path: JSON 配置文件路径
        """
        if config is None:
            config = load_config(config_path)
        camera_cfg = config.get("camera", {})

        self._width = camera_cfg.get("width", 640)
        self._height = camera_cfg.get("height", 480)
        self._fps = camera_cfg.get("fps", 60)
        pixformat = camera_cfg.get("pixformat", "RGB565")

        # 像素格式映射 (新版为 Sensor 类属性)
        fmt_map = {
            "RGB565": Sensor.RGB565,
            "GRAYSCALE": Sensor.GRAYSCALE,
        }
        # YUV422 部分固件用 Sensor.YUV422, 不确定时回退 RGB565
        if hasattr(Sensor, "YUV422"):
            fmt_map["YUV422"] = Sensor.YUV422
        self._pixformat = fmt_map.get(pixformat, Sensor.RGB565)

        # 1. 构造 Sensor 对象 (指定输出通道尺寸)
        self._sensor = Sensor(width=self._width, height=self._height)

        # 2. 复位
        self._sensor.reset()

        # 3. 镜像/翻转 (在 set_framesize 之前设)
        self._sensor.set_hmirror(camera_cfg.get("hmirror", False))
        self._sensor.set_vflip(camera_cfg.get("vflip", False))

        # 4. 输出尺寸 + 像素格式 (chn0)
        self._sensor.set_framesize(width=self._width, height=self._height)
        self._sensor.set_pixformat(self._pixformat)

        # 5/6. MediaManager.init() + sensor.run() 延迟到首次 snapshot() 执行。
        #      原因: 官方要求 Display.init() 必须在 MediaManager.init() 之前;
        #      而框架构造顺序是 Camera 先于 Display。延迟启动可保证
        #      Display 初始化完成后再 init 媒体缓冲, 顺序正确, 上层无需改动。
        self._started = False

        # 图传(可选): 挂在 sensor 第二路输出通道上, 主循环零参与。
        # 由上层用 attach_stream() 注入, 本模块只负责在正确的时机回调它 ——
        # 因为只有这里知道 MediaManager.init() 到底什么时候发生。
        self._stream = None

        # FPS 统计
        self._fps_start = time.ticks_ms()
        self._fps_count = 0
        self._current_fps = 0.0

    def attach_stream(self, stream):
        """
        注入图传对象 (可选)。

        close() 时连带关闭图传(释放 socket 和热点), 采集本身不涉及。
        图传的 start() 由上层自己调, offer() 由上层每帧调 —— 图传是同步编码的,
        不涉及媒体管线时序, 所以不需要 camera 代为回调。
        """
        self._stream = stream

    def _ensure_started(self):
        """首次采集前启动媒体管线 (幂等, 只执行一次)"""
        if self._started:
            return
        MediaManager.init()   # 全局媒体缓冲, 全程只调用一次
        self._sensor.run()
        self._started = True

    def snapshot(self):
        """
        采集一帧图像。

        Returns:
            K230 image 对象
        """
        self._ensure_started()
        img = self._sensor.snapshot()

        # FPS 统计
        self._fps_count += 1
        elapsed = time.ticks_diff(time.ticks_ms(), self._fps_start)
        if elapsed >= 1000:
            self._current_fps = (self._fps_count * 1000.0) / elapsed
            self._fps_count = 0
            self._fps_start = time.ticks_ms()

        return img

    def set_fps(self, fps):
        """设置采集帧率 (新版通过 set_framerate, 部分固件无此方法则忽略)"""
        self._fps = int(fps)
        if hasattr(self._sensor, "set_framerate"):
            try:
                self._sensor.set_framerate(self._fps)
            except Exception:
                pass

    def get_fps(self):
        """
        获取实际帧率。
        """
        return self._current_fps

    def width(self):
        """图像宽度"""
        return self._width

    def height(self):
        """图像高度"""
        return self._height

    def close(self):
        """
        释放摄像头资源。

        按官方例程要求的顺序清理:
            [图传] -> sensor.stop() -> os.exitpoint(ENABLE_SLEEP) -> MediaManager.deinit()
        (Display.deinit() 由 display.py 的 close() 负责, 应在本方法之前调用。)
        图传要先关: 它占着编码器和 sensor 的绑定, 排在 sensor.stop() 之后
        会让搬运线程对着已停的 sensor 取流。
        """
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
        try:
            self._sensor.stop()
        except Exception:
            pass
        try:
            os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
            time.sleep_ms(100)
        except Exception:
            pass
        try:
            MediaManager.deinit()
        except Exception:
            pass
        self._started = False
