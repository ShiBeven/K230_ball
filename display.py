"""
显示模块 —— K230 media.display 封装, 由 config.display.enabled 开关控制。

config.display.type 取值:
    "virt" → Display.VIRT    CanMV IDE 取景窗(开发用)
    "hdmi" → Display.LT9611  HDMI 输出
    "lcd"  → Display.ST7701  LCD 屏 800x480(本项目使用)

MediaManager 的 init/deinit 由 camera.py 统一负责, 本模块不调用 ——
以此保证 Display.init() 先于 MediaManager.init() 的顺序要求。
"""

from media.display import Display as _Display

from utils.config_loader import load_config
from image_utils import draw_detections

# 官方 ST7701 LCD 触摸屏的物理分辨率, 屏就这一款, 固定值
LCD_WIDTH = 800
LCD_HEIGHT = 480

# 自启标志文件: 脱机自启(无 IDE 连接)时由引导代码创建。
# 用于判断是否可以退回 IDE 虚拟显示(脱机时 VIRT 无人取帧会阻塞主循环)。
AUTOSTART_FLAG = "/sdcard/.autostart_running"


def _is_autostart():
    """判断当前是否处于脱机自启模式(标志文件存在即视为是)。"""
    try:
        import os as _os
        _os.stat(AUTOSTART_FLAG)
        return True
    except Exception:
        return False


class Display:
    """
    K230 显示输出封装 (新版 media.display API)。

    支持 IDE虚拟(VIRT) / HDMI(LT9611) / LCD(ST7701) 三种输出。
    对外方法 (show/clear/close) 与旧版保持一致, 上层代码无需改动。
    """

    def __init__(self, config=None, config_path="config.json"):
        """
        初始化显示。

        Args:
            config: 配置 dict (优先)
            config_path: JSON 配置文件路径
        """
        if config is None:
            config = load_config(config_path)
        display_cfg = config.get("display", {})

        if not display_cfg.get("enabled", True):
            self._enabled = False
            return

        self._enabled = True
        disp_type = display_cfg.get("type", "virt")
        self._width = display_cfg.get("width", 640)
        self._height = display_cfg.get("height", 480)
        self._type = disp_type

        # 初始化 K230 display。
        # lcd 分支带 fallback: ST7701 初始化在屏不在时可能抛异常或卡住,
        # 包 try 保证"没接屏也能跑完整项目"。
        # ⚠️ to_ide 必须为 False: 脱机(无 IDE 连接)时没人取帧, 缓冲填满后
        # show_image 会永久阻塞主循环。需查看画面请使用板载 LCD 或图传。
        try:
            if disp_type == "hdmi":
                _Display.init(_Display.LT9611, width=self._width,
                              height=self._height, to_ide=False)
            elif disp_type == "lcd":
                # ST7701 物理分辨率固定 800x480, 不可配置。
                # config 的 width/height 仅对 virt/hdmi 生效; 摄像头保持 640x480,
                # 由 Display 自行缩放上屏。
                _Display.init(_Display.ST7701, width=LCD_WIDTH,
                              height=LCD_HEIGHT, to_ide=False)
            else:
                # 默认: IDE 虚拟显示 (无需外接屏幕)
                _Display.init(_Display.VIRT, width=self._width,
                              height=self._height, fps=100)
        except Exception as e:
            print("  [显示] %s 初始化失败(%r)" % (disp_type, e))
            if disp_type == "lcd":
                # VIRT 只能输出到 IDE 连接; 脱机自启时退回 VIRT 同样会阻塞,
                # 因此脱机时直接关闭显示, 只保留检测与串口。
                if _is_autostart():
                    print("       → 脱机自启中(没有 IDE) → **不退回 VIRT**, "
                          "彻底关显示, 只跑检测+串口。")
                    print("         (VIRT 只往 IDE 推帧, 脱机时必然堵住主循环)")
                    self._enabled = False
                    return
                print("       → 屏没插? 退回 IDE 虚拟显示, 项目照常跑。")
                print("         (接着 IDE 调试时走这个路径, 属正常)")
                try:
                    _Display.init(_Display.VIRT, width=self._width,
                                  height=self._height, fps=100)
                    self._type = "virt"
                except Exception as e2:
                    print("       → 连 VIRT 也失败(%r) → 彻底关显示, "
                          "只跑检测+串口" % (e2,))
                    self._enabled = False
            else:
                self._enabled = False

    def show(self, img, detections=None):
        """
        显示一帧, 可选叠加检测结果。

        Args:
            img: K230 image 对象
            detections: [[class_name, conf, x, y, w, h], ...] 或 None
        """
        if not self._enabled:
            return

        if detections:
            draw_detections(img, detections)

        # 运行中途拔屏时 show_image 可能抛异常; 连续失败则自动关闭显示,
        # 保证检测与串口继续运行。
        try:
            _Display.show_image(img)
            self._fail = 0
        except Exception as e:
            self._fail = getattr(self, "_fail", 0) + 1
            if self._fail == 1:
                print("  [显示] show 失败(%r), 再错几次就关显示继续跑" % (e,))
            if self._fail >= 10:
                print("  [显示] 连续失败 → 关闭显示, 检测与串口照常运行")
                self._enabled = False

    def clear(self):
        """清屏 (新版无直接清屏 API, 显示由下一帧覆盖, 此处空实现)"""
        pass

    def close(self):
        """释放显示资源"""
        if self._enabled:
            try:
                _Display.deinit()
            except Exception:
                pass
            self._enabled = False
