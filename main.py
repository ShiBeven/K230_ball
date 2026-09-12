"""
K230 Vision Framework 主入口。

主循环: 加载配置 → 初始化各模块 → 采集 → 推理 → 用户逻辑 → 云台 → 显示。
自定义处理逻辑写在 user_process() 中; 复杂场景可跳过本文件, 直接组合各模块
(参考 recipes/ 目录下的配方)。
"""

import time
import gc

from camera import Camera
from color_detector import make_detector
from display import Display
from serial_comm import SerialComm
from gimbal_control import GimbalControl, frame_builder_from_config
from utils.config_loader import load_config


def user_process(img, detections):
    """
    用户自定义视觉处理逻辑。

    Args:
        img: K230 image 对象 (当前帧)
        detections: KPU 检测结果 [[class_name, conf, x, y, w, h], ...]
                    KPU 禁用时为 None

    Returns:
        任意用户数据 (可选, 被忽略)
    """
    # ---- 用户代码开始 ----

    # 示例: 打印检测结果
    # if detections:
    #     for d in detections:
    #         print("  {}: conf={:.2f} @ ({:.0f}, {:.0f})".format(
    #             d[0], d[1], d[2] + d[4]/2, d[3] + d[5]/2))

    # 示例: 找到置信度最高的目标, 发送给云台
    # if detections:
    #     best = max(detections, key=lambda d: d[1])
    #     cx = best[2] + best[4] / 2.0
    #     cy = best[3] + best[5] / 2.0
    #     return (cx, cy)  # 返回目标像素坐标

    # ---- 用户代码结束 ----
    return None


# ============================================================
# 主循环
# ============================================================

def main():
    print("=" * 50)
    print("  K230 Vision Framework")
    print("=" * 50)

    # 1. 加载配置
    config = load_config("config.json")
    print("[Main] Config loaded")

    # 2. 初始化摄像头
    cam = Camera(config)
    print("[Main] Camera initialized: {}x{} @ {}fps".format(
        cam._width, cam._height, cam._fps))

    # 3. 初始化检测器 (KPU 神经网络 或 纯视觉, 由 config 决定)
    #    make_detector: kpu.enabled=true 走神经网络; 跑不通或改成 false 时
    #    自动回退到 vision 段的颜色识别。返回对象的 detect(img) 接口一致。
    kpu = make_detector(config)  # 变量名沿用 kpu, 但可能是 ColorDetector

    # 4. 初始化显示 (可选)
    disp = None
    if config.get("display", {}).get("enabled", True):
        try:
            disp = Display(config)
            print("[Main] Display initialized")
        except Exception as e:
            print("[Main] Display init failed: {}".format(e))

    # 5. 初始化串口 (可选)
    ser = None
    if config.get("serial", {}).get("enabled", False):
        try:
            ser_cfg = config["serial"]
            ser = SerialComm(
                uart_id=ser_cfg.get("uart", 2),
                baudrate=ser_cfg.get("baudrate", 115200),
                tx_pin=ser_cfg.get("tx_pin", 4),
                rx_pin=ser_cfg.get("rx_pin", 5),
            )
            print("[Main] Serial initialized")
        except Exception as e:
            print("[Main] Serial init failed: {}".format(e))

    # 6. 初始化云台控制 (可选)
    gimbal = None
    if config.get("gimbal", {}).get("enabled", False) and ser is not None:
        try:
            calib = config.get("calibration", {})
            camera_matrix = calib.get("camera_matrix",
                                       [1838.88, 0, 707.29,
                                        0, 1840.54, 527.68,
                                        0, 0, 1])
            # 按 config 的 gimbal.protocol 选帧格式 (text / binary)
            gimbal = GimbalControl(
                ser, camera_matrix,
                frame_builder=frame_builder_from_config(config))
            print("[Main] Gimbal control initialized")
        except Exception as e:
            print("[Main] Gimbal init failed: {}".format(e))

    print("[Main] Entering main loop...")
    print("-" * 50)

    # 7. 主循环
    target_pixel = None
    frame_count = 0
    fps_timer = time.ticks_ms()

    try:
        while True:
            # 采集一帧
            img = cam.snapshot()
            frame_count += 1

            # KPU 推理
            detections = None
            if kpu is not None:
                try:
                    detections = kpu.detect(img)
                except Exception as e:
                    print("[Main] KPU detect error: {}".format(e))

            # 用户自定义处理
            try:
                result = user_process(img, detections)
                if result is not None and len(result) == 2:
                    target_pixel = result
            except Exception as e:
                print("[Main] user_process error: {}".format(e))

            # 云台瞄准
            if gimbal is not None and target_pixel is not None:
                try:
                    gimbal.aim_at(target_pixel[0], target_pixel[1])
                except Exception as e:
                    print("[Main] Gimbal aim error: {}".format(e))

            # 显示
            if disp is not None:
                disp.show(img, detections)

            # FPS 统计 (每秒输出一次)
            elapsed = time.ticks_diff(time.ticks_ms(), fps_timer)
            if elapsed >= 1000:
                fps = (frame_count * 1000.0) / elapsed
                print("[FPS] {:.1f}  Camera: {:.1f}".format(
                    fps, cam.get_fps()))
                frame_count = 0
                fps_timer = time.ticks_ms()

            # GC (防止 MicroPython 内存碎片)
            if frame_count % 100 == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")

    finally:
        # 清理
        print("[Main] Shutting down...")
        if gimbal:
            gimbal.close()
        if ser:
            ser.close()
        if disp:
            disp.close()
        if kpu:
            kpu.deinit()
        cam.close()
        print("[Main] Done.")


if __name__ == "__main__":
    main()
