"""
Recipe 空白模板。复制本文件 → 重命名 → 在 user_loop 处改写 → 运行。
"""

from camera import Camera
from display import Display
from utils.config_loader import load_config


def main():
    config = load_config("config.json")

    cam = Camera(config)
    disp = Display(config) if config.get("display", {}).get("enabled", True) else None

    """
    # ---- 按需初始化其他模块 ----
    # KPU
    from kpu_tools import KPUModel
    kpu = KPUModel.from_config(config)  # 或 None

    # 串口
    from serial_comm import SerialComm
    ser = SerialComm(uart_id=2, baudrate=115200, tx_pin=4, rx_pin=5)

    # 云台
    from gimbal_control import GimbalControl
    gimbal = GimbalControl(ser, config["calibration"]["camera_matrix"])
    """

    print("[Template] Starting...")
    print("[Template] Press Ctrl+C to stop")

    try:
        while True:
            # ---- 你的视觉逻辑 ----
            img = cam.snapshot()

            # TODO: 在这里写你的处理代码
            # 示例: 颜色识别
            # blobs = img.find_blobs([threshold], pixels_threshold=100)
            # if blobs:
            #     cx, cy = blobs[0].cx(), blobs[0].cy()
            #     img.draw_cross(int(cx), int(cy), color=(0, 255, 0))

            if disp:
                disp.show(img)

    except KeyboardInterrupt:
        print("\n[Template] Stopped by user")
    finally:
        if disp:
            disp.close()
        cam.close()
        print("[Template] Done.")


if __name__ == "__main__":
    main()
