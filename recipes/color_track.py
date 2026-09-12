"""
Recipe: 颜色追踪 —— image.find_blobs() 做 LAB 阈值定位, 不依赖 KPU。

用法: 用 IDE 阈值编辑器调出目标 LAB 阈值填入 config.vision.thresholds, 然后运行。
"""

import time

from camera import Camera
from display import Display
from utils.config_loader import load_config
from image_utils import draw_crosshair


# ---- 颜色阈值 (LAB 色彩空间) ----
# 格式: (L_min, L_max, A_min, A_max, B_min, B_max)
# 默认从 config.json 的 vision.thresholds 读取(用 IDE 阈值编辑器调好后填进 config)。
# 若 config 里没有,才退回用下面这组示例值。
FALLBACK_THRESHOLDS = [
    (0, 100, -20, 30, -128, -20),   # 示例: 某种颜色
]


def main():
    config = load_config("config.json")

    # 优先用 config 里调好的阈值
    vision_cfg = config.get("vision", {})
    thresholds = vision_cfg.get("thresholds") or FALLBACK_THRESHOLDS
    # config 里是 list-of-list, find_blobs 需要 list-of-tuple
    thresholds = [tuple(t) for t in thresholds]
    pixels_threshold = vision_cfg.get("pixels_threshold", 100)
    area_threshold = vision_cfg.get("area_threshold", 100)
    merge = vision_cfg.get("merge", True)

    cam = Camera(config)
    disp = Display(config)

    print("[ColorTrack] Starting...")
    print("[ColorTrack] thresholds =", thresholds)
    print("[ColorTrack] Press Ctrl+C to stop")

    fps_clock = time.clock()

    try:
        while True:
            fps_clock.tick()
            img = cam.snapshot()

            # 查找色块
            blobs = img.find_blobs(thresholds,
                                   pixels_threshold=pixels_threshold,
                                   area_threshold=area_threshold,
                                   merge=merge)

            largest_blob = None
            for blob in blobs:
                if largest_blob is None or blob.area() > largest_blob.area():
                    largest_blob = blob

            if largest_blob:
                cx = largest_blob.cx()
                cy = largest_blob.cy()
                w = largest_blob.w()
                h = largest_blob.h()

                # 画检测框
                img.draw_rectangle(largest_blob.rect(), color=(255, 0, 0), thickness=2)

                # 画十字准星
                draw_crosshair(img, cx, cy, size=10, color=(0, 255, 0))

                # 质心坐标
                img.draw_string_advanced(int(cx) + 10, int(cy) - 20, 20,
                                "({:.0f},{:.0f})".format(cx, cy),
                                color=(255, 255, 0))

                print("  Blob: center=({:.1f},{:.1f}) area={} w={:.0f} h={:.0f}".format(
                    cx, cy, largest_blob.area(), w, h))

            # 帧率: 画到左上角 + 打印到终端
            fps = fps_clock.fps()
            img.draw_string_advanced(5, 5, 20, "FPS: {:.1f}".format(fps),
                                     color=(255, 255, 255))
            print("  FPS: {:.1f}".format(fps))

            if disp:
                disp.show(img)

    except KeyboardInterrupt:
        print("\n[ColorTrack] Stopped by user")
    finally:
        if disp:
            disp.close()
        cam.close()
        print("[ColorTrack] Done.")


if __name__ == "__main__":
    main()
