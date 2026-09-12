# -*- coding: utf-8 -*-
"""
Recipe: 统一检测入口 —— 由 config.detector.mode 切换检测路线。

    "vision" 纯视觉 find_blobs(LAB 阈值), 可选单目测距
    "ai"     神经网络 PipeLine/DetectionApp
    "hybrid" 预留未实现

两条路线输出统一六元组 [class_name, conf, x, y, w, h]。
注意坐标空间不同: vision = 640x480 传感器空间 / ai = 1280x720 rgb888p 空间。

测距(仅 vision): D = 真实尺寸 × fx / 像素径; dx = (u − cx) × D / fx。
像素径取 sqrt(色块像素数), 不用框宽高(误差随距离放大)。
AI 框偏大 1.5~1.8 倍, 故 ai 模式忽略 ranging。

串口(仅 vision): 每帧最多一包 8 字节 track 帧, 内容为相对画面中心的
dx/dy 与像素数; 目标丢失时不发送, 由 MCU 超时判丢。
"""

import time
import os
import math

from utils.config_loader import load_config

VALID_MODES = ("vision", "ai", "hybrid")


# ---------------- 共用小函数(两条路线的下游) ----------------

def pick_best(dets):
    """六元组列表里选置信度最高的一条(手写循环, 不依赖 max 的 key 参数)"""
    best = None
    for d in dets:
        if best is None or d[1] > best[1]:
            best = d
    return best


def fmt_det(det):
    """六元组 -> 终端打印用的短字符串"""
    return "%s %.2f @(%.0f,%.0f) %ux%u" % (
        det[0], det[1], det[2], det[3], int(det[4]), int(det[5]))


# ---------------- vision 路线 ----------------

def loop_vision(config):
    """纯视觉: 框架 Camera/Display + find_blobs + 可选单目测距。

    直接使用 find_blobs 而非 ColorDetector.detect(): 测距需要 blob.pixels()
    (真实像素数), 六元组中不包含 —— 六元组仅作为下游输出契约。
    """
    from camera import Camera        # 延迟导入: ai 模式不碰这套媒体栈
    from display import Display

    vcfg = config.get("vision", {})
    raw = vcfg.get("thresholds") or []
    thresholds = [tuple(t) for t in raw]
    if not thresholds:
        print("[Detect] 错误: config 的 vision.thresholds 为空, 没法检测。")
        print("[Detect] 用 IDE 阈值编辑器(工具→机器视觉→阈值编辑器)调好后填进去。")
        return
    labels = vcfg.get("labels") or []
    pixels_threshold = vcfg.get("pixels_threshold", 100)
    area_threshold = vcfg.get("area_threshold", 100)
    merge = vcfg.get("merge", True)

    # ---- 测距配置(可选) ----
    rcfg = config.get("ranging", {})
    ranging_on = rcfg.get("enabled", False)
    size_mm = float(rcfg.get("target_size_mm", 25.0))
    shape = rcfg.get("shape", "square")
    fx = cx0 = None
    if ranging_on:
        cam_mat = config.get("calibration", {}).get("camera_matrix", [])
        if len(cam_mat) >= 9 and cam_mat[0] > 0:
            fx, cx0 = float(cam_mat[0]), float(cam_mat[2])
        else:
            print("[Detect] 警告: calibration.camera_matrix 缺失/非法, 测距已禁用")
            ranging_on = False
        cam_cfg = config.get("camera", {})
        if (cam_cfg.get("width", 640), cam_cfg.get("height", 480)) != (640, 480):
            print("[Detect] 警告: 内参按 640x480 标定, 当前分辨率不同, 测距结果不可信!")

    # ---- 串口发包配置(可选) ----
    scfg = config.get("serial", {})
    ser = None
    n_sent = 0
    if scfg.get("enabled", False):
        from serial_comm import SerialComm      # 延迟导入: 不用串口就不碰 UART
        ser = SerialComm(uart_id=scfg.get("uart", 2),
                         baudrate=scfg.get("baudrate", 115200),
                         tx_pin=scfg.get("tx_pin", 11),
                         rx_pin=scfg.get("rx_pin", 12),
                         head=scfg.get("head"),
                         end=scfg.get("end"))
        if not ser.is_open():
            print("[Detect] 警告: 串口没开成, 本次不发包(检测照常跑)")
            ser = None

    cam = Camera(config)
    disp = Display(config)
    clock = time.clock()

    print("[Detect] vision 模式启动 | thresholds =", thresholds)
    if ranging_on:
        print("[Detect] 测距开: 目标 %.1fmm %s | fx=%.2f cx=%.2f"
              % (size_mm, shape, fx, cx0))
    if ser is not None:
        print("[Detect] 串口发包开: 每检到目标发 8 字节 track 帧, 丢目标时不发")
    print("[Detect] IDE 点停止或 Ctrl+C 退出")

    try:
        while True:
            os.exitpoint()
            clock.tick()
            img = cam.snapshot()
            img_area = float(img.width() * img.height())

            # 逐类找色块; 六元组喂下游, blob 对象留给测距
            dets = []
            best_blob = None
            best_label = None
            for i, thr in enumerate(thresholds):
                label = labels[i] if i < len(labels) else "color_%d" % i
                blobs = img.find_blobs([thr],
                                       pixels_threshold=pixels_threshold,
                                       area_threshold=area_threshold,
                                       merge=merge)
                if not blobs:
                    continue
                for b in blobs:
                    conf = min(1.0, (b.w() * b.h()) / img_area * 4.0)
                    dets.append([label, conf, float(b.x()), float(b.y()),
                                 float(b.w()), float(b.h())])
                    if best_blob is None or b.pixels() > best_blob.pixels():
                        best_blob = b
                        best_label = label

            fps = clock.fps()

            if best_blob is not None:
                b = best_blob
                x, y, w, h = b.x(), b.y(), b.w(), b.h()
                u = x + w / 2.0
                v = y + h / 2.0
                img.draw_rectangle(x, y, w, h, color=(0, 255, 0), thickness=2)
                img.draw_cross(int(u), int(v), color=(255, 255, 0), size=8)
                img.draw_string_advanced(6, 4, 22, "%s x%d" % (best_label, len(dets)),
                                         color=(0, 255, 0))

                if ranging_on:
                    # 像素径: 方形=√像素数, 圆形=√(4×像素数/π)
                    if shape == "circle":
                        px_d = math.sqrt(4.0 * b.pixels() / math.pi)
                    else:
                        px_d = math.sqrt(b.pixels())
                    if px_d > 0:
                        dist = size_mm * fx / px_d
                        dx = (u - cx0) * dist / fx
                        img.draw_string_advanced(6, 30, 22,
                                                 "D=%.0fmm dx=%+.0fmm" % (dist, dx),
                                                 color=(255, 200, 0))
                        print("d_px=%5.1f | D=%6.1fmm dx=%+6.1fmm | %s | fps=%.1f"
                              % (px_d, dist, dx, fmt_det(pick_best(dets)), fps))
                else:
                    print("  %s | fps=%.1f" % (fmt_det(pick_best(dets)), fps))

                # ---- 发给电控: 相对画面中心的像素偏移 + 色块像素个数 ----
                if ser is not None:
                    off_x = int(u - img.width() / 2.0)     # 左负右正
                    off_y = int(v - img.height() / 2.0)    # 上负下正
                    if ser.send_track(off_x, off_y, b.pixels()):
                        n_sent += 1
                    img.draw_string_advanced(
                        6, img.height() - 52, 20,
                        "TX %+d,%+d px%d" % (off_x, off_y, b.pixels()),
                        color=(0, 255, 255))
            else:
                img.draw_string_advanced(6, 4, 24, "no target", color=(255, 0, 0))
                print("  no target | fps=%.1f" % fps)
                # 目标丢失: 刻意什么都不发, 电控靠超时判丢(负荷最小)

            img.draw_string_advanced(6, img.height() - 28, 20,
                                     "FPS: %.1f" % fps, color=(255, 255, 255))
            disp.show(img)

    except KeyboardInterrupt:
        print("[Detect] 手动停止")
    finally:
        # 顺序: 先 Display.deinit 再 camera(内部 MediaManager.deinit), 不可颠倒
        disp.close()
        cam.close()
        if ser is not None:
            ser.close()
            print("[Detect] 串口已关, 本次共发 %d 包" % n_sent)
        print("[Detect] vision 已退出, 资源已释放")


# ---------------- ai 路线 ----------------

def loop_ai(config):
    """AI: PipeLine/DetectionApp 独占媒体栈(硬约束), 统计写法同绿框版"""
    if config.get("ranging", {}).get("enabled", False):
        print("[Detect] 警告: AI 框虚胖不能测尺寸, ai 模式忽略 ranging 配置")

    from ai_detector import AIDetector   # 延迟导入: vision 模式不碰 libs.PipeLine

    det = AIDetector(config)
    n_frame = 0
    n_hit = 0
    t_mark = time.ticks_ms()

    try:
        while True:
            dets = det.step()
            det.draw_and_show(dets)

            n_frame += 1
            if dets:
                n_hit += 1
            if n_frame % 30 == 0:      # 每 30 帧汇报一次, 别刷屏(AI 路打印密会卡)
                dt_ms = time.ticks_diff(time.ticks_ms(), t_mark)
                best = pick_best(dets)
                print("累计 %d 帧 | 检出 %d 帧 (%.0f%%) | FPS %.1f | %s"
                      % (n_frame, n_hit, n_hit * 100.0 / n_frame,
                         30000.0 / dt_ms,
                         fmt_det(best) if best else "no target"))
                t_mark = time.ticks_ms()
    except KeyboardInterrupt:
        print("[Detect] 手动停止")
    finally:
        det.close()
        if n_frame:
            print("[Detect] 总计 %d 帧, 检出 %d 帧 (%.0f%%)"
                  % (n_frame, n_hit, n_hit * 100.0 / n_frame))
        print("[Detect] ai 已退出, 资源已释放")


# ---------------- 入口分发 ----------------

def main():
    config = load_config("config.json")
    mode = config.get("detector", {}).get("mode", "vision")

    if mode == "vision":
        loop_vision(config)
    elif mode == "ai":
        loop_ai(config)
    elif mode == "hybrid":
        print("[Detect] hybrid(AI+视觉同帧混用)尚未实现 —— 预留位。")
        print("[Detect] DetectionApp 架构下已验证不可行(通道格式互斥),")
        print("[Detect] 将来如需, 按 AIBase+Ai2d 自建方案立项。")
    else:
        print("[Detect] 错误: detector.mode = %r 不认识。" % mode)
        print("[Detect] 合法值: %s (改 config.json 后重跑)" % (VALID_MODES,))


if __name__ == "__main__":
    main()
