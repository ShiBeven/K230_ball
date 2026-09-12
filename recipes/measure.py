# -*- coding: utf-8 -*-
"""
Recipe: 单目视觉测量 —— 基于已标定内参, 一个 fx 导出五种量。

    1. 距离 D      已知真实尺寸时反推     D = 尺寸 × fx / 像素径
    2. 真实尺寸    已知距离时反推         尺寸 = 像素径 × D / fx
    3. 横纵偏移    偏离画面中心的毫米数   dx = (u − cx) × D / fx
    4. 角度        偏离光轴的度数(无需 D) yaw = atan((u − cx) / fx)
    5. 两目标间距  同深度两色块的真实距离 L = 像素距 × D / fx

方向由 config.measure.known 控制:
"size" 已知尺寸求距离 / "distance" 已知距离求尺寸。

注意:
  - 像素径取 sqrt(色块像素数), 不用框宽高 —— 前者误差小且不随距离漂移。
  - 目标尽量置于画面中央(未做去畸变)。
  - 内参仅对 640x480 有效, 改分辨率须按比例换算。
"""

import time
import os
import math

from utils.config_loader import load_config

VALID_KNOWN = ("size", "distance")


# ---------------- 小工具 ----------------

def _center(b):
    """色块中心。优先用 blob 自带质心 cx()/cy(), 没有就退回框中心。"""
    try:
        return float(b.cx()), float(b.cy())
    except (AttributeError, TypeError):
        return b.x() + b.w() / 2.0, b.y() + b.h() / 2.0


def top_two(blobs):
    """按像素数选最大的两个(手写循环, 不依赖 sort 的 key 参数在此固件可用)。"""
    a = b = None
    for x in blobs:
        n = x.pixels()
        if a is None or n > a.pixels():
            a, b = x, a
        elif b is None or n > b.pixels():
            b = x
    out = []
    if a is not None:
        out.append(a)
    if b is not None:
        out.append(b)
    return out


def pixel_diameter(blob, shape):
    """等效像素径。方形=√面积, 圆形=√(4×面积/π)(把面积还原成直径)。"""
    n = blob.pixels()
    if n <= 0:
        return 0.0
    if shape == "circle":
        return math.sqrt(4.0 * n / math.pi)
    return math.sqrt(n)


class Measurer:
    """内参 + 一个已知量 → 五种测量。纯数学, 无硬件依赖, 可在 PC 上单测。"""

    def __init__(self, fx, fy, cx, cy, known, size_mm, distance_mm, shape):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.known = known
        self.size_mm = float(size_mm)
        self.distance_mm = float(distance_mm)
        self.shape = shape

    # ---- 1/2: 距离与尺寸互算 ----

    def distance_from_size(self, px_d):
        """已知真实尺寸 → 距离(mm)。相似三角形。"""
        if px_d <= 0:
            return None
        return self.size_mm * self.fx / px_d

    def size_from_distance(self, px_d):
        """已知距离 → 真实尺寸(mm)。上式反解。"""
        if px_d <= 0:
            return None
        return px_d * self.distance_mm / self.fx

    def solve(self, px_d):
        """按 known 方向算出 (距离, 尺寸)。两个都返回, 已知那个原样带出。"""
        if self.known == "distance":
            return self.distance_mm, self.size_from_distance(px_d)
        return self.distance_from_size(px_d), self.size_mm

    # ---- 3: 横纵偏移 ----

    def offset_mm(self, u, v, dist):
        """目标中心相对画面中心的真实偏移(mm)。dx 右正, dy 下正。"""
        if dist is None:
            return None, None
        return ((u - self.cx) * dist / self.fx,
                (v - self.cy) * dist / self.fy)

    # ---- 4: 角度(不需要距离, 纯几何) ----

    def angle_deg(self, u, v):
        """目标偏离光轴的 yaw/pitch(度)。yaw 右正, pitch 下正。"""
        return (math.degrees(math.atan2(u - self.cx, self.fx)),
                math.degrees(math.atan2(v - self.cy, self.fy)))

    # ---- 5: 两目标真实间距 ----

    def span_mm(self, u1, v1, u2, v2, dist):
        """同一深度平面上两点的真实距离(mm)。两目标一前一后时不准。"""
        if dist is None:
            return None
        px = math.sqrt((u2 - u1) ** 2 + (v2 - v1) ** 2)
        return px * dist / self.fx


# ---------------- 配置装载(失败即明确报错, 不静默跑错数) ----------------

def build_measurer(config):
    """从 config 造 Measurer。任何缺失都打印人话并返回 None。"""
    mcfg = config.get("measure", {})
    known = mcfg.get("known", "size")
    if known not in VALID_KNOWN:
        print("[Measure] 错误: measure.known = %r 不认识。" % known)
        print("[Measure] 合法值: %s ('size'=已知尺寸测距离, "
              "'distance'=已知距离测尺寸)" % (VALID_KNOWN,))
        return None

    cam_mat = config.get("calibration", {}).get("camera_matrix", [])
    if len(cam_mat) < 9 or cam_mat[0] <= 0 or cam_mat[4] <= 0:
        print("[Measure] 错误: calibration.camera_matrix 缺失或非法。")
        print("[Measure] 测量全靠内参, 没它一个数都算不了 —— 先做相机标定。")
        return None

    cam_cfg = config.get("camera", {})
    w, h = cam_cfg.get("width", 640), cam_cfg.get("height", 480)
    if (w, h) != (640, 480):
        print("[Measure] ⚠️ 警告: 内参按 640x480 标定, 当前 %dx%d, "
              "结果不可信!" % (w, h))
        print("[Measure] 要么把 camera 改回 640x480, 要么按比例换算 fx/fy/cx/cy。")

    size_mm = float(mcfg.get("target_size_mm", 25.0))
    distance_mm = float(mcfg.get("distance_mm", 200.0))
    if known == "size" and size_mm <= 0:
        print("[Measure] 错误: known='size' 但 target_size_mm=%s 非正数。"
              % size_mm)
        return None
    if known == "distance" and distance_mm <= 0:
        print("[Measure] 错误: known='distance' 但 distance_mm=%s 非正数。"
              % distance_mm)
        return None

    return Measurer(cam_mat[0], cam_mat[4], cam_mat[2], cam_mat[5],
                    known, size_mm, distance_mm,
                    mcfg.get("shape", "square"))


# ---------------- 主循环 ----------------

def main():
    config = load_config("config.json")
    ms = build_measurer(config)
    if ms is None:
        return

    from camera import Camera        # 延迟导入: PC 上单测 Measurer 时不碰硬件
    from display import Display

    vcfg = config.get("vision", {})
    thresholds = [tuple(t) for t in (vcfg.get("thresholds") or [])]
    if not thresholds:
        print("[Measure] 错误: config 的 vision.thresholds 为空, 找不到目标。")
        print("[Measure] 用 IDE 阈值编辑器(工具→机器视觉→阈值编辑器)调好后填进去。")
        return
    pixels_threshold = vcfg.get("pixels_threshold", 100)
    area_threshold = vcfg.get("area_threshold", 100)
    merge = vcfg.get("merge", True)

    mcfg = config.get("measure", {})
    span_on = mcfg.get("span", False)      # 是否量两目标间距(需画面里有两个色块)

    cam = Camera(config)
    disp = Display(config)
    clock = time.clock()

    print("[Measure] 启动 | fx=%.2f fy=%.2f cx=%.2f cy=%.2f"
          % (ms.fx, ms.fy, ms.cx, ms.cy))
    if ms.known == "size":
        print("[Measure] 方向: 已知尺寸 %.1fmm(%s) → 测距离"
              % (ms.size_mm, ms.shape))
    else:
        print("[Measure] 方向: 已知距离 %.1fmm → 测尺寸(%s)"
              % (ms.distance_mm, ms.shape))
    if span_on:
        print("[Measure] 两目标间距: 开(画面需同时有 2 个色块)")
    print("[Measure] 目标放画面中央最准(边缘有畸变)。IDE 点停止或 Ctrl+C 退出")

    try:
        while True:
            os.exitpoint()
            clock.tick()
            img = cam.snapshot()

            # 找所有色块, 按像素数排序取前两个(最大的当主目标)
            blobs = []
            for thr in thresholds:
                found = img.find_blobs([thr],
                                       pixels_threshold=pixels_threshold,
                                       area_threshold=area_threshold,
                                       merge=merge)
                if found:
                    blobs.extend(found)
            blobs = top_two(blobs)   # 手写选前二, 不依赖 sort 的 key 参数

            fps = clock.fps()

            if not blobs:
                img.draw_string_advanced(6, 4, 24, "no target",
                                         color=(255, 0, 0))
                print("  no target | fps=%.1f" % fps)
            else:
                b = blobs[0]
                u, v = _center(b)
                px_d = pixel_diameter(b, ms.shape)
                dist, size = ms.solve(px_d)
                dx, dy = ms.offset_mm(u, v, dist)
                yaw, pitch = ms.angle_deg(u, v)

                img.draw_rectangle(b.x(), b.y(), b.w(), b.h(),
                                   color=(0, 255, 0), thickness=2)
                img.draw_cross(int(u), int(v), color=(255, 255, 0), size=8)
                img.draw_line(int(ms.cx), 0, int(ms.cx), img.height(),
                              color=(60, 60, 60), thickness=1)
                img.draw_line(0, int(ms.cy), img.width(), int(ms.cy),
                              color=(60, 60, 60), thickness=1)

                # 已知的量灰字, 算出来的量亮字 —— 一眼分清哪个是测量结果
                c_known = (150, 150, 150)
                c_out = (255, 200, 0)
                if ms.known == "size":
                    img.draw_string_advanced(6, 4, 22,
                                             "size %.1fmm (known)" % size,
                                             color=c_known)
                    img.draw_string_advanced(6, 28, 24,
                                             "D = %.1f mm" % dist, color=c_out)
                else:
                    img.draw_string_advanced(6, 4, 22,
                                             "D %.1fmm (known)" % dist,
                                             color=c_known)
                    img.draw_string_advanced(6, 28, 24,
                                             "size = %.2f mm" % size,
                                             color=c_out)
                img.draw_string_advanced(6, 56, 20,
                                         "dx%+.1f dy%+.1f mm" % (dx, dy),
                                         color=(0, 220, 255))
                img.draw_string_advanced(6, 78, 20,
                                         "yaw%+.2f pitch%+.2f deg"
                                         % (yaw, pitch), color=(0, 220, 255))

                line = ("d_px=%5.1f | D=%7.1fmm size=%6.2fmm | "
                        "dx%+6.1f dy%+6.1f | yaw%+6.2f pitch%+6.2f | fps=%.1f"
                        % (px_d, dist, size, dx, dy, yaw, pitch, fps))

                if span_on and len(blobs) >= 2:
                    b2 = blobs[1]
                    u2, v2 = _center(b2)
                    L = ms.span_mm(u, v, u2, v2, dist)
                    img.draw_rectangle(b2.x(), b2.y(), b2.w(), b2.h(),
                                       color=(255, 0, 255), thickness=2)
                    img.draw_line(int(u), int(v), int(u2), int(v2),
                                  color=(255, 0, 255), thickness=2)
                    img.draw_string_advanced(6, 100, 20,
                                             "span = %.1f mm" % L,
                                             color=(255, 0, 255))
                    line += " | span=%.1fmm" % L
                elif span_on:
                    img.draw_string_advanced(6, 100, 20,
                                             "span: need 2 blobs",
                                             color=(255, 120, 120))
                print(line)

            img.draw_string_advanced(6, img.height() - 28, 20,
                                     "FPS: %.1f" % fps, color=(255, 255, 255))
            disp.show(img)

    except KeyboardInterrupt:
        print("[Measure] 手动停止")
    finally:
        # 顺序: 先 Display.deinit 再 camera(内部 MediaManager.deinit), 不可颠倒
        disp.close()
        cam.close()
        print("[Measure] 已退出, 资源已释放")


if __name__ == "__main__":
    main()
