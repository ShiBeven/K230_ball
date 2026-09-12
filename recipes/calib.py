# -*- coding: utf-8 -*-
"""
标定程序 —— 屏上引导采五个点, 最小二乘拟合出 offset_cm 与 scale_correction。

用法: 运行 run_calibration.py, 按屏上提示把球依次摆到指定刻度,
静止后点 [采样] 键; 五个点采完自动拟合并显示结果, 抄入 config.json 即可。

注意: 采样一律使用原始值(offset=0/scale=1) —— 若带旧修正数采样,
新结果会叠加旧值而系统性偏错。本程序不读取 config 中的修正数。
"""

import time

from utils.config_loader import load_config
from recipes.ball_pos import (_cfg_thr, find_tube, find_ball, origin_px,
                              px_to_cm_fixed, TubeRect, draw_zero_line)

# 五个测点。用 ±9 而非 ±10: tube_len_cm=23.5 且 inset_x_frac=0.05 时可找球范围
# 仅 ±10.58cm, 而球半径 0.5cm → 放 ±10 时球边缘几乎压在 ROI 边界上会被切扁
# (形状筛可能误拒, 且质心偏移)。±9 留 1.58cm 余量。
POINTS = (0.0, 5.0, -5.0, 9.0, -9.0)

N_SAMPLE = 60           # 每点连采帧数, 取中位数
MIN_VALID = 20          # 有效帧不足这么多就判本点采样失败(别拿几帧就定标)
TUNE_FILE = "/sdcard/tune.txt"
OUT_FILE = "/sdcard/calib_result.txt"


def _median(vals):
    """中位数。取中位数而非均值: 均值会被个别错读整体拖偏, 中位数天生丢极值。"""
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def fit(points, readings):
    """
    最小二乘拟合 readings → points, 求 offset 与 scale。

    模型 x_out = (x_raw + offset) * scale, 即 x_out = scale*x_raw + scale*offset。
    先按 y = a*x + b 拟合, 再 scale=a, offset=b/a。

    采用五点最小二乘而非单点反推: 五点可让刻度贴标误差均摊;
    线性度已验证, 一个线性 scale 即可, 无需二次项。

    返回 (offset, scale, 残差列表, 最大残差) 或 None。
    """
    n = len(points)
    if n < 2:
        return None
    sx = sy = sxx = sxy = 0.0
    for x, y in zip(readings, points):
        sx += x
        sy += y
        sxx += x * x
        sxy += x * y
    den = n * sxx - sx * sx
    if den == 0 or abs(den) < 1e-9:
        return None
    a = (n * sxy - sx * sy) / den
    b = (sy - a * sx) / n
    if abs(a) < 1e-9:
        return None
    scale = a
    offset = b / a
    res = []
    for x, y in zip(readings, points):
        res.append((x + offset) * scale - y)
    mx = 0.0
    for r in res:
        if abs(r) > mx:
            mx = abs(r)
    return (offset, scale, res, mx)


class CalibUI:
    """采样按钮 + 屏上引导。只管触摸与显示, 不碰检测。"""

    def __init__(self, cfg, cam_w, cam_h):
        self.ok = False
        m = (cfg.get("map") or {})
        self.kx = float(m.get("kx", 0.9870))
        self.bx = float(m.get("bx", 14.6))
        self.ky = float(m.get("ky", 1.1061))
        self.by = float(m.get("by", -32.5))
        self.every = int(cfg.get("read_every_frames", 5)) or 5
        self._down = False
        self._miss = 0
        self._touch = None
        self._rd_fail = 0
        self.hit_req = False        # 待处理的采样请求
        self.skip_req = False       # 待处理的跳过请求
        self._lit = 0
        # 两个大按钮: 左 [采样](占 70%) / 右 [跳过](占 30%)。
        # 采样键做大是因为它要按 5 次; 跳过键小且分开, 避免误触跳过一个测点。
        self.bh = 96
        self.by_ = cam_h - self.bh - 8
        margin = 10
        gap = 24
        zone = cam_w - 2 * margin - gap
        self.aw = int(zone * 0.70)
        self.ax = margin
        self.sw = zone - self.aw
        self.sx = self.ax + self.aw + gap

    def open(self):
        try:
            from machine import TOUCH
            self._touch = TOUCH(0)
            self.ok = True
            print("  触摸已就绪: 屏上点 [采样] 推进")
        except Exception as e:
            print("  ⚠️ 触摸不可用(%r) —— 标定需要触摸屏, 请插好屏再跑" % (e,))

    def poll(self, n_frame):
        if not self.ok or (n_frame % self.every):
            return
        try:
            tp = self._touch.read(1)
            self._rd_fail = 0
        except Exception:
            self._rd_fail += 1
            if self._rd_fail >= 10:
                self.ok = False
                print("  [触摸] 连续读失败 → 停用(屏被拔了?)")
            return
        p = None
        if tp and len(tp):
            if tp[0].event in (2, 3):
                p = tp[0]
        if p is None:
            self._miss += 1
            if self._miss >= 3:
                self._down = False
            return
        self._miss = 0
        if self._down:
            return
        self._down = True
        cx = self.kx * p.x + self.bx
        cy = self.ky * p.y + self.by
        if not (self.by_ <= cy < self.by_ + self.bh):
            return
        if self.ax <= cx < self.ax + self.aw:
            self.hit_req = True
            self._lit = time.ticks_ms() + 250
        elif self.sx <= cx < self.sx + self.sw:
            self.skip_req = True
            self._lit = time.ticks_ms() + 250

    def draw(self, img, busy):
        if not self.ok:
            return
        now = time.ticks_ms()
        lit = now < self._lit
        # 采样键: 采样中转橙色并写 SAMPLING, 否则绿色写 SAMPLE
        if busy:
            col = (200, 110, 0)
            txt = "SAMPLING..."
        elif lit:
            col = (255, 255, 0)
            txt = "SAMPLE"
        else:
            col = (0, 130, 60)
            txt = "SAMPLE"
        img.draw_rectangle(self.ax, self.by_, self.aw, self.bh,
                           color=col, thickness=1, fill=True)
        img.draw_rectangle(self.ax, self.by_, self.aw, self.bh,
                           color=(220, 220, 220), thickness=2)
        img.draw_string_advanced(self.ax + 14, self.by_ + 28, 40, txt,
                                 color=(255, 255, 255))
        img.draw_rectangle(self.sx, self.by_, self.sw, self.bh,
                           color=(90, 60, 60), thickness=1, fill=True)
        img.draw_rectangle(self.sx, self.by_, self.sw, self.bh,
                           color=(200, 200, 200), thickness=2)
        img.draw_string_advanced(self.sx + 10, self.by_ + 32, 30, "SKIP",
                                 color=(230, 200, 200))


def _report(done_pts, done_vals):
    """把拟合结果打成可直接抄进 config 的形式, 并存盘。"""
    print("=" * 58)
    print("  标定采样结果")
    print("=" * 58)
    for p, v in zip(done_pts, done_vals):
        print("    刻度 %+5.1f cm  →  实测原始读数 %+7.3f cm" % (p, v))
    r = fit(done_pts, done_vals)
    if r is None:
        print("  ⚠️ 拟合失败(有效点不足或数据退化), 请重跑标定")
        return None
    offset, scale, res, mx = r
    print("-" * 58)
    print("  拟合结果(五点最小二乘, 模型 x=(x_raw+offset)*scale):")
    print()
    print('      "offset_cm": %.4f,' % offset)
    print('      "scale_correction": %.4f,' % scale)
    print()
    print("  各点残差(修正后与真值之差):")
    for p, e in zip(done_pts, res):
        flag = ""
        if abs(e) > 0.5:
            flag = "  ⚠️ 偏大"
        print("    %+5.1f cm  残差 %+6.3f cm%s" % (p, e, flag))
    print("  最大残差 %.3f cm (容差 1cm 的 %.0f%%)" % (mx, mx * 100))
    if mx > 0.5:
        print("  ⚠️ 最大残差超过容差的一半 —— 可能原因:")
        print("     ① 某个点采样时球没完全静止 ② 刻度线贴歪 ③ 球被 ROI 边缘切到")
        print("     建议重采那个残差最大的点")
    elif mx > 0.3:
        print("  残差偏大但可用。若想更准, 重采残差最大的那个点。")
    else:
        print("  ✅ 残差良好")
    print("-" * 58)
    print("  下一步: 把上面那两行抄进 config.json 的 ball_pos 段,")
    print("          然后 make_installer.py → 重新部署 → 按硬件复位键")
    print("=" * 58)
    # 存盘: 便于稍后抄写
    try:
        with open(OUT_FILE, "w") as f:
            f.write('"offset_cm": %.4f,\n' % offset)
            f.write('"scale_correction": %.4f,\n' % scale)
            f.write("# 采样点(刻度 -> 原始读数):\n")
            for p, v in zip(done_pts, done_vals):
                f.write("#   %+.1f -> %+.4f\n" % (p, v))
            f.write("# 最大残差 %.4f cm\n" % mx)
        print("  结果已存 %s (掉电不丢)" % OUT_FILE)
    except Exception as e:
        print("  ⚠️ 结果存盘失败(终端上面那两行仍然有效): %r" % (e,))
    return (offset, scale)


def main():
    config = load_config("config.json")
    cfg = config.get("ball_pos", {})

    tube_thr = _cfg_thr(cfg.get("tube"), "tube")
    ball_thr = _cfg_thr(cfg.get("ball"), "ball")
    tube_len_cm = float(cfg.get("tube_len_cm", 23.5))
    spec_l_min = int(cfg.get("spec_l_min", 55))
    spec_px_min = int(cfg.get("spec_pixels_min", 2))
    ball_px_min = int(cfg.get("ball_pixels_min", 12))
    ball_max_aspect = float(cfg.get("ball_max_aspect", 2.2))
    spec_inside = bool(cfg.get("require_spec_inside", True))
    inset_frac = float(cfg.get("inset_frac", 0.22))
    inset_x_frac = float(cfg.get("inset_x_frac", 0.05))
    tube_px_min = int(cfg.get("tube_pixels_min", 2000))
    min_aspect = float(cfg.get("tube_min_aspect", 8.0))
    min_fill = float(cfg.get("tube_min_fill", 0.5))
    invert_x = bool(cfg.get("invert_x", False))
    warmup_frames = int(cfg.get("warmup_frames", 20))
    require_spec = bool(cfg.get("require_spec", True))
    band = cfg.get("search_band")
    if band is not None and len(band) == 2:
        band = (float(band[0]), float(band[1]))
    else:
        band = None

    # ⚠️ 若板上存在 tune.txt, 它会覆盖 config 里的阈值 —— 标定必须用与主程序
    # **完全相同**的阈值, 否则标出的 offset/scale 对不上主程序的读数。
    tuned = {}
    try:
        with open(TUNE_FILE, "r") as f:
            for line in f.read().split("\n"):
                line = line.strip()
                if "=" in line:
                    k, _, v = line.partition("=")
                    try:
                        tuned[k.strip()] = float(v.strip())
                    except Exception:
                        pass
    except Exception:
        pass
    if tuned:
        print("  检测到 %s, 采用其中的阈值(与主程序保持一致):" % TUNE_FILE)
        if "spec_l_min" in tuned:
            spec_l_min = int(tuned["spec_l_min"])
        if "spec_pixels_min" in tuned:
            spec_px_min = int(tuned["spec_pixels_min"])
        if "ball_max_aspect" in tuned:
            ball_max_aspect = float(tuned["ball_max_aspect"])
        if "ball_l_max" in tuned:
            ball_thr = (ball_thr[0], int(tuned["ball_l_max"]),
                        ball_thr[2], ball_thr[3], ball_thr[4], ball_thr[5])
        print("     specL=%d ballL=%d aspect=%.1f specPx=%d"
              % (spec_l_min, ball_thr[1], ball_max_aspect, spec_px_min))

    print("=" * 58)
    print("  钢球位置标定 —— 五点最小二乘")
    print("=" * 58)
    print("  测点: %s cm" % ("  ".join("%+.0f" % p for p in POINTS)))
    print("  每点连采 %d 帧取中位数(有效帧 <%d 判失败)" % (N_SAMPLE, MIN_VALID))
    print("  ⚠️ 采样用**原始值**(offset=0/scale=1), 不受 config 现有修正数影响")
    print("  ⚠️ 每点务必等球**完全静止**再点 [采样] —— 拨球过程中的读数会被行程污染")
    print("-" * 58)

    from camera import Camera
    from display import Display

    cam = Camera(config)
    disp = Display(config) if config.get("display", {}).get("enabled", True) else None
    ui = CalibUI(cfg.get("touch") or {}, int(config.get("camera", {}).get("width", 640)),
                 int(config.get("camera", {}).get("height", 480)))
    ui.open()
    if not ui.ok:
        print("  ⛔ 触摸不可用, 标定无法进行。插好屏后重跑。")
        return

    idx = 0                 # 当前测点
    n_frame = 0
    busy = False            # 正在连采
    buf = []
    n_try = 0
    tube = None
    done_pts = []
    done_vals = []
    msg = ""
    msg_until = 0

    try:
        while True:
            img = cam.snapshot()
            n_frame += 1
            warming = n_frame <= warmup_frames

            # 管子每帧重找: 标定期间不锁, 保证与主程序锁定时同一套几何来源
            if tube is None or warming:
                found = find_tube(img, tube_thr, tube_px_min, min_aspect,
                                  min_fill, band)
                if found is not None:
                    tube = TubeRect(found)

            x_raw = None
            if tube is not None and not warming:
                ppcm = tube.w / float(tube_len_cm)
                ox = origin_px(tube)
                ball = find_ball(img, tube, ball_thr, spec_l_min, ball_px_min,
                                 spec_px_min, inset_frac, None, inset_x_frac,
                                 spec_inside, ball_max_aspect)
                if ball is not None:
                    bcx, bcy, bw, bh, has_spec, bpx = ball
                    if not (require_spec and not has_spec):
                        # ⭐ offset=0 / scale=1: 采的是**未修正的原始值**
                        x_raw = px_to_cm_fixed(bcx, ox, ppcm, invert_x, 0.0, 1.0)
                        img.draw_cross(int(bcx), int(bcy), color=(255, 0, 0),
                                       size=10, thickness=2)
                img.draw_rectangle(tube.x, tube.y, tube.w, tube.h,
                                   color=(255, 255, 0), thickness=1)
                # 紫线画在当前该摆的刻度上 —— 球该压在这条线上, 是免费的自检
                if idx < len(POINTS):
                    tx = ox + (-POINTS[idx] if invert_x else POINTS[idx]) * ppcm
                    draw_zero_line(img, tx, tube.y + tube.h // 2, 95)

            ui.poll(n_frame)

            # ---- 采样推进 ----
            if busy:
                n_try += 1
                if x_raw is not None:
                    buf.append(x_raw)
                if n_try >= N_SAMPLE:
                    busy = False
                    if len(buf) >= MIN_VALID:
                        v = _median(buf)
                        done_pts.append(POINTS[idx])
                        done_vals.append(v)
                        pp = max(buf) - min(buf)
                        print("  [点 %d/%d] 刻度 %+.1f → 读数 %+.3f cm  "
                              "(有效 %d/%d 帧, 峰峰值 %.3f)"
                              % (idx + 1, len(POINTS), POINTS[idx], v,
                                 len(buf), N_SAMPLE, pp))
                        if pp > 1.0:
                            print("     ⚠️ 峰峰值 %.2fcm 偏大 —— 球可能没静止, "
                                  "或有帧认到了管壁。建议重跑本点。" % pp)
                        msg = "OK %+.2f" % v
                        idx += 1
                    else:
                        # 有效帧太少 → 不采纳。宁可让人重来, 也不拿几帧定标
                        print("  ⚠️ [点 %d] 有效帧仅 %d/%d(不足 %d) → **本点未采纳**"
                              % (idx + 1, len(buf), N_SAMPLE, MIN_VALID))
                        print("     查: 球在管里吗? 屏上有红十字吗? "
                              "阈值是否需要用主程序的 [TUNE] 先调好?")
                        msg = "FAILED, retry"
                    msg_until = time.ticks_ms() + 2500
                    buf = []
            elif ui.hit_req:
                ui.hit_req = False
                if idx < len(POINTS):
                    busy = True
                    n_try = 0
                    buf = []
                    print("  采样中... 刻度 %+.1f cm, 请保持球静止" % POINTS[idx])
            elif ui.skip_req:
                ui.skip_req = False
                if idx < len(POINTS):
                    print("  已跳过刻度 %+.1f cm(该点不参与拟合)" % POINTS[idx])
                    msg = "SKIPPED"
                    msg_until = time.ticks_ms() + 1500
                    idx += 1

            # ---- 屏上引导 ----
            if warming:
                img.draw_string_advanced(10, 10, 30,
                                         "WARMUP %d/%d" % (n_frame, warmup_frames),
                                         color=(255, 160, 0))
            elif tube is None:
                img.draw_string_advanced(10, 10, 30, "NO TUBE - 找不到绿管",
                                         color=(255, 0, 0))
            elif idx < len(POINTS):
                img.draw_string_advanced(10, 8, 26,
                                         "标定 %d/%d" % (idx + 1, len(POINTS)),
                                         color=(200, 200, 200))
                img.draw_string_advanced(10, 38, 46,
                                         "把球摆到 %+.0f cm" % POINTS[idx],
                                         color=(255, 230, 100))
                if busy:
                    img.draw_string_advanced(10, 92, 26,
                                             "采样中 %d/%d  有效 %d"
                                             % (n_try, N_SAMPLE, len(buf)),
                                             color=(255, 160, 0))
                else:
                    s = ("当前 %+.2f cm" % x_raw) if x_raw is not None \
                        else "看不到球"
                    img.draw_string_advanced(10, 92, 26, s,
                                             color=(0, 255, 255) if x_raw is not None
                                             else (255, 100, 100))
                    img.draw_string_advanced(10, 124, 22, "静止后点 [SAMPLE]",
                                             color=(180, 180, 180))
            else:
                img.draw_string_advanced(10, 20, 40, "标定完成!",
                                         color=(120, 255, 160))
                img.draw_string_advanced(10, 70, 24, "看终端抄那两行数字",
                                         color=(200, 200, 200))

            if msg and time.ticks_ms() < msg_until:
                img.draw_string_advanced(10, 160, 30, msg, color=(120, 255, 160))

            ui.draw(img, busy)
            if disp:
                disp.show(img)

            # 五点采完 → 拟合并结束
            if idx >= len(POINTS):
                if len(done_pts) >= 2:
                    r = _report(done_pts, done_vals)
                    if r is not None and disp:
                        # 结果在屏上停留 10 秒, 可直接拍照记录
                        off, sc = r
                        t_end = time.ticks_ms() + 10000
                        while time.ticks_ms() < t_end:
                            im2 = cam.snapshot()
                            im2.draw_string_advanced(10, 20, 32, "标定完成",
                                                     color=(120, 255, 160))
                            im2.draw_string_advanced(10, 66, 30,
                                                     "offset %+.4f" % off,
                                                     color=(255, 255, 255))
                            im2.draw_string_advanced(10, 104, 30,
                                                     "scale  %.4f" % sc,
                                                     color=(255, 255, 255))
                            im2.draw_string_advanced(10, 148, 22,
                                                     "抄进 config.json 后重新部署",
                                                     color=(200, 200, 200))
                            disp.show(im2)
                else:
                    print("  ⚠️ 有效点不足 2 个, 无法拟合。请重跑标定。")
                break

    except KeyboardInterrupt:
        print("\n[标定] 用户停止")
        if len(done_pts) >= 2:
            print("  已采到 %d 个点, 仍可拟合:" % len(done_pts))
            _report(done_pts, done_vals)
    finally:
        try:
            cam.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

