# -*- coding: utf-8 -*-
"""
钢球一维位置检测配方 —— 测量钢球在绿色凹槽导轨内的 X 坐标(cm),
并将位置误差经串口发送给下位机。

检测原理: 不对钢球本身做颜色阈值检测(镜面球无稳定自身颜色), 而是检测
绿色导轨、在导轨内寻找"非绿色的暗块"作为球体候选, 再用镜面高光特征
排除导轨端口切口的阴影。标尺 px_per_cm = 导轨像素宽 / 实际宽度, 无需相机内参。

主要输出: 触摸屏 UI(目标设定 / 状态切换 / 阈值调节) + ball8 串口帧 + MJPEG 图传。
"""

import time

from utils.config_loader import load_config

# 兼容性: 部分固件的 sort(key=)/max(key=) 行为异常, 循环均为手写实现


def _cfg_thr(seq, name):
    """把 config 里的 6 元 LAB 阈值转成 find_blobs 要的 tuple, 顺手校验。"""
    if seq is None or len(seq) != 6:
        raise ValueError("ball_pos.%s 必须是 6 个数 [Lmin,Lmax,Amin,Amax,Bmin,Bmax]" % name)
    return (int(seq[0]), int(seq[1]), int(seq[2]),
            int(seq[3]), int(seq[4]), int(seq[5]))


def _tube_like(b, min_aspect, min_fill):
    """
    形状判据: 导轨应为宽扁(宽高比大)且实心(填充率高)的色块。

    两个约束缺一不可: 仅按"最宽"筛选会把横跨画面的松散干扰物误认为导轨;
    形状约束与光照无关, 比颜色阈值更稳定。
    """
    if b.h() <= 0 or b.w() <= 0:
        return False
    if b.w() / float(b.h()) < min_aspect:
        return False
    return b.pixels() / float(b.w() * b.h()) >= min_fill


def _pick_tube(blobs):
    """
    在通过形状约束的候选里选导轨: 取像素数最多者而非最宽者。

    按宽度选会输给横跨画面的干扰物; 按面积选则导轨明显胜出,
    且导轨被球切断时 merge 会补回, 抗干扰性更好。
    """
    best = None
    for b in blobs:
        if best is None or b.pixels() > best.pixels():
            best = b
    return best


def _biggest(blobs):
    """返回像素数最多的色块。"""
    best = None
    for b in blobs:
        if best is None or b.pixels() > best.pixels():
            best = b
    return best


def find_tube(img, thr, pixels_min, min_aspect, min_fill, search_band=None):
    """
    查找绿色导轨, 返回 blob 或 None。
    先用形状约束筛掉不像导轨的色块, 再按像素数选取。

    search_band=(y_frac_min, y_frac_max) 时只在指定纵向区间内查找 ——
    相机与导轨相对固定时, 可用于排除画面上下的背景干扰。默认关闭。
    """
    roi = None
    if search_band is not None:
        y0 = int(img.height() * search_band[0])
        y1 = int(img.height() * search_band[1])
        if y1 - y0 >= 8:
            roi = (0, y0, img.width(), y1 - y0)

    if roi is not None:
        blobs = img.find_blobs([thr], roi=roi, pixels_threshold=pixels_min,
                               area_threshold=pixels_min, merge=True)
    else:
        blobs = img.find_blobs([thr], pixels_threshold=pixels_min,
                               area_threshold=pixels_min, merge=True)
    if not blobs:
        return None
    cands = []
    for b in blobs:
        if _tube_like(b, min_aspect, min_fill):
            cands.append(b)
    if not cands:
        return None
    tube = _pick_tube(cands)
    # 导轨必须足够长, 否则可能是其他绿色干扰物
    if tube is None or tube.w() < img.width() * 0.30:
        return None
    return tube


def _spec_hits(c, s, inside):
    """
    高光 s 算不算命中候选块 c。

    inside=False: 高光中心落在框内并放宽 2px 即算命中。
    inside=True : 高光中心须落在框内部收缩一圈的区域内。

    inside=True 用于排除"导轨壁亮边擦框边缘命中"的假阳性: 导轨壁的亮边常
    贴近候选框边缘; 而球是镜面圆球, 高光必然位于中心附近。

    收缩量取 1/4 短边(至少 1px): 过大会导致球位于导轨两端时漏检。
    """
    cx0, cy0, cw, ch = c.x(), c.y(), c.w(), c.h()
    if not inside:
        return (cx0 - 2 <= s.cx() <= cx0 + cw + 2
                and cy0 - 2 <= s.cy() <= cy0 + ch + 2)
    m = ch if ch < cw else cw
    d = m // 4
    if d < 1:
        d = 1
    return (cx0 + d <= s.cx() <= cx0 + cw - d
            and cy0 + d <= s.cy() <= cy0 + ch - d)


def measure_spec_l(img, blob):
    """
    测量候选块内的最大 L 亮度, 供诊断打印。

    光照变暗时 spec_l_min 是最先失效的阈值: 高光亮度整体下移后会被逐帧拒绝,
    现象是"识别不到球"。提供实测值便于据此调整阈值。
    失败返回 None —— 诊断不应影响检测。
    """
    try:
        r = (blob.x(), blob.y(), blob.w(), blob.h())
        st = img.get_statistics(roi=r)
        return st.l_max()
    except Exception:
        return None


def find_ball(img, tube, ball_thr, spec_l_min, ball_pixels_min,
              spec_pixels_min, inset_frac, diag=None, inset_x_frac=0.03,
              spec_inside=True, ball_max_aspect=2.2):
    """
    在绿管 ROI 内找钢球。返回 (cx, cy, w, h, has_spec, pixels) 或 None。

    两步:
      ① 在向内收缩的 ROI 内找"不绿的暗块" = 球体候选
      ② 在同一 ROI 内找高光亮点, 用于校验候选(排除管口切口与边沿阴影)

    ROI 收缩是必要的: 管子上下边沿与两端切口同样"不绿且暗", 且比球更长更连续,
    不收缩则必然选中阴影而非球。

    has_spec=False 的返回值可能来自导轨端口阴影等干扰, 调用方须按
    config.require_spec 决定是否丢弃。设计取舍: 宁可本帧不输出,
    也不输出错误位置。
    """
    tx, ty, tw, th = tube.x, tube.y, tube.w, tube.h

    # 上下各向内收缩(避开管沿阴影), 左右也收一点(避开两端切口)
    dy = int(th * inset_frac)
    # 左右收缩量可配: 光线变暗时导轨端部暗区会被误选, 应适当增大收缩。
    dx = max(4, int(tw * inset_x_frac))
    rx = tx + dx
    ry = ty + dy
    rw = tw - 2 * dx
    rh = th - 2 * dy
    if rw <= 8 or rh <= 2:
        return None
    roi = (rx, ry, rw, rh)

    # ① 暗块 = 球体
    raw_cands = img.find_blobs([ball_thr], roi=roi,
                               pixels_threshold=ball_pixels_min,
                               area_threshold=ball_pixels_min, merge=True)
    if not raw_cands:
        return None

    # 形状筛: 球近圆(宽高比≈1), 导轨壁与端部暗带为细长条。
    # 暗光下 LAB 阈值区间会与导轨壁重叠, 而形状与光照无关。
    # 门限不宜低于 1.5: 球在导轨两端会被 ROI 边缘切掉一部分而变扁。
    cands = []
    for c in raw_cands:
        h = c.h()
        if h <= 0:
            continue
        a = c.w() / float(h)
        if a < 1.0:
            a = 1.0 / a if a > 0 else 999.0   # 竖着的细长条同样要拒
        if a <= ball_max_aspect:
            cands.append(c)
    if not cands:
        # 有暗块但全是细长条 → 本帧只看到导轨壁, 直接返回
        if diag is not None:
            diag["all_elongated"] = len(raw_cands)
        return None

    # ② 高光 = 镜面球签名(唯一能把球和阴影区分开的特征)
    spec_thr = (int(spec_l_min), 100, -30, 30, -30, 30)
    specs = img.find_blobs([spec_thr], roi=roi,
                           pixels_threshold=spec_pixels_min,
                           area_threshold=spec_pixels_min, merge=True)

    # 优先选「含高光」的候选; 高光整体消失时退化为选最大暗块
    best = None
    for c in cands:
        hit = False
        for s in specs:
            if _spec_hits(c, s, spec_inside):
                hit = True
                break
        if hit and (best is None or c.pixels() > best[0].pixels()):
            best = (c, True)
    if best is None:
        b = _biggest(cands)
        if b is None:
            return None
        # 找到暗块但无高光: 记录诊断量, 供调用方区分
        # "阈值偏高误拒真球"与"正确拒绝阴影"两种情形。
        if diag is not None:
            diag["l_max"] = measure_spec_l(img, b)
            diag["n_cands"] = len(cands)
            # 被拒块的宽高比: 细长条说明是导轨壁(判据正常工作),
            # 近方形则可能是真球被误拒(判据过严)。
            if b.h() > 0:
                diag["aspect"] = b.w() / float(b.h())
        best = (b, False)       # has_spec=False → 由调用方按 require_spec 决定要不要

    blob, has_spec = best
    # 末位是 pixels()(真实像素数)而非 w*h: 外接矩形会包含部分导轨背景
    return (float(blob.cx()), float(blob.cy()),
            blob.w(), blob.h(), has_spec, blob.pixels())


def origin_px(tube):
    """
    0 刻度在画面中的 x 像素 = 导轨像素中心。

    独立成函数以保证画面标线与串口输出使用同一原点, 避免两处不一致。
    """
    return tube.x + tube.w / 2.0


def draw_zero_line(img, ox, cy, half_len, color=(190, 90, 230), thickness=6):
    """
    画一条竖直的紫色标线指示目标位置。

    ox 为目标位置的画面 x 像素(目标为 0 时即原点)。标线与串口输出同源,
    可用于目视校验坐标换算是否正确。
    """
    y0 = cy - half_len
    y1 = cy + half_len
    if y0 < 0:
        y0 = 0
    if y1 > img.height() - 1:
        y1 = img.height() - 1
    # 用细矩形填充代替粗线: draw_line 的 thickness 在部分固件上表现不一致
    half_w = thickness // 2
    img.draw_rectangle(int(ox) - half_w, int(y0),
                       thickness, int(y1 - y0),
                       color=color, thickness=1, fill=True)


def draw_arrow_chain(img, x_from, x_to, y, color=(80, 230, 255),
                     spacing=22, size=9, thickness=2):
    """
    从球到目标标线画一串箭头, 指示目标方向。

    箭头指向目标标线, 即串口误差符号的可视化。
    用两条短斜线拼成 V 形, 比实心三角形更清晰且开销更小。
    """
    dist = x_to - x_from
    adist = dist if dist >= 0 else -dist
    if adist < spacing * 0.8:        # 球已经很接近 0 线 → 不画, 免得糊成一团
        return 0
    step = spacing if dist > 0 else -spacing
    # 箭尖朝向目标线方向
    tip = size if dist > 0 else -size
    y = int(y)
    n = 0
    x = x_from + step             # 从球旁边一格起画, 不压在十字上
    # 留一格不画, 避免箭尖压到目标标线上
    while (step > 0 and x < x_to - spacing * 0.5) or \
          (step < 0 and x > x_to + spacing * 0.5):
        xi = int(x)
        img.draw_line(xi, y - size, xi + tip, y, color=color,
                      thickness=thickness)
        img.draw_line(xi, y + size, xi + tip, y, color=color,
                      thickness=thickness)
        x += step
        n += 1
        if n >= 40:               # 上限保护, 防止异常数据画出过多箭头
            break
    return n


class JumpGate:
    """
    跳变闸门 —— 拦截物理上不可能的位置突变。

    中值滤波只能抑制孤立离群点, 连续多帧的坏读会漏进输出;
    跳变闸门基于物理约束(一帧时间内位置变化有上限), 可拦截各类异常读数,
    且对正常运动零延迟。

    连续拒绝 reject_limit 次后强制接受并重新锚定: 处理合法的大位移
    (如手动移动球), 避免永久卡在旧位置。
    """
    __slots__ = ("max_jump", "reject_limit", "last", "n_reject", "n_blocked",
                 "anchor", "n_cluster", "n_forced")

    def __init__(self, max_jump_cm, reject_limit=8):
        self.max_jump = float(max_jump_cm)
        self.reject_limit = int(reject_limit)
        self.last = None
        self.n_reject = 0        # 当前连续拒绝数
        self.n_blocked = 0       # 累计拦掉多少帧(诊断用: 这个数就是错读发生率)
        self.anchor = None       # 被拒读数聚集的候选新位置
        self.n_cluster = 0       # 聚在 anchor 附近的连续拒绝数
        self.n_forced = 0        # 强制放行次数(诊断: >0 说明发生过合法大位移或被顶开)

    def reset(self):
        self.last = None
        self.n_reject = 0
        self.anchor = None
        self.n_cluster = 0

    def check(self, v):
        """返回 True=接受。max_jump<=0 时门限关闭, 一律接受。"""
        if self.max_jump <= 0:
            return True
        if self.last is None:
            self.last = v
            return True
        if abs(v - self.last) <= self.max_jump:
            self.last = v
            self.n_reject = 0
            return True
        self.n_reject += 1
        self.n_blocked += 1
        # 强制放行的条件不是"连拒 N 次", 而是那 N 次读数聚在同一新位置附近。
        # 只数次数的话, 管壁在多个不同位置被误认也能凑满 N 次顶开闸门 —— 即错读
        # 顶开了本该防它的机制。合法大位移的读数会聚集, 错读是散的。
        if self.anchor is None or abs(v - self.anchor) > self.max_jump:
            self.anchor = v          # 新的候选位置, 重新数
            self.n_cluster = 1
        else:
            self.n_cluster += 1
        if self.n_cluster >= self.reject_limit:
            self.last = v
            self.n_reject = 0
            self.n_cluster = 0
            self.anchor = None
            self.n_forced += 1
            return True
        return False


class TubeRect:
    """
    缓存住的摆杆矩形。

    缓存理由: 相机与导轨刚性固连时, 导轨在画面中位置固定, 每帧重找是纯浪费;
    锁定后每帧只搜索球体与高光, 显著降低检测延迟 —— 闭环控制中延迟
    直接转化为相位滞后, 影响稳定性。
    """
    __slots__ = ("x", "y", "w", "h")

    def __init__(self, blob):
        self.x = blob.x()
        self.y = blob.y()
        self.w = blob.w()
        self.h = blob.h()


def px_to_cm_fixed(ball_cx, origin, px_per_cm, invert_x,
                   offset_cm=0.0, scale=1.0):
    """
    固定标尺模式: 标尺与原点使用 config 中的常数, 不从画面测量。

    相机与导轨刚性固连时, 导轨在画面中的位置和宽度是物理常数; 每帧测量
    反而会把它们变成随光照抖动的变量(阈值边界处像素归属不稳定)。
    offset/scale 标定绑定于某一次锁定, 标尺变化会导致标定失配 ——
    因此"每次一致"比"每次重新测量"更重要。
    """
    x_cm = (ball_cx - origin) / px_per_cm
    if invert_x:
        x_cm = -x_cm
    return (x_cm + offset_cm) * scale


def px_to_cm(ball_cx, tube, tube_len_cm, invert_x,
             offset_cm=0.0, scale=1.0):
    """
    像素 → cm。不需要相机内参: 距离恒定 → 固定系数。
        px_per_cm = 绿管像素宽 / 摆杆真实长度
        X = (球中心 − 管中心) / px_per_cm
    返回 (x_cm, px_per_cm)。

    offset_cm / scale 为实测标定的两个修正数: x = (x_raw + offset_cm) * scale
    - offset_cm 吸收加性误差(几何中心与刻度零点的固定偏心等);
    - scale 吸收乘性误差(深度偏移、镜头畸变等随 |X| 增大的项)。
    """
    px_per_cm = tube.w / float(tube_len_cm)
    origin = origin_px(tube)                  # 原点 O = 管中心
    x_cm = (ball_cx - origin) / px_per_cm
    if invert_x:
        x_cm = -x_cm
    x_cm = (x_cm + offset_cm) * scale
    return x_cm, px_per_cm


class MedianFilter:
    """
    滑动中值滤波 —— 抑制位置的随机抖动与偶发离群读数。

    中值而非均值: 待抑制指标是峰峰值(极值), 中值天然丢弃极值;
    偶发误检对均值是污染, 对中值只要不超过半个窗口就不影响输出。

    代价是延迟: 窗口 N 的群延迟为 (N-1)/2 帧, 闭环系统中延迟会转化为
    相位滞后, 因此 N 不宜过大(默认 5, 需要更低延迟时降到 3, 填 1 关闭)。

    实现: 环形缓冲 + 插入排序(N 较小时比通用排序更快)。
    """
    __slots__ = ("n", "buf", "cnt", "idx")

    def __init__(self, n):
        n = int(n)
        if n < 1:
            n = 1
        if n % 2 == 0:
            n += 1          # 偶数窗口没有唯一中值 → 强制奇数
        self.n = n
        self.buf = [0.0] * n
        self.cnt = 0        # 已填入的有效样本数(未满时 buf[:cnt] 才有效)
        self.idx = 0

    def reset(self):
        """丢球太久后调用: 窗口里的旧值已过时, 留着会输出陈旧位置。"""
        self.cnt = 0
        self.idx = 0

    def push(self, v):
        """塞入一个新样本, 返回当前窗口中值。窗口没满时用已有样本的中值(不等待)。"""
        if self.n == 1:
            return v                      # N=1 = 关闭滤波, 零开销直通
        self.buf[self.idx] = v
        self.idx = (self.idx + 1) % self.n
        if self.cnt < self.n:
            self.cnt += 1
        m = self.cnt
        tmp = self.buf[:m]                # 未满时有效值就是 buf[0:cnt]
        for i in range(1, m):             # 插入排序
            key = tmp[i]
            j = i - 1
            while j >= 0 and tmp[j] > key:
                tmp[j + 1] = tmp[j]
                j -= 1
            tmp[j + 1] = key
        return tmp[m // 2]


# ===== 两个工作状态 =====
#
# MODE_HOLD = 手动模式: 目标值由人设定(触摸屏按钮), 球稳定在该点。
# MODE_SEQ  = 序列模式: 目标值按脚本依次走过各位置(如 0 → +5 → -5)。
#
# 序列模式的实现方式: 不新增控制逻辑, 只让目标值按脚本自动移动 ——
# 球由下位机闭环追踪目标, 两种模式在数据链路上完全一致, 区别只是
# target 的来源(触摸按钮 或 Sequence 脚本)。
#
# 注意: 闸门与中值滤波处理的是绝对位置 x_raw, 只在发送串口时才减 target,
# 因此 Sequence 移动目标不会被跳变闸门误拦。
MODE_HOLD = 0
MODE_SEQ = 1
MODE_NAMES = ("HOLD", "SEQ")


class Sequence:
    """
    序列模式的往返脚本: 目标值依次走过 points 中的每个位置, 到达后停留。

    默认 points = [0, +5, -5], 最后一个点到达后停住(终态, 不循环)。

    到达判据 = |x − 目标| ≤ tol_cm 且连续保持 dwell_frames 帧:
    只看单帧会被球冲过目标点的瞬间误判(未真正稳定), 因此要求连续多帧。

    丢帧时既不增加也不清零计数: 丢帧表示"本帧无读数", 不等于"球离开",
    清零会导致计满 dwell 帧几乎不可能。

    超时只报警不自动推进: 与检测侧一致, 宁可停住提示, 也不假装到达。
    """
    __slots__ = ("points", "tol", "dwell", "timeout_ms", "idx", "n_ok",
                 "t_leg", "warned", "done", "running", "n_arrive")

    def __init__(self, points, tol_cm, dwell_frames, leg_timeout_s):
        self.points = [float(p) for p in points] or [0.0]
        self.tol = float(tol_cm)
        self.dwell = int(dwell_frames)
        self.timeout_ms = int(float(leg_timeout_s) * 1000)
        self.idx = 0
        self.n_ok = 0
        self.t_leg = 0
        self.warned = False
        self.done = False
        self.running = False
        self.n_arrive = 0        # 累计到达的点位数

    def start(self):
        """从第一个点重新开始。"""
        self.idx = 0
        self.n_ok = 0
        self.t_leg = time.ticks_ms()
        self.warned = False
        self.done = False
        self.running = True
        self.n_arrive = 0

    def stop(self):
        self.running = False

    def target(self):
        """当前该追的目标位置(cm)。序列走完后恒为最后一个点。"""
        i = self.idx
        if i >= len(self.points):
            i = len(self.points) - 1
        return self.points[i]

    def update(self, x_cm):
        """
        每帧调一次。x_cm=None 表示这一帧没有可信读数。
        返回刚刚到达的点(float)或 None。
        """
        if not self.running or self.done:
            return None
        if x_cm is None:
            return None                      # 未知 ≠ 失败, 见类注释
        wp = self.target()
        d = x_cm - wp
        if (d if d >= 0 else -d) <= self.tol:
            self.n_ok += 1
        elif self.n_ok:
            # 已经攒了一些帧却又跑出容差 → 是路过不是停住, 重新数
            self.n_ok = 0
        if self.n_ok < self.dwell:
            return None
        # 到达
        self.n_arrive += 1
        arrived = wp
        self.n_ok = 0
        self.t_leg = time.ticks_ms()
        self.warned = False
        if self.idx + 1 >= len(self.points):
            self.done = True                 # 停在最后一个点, target 不再变
        else:
            self.idx += 1
        return arrived

    def stuck(self):
        """本段是否已超时(只用来提示, 不改变行为)。"""
        if not self.running or self.done or self.timeout_ms <= 0:
            return False
        return time.ticks_diff(time.ticks_ms(), self.t_leg) > self.timeout_ms

    def status(self):
        """屏上要显示的一行短字符串。"""
        if not self.running:
            return "SEQ idle"
        if self.done:
            return "SEQ done @%+.1f" % self.target()
        return "SEQ %d/%d ->%+.1f  %d/%d" % (
            self.idx + 1, len(self.points), self.target(),
            self.n_ok, self.dwell)


# ===== 触摸调节目标位置 =====

TARGET_FILE = "/sdcard/target.txt"

# 五个增量按钮。使用增量按钮而非滑条: 触摸滑条精度不可控,
# 而每次点击是确定的增量 —— ±1.0 快速到位, ±0.1 微调。
#
# 中间是 RESET 键(一键重置全部状态, 含目标归零), 用于
# 相机被碰、球被移动、序列卡住等情况, 免去重启程序。
BUTTONS = (("-1.0", -1.0), ("-0.1", -0.1), ("RESET", None),
           ("+0.1", +0.1), ("+1.0", +1.0))

# 第 6 个按钮: 图传开关。放在最右侧, 与目标键之间留双倍死区
# (误触图传开关的代价重于误触目标键)。图传默认关闭, 按需开启。
BTN_STREAM = "STREAM"

# 第 7 个按钮: 状态切换 [MODE], 放在**画面右上角**。
# 为什么放上角而不是挤进下面那排: 下面六个键已经占满、再挤会压缩死区,
# 而误点 MODE 的代价最大 —— 它会让球突然开始往 +5 跑(状态①), 比误改目标严重得多。
# 上角离手指常在的下排最远, 是画面上"最不容易蹭到"的位置。
BTN_MODE = "MODE"

# 第 8 个按钮: 阈值调参页开关 [TUNE], 放在 [MODE] 正下方(同一列右上角区)。
BTN_TUNE = "TUNE"

TUNE_FILE = "/sdcard/tune.txt"

# ===== 现场阈值调参 =====
#
# 目标值用增量按钮而阈值可用轨道, 因为两者容错不同:
#   目标值: 误点的代价高且不易察觉 → 必须是确定的增量;
#   阈值:   无唯一正解, 且调错时屏上有直观反馈(识别异常立即可见)。
#
# 四个防错读开关(require_spec / max_jump_cm / warmup_frames / fixed_geom)
# 不开放调参 —— 它们防止严重错读。量程也限定在安全范围内,
# 极限值不会造成危险误判。
#
# (attr, 屏上标签, 下限, 上限, 每格, 小数位)
TUNE_ITEMS = (
    ("spec_l_min",      "specL",  30.0, 80.0, 1.0, 0),
    ("ball_l_max",      "ballL",  30.0, 75.0, 1.0, 0),
    ("ball_max_aspect", "aspect",  1.6,  4.0, 0.1, 1),
    ("spec_pixels_min", "specPx",  2.0, 12.0, 1.0, 0),
)
TUNE_SPEC = {}
for _it in TUNE_ITEMS:
    TUNE_SPEC[_it[0]] = (_it[2], _it[3], _it[4], _it[5])
del _it


class Tuner:
    """
    现场阈值调参页: 显示参数轨道 + 步进增减 + 存盘。
    只持有数值, 不直接修改检测/闸门/串口(与 TouchUI 同样的隔离设计)。

    "选择行"与"修改值"分离: 细轨道仅用于显示与选中(点错代价为零),
    修改使用大按钮步进(每次确定的增量), 规避触摸滑条精度不可控的问题。

    baseline 为 config.json 原值, 轨道上以白色刻痕标示。
    """

    def __init__(self, vals, cam_w, cam_h):
        # vals: {attr: 当前值}, 来自 config(已经过 main() 的解析与默认值)
        self.base = dict(vals)      # config 原值, 只读, 画刻痕用
        self.val = dict(vals)
        self.sel = 0                # 当前选中第几行
        self.on = False             # 调参页是否展开
        self.dirty = False          # 有未同步给检测的改动(主循环消费)
        self.n_change = 0
        # [TUNE] 键位于 [MODE] 正下方, 同宽高
        self.mw = 150
        self.mh = 76
        self.mx = cam_w - self.mw - 10
        self.my = 6 + self.mh + 8
        self._lit_until = 0
        # 轨道画在中部, 避开导轨所在的检测区域。
        self.tr_x = 96                       # 轨道左端(左边留给标签)
        self.tr_w = cam_w - self.tr_x - 130  # 右边留给数值
        self.tr_y0 = int(cam_h * 0.55)
        self.tr_dy = 24
        self.tr_h = 12
        # 底部控制条: 三格布局 [行名][−][+]
        self.bh = 82
        self.by_ = cam_h - self.bh - 8
        gap = 20
        margin = 10
        w3 = (cam_w - 2 * margin - 2 * gap) // 3
        self.cx0 = [margin + i * (w3 + gap) for i in range(3)]
        self.cw = w3

    def load(self):
        """读取调参存档; 无法解析的行直接跳过。"""
        try:
            with open(TUNE_FILE, "r") as f:
                txt = f.read()
        except Exception:
            print("  阈值调参 无存档, 用 config.json 里的值")
            return
        n = 0
        for line in txt.split("\n"):
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            if k not in TUNE_SPEC:
                continue
            try:
                fv = float(v.strip())
            except Exception:
                continue
            lo, hi, _st, _d = TUNE_SPEC[k]
            if lo <= fv <= hi:          # 越界的存档值一律不采纳(量程即安全边界)
                self.val[k] = fv
                n += 1
        if n:
            print("  阈值调参 从 %s 读回 %d 个值:" % (TUNE_FILE, n))
            for attr, label, _lo, _hi, _st, dec in TUNE_ITEMS:
                if self.val[attr] != self.base[attr]:
                    print("     %-7s %s  (config 原值 %s)"
                          % (label, self._fmt(attr), self._fmt(attr, self.base[attr])))
            self.dirty = True

    def save(self):
        """保存调参值; 失败只警告, 内存中的值仍然生效。"""
        try:
            with open(TUNE_FILE, "w") as f:
                for attr, _l, _lo, _hi, _st, _d in TUNE_ITEMS:
                    f.write("%s=%.2f\n" % (attr, self.val[attr]))
            return True
        except Exception as e:
            print("  ⚠️ 阈值存盘失败(内存里仍生效):", e)
            return False

    def _fmt(self, attr, v=None):
        dec = TUNE_SPEC[attr][3]
        if v is None:
            v = self.val[attr]
        return ("%.1f" if dec else "%.0f") % v

    def bump(self, d):
        """选中行 ±1 格。夹在量程内 —— 拖到底也拖不出危险值。"""
        attr, label, lo, hi, step, _dec = TUNE_ITEMS[self.sel]
        old = self.val[attr]
        v = old + d * step
        if v < lo:
            v = lo
        elif v > hi:
            v = hi
        if v == old:
            return False
        self.val[attr] = v
        self.dirty = True
        self.n_change += 1
        ok = self.save()
        print("  [TUNE] %s %s → %s  (config 原值 %s)%s"
              % (label, self._fmt(attr, old), self._fmt(attr),
                 self._fmt(attr, self.base[attr]), "" if ok else " 存盘失败"))
        return True

    def hit(self, cx, cy):
        """
        调参页展开时的命中判定。返回 True = 这一下被调参页吃掉了。

        注意: 返回 True 时调用方不应再判断目标键 —— 调参页展开时下排是
        [行名][−][+], 底下没有目标键了, 但 y 区间是重叠的。
        """
        # 底部控制条
        if self.by_ <= cy < self.by_ + self.bh:
            for i, x0 in enumerate(self.cx0):
                if x0 <= cx < x0 + self.cw:
                    if i == 0:
                        # 左格 = 切到下一行(循环)。不做加减, 所以点错零代价。
                        self.sel = (self.sel + 1) % len(TUNE_ITEMS)
                        print("  [TUNE] 选中 %s = %s"
                              % (TUNE_ITEMS[self.sel][1],
                                 self._fmt(TUNE_ITEMS[self.sel][0])))
                    elif i == 1:
                        self.bump(-1)
                    else:
                        self.bump(+1)
                    return True
            return True         # 落死区: 也算被调参页吃掉, 别透传到目标键
        # 四条轨道: 点哪条选哪条(只切换选中, **不改值**)
        for i in range(len(TUNE_ITEMS)):
            y0 = self.tr_y0 + i * self.tr_dy
            if y0 - 6 <= cy < y0 + self.tr_h + 6 and self.tr_x - 90 <= cx:
                if self.sel != i:
                    self.sel = i
                    print("  [TUNE] 选中 %s = %s"
                          % (TUNE_ITEMS[i][1], self._fmt(TUNE_ITEMS[i][0])))
                return True
        return False

    def draw_button(self, img, now):
        """[TUNE] 键。底色编码状态: 展开=青亮 / 收起=深灰。"""
        if now < self._lit_until:
            col = (255, 255, 0)
        elif self.on:
            col = (0, 140, 150)
        else:
            col = (50, 50, 60)
        img.draw_rectangle(self.mx, self.my, self.mw, self.mh,
                           color=col, thickness=1, fill=True)
        img.draw_rectangle(self.mx, self.my, self.mw, self.mh,
                           color=(230, 230, 230), thickness=2)
        img.draw_string_advanced(self.mx + 8, self.my + 4, 20, BTN_TUNE,
                                 color=(210, 210, 210))
        img.draw_string_advanced(self.mx + 8, self.my + 28, 34,
                                 "ON" if self.on else "OFF",
                                 color=(120, 255, 255) if self.on
                                 else (180, 180, 180))

    def draw_page(self, img, ok_l, rej_l, n_wall, n_round):
        """
        调参页: 四条轨道 + 底部控制条 + 实测 L 读数行。
        """
        # 球 L 与被拒块 L 并排显示, 阈值取两者中间(单看一个数无法定阈值)。
        s = "ball L %s | rej L %s | rej wall%d round%d" % (
            ("%.0f" % ok_l) if ok_l is not None else "--",
            ("%.0f" % rej_l) if rej_l is not None else "--",
            n_wall, n_round)
        # 两者重叠时转红, 表示仅靠 L 已无法区分球与导轨壁
        col = (255, 230, 120)
        if ok_l is not None and rej_l is not None and ok_l <= rej_l:
            col = (255, 90, 90)
        img.draw_string_advanced(10, self.tr_y0 - 26, 20, s, color=col)
        for i, (attr, label, lo, hi, _st, _dec) in enumerate(TUNE_ITEMS):
            y0 = self.tr_y0 + i * self.tr_dy
            cur = self.val[attr]
            span = hi - lo
            sel = (i == self.sel)
            img.draw_string_advanced(4, y0 - 4, 20, label,
                                     color=(255, 255, 255) if sel
                                     else (170, 170, 170))
            # 轨道底
            img.draw_rectangle(self.tr_x, y0, self.tr_w, self.tr_h,
                               color=(45, 45, 55), thickness=1, fill=True)
            # 白刻痕 = config.json 中的原值, 拖回此处即恢复基线。
            bx = self.tr_x + int(self.tr_w * (self.base[attr] - lo) / span)
            img.draw_rectangle(bx - 1, y0 - 4, 3, self.tr_h + 8,
                               color=(255, 255, 255), thickness=1, fill=True)
            # 已填充部分 + 游标
            fw = int(self.tr_w * (cur - lo) / span)
            if fw > 0:
                img.draw_rectangle(self.tr_x, y0, fw, self.tr_h,
                                   color=(0, 150, 160) if sel else (60, 90, 95),
                                   thickness=1, fill=True)
            kx = self.tr_x + fw
            img.draw_rectangle(kx - 5, y0 - 3, 10, self.tr_h + 6,
                               color=(0, 230, 240) if sel else (130, 130, 140),
                               thickness=1, fill=True)
            img.draw_string_advanced(self.tr_x + self.tr_w + 8, y0 - 4, 20,
                                     self._fmt(attr),
                                     color=(255, 255, 255) if sel
                                     else (170, 170, 170))
        # 底部三格 [行名][−][+]
        attr, label, _lo, _hi, step, dec = TUNE_ITEMS[self.sel]
        caps = (label, "-%s" % (("%.1f" if dec else "%.0f") % step),
                "+%s" % (("%.1f" if dec else "%.0f") % step))
        for i, x0 in enumerate(self.cx0):
            img.draw_rectangle(x0, self.by_, self.cw, self.bh,
                               color=(40, 70, 75) if i == 0 else (55, 55, 85),
                               thickness=1, fill=True)
            img.draw_rectangle(x0, self.by_, self.cw, self.bh,
                               color=(200, 200, 200), thickness=2)
            img.draw_string_advanced(x0 + 10, self.by_ + 6, 22, caps[i],
                                     color=(255, 255, 255))
            if i == 0:
                img.draw_string_advanced(x0 + 10, self.by_ + 40, 30,
                                         self._fmt(attr), color=(0, 230, 240))
            else:
                img.draw_string_advanced(x0 + 10, self.by_ + 42, 20, "tap",
                                        color=(160, 160, 160))


class TouchUI:
    """
    触摸交互: 目标值调节按钮 + 存盘。

    本类只管理目标值, 不涉及检测/闸门/滤波 —— 故意的隔离设计,
    触摸失效时最坏后果只是无法修改目标, 检测链路不受影响。
    """

    def __init__(self, cfg, cam_w, cam_h):
        self.ok = False
        self.enabled = bool(cfg.get("enabled", False))
        # 触摸原始坐标 → 640x480 画布坐标的仿射映射(实测标定)。
        # 使用总映射而非分层映射: Display 的缩放/排布方式被复合结果自然吸收,
        # 避免建模中间环节。系数由实测得出, 勿按分辨率比例推算。
        m = cfg.get("map") or {}
        self.kx = float(m.get("kx", 0.9870))
        self.bx = float(m.get("bx", 14.6))
        self.ky = float(m.get("ky", 1.1061))
        self.by = float(m.get("by", -32.5))
        self.tmin = float(cfg.get("target_min", -11.0))
        self.tmax = float(cfg.get("target_max", 11.0))
        self.every = int(cfg.get("read_every_frames", 5))
        if self.every < 1:
            self.every = 1
        self.target = 0.0
        self._down = False        # 手指当前是否按住(去抖状态机)
        self._miss = 0
        self._hit_i = None        # 最近命中的按钮(高亮用)
        self._hit_until = 0
        self._touch = None
        # 按钮几何(画布坐标)。按钮间留死区: 触摸校正存在残差,
        # 死区可防止误触相邻按钮。宁可漏按一次, 不可错按。
        self.bh = 82
        self.by_ = cam_h - self.bh - 8
        gap = 20
        margin = 10
        # 五个目标键排在左侧, 右侧留给图传开关;
        # 两者误触代价不同, 中间留较宽死区。
        zone = int((cam_w - 2 * margin) * 0.76)
        self.bw = (zone - 4 * gap) // len(BUTTONS)
        self.bxs = [margin + i * (self.bw + gap) for i in range(len(BUTTONS))]
        # 图传键
        self.sx = margin + zone + gap
        self.sw = cam_w - margin - self.sx
        if self.sw < 60:              # 画布过窄时不画图传键
            self.sw = 0
        self.stream_on = False        # 由外部每帧同步真实状态进来
        self.stream_toggle = False    # 待处理的开关请求, 由主循环消费并清零
        self._s_lit_until = 0
        # [MODE] 状态切换键: 右上角, 只置标志、不在 poll 中执行动作。
        # 按钮尺寸留有触摸残差余量(角部残差更大), 不宜缩小。
        self.mw = 150
        self.mh = 76
        self.mx = cam_w - self.mw - 10
        self.my = 6
        self.mode = MODE_HOLD         # 由主循环每帧同步真实状态进来(这里只作显示)
        self.mode_toggle = False      # 待处理的切换请求
        self.reset_req = False        # 待处理的一键重置请求(RESET 键)
        self._m_lit_until = 0
        self._r_lit_until = 0
        # 阈值调参页(现场光线不定 → 要能即时改门限)。由 main() 注入,
        # None = 没启用。⚠️ TouchUI 只负责**把触摸事件转给它**, 不读也不改它的值 ——
        # 阈值怎么用是检测那边的事, 这里保持一样的隔离。
        self.tuner = None

    def open(self):
        """初始化触摸并读回上次的目标值。失败只警告不抛 —— 触摸坏了不该让检测停摆。"""
        self.target = self.load()
        if not self.enabled:
            print("  触摸调目标 未启用(config.ball_pos.touch.enabled=false)")
            return
        try:
            from machine import TOUCH
            self._touch = TOUCH(0)
            self.ok = True
            print("  触摸调目标 已启用: 五个按钮 -1.0/-0.1/ZERO/+0.1/+1.0 "
                  "+ [STREAM] 图传开关, 每 %d 帧读一次" % self.every)
            print("     画布x=%.4f*触摸x%+.1f  画布y=%.4f*触摸y%+.1f (实测)"
                  % (self.kx, self.bx, self.ky, self.by))
        except Exception as e:
            # 未接屏时走此路径(属正常) —— 目标值使用上次存档,
            # 检测/闸门/串口全都不受影响(TouchUI 只管目标值, 是故意隔离的)。
            print("  触摸未就绪(%r) → 屏没插? 目标值用存档的 %+.2f cm, "
                  "检测与串口照常" % (e, self.target))

    def load(self):
        """读 /sdcard/target.txt。文件不存在或内容坏 → 回 0.0, 绝不抛异常。"""
        try:
            with open(TARGET_FILE, "r") as f:
                v = float(f.read().strip())
            if self.tmin <= v <= self.tmax:
                print("  目标位置 从 %s 读回 %+.2f cm" % (TARGET_FILE, v))
                return v
            print("  ⚠️ %s 里的 %.2f 超出 [%.1f,%.1f] → 归零"
                  % (TARGET_FILE, v, self.tmin, self.tmax))
        except Exception:
            print("  目标位置 无存档, 从 0.00 cm 起(掉电后会存到 %s)" % TARGET_FILE)
        return 0.0

    def save(self):
        """保存目标值; 失败只警告, 内存中的值仍然生效。"""
        try:
            with open(TARGET_FILE, "w") as f:
                f.write("%.2f" % self.target)
            return True
        except Exception as e:
            print("  ⚠️ 目标值存盘失败(内存里仍生效):", e)
            return False

    def _hit(self, cx, cy):
        """
        画布坐标 → 命中的按钮。落在死区里返回 None(宁可漏不可错)。

        返回: 0..4 = 目标增量键 / "S" = 图传开关 / "M" = 状态切换 /
              "T" = 调参页开关 / "X" = 已被调参页消费掉 / None = 没命中
        """
        # [MODE] 在右上角, 与下排按钮不重叠, 先判断可省一次循环。
        if (self.mx <= cx < self.mx + self.mw
                and self.my <= cy < self.my + self.mh):
            return "M"
        t = self.tuner
        if t is not None:
            # [TUNE] 键在 [MODE] 正下方, 不管页面开没开都要能点(否则开了关不掉)
            if (t.mx <= cx < t.mx + t.mw and t.my <= cy < t.my + t.mh):
                return "T"
            # 调参页展开时必须先由它处理本次点击再判断目标键:
            # 调参页与目标键的 y 区间完全重叠, 顺序颠倒会同时触发两者。
            if t.on and t.hit(cx, cy):
                return "X"
        if not (self.by_ <= cy < self.by_ + self.bh):
            return None
        for i, bx in enumerate(self.bxs):
            if bx <= cx < bx + self.bw:
                return i
        if self.sw and self.sx <= cx < self.sx + self.sw:
            return "S"
        return None

    def poll(self, n_frame):
        """
        每帧调一次, 内部自己按 read_every_frames 节流。返回 True=目标值刚被改过。

        去抖使用状态机而非阻塞等待: 阻塞会卡住主循环导致检测停帧;
        状态机在按住期间主循环照常运行, 只是不重复触发。
        """
        if not self.ok or (n_frame % self.every):
            return False
        try:
            tp = self._touch.read(1)
            self._rd_fail = 0
        except Exception as e:
            # 运行**中途**拔屏 → read 会一直抛。连续失败就彻底停掉触摸,
            # 否则每 5 帧都吃一次异常(异常在 MicroPython 上不便宜)。
            self._rd_fail = getattr(self, "_rd_fail", 0) + 1
            if self._rd_fail >= 10:
                self.ok = False
                print("  [触摸] 连续读失败 → 停用触摸(屏被拔了?), "
                      "目标值锁定在 %+.2f cm, 检测与串口照常" % self.target)
            return False
        # 只有 event 2(DOWN)/3(MOVE) 算按下; UP 也带一个有坐标的点, 不能用 len() 判松手
        p = None
        if tp and len(tp):
            e = tp[0].event
            if e in (2, 3):
                p = tp[0]
        if p is None:
            # 上报稀疏(一次按压可能只报 1 个点) → 连续几次读空才判定松手,
            # 否则两次上报之间的空隙会被误判成松手→同一次按压触发两下
            self._miss += 1
            if self._miss >= 3:
                self._down = False
            return False
        self._miss = 0
        if self._down:            # 还是上一次那下按压, 不重复触发
            return False
        self._down = True

        cx = self.kx * p.x + self.bx
        cy = self.ky * p.y + self.by
        i = self._hit(cx, cy)
        if i is None:
            return False
        if i == "S":
            # 图传开关: 只置一个待处理标志, **不在这里做实际开关** ——
            # 起网要 2 秒、关流要关 socket, 在 poll 里做等于把主循环卡住 2 秒
            # (球在动而我们瞎了)。交给主循环在合适的位置处理。
            self.stream_toggle = True
            self._s_lit_until = time.ticks_ms() + 250
            return False
        if i == "M":
            # 状态切换: 同样只置标志。真正的切换要动 Sequence/闸门/滤波,
            # 让主循环在一个地方统一做 —— 状态的写入点只有一处才好推理。
            self.mode_toggle = True
            self._m_lit_until = time.ticks_ms() + 250
            return False
        if i == "X":
            # 已被调参页消费(它自己完成了选行/加减/存盘) —— 这里什么都不做。
            # 阈值改动经 tuner.dirty 标志交给主循环同步, 和 STREAM/RESET 一个套路。
            return False
        if i == "T":
            # 调参页开合: 纯显示层的事, 不动任何检测状态 → 可以就地做完,
            # 不用像 STREAM(起网 2 秒)/RESET(清缓存) 那样绕主循环。
            self.tuner.on = not self.tuner.on
            self.tuner._lit_until = time.ticks_ms() + 250
            print("  [TUNE] 阈值调参页 %s%s"
                  % ("展开" if self.tuner.on else "收起",
                     " —— 下排按钮暂时变成 [行名][−][+], 目标键已隐藏"
                     if self.tuner.on else " —— 下排恢复五个目标键"))
            return False
        label, delta = BUTTONS[i]
        old = self.target
        if delta is None:
            # RESET: 这里**只置标志并把目标归零**, 重建管子/清闸门/停序列都交给
            # 主循环 —— 那些状态(tube/gate/medf/seq/warmup)全在 main() 的局部变量里,
            # TouchUI 碰不到也**不该**碰到(它的隔离性是故意的: 触摸坏了检测照跑)。
            self.target = 0.0
            self.reset_req = True
            self._r_lit_until = time.ticks_ms() + 400
        else:
            v = self.target + delta
            # 夹到量程内: ROI 左右各内收 3% → 能找球的范围约 ±11.7cm
            self.target = self.tmax if v > self.tmax else (
                self.tmin if v < self.tmin else v)
        self._hit_i = i
        self._hit_until = time.ticks_ms() + 250
        ok = self.save()
        print("  [触摸] %s → 目标 %+.2f → %+.2f cm%s"
              % (label, old, self.target, "" if ok else " (存盘失败)"))
        return True

    def draw(self, img):
        """
        绘制按钮; 命中的按钮高亮, 提供点击反馈。

        ⚠️ 触摸不可用(self.ok=False)时**仍然画右上角那个状态框**, 只是不画按钮:
        状态还能被**电控串口命令**切换, 所以"现在是哪个状态"必须一直看得见。
        少了它, 屏还亮但触摸坏掉时球突然往 +5 跑就完全无法解释。
        """
        now = time.ticks_ms()
        if not self.ok:
            self._draw_mode(img, now)
            return
        if self.tuner is not None:
            self.tuner.draw_button(img, now)
            if self.tuner.on:
                # 调参页展开: 下排让给 [行名][−][+], **不画五个目标键** ——
                # 画了会让人以为能点(而它们已经被 _hit 挡在 "X" 后面了),
                # 显示与可点击区域保持一致; 调参页由主循环绘制(含实测 L 数据)。
                self._draw_mode(img, now)
                return
        lit = self._hit_i if now < self._hit_until else None
        for i, (label, _d) in enumerate(BUTTONS):
            bx = self.bxs[i]
            if i == lit:
                col = (0, 200, 255)
            elif label == "RESET":
                # RESET 常态就用暗红打底: 它现在会清掉一堆状态, 不再是无害的"归零"。
                # 颜色用于区分模式, 降低误触代价。
                col = (95, 35, 40)
            else:
                col = (55, 55, 85)
            img.draw_rectangle(bx, self.by_, self.bw, self.bh,
                               color=col, thickness=1, fill=True)
            img.draw_rectangle(bx, self.by_, self.bw, self.bh,
                               color=(200, 200, 200), thickness=2)
            # RESET 五个字比 ±1.0 长, 字号小一点才不出框
            fs = 26 if label == "RESET" else 32
            img.draw_string_advanced(bx + 8, self.by_ + 24, fs, label,
                                     color=(255, 255, 255))
        # 图传开关: 开=绿底 关=深灰底, 一眼看出当前算力给了谁
        if self.sw:
            if now < self._s_lit_until:
                col = (255, 255, 0)              # 刚点到, 黄闪
            elif self.stream_on:
                col = (0, 130, 60)               # 开着, 绿
            else:
                col = (60, 60, 60)               # 关着, 灰
            img.draw_rectangle(self.sx, self.by_, self.sw, self.bh,
                               color=col, thickness=1, fill=True)
            img.draw_rectangle(self.sx, self.by_, self.sw, self.bh,
                               color=(200, 200, 200), thickness=2)
            img.draw_string_advanced(self.sx + 8, self.by_ + 6, 22, "STREAM",
                                     color=(255, 255, 255))
            img.draw_string_advanced(self.sx + 8, self.by_ + 40, 30,
                                     "ON" if self.stream_on else "OFF",
                                     color=(120, 255, 160) if self.stream_on
                                     else (180, 180, 180))
        self._draw_mode(img, now)

    def _draw_mode(self, img, now):
        # [MODE] 右上角。**底色直接编码当前状态**, 不只是个按钮:
        #   蓝 = HOLD(状态②稳定在某点)  /  橙 = SEQ(状态①正在跑往返)
        # 最危险的误判是"以为在 HOLD 其实在 SEQ"(目标会自动移动),
        # 所以状态显示要比按钮标签更显眼 —— 大字写状态名, 小字写 MODE。
        if now < self._m_lit_until:
            mcol = (255, 255, 0)                  # 刚点到, 黄闪
        elif self.mode == MODE_SEQ:
            mcol = (200, 110, 0)                  # 状态① 橙
        else:
            mcol = (30, 70, 150)                  # 状态② 蓝
        img.draw_rectangle(self.mx, self.my, self.mw, self.mh,
                           color=mcol, thickness=1, fill=True)
        img.draw_rectangle(self.mx, self.my, self.mw, self.mh,
                           color=(230, 230, 230), thickness=2)
        img.draw_string_advanced(self.mx + 8, self.my + 4, 20, BTN_MODE,
                                 color=(210, 210, 210))
        img.draw_string_advanced(self.mx + 8, self.my + 28, 34,
                                 MODE_NAMES[self.mode],
                                 color=(255, 255, 255))


def main():
    config = load_config("config.json")
    cfg = config.get("ball_pos", {})

    tube_thr = _cfg_thr(cfg.get("tube"), "tube")
    ball_thr = _cfg_thr(cfg.get("ball"), "ball")
    spec_l_min = int(cfg.get("spec_l_min", 60))
    tube_len_cm = float(cfg.get("tube_len_cm", 25.0))
    inset_frac = float(cfg.get("inset_frac", 0.22))
    inset_x_frac = float(cfg.get("inset_x_frac", 0.03))
    # 高光必须落在候选块内部(不是擦边) —— 挡管壁亮边冒充球的关键一条
    spec_inside = bool(cfg.get("require_spec_inside", True))
    # 形状筛: 球圆(宽高比≈1) vs 管壁细长。**与光照无关**, 是当前最可靠的一条
    ball_max_aspect = float(cfg.get("ball_max_aspect", 2.2))
    tube_px_min = int(cfg.get("tube_pixels_min", 2000))
    ball_px_min = int(cfg.get("ball_pixels_min", 15))
    spec_px_min = int(cfg.get("spec_pixels_min", 2))
    min_aspect = float(cfg.get("tube_min_aspect", 8.0))
    min_fill = float(cfg.get("tube_min_fill", 0.5))
    band = cfg.get("search_band")           # None 或 [y_frac_min, y_frac_max]
    if band is not None and len(band) == 2:
        band = (float(band[0]), float(band[1]))
    else:
        band = None
    invert_x = bool(cfg.get("invert_x", False))
    show_fps = bool(cfg.get("show_fps", True))
    # 滑动中值滤波: 压抖动, 代价是 (N-1)/2 帧延迟。1=关闭
    med_n = int(cfg.get("median_window", 5))
    # 实测标定的两个修正数
    offset_cm = float(cfg.get("offset_cm", 0.0))
    scale_correction = float(cfg.get("scale_correction", 1.0))
    # 连续丢球多少帧后清空滤波窗口(否则会输出陈旧位置)
    lost_reset = int(cfg.get("lost_reset_frames", 5))
    # 防错读闸门, 不建议关闭
    require_spec = bool(cfg.get("require_spec", True))
    max_jump_cm = float(cfg.get("max_jump_cm", 3.0))
    jump_reject_limit = int(cfg.get("jump_reject_limit", 8))
    # 启动预热帧数: 摄像头 AE/AWB 上电后需要若干帧才收敛, 期间画面偏色
    # 会导致检测失效。预热期丢弃检测结果, 对稳态零代价。
    warmup_frames = int(cfg.get("warmup_frames", 20))
    # 固定标尺模式(见 px_to_cm_fixed 的说明)
    fg = cfg.get("fixed_geom") or {}
    fg_on = bool(fg.get("enabled", False))
    fg_origin = float(fg.get("origin_px", 0.0))
    fg_ppcm = float(fg.get("px_per_cm", 0.0))
    fg_tol = float(fg.get("geom_check_tol", 0.05))
    if fg_on and fg_ppcm <= 0:
        # 未填标尺时退回动态测量, 避免除零。
        print("  ⚠️ fixed_geom.enabled=true 但 px_per_cm 没填 → 本次退回每帧量")
        fg_on = False
    fg_checked = False       # 体检只做一次
    printed_geom = False     # [标定常数] 只打一次(预热结束后的第一次锁定)
    # 叠加层(紫色 0 刻度线 + 指向它的箭头链)
    ov = cfg.get("overlay", {})
    ov_on = bool(ov.get("enabled", True))
    ov_zero_w = int(ov.get("zero_line_width", 6))
    ov_zero_half = int(ov.get("zero_line_half_len", 95))
    ov_gap = int(ov.get("arrow_spacing", 22))
    ov_size = int(ov.get("arrow_size", 9))
    ov_thick = int(ov.get("arrow_thickness", 2))
    # 序列模式的目标点序列
    sq = cfg.get("sequence") or {}
    seq = Sequence(sq.get("points") or [0.0, 5.0, -5.0],
                   float(sq.get("tol_cm", 1.0)),
                   int(sq.get("dwell_frames", 15)),
                   float(sq.get("leg_timeout_s", 20.0)))
    seq_autostart = bool(sq.get("autostart", False))
    # 目标值保存在本机; 串口发送的是误差(下位机 PID 目标恒 0)
    cam_w = int(config.get("camera", {}).get("width", 640))
    cam_h = int(config.get("camera", {}).get("height", 480))
    ui = TouchUI(cfg.get("touch") or {}, cam_w, cam_h)

    # 现场阈值调参页: 光照变化时需即时调整这四个门限。
    # 只提供"区分球与导轨壁"的门限, 不含防错读开关。
    # ball_l_max = ball_thr[1](暗块 L 上限), 从六元阈值中单独抽出。
    tuner = None
    if bool((cfg.get("tune") or {}).get("enabled", True)):
        tuner = Tuner({"spec_l_min": float(spec_l_min),
                       "ball_l_max": float(ball_thr[1]),
                       "ball_max_aspect": float(ball_max_aspect),
                       "spec_pixels_min": float(spec_px_min)}, cam_w, cam_h)
        tuner.load()
        ui.tuner = tuner

    from camera import Camera
    from display import Display

    cam = Camera(config)

    # Display 必须先于图传创建, 顺序不可颠倒:
    # 图传起网里有 time.sleep(2) + 建 socket, 全发生在 Display.init() 之前,
    # 打乱了"Display.init() 必须先于 MediaManager.init()"的顺序要求(见 camera.py)。
    # 屏不亮则触摸 UI 失效, 无法调节目标位置。
    disp = Display(config) if config.get("display", {}).get("enabled", True) else None

    # 图传: WiFi 热点 + MJPEG-over-HTTP, 同步编码。
    # 只构造不启动: 由屏上的 [图传] 按钮按需开启(见 TouchUI),
    # 或 config.video_stream.autostart=true 让它开机自启。
    vs = None
    vs_cfg = config.get("video_stream") or {}
    if vs_cfg.get("enabled", False):
        try:
            from video_stream import VideoStream
            vs = VideoStream(vs_cfg)
            cam.attach_stream(vs)     # 只为收尾时连带关掉它
            if vs_cfg.get("autostart", False):
                if not vs.start():    # 失败它自己会打原因
                    print("  [图传] 自启失败, 可在屏上点 [图传] 按钮重试")
            else:
                print("  [图传] 已就绪但**未启动** → 屏上点 [图传] 按钮开启")
                print("        (不开启则全部算力用于视觉)")
        except Exception as e:
            print("  [图传] 模块加载失败, 跳过图传继续跑视觉: %r" % (e,))
            vs = None

    # 串口: 单向发送误差给电控, ball8 协议见 serial_comm.py
    ser = None
    scfg = config.get("serial", {})
    if scfg.get("enabled", False):
        from serial_comm import SerialComm
        ser = SerialComm(uart_id=int(scfg.get("uart", 2)),
                         baudrate=int(scfg.get("baudrate", 115200)),
                         tx_pin=int(scfg.get("tx_pin", 5)),
                         rx_pin=int(scfg.get("rx_pin", 6)),
                         head=scfg.get("head"), end=scfg.get("end"))
        if not ser.is_open():
            print("  [警告] 串口打开失败, 本次只显示不发送")
            ser = None
        else:
            print("-" * 58)
            print("  串口 UART%d @%d  TX=IO%d  协议 ball8 (只发不收)"
                  % (int(scfg.get("uart", 2)), int(scfg.get("baudrate", 115200)),
                     int(scfg.get("tx_pin", 5))))
            print("  发送量 = **误差** = X实际 − 目标, 单位 0.01cm")
            print("     负 = 球在目标左侧(该往右赶) / 正 = 球在目标右侧(该往左赶)")
            print("     电控还原: err_cm = (xpos - 32768) / 100.0")
            print("  ⚠️ 电控 PID 目标须恒设 0, **不可再减一次目标**"
                  "(减两次 → 球反向跑)")
            if str(scfg.get("head")) == "0xAA" and str(scfg.get("end")) == "0x55":
                print("  ⚠️ 帧头帧尾仍是默认值 0xAA/0x55, 与电控核对真值")
    else:
        print("  串口未启用(config.serial.enabled=false) → 只显示不发送")

    print("=" * 58)
    print("  钢球一维位置检测  (导轨内找非绿暗块 + 高光校验)")
    print("  tube LAB %s" % (tube_thr,))
    print("  ball LAB %s   spec L>%d" % (ball_thr, spec_l_min))
    if spec_l_min < 55:
        # 门限被放宽过 → 明确说出来。否则下次有人对着"报告里写 60"查半天,
        # 或者反过来: 照明恢复后忘了这个数还松着(松门限=切口阴影更容易冒充球)。
        print("  ⚠️ spec_l_min=%d 是**为暗光调过的**(原 60)。切口阴影实测 L=16~18,"
              % spec_l_min)
        print("     所以 %d 仍在安全区; 但照明改善后建议调回 55~60 —— "
              "门限越松, 管壁/切口阴影越容易冒充球。" % spec_l_min)
    print("  高光须在候选块**内部**: %s  ← 挡'管壁亮边擦边命中'的关键一条"
          % ("是(推荐)" if spec_inside else "⚠️否, 擦边也算(管壁易冒充球)"))
    print("     球是圆镜面体, 高光必在中心附近, 物理上不会长在轮廓边缘 ——")
    print("     这是**形状+物理**判据, 比调 LAB 阈值稳(同管子那两条几何约束的思路)")
    if inset_x_frac > 0.035:
        print("  ⚠️ 左右收缩 %.0f%%(原 3%%): 暗光下两端发黑, 收多一点避开端部暗区。"
              % (inset_x_frac * 100))
        print("     代价: 可测范围约 ±%.1fcm。球在两端测不到就把 inset_x_frac 调回 0.03"
              % (tube_len_cm * (0.5 - inset_x_frac)))
    print("  摆杆全长 %.1f cm  → 两端都要在画面里, 尽量占满画面宽" % tube_len_cm)
    if med_n > 1:
        print("  中值滤波 N=%d  → 延迟约 %.0f ms @33fps (压抖动的代价)"
              % (med_n, (med_n - 1) / 2.0 * 30.3))
    else:
        print("  中值滤波 关闭(N=1)")
    print("  防错读闸门①无高光帧: %s" % ("拒绝(推荐)" if require_spec else "⚠️放行"))
    print("  防错读闸门②跳变门限: %s"
          % (("%.1f cm/帧" % max_jump_cm) if max_jump_cm > 0 else "⚠️关闭"))
    if offset_cm != 0.0 or scale_correction != 1.0:
        print("  标定修正 offset %+.3f cm  scale x%.4f" % (offset_cm, scale_correction))
    else:
        print("  标定修正 未做(offset 0 / scale 1) ← 相机固定到摆杆后按 config 注释测")
    if warmup_frames > 0:
        print("  启动预热 %d 帧(约 %.1f 秒): 等 AE/AWB 收敛, 期间不发串口也不统计"
              % (warmup_frames, warmup_frames / 33.0))
    else:
        print("  启动预热 关闭 ← ⚠️前几十帧画面未收敛, 可能发出错位置")
    if fg_on:
        print("  标尺已固定 原点px=%.1f  px_per_cm=%.3f (不再每帧测量)"
              % (fg_origin, fg_ppcm))
    else:
        print("  标尺每帧从管子量 ← 每次上电会变(实测散布1.4cm), "
              "跑一次看下面[标定常数]那行, 抄进 config.fixed_geom")
    print("  屏上: 黄框=绿管 紫线=目标位置 青箭头=该往哪赶 红十字=球")
    print("-" * 58)
    print("  两个工作状态, 右上角 [MODE] 键切换")
    print("  状态② HOLD(蓝, 默认): 目标由触摸按钮设定, 球稳定在该点")
    print("  状态① SEQ (橙): 目标自动走 %s, 到位即折返, 末点停住"
          % (" → ".join("%+.1f" % p for p in seq.points),))
    print("     到达判据: |X−目标|≤%.1fcm 连续 %d 帧(≈%.1f秒), 等球停住而非路过"
          % (seq.tol, seq.dwell, seq.dwell / 33.0))
    print("  [RESET] 键(原 ZERO): 一键重置 —— 目标归零 + 重找绿管 + 重算标尺")
    print("          + 清滤波/闸门锚点 + 停序列 + 重新预热, 相当于软重启检测链")
    if tuner is not None:
        print("-" * 58)
        print("  [TUNE] 键(MODE 正下方): 现场阈值滑条, 光线变化时用它调整")
        print("     四个门限 %s"
              % (" / ".join(it[1] for it in TUNE_ITEMS),))
        print("     操作: 点轨道选行(零代价) → 下排 [−]/[+] 每次一格 → 自动存盘")
        print("     轨道上的白刻痕 = config.json 中的原值, 拖回刻痕即恢复基线")
        print("     页面顶部实时显示 ball L 与 rej L: 门限应卡在这两个数中间,")
        print("        两者重叠会转红 → 那说明别再调 L 了, L 已分不开球和管壁")
        print("     ⛔ 四个防错读开关(require_spec/max_jump_cm/warmup/fixed_geom)")
        print("        **故意不给滑条** —— 它们挡的是 19cm 错读, 关掉球会反向飞")
        print("     量程已写死在安全区内: 拖到底也拖不出危险值(存档越界值不采纳)")
    ui.open()
    print("=" * 58)

    relock_every = int(cfg.get("tube_relock_frames", 0))   # 0=锁死不重找
    lock_tube = bool(cfg.get("tube_lock", True))

    # 计时统计
    n_frame = 0
    n_hit = 0
    n_spec = 0
    t_detect_sum = 0.0
    t_start = time.ticks_ms()
    last_report = t_start

    tube = None             # 缓存的 TubeRect
    n_relock = 0
    # 精度统计: 静态抖动即测量噪声; 滤波前后都统计以便验证滤波效果
    xr_min = None           # raw = 滤波前
    xr_max = None
    x_min = None            # 滤波后(实际输出给电控的量)
    x_max = None
    x_sum = 0.0
    x_n = 0
    last_fps_frame = 0
    last_fps_ms = t_start
    medf = MedianFilter(med_n)
    gate = JumpGate(max_jump_cm, jump_reject_limit)
    n_lost = 0              # 连续丢球帧数
    n_nospec = 0            # 因无高光被拒的帧数
    # 被拒绝块的实测最大 L, 用于判断 spec_l_min 是否设得过高。
    nospec_l_max = None
    nospec_l_sum = 0.0
    nospec_l_n = 0
    # 被拒块的形状分类: 细长 = 导轨壁(判据正常), 近方形 = 可能是真球被误拒
    n_rej_wall = 0
    n_rej_round = 0
    # 成功识别时的高光 L, 与被拒块的 L 对比可确定阈值位置
    ok_l_min = None
    ok_l_sum = 0.0
    ok_l_n = 0
    n_tx = 0                # 成功发出的包数
    n_tx_fail = 0           # 发送失败次数
    # 预热基准帧号: 判据为"距 warm_from 不足 warmup_frames 帧",
    # 使 [RESET] 重置后能重新预热。
    warm_from = 0
    # mode 的写入点仅三处: 本行初值 / autostart / [MODE]键 / [RESET]
    mode = MODE_HOLD
    n_mode_sw = 0           # 切换次数(诊断用)
    n_reset = 0             # 重置次数
    if seq_autostart:
        mode = MODE_SEQ
        seq.start()
        print("  [状态] config.sequence.autostart=true → 开机直接进状态① SEQ")

    try:
        while True:
            img = cam.snapshot()
            n_frame += 1

            warming = (n_frame - warm_from) <= warmup_frames

            # 本帧目标位置: 两种模式仅在此处分岔, 下游完全一致。
            # 整帧只取一次, 避免序列在帧中推进导致显示与串口使用不同目标。
            target_now = seq.target() if mode == MODE_SEQ else ui.target

            t0 = time.ticks_us()
            # 管子固连不动 → 锁定一次后不再重找(省掉每帧一次 find_blobs ≈7.5ms)
            # ⚠️ 预热期强制每帧重找: relock_every=0 意味着"第1帧锁死",
            # 而第1帧恰是 AE/AWB 最没收敛、最不可信的一帧 —— 错的管子矩形
            # 会让原点算错并被**永久**缓存。预热期不锁, 让它跟着画面收敛。
            need_tube = (tube is None
                         or warming
                         or not lock_tube
                         or (relock_every > 0 and n_frame % relock_every == 0))
            if need_tube:
                found = find_tube(img, tube_thr, tube_px_min, min_aspect,
                                  min_fill, band)
                if found is not None:
                    tube = TubeRect(found)
                    n_relock += 1
                    # 预热结束后第一次锁定时打印标定常数(固定标尺模式的输入)。
                    # 标定的 offset/scale 绑定于某一次锁定; 光照变化会影响
                    # 阈值边界像素归属从而改变框宽, 因此打印实测值供核对。
                    if not warming and not printed_geom:
                        printed_geom = True
                        margin_l = tube.x
                        margin_r = img.width() - (tube.x + tube.w)
                        print("  [标定常数] tube x=%d w=%d  原点px=%.1f  "
                              "px_per_cm=%.3f"
                              % (tube.x, tube.w, origin_px(tube),
                                 tube.w / tube_len_cm))
                        # 边距过小是静默失效的预警: 导轨一端出画面会
                        # 导致标尺错误且无报错。
                        print("  [画面边距] 左 %d px  右 %d px  (管占画面 %.0f%%)"
                              % (margin_l, margin_r,
                                 100.0 * tube.w / img.width()))
                        if margin_l < 25 or margin_r < 25:
                            print("     ⚠️ 边距太小! 相机再偏一点管子就出画面 → "
                                  "标尺静默出错。建议相机后退,占画面85~90%")
                    # 几何体检: 只警告, 不自动修正 —— 输出量必须可预测。
                    if fg_on and not fg_checked and not warming and fg_tol > 0:
                        fg_checked = True
                        meas = tube.w / tube_len_cm
                        dev = abs(meas - fg_ppcm) / fg_ppcm
                        if dev > fg_tol:
                            print("  ⚠️⚠️ 几何体检不过: 实测 px_per_cm %.3f vs "
                                  "写死 %.3f (差 %.1f%%)"
                                  % (meas, fg_ppcm, dev * 100))
                            print("     → 相机相对摆杆可能被碰移位了, "
                                  "offset/scale 标定已失效, 要重标!")
                        else:
                            print("  几何体检通过: 实测 px_per_cm %.3f vs 写死 %.3f "
                                  "(差 %.1f%%)" % (meas, fg_ppcm, dev * 100))
                elif not lock_tube:
                    tube = None
            ball = None
            diag = {}
            if tube is not None and not warming:
                ball = find_ball(img, tube, ball_thr, spec_l_min,
                                 ball_px_min, spec_px_min, inset_frac, diag,
                                 inset_x_frac, spec_inside, ball_max_aspect)
                # 无高光的候选可能是阴影干扰, 按 config 决定是否丢弃;
                # 单独计数以区分"真没球"与"有候选但不可信"。
                if ball is not None and require_spec and not ball[4]:
                    n_nospec += 1
                    # 记录被拒块的最大 L(取最大值: 判断"最亮时能否达到阈值")。
                    lv = diag.get("l_max")
                    if lv is not None:
                        if nospec_l_max is None or lv > nospec_l_max:
                            nospec_l_max = lv
                        nospec_l_sum += lv
                        nospec_l_n += 1
                    # 宽高比: 拒掉的是细长条(管壁)还是近方形(球)? 处置相反, 要分清
                    asp = diag.get("aspect")
                    if asp is not None:
                        if asp >= 2.5:
                            n_rej_wall += 1     # 细长 → 大概是管壁/端部暗带
                        else:
                            n_rej_round += 1    # 近方 → 可能是真球被误拒!
                    ball = None
            t_detect_sum += time.ticks_diff(time.ticks_us(), t0) / 1000.0

            x_cm = None
            if tube is not None:
                img.draw_rectangle(tube.x, tube.y, tube.w, tube.h,
                                   color=(255, 255, 0), thickness=1)
                # 紫色 0 刻度线: 球的目标位置。和串口发的 X 同一个原点。
                # 方案① 开启时用写死的原点 —— 屏上的线和串口的数必须同源,
                # 否则会出现"球压在紫线上而串口报 X=+2"这类不一致。
                ox = fg_origin if fg_on else origin_px(tube)
                tube_cy = tube.y + tube.h // 2
                # 标尺: 写死时用常数, 否则从这次锁定的管子量。目标线要用它把 cm 换成像素
                ppcm_now = fg_ppcm if fg_on else (tube.w / float(tube_len_cm))
                # 紫线画在目标位置, 可用于目视核对目标设定是否正确。
                tx_px = ox + (-target_now if invert_x else target_now) * ppcm_now
                if ov_on:
                    draw_zero_line(img, tx_px, tube_cy, ov_zero_half,
                                   thickness=ov_zero_w)
                if ball is not None:
                    bcx, bcy, bw, bh, has_spec, bpx = ball
                    n_hit += 1
                    if has_spec:
                        n_spec += 1
                    # 周期性测量已接受球体的高光 L(get_statistics 有开销,
                    # 不必每帧执行), 与被拒块的 L 对比可确定阈值位置。
                    if n_frame % 15 == 0:
                        lv_ok = None
                        try:
                            st = img.get_statistics(
                                roi=(int(bcx - bw / 2), int(bcy - bh / 2),
                                     max(int(bw), 1), max(int(bh), 1)))
                            lv_ok = st.l_max()
                        except Exception:
                            pass
                        if lv_ok is not None:
                            if ok_l_min is None or lv_ok < ok_l_min:
                                ok_l_min = lv_ok
                            ok_l_sum += lv_ok
                            ok_l_n += 1
                    n_lost = 0
                    if fg_on:
                        px_per_cm = fg_ppcm
                        x_raw = px_to_cm_fixed(bcx, fg_origin, fg_ppcm,
                                               invert_x, offset_cm,
                                               scale_correction)
                    else:
                        x_raw, px_per_cm = px_to_cm(bcx, tube, tube_len_cm,
                                                    invert_x, offset_cm,
                                                    scale_correction)
                    # raw 统计全量收集: raw 峰峰值远大于滤波后说明闸门在起作用
                    if xr_min is None or x_raw < xr_min:
                        xr_min = x_raw
                    if xr_max is None or x_raw > xr_max:
                        xr_max = x_raw
                    # 闸门: 拦截物理上不可能的跳变, 不进入滤波窗口
                    if gate.check(x_raw):
                        # 滤波后的 x_cm 为实际输出量
                        x_cm = medf.push(x_raw)
                        x_sum += x_cm
                        x_n += 1
                        if x_min is None or x_cm < x_min:
                            x_min = x_cm
                        if x_max is None or x_cm > x_max:
                            x_max = x_cm
                        # 误差只在此处计算, 不可上移到闸门/滤波之前:
                        # 闸门与中值滤波必须处理绝对位置 —— 目标值变化是人为瞬变,
                        # 而闸门应判断的是球的物理运动, 两者必须分开。
                        err_cm = x_cm - target_now
                        # 箭头链: 用滤波后的 x_cm 反推起点(不用 bcx),
                        # 保证屏上箭头与串口输出严格一致。
                        if ov_on:
                            bx_shown = ox + x_cm * px_per_cm
                            draw_arrow_chain(img, bx_shown, tx_px, tube_cy,
                                             spacing=ov_gap, size=ov_size,
                                             thickness=ov_thick)
                        img.draw_cross(int(bcx), int(bcy), color=(255, 0, 0),
                                       size=10, thickness=2)
                        # 发送误差: 负 = 球在目标左侧该往右赶, 正 = 反之。
                        # 返回值必须检查 —— 丢弃它等于把"调用了"当成"发出去了",
                        # 而 TX 计数一旦包含失败的次数, 就无法用于判断链路是否通。
                        if ser is not None:
                            if ser.send_ball(err_cm, bpx):
                                n_tx += 1
                            else:
                                n_tx_fail += 1
                        # 同时显示目标/实测/误差, 便于定位问题环节
                        tag = "T%+.1f  X%+.2f  E%+.2f" % (target_now, x_cm, err_cm)
                        if not has_spec:
                            tag += " ?"      # 无高光 → 可信度低, 提示一下
                        img.draw_string_advanced(10, 10, 28, tag,
                                                 color=(0, 255, 255))
                        # 方向提示: 与箭头同源, 用于核对协议符号
                        if err_cm > 0.1:
                            hint = "PUSH LEFT"
                        elif err_cm < -0.1:
                            hint = "PUSH RIGHT"
                        else:
                            hint = "ON TARGET"
                        img.draw_string_advanced(10, 74, 18, hint,
                                                 color=(190, 90, 230))
                    else:
                        # 被闸门拦下: 画面标出来, 但不输出、不进滤波窗口
                        img.draw_cross(int(bcx), int(bcy), color=(255, 0, 255),
                                       size=10, thickness=1)
                        img.draw_string_advanced(10, 10, 26,
                                                "JUMP %+.1f REJECTED" % x_raw,
                                                color=(255, 0, 255))
                    img.draw_string_advanced(10, 44, 18,
                                             "%.1f px/cm" % px_per_cm,
                                             color=(160, 160, 160))
                else:
                    # 丢球: 短暂丢帧不动滤波窗口; 长时间丢失则清空,
                    # 避免输出陈旧位置
                    n_lost += 1
                    if lost_reset > 0 and n_lost == lost_reset:
                        medf.reset()
                        gate.reset()      # 闸门锚点同样过时, 一并清空
                    img.draw_string_advanced(10, 10, 26, "NO BALL",
                                             color=(255, 160, 0))
            else:
                img.draw_string_advanced(10, 10, 26, "NO TUBE",
                                         color=(255, 0, 0))

            if warming:
                # 屏上显示预热进度, 避免误判为程序无响应
                img.draw_string_advanced(10, 10, 26,
                                         "WARMUP %d/%d" % (n_frame, warmup_frames),
                                         color=(255, 160, 0))

            # ===== 序列推进 =====
            # 使用滤波后的绝对位置 x_cm(丢帧/被闸门拦截时为 None),
            # 不使用 x_raw(可能含错读)也不使用误差(序列判断的是绝对位置)。
            if mode == MODE_SEQ:
                arrived = seq.update(x_cm)
                if arrived is not None:
                    if seq.done:
                        print("  [状态①] 到达 %+.1fcm —— 序列走完, "
                              "目标锁定在此点" % arrived)
                    else:
                        print("  [状态①] 到达 %+.1fcm → 折返, 下一目标 %+.1fcm"
                              % (arrived, seq.target()))
                elif seq.stuck() and not seq.warned:
                    # 只警告一次, 不自动推进 —— 宁可停住提示, 也不假装到达。
                    seq.warned = True
                    print("  ⚠️ [状态①] 卡在目标 %+.1fcm 超过 %.0f 秒未到位"
                          % (seq.target(), seq.timeout_ms / 1000.0))
                    print("     查: 球是否卡住/电控是否在追这个目标/"
                          "是否大量帧被闸门拦(看下面 拦: 那两个数)")

            # 触摸轮询: 预热期也响应(可用于预设目标);
            # 置于检测之后、上屏之前, 使本帧的触摸立即反映到显示。
            ui.poll(n_frame)

            # ===== 阈值调参改动同步 =====
            # Tuner 只置 dirty 标志, 参数的实际更新集中在此处,
            # 保证同一帧内检测/打印/显示使用同一组参数。
            if tuner is not None and tuner.dirty:
                tuner.dirty = False
                spec_l_min = int(tuner.val["spec_l_min"])
                spec_px_min = int(tuner.val["spec_pixels_min"])
                ball_max_aspect = float(tuner.val["ball_max_aspect"])
                # ball 阈值是 6 元 tuple(不可变) → 换 L 上限要整条重建
                ball_thr = (ball_thr[0], int(tuner.val["ball_l_max"]),
                            ball_thr[2], ball_thr[3], ball_thr[4], ball_thr[5])
                # 阈值变更后清空统计量: 旧统计与新判据混用会误导调参判断。
                xr_min = xr_max = x_min = x_max = None
                x_sum = 0.0
                x_n = 0
                nospec_l_max = None
                nospec_l_sum = 0.0
                nospec_l_n = 0
                n_rej_wall = 0
                n_rej_round = 0
                ok_l_min = None
                ok_l_sum = 0.0
                ok_l_n = 0
                # 闸门锚点不清: 阈值变化不代表球位置变化;
                # 若新阈值导致识别偏移, 闸门会自行拦截。

            # ===== [MODE] 状态切换 =====
            # 切换动作集中在此处, poll 只置标志。
            if ui.mode_toggle:
                ui.mode_toggle = False
                n_mode_sw += 1
                if mode == MODE_HOLD:
                    mode = MODE_SEQ
                    seq.start()
                    print("  [MODE] → 状态① SEQ: 目标自动走 %s"
                          % (" → ".join("%+.1f" % p for p in seq.points),))
                else:
                    mode = MODE_HOLD
                    seq.stop()
                    print("  [MODE] → 状态② HOLD: 目标回到触摸设定值 %+.2fcm"
                          % ui.target)
                # 目标跳变不清闸门锚点: 锚点记录绝对位置, 球未移动则锚点仍有效。

            # ===== [RESET] 一键重置(触摸或电控都汇到这里) =====
            if ui.reset_req:
                ui.reset_req = False
                n_reset += 1
                # 顺序: 先停序列并回到 HOLD, 再清缓存几何与历史, 最后重新预热。
                mode = MODE_HOLD
                seq.stop()
                tube = None            # 重找绿管 → 原点与 px_per_cm 全部重算
                printed_geom = False   # 让 [标定常数] 那行重新打一次, 方便核对新值
                fg_checked = False     # 几何体检也重做(相机可能刚被碰过)
                medf.reset()
                gate.reset()
                n_lost = 0
                warm_from = n_frame    # 重新预热: 等 AE/AWB 收敛再信画面
                x_cm = None
                # 抖动统计也清: 留着旧极值会让重置后的报表永远显示重置前那个大峰峰值,
                # 看不出重置有没有改善(报表要能反映当前状态, 否则等于没有报表)。
                xr_min = xr_max = x_min = x_max = None
                x_sum = 0.0
                x_n = 0
                print("  [RESET] 已重置: 目标归零 / 重找绿管+重算标尺 / "
                      "清滤波与闸门 / 停序列回状态② / 重新预热 %d 帧" % warmup_frames)
                print("          (统计也已清零 —— 报表反映的是重置之后的状态)")

            ui.mode = mode          # 同步给 UI 只为显示, UI 不持有状态
            ui.draw(img)

            # 调参页由主循环绘制: 需要实测 L 统计数据。
            # 那两个数是本页核心 —— 据"球 L"与"被拒块 L"把门限卡在中间。
            if tuner is not None and tuner.on and ui.ok:
                tuner.draw_page(img,
                                ok_l_min, nospec_l_max,
                                n_rej_wall, n_rej_round)

            # 状态① 的进度条一行: 走到第几段、还差几帧算稳住。
            # 画在左侧中部(避开上面的 T/X/E 和下面那排按钮)。
            # 卡住时转红字, 明确显示"在等待而非在运动"。
            if mode == MODE_SEQ:
                if seq.stuck():
                    scol = (255, 60, 60)
                elif seq.done:
                    scol = (120, 255, 160)
                else:
                    scol = (255, 190, 60)
                img.draw_string_advanced(10, 100, 22, seq.status(), color=scol)
                if seq.stuck():
                    img.draw_string_advanced(10, 126, 20, "STUCK - not advancing",
                                             color=(255, 60, 60))

            if show_fps:
                el = time.ticks_diff(time.ticks_ms(), t_start)
                fps = n_frame * 1000.0 / el if el > 0 else 0.0
                img.draw_string_advanced(10, img.height() - 30, 18,
                                         "%.1f fps" % fps, color=(255, 255, 0))
                # 图传开启时在屏角显示 IP, 无需连接电脑查询
                if vs is not None and vs.is_on() and vs.ip():
                    img.draw_string_advanced(img.width() - 210, img.height() - 30,
                                             18, "RTSP %s" % vs.ip(),
                                             color=(0, 255, 128))

            if disp:
                disp.show(img)

            # 图传: 放在 disp.show 之后 → 传出去的画面**带全部叠加层**(黄框/
            # 紫线/箭头/坐标数字), 回传画面自带数据标注。
            if vs is not None:
                # 屏上 [STREAM] 键的开关请求在这里处理, 不在 ui.poll 里 ——
                # 起网要 2 秒、关流要关 socket, 放 poll 里等于把主循环卡 2 秒
                # (球在动而我们瞎了)。这里已经出完这一帧, 代价最小。
                if ui.stream_toggle:
                    ui.stream_toggle = False
                    if vs.is_on():
                        vs.close()
                        print("  [图传] 已关闭 → 全部算力给视觉")
                    else:
                        print("  [图传] 启动中(约2秒, 这期间会掉几帧)...")
                        vs.start()
                    # 清掉滤波和闸门的锚点: 刚才卡的那 2 秒里球可能被移动过,
                    # 用旧锚点判跳变会把之后正常的读数全拦掉。
                    medf.reset()
                    gate.reset()
                ui.stream_on = vs.is_on()
                vs.offer(img)
            elif ui.stream_toggle:
                # 模块没加载(config.video_stream.enabled=false)却点了按钮 →
                # 提供明确的点击反馈。
                ui.stream_toggle = False
                print("  [图传] 模块未加载: config.video_stream.enabled=false")
                print("        要用图传得改 config 重新部署, 按钮改不了这个")

            # 每 2 秒往终端汇报一次(别每帧打, 会拖慢)
            now = time.ticks_ms()
            if time.ticks_diff(now, last_report) >= 2000:
                # 瞬时 FPS(增量算) 比累计平均有意义 —— 累计值会被启动阶段拖低
                d_f = n_frame - last_fps_frame
                d_ms = time.ticks_diff(now, last_fps_ms)
                fps_now = d_f * 1000.0 / d_ms if d_ms > 0 else 0.0
                if x_n > 1:
                    jit = "%.2f→%.2f" % (xr_max - xr_min, x_max - x_min)
                else:
                    jit = "--"
                # 有效% 的分母要扣掉预热帧, 否则预热被算成"丢球"、白白拉低指标
                n_eff = n_frame - warmup_frames
                if n_eff < 1:
                    n_eff = 1
                # 图传状态: 只读计数器, 不做任何可能失败的调用
                vs_tag = ""
                if vs is not None and vs.is_on():
                    try:
                        n_en, n_st, n_se, n_cl = vs.stats_full()
                        vs_tag = "  图传%d帧/%d连接" % (n_st, n_cl)
                        if n_se:
                            vs_tag += "(丢%d)" % n_se
                    except Exception:
                        pass
                # 状态标签: 报表第一眼就能看出当前跑的是哪个状态
                mtag = MODE_NAMES[mode]
                if mode == MODE_SEQ:
                    mtag += "(%s)" % seq.status()[4:]     # 去掉重复的 "SEQ "
                # 发送失败只在真发生时才显示 —— 平时为 0, 出现即说明串口有问题
                tx_tag = "%d" % n_tx
                if n_tx_fail:
                    tx_tag += "(失败%d)" % n_tx_fail
                print("  帧 %d  FPS %.1f(瞬时)  检测 %.1f ms  有效 %.0f%%  "
                      "[%s] 目标%+.1f X=%s 误差=%s  抖动 %s cm  "
                      "拦:无高光%d/跳变%d  TX %s%s"
                      % (n_frame, fps_now, t_detect_sum / n_frame,
                         100.0 * n_hit / n_eff, mtag, target_now,
                         ("%+.2f" % x_cm) if x_cm is not None else "--",
                         ("%+.2f" % (x_cm - target_now)) if x_cm is not None else "--",
                         jit, n_nospec, gate.n_blocked, tx_tag, vs_tag))
                # 阈值诊断: 把球与被拒块的实测 L 并排打出, 门限该卡在两者中间。
                # 只在真被拒过时才打, 平时不刷屏。
                if nospec_l_n or ok_l_n:
                    s_ok = ("球 L 最低 %.0f/均 %.0f" % (ok_l_min, ok_l_sum / ok_l_n)
                            if ok_l_n else "球 L 无样本(一帧都没认出)")
                    s_no = ("被拒块 L 最亮 %.0f/均 %.0f"
                            % (nospec_l_max, nospec_l_sum / nospec_l_n)
                            if nospec_l_n else "无被拒块")
                    # 标出被调参页修改过的门限, 避免误引 config.json 旧值。
                    tune_tag = ""
                    if tuner is not None and tuner.n_change:
                        chg = [it[1] for it in TUNE_ITEMS
                               if tuner.val[it[0]] != tuner.base[it[0]]]
                        if chg:
                            tune_tag = "  ⚠️[TUNE]已改: %s" % ",".join(chg)
                    print("     [阈值诊断] %s | %s | 门限=%d%s"
                          % (s_ok, s_no, spec_l_min, tune_tag))
                    if nospec_l_n:
                        # ⚠️ 判"是不是真球被误拒"必须**形状和 L 一起看**, 只看形状会喊狼:
                        # 切口阴影也可以是近方形的, 但它 L≈17(球 L 是 70+),
                        # 那类拒绝是正确行为。
                        if nospec_l_max is not None and nospec_l_max < 30:
                            verdict = "L≤%.0f=切口阴影级, **全部拒得对**" % nospec_l_max
                        elif n_rej_round:
                            verdict = ("近方形%d 且 L 到 %.0f → **可能是真球被误拒**"
                                       % (n_rej_round, nospec_l_max))
                        else:
                            verdict = "全是细长条(管壁/端部), 拒得对"
                        print("        被拒块: 细长%d / 近方形%d —— %s"
                              % (n_rej_wall, n_rej_round, verdict))
                # 球静止时输出峰峰值应 <0.3cm; 达到 cm 级则可能正在输出误读。
                # 但峰峰值大不等于有误读 —— 球在移动时它就是行程。
                # 因此此处不下结论, 只列出判据供判断:
                #   跳变拦截为 0 = 帧间连续 = 球确实在移动(误读会触发大量拦截)
                #   球 L 与被拒块 L 分离 = 识别的是球而非管壁
                if x_n > 5 and x_min is not None and (x_max - x_min) > 2.0:
                    pp = x_max - x_min
                    if gate.n_blocked == 0 and (nospec_l_max is None
                                                or nospec_l_max < 30):
                        print("  ℹ️ 峰峰值 %.1fcm —— 但跳变拦截 0 且被拒块只有阴影级 L,"
                              " 这是**球在移动的行程**, 不是噪声。" % pp)
                        print("     要测静态噪声: **先摆好球再启动程序**(或按 RESET 重置统计)")
                    else:
                        print("  ⛔ 峰峰值 %.1fcm 且 跳变拦截%d/无高光%d —— "
                              "可能**真有错读在发出去**" % (pp, gate.n_blocked, n_nospec))
                        print("     看屏上红十字是否粘在管壁/管端上; 球静止时应 <0.3cm")
                    # 三种情形给三种相反的处置 —— 这才是这行诊断的价值
                    if ok_l_n and nospec_l_n and ok_l_min > nospec_l_max:
                        mid = int((ok_l_min + nospec_l_max) / 2)
                        print("        → 两者**已分开**(球最低%.0f > 被拒最亮%.0f), "
                              "门限设 %d 最稳" % (ok_l_min, nospec_l_max, mid))
                    elif ok_l_n and nospec_l_n:
                        print("        → ⚠️两者**重叠**, 光靠 L 分不开 —— "
                              "别再调 L 了, 去装遮光罩/补光让球的高光亮回来")
                    elif not ok_l_n and n_rej_round:
                        print("        → ⚠️一帧都没认出球, 且被拒的是近方形 → "
                              "**门限太严**, 把 spec_l_min 降到 %d 试试"
                              % max(int(nospec_l_max) - 5, 25))
                last_report = now
                last_fps_frame = n_frame
                last_fps_ms = now

    except KeyboardInterrupt:
        print("\n[BallPos] 用户停止")
    except Exception as _e:
        # 先打印异常原因再进入 finally: 若 finally 中的 close 阻塞,
        # 异常将无法到达引导层的 except, 日志中会缺失退出原因。
        print("\n" + "!" * 58)
        print("[BallPos] ⛔ 主循环异常退出: %r" % (_e,))
        print("         第 %d 帧, 运行 %.1f 秒"
              % (n_frame, time.ticks_diff(time.ticks_ms(), t_start) / 1000.0))
        # 不要直接调用 print_exception(_e): 不带 file 参数时它会写真实 stdout,
        # 脱机模式下该通道无人读取会阻塞。先捕获为字符串再通过 print 输出。
        try:
            import sys as _sys
            import io as _io
            _buf = _io.StringIO()
            _sys.print_exception(_e, _buf)
            for _ln in _buf.getvalue().split("\n"):
                if _ln.strip():
                    print("    " + _ln)
        except Exception as _e2:
            # StringIO 不可用时至少把类型名留下, 别把这里变成新的失败点
            print("    (traceback 抓取失败: %r; 异常本身: %r)" % (_e2, _e))
        print("!" * 58)
        raise
    finally:
        el = time.ticks_diff(time.ticks_ms(), t_start)
        if n_frame > 0 and el > 0:
            print("-" * 58)
            print("  总帧数 %d   平均 FPS %.1f   检测平均 %.1f ms/帧"
                  % (n_frame, n_frame * 1000.0 / el, t_detect_sum / n_frame))
            # 预热帧不参与"有效率"的分母(它们本来就不该出球)
            n_eff = max(n_frame - warmup_frames, 1)
            print("  确认球 %d 帧 (%.0f%%), 其中含高光 %d 帧 (%.0f%%)"
                  % (n_hit, 100.0 * n_hit / n_eff, n_spec,
                     100.0 * n_spec / max(n_hit, 1)))
            print("  ── 两道防错读闸门的战果 ──")
            print("  无高光被拒 %d 帧 (%.0f%%)%s"
                  % (n_nospec, 100.0 * n_nospec / n_eff,
                     "" if require_spec else "  ← require_spec=false, 未启用"))
            if nospec_l_n or ok_l_n:
                print("  ── 阈值诊断(定 spec_l_min 就看这几行) ──")
                if ok_l_n:
                    print("  被**接受**的球: 高光 L 最低 %.0f / 平均 %.0f (%d 个样本)"
                          % (ok_l_min, ok_l_sum / ok_l_n, ok_l_n))
                else:
                    print("  被**接受**的球: 无样本 —— 一帧都没认出来")
                if nospec_l_n:
                    print("  被**拒绝**的块: 高光 L 最亮 %.0f / 平均 %.0f"
                          % (nospec_l_max, nospec_l_sum / nospec_l_n))
                    print("    形状: 细长 %d 帧(管壁/端部暗带, 拒得对) / "
                          "近方形 %d 帧(**可能是真球被误拒**)"
                          % (n_rej_wall, n_rej_round))
                print("  当前门限 spec_l_min=%d" % spec_l_min)
                if tuner is not None and tuner.n_change:
                    # 调出的值可抄回 config.json —— 仅存于 tune.txt 时,
                    # 换板或重烧固件即丢失。故此处直接打印成可抄录的形式。
                    print("  ── [TUNE] 本次运行共改动 %d 次, 结果如下 ──" % tuner.n_change)
                    for attr, label, _lo, _hi, _st, _d in TUNE_ITEMS:
                        cur = tuner.val[attr]
                        bas = tuner.base[attr]
                        print("    %-16s %s%s"
                              % (attr, tuner._fmt(attr),
                                 "" if cur == bas
                                 else "   (config 原值 %s ← **已改**)"
                                      % tuner._fmt(attr, bas)))
                    print("  调定后请抄回 config.json (ball_l_max 对应 ball 的第 2 个数),")
                    print("     只存在 %s 里的话, 换板或重烧固件就丢了。" % TUNE_FILE)
                if ok_l_n and nospec_l_n:
                    if ok_l_min > nospec_l_max:
                        print("  ✅ 两者已分开 → 门限设 %d 最稳(取中点)"
                              % int((ok_l_min + nospec_l_max) / 2))
                    else:
                        print("  ⚠️ 两者重叠 → 仅靠 L 无法区分, 请勿继续调该阈值;")
                        print("     去装遮光罩/补光, 让球的高光亮回来 = 唯一正解。")
            print("  跳变被拦 %d 帧 (门限 %.1f cm)" % (gate.n_blocked, max_jump_cm))
            if ser is not None:
                print("  串口发出 %d 包 (一帧一包, 被两道闸门拦下的帧不发送)" % n_tx)
                if n_tx_fail:
                    print("  ⚠️ write 失败 %d 次 —— 串口链路有问题, 查接线与引脚配置"
                          % n_tx_fail)
            if gate.n_blocked > 0 or n_nospec > 0:
                print("    ↑ 拦下的是会使电控反向控制的误读, 计数大说明它们确实存在。")
                print("      若无高光比例 >40%, 说明照明条件不足, 需用 [TUNE] 调门限。")
            print("  摆杆锁定 %d 次%s"
                  % (n_relock, "(锁死模式)" if lock_tube and relock_every == 0
                     else ""))
            print("  ── 状态机 ──")
            print("  结束时状态: %s   切换 %d 次   重置 %d 次"
                  % (MODE_NAMES[mode], n_mode_sw, n_reset))
            if seq.running or seq.n_arrive:
                print("  状态① 序列: 到达 %d/%d 个点%s"
                      % (seq.n_arrive, len(seq.points),
                         "  ✅走完" if seq.done else
                         ("  ⚠️卡在 %+.1fcm" % seq.target() if seq.stuck()
                          else "  (未走完)")))
            if x_n > 1:
                pp_raw = xr_max - xr_min
                pp_med = x_max - x_min
                print("  X 统计: 均值 %+.2f  范围 %+.2f..%+.2f" % (x_sum / x_n, x_min, x_max))
                print("  峰峰值: 闸门前 %.2f cm  →  输出 %.2f cm (中值 N=%d)"
                      % (pp_raw, pp_med, med_n))
                print("  ↑ 只看**输出**那个数。球静止时它就是测量噪声,")
                print("    建议静态噪声 <0.3cm, 为机械与控制误差留余量。")
                if pp_raw > 3.0:
                    print("  ⚠️ 闸门前 %.1f cm 说明确实发生过错读(球在管里跑不了这么远),"
                          % pp_raw)
                    print("     两道闸门把它们挡在输出之外了 —— 闸门前的数大是正常的。")
                print("  ⚠️ 这个统计只在球静止时才是噪声; 球在滚时它是行程, 别当噪声看。")
            print("  → 传感器上限约 33fps。检测 <15ms 说明延迟已不是瓶颈;")
            print("    球杆闭环里检测延迟=相位滞后, 比帧率数字更影响稳定性。")
            print("-" * 58)
        if ser is not None:
            ser.close()      # 不关的话下次跑会占着引脚打不开
        if disp:
            disp.close()
        cam.close()
        print("[BallPos] Done.")


if __name__ == "__main__":
    # 直接在 IDE 里运行本文件时的自保护: IDE 的工作目录不是框架目录,
    # 会导致 load_config("config.json") 报 ENOENT、import utils 报 ImportError。
    # 推荐入口是 run_ball_pos.py, 此处做兜底处理。
    try:
        import os
        import sys
        _d = "/sdcard/k230_vision"
        if os.getcwd() != _d:
            os.chdir(_d)
        if _d not in sys.path:
            sys.path.insert(0, _d)
    except Exception:
        pass
    main()
