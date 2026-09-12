# -*- coding: utf-8 -*-
"""
开机自启安装器 —— 在 CanMV IDE 中运行本文件, 使板子上电后自动运行视觉主程序。

原理: 向 /sdcard/main.py 写入一段引导代码(固件上电会执行该文件)。
再次运行本文件时, 会先输出上一次运行的日志, 可用于排查自启问题。

关键设计 —— stdout 接管:
    自启模式下没有 IDE 连接, stdout 无人读取; 向无读者的管道写入,
    缓冲区填满后 write 会永久阻塞(表现为"运行十几秒后卡住")。
    因此引导代码在 import 框架之前将 print 替换为只写内存的版本:
      ① 逐模块注入 (本固件无法修改 builtins.print)
      ② PREIMPORT 提前导入延迟加载的模块
      ③ _sweep 持续发现并注入新模块
    运行期输出保留在内存(最近 RING_LINES 行), 异常退出时落盘;
    另有约 30 秒一次的心跳写入日志。

停用方式(从软到硬):
    · 上电后 3 秒内让 IDE 连接并停止, 或串口发送 Ctrl-C
    · 拔出 TF 卡在电脑上创建空文件 noboot  → 下次开机直接进 REPL
    · 拔出 TF 卡删除 main.py                → 彻底停用
    · 连续 3 次开机未跑满 12 秒时, 引导代码自动创建 noboot 停用自启
"""

import os
import sys
import time

# ---- 板上路径与行为参数。改这里再重新跑本文件(引导代码是照这些常量生成的) ----
TARGET       = '/sdcard/main.py'            # 固件上电执行的文件, 不可改名
DIR          = '/sdcard/k230_vision'        # 框架目录
MODULE       = 'ball_pos'                   # 要跑的模块(在 DIR/recipes/ 下)
FUNC         = 'main'
LOG          = '/sdcard/autostart.log'      # 本次开机日志; 上一份自动存成 .prev
NOBOOT       = '/sdcard/noboot'             # 这个文件存在就跳过自启
STRIKES      = '/sdcard/autostart.strikes'  # 连续没跑起来的次数
# 自启运行中的标志。display.py 读它来判断"现在没有 IDE 在读" → 不退回 VIRT
# (VIRT 只往 IDE 推帧, 脱机时必然堵住主循环)。两边这个路径必须一致。
RUNFLAG      = '/sdcard/.autostart_running'
GRACE_S      = 3                            # 逃生窗口秒数(这期间 Ctrl-C 能掐掉自启)
MAX_STRIKES  = 3                            # 连续失败这么多次就自动停用
HEALTHY_MS   = 12000                        # 跑满这么久算"起来了", 失败计数清零
HEARTBEAT_MS = 30000                        # 心跳落盘间隔
# 运行期输出在内存保留的行数(异常退出时落盘), 用于脱机排查。
RING_LINES   = 200

# ⭐这几个模块必须**提前** import。原因: 本固件禁止改 builtins.print(镜像里
# builtins_override 符号数为 0 = MICROPY_CAN_OVERRIDE_BUILTINS 关), 接管 print
# 只能逐模块注入; 而框架把这些模块写成在 main() 里延迟 import(ball_pos.py
# L1341-1378), 那时注入已经做完 → 它们会绕过接管、直接写真 stdout 而卡死。
# 提前拉进来, 注入才盖得住。这些模块体内只有 import 与 class/def, 无副作用。
PREIMPORT = ["camera", "display", "serial_comm", "image_utils", "video_stream"]

_MARK = '开机自启引导'                       # 用来认出 main.py 是不是本文件写的

# 把 ACTION 改成 "off" 再运行一次 = 停用自启(板上没法输入按键, 所以做成常量开关)。
ACTION = "on"


# ============================================================
# 写进 /sdcard/main.py 的引导代码。@@X@@ 由下面 _render() 替换成上面的常量。
# 用 replace 而不是 % 格式化, 因为这段代码里自己带着 %d/%s。
# ============================================================
BOOT = '''# -*- coding: utf-8 -*-
# /sdcard/main.py —— 开机自启引导。由 install_autostart.py 生成, 别手改这个文件。
#
# 停用(从软到硬): ①上电 3 秒内 Ctrl-C  ②在 /sdcard/ 建空文件 noboot
#                 ③删掉本文件  ④把 install_autostart.py 的 ACTION 改 "off" 跑一次
#
# ⭐这里为什么几乎不 print: 自启时 stdout **没有人读**(IDE 没连、串口没人收),
#   缓冲写满之后那一次 write 就永久阻塞 —— 这正是"上电跑十几秒后定住, 而同一份
#   代码在 IDE 里手动跑却一切正常"的原因(IDE 会一直把 stdout 抽走)。
#   所以: import 框架**之前**先把 print 换成只写内存的版本, 定期/出错时才落盘。
#   代价是终端只有开头那几行, 这是对的, 不是没跑 —— 进度看日志文件。

import os
import sys
import time

DIR          = "@@DIR@@"
MODULE       = "@@MODULE@@"
FUNC         = "@@FUNC@@"
LOG          = "@@LOG@@"
NOBOOT       = "@@NOBOOT@@"
STRIKES      = "@@STRIKES@@"
RUNFLAG      = "@@RUNFLAG@@"
GRACE_S      = @@GRACE_S@@
MAX_STRIKES  = @@MAX_STRIKES@@
HEALTHY_MS   = @@HEALTHY_MS@@
HEARTBEAT_MS = @@HEARTBEAT_MS@@
RING_LINES   = @@RING_LINES@@
# 要提前 import 的模块。⭐为什么需要这个: 本固件**禁止**改 builtins.print
# (镜像里 builtins_override 符号数为 0), 所以只能逐模块注入; 而框架把这几个
# 模块写成在 main() 里**延迟** import —— 那时注入早就做完了, 它们会漏到真
# stdout 上。提前 import 好, 注入才盖得住。这几个模块体内只有 import 和
# class/def, 没有副作用, 提前导入是安全的(已逐个看过)。
# 加了新的会 print 的模块就往这里补一个名字。漏了也有下面 _sweep() 兜。
PREIMPORT    = @@PREIMPORT@@

_rp = print                 # 真 print 的引用, 换掉之后还想往终端写就用它
_t0 = time.ticks_ms()

_pend = []                  # 待落盘的行
_dropped = 0                # 因超过 RING_LINES 被丢掉的行数
_last_flush = _t0
_healthy = False            # 是否已判定"起来了"(见 HEALTHY_MS)


def _ms():
    return time.ticks_diff(time.ticks_ms(), _t0)


def _exists(p):
    try:
        os.stat(p)
        return True
    except OSError:
        return False


def _strikes_read():
    try:
        with open(STRIKES) as f:
            return int(f.read().strip() or "0")
    except Exception:
        return 0


def _strikes_write(n):
    try:
        with open(STRIKES, "w") as f:
            f.write("%d" % n)
    except OSError:
        pass


def _put(line):
    # 只进内存。行数封顶, 超了丢最老的并记账(日志里会写明省略了多少行)。
    global _dropped
    _pend.append("[%7dms] %s" % (_ms(), line))
    while len(_pend) > RING_LINES:
        del _pend[0]
        _dropped += 1


def _flush(tag=None):
    global _dropped, _last_flush
    _last_flush = time.ticks_ms()
    if not _pend and tag is None:
        return
    try:
        f = open(LOG, "a")
    except OSError:
        del _pend[:]        # 存储不可用时丢弃, 避免内存累积
        return
    try:
        if _dropped:
            f.write("... 省略 %d 行(内存只留最近 %d 行) ...\\n"
                    % (_dropped, RING_LINES))
            _dropped = 0
        for ln in _pend:
            f.write(ln)
            f.write("\\n")
        if tag is not None:
            f.write("[%7dms] %s\\n" % (_ms(), tag))
    except Exception:
        pass
    try:
        f.close()           # close 才真正落盘, 断电前没 close 的内容会丢
    except Exception:
        pass
    del _pend[:]


def _print(*a, **kw):
    # 顶掉的 print: 只写内存, 不碰 stdout(为什么见文件头)。
    # 整体包 try —— 它现在被框架每帧调用, 自己绝不能抛; 而 KeyboardInterrupt
    # 属 BaseException, 不会被这里吞掉, Ctrl-C 仍能停。
    global _healthy
    try:
        s = kw.get("sep", " ").join([str(x) for x in a])
    except Exception:
        s = "<print 参数转字符串失败>"
    try:
        _put(s)
        _sweep()        # 有新模块进来就顺手接管(延迟 import 的兜底)
        if not _healthy and _ms() >= HEALTHY_MS:
            # 跑够久了 = 这次开机是成功的, 把失败计数清掉(见 MAX_STRIKES)。
            # 挂在 print 上是因为板上没有安全的定时器回调, 而框架每 2 秒必打一行
            # 报表 —— 换句话说"还在打 print"本身就是活着的证据, 正好当心跳用。
            # 已知局限: 若程序**在这之前**就卡死, 计数不会清 -> 会被算作失败。
            # 这个方向是对的: 宁可把可疑的一次算成失败(代价=多一次手动重启),
            # 也不能把卡死误判成成功(代价=永远拿不回 REPL)。
            _healthy = True
            _put("已连续运行 %d 秒 -> 判定启动成功, 失败计数清零"
                 % (HEALTHY_MS // 1000))
            _strikes_write(0)
        if time.ticks_diff(time.ticks_ms(), _last_flush) >= HEARTBEAT_MS:
            _flush("(心跳: 此刻还活着)")
    except Exception:
        pass


def _swap_print(fn):
    # 两层都换, 因为哪层有效取决于固件, 而我们上电前无法确定是哪种:
    #   ①builtins.print —— 一次覆盖所有模块(**包括之后才 import 进来的**),
    #      但 MicroPython 只在编译开了 CAN_OVERRIDE_BUILTINS 时才允许写。
    #   ②逐模块注入 —— 模块 globals 一定可写, 这是保底层。查找顺序是
    #      局部 -> 模块globals -> builtins, 所以注入②就能盖住 builtins。
    #      局限: 只覆盖**此刻已在** sys.modules 里的模块。
    # 返回 ① 是否成功。①失败时, main() 里延迟 import 的模块(如 video_stream)
    # 仍会往真 stdout 打 —— 那种情形靠 noboot 计数兜底(见 MAX_STRIKES)。
    ok = False
    try:
        import builtins
        builtins.print = fn
        ok = builtins.print is fn       # 有些固件静默忽略赋值, 必须回读确认
    except Exception:
        ok = False
    for _n in list(sys.modules):
        try:
            setattr(sys.modules[_n], "print", fn)   # C 模块会抛, 跳过即可
        except Exception:
            pass
    return ok


_n_seen = 0


def _sweep():
    """把新出现的模块也注入一遍。兜的是"延迟 import 漏网"那一类:
    PREIMPORT 只覆盖我们知道的模块, 而 sys.modules 变长了就说明有新模块进来。
    比长度比逐个 setattr 便宜得多, 所以放在每帧都会走的 _print 里也不心疼。"""
    global _n_seen
    n = len(sys.modules)
    if n != _n_seen:
        _n_seen = n
        for _n2 in list(sys.modules):
            try:
                setattr(sys.modules[_n2], "print", _print)
            except Exception:
                pass


# ---------------- 从这里开始是流程 ----------------

if _exists(NOBOOT):
    _rp("[自启] 检测到 %s -> 跳过自启, 进入 REPL" % NOBOOT)
    _rp("[自启] 想恢复自启: 删掉那个文件")
    raise SystemExit

_n_strike = _strikes_read()
if _n_strike >= MAX_STRIKES:
    # 连着几次开机都没跑起来 -> 自己把自己关掉。否则"卡死->只能断电->又卡死"
    # 会让板子永远回不到 REPL, 那是最糟的状态。
    try:
        with open(NOBOOT, "w") as _f:
            _f.write("自启连续 %d 次未跑满 %d 秒, 已自动停用\\n"
                     % (_n_strike, HEALTHY_MS // 1000))
    except OSError:
        pass
    _strikes_write(0)
    _rp("!" * 46)
    _rp("[自启] 连续 %d 次开机没跑起来 -> 已自动停用, 现在进 REPL" % _n_strike)
    _rp("[自启] 看上次日志: %s (或运行 install_autostart.py, 它会打出来)" % LOG)
    _rp("[自启] 修好后删掉 %s 即恢复" % NOBOOT)
    _rp("!" * 46)
    raise SystemExit

_rp("=" * 46)
_rp("[自启] %s 将在 %d 秒后启动。现在按 Ctrl-C / IDE 停止键可中断" % (MODULE, GRACE_S))
_rp("[自启] 启动后终端**不再输出**(那是故意的, 见 main.py 文件头)")
_rp("[自启] 进度与报错都写在 %s" % LOG)
_rp("=" * 46)
try:
    for _i in range(GRACE_S, 0, -1):
        _rp("[自启] %d ..." % _i)
        for _j in range(5):
            time.sleep_ms(200)      # 切碎了睡, Ctrl-C 响应快一些
except KeyboardInterrupt:
    _strikes_write(0)               # 人主动掐的, 不算失败
    _rp("[自启] 已中断, 进入 REPL")
    raise SystemExit

# 日志轮换: 上一次日志保留为 .prev。
# 捕 Exception 而不只是 OSError: 万一某固件没有 os.rename, AttributeError 会
# 在这里把整个引导打死 —— 而这一步只是"留个备份", 失败了也不该影响启动。
try:
    os.remove(LOG + ".prev")
except Exception:
    pass
try:
    os.rename(LOG, LOG + ".prev")
except Exception:
    pass

_put("==== 开机自启 ====")
_put("目录 %s   模块 %s.%s()" % (DIR, MODULE, FUNC))
_put("本次为连续第 %d 次尝试(跑满 %d 秒即清零; 满 %d 次自动停用)"
     % (_n_strike + 1, HEALTHY_MS // 1000, MAX_STRIKES))
_flush()
_strikes_write(_n_strike + 1)       # 必须在启动**前**记账: 真卡死了就没机会记了

# 立自启标志: display.py 靠它判断"现在没有 IDE", 从而不退回只往 IDE 推帧的
# VIRT 显示(脱机退到 VIRT = 换个姿势撞同一面墙, 见 display.py 的注释)。
# 必须在 import 框架**之前**建 —— Display 是在 main() 里初始化的, 但早建无害。
try:
    with open(RUNFLAG, "w") as _f:
        _f.write("由 main.py 建, 退出时删。残留=上次硬断电\\n")
except OSError:
    pass

_builtin_ok = _swap_print(_print)   # 必须在 import 框架**之前**换
# 本固件预期就是"改不了 builtins"(镜像里 builtins_override 符号数为 0), 所以那
# 不算故障 —— 真正干活的是逐模块注入 + PREIMPORT + _sweep 这三层。打这行只为
# 万一换了固件能一眼看出走的是哪条路。
_put("print 接管: 逐模块注入 已完成; builtins %s"
     % ("也成功(此固件允许改, 覆盖面最好)" if _builtin_ok
        else "改不了(本固件的预期行为, 已由 PREIMPORT+_sweep 兜住)"))

_code = 0
try:
    # chdir 与 sys.path 两步都不能省, 且顺序不能颠倒:
    #   框架里 load_config("config.json") 用的是**相对路径** -> 不 chdir 报 ENOENT;
    #   框架装在 /sdcard/k230_vision/ 而固件的搜索路径只到 /sdcard -> 不进
    #   sys.path 则 import utils.config_loader 失败。
    #   (与手动入口 run_ball_pos.py 的路径一致。)
    os.chdir(DIR)
    if DIR not in sys.path:
        sys.path.insert(0, DIR)
    _r = DIR + "/recipes"           # recipes 没有 __init__.py, 直接进 path 平铺导入
    if _r not in sys.path:
        sys.path.insert(0, _r)
    _put("cwd=%s" % os.getcwd())
    # 先把"框架在 main() 里才 import"的那几个模块拉进来, 好让下面的注入盖住它们
    # (原因见 PREIMPORT 的注释)。单个失败不致命: 框架自己 import 时会再报一次,
    # 那时的报错信息比这里更准, 让它去报。
    _pre_ok, _pre_bad = [], []
    for _m in PREIMPORT:
        try:
            __import__(_m)
            _pre_ok.append(_m)
        except Exception as _pe:
            _pre_bad.append("%s(%r)" % (_m, _pe))
    _mod = __import__(MODULE)
    _swap_print(_print)             # import 带进来的新模块也要接管
    # 给 _sweep 定基线(模块级流程里直接赋值即是全局, 不需要 global 声明)
    _n_seen = len(sys.modules)
    _put("预导入 %d 个: %s" % (len(_pre_ok), " ".join(_pre_ok) or "无"))
    if _pre_bad:
        _put("  预导入失败(不致命, 框架自己 import 时会再报): %s"
             % " ".join(_pre_bad))
    _put("import 完成, 进 %s.%s()" % (MODULE, FUNC))
    _flush()
    getattr(_mod, FUNC)()
    _put("%s.%s() 正常返回" % (MODULE, FUNC))
except KeyboardInterrupt:
    _strikes_write(0)
    _put("用户中断")
except BaseException as _e:
    _code = 1
    _put("!!!! 异常退出: %r" % (_e,))
    _flush()
    try:
        with open(LOG, "a") as _f:
            sys.print_exception(_e, _f)
    except Exception:
        pass
finally:
    # 撤自启标志: 之后手动在 IDE 里跑时, display.py 才允许用 IDE 取景窗。
    # 万一这里没跑到(硬断电), 下次启动时引导会重建它 —— 那个方向是安全的
    # (残留标志只会让"手动跑"少一个取景窗, 不会让"自启"卡死)。
    try:
        os.remove(RUNFLAG)
    except Exception:
        pass
    _swap_print(_rp)                # ⭐必须还原, 否则掉到 REPL 后 print 全进内存
    _flush("---- 结束 ----")
    _rp("[自启] 已退出%s。日志: %s" % ("(异常, 见日志末尾)" if _code else "", LOG))
    _rp("[自启] 板上看日志: 运行 install_autostart.py, 它会把日志打出来")
'''


def _render():
    """把 BOOT 里的 @@名字@@ 占位替换成本文件顶部的常量, 生成最终引导代码。"""
    subs = (("DIR", DIR), ("MODULE", MODULE), ("FUNC", FUNC), ("LOG", LOG),
            ("NOBOOT", NOBOOT), ("STRIKES", STRIKES), ("RUNFLAG", RUNFLAG),
            ("GRACE_S", GRACE_S),
            ("MAX_STRIKES", MAX_STRIKES), ("HEALTHY_MS", HEALTHY_MS),
            ("HEARTBEAT_MS", HEARTBEAT_MS), ("RING_LINES", RING_LINES),
            ("PREIMPORT", repr(list(PREIMPORT))))
    code = BOOT
    for k, v in subs:
        code = code.replace("@@%s@@" % k, str(v))
    if "@@" in code:
        raise ValueError("引导代码里还有没替换的占位符, 别装")
    return code


def _exists(p):
    try:
        os.stat(p)
        return True
    except OSError:
        return False


def _rm(p):
    try:
        os.remove(p)
    except OSError:
        pass


def _dump(path, tail=70):
    """打出日志尾部。板上没有文本编辑器, 这是唯一能看日志的办法。"""
    if not _exists(path):
        return False
    try:
        with open(path) as f:
            lines = f.read().split("\n")
    except Exception as e:
        print("  (%s 读不出来: %r)" % (path, e))
        return False
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        print("  (%s 是空的)" % path)
        return False
    print("  --- %s  共 %d 行%s ---"
          % (path, len(lines), ", 只显示最后 %d 行" % tail if len(lines) > tail else ""))
    for ln in lines[-tail:]:
        print("  " + ln)
    return True


def _show_logs():
    """上一次开机发生了什么。⭐怎么读: 看最后一行的时间戳 ——
    有 '---- 结束 ----' = 正常收尾; 停在心跳上 = 那之后卡住了, 卡点就在末行之后。"""
    print("=" * 58)
    print("  上一次开机的日志")
    print("=" * 58)
    any_log = False
    for p, why in ((LOG, "最近一次"), (LOG + ".prev", "再上一次")):
        if _exists(p):
            print("  【%s】" % why)
            any_log = _dump(p) or any_log
            print()
    if not any_log:
        print("  还没有日志(没自启过, 或自启过但一行都没落盘)。")
        print()


def _preflight():
    """装之前先确认框架在板上。装了自启却没框架 = 开机必崩, 白搭一次断电。"""
    need = (DIR + "/recipes/" + MODULE + ".py", DIR + "/config.json")
    miss = [p for p in need if not _exists(p)]
    if miss:
        print("=" * 58)
        print("  ⛔ 没装成: 框架文件缺失")
        for p in miss:
            print("     找不到 %s" % p)
        print("  先在 IDE 里运行 install_to_board.py 把框架装到板上, 再跑本文件。")
        print("=" * 58)
        return False
    return True


def enable():
    if not _preflight():
        return
    code = _render()
    tmp = TARGET + ".new"
    # ⭐先写临时文件、校验通过后才改名顶上去。不直接写 TARGET 是因为:
    #   写一半失败(卡满/拔卡/写错误)会留下一个**截断的** main.py, 而截断处若刚好
    #   在"已换掉 print"之后、"进 try"之前, 那就是最坏的状态 —— 开机后掉到 REPL
    #   但 print 是哑的, 打什么都没反应, 比明摆着的崩溃难查得多。
    #   改名这一步要么成要么不成: 板上留着的永远是"完整的新版"或"原样的旧版"。
    try:
        with open(tmp, "w") as f:
            f.write(code)       # 注: 板上 open 没有 encoding 参数, 一律 UTF-8
    except Exception as e:
        print("  ⛔ 写 %s 失败: %r" % (tmp, e))
        print("     %s 未被改动。查 TF 卡是否写满/写保护。" % TARGET)
        _rm(tmp)
        return
    # 回读校验。写自启这件事的特点是: 写坏了要到下次上电才知道, 而那时板子可能
    # 已经进不去 REPL —— 所以现在就把内容读回来核对, 别赌。
    try:
        with open(tmp) as f:
            back = f.read()
    except Exception as e:
        print("  ⛔ 写完读不回来: %r" % (e,))
        print("     %s 未被改动, 现状安全。查 TF 卡。" % TARGET)
        _rm(tmp)
        return
    if back != code:
        print("  ⛔ 回读内容与写入的不一致(%d / %d 字节) —— TF 卡有问题。"
              % (len(back), len(code)))
        print("     %s 未被改动, 现状安全。重跑本文件或换卡。" % TARGET)
        _rm(tmp)
        return
    _rm(TARGET)                 # FAT 上 rename 不覆盖已存在的文件, 先删
    try:
        os.rename(tmp, TARGET)
    except Exception as e:
        print("  ⛔ 改名 %s -> %s 失败: %r" % (tmp, TARGET, e))
        print("     ⚠️ 现在板上**没有** %s = 自启未启用(开机直接进 REPL, 安全)。" % TARGET)
        print("     重跑本文件。完整内容还在 %s, 也可手工改名。" % tmp)
        return
    # 之前的 noboot / 失败计数会让新装的自启直接被跳过, 一并清掉。
    # RUNFLAG 也清: 上次硬断电会留下它, 而它残留会让**手动**跑时也没有 IDE
    # 取景窗 —— 那种"少了个窗口却没人告诉你为什么"最难查。
    for p, why in ((NOBOOT, "noboot 跳过标记"), (STRIKES, "失败计数"),
                   (RUNFLAG, "自启运行中标志(上次硬断电残留)")):
        if _exists(p):
            try:
                os.remove(p)
                print("  已清除旧的 %s (%s)" % (why, p))
            except OSError as e:
                print("  ⚠️ 清 %s 失败: %r —— 它还在的话自启会被跳过" % (p, e))
    print("=" * 58)
    print("  ✅ 开机自启已启用   %s  (%d 字节)" % (TARGET, len(back)))
    print("=" * 58)
    print("  验证: **拔 Type-C 断电再上电**(软复位不一定重跑 main.py)。")
    print("        终端应出现 %d 秒倒计时, 之后就没输出了 —— 那是正常的。" % GRACE_S)
    print()
    print("  ⭐终端没输出不等于没跑: 自启时 stdout 没人读, 写满会永久阻塞,")
    print("     所以引导把 print 全接到了日志里。看进度 → 日志文件。")
    print("  提示: 需要实时输出时, 请连接 IDE 手动运行 run_ball_pos.py。")
    print()
    print("  排查: 再运行一次本文件, 顶部会打出上次开机的日志。")
    print("        %s   (上一份 .prev)" % LOG)
    print()
    print("  停用(从软到硬):")
    print("    · 上电 %d 秒内按 Ctrl-C / IDE 停止键" % GRACE_S)
    print("    · 把本文件顶部 ACTION 改成 \"off\" 再运行一次")
    print("    · 板子已经连不上时: 拔 TF 卡插电脑, 建一个空文件 %s" % NOBOOT)
    print("      (或直接删掉 %s)" % TARGET)
    print("    · 兜底: 连续 %d 次开机没跑满 %d 秒, 引导会自己建 noboot 停用,"
          % (MAX_STRIKES, HEALTHY_MS // 1000))
    print("      免得卡死→断电→又卡死, 永远拿不回 REPL。")
    print("=" * 58)


def disable():
    # 建 noboot 而不只是删 main.py: 删文件在 FAT 上偶有不落盘, 而 noboot 是
    # 引导代码进业务逻辑前第一件事就检查的 —— 两个一起来, 才是真关掉了。
    done = []
    try:
        with open(NOBOOT, "w") as f:
            f.write("由 install_autostart.py 停用\n")
        done.append("已建 %s (开机跳过自启)" % NOBOOT)
    except OSError as e:
        done.append("⚠️ 建 %s 失败: %r" % (NOBOOT, e))
    if _exists(TARGET):
        # 按**字节**读并按字节找标记: 文本模式 read(n) 可能把多字节汉字截断,
        # 而标记本身是中文 —— 那会让判据在完全正常的文件上误判。
        try:
            with open(TARGET, "rb") as f:
                head = f.read(600)
        except Exception:
            head = b""
        if _MARK.encode("utf-8") in head:
            try:
                os.remove(TARGET)
                done.append("已删 %s" % TARGET)
            except OSError as e:
                done.append("⚠️ 删 %s 失败: %r" % (TARGET, e))
        else:
            # 里面不是我们写的东西 —— 可能是用户自己的脚本, 不能替他删。
            done.append("⚠️ %s 不是本文件生成的(没有「%s」标记), 已保留未删。"
                        % (TARGET, _MARK))
            done.append("   noboot 已建好, 自启照样跳过; 要删请自己确认内容。")
    else:
        done.append("%s 本来就不存在" % TARGET)
    try:
        os.remove(STRIKES)
    except OSError:
        pass
    print("=" * 58)
    print("  开机自启已停用")
    print("=" * 58)
    for s in done:
        print("  " + s)
    print("  恢复: 把顶部 ACTION 改回 \"on\" 再运行一次。")
    print("=" * 58)


def main():
    _show_logs()        # 先打日志: 这个文件同时是"上次为什么没起来"的排查入口
    if ACTION == "off":
        disable()
    elif ACTION == "on":
        enable()
    else:
        print("  ⛔ ACTION 只能是 \"on\" 或 \"off\", 现在是 %r" % (ACTION,))


if __name__ == "__main__":
    main()
