"""
标定入口 —— CanMV IDE 打开本文件直接运行。

屏上会引导你把球依次摆到 0 / +5 / -5 / +9 / -9 五个刻度, 每个位置静止后点屏上
[SAMPLE] 键, 程序连采 60 帧取中位数; 五点采完自动最小二乘拟合, 结果显示在屏上
并打印到终端, 抄进 config.json 的 ball_pos 段即可。
"""

import os
import sys

BOARD_DIR = "/sdcard/k230_vision"

# chdir → sys.path → import, 顺序不可颠倒(同 run_ball_pos.py)
try:
    os.chdir(BOARD_DIR)
except Exception as e:
    print("[标定] chdir 到 %s 失败: %r" % (BOARD_DIR, e))
    print("       框架未安装。运行 install_to_board.py 后按硬件复位键。")
    raise SystemExit(1)

for p in (BOARD_DIR, BOARD_DIR + "/recipes"):
    if p not in sys.path:
        sys.path.insert(0, p)

import recipes.calib as calib

if __name__ == "__main__":
    calib.main()
