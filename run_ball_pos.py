"""
一键启动: 视觉检测 + 触摸屏 UI + 串口发送 + 图传。在 CanMV IDE 中打开并运行。

图传地址(config.video_stream.enabled=true 时): 连热点 K230_BALL / 12345678,
浏览器开 http://192.168.169.1:8080/
"""

import os
import sys

BOARD_DIR = "/sdcard/k230_vision"

# chdir → sys.path → import, 三步顺序不可颠倒:
# IDE 运行脚本时工作目录不是框架目录, 不 chdir 则 load_config("config.json") 报 ENOENT;
# 不进 sys.path 则 import utils 失败(recipes/ 与 utils/ 无完整包结构, MicroPython 不认)。
try:
    os.chdir(BOARD_DIR)
except Exception as e:
    print("[启动] chdir 到 %s 失败: %r" % (BOARD_DIR, e))
    print("       框架未安装。运行 install_to_board.py 后按硬件复位键。")
    raise SystemExit(1)

for p in (BOARD_DIR, BOARD_DIR + "/recipes"):
    if p not in sys.path:
        sys.path.insert(0, p)

print("[启动] cwd = %s" % os.getcwd())

import ball_pos

if __name__ == "__main__":
    ball_pos.main()
