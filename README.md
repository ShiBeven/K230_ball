# K230 Vision Framework

一个运行在 **K230 (CanMV)** 上的轻量级视觉框架，提供颜色识别、KPU 神经网络推理、单目测距/测量、串口通信、云台控制与无线图传等模块化能力，并附带面向"平衡滚球"场景的完整应用配方。

## 功能特性

- **双检测路线**: 纯视觉 (`find_blobs` LAB 颜色阈值) 与 KPU 神经网络 (nncase `.kmodel`) 可通过配置一键切换，输出统一六元组 `[class, conf, x, y, w, h]`
- **单目测量**: 基于相机内参标定，一个 `fx` 导出距离、尺寸、偏移、角度、双目标间距五种量
- **串口通信**: 裸字节 / 8 字节定长帧 / CRC16 帧 / 文本行四种模式，含定长帧组包纯函数，便于 PC 端单测
- **云台控制**: 像素坐标 → 偏角解算 (针孔模型) → 串口输出，帧格式可自定义
- **无线图传**: WiFi 热点 + MJPEG-over-HTTP，手机/电脑浏览器直接观看，无需客户端
- **触摸屏 UI**: 目标设定、状态切换、现场阈值调参 (ST7701 触摸屏)
- **开机自启**: 生成引导代码写入 `/sdcard/main.py`，含 stdout 接管与故障自动停用机制
- **脱机可靠运行**: 显示可选、异常隔离 —— 图传/触摸/显示任一失效不影响核心检测与串口输出

## 硬件要求

| 项目 | 要求 |
|---|---|
| 主控板 | K230 开发板 (嘉楠 Kendryte K230, CanMV 固件) |
| 摄像头 | OV5647 (或兼容 sensor) |
| 显示 (可选) | 官方 ST7701 触摸屏 (800×480) 或 HDMI 显示器 |
| 通信 | UART (与下位机 MCU 互联, 3.3V 电平) |

## 目录结构

```
k230_vision/
├── main.py                  # 框架主入口 (通用骨架循环)
├── config.json              # 全局配置 (相机/检测/串口/图传/标定等)
├── camera.py                # 摄像头封装 (media.sensor)
├── display.py               # 显示封装 (VIRT / HDMI / LCD)
├── color_detector.py        # 颜色阈值检测器
├── kpu_tools.py             # KPU 模型加载与推理
├── ai_detector.py           # PipeLine/DetectionApp 封装 (ai 模式)
├── model_utils.py           # 模型后处理 (NMS / YOLO 解析, 纯函数)
├── image_utils.py           # 图像绘制 / 裁剪 / 亚像素质心
├── serial_comm.py           # 串口通信 (四种模式)
├── gimbal_control.py        # 云台控制 (像素→角度→串口)
├── video_stream.py          # 无线图传 (MJPEG-over-HTTP)
├── run_ball_pos.py          # 钢球位置检测 一键启动入口
├── run_calibration.py       # 五点标定 一键启动入口
├── install_autostart.py     # 开机自启安装器
├── utils/
│   ├── config_loader.py     # JSON 配置加载
│   ├── math_utils.py        # 数学工具 (坐标转换/测距/角度)
│   ├── crc.py               # CRC8 / CRC16
│   └── ekf.py               # 扩展卡尔曼滤波
└── recipes/                 # 应用配方 (独立可运行入口)
    ├── ball_pos.py          # 钢球一维位置检测 (完整应用)
    ├── calib.py             # 五点标定
    ├── detect.py            # 通用目标检测 (vision/ai 切换)
    ├── measure.py           # 单目测量 (五种量)
    ├── color_track.py       # 颜色追踪 (轻量)
    └── template.py          # 新配方模板
```

## 快速开始

### 1. 部署到板子

将本目录整体复制到 K230 的 TF 卡 `/sdcard/k230_vision/` (或通过 CanMV IDE 逐文件上传)。

### 2. 运行通用检测

在 CanMV IDE 中新建脚本并运行:

```python
import sys, os
os.chdir("/sdcard/k230_vision")
sys.path.insert(0, "/sdcard/k230_vision")
import recipes.detect as dt
dt.main()
```

检测路线由 `config.json` 的 `detector.mode` 决定:

| mode | 说明 |
|---|---|
| `"vision"` | 纯视觉 `find_blobs` (LAB 阈值), 可选单目测距 |
| `"ai"` | 神经网络推理 (需在板上部署 `.kmodel` 部署包) |
| `"hybrid"` | 预留, 未实现 |

### 3. 运行钢球位置检测

```python
import sys, os
os.chdir("/sdcard/k230_vision")
sys.path.insert(0, "/sdcard/k230_vision")
import ball_pos
ball_pos.main()
```

或直接在 IDE 中打开 `run_ball_pos.py` 运行。

### 4. 标定

首次使用钢球位置检测前，需进行五点标定:

1. 运行 `run_calibration.py`
2. 按屏上提示将球依次摆到 0 / +5 / −5 / +9 / −9 刻度，每个位置静止后点 [SAMPLE]
3. 拟合结果 (offset / scale) 抄入 `config.json` 的 `ball_pos` 段

## 核心配置说明 (`config.json`)

```jsonc
{
  "camera":   { "width": 640, "height": 480, "fps": 60 },
  "detector": { "mode": "vision" },          // vision / ai / hybrid
  "vision":   { "thresholds": [[...]] },      // LAB 阈值, 用 IDE 阈值编辑器调
  "ranging":  { "enabled": true, "target_size_mm": 25.0, "shape": "square" },
  "serial":   { "enabled": true, "uart": 2, "tx_pin": 5, "rx_pin": 6 },
  "display":  { "type": "lcd" },              // virt / hdmi / lcd
  "video_stream": { "enabled": true, "autostart": false }
}
```

完整字段说明见文件内各 `_comment` 注释。

## 串口协议 (ball8 帧)

8 字节定长帧，单向发送 (视觉 → MCU):

| 字节 | 内容 |
|---|---|
| `[0]` | 帧头 (`config.serial.head`, 默认 `0xA5`) |
| `[1][2]` | 误差 x (大端, 值 = err_0.01cm + 32768) |
| `[3][4]` | 保留 (恒 32768) |
| `[5][6]` | 目标像素数 / 10 (大端, 作粗略置信度) |
| `[7]` | 帧尾 (`config.serial.end`, 默认 `0x5A`) |

发送原则: 每帧最多一包; 无可信读数时不发送任何字节，由 MCU 侧超时判丢。

另有通用 `track8` 帧 (画面中心偏移 + 像素数) 供 `detect.py` 使用，帧格式见 `serial_comm.py` 头部注释。

## 开机自启

在 CanMV IDE 中运行 `install_autostart.py` 即可安装 (向 `/sdcard/main.py` 写入引导代码)。

停用方式 (任选其一):
- 上电 3 秒内通过 IDE / 串口发送 Ctrl-C
- 在 TF 卡根目录创建空文件 `noboot`
- 删除 `/sdcard/main.py`
- 连续 3 次启动失败会自动停用

## 编写自己的配方

复制 `recipes/template.py` 开始，或参考 `recipes/detect.py` 组合框架各模块:

```python
from camera import Camera
from display import Display
from utils.config_loader import load_config

config = load_config("config.json")
cam = Camera(config)
disp = Display(config)

while True:
    img = cam.snapshot()
    # ... 你的视觉处理逻辑 ...
    disp.show(img)
```

## 常见问题

| 现象 | 处理 |
|---|---|
| vision 模式检测不到目标 | 用 IDE 阈值编辑器 (工具 → 机器视觉 → 阈值编辑器) 重调 LAB 阈值 |
| ai 模式卡在模型加载 | 检查部署包目录与 `deploy_config.json` 是否存在; nncase 版本需与训练一致 |
| 改了 config 不生效 | 重新部署到板上 (板上文件不会被 IDE 端修改自动同步) |
| 串口无输出 | 确认 tx/rx 为 IO 编号 (IO5/IO6), 检查共地与电平匹配 |
| `[Serial] Failed to open UART` | 引脚被占用, 尝试更换 UART 或引脚 |
| IDE 停止后摄像头被占用 | 断开重连 IDE, 或按板上复位键 |

## 许可证

[MIT](LICENSE)
