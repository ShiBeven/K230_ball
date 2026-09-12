# -*- coding: utf-8 -*-
"""
AI 检测器 —— 封装 PipeLine/DetectionApp 推理链路, 供 ai 模式使用。

架构约束: PipeLine/DetectionApp 必须独占媒体初始化(相机+显示), 与框架的
Camera/Display 并存会死锁在模型加载阶段, 因此本类自管取图与显示。

坐标空间: step() 返回的六元组位于 rgb888p 取图空间(1280x720); 画框时需按
display_size/rgb888p_size 换算到显示空间。该空间与标定内参的 640x480 不同,
ai 模式不支持测距。

检测结果结构: {'scores': [...], 'idx': [...], 'boxes': [[x1, y1, x2, y2]]}
其中 boxes 为左上+右下角点(不是 x, y, w, h)。
"""
import gc
from libs.PlatTasks import DetectionApp
from libs.PipeLine import PipeLine
from libs.Utils import read_json

# ---- 画框样式(与绿框版一致, 想调就改这儿) ----
BOX_COLOR = (255, 0, 255, 0)   # ARGB: 不透明纯绿
BOX_THICK = 5
TEXT_SIZE = 32
SHOW_TEXT = True

# AI 取图分辨率(部署包默认), 与标定的 640x480 不同, 见文件头说明
RGB888P_SIZE = [1280, 720]


class AIDetector:
    """PipeLine + DetectionApp 封装。用法: 构造 -> 循环 step()/draw_and_show() -> close()"""

    def __init__(self, config):
        ai_cfg = config.get("ai_deploy", {})
        root = ai_cfg.get("deploy_dir", "/sdcard/mp_deployment_source")
        if root.endswith("/"):
            root = root[:-1]
        cfg_path = root + "/deploy_config.json"

        # 部署包不存在时立即明确报错, 避免进入加载环节后无提示卡死
        try:
            f = open(cfg_path)
            f.close()
        except OSError:
            print("[AIDetector] 错误: 未找到部署包配置 {}".format(cfg_path))
            print("[AIDetector] 请检查 config.json 的 ai_deploy.deploy_dir,")
            print("[AIDetector] 以及板上是否已拷入 mp_deployment_source 整个目录")
            raise SystemExit(1)

        deploy_conf = read_json(cfg_path)
        kmodel_path = root + "/" + deploy_conf["kmodel_path"]
        self._labels = deploy_conf["categories"]
        confidence_threshold = deploy_conf["confidence_threshold"]
        nms_threshold = deploy_conf["nms_threshold"]
        model_input_size = deploy_conf["img_size"]
        model_type = deploy_conf["model_type"]

        conf_override = ai_cfg.get("conf_override")
        if conf_override is not None:
            confidence_threshold = conf_override

        anchors = []
        if model_type == "AnchorBaseDet":
            # 3 组 anchors 必须拼接(部署包约定)
            anchors = (deploy_conf["anchors"][0]
                       + deploy_conf["anchors"][1]
                       + deploy_conf["anchors"][2])

        print("[AIDetector] 置信度阈值 =", confidence_threshold,
              " 标签 =", self._labels)
        print("[AIDetector] 加载模型中(首次加载需要较长时间)...")

        # display_mode 用 "hdmi": PipeLine 不支持 virt, 但库内部 to_ide=True
        # 会把画面推到 IDE 取景窗, 无需真接 HDMI 屏。
        # ⚠️ 库内部的 to_ide 无法关闭, 因此脱机(无 IDE)运行 AI 路线会阻塞。
        self._pl = PipeLine(rgb888p_size=RGB888P_SIZE, display_mode="hdmi")
        self._pl.create()
        display_size = self._pl.get_display_size()
        self._det_app = DetectionApp(
            "video", kmodel_path, self._labels, model_input_size, anchors,
            model_type, confidence_threshold, nms_threshold,
            RGB888P_SIZE, display_size, debug_mode=0)
        self._det_app.config_preprocess()

        # rgb888p 空间 -> 显示空间 的画框缩放比例
        self._sx = display_size[0] / RGB888P_SIZE[0]
        self._sy = display_size[1] / RGB888P_SIZE[1]
        print("[AIDetector] 取图 %dx%d -> 显示 %dx%d, 画框缩放 %.3f / %.3f"
              % (RGB888P_SIZE[0], RGB888P_SIZE[1],
                 display_size[0], display_size[1], self._sx, self._sy))
        print("[AIDetector] 模型就绪, 开始检测")

    def labels(self):
        return self._labels

    def step(self):
        """
        取一帧并推理。

        Returns:
            [[class_name, conf, x, y, w, h], ...]  坐标在 rgb888p 空间(1280x720),
            x,y=框左上角。无检出返回空列表。
        """
        img = self._pl.get_frame()
        res = self._det_app.run(img)

        dets = []
        boxes = res.get("boxes") if res else None
        scores = res.get("scores") if res else None
        idxs = res.get("idx") if res else None
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                if idxs is not None:
                    name = self._labels[int(idxs[i])]
                else:
                    name = "?"
                conf = float(scores[i]) if scores is not None else 0.0
                dets.append([name, conf, float(x1), float(y1),
                             float(x2 - x1), float(y2 - y1)])
        return dets

    def draw_and_show(self, dets):
        """把六元组画到 OSD 层并刷新显示(内部做 rgb888p->显示空间换算)"""
        # 先清空上一帧 OSD, 否则旧框残留成拖影
        self._pl.osd_img.clear()
        for d in dets:
            name, conf = d[0], d[1]
            x1 = int(d[2] * self._sx)
            y1 = int(d[3] * self._sy)
            w = int(d[4] * self._sx)
            h = int(d[5] * self._sy)
            self._pl.osd_img.draw_rectangle(x1, y1, w, h,
                                            color=BOX_COLOR, thickness=BOX_THICK)
            if SHOW_TEXT:
                txt = "%s %.2f" % (name, conf)
                ty = y1 - TEXT_SIZE - 2          # 标签放框上方
                if ty < 0:
                    ty = y1 + h + 2              # 贴顶就翻到框下方
                self._pl.osd_img.draw_string_advanced(x1, ty, TEXT_SIZE,
                                                      txt, color=BOX_COLOR)
        self._pl.show_image()
        gc.collect()

    def close(self):
        """释放推理与媒体资源(顺序与绿框版一致)"""
        try:
            self._det_app.deinit()
        except Exception:
            pass
        try:
            self._pl.destroy()
        except Exception:
            pass
