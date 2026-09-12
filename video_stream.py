"""
无线图传 —— WiFi 热点 + MJPEG-over-HTTP, 浏览器直接观看, 无需客户端。

    sensor 640x480 → 主循环检测/上屏/串口 → offer(img)
                                             └ 每 N 帧: 缩放 → to_jpeg → socket

编码在主循环内同步执行, 由 every_n_frames 控制频次。
本平台 MicroPython 单核分时, 线程化编码不能降低主循环开销。

平台约束:
  1. 只用 img.to_jpeg(), 绝不用 img.compress() —— 后者原地压缩 sensor
     帧缓冲, 会挂死主线程。
  2. copy()/mean_pool() 在此固件上原地修改源图, 缩放结果只能用一次,
     且不可赋回 img。
  3. AP 的 config() 只接受 ssid 与 key; 读回配置用 ap.info()。

设计原则: 图传是可牺牲的一方 —— 任何异常只关闭图传自身, 不影响视觉与串口。
"""

import time


class VideoStream:
    """
    WiFi AP + MJPEG-over-HTTP 图传（同步编码版）。

    用法:
        vs = VideoStream(cfg)
        vs.start()                      # 起网 + 起监听 socket
        ...主循环每帧: vs.offer(img)     # 内部按 every_n_frames 决定编不编
        vs.close()

    offer() 大多数帧直接返回; 只在该编码的帧才有开销, 代价由
    every_n_frames 决定, 可预测。
    """

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))

        net = cfg.get("net") or {}
        self.mode = str(net.get("mode", "ap"))          # ap | sta
        self.ssid = str(net.get("ssid", "K230_BALL"))
        self.key = str(net.get("key", "12345678"))
        self.sta_wait_s = float(net.get("sta_wait_s", 10.0))

        v = cfg.get("video") or {}
        self.quality = int(v.get("quality", 50))
        self.scale = float(v.get("scale", 0.5))
        # 每 N 帧编一帧, 是控制图传开销的主要参数。
        self.every = max(1, int(v.get("every_n_frames", 6)))

        # 每编 N 帧主动 gc.collect() 一次, 避免累积成一次长停顿。
        self.gc_every = max(1, int(v.get("gc_every_n_encodes", 8)))

        h = cfg.get("http") or {}
        self.port = int(h.get("port", 8080))
        # accept 超时: 主循环内不能阻塞, 用极短超时轮询
        self.accept_ms = int(h.get("accept_timeout_ms", 5))

        self._ok = False
        self._ip = None
        self._url = None
        self._srv = None
        self._cli = None          # 当前客户端(同时只服务一个)
        self._n = 0               # offer 计数
        self._n_enc = 0
        self._n_sent = 0
        self._n_err = 0
        self._t_enc = 0.0
        self._t_send = 0.0
        self._n_bytes = 0
        self._gc_n = 0
        self._warned_noscale = False

    # ---------- 对外 ----------

    def is_on(self):
        return self._ok

    def url(self):
        return self._url

    def ip(self):
        return self._ip

    def stats(self):
        return self._n_sent, self._n_err

    def stats_full(self):
        """(编码帧数, 发出帧数, 出错次数, 当前连接数)"""
        return (self._n_enc, self._n_sent, self._n_err,
                1 if self._cli is not None else 0)

    def timing(self):
        """(平均编码ms, 平均发送ms, 平均KB)"""
        n = max(self._n_enc, 1)
        m = max(self._n_sent, 1)
        return (self._t_enc / n, self._t_send / m, self._n_bytes / m / 1024.0)

    def offer(self, img):
        """
        主循环每帧调一次。

        绝大多数帧直接返回(一次取模); 只在该编的那一帧才编码+发送。
        全程 try 包死, 任何异常只关掉图传自己, 绝不向上抛。
        """
        if not self._ok:
            return
        self._n += 1
        if self._n % self.every:
            return
        try:
            if self._cli is None:
                self._try_accept()
                if self._cli is None:
                    return
            self._send_frame(img)
        except Exception as e:
            # 客户端断开是常态(关网页/切后台), 不是故障 —— 丢掉这个连接等下一个
            self._n_err += 1
            self._drop_client(e)

    def start(self):
        """起网 + 起监听 socket。返回 True/False, 不抛异常。"""
        if not self.enabled:
            return False
        try:
            ip = self._net_up()
            if ip is None:
                self._ok = False
                return False

            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(socket.getaddrinfo("0.0.0.0", self.port)[0][-1])
            s.listen(1)
            # 非阻塞 accept: 主循环里"顺手看一眼有没有人连", 绝不能阻塞
            s.settimeout(self.accept_ms / 1000.0)
            self._srv = s
            self._url = "http://%s:%d/" % (ip, self.port)
            self._ok = True

            print("=" * 58)
            print("  [图传] 已开启   %s" % self._url)
            if self.mode == "ap":
                print("  先连热点: SSID=%s  密码=%s" % (self.ssid, self.key))
            print("  看画面: 手机/电脑**浏览器**直接打开上面网址(不用装App)")
            print("  录像:   PC 端跑 tools/图传接收_录像.py")
            print("  MJPEG q=%d 缩放x%.2f 每%d帧编一帧"
                  " → 图传约%.1ffps, 主循环额外约%.1fms/帧"
                  % (self.quality, self.scale, self.every,
                     33.0 / self.every, 25.0 / self.every))
            print("=" * 58)
            return True
        except Exception as e:
            print("  [图传] 启动失败 → 关闭图传, 视觉照常运行: %r" % (e,))
            self._ok = False
            return False

    def close(self):
        self._ok = False
        for o in (self._cli, self._srv):
            try:
                if o is not None:
                    o.close()
            except Exception:
                pass
        self._cli = None
        self._srv = None

    # ---------- 内部 ----------

    def _drop_client(self, why=None):
        try:
            if self._cli is not None:
                self._cli.close()
        except Exception:
            pass
        if self._cli is not None and why is not None and self._n_err <= 5:
            print("  [图传] 客户端断开(正常, 等下一个): %r" % (why,))
        self._cli = None

    def _try_accept(self):
        """顺手看一眼有没有人连。超时=没人连, 立刻返回, 不阻塞主循环。"""
        try:
            cli, addr = self._srv.accept()
        except Exception:
            return          # 超时即"没人连", 是正常路径
        try:
            # 发送超时必须很短: 客户端阻塞时避免长时间占用主循环。
            # (再叠加 GIL 抢占更明显)。60ms = 两帧的预算, 超了就丢这帧,
            # 图传掉几帧无所谓, 主循环卡住不行。
            cli.settimeout(0.06)
            try:
                cli.recv(512)       # 读掉请求头, 不解析路径
            except Exception:
                pass
            cli.send(b"HTTP/1.0 200 OK\r\n"
                     b"Cache-Control: no-store\r\n"
                     b"Pragma: no-cache\r\n"
                     b"Connection: close\r\n"
                     b"Content-Type: multipart/x-mixed-replace; "
                     b"boundary=k230\r\n\r\n")
            self._cli = cli
            print("  [图传] 客户端接入 %s" % (addr,))
        except Exception as e:
            print("  [图传] 握手失败: %r" % (e,))
            try:
                cli.close()
            except Exception:
                pass

    def _send_frame(self, img):
        """
        编一帧并发出去。

        内存优化(避免 MicroPython 堆碎片化导致的性能衰减):
        ① 用 to_jpeg 的 roi/scale 参数一步到位, 避免整图 copy;
        ② 头部与数据分开发送, 避免整帧数据的拼接拷贝;
        ③ 发送完成后主动 GC(此时临时对象刚成为垃圾, 回收效率最高)。
        """
        t_a = time.ticks_us()
        # 直接用 to_jpeg 的 scale 参数, 避免整图 copy。
        # 注意: 禁用 img.compress()(原地压缩) 与对帧缓冲赋值 copy 结果。
        try:
            j = img.to_jpeg(quality=self.quality,
                            x_scale=self.scale, y_scale=self.scale)
        except TypeError:
            # 老固件的 to_jpeg 不吃 scale 参数 → 退回"先 copy 再编"。
            # 记住这条分支比较费内存, 若走到这里 every_n_frames 要调大些。
            if not self._warned_noscale:
                self._warned_noscale = True
                print("  [图传] to_jpeg 不支持 x_scale → 退回 copy 路径"
                      "(更费内存, 建议 every_n_frames 调大)")
            j = img.copy(x_scale=self.scale,
                         y_scale=self.scale).to_jpeg(quality=self.quality)
        buf = j.bytearray()
        self._n_enc += 1
        t_b = time.ticks_us()
        self._t_enc += time.ticks_diff(t_b, t_a) / 1000.0

        # 头部与数据分开发送: 两次 send 的调用开销小于整帧数据的拼接拷贝。
        cli = self._cli
        cli.send(b"--k230\r\nContent-Type: image/jpeg\r\n"
                 b"Content-Length: %d\r\n\r\n" % len(buf))
        cli.send(buf)
        cli.send(b"\r\n")
        self._n_sent += 1
        self._t_send += time.ticks_diff(time.ticks_us(), t_b) / 1000.0
        self._n_bytes += len(buf)

        # 定期主动 GC, 避免累积触发一次长停顿的大回收。
        self._gc_n += 1
        if self._gc_n >= self.gc_every:
            self._gc_n = 0
            try:
                import gc
                gc.collect()
            except Exception:
                pass

    def _net_up(self):
        """启动 WiFi (AP/STA), 返回 IP 或 None。"""
        try:
            import network
        except Exception as e:
            print("  [图传] import network 失败: %r" % (e,))
            return None
        try:
            if self.mode == "ap":
                ap = network.WLAN(network.AP_IF)
                ap.config(ssid=self.ssid, key=self.key)   # 只吃这两个关键字
                time.sleep(2)
                try:
                    print("  [图传] AP info: %s" % (ap.info(),))
                except Exception:
                    pass
                ip = ap.ifconfig()[0]
            else:
                sta = network.WLAN(network.STA_IF)
                sta.connect(self.ssid, self.key)   # 不要 active(True), 禁忌3
                t0 = time.ticks_ms()
                while time.ticks_diff(time.ticks_ms(), t0) < self.sta_wait_s * 1000:
                    if sta.isconnected():
                        break
                    time.sleep_ms(300)
                if not sta.isconnected():
                    print("  [图传] STA 连 %r 超时" % (self.ssid,))
                    return None
                ip = sta.ifconfig()[0]
            if not ip or ip == "0.0.0.0":
                print("  [图传] 没拿到 IP(%r)" % (ip,))
                return None
            self._ip = ip
            return ip
        except Exception as e:
            print("  [图传] 起网失败: %r" % (e,))
            return None
