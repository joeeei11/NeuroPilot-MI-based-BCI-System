# -*- coding: utf-8 -*-
# core/eeg_worker.py
# 核心业务工作类：EEG采集、处理、存储、预测
# Phase 14 Step 1: 流量埋点 (Traffic Instrumentation)

import time
import csv
import struct
import socket
import select
import logging
import numpy as np
from datetime import datetime

from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer, pyqtSlot

# 引入 Core 模块
from . import dsp
from .models import CSP
from .data_manager import DataManager

# 引入 sklearn
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 可选硬件依赖
try:
    import serial
except ImportError:
    serial = None

try:
    from bluetooth import BluetoothSocket, RFCOMM
except ImportError:
    BluetoothSocket, RFCOMM = None, None


class RingBuffer:
    """高性能环形缓冲区"""

    def __init__(self, n_channels, maxlen):
        self.n_channels = n_channels
        self.maxlen = maxlen
        self.buf = np.zeros((maxlen, n_channels), dtype=np.float32)
        self.idx = 0
        self.full = False

    def append(self, samples):
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        n = samples.shape[0]
        if n == 0: return

        if n > self.maxlen:
            self.buf[:] = samples[-self.maxlen:, :]
            self.idx = 0
            self.full = True
            return

        remain = self.maxlen - self.idx
        if n <= remain:
            self.buf[self.idx: self.idx + n, :] = samples
            self.idx += n
        else:
            self.buf[self.idx:, :] = samples[:remain]
            overflow = n - remain
            self.buf[:overflow, :] = samples[remain:]
            self.idx = overflow
            self.full = True

        if self.idx >= self.maxlen:
            self.idx = 0
            self.full = True

    def get_last(self, n):
        if not self.full and self.idx < n:
            return None
        if self.idx >= n:
            return self.buf[self.idx - n: self.idx, :].copy()
        else:
            part1 = self.buf[self.idx - n + self.maxlen:, :]
            part2 = self.buf[:self.idx, :]
            return np.vstack((part1, part2))


class AcquisitionThread(QThread):
    """
    采集线程 (Phase 14: 增加流量埋点)
    """
    connection_result = pyqtSignal(bool, str)  # 连接结果(成功/失败, 信息)
    data_ready = pyqtSignal(object)  # 数据块
    error_occurred = pyqtSignal(str)  # 运行时错误

    # [Phase 14] 流量监控信号: (方向 "TX"/"RX", 内容 bytes/str)
    sig_traffic = pyqtSignal(str, object)

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self._running = False
        self._paused = False

    def run(self):
        self._running = True
        mode = self.cfg.get('mode', 'demo')
        srate = self.cfg.get('srate', 250)
        n_ch = self.cfg.get('n_channels', 8)

        # 资源句柄
        ser = None
        bt = None
        sock = None

        # --- Stage 1: 建立物理连接 ---
        try:
            if mode == 'serial':
                if not serial: raise RuntimeError("Missing pyserial")
                ser = serial.Serial(self.cfg['port'], self.cfg.get('baud', 115200), timeout=0.1)

            elif mode == 'bluetooth':
                if not BluetoothSocket: raise RuntimeError("Missing pybluez")
                bt = BluetoothSocket(RFCOMM)
                bt.connect((self.cfg['bt_addr'], 1))

            elif mode == 'tcp':
                ip = self.cfg.get('ip', '127.0.0.1')
                port = int(self.cfg.get('port', 8712))

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3.0)  # 连接超时 3s
                sock.connect((ip, port))

                # 连接建立后，暂时设为非阻塞以便用 select 进行首包校验
                sock.setblocking(False)

                # NeuSenW 参数
                srate = 1000
                n_ch = 9
                pack_points = 40
                bufsize = n_ch * pack_points * 4  # 1440 bytes

                # --- Stage 2: 首包校验 (Handshake via Data) ---
                # 等待数据到来，超时 2s。如果硬件没开，这里会超时。
                rlist, _, _ = select.select([sock], [], [], 2.0)
                if not rlist:
                    raise TimeoutError("连接建立，但未收到数据流 (请检查设备电源/采集软件)")

                # [Phase 14] 埋点：连接建立
                self.sig_traffic.emit("INFO", f"TCP Handshake OK with {ip}:{port}")

            # 握手成功
            self.connection_result.emit(True, f"已连接 ({mode}) - 数据流正常")

        except Exception as e:
            self.connection_result.emit(False, str(e))
            self._running = False
            # 清理
            if sock: sock.close()
            if ser: ser.close()
            if bt: bt.close()
            return  # 退出线程

        # --- Stage 3: 采集循环 ---
        # 演示模式参数
        t_sim = 0.0
        dt_sim = 1.0 / srate

        try:
            while self._running:
                if self._paused:
                    self.msleep(10)
                    continue

                chunk = None

                if mode == 'demo':
                    n_gen = max(1, int(srate * 0.04))
                    time_vec = t_sim + np.arange(n_gen) * dt_sim
                    sig = np.random.randn(n_gen, n_ch) * 2.0
                    for c in range(min(n_ch, 4)):
                        sig[:, c] += 10.0 * np.sin(2 * np.pi * 10.0 * time_vec)
                    chunk = sig.astype(np.float32)
                    t_sim = time_vec[-1]
                    self.msleep(40)

                elif mode == 'serial':
                    if ser.in_waiting:
                        raw_s = ser.read(ser.in_waiting)
                        # [Phase 14] 串口流量摘要
                        self.sig_traffic.emit("RX", f"Serial: {len(raw_s)} bytes")

                        lines = raw_s.decode(errors='ignore').split('\n')
                        vals = []
                        for line in lines:
                            parts = line.split(',')
                            if len(parts) >= n_ch:
                                try:
                                    v = [float(p) for p in parts[:n_ch]]
                                    vals.append(v)
                                except:
                                    pass
                        if vals:
                            chunk = np.array(vals, dtype=np.float32)
                    else:
                        self.msleep(10)

                elif mode == 'tcp':
                    # 使用 select 监听，超时 1.0s (心跳)
                    rlist, _, _ = select.select([sock], [], [], 1.0)

                    if not rlist:
                        continue  # 超时，检查 _running

                    # 读取数据 (阻塞直到读满 bufsize 或报错)
                    try:
                        sock.setblocking(True)  # 临时切回阻塞以确保 MSG_WAITALL
                        raw_bytes = sock.recv(bufsize, socket.MSG_WAITALL)
                        sock.setblocking(False)  # 切回非阻塞给 select 用

                        if not raw_bytes:
                            raise RuntimeError("远程主机关闭了连接")

                        if len(raw_bytes) != bufsize:
                            continue  # 丢包

                        # [Phase 14] TCP 流量摘要 (仅长度，防刷屏)
                        self.sig_traffic.emit("RX", f"TCP Packet: {len(raw_bytes)} bytes")

                        # 解析
                        n_items = n_ch * pack_points
                        vals = struct.unpack('<' + ('f' * n_items), raw_bytes)
                        chunk = np.array(vals, dtype=np.float32).reshape(pack_points, n_ch)

                    except BlockingIOError:
                        continue
                    except Exception as e:
                        raise e

                # 发送数据
                if chunk is not None and chunk.size > 0:
                    self.data_ready.emit(chunk)

        except Exception as e:
            if self._running:
                self.error_occurred.emit(f"采集异常: {e}")
        finally:
            try:
                if ser: ser.close()
                if bt: bt.close()
                if sock: sock.close()
            except:
                pass


class EEGWorker(QObject):
    """
    EEG 业务控制器
    """
    sig_connected = pyqtSignal(bool, str)
    sig_samples_ready = pyqtSignal(object)
    sig_prediction_result = pyqtSignal(str, float)
    sig_status_msg = pyqtSignal(str)

    # [Phase 14] 转发流量信号
    sig_traffic_monitor = pyqtSignal(str, object)

    def __init__(self):
        super().__init__()
        self.acq_thread = None
        self.buffer = None
        self.last_config = {}

        self.csv_file = None
        self.csv_writer = None

        self.srate = 250
        self.n_channels = 8
        self.band_pass = (8.0, 30.0)
        self.notch_freq = 50.0

        self.model_csp = None
        self.model_clf = None
        self.scaler = StandardScaler()
        self.model_ready = False

        self.predict_timer = QTimer()
        self.predict_timer.timeout.connect(self._perform_prediction)
        self.win_size_sec = 1.0
        self._last_pred_label = "unknown"

    @pyqtSlot(dict)
    def start_acquisition(self, config: dict):
        if self.acq_thread and self.acq_thread.isRunning():
            return

        self.last_config = config
        self.sig_status_msg.emit("正在握手 (等待数据流)...")

        self.acq_thread = AcquisitionThread(config)
        self.acq_thread.connection_result.connect(self._on_thread_connection_result)
        self.acq_thread.data_ready.connect(self._on_data_received)
        self.acq_thread.error_occurred.connect(self._on_acq_error)

        # [Phase 14] 连接流量信号转发
        self.acq_thread.sig_traffic.connect(self.sig_traffic_monitor)

        self.acq_thread.start()

    @pyqtSlot(bool, str)
    def _on_thread_connection_result(self, success, msg):
        """处理握手结果"""
        if success:
            self._init_runtime_resources()
            self.sig_connected.emit(True, msg)
            self.sig_status_msg.emit(msg)
        else:
            self.sig_connected.emit(False, msg)
            self.sig_status_msg.emit(f"连接失败: {msg}")
            self.stop_acquisition()

    def _init_runtime_resources(self):
        self.srate = self.last_config.get('srate', 250)
        self.n_channels = self.last_config.get('n_channels', 8)
        if self.last_config.get('mode') == 'tcp':
            self.srate = 1000
            self.n_channels = 9

        self.buffer = RingBuffer(self.n_channels, int(self.srate * 10))

        try:
            path = DataManager().get_new_eeg_file_path(
                subject_name=self.last_config.get('subject', 'Guest'),
                session_id=datetime.now().strftime("%Y%m%d")
            )
            self.csv_file = open(path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            cols = ['time'] + [f'CH{i + 1}' for i in range(self.n_channels)]
            self.csv_writer.writerow(cols)
            self.sig_status_msg.emit(f"数据记录于: {path}")
        except Exception as e:
            self.sig_status_msg.emit(f"文件创建失败: {e}")

    @pyqtSlot()
    def stop_acquisition(self):
        if self.acq_thread:
            self.acq_thread._running = False
            self.acq_thread.quit()
            self.acq_thread.wait()
            self.acq_thread = None

        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

        self.predict_timer.stop()
        self.sig_connected.emit(False, "已断开")

    @pyqtSlot(bool)
    def toggle_prediction(self, running: bool):
        if running:
            if not self.model_ready:
                self.sig_status_msg.emit("模型未训练，无法开启预测")
                return
            interval = 500
            self.predict_timer.start(interval)
            self.sig_status_msg.emit("在线预测已开启")
        else:
            self.predict_timer.stop()
            self.sig_status_msg.emit("在线预测已关闭")

    def _on_data_received(self, chunk: np.ndarray):
        if chunk is None or chunk.size == 0: return

        if self.csv_writer:
            now = time.time()
            dt = 1.0 / self.srate
            rows = []
            for i in range(len(chunk)):
                t_point = now - (len(chunk) - 1 - i) * dt
                row = [f"{t_point:.3f}"] + chunk[i].tolist()
                rows.append(row)
            self.csv_writer.writerows(rows)

        if self.buffer:
            self.buffer.append(chunk)

        self.sig_samples_ready.emit(chunk)

    def _on_acq_error(self, msg):
        self.sig_status_msg.emit(f"采集错误: {msg}")
        self.stop_acquisition()

    def _perform_prediction(self):
        if not self.buffer: return
        n_samples = int(self.win_size_sec * self.srate)
        raw = self.buffer.get_last(n_samples)
        if raw is None: return

        eeg_data = raw[:, :8]

        eeg_data = dsp.notch_filter(eeg_data, self.srate, freq=self.notch_freq)
        eeg_data = dsp.butter_filter(eeg_data, self.srate,
                                     f_low=self.band_pass[0],
                                     f_high=self.band_pass[1])

        eeg_data_T = eeg_data.T
        try:
            feat = self.model_csp.transform(eeg_data_T)
            feat = self.scaler.transform(feat)

            if hasattr(self.model_clf, "predict_proba"):
                probs = self.model_clf.predict_proba(feat)[0]
                pred_idx = np.argmax(probs)
                confidence = probs[pred_idx]
            else:
                pred_idx = self.model_clf.predict(feat)[0]
                confidence = 1.0

            label = "left" if pred_idx == 0 else "right"
            self._last_pred_label = label
            self.sig_prediction_result.emit(label, confidence)
        except Exception:
            pass

    def train_model(self, X_left, X_right, method='svm'):
        if len(X_left) < 2 or len(X_right) < 2:
            self.sig_status_msg.emit("样本不足")
            return

        try:
            X = np.concatenate([np.stack(X_left), np.stack(X_right)], axis=0)
            y = np.array([0] * len(X_left) + [1] * len(X_right))

            self.model_csp = CSP(n_components=4)
            self.model_csp.fit(X, y)

            feats = self.model_csp.transform(X)
            self.scaler.fit(feats)
            feats_norm = self.scaler.transform(feats)

            if method == 'knn':
                self.model_clf = KNeighborsClassifier(n_neighbors=3)
            else:
                self.model_clf = SVC(kernel='rbf', probability=True)

            self.model_clf.fit(feats_norm, y)
            self.model_ready = True
            self.sig_status_msg.emit(f"模型训练完成 (Samples: {len(y)})")

        except Exception as e:
            self.sig_status_msg.emit(f"训练失败: {e}")
            logging.error(f"Train error: {e}", exc_info=True)