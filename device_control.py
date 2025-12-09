# -*- coding: utf-8 -*-
# device_control.py
# 外设控制模块 (Phase 14 Step 1: Debug Traffic Signals)
# 包含: DeviceBackend (逻辑层) + ControlPanel (UI层)
# 状态: Instrumented for Debugging

import socket
import logging
from typing import Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QStackedWidget
)

# --- Fluent Widgets ---
from qfluentwidgets import (
    CardWidget, SimpleCardWidget,
    PrimaryPushButton, PushButton, ToolButton,
    ComboBox, LineEdit, SwitchButton,
    TitleLabel, SubtitleLabel, BodyLabel, CaptionLabel, StrongBodyLabel,
    FluentIcon as FIF, IconWidget, InfoBar, InfoBarPosition, InfoBadge, InfoLevel
)

# --- Config Manager ---
from core.config_manager import cfg

# 可选依赖
try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None

try:
    from bluetooth import BluetoothSocket, RFCOMM
except ImportError:
    BluetoothSocket, RFCOMM = None, None


class DeviceBackend(QObject):
    """
    设备控制后端逻辑 (I/O Layer)
    负责串口/蓝牙/TCP的物理连接与数据收发
    """
    sig_connected = pyqtSignal(bool, str)
    sig_feedback = pyqtSignal(str)
    sig_send_result = pyqtSignal(bool, str)
    sig_status_msg = pyqtSignal(str)

    # [Phase 14 New] 流量监控信号: (方向 "TX"/"RX", 内容 bytes/str)
    sig_traffic = pyqtSignal(str, object)

    def __init__(self):
        super().__init__()
        self._log = logging.getLogger("NeuroPilot.Device")

        self.ser: Optional["serial.Serial"] = None
        self.bt: Optional["BluetoothSocket"] = None
        self.sock: Optional["socket.socket"] = None

        self.mode = "Serial"
        self._rx_buffer = bytearray()
        self._busy = False

        # 轮询定时器 (20Hz)
        self.rx_timer = QTimer(self)
        self.rx_timer.setInterval(50)
        self.rx_timer.timeout.connect(self._poll_feedback)

    def get_serial_ports(self):
        if serial is None: return ["未安装 pyserial"]
        ports = [p.device for p in list_ports.comports()]
        return ports if ports else ["无可用串口"]

    def connect_device(self, cfg: dict):
        self.mode = cfg.get("mode", "Serial")
        try:
            if self.mode == "Serial":
                if not serial: raise RuntimeError("缺少 pyserial")
                port = cfg["port"]
                baud = cfg.get("baud", 115200)
                self.ser = serial.Serial(port, baudrate=baud, timeout=0.05)
                msg = f"串口已连接: {port}@{baud}"

            elif self.mode == "Bluetooth":
                if not BluetoothSocket: raise RuntimeError("缺少 pybluez")
                addr = cfg["bt_addr"]
                self.bt = BluetoothSocket(RFCOMM)
                self.bt.connect((addr, 1))
                self.bt.setblocking(False)
                msg = f"蓝牙已连接: {addr}"

            elif self.mode == "WiFi":
                ip = cfg["ip"]
                port = cfg["port"]
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(3.0)
                self.sock.connect((ip, port))
                self.sock.setblocking(False)
                msg = f"WiFi已连接: {ip}:{port}"
            else:
                raise ValueError("未知模式")

            self.rx_timer.start()
            self._log.info(msg)
            self.sig_connected.emit(True, msg)
            self.sig_traffic.emit("INFO", f"System Connected: {msg}")

        except Exception as e:
            self._log.error(f"Connect failed: {e}")
            self.sig_connected.emit(False, str(e))

    def disconnect(self):
        self.rx_timer.stop()
        try:
            if self.ser: self.ser.close(); self.ser = None
            if self.bt: self.bt.close(); self.bt = None
            if self.sock: self.sock.close(); self.sock = None
        except:
            pass

        self.sig_connected.emit(False, "已断开")
        self.sig_traffic.emit("INFO", "System Disconnected")

    def send_data(self, payload: bytes):
        if not self.is_connected():
            self.sig_send_result.emit(False, "未连接")
            return

        if self._busy:
            self.sig_send_result.emit(False, "设备忙")
            return

        self._busy = True
        ok, msg = False, ""

        try:
            if self.ser:
                self.ser.write(payload)
                self.ser.flush()
                ok, msg = True, "串口发送成功"
            elif self.bt:
                self.bt.send(payload)
                ok, msg = True, "蓝牙发送成功"
            elif self.sock:
                self.sock.sendall(payload)
                ok, msg = True, "WiFi发送成功"

            # [Phase 14] 埋点发送数据
            if ok:
                self.sig_traffic.emit("TX", payload)

        except Exception as e:
            ok, msg = False, f"发送异常: {e}"

        self.sig_send_result.emit(ok, msg)
        # 短暂忙碌后释放，防止高频点击
        QTimer.singleShot(150, lambda: setattr(self, '_busy', False))

    def is_connected(self):
        return any([self.ser, self.bt, self.sock])

    def _poll_feedback(self):
        """轮询读取设备返回数据"""
        try:
            data = None
            if self.ser and self.ser.in_waiting:
                data = self.ser.read(self.ser.in_waiting)
            elif self.bt:
                try:
                    data = self.bt.recv(1024)
                except:
                    pass
            elif self.sock:
                try:
                    data = self.sock.recv(1024)
                except:
                    pass

            if data:
                # [Phase 14] 埋点接收数据 (原始字节)
                self.sig_traffic.emit("RX", data)

                self._rx_buffer.extend(data)
                while b"\n" in self._rx_buffer:
                    line, _, rest = self._rx_buffer.partition(b"\n")
                    self._rx_buffer = bytearray(rest)
                    text = line.decode("utf-8", errors="ignore").strip()
                    if text:
                        self.sig_feedback.emit(text)
        except Exception as e:
            self._log.error(f"RX Error: {e}")


class ControlPanel(QWidget):
    """
    设备控制面板 (UI Layer)
    全 Fluent Design，支持配置持久化
    """
    info = pyqtSignal(str)
    device_feedback = pyqtSignal(str)
    send_result = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DeviceControl")

        # 实例化后端
        self.backend = DeviceBackend()
        self._bind_backend()

        self._init_ui()
        self._refresh_ports()

        # 关键：加载上次配置
        self._load_settings()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # ==========================================
        # 1. 顶部连接配置卡片
        # ==========================================
        self.conn_card = SimpleCardWidget()
        h_conn = QHBoxLayout(self.conn_card)
        h_conn.setContentsMargins(20, 16, 20, 16)
        h_conn.setSpacing(16)

        icon = IconWidget(FIF.IOT)
        icon.setFixedSize(36, 36)

        info_l = QVBoxLayout()
        info_l.setSpacing(4)
        info_l.addWidget(SubtitleLabel("外设连接配置", self))
        self.lbl_status = CaptionLabel("当前状态: 未连接", self)
        self.lbl_status.setTextColor(QColor(120, 120, 120), QColor(150, 150, 150))
        info_l.addWidget(self.lbl_status)

        # 模式选择
        self.cmb_mode = ComboBox()
        self.cmb_mode.addItems(["串口 (Serial)", "蓝牙 (Bluetooth)", "WiFi (TCP)"])
        self.cmb_mode.setFixedWidth(140)
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)

        # 动态输入区 (Stacked Widget)
        self.stack_input = QStackedWidget()
        self.stack_input.setFixedHeight(36)

        # Page 1: Serial
        w_serial = QWidget()
        l_serial = QHBoxLayout(w_serial)
        l_serial.setContentsMargins(0, 0, 0, 0)
        self.cmb_port = ComboBox()
        self.cmb_port.setMinimumWidth(120)
        self.btn_refresh = ToolButton(FIF.SYNC)
        self.btn_refresh.setToolTip("刷新串口列表")
        self.btn_refresh.clicked.connect(self._refresh_ports)
        self.ed_baud = LineEdit()
        self.ed_baud.setPlaceholderText("波特率")
        self.ed_baud.setFixedWidth(80)
        l_serial.addWidget(self.cmb_port)
        l_serial.addWidget(self.btn_refresh)
        l_serial.addWidget(CaptionLabel("波特:", self))
        l_serial.addWidget(self.ed_baud)
        self.stack_input.addWidget(w_serial)

        # Page 2: Bluetooth
        w_bt = QWidget()
        l_bt = QHBoxLayout(w_bt)
        l_bt.setContentsMargins(0, 0, 0, 0)
        self.ed_bt_addr = LineEdit()
        self.ed_bt_addr.setPlaceholderText("蓝牙 MAC 地址 (00:11:22...)")
        l_bt.addWidget(self.ed_bt_addr)
        self.stack_input.addWidget(w_bt)

        # Page 3: WiFi
        w_wifi = QWidget()
        l_wifi = QHBoxLayout(w_wifi)
        l_wifi.setContentsMargins(0, 0, 0, 0)
        self.ed_ip = LineEdit()
        self.ed_ip.setPlaceholderText("IP 地址")
        self.ed_tcp_port = LineEdit()
        self.ed_tcp_port.setPlaceholderText("端口")
        self.ed_tcp_port.setFixedWidth(70)
        l_wifi.addWidget(self.ed_ip)
        l_wifi.addWidget(CaptionLabel("端口:", self))
        l_wifi.addWidget(self.ed_tcp_port)
        self.stack_input.addWidget(w_wifi)

        # 操作按钮
        self.btn_connect = PrimaryPushButton(FIF.IOT, "连接")
        self.btn_connect.clicked.connect(self._do_connect)
        self.btn_disconnect = PushButton(FIF.CANCEL, "断开")
        self.btn_disconnect.clicked.connect(self.backend.disconnect)
        self.btn_disconnect.setEnabled(False)

        # 组装顶部
        h_conn.addWidget(icon)
        h_conn.addLayout(info_l)
        h_conn.addWidget(self.cmb_mode)
        h_conn.addWidget(self.stack_input)
        h_conn.addStretch(1)
        h_conn.addWidget(self.btn_connect)
        h_conn.addWidget(self.btn_disconnect)

        layout.addWidget(self.conn_card)

        # ==========================================
        # 2. 指令控制卡片
        # ==========================================
        self.ctrl_card = CardWidget()
        ctrl_l = QVBoxLayout(self.ctrl_card)
        ctrl_l.setContentsMargins(20, 16, 20, 16)
        ctrl_l.setSpacing(16)

        ctrl_l.addWidget(StrongBodyLabel("指令控制与自动化", self))

        # 按钮行
        row_btn = QHBoxLayout()
        self.btn_left = PrimaryPushButton(FIF.LEFT_ARROW, "发送: 左手 (L)")
        self.btn_right = PrimaryPushButton(FIF.RIGHT_ARROW, "发送: 右手 (R)")
        self.btn_trigger = PushButton(FIF.SEND, "发送 Trigger")

        self.btn_left.clicked.connect(lambda: self._send_cmd("left"))
        self.btn_right.clicked.connect(lambda: self._send_cmd("right"))
        self.btn_trigger.clicked.connect(self.sendTrigger)

        row_btn.addWidget(self.btn_left)
        row_btn.addWidget(self.btn_right)
        row_btn.addWidget(self.btn_trigger)
        ctrl_l.addLayout(row_btn)

        # 自动化选项 (使用 SwitchButton)
        row_sw = QHBoxLayout()
        self.sw_auto = SwitchButton()
        self.sw_auto.setOnText("自动发送开启")
        self.sw_auto.setOffText("手动模式")

        self.sw_strict = SwitchButton()
        self.sw_strict.setOnText("严格模式 (仅成功时发送)")
        self.sw_strict.setOffText("宽松模式")

        row_sw.addWidget(self.sw_auto)
        row_sw.addSpacing(24)
        row_sw.addWidget(self.sw_strict)
        row_sw.addStretch(1)

        ctrl_l.addLayout(row_sw)

        layout.addWidget(self.ctrl_card)
        layout.addStretch(1)

        self._on_mode_changed()

    # ==========================================
    # 配置持久化 (Persistence)
    # ==========================================
    def _load_settings(self):
        """启动时读取配置"""
        idx = cfg.get("Device", "mode_idx", 0, int)
        self.cmb_mode.setCurrentIndex(idx)
        self.stack_input.setCurrentIndex(idx)

        self.ed_baud.setText(cfg.get("Device", "baud", "115200", str))
        self.ed_bt_addr.setText(cfg.get("Device", "bt_addr", "00:11:22:33:44:55", str))
        self.ed_ip.setText(cfg.get("Device", "ip", "192.168.4.1", str))
        self.ed_tcp_port.setText(cfg.get("Device", "tcp_port", "8080", str))

        self.sw_auto.setChecked(cfg.get("Device", "auto_send", True, bool))
        self.sw_strict.setChecked(cfg.get("Device", "strict_mode", True, bool))

    def closeEvent(self, e):
        """关闭时保存配置"""
        cfg.set("Device", "mode_idx", self.cmb_mode.currentIndex())
        cfg.set("Device", "baud", self.ed_baud.text())
        cfg.set("Device", "bt_addr", self.ed_bt_addr.text())
        cfg.set("Device", "ip", self.ed_ip.text())
        cfg.set("Device", "tcp_port", self.ed_tcp_port.text())
        cfg.set("Device", "auto_send", self.sw_auto.isChecked())
        cfg.set("Device", "strict_mode", self.sw_strict.isChecked())
        super().closeEvent(e)

    # ==========================================
    # 逻辑实现
    # ==========================================
    def _bind_backend(self):
        self.backend.sig_connected.connect(self._on_connected)
        self.backend.sig_feedback.connect(self.device_feedback)
        self.backend.sig_send_result.connect(self._on_send_result)
        self.backend.sig_feedback.connect(lambda s: self.info.emit(f"设备反馈: {s}"))

    def _on_mode_changed(self):
        idx = self.cmb_mode.currentIndex()
        self.stack_input.setCurrentIndex(idx)

    def _refresh_ports(self):
        self.cmb_port.clear()
        ports = self.backend.get_serial_ports()
        self.cmb_port.addItems(ports)

    def _do_connect(self):
        mode_txt = self.cmb_mode.currentText()
        cfg_conn = {}
        if "串口" in mode_txt:
            cfg_conn["mode"] = "Serial"
            cfg_conn["port"] = self.cmb_port.currentText()
            try:
                cfg_conn["baud"] = int(self.ed_baud.text())
            except:
                cfg_conn["baud"] = 115200
        elif "蓝牙" in mode_txt:
            cfg_conn["mode"] = "Bluetooth"
            cfg_conn["bt_addr"] = self.ed_bt_addr.text()
        else:
            cfg_conn["mode"] = "WiFi"
            cfg_conn["ip"] = self.ed_ip.text()
            try:
                cfg_conn["port"] = int(self.ed_tcp_port.text())
            except:
                cfg_conn["port"] = 8080

        self.btn_connect.setEnabled(False)
        self.btn_connect.setText("连接中...")
        self.backend.connect_device(cfg_conn)

    def _on_connected(self, success, msg):
        self.btn_connect.setEnabled(not success)
        self.btn_connect.setText("连接" if not success else "已连接")
        self.btn_disconnect.setEnabled(success)
        self.cmb_mode.setEnabled(not success)

        if success:
            self.lbl_status.setText(f"状态: {msg}")
            InfoBar.success("连接成功", msg, parent=self)
        else:
            self.lbl_status.setText("状态: 连接失败")
            InfoBar.error("连接失败", msg, parent=self)

    def _send_cmd(self, cmd_type):
        payload = b''
        if cmd_type == 'left':
            payload = b'L\n'
        elif cmd_type == 'right':
            payload = b'R\n'
        elif cmd_type == 'trigger':
            payload = b'T\n'

        self.backend.send_data(payload)

    def _on_send_result(self, ok, msg):
        self.send_result.emit(ok, msg)
        if not ok:
            InfoBar.warning("发送失败", msg, position=InfoBarPosition.TOP_RIGHT, parent=self)

    # --- 外部接口 (Main 调用) ---
    def handle_trial_result(self, pred: str, success: bool):
        if not self.sw_auto.isChecked():
            return

        if self.sw_strict.isChecked() and not success:
            self.info.emit("自动发送: 跳过 (预测未成功)")
            return

        self._send_cmd(pred)

    def sendTrigger(self):
        self._send_cmd('trigger')

    def sendTrigger_end(self):
        self.backend.send_data(b'E\n')