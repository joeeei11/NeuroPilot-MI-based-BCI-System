# -*- coding: utf-8 -*-
# eeg_module.py
# 脑电控制模块 (Phase 12: Config Persistence)
# 变更：接入 ConfigManager 实现参数自动保存与恢复

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QApplication
)

# Fluent Widgets 组件
from qfluentwidgets import (
    CardWidget, SimpleCardWidget,
    PrimaryPushButton, PushButton, SwitchButton,
    ComboBox, LineEdit, DoubleSpinBox, SpinBox,
    ProgressBar, InfoBadge, InfoLevel,
    TitleLabel, SubtitleLabel, BodyLabel, CaptionLabel, StrongBodyLabel,
    FluentIcon as FIF, IconWidget
)

# 引入核心业务逻辑
from core.eeg_worker import EEGWorker
# 引入配置管家
from core.config_manager import cfg


class EEGModule(QWidget):
    """
    脑电控制面板
    """

    # 信号定义
    info = pyqtSignal(str)
    classified = pyqtSignal(str, float)
    trial_result = pyqtSignal(str, bool)
    raw_data_ready = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("EEGModule")

        # --- 1. 后台 Worker 线程初始化 ---
        self.worker_thread = QThread()
        self.worker = EEGWorker()
        self.worker.moveToThread(self.worker_thread)

        # 信号连接 (只连一次)
        self.worker.sig_connected.connect(self._on_worker_connected)
        self.worker.sig_status_msg.connect(self._on_worker_msg)
        self.worker.sig_prediction_result.connect(self._on_worker_prediction)
        self.worker.sig_samples_ready.connect(self._on_worker_samples)

        self.worker_thread.start()

        # --- 2. 内部状态 ---
        self._voting = False
        self._votes = []
        self._last_pred = "unknown"

        # 训练采集状态
        self._capture_state = None
        self._train_samples = {'left': [], 'right': []}

        # --- 3. 构建界面 ---
        self._init_ui()

        # --- 4. 加载配置 ---
        self._load_settings()
        self._update_input_fields()
        self._update_ui_state(False)

    def _init_ui(self):
        """构建 Fluent 风格界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # 标题区
        header_l = QHBoxLayout()
        icon = IconWidget(FIF.HEART)
        icon.setFixedSize(32, 32)
        title = TitleLabel("脑电信号处理与分类")
        header_l.addWidget(icon)
        header_l.addSpacing(12)
        header_l.addWidget(title)
        header_l.addStretch()
        layout.addLayout(header_l)

        # ==========================================
        # 卡片 1: 设备连接
        # ==========================================
        conn_card = SimpleCardWidget()
        conn_l = QGridLayout(conn_card)
        conn_l.setContentsMargins(20, 20, 20, 20)
        conn_l.setVerticalSpacing(16)
        conn_l.setHorizontalSpacing(16)

        conn_l.addWidget(StrongBodyLabel("设备连接参数"), 0, 0, 1, 4)

        # 模式选择
        self.mode_combo = ComboBox()
        self.mode_combo.addItems(["演示模式", "串口 (Serial)", "蓝牙 (Bluetooth)", "NeuSenW TCP"])
        self.mode_combo.currentTextChanged.connect(self._update_input_fields)

        conn_l.addWidget(CaptionLabel("采集模式"), 1, 0)
        conn_l.addWidget(self.mode_combo, 1, 1, 1, 3)

        # 参数输入
        self.port_edit = LineEdit();
        self.port_edit.setPlaceholderText("例如 COM3")
        self.baud_edit = LineEdit();
        self.baud_edit.setPlaceholderText("115200")
        self.bt_edit = LineEdit();
        self.bt_edit.setPlaceholderText("MAC 地址")
        self.tcp_host = LineEdit();
        self.tcp_host.setPlaceholderText("IP 地址")
        self.tcp_port = LineEdit();
        self.tcp_port.setPlaceholderText("端口")

        conn_l.addWidget(CaptionLabel("串口/波特"), 2, 0)
        conn_l.addWidget(self.port_edit, 2, 1)
        conn_l.addWidget(self.baud_edit, 2, 2, 1, 2)

        conn_l.addWidget(CaptionLabel("蓝牙地址"), 3, 0)
        conn_l.addWidget(self.bt_edit, 3, 1, 1, 3)

        conn_l.addWidget(CaptionLabel("TCP IP/端口"), 4, 0)
        conn_l.addWidget(self.tcp_host, 4, 1)
        conn_l.addWidget(self.tcp_port, 4, 2, 1, 2)

        # 按钮
        btn_box = QHBoxLayout()
        self.btn_connect = PrimaryPushButton(FIF.IOT, "连接设备")
        self.btn_disconnect = PushButton(FIF.CANCEL, "断开连接")
        self.btn_connect.clicked.connect(self._on_btn_connect)
        self.btn_disconnect.clicked.connect(self._on_btn_disconnect)

        btn_box.addWidget(self.btn_connect)
        btn_box.addWidget(self.btn_disconnect)
        conn_l.addLayout(btn_box, 5, 0, 1, 4)

        layout.addWidget(conn_card)

        # ==========================================
        # 卡片 2: 在线预测
        # ==========================================
        online_card = CardWidget()
        online_l = QHBoxLayout(online_card)
        online_l.setContentsMargins(24, 20, 24, 20)

        sw_l = QVBoxLayout()
        sw_l.addWidget(StrongBodyLabel("在线分类预测"))
        self.chk_online = SwitchButton()
        self.chk_online.setOnText("开启")
        self.chk_online.setOffText("关闭")
        self.chk_online.checkedChanged.connect(self._on_chk_online)
        sw_l.addWidget(self.chk_online)
        sw_l.addStretch()

        res_l = QVBoxLayout()
        res_l.setAlignment(Qt.AlignCenter)
        self.res_title = CaptionLabel("实时预测结果")
        self.res_label = SubtitleLabel("等待数据...")
        self.res_label.setTextColor(QColor("#007AFF"), QColor("#007AFF"))
        res_l.addWidget(self.res_title)
        res_l.addWidget(self.res_label)

        online_l.addLayout(sw_l)
        online_l.addStretch(1)
        online_l.addLayout(res_l)
        layout.addWidget(online_card)

        # ==========================================
        # 卡片 3: 模型校准
        # ==========================================
        calib_card = CardWidget()
        cg = QGridLayout(calib_card)
        cg.setContentsMargins(20, 20, 20, 20)
        cg.setVerticalSpacing(16)

        cg.addWidget(StrongBodyLabel("模型快速校准"), 0, 0, 1, 4)

        self.spin_win = DoubleSpinBox();
        self.spin_win.setValue(1.0);
        self.spin_win.setPrefix("窗口: ")
        self.spin_count = SpinBox();
        self.spin_count.setValue(10);
        self.spin_count.setPrefix("数量: ")
        cg.addWidget(self.spin_win, 1, 0, 1, 2)
        cg.addWidget(self.spin_count, 1, 2, 1, 2)

        self.btn_cap_left = PushButton(FIF.LEFT_ARROW, "采集左手")
        self.btn_cap_right = PushButton(FIF.RIGHT_ARROW, "采集右手")
        self.btn_cap_left.clicked.connect(lambda: self._start_capture('left'))
        self.btn_cap_right.clicked.connect(lambda: self._start_capture('right'))

        cg.addWidget(self.btn_cap_left, 2, 0, 1, 2)
        cg.addWidget(self.btn_cap_right, 2, 2, 1, 2)

        self.progress_bar = ProgressBar()
        self.progress_bar.setValue(0)
        self.lbl_calib_status = CaptionLabel("就绪")
        cg.addWidget(self.lbl_calib_status, 3, 0, 1, 4)
        cg.addWidget(self.progress_bar, 4, 0, 1, 4)

        self.btn_train = PrimaryPushButton(FIF.EDUCATION, "开始训练模型")
        self.btn_train.clicked.connect(self._on_btn_train)
        cg.addWidget(self.btn_train, 5, 0, 1, 4)

        layout.addWidget(calib_card)
        layout.addStretch()

    # ==========================================
    # 配置持久化 (Configuration)
    # ==========================================
    def _load_settings(self):
        """启动时加载"""
        self.mode_combo.setCurrentIndex(cfg.get("EEG", "mode_idx", 0, int))
        self.port_edit.setText(cfg.get("EEG", "port", "COM3", str))
        self.baud_edit.setText(cfg.get("EEG", "baud", "115200", str))
        self.bt_edit.setText(cfg.get("EEG", "bt_addr", "00:11:22:33:44:55", str))
        self.tcp_host.setText(cfg.get("EEG", "tcp_ip", "127.0.0.1", str))
        self.tcp_port.setText(cfg.get("EEG", "tcp_port", "8712", str))

        # 校准参数
        self.spin_win.setValue(cfg.get("EEG", "calib_win", 1.0, float))
        self.spin_count.setValue(cfg.get("EEG", "calib_count", 10, int))

    def _save_settings(self):
        """保存当前所有输入"""
        cfg.set("EEG", "mode_idx", self.mode_combo.currentIndex())
        cfg.set("EEG", "port", self.port_edit.text())
        cfg.set("EEG", "baud", self.baud_edit.text())
        cfg.set("EEG", "bt_addr", self.bt_edit.text())
        cfg.set("EEG", "tcp_ip", self.tcp_host.text())
        cfg.set("EEG", "tcp_port", self.tcp_port.text())

        cfg.set("EEG", "calib_win", self.spin_win.value())
        cfg.set("EEG", "calib_count", self.spin_count.value())

    # ==========================================
    # 逻辑核心
    # ==========================================

    def _update_input_fields(self):
        """根据模式启用/禁用输入框"""
        m = self.mode_combo.currentText()
        is_serial = "串口" in m
        is_bt = "蓝牙" in m
        is_tcp = "TCP" in m

        self.port_edit.setEnabled(is_serial)
        self.baud_edit.setEnabled(is_serial)
        self.bt_edit.setEnabled(is_bt)
        self.tcp_host.setEnabled(is_tcp)
        self.tcp_port.setEnabled(is_tcp)

    def _update_ui_state(self, connected: bool):
        """连接状态改变时更新UI"""
        self.btn_connect.setEnabled(not connected)
        self.btn_connect.setText("连接设备" if not connected else "已连接")
        self.btn_disconnect.setEnabled(connected)
        self.mode_combo.setEnabled(not connected)

        if not connected:
            self.chk_online.setChecked(False)
            self.res_label.setText("未连接")
        else:
            self.port_edit.setEnabled(False)
            self.baud_edit.setEnabled(False)
            self.bt_edit.setEnabled(False)
            self.tcp_host.setEnabled(False)
            self.tcp_port.setEnabled(False)

    def _on_btn_connect(self):
        """执行连接"""
        self._save_settings()  # 保存参数

        mode_text = self.mode_combo.currentText()
        cfg = {'srate': 250, 'n_channels': 8}

        if "串口" in mode_text:
            cfg['mode'] = 'serial'
            cfg['port'] = self.port_edit.text().strip()
            try:
                cfg['baud'] = int(self.baud_edit.text().strip())
            except:
                cfg['baud'] = 115200
        elif "蓝牙" in mode_text:
            cfg['mode'] = 'bluetooth'
            cfg['bt_addr'] = self.bt_edit.text().strip()
        elif "TCP" in mode_text:
            cfg['mode'] = 'tcp'
            cfg['ip'] = self.tcp_host.text().strip()
            try:
                cfg['port'] = int(self.tcp_port.text().strip())
            except:
                cfg['port'] = 8712
            cfg['srate'] = 1000
            cfg['n_channels'] = 9
        else:
            cfg['mode'] = 'demo'

        # 禁用按钮，显示状态
        self.btn_connect.setEnabled(False)
        self.btn_connect.setText("握手中...")
        self.worker.start_acquisition(cfg)

    def _on_btn_disconnect(self):
        self.worker.stop_acquisition()
        self._update_input_fields()

    def _on_chk_online(self, checked):
        self.worker.toggle_prediction(checked)
        if checked:
            self.res_label.setText("分析中...")
        else:
            self.res_label.setText("已停止")

    # --- Worker 回调 ---

    def _on_worker_connected(self, is_connected, msg):
        self.info.emit(msg)
        self._update_ui_state(is_connected)
        if not is_connected:
            self.btn_connect.setText("连接设备")
            self.btn_connect.setEnabled(True)

    def _on_worker_msg(self, msg):
        self.info.emit(msg)

    def _on_worker_prediction(self, label, prob):
        self.classified.emit(label, prob)
        txt = "左手" if label == "left" else "右手"
        self.res_label.setText(f"{txt} ({prob:.2f})")
        self._last_pred = label
        if self._voting:
            self._votes.append(label)

    def _on_worker_samples(self, chunk):
        self.raw_data_ready.emit(chunk)
        if self._capture_state:
            self._process_capture_chunk(chunk)

    # --- 校准逻辑 ---

    def _start_capture(self, target):
        if self._capture_state: return
        win_len = self.spin_win.value()
        count = self.spin_count.value()

        self._capture_state = {
            'target': target,
            'needed': count,
            'collected': 0,
            'samples_needed': int(win_len * self.worker.srate),
            'buffer': []
        }
        self._train_samples[target] = []

        t_str = "左手" if target == "left" else "右手"
        self.lbl_calib_status.setText(f"正在采集 {t_str} ...")
        self.progress_bar.setValue(0)
        self.info.emit(f"开始采集 {t_str}")

    def _process_capture_chunk(self, chunk):
        st = self._capture_state
        if not st: return

        for i in range(len(chunk)):
            st['buffer'].append(chunk[i])
            if len(st['buffer']) >= st['samples_needed']:
                seg = np.array(st['buffer']).T
                self._train_samples[st['target']].append(seg)
                st['buffer'] = []
                st['collected'] += 1

                pct = int(st['collected'] / st['needed'] * 100)
                self.progress_bar.setValue(pct)

                if st['collected'] >= st['needed']:
                    self._finish_capture()
                    break

    def _finish_capture(self):
        t_str = "左手" if self._capture_state['target'] == "left" else "右手"
        self.lbl_calib_status.setText(f"{t_str} 采集完成")
        self.info.emit(f"{t_str} 采集完成")
        self._capture_state = None

    def _on_btn_train(self):
        left = self._train_samples['left']
        right = self._train_samples['right']
        if not left or not right:
            self.info.emit("训练失败: 请先采集双侧样本")
            return

        self.lbl_calib_status.setText("正在训练模型...")
        self.worker.train_model(left, right)
        self.lbl_calib_status.setText("模型训练完成")

    # --- 范式接口 ---

    def begin_trial(self, label):
        self._votes = []
        self._voting = True
        self.res_label.setText(f"投票中 (目标: {label})")

    def end_trial(self, intended):
        self._voting = False
        if not self._votes:
            pred = self._last_pred
        else:
            c_l = self._votes.count("left")
            c_r = self._votes.count("right")
            pred = "left" if c_l >= c_r else "right"

        success = (pred == intended)
        self.trial_result.emit(pred, success)
        txt = "左手" if pred == "left" else "右手"
        icon = "✅" if success else "❌"
        self.res_label.setText(f"最终判定: {txt} {icon}")

    def closeEvent(self, e):
        self._save_settings()
        self.worker.stop_acquisition()
        self.worker_thread.quit()
        self.worker_thread.wait()
        super().closeEvent(e)