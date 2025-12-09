# -*- coding: utf-8 -*-
# main.py
# NeuroPilot 主程序 (Phase 14 Integrated & Fixed)

import sys
import os
import logging

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QApplication

# 引入 Fluent Widgets
from qfluentwidgets import (
    FluentWindow, NavigationItemPosition, FluentIcon as FIF,
    SplashScreen, setTheme, Theme
)

# 业务模块
from login_dialog import LoginDialog
from dashboard_module import DashboardPage
from task_module import TaskModule
from eeg_module import EEGModule
from device_control import ControlPanel
# 新增调试模块
from debug_module import DebugPanel

# 可选模块
try:
    from ml_module import MLTrainerPanel
except ImportError:
    MLTrainerPanel = None
try:
    from log_module import LogPanel
except ImportError:
    LogPanel = None
try:
    from subject_manager import SubjectManager
except ImportError:
    SubjectManager = None
try:
    from data_module import DataAnalyticsPanel
except ImportError:
    DataAnalyticsPanel = None


class MainWindow(FluentWindow):
    def __init__(self, username: str):
        super().__init__()
        self.username = username
        self.logger = self._setup_logger()

        # 1. 窗口基础设置
        self.init_window()

        # 2. 实例化子页面
        self.dashboard_page = DashboardPage(username)
        self.dashboard_page.setObjectName("dashboardInterface")

        self.task_page = TaskModule()
        self.task_page.setObjectName("taskInterface")

        self.eeg_page = EEGModule()
        self.eeg_page.setObjectName("eegInterface")

        self.device_page = ControlPanel()
        self.device_page.setObjectName("deviceInterface")

        # [Phase 14] 调试模块
        self.debug_page = DebugPanel()
        self.debug_page.setObjectName("debugInterface")

        self.ml_page = None
        if MLTrainerPanel:
            self.ml_page = MLTrainerPanel()
            self.ml_page.setObjectName("mlInterface")

        self.data_page = None
        if DataAnalyticsPanel:
            self.data_page = DataAnalyticsPanel()
            self.data_page.setObjectName("dataInterface")

        self.subject_page = None
        if SubjectManager:
            self.subject_page = SubjectManager()
            self.subject_page.setObjectName("subjectInterface")

        self.log_page = None
        if LogPanel:
            self.log_page = LogPanel()
            self.log_page.setObjectName("logInterface")

        # 3. 初始化导航栏
        self.init_navigation()

        # 4. 信号绑定 (关键集成)
        self._bind_signals()

        # 5. 关闭启动页
        self.splashScreen.finish()

    def init_window(self):
        self.resize(1200, 800)
        self.setMinimumSize(960, 640)
        self.setWindowTitle('NeuroPilot 脑机接口康复系统')

        # 居中
        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)

        # 启动画面
        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(100, 100))
        self.show()

    def init_navigation(self):
        # 首页
        self.addSubInterface(self.dashboard_page, FIF.HOME, '仪表盘')

        # 核心业务
        self.addSubInterface(self.task_page, FIF.GAME, '运动范式')
        self.addSubInterface(self.eeg_page, FIF.HEART, '脑电采集')
        self.addSubInterface(self.device_page, FIF.IOT, '外设控制')

        # 数据与分析
        if self.ml_page:
            self.addSubInterface(self.ml_page, FIF.EDUCATION, '模型训练')
        if self.data_page:
            # [FIXED] 使用有效的图标
            self.addSubInterface(self.data_page, FIF.MARKET, '数据分析')

        # 调试工具 (Phase 14)
        self.addSubInterface(self.debug_page, FIF.DEVELOPER_TOOLS, '调试控制台')

        # 管理
        if self.subject_page:
            self.addSubInterface(self.subject_page, FIF.PEOPLE, '受试者管理')

        # 底部
        if self.log_page:
            self.addSubInterface(self.log_page, FIF.DOCUMENT, '系统日志', NavigationItemPosition.BOTTOM)

    def _setup_logger(self):
        logger = logging.getLogger("NeuroPilot")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
            handler.setFormatter(fmt)
            logger.addHandler(handler)
        return logger

    def _bind_signals(self):
        """Logic Glue"""
        # Dashboard 聚合
        self.dashboard_page.bind_task_module(self.task_page)
        self.dashboard_page.bind_eeg_module(self.eeg_page)
        self.dashboard_page.bind_device_control(self.device_page)

        # 日志聚合
        def log_proxy(text):
            self.logger.info(text)
            if self.log_page:
                self.log_page.append_record("System", "INFO", text)

        for p in [self.dashboard_page, self.task_page, self.eeg_page,
                  self.device_page, self.ml_page, self.data_page, self.log_page, self.debug_page]:
            if p and hasattr(p, 'info'):
                p.info.connect(log_proxy)

        # 范式联动
        self.task_page.stage.connect(self.on_stage_changed)
        self.eeg_page.trial_result.connect(self.on_trial_result)
        # 数据模块记录试次
        if self.data_page:
            self.eeg_page.trial_result.connect(self.data_page.notify_trial_result)
            self.device_page.send_result.connect(self.data_page.notify_device_send)

        # 设备反馈
        self.device_page.device_feedback.connect(lambda s: log_proxy(f"Device: {s}"))
        self.device_page.send_result.connect(self.on_device_send)

        # [Phase 14] 调试信号集成
        # 1. Device Traffic -> Debug Log
        self.device_page.backend.sig_traffic.connect(self.debug_page.append_device_log)

        # 2. EEG Traffic -> Debug Log
        # 注意: eeg_page.worker 是在 __init__ 中创建的 QObject，可以直接访问
        self.eeg_page.worker.sig_traffic_monitor.connect(self.debug_page.append_eeg_log)

        # 3. Debug Send -> Device Backend
        self.debug_page.request_send_device.connect(self.device_page.backend.send_data)

        if self.log_page:
            self.log_page.attach_python_logging(self.logger)

    def on_stage_changed(self, name, idx):
        # 意图判断
        try:
            is_left = (self.task_page.task.currentIndex() == 0)
            label = "left" if is_left else "right"
        except:
            label = "left"

        if name == "运动想象":
            # 记录数据 (Data Module)
            if self.data_page:
                self.data_page.notify_trial_started(
                    self.username,
                    "左手" if label == "left" else "右手",
                    self.task_page.fix.value(),
                    self.task_page.cue.value(),
                    self.task_page.imag.value(),
                    self.task_page.rest.value()
                )

            if hasattr(self.device_page, "sendTrigger"):
                self.device_page.sendTrigger()
            self.eeg_page.begin_trial(label)

        elif name == "休息结束":
            self.eeg_page.end_trial(label)
            if hasattr(self.device_page, "sendTrigger_end"):
                self.device_page.sendTrigger_end()

    def on_trial_result(self, pred, success):
        try:
            intended = "左手" if self.task_page.task.currentIndex() == 0 else "右手"
            self.task_page.notify_trial_result(pred, success, intended)
        except:
            pass
        try:
            self.device_page.handle_trial_result(pred, success)
        except:
            pass

    def on_device_send(self, ok, msg):
        try:
            self.task_page.notify_device_send(ok, msg)
        except:
            pass


def main():
    # 高 DPI 适配
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    setTheme(Theme.LIGHT)

    # 登录
    login = LoginDialog()
    if login.exec_() != LoginDialog.Accepted:
        sys.exit(0)

    username = login.user_edit.text() or "User"

    w = MainWindow(username)
    w.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()