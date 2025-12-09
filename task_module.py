# -*- coding: utf-8 -*-
# task_module.py
# 运动范式模块 (Fluent Design Phase 6.2 Hotfix)
# 修复: 移除 InfoLevel.NORMAL/INFO 导致的崩溃
# 修复: 使用 InfoLevel.WARNING 替代 ATTENTION 以保证最大兼容性

import os
import csv
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSettings, QUrl
from PyQt5.QtGui import QFont, QMovie, QDesktopServices
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QStackedLayout, QFileDialog, QFrame
)

# Fluent Widgets
from qfluentwidgets import (
    CardWidget, SimpleCardWidget, ElevatedCardWidget,
    PrimaryPushButton, PushButton, ToolButton, SwitchButton,
    ComboBox, DoubleSpinBox, SpinBox, LineEdit,
    TitleLabel, SubtitleLabel, BodyLabel, StrongBodyLabel, CaptionLabel,
    InfoBadge, InfoLevel, FluentIcon as FIF,
    SmoothScrollArea, VBoxLayout
)

# 默认资源路径
LEFT_GIF_DEFAULT = "assets/left_hand_grasp.gif"
RIGHT_GIF_DEFAULT = "assets/right_hand_grasp.gif"


class FluentStageBar(QWidget):
    """使用 InfoBadge 显示的阶段指示条"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.badges = []
        stages = ["注视点", "方向提示", "运动想象", "休息"]

        for name in stages:
            # 默认创建 INFO 级别的 Badge (蓝色/灰色)
            # 注意: InfoBadge.info() 是静态工厂方法，内部会自动处理 Level
            badge = InfoBadge.info(name)
            layout.addWidget(badge)
            self.badges.append(badge)

        layout.addStretch(1)

    def highlight(self, idx: int):
        # 0: Fix, 1: Cue, 2: Imag, 3: Rest
        for i, badge in enumerate(self.badges):
            if i == idx:
                # 仅在特殊阶段设置特殊颜色
                if i == 2:  # 想象阶段 -> 警告色 (橙色)
                    try:
                        # 尝试使用 WARNING (最通用)
                        badge.setLevel(InfoLevel.WARNING)
                    except AttributeError:
                        pass
                elif i == 3:  # 休息 -> 成功色 (绿色)
                    try:
                        badge.setLevel(InfoLevel.SUCCESS)
                    except AttributeError:
                        pass
                else:
                    # 其他阶段保持默认，不调用 setLevel 以避免 'NORMAL'/'INFO' 报错
                    pass
            else:
                # 非高亮阶段不强制重置，避免触发崩溃
                # 下次重绘或状态切换时，InfoBadge 通常会保持最后状态或由样式表控制
                pass


class StimulusArea(ElevatedCardWidget):
    """刺激呈现区域 (纯白卡片)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBorderRadius(10)
        # 强制白色背景
        self.setStyleSheet("ElevatedCardWidget { background-color: #FFFFFF; border: 1px solid #E5E5E5; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 内部堆栈布局
        self.container = QWidget()
        self.stack = QStackedLayout(self.container)

        # 1. 注视点 (+)
        self.fix = QLabel("+")
        self.fix.setAlignment(Qt.AlignCenter)
        self.fix.setStyleSheet("font-size: 80px; color: #000000; font-weight: bold;")

        # 2. 提示 (←/→)
        self.cue = QLabel("←")
        self.cue.setAlignment(Qt.AlignCenter)
        self.cue.setStyleSheet("font-size: 100px; color: #007AFF; font-weight: bold;")

        # 3. GIF
        self.gif_label = QLabel()
        self.gif_label.setAlignment(Qt.AlignCenter)
        self.gif_label.setScaledContents(True)
        self.movie = QMovie(self)
        self.gif_label.setMovie(self.movie)

        # 4. 休息
        self.rest = QLabel("休息")
        self.rest.setAlignment(Qt.AlignCenter)
        self.rest.setStyleSheet("font-size: 32px; color: #666666;")

        self.stack.addWidget(self.fix)
        self.stack.addWidget(self.cue)
        self.stack.addWidget(self.gif_label)
        self.stack.addWidget(self.rest)

        self.stack.setCurrentIndex(3)
        layout.addWidget(self.container)

    def show_fix(self):
        self.stack.setCurrentIndex(0)

    def show_cue(self, is_left: bool):
        self.cue.setText("←" if is_left else "→")
        self.cue.setStyleSheet(f"font-size: 100px; color: {'#007AFF' if is_left else '#FF4D4F'}; font-weight: bold;")
        self.stack.setCurrentIndex(1)

    def show_gif(self, path: str):
        if not path or not os.path.exists(path):
            self.rest.setText(f"GIF丢失: {path}")
            self.stack.setCurrentIndex(3)
            return
        self.movie.stop()
        self.movie.setFileName(path)
        self.movie.setCacheMode(QMovie.CacheAll)
        self.movie.start()
        self.stack.setCurrentIndex(2)

    def show_rest(self, text="休息"):
        self.movie.stop()
        self.rest.setText(text)
        self.stack.setCurrentIndex(3)


class TaskModule(QWidget):
    """
    运动范式模块 (Fluent Design)
    """
    info = pyqtSignal(str)
    stage = pyqtSignal(str, int)
    trial_progress = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setObjectName("TaskModule")

        # 设置持久化
        self.settings = QSettings("NeuroPilot", "TaskModule")
        self.left_gif_path = self.settings.value("left_gif", LEFT_GIF_DEFAULT, type=str)
        self.right_gif_path = self.settings.value("right_gif", RIGHT_GIF_DEFAULT, type=str)

        # 内部状态
        self._running = False
        self._timers = []
        self._loop_left = 0
        self._iti_ms = 0
        self._total_trials = 0
        self._records = []
        self._last_idx = None
        self._cnt_total = 0
        self._cnt_succ = 0
        self._cnt_send = 0

        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # =================================================
        # 左侧：设置区 (使用 ScrollArea 防止屏幕过小)
        # =================================================
        self.settings_area = SmoothScrollArea()
        self.settings_area.setFixedWidth(360)
        self.settings_area.setWidgetResizable(True)
        self.settings_area.setStyleSheet("background-color: transparent; border: none;")

        settings_content = QWidget()
        settings_layout = QVBoxLayout(settings_content)
        settings_layout.setContentsMargins(0, 0, 10, 0)
        settings_layout.setSpacing(16)

        # 1. 参数设置卡片
        param_card = CardWidget()
        param_l = QVBoxLayout(param_card)
        param_l.setContentsMargins(16, 16, 16, 16)
        param_l.setSpacing(12)

        param_l.addWidget(StrongBodyLabel("任务参数"))

        # 任务类型
        self.task = ComboBox()
        self.task.addItems(["左手抓握 (Left)", "右手抓握 (Right)"])
        param_l.addWidget(CaptionLabel("任务类型"))
        param_l.addWidget(self.task)

        # 时间参数
        self.fix = DoubleSpinBox();
        self._cfg_spin(self.fix, 2.0)
        self.cue = DoubleSpinBox();
        self._cfg_spin(self.cue, 1.25)
        self.imag = DoubleSpinBox();
        self._cfg_spin(self.imag, 4.0)
        self.rest = DoubleSpinBox();
        self._cfg_spin(self.rest, 1.0)

        param_l.addWidget(CaptionLabel("注视点时长 (s)"))
        param_l.addWidget(self.fix)
        param_l.addWidget(CaptionLabel("提示时长 (s)"))
        param_l.addWidget(self.cue)
        param_l.addWidget(CaptionLabel("想象时长 (s)"))
        param_l.addWidget(self.imag)
        param_l.addWidget(CaptionLabel("休息时长 (s)"))
        param_l.addWidget(self.rest)

        # 循环设置
        loop_layout = QHBoxLayout()
        loop_layout.addWidget(StrongBodyLabel("自动循环"))
        self.loop_switch = SwitchButton()
        self.loop_switch.setOnText("开")
        self.loop_switch.setOffText("关")
        loop_layout.addStretch()
        loop_layout.addWidget(self.loop_switch)
        param_l.addLayout(loop_layout)

        self.n_trials = SpinBox();
        self.n_trials.setRange(1, 999);
        self.n_trials.setValue(5)
        self.iti = DoubleSpinBox();
        self._cfg_spin(self.iti, 2.0)
        self.n_trials.setEnabled(False)
        self.iti.setEnabled(False)

        self.loop_switch.checkedChanged.connect(lambda c: (self.n_trials.setEnabled(c), self.iti.setEnabled(c)))

        param_l.addWidget(CaptionLabel("循环次数"))
        param_l.addWidget(self.n_trials)
        param_l.addWidget(CaptionLabel("间隔时间 (s)"))
        param_l.addWidget(self.iti)

        # 2. 素材设置卡片
        gif_card = CardWidget()
        gif_l = QVBoxLayout(gif_card)
        gif_l.setContentsMargins(16, 16, 16, 16)
        gif_l.setSpacing(12)
        gif_l.addWidget(StrongBodyLabel("视觉素材 (GIF)"))

        # 左手 GIF
        gif_l.addWidget(CaptionLabel("左手 GIF 路径"))
        l_row = QHBoxLayout()
        self.left_edit = LineEdit()
        self.left_edit.setText(self.left_gif_path)
        self.btn_l_browse = ToolButton(FIF.FOLDER)
        self.btn_l_browse.clicked.connect(lambda: self._pick_gif("left"))
        l_row.addWidget(self.left_edit)
        l_row.addWidget(self.btn_l_browse)
        gif_l.addLayout(l_row)

        # 右手 GIF
        gif_l.addWidget(CaptionLabel("右手 GIF 路径"))
        r_row = QHBoxLayout()
        self.right_edit = LineEdit()
        self.right_edit.setText(self.right_gif_path)
        self.btn_r_browse = ToolButton(FIF.FOLDER)
        self.btn_r_browse.clicked.connect(lambda: self._pick_gif("right"))
        r_row.addWidget(self.right_edit)
        r_row.addWidget(self.btn_r_browse)
        gif_l.addLayout(r_row)

        # 3. 统计导出
        stat_card = SimpleCardWidget()
        stat_l = QVBoxLayout(stat_card)
        stat_l.setContentsMargins(16, 16, 16, 16)
        self.lbl_stats = CaptionLabel("总完成: 0 | 成功: 0")
        self.btn_export = PushButton(FIF.SHARE, "导出 CSV")
        self.btn_export.clicked.connect(self.export_csv)
        stat_l.addWidget(StrongBodyLabel("数据统计"))
        stat_l.addWidget(self.lbl_stats)
        stat_l.addSpacing(4)
        stat_l.addWidget(self.btn_export)

        # 添加到 ScrollArea
        settings_layout.addWidget(param_card)
        settings_layout.addWidget(gif_card)
        settings_layout.addWidget(stat_card)
        settings_layout.addStretch(1)

        self.settings_area.setWidget(settings_content)

        # =================================================
        # 右侧：视觉呈现区
        # =================================================
        visual_area = QWidget()
        v_layout = QVBoxLayout(visual_area)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(16)

        # 顶部：标题 + 阶段条
        top_bar = QHBoxLayout()
        title_l = QVBoxLayout()
        title_l.setSpacing(4)
        title_l.addWidget(TitleLabel("运动范式演示"))
        self.subtitle = CaptionLabel("请保持专注，根据提示进行运动想象")
        title_l.addWidget(self.subtitle)

        self.stage_bar = FluentStageBar()

        top_bar.addLayout(title_l)
        top_bar.addStretch(1)
        top_bar.addWidget(self.stage_bar)

        # 中部：刺激呈现 (ElevatedCardWidget)
        self.stim = StimulusArea()

        # 底部：控制按钮
        ctl_bar = QHBoxLayout()
        self.btn_start = PrimaryPushButton(FIF.CARE_RIGHT_SOLID, "开始试次")
        self.btn_start.setFixedWidth(140)
        self.btn_start.clicked.connect(self.start_trial)

        self.btn_stop = PushButton(FIF.PAUSE, "停止任务")
        self.btn_stop.setFixedWidth(120)
        self.btn_stop.clicked.connect(self.abort_trial)
        self.btn_stop.setEnabled(False)

        ctl_bar.addStretch(1)
        ctl_bar.addWidget(self.btn_start)
        ctl_bar.addSpacing(16)
        ctl_bar.addWidget(self.btn_stop)
        ctl_bar.addStretch(1)

        v_layout.addLayout(top_bar)
        v_layout.addWidget(self.stim, 1)  # 拉伸
        v_layout.addLayout(ctl_bar)

        # 整体组装
        main_layout.addWidget(self.settings_area)
        main_layout.addWidget(visual_area, 1)  # 右侧占主要空间

    def _cfg_spin(self, spin, val):
        spin.setSingleStep(0.25)
        spin.setRange(0.25, 60.0)
        spin.setValue(val)

    def _pick_gif(self, side):
        path, _ = QFileDialog.getOpenFileName(self, "选择 GIF", "", "GIF Files (*.gif)")
        if path:
            if side == "left":
                self.left_edit.setText(path)
                self.left_gif_path = path
                self.settings.setValue("left_gif", path)
            else:
                self.right_edit.setText(path)
                self.right_gif_path = path
                self.settings.setValue("right_gif", path)

    # --- 核心业务逻辑 ---

    def start_trial(self):
        if self._running: return
        is_left = (self.task.currentIndex() == 0)

        # 参数
        fix_ms = int(self.fix.value() * 1000)
        cue_ms = int(self.cue.value() * 1000)
        imag_ms = int(self.imag.value() * 1000)
        rest_ms = int(self.rest.value() * 1000)

        # 循环初始化
        if self._loop_left == 0:
            self._loop_left = self.n_trials.value() if self.loop_switch.isChecked() else 1
            self._total_trials = self._loop_left
            self._iti_ms = int(self.iti.value() * 1000) if self.loop_switch.isChecked() else 0

        self._running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.task.setEnabled(False)  # 锁定任务选择
        self._clear_timers()

        # 阶段 1: 注视点
        self.info.emit("阶段: 注视点")
        self.stage.emit("注视点", 0)
        self.stage_bar.highlight(0)
        self.stim.show_fix()
        self.subtitle.setText("保持静止，注视屏幕中心...")

        # 创建记录
        rec = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "左手" if is_left else "右手", "", "",
               self.fix.value(), self.cue.value(), self.imag.value(), self.rest.value(), "", ""]
        self._records.append(rec)
        self._last_idx = len(self._records) - 1

        # 时间轴
        t1 = fix_ms
        t2 = t1 + cue_ms
        t3 = t2 + imag_ms
        t4 = t3 + rest_ms

        self._add_timer(t1, lambda: self._enter_cue(is_left))
        self._add_timer(t2, self._enter_imag)
        self._add_timer(t3, self._enter_rest)
        self._add_timer(t4, self._finish_one)

        self.trial_progress.emit(self._total_trials - self._loop_left + 1, self._total_trials)

    def _enter_cue(self, is_left):
        self.info.emit("阶段: 提示")
        self.stage.emit("方向提示", 1)
        self.stage_bar.highlight(1)
        self.stim.show_cue(is_left)
        self.subtitle.setText(f"提示: {'左' if is_left else '右'} (准备想象)")

    def _enter_imag(self):
        self.info.emit("阶段: 想象")
        self.stage.emit("运动想象", 2)
        self.stage_bar.highlight(2)

        is_left = (self.task.currentIndex() == 0)
        path = (self.left_edit.text() or self.left_gif_path) if is_left else \
            (self.right_edit.text() or self.right_gif_path)

        # 路径兜底
        if not path or not os.path.exists(path):
            path = LEFT_GIF_DEFAULT if is_left else RIGHT_GIF_DEFAULT

        self.stim.show_gif(path)
        self.subtitle.setText("开始运动想象 (Motor Imagery)...")

    def _enter_rest(self):
        self.info.emit("阶段: 休息")
        self.stage.emit("休息结束", 3)
        self.stage_bar.highlight(3)
        self.stim.show_rest("休息")
        self.subtitle.setText("放松...")

    def _finish_one(self):
        self._cnt_total += 1
        self._update_stats()
        self._loop_left -= 1
        self._running = False

        if self._loop_left > 0 and self.loop_switch.isChecked():
            self.stim.show_rest(f"下一次试次将在 {self.iti.value():.1f}s 后开始")
            self._add_timer(self._iti_ms, self.start_trial)
        else:
            self._reset_state()
            self.stim.show_rest("任务结束")
            self.subtitle.setText("所有试次已完成")
            self.info.emit("任务结束")

    def abort_trial(self):
        if not self._running: return
        self._reset_state()
        self._clear_timers()
        self.stim.show_rest("已中止")
        self.subtitle.setText("任务已手动中止")
        self.stage_bar.highlight(3)

    def _reset_state(self):
        self._running = False
        self._loop_left = 0
        self._total_trials = 0
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.task.setEnabled(True)

    def _add_timer(self, ms, func):
        t = QTimer(self)
        t.setSingleShot(True)
        t.timeout.connect(func)
        t.start(ms)
        self._timers.append(t)

    def _clear_timers(self):
        for t in self._timers:
            t.stop()
            t.deleteLater()
        self._timers.clear()

    def _update_stats(self):
        self.lbl_stats.setText(f"总完成: {self._cnt_total} | 成功: {self._cnt_succ}")

    def notify_trial_result(self, pred, success, intended):
        if self._last_idx is not None and self._last_idx < len(self._records):
            self._records[self._last_idx][2] = pred
            self._records[self._last_idx][3] = "是" if success else "否"

        if success:
            self._cnt_succ += 1
            self._update_stats()

    def notify_device_send(self, ok, msg):
        if self._last_idx is not None:
            self._records[self._last_idx][8] = "是" if ok else "否"
            self._records[self._last_idx][9] = msg

    def export_csv(self):
        if not self._records: return
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", "trials.csv", "CSV (*.csv)")
        if path:
            try:
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f)
                    w.writerow(["时间", "意图", "预测", "成功", "Fix", "Cue", "Imag", "Rest", "发送", "反馈"])
                    w.writerows(self._records)
                self.info.emit(f"导出成功: {path}")
            except Exception as e:
                self.info.emit(f"导出失败: {e}")