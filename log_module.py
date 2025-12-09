# -*- coding: utf-8 -*-
# log_module.py
#
# 系统日志面板（最小侵入接入）：
# - 提供 GUI 日志查看（按级别过滤、搜索关键字、自动滚动、清空、导出、打开日志目录）
# - 提供 attach_python_logging(logger) 方法，把 Python logging 接入到面板
# - 提供 append_record(source, level, message) 方法，供主程序把各模块 info/事件回灌到面板
#
# 使用：在 main.py 里创建 LogPanel()，作为一个新标签加入 QTabWidget；
#      初始化 Python 日志 logger，并调用 log_panel.attach_python_logging(logger)。
#      同时在你的回调 on_info/on_stage/on_trial_result/on_device_* 中调用 self._log.info(...)
#      和 self.log_page.append_record(...)

import os
import logging
from datetime import datetime

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QLabel,
    QComboBox, QLineEdit, QPushButton, QTextEdit, QFileDialog, QMessageBox
)

APPLE_BLUE = "#007AFF"

class QtLogHandler(logging.Handler):
    """把 Python logging 记录转发到 LogPanel"""
    def __init__(self, panel):
        super().__init__()
        self.panel = panel

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.panel.append_record(
                source=getattr(record, "name", "Logger"),
                level=record.levelname,
                message=msg
            )
        except Exception:
            pass


class LogPanel(QWidget):
    """系统日志查看器"""
    info = pyqtSignal(str)

    def __init__(self, log_dir: str = "logs", log_file: str = "neuro_pilot.log", parent=None):
        super().__init__(parent)
        self.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.log_dir = log_dir
        self.log_file = log_file
        os.makedirs(self.log_dir, exist_ok=True)
        self._records = []  # [(ts, source, level, message)]
        self._auto_scroll = True
        self._attached_handler = None

        self._build_ui()
        self._apply_styles()

    # ---------- UI ----------
    def _build_ui(self):
        ctrl = QGroupBox("日志控制")
        g = QGridLayout()

        self.level_combo = QComboBox()
        self.level_combo.addItems(["ALL", "INFO", "WARNING", "ERROR", "DEBUG"])
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("输入关键字过滤（模块名/文本）")
        self.btn_clear = QPushButton("清空")
        self.btn_export = QPushButton("导出")
        self.btn_open_dir = QPushButton("打开日志目录")
        self.btn_toggle_scroll = QPushButton("自动滚动：开")
        self.btn_toggle_scroll.setCheckable(True)
        self.btn_toggle_scroll.setChecked(True)

        g.addWidget(QLabel("级别"), 0, 0);     g.addWidget(self.level_combo, 0, 1)
        g.addWidget(QLabel("搜索"), 0, 2);     g.addWidget(self.search_edit, 0, 3, 1, 3)
        g.addWidget(self.btn_clear, 0, 6)
        g.addWidget(self.btn_export, 0, 7)
        g.addWidget(self.btn_open_dir, 0, 8)
        g.addWidget(self.btn_toggle_scroll, 0, 9)
        ctrl.setLayout(g)

        self.text = QTextEdit()
        self.text.setReadOnly(True)

        root = QVBoxLayout()
        root.addWidget(ctrl)
        root.addWidget(self.text, 1)
        self.setLayout(root)

        # 事件
        self.level_combo.currentIndexChanged.connect(self._refresh_view)
        self.search_edit.textChanged.connect(self._refresh_view)
        self.btn_clear.clicked.connect(self.clear)
        self.btn_export.clicked.connect(self.export_logs)
        self.btn_open_dir.clicked.connect(self.open_dir)
        self.btn_toggle_scroll.clicked.connect(self._toggle_scroll)

    def _apply_styles(self):
        self.setStyleSheet(f"""
            QWidget {{
                background: #FFFFFF;
                color: #323232;
                font-family: "Microsoft YaHei","微软雅黑",Arial;
                font-size: 14px;
            }}
            QGroupBox {{
                border: 1px solid #E6E6E6;
                border-radius: 12px;
                padding: 8px;
                margin-top: 8px;
                background: #FAFAFA;
                font-weight: bold;
            }}
            QLineEdit, QComboBox {{
                border: 1px solid #D0D0D0;
                border-radius: 8px;
                padding: 6px 10px;
                background: #F7F7F7;
                min-width: 140px;
            }}
            QLineEdit:focus, QComboBox:focus {{
                border: 1px solid {APPLE_BLUE};
                background: #FFFFFF;
            }}
            QPushButton {{
                background: {APPLE_BLUE};
                color: #FFF;
                padding: 8px 14px;
                border-radius: 10px;
                font-weight: bold;
                border: none;
                min-width: 100px;
            }}
            QPushButton:hover {{ background: #1A84FF; }}
            QPushButton:pressed {{ background: #0062CC; }}
        """)

    # ---------- 公共方法 ----------
    def attach_python_logging(self, logger: logging.Logger, level=logging.INFO):
        """把 Python logging 接入到面板，同时返回该 handler 以便外部管理"""
        if self._attached_handler is not None:
            try:
                logger.removeHandler(self._attached_handler)
            except Exception:
                pass
            self._attached_handler = None

        handler = QtLogHandler(self)
        handler.setLevel(level)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        self._attached_handler = handler
        self.info.emit("系统日志：已接入 Python logging")

    def append_record(self, source: str, level: str, message: str):
        """外部回灌一条日志"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._records.append((ts, source or "Unknown", level.upper(), message.strip()))
        # 追加到文本框（按当前过滤设置）
        if self._pass_filter(self._records[-1]):
            self._append_to_view(self._records[-1])

    def clear(self):
        self._records.clear()
        self.text.clear()
        self.info.emit("系统日志：已清空")

    def export_logs(self):
        """导出当前筛选结果为文本文件"""
        path, _ = QFileDialog.getSaveFileName(self, "导出日志", "neuro_pilot_log.txt", "Text Files (*.txt)")
        if not path:
            return
        to_write = []
        for rec in self._records:
            if self._pass_filter(rec):
                to_write.append(self._format(rec))
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(to_write))
            QMessageBox.information(self, "完成", f"日志已导出到：{path}")
        except Exception as e:
            QMessageBox.critical(self, "失败", f"导出失败：{e}")

    def open_dir(self):
        """打开日志目录（仅提示路径，避免依赖平台差异）"""
        absdir = os.path.abspath(self.log_dir)
        QMessageBox.information(self, "日志目录", f"日志目录路径：\n{absdir}")

    # ---------- 内部 ----------
    def _toggle_scroll(self):
        self._auto_scroll = not self._auto_scroll
        self.btn_toggle_scroll.setChecked(self._auto_scroll)
        self.btn_toggle_scroll.setText(f"自动滚动：{'开' if self._auto_scroll else '关'}")

    def _format(self, rec):
        ts, source, level, message = rec
        return f"{ts} [{level}] {source}: {message}"

    def _append_to_view(self, rec):
        self.text.append(self._format(rec))
        if self._auto_scroll:
            self.text.moveCursor(QTextCursor.End)

    def _pass_filter(self, rec):
        ts, source, level, message = rec
        lv = self.level_combo.currentText().upper()
        kw = self.search_edit.text().strip().lower()
        # 级别过滤
        if lv != "ALL" and level != lv:
            return False
        # 关键字过滤
        if kw:
            text = f"{ts} {source} {level} {message}".lower()
            if kw not in text:
                return False
        return True

    def _refresh_view(self):
        self.text.clear()
        for rec in self._records:
            if self._pass_filter(rec):
                self._append_to_view(rec)
