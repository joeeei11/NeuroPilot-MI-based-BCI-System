# -*- coding: utf-8 -*-
# log_viewer.py
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QUrl
from PyQt5.QtGui import QFont, QTextCursor, QDesktopServices
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QPlainTextEdit, QFileDialog, QMessageBox
)
import os

DEFAULT_LOG = os.path.join("logs", "system_log.log")

class LogViewerWidget(QWidget):
    """系统日志查看器（实时 tail）"""
    new_line = pyqtSignal(str)

    def __init__(self, log_path=DEFAULT_LOG, parent=None):
        super().__init__(parent)
        self.log_path = log_path
        self._last_pos = 0
        self._paused = False
        self.setFont(QFont("Microsoft YaHei", 10))

        # 顶部工具条
        self.path_label = QLabel(f"日志：{os.path.abspath(self.log_path)}")
        self.level_combo = QComboBox(); self.level_combo.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR"])
        self.level_combo.setCurrentText("ALL")
        self.search_edit = QLineEdit(); self.search_edit.setPlaceholderText("关键词过滤（回车应用）")

        self.btn_refresh = QPushButton("刷新")
        self.btn_clear = QPushButton("清屏")
        self.btn_export = QPushButton("导出")
        self.btn_open = QPushButton("打开目录")
        self.btn_pause = QPushButton("暂停追踪")

        self.text = QPlainTextEdit(); self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.text.setStyleSheet("QPlainTextEdit{background:#0A0A0A;color:#EAEAEA;border-radius:10px;}")

        bar1 = QHBoxLayout(); bar1.addWidget(self.path_label); bar1.addStretch(1)
        bar2 = QHBoxLayout()
        bar2.addWidget(QLabel("等级：")); bar2.addWidget(self.level_combo)
        bar2.addSpacing(10)
        bar2.addWidget(QLabel("搜索：")); bar2.addWidget(self.search_edit, 1)
        bar2.addSpacing(10)
        for b in (self.btn_refresh, self.btn_clear, self.btn_export, self.btn_open, self.btn_pause):
            bar2.addWidget(b)

        root = QVBoxLayout()
        root.addLayout(bar1); root.addLayout(bar2); root.addWidget(self.text, 1)
        self.setLayout(root)

        self.setStyleSheet("""
            QWidget { background:#FFFFFF; color:#323232; }
            QLabel { font-weight:bold; }
            QLineEdit, QComboBox { border:1px solid #D0D0D0; border-radius:8px; padding:6px 10px; background:#F7F7F7; }
            QLineEdit:focus, QComboBox:focus { border:1px solid #007AFF; background:#FFFFFF; }
            QPushButton { background:#007AFF; color:#FFF; border:none; border-radius:10px; padding:8px 14px; font-weight:bold; }
            QPushButton:hover { background:#1A84FF; } QPushButton:pressed { background:#0062CC; }
        """)

        # 信号
        self.btn_refresh.clicked.connect(self._load_all)
        self.btn_clear.clicked.connect(self.text.clear)
        self.btn_export.clicked.connect(self._export)
        self.btn_open.clicked.connect(self._open_dir)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.search_edit.returnPressed.connect(self._apply_filter)
        self.level_combo.currentTextChanged.connect(self._apply_filter)

        self.timer = QTimer(self); self.timer.timeout.connect(self._tail); self.timer.start(600)
        self._load_all()

    def _open_dir(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(os.path.dirname(self.log_path))))

    def _toggle_pause(self):
        self._paused = not self._paused
        self.btn_pause.setText("继续追踪" if self._paused else "暂停追踪")

    def _export(self):
        if not os.path.exists(self.log_path):
            QMessageBox.information(self, "提示", "日志文件不存在。"); return
        fn, _ = QFileDialog.getSaveFileName(self, "导出日志", "system_log.txt", "文本文件 (*.txt)")
        if not fn: return
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="ignore") as f_in, \
                 open(fn, "w", encoding="utf-8") as f_out:
                f_out.write(f_in.read())
            QMessageBox.information(self, "完成", f"已导出：{fn}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{e}")

    def _load_all(self):
        self.text.clear()
        self._last_pos = 0
        self._tail(force_all=True)

    def _apply_filter(self):
        self._load_all()

    def _tail(self, force_all=False):
        if self._paused:
            return
        if not os.path.exists(self.log_path):
            if force_all and self.text.toPlainText().strip() == "":
                self.text.appendPlainText("（暂无日志，运行后自动生成 logs/system_log.log）")
            return
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="ignore") as f:
                if not force_all:
                    f.seek(self._last_pos)
                lines = f.readlines()
                self._last_pos = f.tell()
        except Exception:
            return

        if not lines: return

        level = self.level_combo.currentText()
        keyword = self.search_edit.text().strip()

        for s in lines:
            line = s.rstrip("\n")
            lvl = None
            # 解析格式：asctime - LEVEL - logger - message
            parts = line.split(" - ", 3)
            if len(parts) >= 3:
                lvl = parts[1].strip()

            if level != "ALL" and (lvl is None or lvl.upper() != level):
                continue
            if keyword and (keyword.lower() not in line.lower()):
                continue

            color = "#EAEAEA"
            if lvl == "ERROR": color = "#FF6B6B"
            elif lvl == "WARNING": color = "#FFD166"
            elif lvl == "INFO": color = "#A8E6CF"
            elif lvl == "DEBUG": color = "#B2EBF2"

            html = f'<pre style="margin:0;color:{color};">{self._esc(line)}</pre>'
            self.text.appendHtml(html)
            self.text.moveCursor(QTextCursor.End)
            self.new_line.emit(line)

    @staticmethod
    def _esc(t: str) -> str:
        return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
