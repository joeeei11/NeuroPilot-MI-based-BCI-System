# -*- coding: utf-8 -*-
# login_dialog.py
# 登录界面 (Fluent Design 重构版)

import sys
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QWidget, QGraphicsDropShadowEffect
)

# 引入 Fluent Widgets
from qfluentwidgets import (
    LineEdit, PasswordLineEdit, CheckBox,
    PrimaryPushButton, PushButton,
    TitleLabel, BodyLabel,
    InfoBar, InfoBarPosition,
    IconWidget, FluentIcon as FIF,
    setTheme, Theme
)


class LoginDialog(QDialog):
    """
    Fluent 风格登录对话框
    无边框、阴影卡片、现代控件
    """
    info = pyqtSignal(str)

    def __init__(self, parent=None, fixed_user: str = "admin", fixed_pass: str = "123456"):
        super().__init__(parent)
        self.fixed_user = fixed_user
        self.fixed_pass = fixed_pass

        # 1. 窗口属性设置
        # 无边框 + 对话框模式
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        # 背景透明 (为了显示圆角和阴影)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(420, 560)

        # 拖拽移动支持
        self._is_dragging = False
        self._drag_pos = QPoint()

        self._init_ui()

    def _init_ui(self):
        """初始化 UI"""

        # 2. 主容器 (卡片)
        self.card = QWidget(self)
        self.card.setGeometry(10, 10, 400, 540)  # 留出边距给阴影
        self.card.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #E5E5E5;
                border-radius: 12px;
            }
        """)

        # 3. 阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 30))
        self.card.setGraphicsEffect(shadow)

        # 4. 垂直布局
        layout = QVBoxLayout(self.card)
        layout.setContentsMargins(40, 50, 40, 40)
        layout.setSpacing(16)

        # --- 图标区 ---
        self.icon_widget = IconWidget(FIF.EDUCATION)
        self.icon_widget.setFixedSize(64, 64)
        layout.addWidget(self.icon_widget, 0, Qt.AlignHCenter)

        layout.addSpacing(8)

        # --- 标题区 ---
        title = TitleLabel("NeuroPilot", self)
        subtitle = BodyLabel("运动想象上肢康复系统", self)
        subtitle.setTextColor(QColor(96, 96, 96), QColor(96, 96, 96))  # 灰色

        layout.addWidget(title, 0, Qt.AlignHCenter)
        layout.addWidget(subtitle, 0, Qt.AlignHCenter)

        layout.addSpacing(24)

        # --- 输入区 ---
        self.user_edit = LineEdit(self)
        self.user_edit.setPlaceholderText("请输入账号")
        self.user_edit.setClearButtonEnabled(True)
        # 支持回车跳转
        self.user_edit.returnPressed.connect(lambda: self.pass_edit.setFocus())

        self.pass_edit = PasswordLineEdit(self)
        self.pass_edit.setPlaceholderText("请输入密码")
        self.pass_edit.returnPressed.connect(self._try_login)

        layout.addWidget(self.user_edit)
        layout.addWidget(self.pass_edit)

        # --- 选项区 ---
        self.remember_chk = CheckBox("记住密码", self)
        layout.addWidget(self.remember_chk)

        layout.addSpacing(24)

        # --- 按钮区 ---
        self.btn_login = PrimaryPushButton("登录", self)
        self.btn_login.clicked.connect(self._try_login)
        self.btn_login.setFixedHeight(36)

        self.btn_cancel = PushButton("取消", self)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_cancel.setFixedHeight(36)

        layout.addWidget(self.btn_login)
        layout.addWidget(self.btn_cancel)
        layout.addStretch(1)

        # --- 底部提示 ---
        hint = BodyLabel(f"默认账号: {self.fixed_user} / {self.fixed_pass}", self)
        hint.setTextColor(QColor(150, 150, 150), QColor(150, 150, 150))
        hint.setFont(QFont("Microsoft YaHei", 9))
        layout.addWidget(hint, 0, Qt.AlignHCenter)

        # 默认焦点
        self.user_edit.setFocus()

    def _try_login(self):
        """验证逻辑"""
        u = self.user_edit.text().strip()
        p = self.pass_edit.text().strip()

        if u == self.fixed_user and p == self.fixed_pass:
            self.info.emit(f"登录成功: {u}")
            self.accept()
        else:
            # 使用 InfoBar 显示错误，而不是 Label
            InfoBar.error(
                title='登录失败',
                content="账号或密码错误，请重试。",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self.card
            )
            self.pass_edit.clear()
            self.pass_edit.setFocus()
            self.info.emit("登录失败: 密码错误")

    # --- 窗口拖拽逻辑 ---
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_dragging = True
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._is_dragging:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._is_dragging = False


# 独立测试
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    # 启用高 DPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    setTheme(Theme.LIGHT)

    w = LoginDialog()
    if w.exec_():
        print("Login Success")
    else:
        print("Login Cancelled")
    sys.exit()