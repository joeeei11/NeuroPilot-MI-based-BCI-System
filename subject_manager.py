
# -*- coding: utf-8 -*-
"""
subject_manager.py

受试者管理模块，实现一个独立的标签页，用于录入和维护受试者（病人）信息。
字段包括：
  - id: 自增主键
  - name: 姓名或编号
  - age: 年龄
  - gender: 性别（男/女/其他）
  - contact: 联系方式
  - dominant_hand: 惯用手（左/右/双）
  - onset_time: 发病时间（可自由输入）
"""
import sqlite3
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QFormLayout,
    QLineEdit, QSpinBox, QComboBox
)

class SubjectFormDialog(QDialog):
    """用于新增或编辑受试者信息的表单对话框"""
    def __init__(self, parent=None, subject=None):
        super().__init__(parent)
        self.setWindowTitle("受试者信息")
        self.setMinimumWidth(320)
        self.subject = subject or {}
        # 表单字段
        self.edit_name = QLineEdit(self.subject.get('name', ''))
        self.spin_age = QSpinBox(); self.spin_age.setRange(0, 120)
        if 'age' in self.subject:
            try:
                self.spin_age.setValue(int(self.subject['age']))
            except Exception:
                self.spin_age.setValue(0)
        self.cmb_gender = QComboBox(); self.cmb_gender.addItems(["男","女","其他"])
        if 'gender' in self.subject and self.subject['gender'] in ["男","女","其他"]:
            index = ["男","女","其他"].index(self.subject['gender'])
            self.cmb_gender.setCurrentIndex(index)
        self.edit_contact = QLineEdit(self.subject.get('contact', ''))
        self.cmb_hand = QComboBox(); self.cmb_hand.addItems(["左手","右手","双手"])
        if 'dominant_hand' in self.subject and self.subject['dominant_hand'] in ["左手","右手","双手"]:
            idx = ["左手","右手","双手"].index(self.subject['dominant_hand'])
            self.cmb_hand.setCurrentIndex(idx)
        self.edit_onset = QLineEdit(self.subject.get('onset_time', ''))

        # 布局
        form = QFormLayout()
        form.addRow("姓名/代号：", self.edit_name)
        form.addRow("年龄：", self.spin_age)
        form.addRow("性别：", self.cmb_gender)
        form.addRow("联系方式：", self.edit_contact)
        form.addRow("惯用手：", self.cmb_hand)
        form.addRow("发病时间：", self.edit_onset)

        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("保存")
        self.btn_cancel = QPushButton("取消")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def get_data(self):
        """返回表单数据"""
        return {
            'name': self.edit_name.text().strip(),
            'age': self.spin_age.value(),
            'gender': self.cmb_gender.currentText(),
            'contact': self.edit_contact.text().strip(),
            'dominant_hand': self.cmb_hand.currentText(),
            'onset_time': self.edit_onset.text().strip(),
        }

class SubjectManager(QWidget):
    """受试者管理面板，实现受试者列表展示和增删改功能"""
    def __init__(self, db_path="data.db", parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_table()
        self.setWindowTitle("受试者管理")
        self.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self._build_ui()
        self.load_subjects()

    def _create_table(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                age INTEGER,
                gender TEXT,
                contact TEXT,
                dominant_hand TEXT,
                onset_time TEXT
            )
        """)
        self.conn.commit()

    def _build_ui(self):
        # 主布局
        main_layout = QVBoxLayout()
        # 表格
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "ID", "姓名/代号", "年龄", "性别", "联系方式", "惯用手", "发病时间"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        # 按钮
        btn_box = QGroupBox("操作")
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("新增")
        self.btn_edit = QPushButton("编辑")
        self.btn_delete = QPushButton("删除")
        self.btn_add.clicked.connect(self.add_subject)
        self.btn_edit.clicked.connect(self.edit_subject)
        self.btn_delete.clicked.connect(self.delete_subject)
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_edit)
        btn_layout.addWidget(self.btn_delete)
        btn_layout.addStretch(1)
        btn_box.setLayout(btn_layout)
        # 组装
        main_layout.addWidget(self.table)
        main_layout.addWidget(btn_box)
        self.setLayout(main_layout)
        # 样式（沿用苹果风配色）
        self.setStyleSheet("""
            QWidget { background:#FFFFFF; color:#323232; font-size:14px; }
            QGroupBox { border:1px solid #E6E6E6; border-radius:12px; padding:10px; margin-top:10px; background:#FAFAFA; font-weight:bold; }
            QPushButton { background:#007AFF; color:#FFFFFF; padding:8px 16px; border-radius:10px; font-weight:bold; border:none; }
            QPushButton:hover { background:#1A84FF; }
            QPushButton:pressed { background:#0062CC; }
            QPushButton:disabled { background:#E0E0E0; color:#9E9E9E; }
            QTableWidget { border:1px solid #E6E6E6; border-radius:8px; }
        """)

    # --- 数据加载与刷新 ---
    def load_subjects(self):
        """从数据库加载所有受试者到表格"""
        c = self.conn.cursor()
        c.execute("SELECT id, name, age, gender, contact, dominant_hand, onset_time FROM subjects ORDER BY id")
        rows = c.fetchall()
        self.table.setRowCount(len(rows))
        for row_idx, row_data in enumerate(rows):
            for col_idx, value in enumerate(row_data):
                item = QTableWidgetItem(str(value) if value is not None else "")
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)  # 禁止直接编辑
                self.table.setItem(row_idx, col_idx, item)

    # --- 操作函数 ---
    def add_subject(self):
        dlg = SubjectFormDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            data = dlg.get_data()
            # 插入数据库
            c = self.conn.cursor()
            c.execute(
                "INSERT INTO subjects (name, age, gender, contact, dominant_hand, onset_time) VALUES (?,?,?,?,?,?)",
                (data['name'], data['age'], data['gender'], data['contact'], data['dominant_hand'], data['onset_time'])
            )
            self.conn.commit()
            self.load_subjects()

    def edit_subject(self):
        # 获取当前选中的行
        current_row = self.table.currentRow()
        if current_row < 0:
            return
        subject_id = int(self.table.item(current_row, 0).text())
        # 读取该受试者数据
        c = self.conn.cursor()
        c.execute("SELECT name, age, gender, contact, dominant_hand, onset_time FROM subjects WHERE id=?", (subject_id,))
        row = c.fetchone()
        if not row:
            return
        subj = {
            'id': subject_id,
            'name': row[0],
            'age': row[1],
            'gender': row[2],
            'contact': row[3],
            'dominant_hand': row[4],
            'onset_time': row[5]
        }
        dlg = SubjectFormDialog(self, subject=subj)
        if dlg.exec_() == QDialog.Accepted:
            data = dlg.get_data()
            c.execute(
                "UPDATE subjects SET name=?, age=?, gender=?, contact=?, dominant_hand=?, onset_time=? WHERE id=?",
                (data['name'], data['age'], data['gender'], data['contact'], data['dominant_hand'], data['onset_time'], subject_id)
            )
            self.conn.commit()
            self.load_subjects()

    def delete_subject(self):
        current_row = self.table.currentRow()
        if current_row < 0:
            return
        subject_id = int(self.table.item(current_row, 0).text())
        c = self.conn.cursor()
        c.execute("DELETE FROM subjects WHERE id=?", (subject_id,))
        self.conn.commit()
        self.load_subjects()
