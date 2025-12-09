# -*- coding: utf-8 -*-
# core/data_manager.py
# 统一数据管理层：负责 SQLite 数据库交互、文件路径生成、数据持久化
# 纯 Python 实现，无 GUI 依赖，线程安全

import os
import sqlite3
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any


class DataManager:
    """
    数据管理单例类 (Singleton)。
    集中管理受试者信息、实验记录 (Trials) 和原始数据文件路径。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(DataManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_dir="data", db_name="neuro_pilot.db"):
        if getattr(self, "_initialized", False):
            return

        # 初始化目录
        self.root_dir = db_dir
        self.eeg_dir = os.path.join(self.root_dir, "raw_eeg")
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.eeg_dir, exist_ok=True)

        # 初始化数据库
        self.db_path = os.path.join(self.root_dir, db_name)
        # check_same_thread=False 允许在多线程中使用同一个连接（配合锁使用）
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # 将 row_factory 设置为 Row，以便像字典一样访问列
        self.conn.row_factory = sqlite3.Row

        self._create_tables()
        self._initialized = True

    def _create_tables(self):
        """初始化表结构"""
        with self._lock:
            c = self.conn.cursor()

            # 1. 受试者表
            c.execute("""
                CREATE TABLE IF NOT EXISTS subjects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER,
                    gender TEXT,
                    contact TEXT,
                    dominant_hand TEXT,
                    onset_time TEXT,
                    created_at TEXT
                )
            """)

            # 2. 试次记录表 (Trials)
            c.execute("""
                CREATE TABLE IF NOT EXISTS trials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,           -- 试次发生时间
                    session_id TEXT,          -- 会话ID (通常是日期)
                    subject_id INTEGER,       -- 关联 subjects.id
                    subject_name TEXT,        -- 冗余存储方便查询
                    intended_label TEXT,      -- 真实意图 (左/右)
                    predicted_label TEXT,     -- 模型预测
                    is_success INTEGER,       -- 1=成功, 0=失败

                    -- 时间参数 (秒)
                    fix_duration REAL,
                    cue_duration REAL,
                    imag_duration REAL,
                    rest_duration REAL,

                    -- 外设状态
                    send_status INTEGER,      -- 1=发送成功, 0=失败
                    device_msg TEXT,          -- 设备反馈信息

                    -- 原始数据关联
                    raw_file_path TEXT        -- 关联的 EEG CSV 文件路径
                )
            """)
            self.conn.commit()

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    # ======================================================
    # 原始数据文件管理 (File Paths)
    # ======================================================

    def get_new_eeg_file_path(self, subject_name: str, session_id: str = None) -> str:
        """
        生成规范化的 EEG 原始数据保存路径。
        格式: data/raw_eeg/{subject}_{session}_{timestamp}.csv
        """
        if not session_id:
            session_id = datetime.now().strftime("%Y%m%d")

        timestamp = datetime.now().strftime("%H%M%S")
        # 清理文件名中的非法字符
        safe_name = "".join([c for c in subject_name if c.isalnum() or c in (' ', '_')]).strip().replace(' ', '_')

        filename = f"{safe_name}_{session_id}_{timestamp}.csv"
        return os.path.join(self.eeg_dir, filename)