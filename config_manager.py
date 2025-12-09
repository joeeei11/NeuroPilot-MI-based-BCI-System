# -*- coding: utf-8 -*-
# core/config_manager.py
# 全局配置管理器 (Configuration Persistence Layer)
# 职责：统一管理所有模块的持久化参数，防止重启后设置丢失。
# 架构：基于 PyQt5.QSettings 的单例封装

import threading
from PyQt5.QtCore import QSettings
from typing import Any, Optional


class ConfigManager:
    """
    配置管理单例类。
    使用 QSettings (注册表/INI) 进行持久化存储。

    Usage:
        cfg = ConfigManager()
        cfg.set("EEG", "port", "COM3")
        port = cfg.get("EEG", "port", "COM1")
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        # 组织名称, 应用名称 (决定注册表路径或INI文件名)
        self.settings = QSettings("NeuroPilot", "SystemConfig")
        self._initialized = True

    def set(self, section: str, key: str, value: Any):
        """
        保存配置项
        :param section: 模块名 (如 'EEG', 'Device', 'ML')
        :param key: 参数名
        :param value: 参数值 (支持 int, float, bool, str)
        """
        self.settings.beginGroup(section)
        self.settings.setValue(key, value)
        self.settings.endGroup()
        # 立即同步到磁盘，防止崩溃丢失
        self.settings.sync()

    def get(self, section: str, key: str, default: Any = None, type_hint: type = None) -> Any:
        """
        读取配置项
        :param section: 模块名
        :param key: 参数名
        :param default: 默认值 (当键不存在时返回)
        :param type_hint: 强制类型转换 (如 int, bool, float)
        :return: 配置值
        """
        self.settings.beginGroup(section)

        # QSettings.value() 第二个参数是默认值
        val = self.settings.value(key, default)
        self.settings.endGroup()

        if val is None:
            return default

        # 智能类型转换
        try:
            if type_hint is not None:
                if type_hint == bool:
                    # 处理布尔值的特殊情况 (QSettings 可能存为 'true'/'false')
                    if isinstance(val, str):
                        return val.lower() == 'true'
                    return bool(val)
                elif type_hint == int:
                    return int(val)
                elif type_hint == float:
                    return float(val)
                elif type_hint == str:
                    return str(val)
                else:
                    return type_hint(val)

            # 如果没有 type_hint，QSettings 通常会自动推断，但有时会返回字符串
            # 这里做一个简单的自动修正
            if isinstance(default, bool) and isinstance(val, str):
                return val.lower() == 'true'
            if isinstance(default, int) and isinstance(val, str):
                return int(val)
            if isinstance(default, float) and isinstance(val, str):
                return float(val)

        except Exception:
            # 转换失败则返回默认值
            return default

        return val

    def clear_section(self, section: str):
        """清空某个模块的所有配置"""
        self.settings.beginGroup(section)
        self.settings.remove("")
        self.settings.endGroup()


# 全局单例实例，供外部直接 import
# from core.config_manager import cfg
cfg = ConfigManager()