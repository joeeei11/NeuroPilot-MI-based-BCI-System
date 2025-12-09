# -*- coding: utf-8 -*-
# core/dsp.py
# 纯数学/信号处理函数库 (Pure Python/Numpy/Scipy)
# 严禁引入 PyQt5

import numpy as np
from scipy import signal


def butter_filter(data: np.ndarray, fs: float, f_low=None, f_high=None, order=4, axis=0) -> np.ndarray:
    """
    通用 Butterworth 滤波器 (带通/低通/高通)。

    Args:
        data: 输入数据 (n_samples, n_channels) 或 (n_samples,)
        fs: 采样率 (Hz)
        f_low: 低截止频率 (Hz)，若为 None 则可能为低通
        f_high: 高截止频率 (Hz)，若为 None 则可能为高通
        order: 滤波器阶数
        axis: 沿哪个轴滤波 (默认0，即沿时间轴)

    Returns:
        滤波后的数据，形状同 data
    """
    if fs <= 0 or data is None or data.size == 0:
        return data

    nyq = 0.5 * fs
    btype = 'band'
    Wn = None

    # 逻辑判定：带通、低通、高通
    if f_low and f_high:
        if f_low >= f_high or f_high >= nyq:
            return data  # 参数不合法，原样返回
        btype = 'band'
        Wn = [f_low / nyq, f_high / nyq]

    elif f_high:
        if f_high >= nyq:
            return data
        btype = 'low'
        Wn = f_high / nyq

    elif f_low:
        if f_low >= nyq:
            return data
        btype = 'high'
        Wn = f_low / nyq

    else:
        # 未指定频率，原样返回
        return data

    try:
        b, a = signal.butter(order, Wn, btype=btype)
        return signal.filtfilt(b, a, data, axis=axis)
    except Exception:
        # 兜底：若滤波失败（如数据过短），原样返回
        return data


def notch_filter(data: np.ndarray, fs: float, freq=50.0, Q=30.0, axis=0) -> np.ndarray:
    """
    IIR 陷波滤波器 (去除工频干扰)。

    Args:
        data: 输入数据
        fs: 采样率
        freq: 陷波中心频率 (默认 50Hz)
        Q: 质量因子 (越高越窄)
    """
    if fs <= 0 or data is None or data.size == 0:
        return data

    w0 = freq / (fs / 2.0)
    if w0 >= 1.0 or w0 <= 0:
        return data

    try:
        b, a = signal.iirnotch(w0, Q)
        return signal.filtfilt(b, a, data, axis=axis)
    except Exception:
        return data


def compute_psd(data: np.ndarray, fs: float, nperseg=512, axis=0):
    """
    计算功率谱密度 (Welch方法)。

    Returns:
        f: 频率轴
        pxx: 功率谱密度
    """
    if data is None or data.size == 0:
        return np.array([]), np.array([])

    # 自动调整 nperseg，防止数据过短报错
    n_samples = data.shape[axis] if data.ndim > 0 else len(data)
    if nperseg > n_samples:
        nperseg = n_samples

    try:
        f, pxx = signal.welch(data, fs=fs, nperseg=nperseg, axis=axis)
        return f, pxx
    except Exception:
        return np.array([]), np.array([])