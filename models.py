# -*- coding: utf-8 -*-
# core/models.py
# 机器学习模型与特征提取库
# 严禁引入 PyQt5

import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(BaseEstimator, TransformerMixin):
    """
    共空间模式 (Common Spatial Pattern) 滤波器。
    支持两种后端实现：
    1. 'vectorized' (默认): 基于 numpy 矩阵运算，速度快，代码简洁。
    2. 'loop' (兼容): 对应原 CSP_2.py 的实现，基于循环计算协方差。

    属性:
        filters_ (m_filters * 2, n_channels): 空间滤波器矩阵
    """

    def __init__(self, n_components=4, backend='vectorized'):
        """
        Args:
            n_components: 保留的特征分量总数 (通常为偶数，如 4 表示取首2和尾2)
            backend: 'vectorized' 或 'loop'
        """
        self.n_components = n_components
        self.backend = backend
        self.filters_ = None
        # 兼容 sklearn 习惯
        self.patterns_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y):
        """
        训练 CSP 模型。

        Args:
            X: 训练数据, 形状 (n_trials, n_channels, n_samples)
            y: 标签, 形状 (n_trials,), 必须包含两个不同的类别 (如 0 和 1)
        """
        # 检查输入
        if X.ndim != 3:
            raise ValueError("CSP fit expects 3D input: (n_trials, n_channels, n_samples)")

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP requires exactly 2 classes.")

        # 路由到具体实现
        if self.backend == 'loop':
            self._fit_loop(X, y, classes)
        else:
            self._fit_vectorized(X, y, classes)

        return self

    def transform(self, X):
        """
        提取特征 (Log-Variance)。

        Args:
            X: 数据, 形状 (n_trials, n_channels, n_samples)
               或者单次数据 (n_channels, n_samples)

        Returns:
            features: 形状 (n_trials, n_components)
        """
        if self.filters_ is None:
            raise RuntimeError("CSP not fitted yet!")

        # 兼容单次输入 (n_channels, n_samples) -> 扩维
        input_is_2d = False
        if X.ndim == 2:
            X = X[np.newaxis, :, :]
            input_is_2d = True

        n_trials = X.shape[0]
        feats = np.zeros((n_trials, self.n_components))

        # 投影: Z = W.T * X  (这里 W=filters_.T)
        # filters_ shape: (n_comps, n_channels)
        # X[i] shape: (n_channels, n_samples)

        for i in range(n_trials):
            # 空间滤波 -> (n_components, n_samples)
            Z = np.dot(self.filters_, X[i])
            # 计算方差 -> (n_components,)
            var = np.var(Z, axis=1)
            # Log 变换并归一化
            feats[i] = np.log(var / np.sum(var))

        return feats

    def _fit_vectorized(self, X, y, classes):
        """基于矩阵运算的高效实现"""
        n_channels = X.shape[1]

        # 计算两类的平均协方差矩阵
        covs = []
        for cls in classes:
            X_cls = X[y == cls]
            # 计算该类所有试次的协方差并平均
            # cov(X) = (X * X.T) / trace
            C_sum = np.zeros((n_channels, n_channels))
            for i in range(len(X_cls)):
                trial = X_cls[i]
                # 均值移除 (虽通常预处理做过，但为了保险)
                trial = trial - trial.mean(axis=1, keepdims=True)
                c = np.dot(trial, trial.T)
                c /= np.trace(c)  # 归一化 trace
                C_sum += c
            covs.append(C_sum / len(X_cls))

        cov0, cov1 = covs[0], covs[1]
        cov_combined = cov0 + cov1

        # 广义特征值分解: cov0 * w = lambda * (cov0 + cov1) * w
        # 这里的实现等价于求解 P * cov0 * P.T 的特征值，其中 P 是白化矩阵
        # 为了简单，直接用 scipy.linalg.eigh 求解广义特征值问题
        # Ax = lambda Bx
        eigvals, eigvecs = scipy.linalg.eigh(cov0, cov_combined)

        # eigh 返回的是升序，我们需要降序 (最大特征值对应第一类，最小特征值对应第二类)
        # 同时要取两端
        # sorted indices descending
        ix = np.argsort(np.abs(eigvals))[::-1]
        eigvecs = eigvecs[:, ix]

        # 取滤波器: 前 m 个和 后 m 个
        m = self.n_components // 2
        filters_top = eigvecs[:, :m]
        filters_bot = eigvecs[:, -m:]

        # 组合并转置，使得 filters_ 的形状为 (n_components, n_channels)
        self.filters_ = np.concatenate([filters_top, filters_bot], axis=1).T

    def _fit_loop(self, X, y, classes):
        """
        基于 CSP_2.py 的逻辑复刻
        注：原 CSP_2.py 的 fit 逻辑中混合了 loop 和 linalg，这里进行了清理
        """
        n_trials, n_channels, n_samples = X.shape
        cov_x = np.zeros((2, n_channels, n_channels))

        # 1. 累加协方差
        for i in range(n_trials):
            x_trial = X[i]
            y_trial = 0 if y[i] == classes[0] else 1  # 映射到 0/1 索引

            c = np.dot(x_trial, x_trial.T)
            c /= np.trace(c)
            cov_x[y_trial] += c

        # 2. 平均
        # 计算每个类别的数量
        n0 = np.sum(y == classes[0])
        n1 = np.sum(y == classes[1])
        cov_x[0] /= n0
        cov_x[1] /= n1

        cov_combined = cov_x[0] + cov_x[1]

        # 3. 特征分解 (scipy.linalg.eig)
        # 原代码：scipy.linalg.eig(cov_combined, cov_x[0]) 这里的物理意义有点反直觉
        # 通常是 eig(cov_0, cov_combined) 或者 先白化 cov_combined
        # 但为了保持和 CSP_2.py 逻辑一致，我们尽量还原它的数学过程
        # CSP_2.py: eig(cov_combined, cov_x[0]) -> 这实际上是在求 (cov_x[0])^-1 * cov_combined 的特征值

        # 修正：为了保证效果，这里建议还是使用标准的广义特征分解
        # 如果必须严格复刻 CSP_2.py 的特殊写法：
        vals, vecs = scipy.linalg.eig(cov_combined, cov_x[0])

        # 排序
        sort_indices = np.argsort(np.abs(vals))[::-1]
        vecs = vecs[:, sort_indices]

        # CSP_2.py 取的是 transpose
        u_mat = np.transpose(vecs)

        # 选择特征向量 (首 m 个 和 尾 m 个)
        # CSP_2.py 的 transform 逻辑略显复杂，这里我们将其统一到标准的 filters 格式
        # 原 transform 逻辑是：取 u_mat 前 m 行 和 后 m 行
        m = self.n_components // 2

        # 注意：因为上面 eig 的参数顺序问题，这里的特征值含义可能与标准相反
        # 但 CSP 本质是寻找差异最大化，只要取两头即可
        f1 = u_mat[:m, :]
        f2 = u_mat[-m:, :]

        self.filters_ = np.concatenate([f1, f2], axis=0)