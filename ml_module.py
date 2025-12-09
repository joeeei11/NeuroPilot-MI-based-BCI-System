# -*- coding: utf-8 -*-
# ml_module.py
# 模型训练与优化 (Fluent Design + Config Persistence)
# 职责：离线训练、网格搜索、交叉验证、模型对比
# 状态：Final Release

import os
import io
import pickle
import logging
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
)

# --- Fluent Widgets ---
from qfluentwidgets import (
    SmoothScrollArea, CardWidget, SimpleCardWidget, ElevatedCardWidget,
    PrimaryPushButton, PushButton, CheckBox,
    ComboBox, DoubleSpinBox, SpinBox, LineEdit, TextEdit,
    TitleLabel, CaptionLabel, StrongBodyLabel, BodyLabel,
    FluentIcon as FIF, IconWidget, InfoBar, InfoBarPosition
)

# --- Config Manager (Phase 12) ---
from core.config_manager import cfg

# --- Matplotlib ---
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Segoe UI", "Microsoft YaHei", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Scikit-learn ---
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA


def _parse_param_grid(grid_text: str):
    """解析简易网格字符串 -> dict"""
    grid = {}
    if not grid_text or not grid_text.strip():
        return grid
    parts = grid_text.split(";")
    for p in parts:
        p = p.strip()
        if not p or "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        vals = []
        for token in v.split(","):
            token = token.strip()
            if token == "": continue
            try:
                if "." in token:
                    vals.append(float(token))
                else:
                    vals.append(int(token))
            except:
                vals.append(token)
        if vals:
            grid[k] = vals
    return grid


class MplCanvas(FigureCanvas):
    """
    2x2 子图画布，适配 Fluent Light 主题
    """

    def __init__(self, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.fig.patch.set_facecolor('white')

        self.ax_cm = self.fig.add_subplot(2, 2, 1)
        self.ax_roc = self.fig.add_subplot(2, 2, 2)
        self.ax_lc = self.fig.add_subplot(2, 2, 3)
        self.ax_cmp = self.fig.add_subplot(2, 2, 4)

        super().__init__(self.fig)


class MLTrainerPanel(QWidget):
    """
    模型训练与优化面板
    """
    info = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MLTrainer")
        self._log = logging.getLogger("NeuroPilot.ML")

        self.df = None
        self.X = None
        self.y = None
        self.classes_ = None
        self.model = None
        self._last_split = None

        self._init_ui()

        # 核心：加载上次配置
        self._load_settings()

    def _init_ui(self):
        # 1. 根布局：SmoothScrollArea
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = SmoothScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: transparent; border: none;")

        self.content_widget = QWidget()
        self.scroll_area.setWidget(self.content_widget)
        main_layout.addWidget(self.scroll_area)

        # 2. 内容布局
        self.v_layout = QVBoxLayout(self.content_widget)
        self.v_layout.setContentsMargins(24, 24, 24, 24)
        self.v_layout.setSpacing(20)

        # ==========================================
        # Header
        # ==========================================
        self.header = SimpleCardWidget()
        h_header = QHBoxLayout(self.header)
        h_header.setContentsMargins(24, 16, 24, 16)

        icon = IconWidget(FIF.EDUCATION)
        icon.setFixedSize(40, 40)

        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title_box.addWidget(TitleLabel("机器学习模型训练", self))
        title_box.addWidget(CaptionLabel("基于 Scikit-learn 的离线训练、评估与优化流程", self))

        h_header.addWidget(icon)
        h_header.addLayout(title_box)
        h_header.addStretch(1)

        self.v_layout.addWidget(self.header)

        # ==========================================
        # Section A: 数据配置 (Card)
        # ==========================================
        self.data_card = CardWidget()
        l_data = QVBoxLayout(self.data_card)
        l_data.setContentsMargins(20, 16, 20, 16)
        l_data.setSpacing(12)

        l_data.addWidget(StrongBodyLabel("📁 数据源配置", self))

        # Row 1
        row_d1 = QHBoxLayout()
        self.ed_target = LineEdit()
        self.ed_target.setPlaceholderText("目标列名 (例如 label)")
        self.ed_target.setFixedWidth(120)

        self.ed_features = LineEdit()
        self.ed_features.setPlaceholderText("特征列 (留空自动识别)")

        row_d1.addWidget(CaptionLabel("目标列:", self))
        row_d1.addWidget(self.ed_target)
        row_d1.addWidget(CaptionLabel("特征列:", self))
        row_d1.addWidget(self.ed_features)
        l_data.addLayout(row_d1)

        # Row 2
        row_d2 = QHBoxLayout()
        self.split_spin = DoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setSingleStep(0.05)
        self.split_spin.setFixedWidth(100)

        self.btn_load_csv = PushButton(FIF.FOLDER, "导入 CSV", self)
        self.btn_demo = PushButton(FIF.IOT, "生成演示数据", self)
        self.btn_preview = PushButton(FIF.VIEW, "预览数据", self)

        row_d2.addWidget(CaptionLabel("测试集比例:", self))
        row_d2.addWidget(self.split_spin)
        row_d2.addStretch(1)
        row_d2.addWidget(self.btn_load_csv)
        row_d2.addWidget(self.btn_demo)
        row_d2.addWidget(self.btn_preview)
        l_data.addLayout(row_d2)

        self.v_layout.addWidget(self.data_card)

        # ==========================================
        # Section B: 算法与特征 (Split Layout)
        # ==========================================
        row_algo = QHBoxLayout()
        row_algo.setSpacing(20)

        # --- B1. 算法参数 ---
        self.algo_card = CardWidget()
        l_algo = QVBoxLayout(self.algo_card)
        l_algo.setContentsMargins(20, 16, 20, 16)
        l_algo.setSpacing(12)
        l_algo.addWidget(StrongBodyLabel("⚙️ 算法配置", self))

        self.cmb_algo = ComboBox()
        self.cmb_algo.addItems(["SVM (RBF)", "SVM (Linear)", "KNN", "LogisticRegression", "RandomForest"])
        self.cmb_algo.currentIndexChanged.connect(self._on_algo_changed)

        self.ed_grid = LineEdit()
        self.ed_grid.setPlaceholderText("参数网格 (Grid Search)")

        row_cv = QHBoxLayout()
        self.cv_spin = SpinBox()
        self.cv_spin.setRange(2, 20)
        self.cv_spin.setFixedWidth(100)

        self.cmb_score = ComboBox()
        self.cmb_score.addItems(["accuracy", "f1_macro", "roc_auc_ovr"])

        row_cv.addWidget(CaptionLabel("CV折数:", self))
        row_cv.addWidget(self.cv_spin)
        row_cv.addWidget(CaptionLabel("评分:", self))
        row_cv.addWidget(self.cmb_score)

        l_algo.addWidget(CaptionLabel("分类器模型:", self))
        l_algo.addWidget(self.cmb_algo)
        l_algo.addWidget(CaptionLabel("参数网格 (分号分隔):", self))
        l_algo.addWidget(self.ed_grid)
        l_algo.addLayout(row_cv)
        l_algo.addStretch(1)

        row_algo.addWidget(self.algo_card, 1)

        # --- B2. 特征工程 ---
        self.feat_card = CardWidget()
        l_feat = QVBoxLayout(self.feat_card)
        l_feat.setContentsMargins(20, 16, 20, 16)
        l_feat.setSpacing(12)
        l_feat.addWidget(StrongBodyLabel("🔧 特征工程", self))

        self.chk_standardize = CheckBox("标准化 (StandardScaler)")

        # KBest
        row_k = QHBoxLayout()
        self.chk_kbest = CheckBox("SelectKBest")
        self.cmb_kbest_score = ComboBox()
        self.cmb_kbest_score.addItems(["f_classif", "mutual_info"])
        self.spin_k = SpinBox()
        self.spin_k.setRange(1, 9999)
        row_k.addWidget(self.chk_kbest)
        row_k.addWidget(self.cmb_kbest_score)
        row_k.addWidget(CaptionLabel("k=", self))
        row_k.addWidget(self.spin_k)

        # PCA
        row_pca = QHBoxLayout()
        self.chk_pca = CheckBox("PCA 降维")
        self.spin_pca = SpinBox()
        self.spin_pca.setRange(1, 9999)
        row_pca.addWidget(self.chk_pca)
        row_pca.addWidget(CaptionLabel("n_comp=", self))
        row_pca.addWidget(self.spin_pca)

        l_feat.addWidget(self.chk_standardize)
        l_feat.addLayout(row_k)
        l_feat.addLayout(row_pca)
        l_feat.addStretch(1)

        row_algo.addWidget(self.feat_card, 1)

        self.v_layout.addLayout(row_algo)

        # ==========================================
        # Section C: 训练控制 (Card)
        # ==========================================
        self.train_card = CardWidget()
        l_train = QHBoxLayout(self.train_card)
        l_train.setContentsMargins(24, 20, 24, 20)
        l_train.setSpacing(16)

        self.btn_train = PrimaryPushButton(FIF.PLAY, "开始训练", self)
        self.btn_train.setFixedWidth(140)

        # FIXED: Use valid icon FIF.MARKET for charts
        self.btn_lc = PushButton(FIF.MARKET, "绘制学习曲线", self)

        self.btn_save = PushButton(FIF.SAVE, "保存模型", self)
        self.btn_load = PushButton(FIF.FOLDER, "加载模型", self)

        l_train.addWidget(self.btn_train)
        l_train.addWidget(self.btn_lc)
        l_train.addStretch(1)
        l_train.addWidget(self.btn_save)
        l_train.addWidget(self.btn_load)

        self.v_layout.addWidget(self.train_card)

        # ==========================================
        # Section D: 批量对比 (Card)
        # ==========================================
        self.cmp_card = CardWidget()
        l_cmp = QVBoxLayout(self.cmp_card)
        l_cmp.setContentsMargins(20, 16, 20, 16)

        l_cmp.addWidget(StrongBodyLabel("⚖️ 批量算法对比 (可选)", self))

        row_cmp = QHBoxLayout()
        self.chk_cmp_svm_rbf = CheckBox("SVM-RBF")
        self.chk_cmp_svm_lin = CheckBox("SVM-Lin")
        self.chk_cmp_knn = CheckBox("KNN")
        self.chk_cmp_lr = CheckBox("LogReg")
        self.chk_cmp_rf = CheckBox("RF")

        self.btn_cmp = PrimaryPushButton(FIF.SYNC, "运行对比", self)

        row_cmp.addWidget(self.chk_cmp_svm_rbf)
        row_cmp.addWidget(self.chk_cmp_svm_lin)
        row_cmp.addWidget(self.chk_cmp_knn)
        row_cmp.addWidget(self.chk_cmp_lr)
        row_cmp.addWidget(self.chk_cmp_rf)
        row_cmp.addStretch(1)
        row_cmp.addWidget(self.btn_cmp)

        l_cmp.addLayout(row_cmp)
        self.v_layout.addWidget(self.cmp_card)

        # ==========================================
        # Section E: 结果展示 (Elevated Card)
        # ==========================================
        self.res_card = ElevatedCardWidget()
        l_res = QVBoxLayout(self.res_card)
        l_res.setContentsMargins(0, 0, 0, 0)
        l_res.setSpacing(0)

        self.txt_report = TextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setMaximumHeight(150)
        self.txt_report.setPlaceholderText("训练日志与分类报告将显示在这里...")
        self.txt_report.setStyleSheet(
            "TextEdit { background: #FAFAFA; border: none; border-bottom: 1px solid #E5E5E5; }")

        self.canvas = MplCanvas(width=10, height=8, dpi=100)

        l_res.addWidget(self.txt_report)
        l_res.addWidget(self.canvas)

        self.v_layout.addWidget(self.res_card)

        # 信号绑定
        self.btn_load_csv.clicked.connect(self._load_csv)
        self.btn_demo.clicked.connect(self._gen_demo)
        self.btn_preview.clicked.connect(self._preview)
        self.btn_train.clicked.connect(self._train)
        self.btn_save.clicked.connect(self._save_model)
        self.btn_load.clicked.connect(self._load_model)
        self.btn_lc.clicked.connect(self._draw_learning_curve)
        self.btn_cmp.clicked.connect(self._run_comparison)

        os.makedirs("data/models", exist_ok=True)

    # ======================================================
    # 配置持久化 (Phase 12)
    # ======================================================
    def _load_settings(self):
        """启动时读取配置"""
        self.ed_target.setText(cfg.get("ML", "target", "label", str))
        self.ed_features.setText(cfg.get("ML", "features", "", str))
        self.split_spin.setValue(cfg.get("ML", "split", 0.2, float))

        self.cmb_algo.setCurrentIndex(cfg.get("ML", "algo_idx", 0, int))
        # 默认参数根据算法而定，但优先读取保存值
        default_grid = "C=0.1,1,10; gamma=scale,auto"
        self.ed_grid.setText(cfg.get("ML", "grid", default_grid, str))
        self.cv_spin.setValue(cfg.get("ML", "cv", 5, int))

        self.chk_standardize.setChecked(cfg.get("ML", "std", True, bool))
        self.chk_kbest.setChecked(cfg.get("ML", "kbest", False, bool))
        self.spin_k.setValue(cfg.get("ML", "k", 20, int))
        self.chk_pca.setChecked(cfg.get("ML", "pca", False, bool))
        self.spin_pca.setValue(cfg.get("ML", "n_pca", 10, int))

    def closeEvent(self, e):
        """关闭时保存配置"""
        cfg.set("ML", "target", self.ed_target.text())
        cfg.set("ML", "features", self.ed_features.text())
        cfg.set("ML", "split", self.split_spin.value())

        cfg.set("ML", "algo_idx", self.cmb_algo.currentIndex())
        cfg.set("ML", "grid", self.ed_grid.text())
        cfg.set("ML", "cv", self.cv_spin.value())

        cfg.set("ML", "std", self.chk_standardize.isChecked())
        cfg.set("ML", "kbest", self.chk_kbest.isChecked())
        cfg.set("ML", "k", self.spin_k.value())
        cfg.set("ML", "pca", self.chk_pca.isChecked())
        cfg.set("ML", "n_pca", self.spin_pca.value())

        super().closeEvent(e)

    # ======================================================
    # 辅助功能
    # ======================================================

    def _show_msg(self, title, content, success=True):
        self.info.emit(f"{title}: {content}")
        if success:
            InfoBar.success(
                title=title, content=content,
                orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP_RIGHT,
                duration=3000, parent=self
            )
        else:
            InfoBar.error(
                title=title, content=content,
                orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP_RIGHT,
                duration=3000, parent=self
            )

    def _style_axis(self, ax, title=""):
        """统一图表风格：去边框、深灰字体、虚线网格"""
        ax.set_title(title, fontsize=10, fontweight='bold', color='#333333', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E0E0E0')
        ax.spines['bottom'].set_color('#E0E0E0')
        ax.tick_params(axis='x', colors='#606060', labelsize=8)
        ax.tick_params(axis='y', colors='#606060', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.3, color='#C0C0C0')

    def _clear_axes(self):
        self.canvas.ax_cm.clear()
        self.canvas.ax_roc.clear()
        self.canvas.ax_lc.clear()
        self.canvas.draw()

    # ======================================================
    # 业务逻辑
    # ======================================================

    def _on_algo_changed(self):
        # 仅当输入框为空或为默认值时才自动填充，避免覆盖用户自定义值
        # 这里简单起见，仅提供默认提示
        pass

    def _load_csv(self):
        if pd is None:
            self._show_msg("错误", "未安装 pandas", False)
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择特征CSV", "data", "CSV Files (*.csv)")
        if not path: return
        try:
            self.df = pd.read_csv(path)
            self._show_msg("成功", f"CSV已载入: {os.path.basename(path)} {self.df.shape}")
            self._extract_Xy()
        except Exception as e:
            self._show_msg("失败", str(e), False)

    def _gen_demo(self):
        n = 300
        rng = np.random.RandomState(0)
        mu0 = np.zeros(12)
        mu1 = np.r_[np.ones(6) * 0.8, np.ones(6) * -0.8]
        cov = 0.3 * np.eye(12)
        X0 = rng.multivariate_normal(mu0, cov, size=n // 2)
        X1 = rng.multivariate_normal(mu1, cov, size=n // 2)
        X = np.vstack([X0, X1]).astype(np.float32)
        y = np.array([0] * (n // 2) + [1] * (n // 2))

        if pd is None:
            self.df = None
            self.X, self.y = X, y
            self.classes_ = np.array([0, 1])
            self._show_msg("成功", "已生成演示数据 (无Pandas模式)", True)
            return

        cols = [f"f{i + 1}" for i in range(X.shape[1])]
        self.df = pd.DataFrame(X, columns=cols)
        self.df["label"] = y
        self._show_msg("成功", "已生成演示数据", True)
        self._extract_Xy()

    def _preview(self):
        if self.df is None:
            self._show_msg("提示", "请先导入数据", False)
            return
        buf = io.StringIO()
        self.df.head(10).to_string(buf, index=False)
        self.txt_report.setPlainText(f"数据预览 (前10行):\n{buf.getvalue()}")

    def _extract_Xy(self):
        if self.df is None: return
        target = self.ed_target.text().strip() or "label"
        if target not in self.df.columns:
            self._show_msg("错误", f"列名 {target} 不存在", False)
            return

        feats_txt = self.ed_features.text().strip()
        if feats_txt:
            feats = [c.strip() for c in feats_txt.split(",") if c.strip()]
        else:
            feats = [c for c in self.df.columns if c != target]

        X = self.df[feats].values.astype(np.float32)
        y_raw = self.df[target].values
        classes, y_enc = np.unique(y_raw, return_inverse=True)
        self.X, self.y = X, y_enc
        self.classes_ = classes
        self.txt_report.setPlainText(f"特征提取完成: X={X.shape}, y={y_enc.shape}\n类别: {classes}")

    def _build_pipeline(self):
        steps = []
        if self.chk_kbest.isChecked():
            k = self.spin_k.value()
            sc = f_classif if self.cmb_kbest_score.currentText() == "f_classif" else mutual_info_classif
            steps.append(("select", SelectKBest(score_func=sc, k=k)))
        if self.chk_standardize.isChecked():
            steps.append(("scaler", StandardScaler()))
        if self.chk_pca.isChecked():
            steps.append(("pca", PCA(n_components=self.spin_pca.value(), random_state=0)))

        name = self.cmb_algo.currentText()
        grid_user = _parse_param_grid(self.ed_grid.text())

        if name == "SVM (RBF)":
            est = SVC(kernel="rbf", probability=True, random_state=0)
            defs = {"clf__C": [1.0], "clf__gamma": ["scale"]}
        elif name == "SVM (Linear)":
            est = SVC(kernel="linear", probability=True, random_state=0)
            defs = {"clf__C": [1.0]}
        elif name == "KNN":
            est = KNeighborsClassifier()
            defs = {"clf__n_neighbors": [5]}
        elif name == "LogisticRegression":
            est = LogisticRegression(max_iter=300, random_state=0)
            defs = {"clf__C": [1.0]}
        elif name == "RandomForest":
            est = RandomForestClassifier(random_state=0)
            defs = {"clf__n_estimators": [100]}
        else:
            est = SVC(probability=True)
            defs = {}

        steps.append(("clf", est))
        pipe = Pipeline(steps)
        grid = {f"clf__{k}": v for k, v in grid_user.items()} if grid_user else defs
        return pipe, grid

    def _train(self):
        if self.X is None or self.y is None:
            self._show_msg("提示", "无数据", False)
            return

        try:
            test_size = self.split_spin.value()
            Xtr, Xte, ytr, yte = train_test_split(self.X, self.y, test_size=test_size, stratify=self.y, random_state=0)
            self._last_split = (Xtr, Xte, ytr, yte)

            pipe, grid = self._build_pipeline()
            cv = self.cv_spin.value()
            scoring = self.cmb_score.currentText()

            self._show_msg("开始训练", f"CV={cv}, Grid={grid}", True)
            self._clear_axes()

            gs = GridSearchCV(pipe, grid, scoring=scoring, cv=StratifiedKFold(cv, shuffle=True, random_state=0),
                              n_jobs=-1)
            gs.fit(Xtr, ytr)

            self.model = gs.best_estimator_

            # Eval
            ypred = self.model.predict(Xte)
            report = classification_report(yte, ypred, target_names=[str(c) for c in self.classes_])
            cm = confusion_matrix(yte, ypred)

            self.txt_report.setPlainText(f"最优参数: {gs.best_params_}\n最佳CV分数: {gs.best_score_:.4f}\n\n{report}")

            # Plot CM
            self._style_axis(self.canvas.ax_cm, "Confusion Matrix")
            self.canvas.ax_cm.imshow(cm, cmap="Blues")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    self.canvas.ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center")

            # Plot ROC
            self._style_axis(self.canvas.ax_roc, "ROC Curve")
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(Xte)
                if probs.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(yte, probs[:, 1])
                    self.canvas.ax_roc.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}", color="#FF6B6B")
                else:
                    for i in range(probs.shape[1]):
                        fpr, tpr, _ = roc_curve((yte == i).astype(int), probs[:, i])
                        self.canvas.ax_roc.plot(fpr, tpr, label=f"C{i} AUC={auc(fpr, tpr):.2f}")
                self.canvas.ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3)
                self.canvas.ax_roc.legend(fontsize=8)

            self.canvas.draw()

        except Exception as e:
            self._show_msg("训练异常", str(e), False)

    def _draw_learning_curve(self):
        if self.X is None: return
        try:
            pipe, _ = self._build_pipeline()
            cv = self.cv_spin.value()
            self._style_axis(self.canvas.ax_lc, "Learning Curve")

            ts, tr_sc, va_sc = learning_curve(pipe, self.X, self.y, cv=StratifiedKFold(cv), n_jobs=-1)
            tr_mean = np.mean(tr_sc, axis=1)
            va_mean = np.mean(va_sc, axis=1)

            self.canvas.ax_lc.plot(ts, tr_mean, 'o-', color="#4ECDC4", label="Train")
            self.canvas.ax_lc.plot(ts, va_mean, 's-', color="#FF6B6B", label="Valid")
            self.canvas.ax_lc.legend(fontsize=8)
            self.canvas.draw()
        except Exception as e:
            self._show_msg("绘图失败", str(e), False)

    def _save_model(self):
        if not self.model: return
        path, _ = QFileDialog.getSaveFileName(self, "保存模型", "data/models/model.pkl", "Pickle (*.pkl)")
        if path:
            try:
                with open(path, "wb") as f:
                    pickle.dump({"model": self.model, "classes": self.classes_}, f)
                self._show_msg("成功", f"保存至 {path}", True)
            except Exception as e:
                self._show_msg("失败", str(e), False)

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "加载模型", "data/models", "Pickle (*.pkl)")
        if path:
            try:
                with open(path, "rb") as f:
                    d = pickle.load(f)
                self.model = d["model"]
                self.classes_ = d["classes"]
                self.txt_report.setPlainText(f"模型已加载: {os.path.basename(path)}")
            except Exception as e:
                self._show_msg("失败", str(e), False)

    def _run_comparison(self):
        if self.X is None: return
        try:
            Xtr, Xte, ytr, yte = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y, random_state=0)
            res = []

            cands = []
            if self.chk_cmp_svm_rbf.isChecked(): cands.append(("SVM-RBF", SVC(probability=True)))
            if self.chk_cmp_svm_lin.isChecked(): cands.append(("SVM-Lin", SVC(kernel='linear', probability=True)))
            if self.chk_cmp_knn.isChecked(): cands.append(("KNN", KNeighborsClassifier()))
            if self.chk_cmp_lr.isChecked(): cands.append(("LR", LogisticRegression()))
            if self.chk_cmp_rf.isChecked(): cands.append(("RF", RandomForestClassifier()))

            if not cands: return

            txt = "批量对比结果:\n"
            base_steps = []
            if self.chk_standardize.isChecked(): base_steps.append(("sc", StandardScaler()))

            scores = []
            names = []

            for name, clf in cands:
                p = Pipeline(base_steps + [("clf", clf)])
                p.fit(Xtr, ytr)
                s = p.score(Xte, yte)
                res.append((name, s))
                txt += f"{name}: {s:.4f}\n"
                scores.append(s)
                names.append(name)

            self.txt_report.setPlainText(txt)

            self._style_axis(self.canvas.ax_cmp, "Batch Comparison")
            self.canvas.ax_cmp.bar(names, scores, color=['#4ECDC4', '#FF6B6B', '#C7F464', '#556270', '#C44D58'])
            self.canvas.ax_cmp.set_ylim(0, 1.05)
            self.canvas.draw()

        except Exception as e:
            self._show_msg("对比失败", str(e), False)