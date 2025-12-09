# NeuroPilot-MI-based-BCI-System
-----

**NeuroPilot** 是一个基于 Python 和 PyQt5 开发的现代化脑机接口（BCI）上肢康复训练平台。该系统专注于运动想象（Motor Imagery, MI）范式，集成了信号采集、范式呈现、实时/离线数据分析、机器学习模型训练以及外部康复设备控制功能。

系统界面采用 **Fluent Design** 风格（基于 `qfluentwidgets`），提供美观且直观的交互体验。

## ✨ 主要功能

  * **🛡️ 现代化交互界面**: 采用 Windows 11 风格的 Fluent UI，包含仪表盘、导航栏及深色/浅色主题适配。
  * **🎮 运动想象范式 (Task)**: 标准 MI 训练流程（注视 -\> 提示 -\> 想象 -\> 休息），支持自定义左右手示教 GIF 动画。
  * **🧠 信号采集与处理 (EEG)**:
      * 支持多模态连接：串口 (Serial)、蓝牙 (Bluetooth)、TCP 网络。
      * 实时波形显示（高性能 RingBuffer + 动态去直流）。
      * 集成常见预处理（滤波、下采样）。
  * **🤖 机器学习工坊 (ML)**:
      * 内置 `scikit-learn` 管道。
      * 支持 SVM, KNN, RF, LR 等多种算法。
      * 提供网格搜索 (GridSearch)、交叉验证、混淆矩阵及 ROC 曲线绘制。
  * **🕹️ 外设控制 (Device)**: 通过指令（如 'L', 'R'）控制外部康复机器人或外骨骼，支持自动化触发模式。
  * **📊 数据分析 (Data)**: 基于 SQLite 的训练日志管理，支持学习曲线可视化、EEG 频谱分析 (PSD) 及数据导出 (CSV/JSON)。
  * **🔧 调试与日志**: 集成硬件流量监视器（HEX/ASCII）和系统级日志查看器。

## 🛠️ 项目结构

```text
NeuroPilot/
├── assets/                  # 资源文件 (GIF, 图标等)
├── core/                    # 核心逻辑 (e.g., eeg_worker, dsp)
├── data/                    # 数据库与模型存储
├── logs/                    # 系统日志
├── main.py                  # 程序入口
├── login_dialog.py          # 登录界面
├── dashboard_module.py      # 仪表盘 (实时绘图)
├── task_module.py           # 范式呈现 (刺激界面)
├── eeg_module.py            # 脑电采集配置
├── ml_module.py             # 机器学习训练面板
├── device_control.py        # 外设控制模块
├── data_module.py           # 数据分析与回放
├── subject_manager.py       # 受试者管理
├── debug_module.py          # 调试控制台
├── log_module.py            # 日志查看器
└── log_viewer.py            # 独立日志组件
```

## 🚀 快速开始

### 1\. 环境要求

确保安装 Python 3.8+，并安装以下依赖库：

```bash
pip install PyQt5 numpy pandas scipy scikit-learn matplotlib pyserial PyQt-Fluent-Widgets
# 可选依赖 (视硬件连接方式而定)
# pip install pybluez  # 蓝牙支持
# pip install pyqtgraph # 高性能绘图 (默认使用 matplotlib)
```

### 2\. 运行系统

在项目根目录下运行：

```bash
python main.py
```

### 3\. 登录

系统默认测试账号：

  * **账号**: `admin`
  * **密码**: `123456`

*(可在 `login_dialog.py` 中修改默认凭证)*

## 📖 使用指南

1.  **连接设备**: 在 **仪表盘** 或 **脑电采集** 页面配置端口/IP并连接 EEG 设备。
2.  **受试者录入**: 在 **受试者管理** 中添加病人信息。
3.  **范式训练**:
      * 进入 **运动范式** 页面，设置试次数量和时长。
      * 点击“开始试次”，用户跟随屏幕提示进行左右手运动想象。
4.  **模型训练**:
      * 数据采集完成后，进入 **模型训练** 页面。
      * 导入 CSV 数据，选择算法（如 SVM-RBF），点击“开始训练”生成模型。
5.  **在线控制**:
      * 连接外部康复设备（**外设控制** 页）。
      * 在 **仪表盘** 开启“在线预测”，系统将实时识别意图并发送指令驱动设备。

## 📄 许可证

本项目开源，仅供科研与学习使用。

-----

**开发者**: 湖南科技大学玺人
**构建时间**: 2025
