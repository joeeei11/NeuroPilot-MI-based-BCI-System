# core/__init__.py
# 核心算法库初始化
# 这里的代码不依赖任何 PyQt 界面库

from .dsp import butter_filter, notch_filter, compute_psd
from .models import CSP