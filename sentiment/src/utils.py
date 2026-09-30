"""
辅助函数：随机种子、目录创建。
"""
import os
import random
import numpy as np
import torch

from src.config import config


def set_seed():
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs():
    for d in [config.data_dir, config.model_save_dir, config.log_dir, config.output_dir]:
        os.makedirs(d, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
