"""
集中管理：模型名、路径、训练超参、随机种子。
"""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ==================== 模型 ====================
    model_name: str = "hfl/chinese-roberta-wwm-ext"
    num_labels: int = 2                     # 正/负 二分类

    # ==================== 数据集 ====================
    dataset_name: str = "lansinuote/ChnSentiCorp"    # seamew 镜像，无 GFW 问题
    val_split_ratio: float = 0.1            # 无 val 集时从 train 切的比例

    # ==================== 路径 ====================
    project_root: str = field(default_factory=lambda: os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    data_dir: str = ""
    model_save_dir: str = ""
    log_dir: str = ""
    output_dir: str = ""

    # ==================== 训练超参 ====================
    max_length: int = 128
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    fp16: bool = False                      # Windows GPU 上建议 False

    # ==================== 随机种子 ====================
    seed: int = 42

    def __post_init__(self):
        """自动推导子目录路径。"""
        if not self.data_dir:
            self.data_dir = os.path.join(self.project_root, "data")
        if not self.model_save_dir:
            self.model_save_dir = os.path.join(self.project_root, "models")
        if not self.log_dir:
            self.log_dir = os.path.join(self.project_root, "logs")
        if not self.output_dir:
            self.output_dir = os.path.join(self.project_root, "outputs")


config = Config()
