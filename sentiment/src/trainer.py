"""
训练调度中心：串联 config / data / model / metrics，一键完成训练。

这是整个 pipeline 的核心枢纽，向下游 evaluate.py 和 inference.py
提供训练好的 model 和 trainer。
"""
import logging
from transformers import Trainer, TrainingArguments

from src.config import config
from src.utils import set_seed, ensure_dirs, get_device
from src.data_loader import load_and_tokenize, collate_fn
from src.model import build_model
from src.metrics import compute_metrics

logger = logging.getLogger(__name__)


def train():
    """执行完整训练流程。

    步骤:
        1. 固定随机种子
        2. 创建输出目录
        3. 加载 + tokenize 数据
        4. 构建模型
        5. 配置 TrainingArguments
        6. 组装 Trainer (含动态 padding 和评估指标)
        7. 训练
        8. 保存最佳模型

    Returns:
        trainer, model, tokenizer
        - trainer:   HF Trainer 实例，下游 evaluate.py 用 trainer.evaluate()
        - model:     训练好的模型，下游 inference.py 做预测
        - tokenizer: tokenizer，Captum notebook decode 时用
    """
    # ---- 1. 可复现性 ----
    set_seed()
    device = get_device()
    logger.info("使用设备: %s", device)

    # ---- 2. 目录 ----
    ensure_dirs()

    # ---- 3. 数据 ----
    train_ds, val_ds, test_ds, tokenizer = load_and_tokenize()
    data_collator = collate_fn(tokenizer)

    # ---- 4. 模型 ----
    model = build_model()
    model.to(device)

    # ---- 5. 训练参数 (全部来自 config) ----
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,

        # 批次
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,

        # 优化器
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_epochs,
        warmup_ratio=config.warmup_ratio,

        # 评估 & 保存
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,

        # 精度
        fp16=config.fp16,

        # 可复现
        seed=config.seed,
        data_seed=config.seed,

        # 日志
        logging_dir=config.log_dir,
        report_to=["tensorboard"],   # 训练曲线可视化，论文截图用
    )

    # ---- 6. 组装 Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ---- 7. 训练 ----
    logger.info("开始训练...")
    trainer.train()

    # ---- 8. 保存模型 ----
    trainer.save_model(config.model_save_dir)
    tokenizer.save_pretrained(config.model_save_dir)
    logger.info("模型已保存至: %s", config.model_save_dir)

    return trainer, model, tokenizer
