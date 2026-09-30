"""
测试集评估：加载已训练模型，输出完整评估报告。

产出：
  - Accuracy / Precision / Recall / F1（macro avg）
  - 混淆矩阵
  - 每类别详细指标（precision, recall, f1, support）
  - 结果自动保存至 outputs/eval_results.json
"""
import json
import logging
import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.config import config
from src.utils import set_seed, ensure_dirs, get_device
from src.data_loader import load_and_tokenize, collate_fn
from src.metrics import compute_metrics, get_confusion_matrix, get_classification_report

logger = logging.getLogger(__name__)


def evaluate(model_dir: str = None):
    """在测试集上评估已训练模型。

    Args:
        model_dir: 模型保存路径，默认取 config.model_save_dir

    Returns:
        dict: 包含所有评估指标的结果字典
    """
    set_seed()
    ensure_dirs()

    model_dir = model_dir or config.model_save_dir

    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"模型目录不存在: {model_dir}。请先运行 trainer.py 训练模型。"
        )

    device = get_device()
    logger.info("评估设备: %s", device)
    logger.info("加载模型: %s", model_dir)

    # ---- 1. 加载模型 & tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    # ---- 2. 加载测试数据 ----
    _, _, test_ds, _ = load_and_tokenize(tokenizer)
    data_collator = collate_fn(tokenizer)
    logger.info("测试集样本数: %d", len(test_ds))

    # ---- 3. 构造轻量 Trainer（仅用于 predict）----
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_eval_batch_size=config.eval_batch_size,
        remove_unused_columns=False,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
    )

    # ---- 4. 预测 ----
    logger.info("正在评估...")
    predictions = trainer.predict(test_ds)
    logits = predictions.predictions          # (N, 2)
    labels = predictions.label_ids            # (N,)
    preds = np.argmax(logits, axis=-1)        # (N,)

    # ---- 5. 计算指标 ----
    metrics = compute_metrics(predictions)     # acc, prec, rec, f1 (macro)
    cm = get_confusion_matrix(logits, labels)
    report = get_classification_report(logits, labels)

    # ---- 6. 汇总结果 ----
    results = {
        "model": config.model_name,
        "dataset": config.dataset_name,
        "test_samples": len(test_ds),
        "metrics": {
            "accuracy": round(float(metrics["accuracy"]), 4),
            "precision_macro": round(float(metrics["precision"]), 4),
            "recall_macro": round(float(metrics["recall"]), 4),
            "f1_macro": round(float(metrics["f1"]), 4),
        },
        "confusion_matrix": {
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1]),
        },
        "per_class": {
            "negative": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in report["neg"].items()},
            "positive": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in report["pos"].items()},
        },
    }

    # ---- 7. 打印结果 ----
    m = results["metrics"]
    cm_r = results["confusion_matrix"]
    pc = results["per_class"]

    logger.info("=" * 55)
    logger.info("              测试集评估结果")
    logger.info("=" * 55)
    logger.info("  Accuracy : %.4f", m["accuracy"])
    logger.info("  Precision: %.4f  (macro avg)", m["precision_macro"])
    logger.info("  Recall   : %.4f  (macro avg)", m["recall_macro"])
    logger.info("  F1       : %.4f  (macro avg)", m["f1_macro"])
    logger.info("-" * 55)
    logger.info("  混淆矩阵:")
    logger.info("             预测负类  预测正类")
    logger.info("    真实负类    %4d      %4d", cm_r["TN"], cm_r["FP"])
    logger.info("    真实正类    %4d      %4d", cm_r["FN"], cm_r["TP"])
    logger.info("-" * 55)
    logger.info("  负类: P=%.4f  R=%.4f  F1=%.4f  support=%d",
                pc["negative"]["precision"], pc["negative"]["recall"],
                pc["negative"]["f1"], pc["negative"]["support"])
    logger.info("  正类: P=%.4f  R=%.4f  F1=%.4f  support=%d",
                pc["positive"]["precision"], pc["positive"]["recall"],
                pc["positive"]["f1"], pc["positive"]["support"])
    logger.info("=" * 55)

    # ---- 8. 保存 ----
    save_path = os.path.join(config.output_dir, "eval_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("评估结果已保存: %s", save_path)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    evaluate()
