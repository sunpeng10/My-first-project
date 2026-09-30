"""
评估指标

为 HuggingFace Trainer 提供 compute_metrics 回调，在每轮 eval 后自动计算。

指标说明（二分类场景，macro 平均）:
  - Accuracy  : (TP + TN) / Total        — 整体正确率，类别均衡时直观
  - Precision : TP / (TP + FP)           — "模型说正面的，有多少真的正面" (减少误判)
  - Recall    : TP / (TP + FN)           — "真正正面的，模型找出了多少" (减少漏判)
  - F1        : 2 * P * R / (P + R)      — Precision/Recall 调和平均，综合评价
  - macro avg : 对每个类别分别计算再取平均，关注少数类表现
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(eval_pred) -> dict:
    """Trainer 每轮 eval 后自动调用。

    HF Trainer 内部逻辑:
        logits = model(input_ids, attention_mask).logits   # shape: (batch, num_labels)
        labels = batch["label"]
        eval_pred = EvalPrediction(predictions=logits, label_ids=labels)
        result = compute_metrics(eval_pred)

    Args:
        eval_pred: 含 .predictions (ndarray, shape [N, 2]) 和 .label_ids (ndarray, shape [N,])

    Returns:
        dict: {"accuracy": float, "precision": float, "recall": float, "f1": float}
              键名 "f1" 已注册到 config.metric_for_best_model = "eval_f1"
    """
    logits = eval_pred.predictions          # (N, num_labels) 原始 logits
    labels = eval_pred.label_ids            # (N,) 真实标签

    preds = np.argmax(logits, axis=-1)      # logit → 类别 ID，取概率最大的那个

    # 四个核心指标，均用 macro 平均（每类权重相同，不受样本数影响）
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def get_confusion_matrix(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """计算混淆矩阵。

    返回 shape [2, 2]:
        [[TN, FP],
         [FN, TP]]

    用于后续可解释性分析报告中可视化。
    """
    preds = np.argmax(logits, axis=-1)
    return confusion_matrix(labels, preds)


def get_classification_report(logits: np.ndarray, labels: np.ndarray) -> dict:
    """返回详细的每类别指标，方便写入 reports 或论文表格。

    Returns:
        {
            "neg": {"precision": ..., "recall": ..., "f1": ..., "support": ...},
            "pos": {"precision": ..., "recall": ..., "f1": ..., "support": ...},
        }
    """
    preds = np.argmax(logits, axis=-1)

    # per-class precision, recall, f1, support
    p, r, f, s = [], [], [], []
    for cls_id in [0, 1]:
        tp = int(np.sum((preds == cls_id) & (labels == cls_id)))
        fp = int(np.sum((preds == cls_id) & (labels != cls_id)))
        fn = int(np.sum((preds != cls_id) & (labels == cls_id)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_ = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else 0.0
        support = int(np.sum(labels == cls_id))

        p.append(prec)
        r.append(rec_)
        f.append(f1_)
        s.append(support)

    return {
        "neg": {"precision": p[0], "recall": r[0], "f1": f[0], "support": s[0]},
        "pos": {"precision": p[1], "recall": r[1], "f1": f[1], "support": s[1]},
    }
