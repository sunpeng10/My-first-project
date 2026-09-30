"""
预测模块：对新文本做情感分类。

三种使用场景：
  1. predict()          — 单条预测，返回 (标签, 置信度)
  2. predict_batch()    — 批量预测
  3. predict_with_embeddings() — 返回 embeddings + logits，供 Captum 归因分析

所有接口自动切换 CPU/GPU。
"""
import logging
from typing import Dict, List, Tuple, Union

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import config
from src.utils import get_device
from src.model import CaptumFriendlyWrapper, get_embeddings_from_ids

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "negative", 1: "positive"}


# ============================================================
#  核心预测函数
# ============================================================

def predict(
    text: str,
    model: AutoModelForSequenceClassification = None,
    tokenizer: AutoTokenizer = None,
) -> Dict:
    """单条文本情感预测。

    Args:
        text:     输入文本
        model:    已加载的 HF 模型（可选，不传则从 model_dir 加载）
        tokenizer: 已加载的 tokenizer（可选）

    Returns:
        {"text": str, "label": "positive"|"negative",
         "label_id": 0|1, "score": float}
    """
    model, tokenizer, device = _ensure_model(model, tokenizer)

    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits                 # (1, 2)
        probs = torch.softmax(logits, dim=-1)            # (1, 2)
        label_id = int(torch.argmax(probs, dim=-1).item())
        score = float(probs[0, label_id].item())

    return {
        "text": text,
        "label": LABEL_MAP[label_id],
        "label_id": label_id,
        "score": round(score, 4),
    }


def predict_batch(
    texts: List[str],
    model: AutoModelForSequenceClassification = None,
    tokenizer: AutoTokenizer = None,
    batch_size: int = None,
) -> List[Dict]:
    """批量情感预测。

    Args:
        texts:      文本列表
        model:      已加载的 HF 模型（可选）
        tokenizer:  已加载的 tokenizer（可选）
        batch_size: 预测 batch 大小，默认取 config.eval_batch_size

    Returns:
        [{"text": ..., "label": ..., "label_id": ..., "score": ...}, ...]
    """
    model, tokenizer, device = _ensure_model(model, tokenizer)
    batch_size = batch_size or config.eval_batch_size

    def collate(batch_texts):
        return tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )

    dataloader = DataLoader(texts, batch_size=batch_size, collate_fn=collate)

    results = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
            label_ids = torch.argmax(probs, dim=-1)
            scores = probs[range(len(label_ids)), label_ids]

            for i, t in enumerate(texts[len(results):len(results) + len(label_ids)]):
                results.append({
                    "text": t,
                    "label": LABEL_MAP[label_ids[i].item()],
                    "label_id": label_ids[i].item(),
                    "score": round(scores[i].item(), 4),
                })

    return results


# ============================================================
#  Captum 桥接：同时返回 embedding 和 logits
# ============================================================

def predict_with_embeddings(
    text: str,
    model: AutoModelForSequenceClassification = None,
    tokenizer: AutoTokenizer = None,
) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    """预测 + 返回 embedding，用于 Captum 归因分析。

    单次调用同时返回：
      - 预测结果 dict
      - inputs_embeds:  embedding 层输出，传给 Captum IG/Saliency
      - logits:         模型原始输出

    用法（在 Captum notebook 中）:
        result, emb, logits = predict_with_embeddings(text, model, tokenizer)

        # Integrated Gradients
        wrapper = CaptumFriendlyWrapper(model)
        ig = IntegratedGradients(wrapper)
        attr = ig.attribute(emb, target=1, ...)

    Args:
        text:      输入文本
        model:     已加载的 HF 模型（需与 CaptumFriendlyWrapper 共用）
        tokenizer: 已加载的 tokenizer

    Returns:
        (result_dict, inputs_embeds, logits)
    """
    model, tokenizer, device = _ensure_model(model, tokenizer)

    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    ).to(device)

    # 提取 embedding（Captum 入口）
    emb_kwargs = {"input_ids": inputs["input_ids"]}
    if "token_type_ids" in inputs:
        emb_kwargs["token_type_ids"] = inputs["token_type_ids"]
    emb = get_embeddings_from_ids(model, **emb_kwargs)

    # 通过 wrapper 做 forward（保持与 Captum 分析一致的计算路径）
    wrapper = CaptumFriendlyWrapper(model)
    model.eval()
    with torch.no_grad():
        logits = wrapper(emb, attention_mask=inputs.get("attention_mask"))
        probs = torch.softmax(logits, dim=-1)
        label_id = int(torch.argmax(probs, dim=-1).item())
        score = float(probs[0, label_id].item())

    result = {
        "text": text,
        "label": LABEL_MAP[label_id],
        "label_id": label_id,
        "score": round(score, 4),
    }

    return result, emb, logits


# ============================================================
#  内部辅助
# ============================================================

def _ensure_model(model, tokenizer):
    """确保 model 和 tokenizer 已加载，未传则从 model_dir 加载。"""
    if model is None or tokenizer is None:
        model_dir = config.model_save_dir
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = get_device()
    model.to(device)
    return model, tokenizer, device


# ============================================================
#  CLI 快速测试
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # 测试单条
    r = predict("这家酒店的服务态度非常好，下次还会再来！")
    print(r)

    # 测试批量
    texts = [
        "这家酒店的服务态度非常好，下次还会再来！",
        "房间又脏又乱，非常失望，不推荐。",
        "位置还不错，但是价格有点贵。",
    ]
    for r in predict_batch(texts):
        print(f"[{r['label']}] ({r['score']:.2%}) {r['text']}")
