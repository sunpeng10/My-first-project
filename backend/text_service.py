"""
文本情感分析服务：RoBERTa 分类 + Captum 归因（Integrated Gradients / Saliency）。

- 复用原项目 sentiment_baseline 的模型权重与 `CaptumFriendlyWrapper` /
  `get_embeddings_from_ids` 桥接代码，不重新实现、不重新训练。
- 模型在进程内只加载一次（懒加载 + 线程锁保护）。
- 显著性字段 `char_range` 由 tokenizer 的 `return_offsets_mapping` 可靠计算，
  不伪造。中文 RoBERTa-wwm 的 token 粒度是「单字」，因此 words 为 token 级。
"""
import logging
import sys
import threading

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config
from .utils import get_device

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "negative", 1: "positive"}

# RoBERTa / BERT 的特殊 token id
_SPECIAL_IDS = {0, 101, 102}  # [PAD], [CLS], [SEP]


class TextService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.wrapper_cls = None
        self.get_embeddings = None
        self.device = get_device()
        self.loaded = False
        self.load_error = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 模型加载（进程内一次）
    # ------------------------------------------------------------------
    def load(self):
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            try:
                # 复用原项目的桥接代码（CaptumFriendlyWrapper / get_embeddings_from_ids）
                if config.SENTIMENT_ROOT not in sys.path:
                    sys.path.insert(0, config.SENTIMENT_ROOT)
                from src.model import (  # noqa: E402
                    CaptumFriendlyWrapper,
                    get_embeddings_from_ids,
                )

                self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_DIR)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    config.TEXT_MODEL_DIR
                )
                self.model.to(self.device)
                self.model.eval()
                self.wrapper_cls = CaptumFriendlyWrapper
                self.get_embeddings = get_embeddings_from_ids
                self.loaded = True
                logger.info(
                    "text model loaded: %s (%s)", config.TEXT_MODEL_DIR, self.device
                )
            except Exception as exc:  # noqa: BLE001
                self.load_error = str(exc)
                self.loaded = False
                logger.exception("failed to load text model")

    def _ensure_loaded(self):
        if not self.loaded:
            self.load()
        if not self.loaded:
            raise RuntimeError(f"text model not loaded: {self.load_error}")

    # ------------------------------------------------------------------
    # 推理 + 归因
    # ------------------------------------------------------------------
    def predict_and_explain(self, text: str, method: str = "ig") -> dict:
        self._ensure_loaded()

        # 1. 分词（不 pad，保留真实 token 边界与 char 偏移）
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=config.MAX_TEXT_LENGTH,  # 足够长；实际仍受模型 512 上限约束
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        offsets = enc["offset_mapping"][0].tolist()
        token_ids = enc["input_ids"][0].tolist()
        token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)

        # 2. 复用原项目 embedding 桥接 + wrapper 前向
        wrapper = self.wrapper_cls(self.model)
        emb = self.get_embeddings(
            self.model, input_ids, attention_mask=attention_mask
        )

        with torch.no_grad():
            logits = wrapper(emb, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
            label_id = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs[0, label_id].item())

        # 3. Captum 归因
        attr = self._attribute(wrapper, emb, attention_mask, label_id, method)

        # 4. 聚合到 token 级
        attr_sum = attr.squeeze(0).sum(dim=-1)  # (seq_len,)
        attr_sum = attr_sum.detach().cpu()

        real_indices = [
            i
            for i, tid in enumerate(token_ids)
            if tid not in _SPECIAL_IDS and offsets[i] != [0, 0]
        ]

        magnitudes = torch.abs(attr_sum[[real_indices]])
        max_mag = float(magnitudes.max()) if magnitudes.numel() else 0.0

        words = []
        for pos, i in enumerate(real_indices):
            signed = float(attr_sum[i].item())
            mag = abs(signed)
            score = (mag / max_mag) if max_mag > 1e-12 else 0.0
            words.append(
                {
                    "word": token_texts[i].lstrip("#"),
                    "score": round(score, 4),
                    "score_raw": round(signed, 4),
                    "token_index": pos,
                    "token_text": token_texts[i],
                    "char_range": [int(offsets[i][0]), int(offsets[i][1])],
                }
            )

        return {
            "text": text,
            "label": LABEL_MAP[label_id],
            "label_id": label_id,
            "confidence": round(confidence, 4),
            "method": method,
            "words": words,
        }

    def _attribute(self, wrapper, emb, attention_mask, label_id, method):
        if method == "ig":
            from captum.attr import IntegratedGradients

            ig = IntegratedGradients(wrapper)
            return ig.attribute(
                emb,
                target=label_id,
                n_steps=20,
                internal_batch_size=1,
                additional_forward_args=(attention_mask,),
            )
        if method == "saliency":
            from captum.attr import Saliency

            sal = Saliency(wrapper)
            return sal.attribute(
                emb,
                target=label_id,
                abs=True,
                additional_forward_args=(attention_mask,),
            )
        raise ValueError(f"unsupported method: {method}")
