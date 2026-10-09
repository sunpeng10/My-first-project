"""
文本情感分析服务：RoBERTa 分类 + Captum 归因（Integrated Gradients / Saliency）。

- 复用原项目 sentiment_baseline 的模型权重与 `CaptumFriendlyWrapper` /
  `get_embeddings_from_ids` 桥接代码，不重新实现、不重新训练。
- 模型在进程内只加载一次（懒加载 + 线程锁保护）。
- 显著性字段 `char_range` 由 tokenizer 的 `return_offsets_mapping` 可靠计算，
  不伪造。中文 RoBERTa-wwm 的 token 粒度是「单字」，因此 `tokens` 为字级；
  另用 jieba 分词把字级归因聚合成 `words`（词级），供前端切换「字/词」视图。
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

# jieba 词级聚合（可选依赖）：未安装时退化为仅字级显著性
try:
    import jieba as _jieba
except Exception:  # noqa: BLE001
    _jieba = None
_jieba_warmed = False


def aggregate_word_saliency(text: str, tokens: list, mags: list) -> list:
    """把字级显著性按 jieba 分词聚合成词级。

    RoBERTa-wwm 的 token 粒度是单字，char_range 只能表达「字」。这里用
    jieba 在原文上分词（返回 [start, end) 字符区间），把落在每个词区间内
    的字级归因聚合到词：重要性用 L1 幅度求和，方向用带符号和。

    Args:
        text:   原始文本（已 strip，与 tokenizer 输入一致）
        tokens: 字级 token 列表，每项含 char_range=[start, end) 与 score_raw
        mags:   与 tokens 对齐的 L1 幅度（未归一化）

    Returns:
        词级列表 [{"word", "score_raw", "score", "char_range", "char_count"}]
        jieba 不可用时返回空列表（前端据此隐藏「词」视图）。
    """
    if _jieba is None:
        return []

    # 字符位置 -> 覆盖它的字级 token 下标（单字 token 居多；多字 token 也兼容）
    char_to_token = {}
    for idx, t in enumerate(tokens):
        s, e = t["char_range"]
        for c in range(s, e):
            char_to_token[c] = idx

    words = []
    for word, start, end in _jieba.tokenize(text):
        idxs = sorted(
            {char_to_token[c] for c in range(start, end) if c in char_to_token}
        )
        if not idxs:
            continue
        raw = sum(tokens[i]["score_raw"] for i in idxs)  # 带符号和（方向）
        mag = sum(mags[i] for i in idxs)                 # L1 幅度（重要性）
        words.append(
            {
                "word": word,
                "score_raw": round(raw, 4),
                "char_range": [start, end],
                "char_count": len(idxs),
                "_mag": mag,
            }
        )

    max_mag = max((w["_mag"] for w in words), default=0.0)
    for w in words:
        w["score"] = round((w["_mag"] / max_mag) if max_mag > 1e-12 else 0.0, 4)
        w.pop("_mag", None)
    return words


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
                self._warm_jieba()
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

    @staticmethod
    def _warm_jieba():
        """预热 jieba 前缀词典（首次分词需构建词典，约数百毫秒）。

        惰性且幂等，未安装 jieba 时静默跳过。放在 load() 里执行，
        避免把延迟摊到第一次 /api/text 请求上。
        """
        global _jieba_warmed
        if _jieba is None or _jieba_warmed:
            return
        _jieba.lcut("预热")
        _jieba_warmed = True

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
        #    重要性（score）用归因向量的 L1 幅度，而非带符号求和——
        #    带符号求和会让正负分量抵消，把真正的关键字（如"差"）错误压到接近 0。
        attr_flat = attr.squeeze(0).detach().cpu()          # (seq_len, hidden)
        attr_signed = attr_flat.sum(dim=-1)                 # 带符号和 → score_raw（方向）
        attr_mag = attr_flat.abs().sum(dim=-1)              # L1 幅度 → score（重要性）

        real_indices = [
            i
            for i, tid in enumerate(token_ids)
            if tid not in _SPECIAL_IDS and offsets[i] != [0, 0]
        ]

        magnitudes = attr_mag[[real_indices]]
        max_mag = float(magnitudes.max()) if magnitudes.numel() else 0.0

        tokens = []
        mags = []  # 与 tokens 对齐的原始 L1 幅度，供词级聚合
        for pos, i in enumerate(real_indices):
            signed = float(attr_signed[i].item())
            mag = float(attr_mag[i].item())
            score = (mag / max_mag) if max_mag > 1e-12 else 0.0
            mags.append(mag)
            tokens.append(
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
            "tokens": tokens,                                    # 字级显著性（原 words）
            "words": aggregate_word_saliency(text, tokens, mags),  # 词级显著性（jieba）
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
