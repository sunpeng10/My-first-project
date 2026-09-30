"""
模型层：chinese-roberta-wwm-ext + 二分类头

提供三个接口：
  1. build_model()               → 标准 HuggingFace 模型 (训练/推理)
  2. CaptumFriendlyWrapper(model) → 接受 inputs_embeds 的 forward (Captum 归因)
  3. get_embeddings_from_ids()   → input_ids → inputs_embeds 桥接 (复用)
"""
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from src.config import config


def build_model() -> AutoModelForSequenceClassification:
    """标准 HuggingFace 模型加载。"""
    return AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
    )


# ============================================================
#  CaptumFriendlyWrapper
# ============================================================

class CaptumFriendlyWrapper(nn.Module):
    """
    将 HuggingFace 模型包装为可接受 embedding 输入的版本。

    Captum 的 Saliency / Integrated Gradients / DeepLift 等方法
    要求输入是连续可微的张量，而不是离散的 input_ids。

    设计要点：
      - 使用 hf_model.base_model_prefix 动态获取 encoder，
        自动兼容 roberta / bert / electra 等不同模型架构
      - 不依赖硬编码的模型名，换模型只需改 config.model_name

    用法:
        model = build_model()
        wrapper = CaptumFriendlyWrapper(model)

        # 通过桥接函数获取 embedding
        emb = get_embeddings_from_ids(model, input_ids, attention_mask)
        # 传入 wrapper 做 Captum 归因
        logits = wrapper(emb, attention_mask=attention_mask)
    """

    def __init__(self, hf_model: AutoModelForSequenceClassification):
        super().__init__()
        # 动态获取 encoder（roberta / bert / electra ...）
        prefix = hf_model.base_model_prefix           # "roberta"
        self.encoder = getattr(hf_model, prefix)       # hf_model.roberta
        self.classifier = hf_model.classifier          # dense → dropout → out_proj
        self.config = hf_model.config                  # 方便调用方检查 hidden_size 等

    def forward(self, inputs_embeds, attention_mask=None):
        """
        Args:
            inputs_embeds:  (batch, seq_len, hidden=768)
            attention_mask: (batch, seq_len)

        Returns:
            logits: (batch, num_labels)
        """
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        pooled = encoder_outputs.pooler_output        # [CLS] → Tanh → Dense
        logits = self.classifier(pooled)
        return logits


# ============================================================
#  Embedding 桥接函数
# ============================================================

def get_embeddings_from_ids(
    hf_model: AutoModelForSequenceClassification,
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
    """从 input_ids 提取连续 embedding 向量。

    这是离散 token → 连续表示的「桥接函数」。所有 Captum 分析脚本
    (Saliency, Integrated Gradients, DeepLift 等) 统一通过此函数获取
    inputs_embeds，避免每个 notebook 重复实现 embedding 提取逻辑。

    Args:
        hf_model:        已加载的 HF 模型
        input_ids:       (batch, seq_len) 或 (seq_len,)
        token_type_ids:  可选，BERT 系模型需要；RoBERTa 传 None
        attention_mask:  可选，(batch, seq_len)

    Returns:
        inputs_embeds: (batch, seq_len, hidden_size) 或 (seq_len, hidden_size)

    示例:
        model = build_model()
        emb = get_embeddings_from_ids(model, input_ids, attention_mask=mask)
        wrapper = CaptumFriendlyWrapper(model)
        logits = wrapper(emb, attention_mask=mask)
        # 然后传给 Captum: ig.attribute(emb, target=1, ...)
    """
    # 确保是 2D tensor（单条样本扩展 batch 维度）
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        if token_type_ids is not None and token_type_ids.dim() == 1:
            token_type_ids = token_type_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

    prefix = hf_model.base_model_prefix
    embedding_layer = getattr(hf_model, prefix).embeddings

    # RoBERTa 不需要 token_type_ids；BERT 需要
    kwargs = {}
    if token_type_ids is not None:
        kwargs["token_type_ids"] = token_type_ids

    return embedding_layer(input_ids=input_ids, **kwargs)
