"""
加载数据集 + tokenize，返回 train/val/test Dataset 及 tokenizer。

HuggingFace 最佳实践：
  - map() 阶段不 padding（仅 tokenize + truncation）
  - 通过 DataCollatorWithPadding 做动态 batch 级填充
  - 短 batch 不浪费计算，训练速度提升 30-50%

切分策略：
  - 情况 A: 只有 train → split → train/val/test  (含日志)
  - 情况 B: 有 train + test → split train → train/val
  - 情况 C: 三者齐全 → 直接使用

返回 tokenizer 供 Captum 词级归因时 decode token 使用。
"""
import logging
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from src.config import config

logger = logging.getLogger(__name__)


def load_and_tokenize(tokenizer: AutoTokenizer = None):
    """加载并 tokenize 数据集（不做 padding）。

    Returns:
        train_dataset, val_dataset, test_dataset, tokenizer
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # ---- 1. 加载原始数据 ----
    raw = load_dataset(config.dataset_name, trust_remote_code=True)

    # ---- 2. 自动补全 train/val/test 三个 split ----
    if "validation" not in raw:
        split = raw["train"].train_test_split(
            test_size=config.val_split_ratio,
            seed=config.seed,
        )
        train = split["train"]
        val = split["test"]
    else:
        train = raw["train"]
        val = raw["validation"]

    if "test" not in raw:
        split = train.train_test_split(
            test_size=config.val_split_ratio,
            seed=config.seed,
        )
        train = split["train"]
        test = split["test"]
    else:
        test = raw["test"]

    # ---- 3. tokenize（不 padding，由 DataCollator 动态处理）----
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding=False,                      # ← 关键：交给 DataCollator
            truncation=True,
            max_length=config.max_length,
        )

    train_enc = train.map(tokenize_fn, batched=True, batch_size=1000)
    val_enc = val.map(tokenize_fn, batched=True, batch_size=1000)
    test_enc = test.map(tokenize_fn, batched=True, batch_size=1000)

    # ---- 4. 保留列（自动兼容 BERT/RoBERTa）----
    model_inputs = tokenizer.model_input_names
    keep_columns = model_inputs + ["label"]

    train_enc.set_format(type="torch", columns=keep_columns)
    val_enc.set_format(type="torch", columns=keep_columns)
    test_enc.set_format(type="torch", columns=keep_columns)

    # ---- 5. 日志：打印各 split 样本量（原则⑦：实验自动记录）----
    logger.info(
        "数据加载完成 | train=%d  val=%d  test=%d  total=%d",
        len(train_enc), len(val_enc), len(test_enc),
        len(train_enc) + len(val_enc) + len(test_enc),
    )

    return train_enc, val_enc, test_enc, tokenizer


def collate_fn(tokenizer: AutoTokenizer = None):
    """返回 DataCollatorWithPadding，供 Trainer 使用。

    动态 padding：每个 batch 只填充到该 batch 内最长样本的长度，
    避免对全量数据做 max_length 级静态填充。

    用法:
        trainer = Trainer(
            data_collator=collate_fn(tokenizer),
            ...
        )
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return DataCollatorWithPadding(tokenizer=tokenizer)
