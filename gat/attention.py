"""GAT 注意力权重提取。

使用 PyG GATConv 官方 `return_attention_weights=True` 接口。
返回的 alpha 形状为 (E + N, heads)，其中前 E 行对应输入的原始边
（GATConv 默认 add_self_loops=True，后 N 行是 self-loop，不输出）。

输出：
- att_df: 每条边 × 每层 × 每头 的注意力（gat_attention.csv）
- edge_mean_attn: 与 edge_list 对齐的逐边平均注意力（第一层跨头平均，用于 JSON）
"""
import numpy as np
import pandas as pd
import torch


def extract_attention(model, data, edge_list, device):
    model.eval()
    with torch.no_grad():
        model(data.x.to(device), data.edge_index.to(device), return_attention=True)

    E = len(edge_list)
    rows = []
    edge_attn_accum = np.zeros(E, dtype=np.float64)

    for layer, (ei, alpha) in enumerate(model.attentions, start=1):
        a = alpha.cpu().numpy()          # (E + N, heads)
        n_heads = a.shape[1]
        for h in range(n_heads):
            for i in range(E):
                src_uid, tgt_uid, _ = edge_list[i]
                rows.append((int(src_uid), int(tgt_uid), layer, h, float(a[i, h])))
        if layer == 1:
            edge_attn_accum = a[:E, :].mean(axis=1)   # 第一层跨头平均

    att_df = pd.DataFrame(
        rows, columns=["source_uid", "target_uid", "layer", "head", "attention_weight"]
    )
    return att_df, edge_attn_accum.tolist()
