"""两层 GAT 模型（PyTorch Geometric）。

结构：
  Input -> GATConv(64, heads=4) -> ELU -> Dropout
        -> GATConv(32, heads=1, concat=False) -> Dropout -> Linear(2)

支持 return_attention 模式，便于导出注意力权重。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from . import config


class GAT(nn.Module):
    def __init__(self, in_dim, hidden=config.HIDDEN, heads=config.HEADS,
                 out_hidden=config.OUT_HIDDEN, num_classes=config.NUM_CLASSES,
                 dropout=config.DROPOUT):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, out_hidden, heads=1,
                             concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(out_hidden, num_classes)
        self.attentions = None  # 每层 [(edge_index, alpha), ...]

    def forward(self, x, edge_index, return_attention=False):
        if return_attention:
            self.attentions = []
            h, (ei1, a1) = self.conv1(x, edge_index, return_attention_weights=True)
            h = F.elu(h)
            h = self.dropout(h)
            h, (ei2, a2) = self.conv2(h, edge_index, return_attention_weights=True)
            h = self.dropout(h)
            out = self.lin(h)
            self.attentions = [(ei1, a1), (ei2, a2)]
            return out

        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = self.dropout(h)
        return self.lin(h)

    def predict_proba(self, x, edge_index):
        logits = self.forward(x, edge_index, return_attention=False)
        return F.softmax(logits, dim=1)
