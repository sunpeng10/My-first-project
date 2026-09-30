"""PyG 数据构建与 GAT 训练。

节点级 transductive 学习：整张子图参与消息传播，loss 只在 train_mask 上计算，
val/test 用于评估。类别不平衡通过 class weight 缓解。
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from . import config
from .model import GAT
from .utils import set_seed


def stratified_masks(y, seed=config.SEED):
    """70/15/15 分层划分 train/val/test 布尔 mask。"""
    y = np.asarray(y)
    n = len(y)
    idx = np.arange(n)

    train_idx, rest_idx = train_test_split(
        idx, train_size=config.TRAIN_RATIO, stratify=y,
        random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
        stratify=y[rest_idx], random_state=seed,
    )

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def build_pyg_data(comp_nodes, X_scaled, edge_list, labels_df, seed=config.SEED):
    """构造 PyG Data。

    - comp_nodes: 最大弱连通分量 uid 列表（顺序即节点索引）
    - X_scaled:   标准化特征矩阵 (N, F)
    - edge_list:  [(src_uid, tgt_uid, weight), ...]
    - labels_df:  index=uid，含 pseudo_label
    """
    uid2idx = {uid: i for i, uid in enumerate(comp_nodes)}
    y = labels_df["pseudo_label"].reindex(comp_nodes).fillna(0).astype("int64").values

    src = []
    tgt = []
    w = []
    for s, t, weight in edge_list:
        src.append(uid2idx[s])
        tgt.append(uid2idx[t])
        w.append(weight)

    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float)

    train_mask, val_mask, test_mask = stratified_masks(y, seed)

    data = Data(
        x=torch.tensor(X_scaled, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=torch.tensor(train_mask),
        val_mask=torch.tensor(val_mask),
        test_mask=torch.tensor(test_mask),
    )
    data.edge_weight = edge_weight
    return data, uid2idx


def make_model(in_dim):
    return GAT(in_dim)


def class_weights(y_tensor):
    y = y_tensor.numpy()
    n = len(y)
    counts = np.bincount(y, minlength=config.NUM_CLASSES)
    # balanced: weight_i = n / (num_classes * n_i)
    w = [n / (config.NUM_CLASSES * max(c, 1)) for c in counts]
    return torch.tensor(w, dtype=torch.float)


def train_model(model, data, device, epochs=config.EPOCHS, lr=config.LR,
                weight_decay=config.WEIGHT_DECAY, patience=config.PATIENCE):
    """训练 GAT，早停监控 validation F1，返回最优模型与历史。"""
    set_seed(config.SEED)
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights(data.y).to(device))

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_pred = out[data.val_mask].argmax(dim=1).cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
        val_f1 = f1_score(val_true, val_pred, average="binary",
                          pos_label=1, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        history.append((epoch, float(loss.detach().cpu()), float(val_f1)))

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_f1, history
