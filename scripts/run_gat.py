"""GAT 核心传播节点识别 —— 运行入口。

用法：
    cd /d D:\\BASNet
    python scripts/run_gat.py                 # 完整流程（训练 + 评估 + 导出）
    python scripts/run_gat.py --epochs 200
    python scripts/run_gat.py --topk 20
    python scripts/run_gat.py --mode train
    python scripts/run_gat.py --mode evaluate
    python scripts/run_gat.py --mode predict
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from gat import config, utils                       # noqa: E402
from gat.graph_builder import (build_graph,         # noqa: E402
                               load_comment_edges, subgraph_edges)
from gat.features import (build_feature_frame,      # noqa: E402
                          load_author_uids, load_node_tables,
                          preprocess_features)
from gat.labels import build_pseudo_labels          # noqa: E402
from gat.train import (build_pyg_data, make_model,  # noqa: E402
                       train_model)
from gat.evaluate import compute_metrics            # noqa: E402
from gat.attention import extract_attention         # noqa: E402
from gat.inference import export_all                # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="GAT 核心传播节点识别")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--mode", choices=["all", "train", "evaluate", "predict"],
                   default="all")
    p.add_argument("--device", default=None, help="cpu / cuda，默认自动")
    return p.parse_args()


def banner():
    print("=" * 50)
    print("GAT CORE NODE IDENTIFICATION")
    print("=" * 50)


def build_pipeline():
    """加载数据、构图、选最大连通分量、构造特征与标签，返回上下文 dict。"""
    print("[1] Loading data...")
    edges_df = load_comment_edges()
    node_metrics, user_features = load_node_tables()
    author_uids = load_author_uids()

    print("[2] Building graph...")
    G, comp_nodes, graph_stats = build_graph(edges_df)

    print("[3] Graph statistics...")
    print(f"    original nodes: {graph_stats['original_nodes']}")
    print(f"    original edge rows: {graph_stats['original_edge_rows']}")
    print(f"    self-loops: {graph_stats['self_loop_rows']}")

    print("[4] Selecting largest component...")
    if not comp_nodes:
        raise RuntimeError("无最大连通分量，图为空或全部为孤立节点")
    edge_list = subgraph_edges(G, comp_nodes)
    graph_stats["largest_component_edges"] = len(edge_list)

    print("[5] Building node features...")
    feature_df, feature_cols = build_feature_frame(
        edges_df, node_metrics, user_features, author_uids, comp_nodes)
    X_scaled, features_used, features_dropped = preprocess_features(
        feature_df, feature_cols)
    graph_stats["feature_dim"] = len(features_used)

    print("[6] Building pseudo labels...")
    labels_df, threshold, degenerate = build_pseudo_labels(node_metrics, comp_nodes)
    pos = int(labels_df["pseudo_label"].sum())
    neg = int(len(labels_df) - pos)
    if degenerate:
        print(f"    NOTE: top-10% threshold degenerates to 0; "
              f"positive = core_score > 0 ({pos} nodes)")

    print("[7] Building PyG graph...")
    data, uid2idx = build_pyg_data(comp_nodes, X_scaled, edge_list, labels_df,
                                   seed=config.SEED)

    return {
        "edges_df": edges_df,
        "G": G,
        "comp_nodes": comp_nodes,
        "graph_stats": graph_stats,
        "edge_list": edge_list,
        "feature_df": feature_df,
        "features_used": features_used,
        "features_dropped": features_dropped,
        "labels_df": labels_df,
        "data": data,
        "uid2idx": uid2idx,
        "node_metrics": node_metrics,
        "user_features": user_features,
        "threshold": threshold,
        "degenerate": degenerate,
        "pos": pos,
        "neg": neg,
        "component_size": graph_stats["largest_wcc_size"],
    }


def run(args):
    banner()
    utils.set_seed(args.seed)
    device = utils.get_device() if args.device is None else torch.device(args.device)
    print(f"    device: {device}")

    ctx = build_pipeline()

    do_train = args.mode in ("all", "train")
    do_export = args.mode in ("all", "evaluate", "predict")

    in_dim = ctx["graph_stats"]["feature_dim"]
    model = make_model(in_dim)

    if do_train:
        print("[8] Training GAT...")
        model, best_val_f1, history = train_model(
            model, ctx["data"], device, epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, patience=args.patience)
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "in_dim": in_dim,
                    "features_used": ctx["features_used"],
                    "best_val_f1": float(best_val_f1)},
                   config.MODEL_PATH)
        print(f"    model saved to: {config.MODEL_PATH}")
    else:
        print("[8] Loading saved model (skip training)...")
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(
                f"模型不存在: {config.MODEL_PATH}\n"
                "请先运行: python scripts/run_gat.py --mode train")
        ckpt = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        best_val_f1 = float(ckpt.get("best_val_f1", float("nan")))

    if not do_export:
        print("Done (train only).")
        return

    print("[9] Evaluating...")
    metrics, gat_score, pred = compute_metrics(
        model, ctx["data"], device, ctx["labels_df"], ctx["comp_nodes"])

    print("[10] Extracting attention...")
    att_df, edge_mean_attn = extract_attention(
        model, ctx["data"], ctx["edge_list"], device)

    print("[11] Exporting results...")
    metrics.update({
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "dropout": config.DROPOUT,
        "device": str(device),
        "best_val_f1": round(float(best_val_f1), 4),
        "positive_labels": ctx["pos"],
        "negative_labels": ctx["neg"],
        "core_score_threshold": round(float(ctx["threshold"]), 4),
        "label_degenerate": bool(ctx["degenerate"]),
        "largest_component_nodes": ctx["graph_stats"]["largest_wcc_size"],
        "largest_component_edges": ctx["graph_stats"]["largest_component_edges"],
        "feature_dim": ctx["graph_stats"]["feature_dim"],
    })

    ctx.update({
        "gat_score": gat_score,
        "pred": pred,
        "att_df": att_df,
        "edge_mean_attn": edge_mean_attn,
        "metrics": metrics,
        "topk": args.topk,
    })
    paths, node_scores, top_nodes, graph_json = export_all(ctx)

    _print_summary(ctx["graph_stats"], metrics, paths)
    return paths


def _print_summary(graph_stats, metrics, paths):
    print("")
    print("-" * 50)
    print("SUMMARY")
    print("-" * 50)
    print(f"Original nodes:            {graph_stats['original_nodes']}")
    print(f"Original edges (unique):   {graph_stats['original_edges_unique']}")
    print(f"Largest component nodes:   {graph_stats['largest_wcc_size']}")
    print(f"Largest component edges:   {graph_stats['largest_component_edges']}")
    print(f"Component ratio:           {graph_stats['component_ratio']}")
    print("")
    print(f"Feature dimension:         {metrics['feature_dim']}")
    print(f"Positive labels:           {metrics['positive_labels']}")
    print(f"Negative labels:           {metrics['negative_labels']}")
    print("")
    print(f"Best validation F1:        {metrics['best_val_f1']}")
    print(f"Test Accuracy:             {metrics['test_accuracy']}")
    print(f"Test Precision:            {metrics['test_precision']}")
    print(f"Test Recall:               {metrics['test_recall']}")
    print(f"Test F1:                   {metrics['test_f1']}")
    print(f"ROC-AUC:                   {metrics['roc_auc']}")
    print("")
    for k in ("gat_vs_pagerank_overlap_at_10", "gat_vs_pagerank_overlap_at_20",
              "gat_vs_pagerank_overlap_at_50"):
        kk = k.split("_at_")[-1]
        print(f"Top-{kk} overlap: {metrics[k]}")
    print("")
    print("Outputs saved to:")
    for name, p in paths.items():
        print(f"  {p}")
    print("-" * 50)


if __name__ == "__main__":
    run(parse_args())
