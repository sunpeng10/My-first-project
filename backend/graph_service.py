"""
传播网络服务：直接读取 GAT 已生成的结果文件，不重跑 GAT / PageRank / 训练。

数据来源（只读）：
- results/gat_graph.json       —— 图结构（2848 nodes / 2999 edges）
- results/gat_node_scores.csv  —— 节点详情 + 核心节点排名（按 pagerank 降序）
- results/gat_attention.csv    —— 注意力边（约 14995 条）

启动时全部读入内存并建索引，避免每次请求扫描 CSV。
"""
import json
import logging
import os
import threading

import pandas as pd

from . import config
from .utils import APIError

logger = logging.getLogger(__name__)


class GraphService:
    def __init__(self):
        self.graph = None            # {"nodes": [...], "edges": [...]}
        self.top_nodes = None        # list[dict]（含 rank）
        self.node_map = None         # uid -> dict
        self.attention_index = None  # uid -> list[dict]
        self.attention_sorted = None # list[dict]，按 attention_weight 降序
        self.author_posts = None     # author_uid(str) -> list[dict]（该作者的微博）
        self.loaded = False
        self.load_error = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 加载（进程内一次）
    # ------------------------------------------------------------------
    def load(self):
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            try:
                # 1. 图结构（原样读取 JSON）
                with open(config.GAT_GRAPH_JSON, "r", encoding="utf-8") as f:
                    self.graph = json.load(f)

                # 2. 节点详情 -> uid 索引（全量节点）
                node_df = pd.read_csv(config.GAT_NODE_SCORES_CSV, encoding="utf-8-sig")
                node_df["uid"] = node_df["uid"].astype(str)
                self.node_map = {
                    str(row["uid"]): self._clean_node_row(row)
                    for _, row in node_df.iterrows()
                }

                # 3. 核心节点：全量节点按 pagerank 降序取前 TOP_NODES_MAX_LIMIT
                #    gat_score 因 pseudo-label 退化而在作者群体内饱和≈1.0，
                #    故改用 pagerank 作为排序键，使 Top 排名具有区分度。
                top_df = (
                    node_df.sort_values(
                        "pagerank", ascending=False, na_position="last"
                    ).reset_index(drop=True)
                )
                top_df.insert(0, "rank", list(range(1, len(top_df) + 1)))
                self.top_nodes = top_df.head(config.TOP_NODES_MAX_LIMIT).to_dict("records")

                # 4. 注意力边 -> uid 索引（source/target 双向），并按权重降序缓存
                att_df = pd.read_csv(config.GAT_ATTENTION_CSV, encoding="utf-8-sig")
                att_df["source_uid"] = att_df["source_uid"].astype(str)
                att_df["target_uid"] = att_df["target_uid"].astype(str)

                edges = []
                index = {}
                for _, row in att_df.iterrows():
                    edge = {
                        "source_uid": str(row["source_uid"]),
                        "target_uid": str(row["target_uid"]),
                        "layer": int(row["layer"]),
                        "head": int(row["head"]),
                        "attention_weight": float(row["attention_weight"]),
                    }
                    edges.append(edge)
                    for uid in (edge["source_uid"], edge["target_uid"]):
                        index.setdefault(uid, []).append(edge)

                self.attention_index = index
                self.attention_sorted = sorted(
                    edges, key=lambda e: e["attention_weight"], reverse=True
                )

                # 5. 作者→微博映射（post_authors.csv），用于节点详情展示该作者微博
                author_posts = {}
                if os.path.exists(config.POST_AUTHORS_CSV):
                    authors_df = pd.read_csv(
                        config.POST_AUTHORS_CSV, encoding="utf-8-sig"
                    )
                    authors_df["author_uid"] = authors_df["author_uid"].astype(str)
                    images_dir = config.WEIBO_IMAGES_DIR
                    for _, row in authors_df.iterrows():
                        uid = str(row["author_uid"])
                        image_id = str(row["image_id"])
                        post = {
                            "image_id": image_id,
                            "weibo_url": (
                                str(row["weibo_url"]) if pd.notna(row["weibo_url"]) else ""
                            ),
                            "text": str(row["text"]) if pd.notna(row["text"]) else "",
                            "has_image": os.path.exists(
                                os.path.join(images_dir, f"{image_id}.jpg")
                            ),
                        }
                        author_posts.setdefault(uid, []).append(post)
                self.author_posts = author_posts

                self.loaded = True
                logger.info(
                    "gat graph loaded: %d nodes, %d edges, %d attention rows",
                    len(self.graph.get("nodes", [])),
                    len(self.graph.get("edges", [])),
                    len(edges),
                )
            except Exception as exc:  # noqa: BLE001
                self.load_error = str(exc)
                self.loaded = False
                logger.exception("failed to load gat graph data")

    def _ensure_loaded(self):
        if not self.loaded:
            self.load()
        if not self.loaded:
            raise RuntimeError(f"gat graph data not loaded: {self.load_error}")

    @staticmethod
    def _clean_node_row(row) -> dict:
        """把 CSV 行转成 JSON 友好的 dict，数值类型尽量精确。"""
        d = {}
        for k, v in row.items():
            k = str(k)
            if k == "uid":
                d[k] = str(v)
            elif k == "username":
                d[k] = str(v) if pd.notna(v) else ""
            else:
                d[k] = _to_number(v)
        return d

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------
    def get_graph(self) -> dict:
        self._ensure_loaded()
        return self.graph

    def get_top_nodes(self, limit: int) -> list:
        self._ensure_loaded()
        return self.top_nodes[:limit]

    def get_node(self, uid: str):
        self._ensure_loaded()
        node = self.node_map.get(str(uid))
        if node is None:
            return None
        # 返回副本，附加该作者发布的微博（不污染 node_map 缓存）
        result = dict(node)
        result["authored_posts"] = self.author_posts.get(str(uid), [])
        return result

    def get_attention(self, uid: str = None, limit: int = 1000) -> dict:
        self._ensure_loaded()
        if uid is None:
            return {
                "total_edges": len(self.attention_sorted),
                "edges": self.attention_sorted[:limit],
            }
        matches = self.attention_index.get(str(uid), [])
        return {
            "uid": str(uid),
            "count": len(matches),
            "edges": matches[:limit],
        }


def _to_number(v):
    """尽力把值转成 int/float，失败则原样返回。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    s = str(v).strip()
    if not s:
        return None
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s
