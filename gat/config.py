"""GAT 核心传播节点识别 —— 全局配置。

本阶段只做一件事：基于真实 User→User 评论网络，用 GAT 学习节点表示，
输出核心传播节点排名。不做点赞/评论/转发量回归，不引入异构图神经网络。
"""
import os

# ---------------------------------------------------------------------------
# 数据路径（只读，绝不修改）
# ---------------------------------------------------------------------------
DATA_DIR = r"D:\WeiboCrawler\data"
RESULTS_DIR_WB = r"D:\WeiboCrawler\results"

COMMENT_EDGES = os.path.join(DATA_DIR, "comment_edges.csv")
NODE_METRICS = os.path.join(RESULTS_DIR_WB, "comment_network_node_metrics.csv")
USER_FEATURES = os.path.join(RESULTS_DIR_WB, "heterogeneous_graph_user_features.csv")
POST_FEATURES = os.path.join(RESULTS_DIR_WB, "heterogeneous_graph_post_features.csv")
USER_MAPPING = os.path.join(DATA_DIR, "weibo_user_mapping.csv")

# ---------------------------------------------------------------------------
# 输出路径
# ---------------------------------------------------------------------------
BASE_DIR = r"D:\BASNet"
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gat_best.pt")

# ---------------------------------------------------------------------------
# pseudo-label 构造（弱监督，非人工标注）
#   core_score = alpha*norm(pagerank) + beta*norm(in_degree) + gamma*norm(weighted_in_degree)
# ---------------------------------------------------------------------------
CORE_ALPHA = 0.5
CORE_BETA = 0.3
CORE_GAMMA = 0.2
TOP_PERCENT = 0.10          # Top 10% -> core_label = 1

# ---------------------------------------------------------------------------
# 节点特征（避免 target leakage）
#   绝不使用：pagerank / in_degree / weighted_in_degree / core_score
#   不直接使用：total_degree（含 in_degree）
#   component_size 在最大弱连通分量内为常数，故不纳入训练特征（仍写入输出 CSV）
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "out_degree",            # 出度：评论过多少个不同作者
    "weighted_out_degree",   # 加权出度：发出的评论总数（== comment_count）
    "posts_commented",       # 评论过的不同帖子数
    "posts_authored",        # 原创帖子数
    "author_flag",           # 是否原创作者（0/1，来自 weibo 作者名单）
    "avg_comment_length",    # 平均评论文本长度
]
LOG1P_FEATURES = {
    "out_degree", "weighted_out_degree", "posts_commented",
    "posts_authored", "avg_comment_length",
}

# ---------------------------------------------------------------------------
# GAT 模型结构
# ---------------------------------------------------------------------------
HIDDEN = 64                 # 第一层每头输出维度
HEADS = 4                   # 第一层注意力头数 -> 64*4 = 256
OUT_HIDDEN = 32             # 第二层输出维度（heads=1, concat=False）
NUM_CLASSES = 2             # core / non-core
DROPOUT = 0.2

# ---------------------------------------------------------------------------
# 训练参数
# ---------------------------------------------------------------------------
EPOCHS = 200
LR = 0.005
WEIGHT_DECAY = 5e-4
PATIENCE = 30               # early stopping 监控 validation F1
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

# ---------------------------------------------------------------------------
# 输出配置
# ---------------------------------------------------------------------------
TOP_N_NODES = 100           # gat_top_nodes.csv 保存前 100
OVERLAP_KS = [10, 20, 50]   # Top-K overlap 评估

CN_FONTS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun",
            "Arial Unicode MS"]
