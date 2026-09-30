"""
集中配置：路径、模型位置、上传限制、CORS、API 元信息。

只读配置，不含任何模型加载逻辑。修改路径时只改这里。
"""
import os

# ---------------------------------------------------------------------------
# 目录结构
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)  # D:\BASNet

# ---------------------------------------------------------------------------
# 文本模型（RoBERTa）—— 位于独立项目目录，只读加载，不复制、不重训
# ---------------------------------------------------------------------------
SENTIMENT_ROOT = os.environ.get(
    "SENTIMENT_ROOT",
    r"D:\vscode python files\sentiment_baseline",
)
TEXT_MODEL_DIR = os.path.join(SENTIMENT_ROOT, "models")

# ---------------------------------------------------------------------------
# 图像模型（BASNet）—— 已训练权重
# ---------------------------------------------------------------------------
BASNET_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "basnet", "saved_models", "basnet_bsi", "basnet.pth"
)

# ---------------------------------------------------------------------------
# GAT 传播网络 —— 直接读取已生成结果，不重跑 GAT / PageRank / 训练
# ---------------------------------------------------------------------------
GAT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
GAT_GRAPH_JSON = os.path.join(GAT_RESULTS_DIR, "gat_graph.json")
GAT_TOP_NODES_CSV = os.path.join(GAT_RESULTS_DIR, "gat_top_nodes.csv")
GAT_NODE_SCORES_CSV = os.path.join(GAT_RESULTS_DIR, "gat_node_scores.csv")
GAT_ATTENTION_CSV = os.path.join(GAT_RESULTS_DIR, "gat_attention.csv")

# ---------------------------------------------------------------------------
# 微博帖子数据 —— 作者→微博映射（image_id, weibo_url, author_uid, text）与本地图片
# ---------------------------------------------------------------------------
POST_AUTHORS_CSV = os.path.join(
    PROJECT_ROOT, "visual_saliency_analysis", "graph", "post_authors.csv"
)
WEIBO_IMAGES_DIR = r"D:\WeiboCrawler\images"

# ---------------------------------------------------------------------------
# 上传 / 输出目录（运行时创建）
# ---------------------------------------------------------------------------
UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")
OUTPUT_DIR = os.path.join(BACKEND_DIR, "outputs")

# ---------------------------------------------------------------------------
# 输入校验限制
# ---------------------------------------------------------------------------
MAX_TEXT_LENGTH = 5000                    # 文本最大字符数
MAX_IMAGE_SIZE = 10 * 1024 * 1024         # 图片最大 10 MB
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# ---------------------------------------------------------------------------
# 分页 / 查询限制
# ---------------------------------------------------------------------------
TOP_NODES_DEFAULT_LIMIT = 20
TOP_NODES_MAX_LIMIT = 100
ATTENTION_DEFAULT_LIMIT = 1000
ATTENTION_MAX_LIMIT = 10000

# ---------------------------------------------------------------------------
# CORS（开发环境允许 Vue dev server；生产环境应收紧）
# ---------------------------------------------------------------------------
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# ---------------------------------------------------------------------------
# API 元信息
# ---------------------------------------------------------------------------
API_NAME = "Multimodal Public Opinion Analysis API"
API_VERSION = "1.0.0"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000
