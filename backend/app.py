"""
Multimodal Public Opinion Analysis API — Flask 启动入口。

启动方式（在 D:\\BASNet 目录下）：
    python -m backend.app
    python backend/app.py

只做 API 封装，加载已有模型与结果，不训练、不改原始数据。
"""
import logging
import os
import re
import sys

# 支持 `python backend/app.py`：把项目根目录加入 sys.path，
# 以便 `from backend import ...` 两种启动方式都能工作。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flask import Flask, jsonify, request, send_file, send_from_directory  # noqa: E402
from flask_cors import CORS  # noqa: E402

from backend import config  # noqa: E402
from backend.schemas import (  # noqa: E402
    parse_limit,
    validate_image_file,
    validate_text_request,
)
from backend.utils import APIError, error_payload, success_payload  # noqa: E402
from backend.text_service import TextService  # noqa: E402
from backend.image_service import ImageService  # noqa: E402
from backend.graph_service import GraphService  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("backend.app")

# ---------------------------------------------------------------------------
# 服务单例（进程内各加载一次）
# ---------------------------------------------------------------------------
text_service = TextService()
image_service = ImageService()
graph_service = GraphService()


def create_app() -> Flask:
    app = Flask(__name__)

    # CORS：开发环境允许 Vue dev server；生产环境应限制 origin（见 README）
    CORS(app, resources={r"/api/*": {"origins": config.CORS_ORIGINS}})

    # 全局上传体积上限（图片另有精确校验）
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_IMAGE_SIZE

    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    _register_routes(app)
    _register_error_handlers(app)
    return app


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
def _register_routes(app: Flask):
    @app.route("/api", methods=["GET"])
    def api_info():
        return jsonify(
            {
                "name": config.API_NAME,
                "version": config.API_VERSION,
                "endpoints": [
                    "POST /api/text",
                    "POST /api/image",
                    "GET /api/image/saliency/<filename>",
                    "GET /api/post-image/<image_id>",
                    "POST /api/post-image/<image_id>/saliency",
                    "GET /api/graph",
                    "GET /api/top-nodes",
                    "GET /api/node/<uid>",
                    "GET /api/attention",
                    "GET /api/health",
                ],
            }
        )

    @app.route("/api/health", methods=["GET"])
    def api_health():
        return jsonify(
            success_payload(
                {
                    "status": "ok",
                    "text_model": "loaded" if text_service.loaded else "not_loaded",
                    "image_model": "loaded" if image_service.loaded else "not_loaded",
                    "gat_graph": "loaded" if graph_service.loaded else "not_loaded",
                }
            )
        )

    # ---------------- 文本 ----------------
    @app.route("/api/text", methods=["POST"])
    def api_text():
        try:
            data = request.get_json(silent=True) or {}
            text, method = validate_text_request(data)
            result = text_service.predict_and_explain(text, method)
            return jsonify(success_payload(result))
        except APIError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("text inference error")
            return jsonify(
                error_payload("INTERNAL_ERROR", "internal server error")
            ), 500

    # ---------------- 图片 ----------------
    @app.route("/api/image", methods=["POST"])
    def api_image():
        try:
            file_storage = request.files.get("file")
            data, ext = validate_image_file(file_storage)
            result = image_service.analyze(data, file_storage.filename or "image")
            return jsonify(success_payload(result))
        except APIError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("image inference error")
            return jsonify(
                error_payload("INTERNAL_ERROR", "internal server error")
            ), 500

    @app.route("/api/image/saliency/<filename>", methods=["GET"])
    def api_image_saliency(filename):
        # 只允许后端生成的 uuid.png 文件名，防止路径穿越
        if not re.match(r"^[0-9a-f]{32}\.png$", filename):
            raise APIError("NOT_FOUND", "saliency map not found", 404)
        return send_from_directory(config.OUTPUT_DIR, filename)

    # ---------------- 微博帖子原图 / 显著性检测 ----------------
    def _post_image_path(image_id: str):
        """校验 image_id 并返回本地图片路径，不存在或非法则返回 None。"""
        if not re.match(r"^wb_\d{4}$", image_id):
            return None
        path = os.path.join(config.WEIBO_IMAGES_DIR, f"{image_id}.jpg")
        return path if os.path.exists(path) else None

    @app.route("/api/post-image/<image_id>", methods=["GET"])
    def api_post_image(image_id):
        try:
            path = _post_image_path(image_id)
            if path is None:
                raise APIError("NOT_FOUND", "post image not found", 404)
            return send_file(path, mimetype="image/jpeg")
        except APIError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("post-image error")
            return jsonify(
                error_payload("INTERNAL_ERROR", "internal server error")
            ), 500

    @app.route("/api/post-image/<image_id>/saliency", methods=["POST"])
    def api_post_image_saliency(image_id):
        try:
            path = _post_image_path(image_id)
            if path is None:
                raise APIError("NOT_FOUND", "post image not found", 404)
            with open(path, "rb") as f:
                image_bytes = f.read()
            result = image_service.analyze(image_bytes, f"{image_id}.jpg")
            return jsonify(success_payload(result))
        except APIError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("post-image saliency error")
            return jsonify(
                error_payload("INTERNAL_ERROR", "internal server error")
            ), 500

    # ---------------- 传播网络 ----------------
    @app.route("/api/graph", methods=["GET"])
    def api_graph():
        try:
            return jsonify(success_payload(graph_service.get_graph()))
        except APIError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("graph error")
            return jsonify(
                error_payload("INTERNAL_ERROR", "internal server error")
            ), 500

    @app.route("/api/top-nodes", methods=["GET"])
    def api_top_nodes():
        try:
            limit = parse_limit(
                request.args.get("limit"),
                config.TOP_NODES_DEFAULT_LIMIT,
                config.TOP_NODES_MAX_LIMIT,
            )
            nodes = graph_service.get_top_nodes(limit)
            return jsonify(success_payload({"count": len(nodes), "nodes": nodes}))
        except APIError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("top-nodes error")
            return jsonify(
                error_payload("INTERNAL_ERROR", "internal server error")
            ), 500

    @app.route("/api/node/<uid>", methods=["GET"])
    def api_node(uid):
        try:
            node = graph_service.get_node(uid)
            if node is None:
                raise APIError("NOT_FOUND", f"node '{uid}' not found", 404)
            return jsonify(success_payload(node))
        except APIError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("node error")
            return jsonify(
                error_payload("INTERNAL_ERROR", "internal server error")
            ), 500

    @app.route("/api/attention", methods=["GET"])
    def api_attention():
        try:
            uid = request.args.get("uid")
            limit = parse_limit(
                request.args.get("limit"),
                config.ATTENTION_DEFAULT_LIMIT,
                config.ATTENTION_MAX_LIMIT,
            )
            result = graph_service.get_attention(uid, limit)
            return jsonify(success_payload(result))
        except APIError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("attention error")
            return jsonify(
                error_payload("INTERNAL_ERROR", "internal server error")
            ), 500


# ---------------------------------------------------------------------------
# 统一错误处理
# ---------------------------------------------------------------------------
def _register_error_handlers(app: Flask):
    @app.errorhandler(APIError)
    def handle_api_error(e: APIError):
        return jsonify(error_payload(e.code, e.message)), e.status_code

    @app.errorhandler(404)
    def handle_not_found(e):
        return jsonify(error_payload("NOT_FOUND", "resource not found")), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(e):
        return jsonify(error_payload("INVALID_REQUEST", "method not allowed")), 405

    @app.errorhandler(413)
    def handle_too_large(e):
        return jsonify(
            error_payload("FILE_TOO_LARGE", "request body too large")
        ), 413

    @app.errorhandler(500)
    def handle_internal(e):
        logger.exception("unhandled internal error")
        return jsonify(error_payload("INTERNAL_ERROR", "internal server error")), 500


app = create_app()


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print(config.API_NAME)
    print("=" * 50)
    print()

    print("[1] Loading text model...")
    text_service.load()
    print("    -> " + ("loaded" if text_service.loaded else f"FAILED: {text_service.load_error}"))

    print("[2] Loading image model...")
    image_service.load()
    print("    -> " + ("loaded" if image_service.loaded else f"FAILED: {image_service.load_error}"))

    print("[3] Loading GAT graph...")
    graph_service.load()
    print("    -> " + ("loaded" if graph_service.loaded else f"FAILED: {graph_service.load_error}"))

    print("[4] Flask API ready")
    print()
    print("Server:")
    print(f"http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    print()
    print("Endpoints:")
    print("POST /api/text")
    print("POST /api/image")
    print("GET  /api/image/saliency/<filename>")
    print("GET  /api/post-image/<image_id>")
    print("POST /api/post-image/<image_id>/saliency")
    print("GET  /api/graph")
    print("GET  /api/top-nodes")
    print("GET  /api/node/<uid>")
    print("GET  /api/attention")
    print("GET  /api/health")
    print()
    print("=" * 50)

    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=False)
