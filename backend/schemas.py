"""
请求字段校验：把「校验规则」与「业务逻辑」分离。

所有校验失败统一抛出 utils.APIError，由 Flask errorhandler 转成统一 JSON。
"""
from . import config
from .utils import APIError


# ---------------------------------------------------------------------------
# 文本请求
# ---------------------------------------------------------------------------
SUPPORTED_TEXT_METHODS = {"ig", "saliency"}
DEFAULT_TEXT_METHOD = "ig"


def validate_text_request(data: dict) -> tuple:
    """
    校验 /api/text 请求体。

    Returns:
        (text, method)

    Raises:
        APIError(400 INVALID_REQUEST)  文本缺失/为空/超长
        APIError(400 INVALID_REQUEST)  method 不支持
    """
    if not isinstance(data, dict):
        raise APIError("INVALID_REQUEST", "request body must be JSON object", 400)

    text = data.get("text")
    if text is None:
        raise APIError("INVALID_REQUEST", "text is required", 400)

    if not isinstance(text, str):
        raise APIError("INVALID_REQUEST", "text must be a string", 400)

    text = text.strip()
    if not text:
        raise APIError("INVALID_REQUEST", "text cannot be empty", 400)

    if len(text) > config.MAX_TEXT_LENGTH:
        raise APIError(
            "INVALID_REQUEST",
            f"text too long (max {config.MAX_TEXT_LENGTH} characters)",
            400,
        )

    method = data.get("method", DEFAULT_TEXT_METHOD)
    if not isinstance(method, str):
        raise APIError("INVALID_REQUEST", "method must be a string", 400)
    method = method.strip().lower()
    if method not in SUPPORTED_TEXT_METHODS:
        raise APIError("INVALID_REQUEST", "Unsupported explanation method", 400)

    return text, method


# ---------------------------------------------------------------------------
# 图片请求
# ---------------------------------------------------------------------------
def validate_image_file(file_storage) -> bytes:
    """
    校验上传的图片。

    Returns:
        文件原始字节（已完整读入内存）

    Raises:
        APIError(400 INVALID_REQUEST)  缺文件 / 非法扩展名 / 无法读取
        APIError(413 FILE_TOO_LARGE)   超过大小限制
    """
    if file_storage is None:
        raise APIError("INVALID_REQUEST", "file is required", 400)

    filename = file_storage.filename or ""
    if "." not in filename:
        raise APIError("INVALID_REQUEST", "file must have an extension", 400)

    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in config.ALLOWED_IMAGE_EXTENSIONS:
        raise APIError(
            "INVALID_REQUEST",
            f"unsupported image type '{ext}' (allowed: "
            + ", ".join(sorted(config.ALLOWED_IMAGE_EXTENSIONS))
            + ")",
            400,
        )

    data = file_storage.read()
    if len(data) == 0:
        raise APIError("INVALID_REQUEST", "file is empty", 400)

    if len(data) > config.MAX_IMAGE_SIZE:
        raise APIError(
            "FILE_TOO_LARGE",
            f"file too large (max {config.MAX_IMAGE_SIZE // (1024 * 1024)} MB)",
            413,
        )

    return data, ext


# ---------------------------------------------------------------------------
# 查询参数校验
# ---------------------------------------------------------------------------
def parse_limit(value, default: int, maximum: int) -> int:
    """解析 ?limit= 查询参数，越界/非法时给出明确错误或回退默认值。"""
    if value is None:
        return default
    try:
        limit = int(value)
    except (TypeError, ValueError):
        raise APIError("INVALID_REQUEST", "limit must be an integer", 400)
    if limit < 1:
        raise APIError("INVALID_REQUEST", "limit must be >= 1", 400)
    return min(limit, maximum)
