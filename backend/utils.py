"""
通用工具：统一错误类型、响应构造、安全文件名、设备选择。
"""
import re
import uuid

import torch


# ---------------------------------------------------------------------------
# 统一错误类型
# ---------------------------------------------------------------------------
class APIError(Exception):
    """业务错误，携带 HTTP 状态码与错误码，由 Flask errorhandler 统一转 JSON。"""

    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def error_payload(code: str, message: str) -> dict:
    """统一错误响应体（见 README「错误码」）。"""
    return {"success": False, "error": {"code": code, "message": message}}


def success_payload(data) -> dict:
    """统一成功响应体。"""
    return {"success": True, "data": data}


# ---------------------------------------------------------------------------
# 设备选择
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """CUDA 可用则 cuda，否则 cpu（当前 GAT / BASNet 环境均为 CPU）。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 文件名安全
# ---------------------------------------------------------------------------
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def safe_original_name(filename: str) -> str:
    """去除路径部分，只保留安全文件名（拒绝路径穿越）。"""
    name = filename.replace("\\", "/").split("/")[-1]
    if not _SAFE_NAME_RE.match(name):
        # 非法字符一律拒绝，返回空让调用方决定
        return ""
    return name


def make_output_name(extension: str = "png") -> str:
    """生成唯一输出文件名，避免覆盖与碰撞。"""
    return f"{uuid.uuid4().hex}.{extension.lstrip('.').lower()}"
