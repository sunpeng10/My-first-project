"""
test_api.py — 后端 API 集成测试。

前置：先启动后端（在 D:\\BASNet 下 `python -m backend.app`），再运行本脚本：
    python scripts/test_api.py

自动从 visual_saliency_analysis/data/images/weibo 选取真实图片测试 /api/image，
无需手动输入路径。
"""
import glob
import json
import os
import sys

import requests

BASE = os.environ.get("API_BASE", "http://127.0.0.1:5000")

# 项目根目录（用于定位真实测试图片）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEIBO_IMAGES = os.path.join(
    _PROJECT_ROOT, "visual_saliency_analysis", "data", "images", "weibo"
)

_PASS = 0
_FAIL = 0


def check(name, ok, detail=""):
    global _PASS, _FAIL
    status = "PASS" if ok else "FAIL"
    if ok:
        _PASS += 1
    else:
        _FAIL += 1
    line = f"[{status}] {name}"
    if detail:
        line += f"  -> {detail}"
    print(line)


def get(path, **kwargs):
    return requests.get(BASE + path, timeout=120, **kwargs)


def post_json(path, payload, **kwargs):
    return requests.post(BASE + path, json=payload, timeout=120, **kwargs)


def main():
    # 1. GET /api
    r = get("/api")
    ok = r.status_code == 200 and r.json().get("name")
    check("GET /api", ok, f"status={r.status_code} name={r.json().get('name')}")

    # 2. GET /api/health
    r = get("/api/health")
    body = r.json()
    data = body.get("data", {})
    ok = r.status_code == 200 and data.get("status") == "ok"
    check(
        "GET /api/health",
        ok,
        f"status={r.status_code} text={data.get('text_model')} "
        f"image={data.get('image_model')} gat={data.get('gat_graph')}",
    )

    # 3. GET /api/graph
    r = get("/api/graph")
    body = r.json()
    g = body.get("data", {})
    ok = r.status_code == 200 and "nodes" in g and "edges" in g
    check(
        "GET /api/graph",
        ok,
        f"status={r.status_code} nodes={len(g.get('nodes', []))} edges={len(g.get('edges', []))}",
    )

    # 4. GET /api/top-nodes?limit=5
    r = get("/api/top-nodes", params={"limit": 5})
    body = r.json()
    data = body.get("data", {})
    ok = r.status_code == 200 and data.get("count") == 5
    first_uid = data.get("nodes", [{}])[0].get("uid") if data.get("nodes") else None
    check(
        "GET /api/top-nodes?limit=5",
        ok,
        f"status={r.status_code} count={data.get('count')} first_uid={first_uid}",
    )

    # 取一个真实 UID 用于 node / attention 测试
    real_uid = first_uid
    if real_uid is None:
        # 兜底：从 node_scores CSV 取
        import csv as _csv
        scores = os.path.join(_PROJECT_ROOT, "results", "gat_node_scores.csv")
        with open(scores, encoding="utf-8-sig") as f:
            real_uid = next(_csv.DictReader(f))["uid"]

    # 5. GET /api/node/<uid>
    r = get(f"/api/node/{real_uid}")
    body = r.json()
    node = body.get("data", {})
    ok = r.status_code == 200 and str(node.get("uid")) == str(real_uid)
    check(
        "GET /api/node/<uid>",
        ok,
        f"status={r.status_code} uid={node.get('uid')} username={node.get('username')}",
    )

    # 5b. 不存在的 UID -> 404
    r = get("/api/node/99999999999999999")
    ok = r.status_code == 404 and r.json().get("error", {}).get("code") == "NOT_FOUND"
    check("GET /api/node/<missing> -> 404", ok, f"status={r.status_code}")

    # 6. GET /api/attention?uid=<uid>
    r = get("/api/attention", params={"uid": real_uid, "limit": 20})
    body = r.json()
    data = body.get("data", {})
    ok = r.status_code == 200 and "edges" in data
    check(
        "GET /api/attention?uid=...",
        ok,
        f"status={r.status_code} count={data.get('count')} returned={len(data.get('edges', []))}",
    )

    # 6b. 不带 uid -> 摘要
    r = get("/api/attention", params={"limit": 10})
    body = r.json()
    data = body.get("data", {})
    ok = r.status_code == 200 and "total_edges" in data
    check(
        "GET /api/attention (summary)",
        ok,
        f"status={r.status_code} total_edges={data.get('total_edges')}",
    )

    # 7. POST /api/text
    r = post_json("/api/text", {"text": "这个手机太差了"})
    body = r.json()
    data = body.get("data", {})
    ok = (
        r.status_code == 200
        and data.get("label") in ("positive", "negative")
        and isinstance(data.get("words"), list)
        and len(data["words"]) > 0
    )
    check(
        "POST /api/text",
        ok,
        f"status={r.status_code} label={data.get('label')} "
        f"confidence={data.get('confidence')} words={len(data.get('words', []))}",
    )

    # 7b. method=saliency
    r = post_json("/api/text", {"text": "服务很好", "method": "saliency"})
    body = r.json()
    ok = r.status_code == 200 and body.get("data", {}).get("method") == "saliency"
    check("POST /api/text (saliency)", ok, f"status={r.status_code}")

    # 7c. 非法 method -> 400
    r = post_json("/api/text", {"text": "hello", "method": "lime"})
    ok = r.status_code == 400 and r.json().get("error", {}).get("code") == "INVALID_REQUEST"
    check("POST /api/text (bad method) -> 400", ok, f"status={r.status_code}")

    # 7d. 空文本 -> 400
    r = post_json("/api/text", {"text": "   "})
    ok = r.status_code == 400
    check("POST /api/text (empty) -> 400", ok, f"status={r.status_code}")

    # 8. POST /api/image（自动选真实图片）
    candidates = (
        glob.glob(os.path.join(_WEIBO_IMAGES, "*.jpg"))
        + glob.glob(os.path.join(_WEIBO_IMAGES, "*.jpeg"))
        + glob.glob(os.path.join(_WEIBO_IMAGES, "*.png"))
    )
    if candidates:
        img_path = sorted(candidates)[0]
        with open(img_path, "rb") as f:
            r = requests.post(
                BASE + "/api/image",
                files={"file": (os.path.basename(img_path), f, "image/jpeg")},
                timeout=300,
            )
        body = r.json()
        data = body.get("data", {})
        ok = (
            r.status_code == 200
            and "saliency" in data
            and "saliency_map_url" in data
            and data.get("saliency", {}).get("saliency_area_ratio") is not None
        )
        check(
            "POST /api/image",
            ok,
            f"status={r.status_code} file={os.path.basename(img_path)} "
            f"area_ratio={data.get('saliency', {}).get('saliency_area_ratio')}",
        )

        # 8b. 显著性图可访问
        sal_url = data.get("saliency_map_url", "")
        if sal_url:
            sr = get(sal_url)
            check(
                "GET /api/image/saliency/<file>",
                sr.status_code == 200 and sr.headers.get("content-type", "").startswith("image"),
                f"status={sr.status_code} type={sr.headers.get('content-type')}",
            )
    else:
        check("POST /api/image", False, "no weibo image found")

    # 8c. 非法扩展名 -> 400
    r = requests.post(
        BASE + "/api/image",
        files={"file": ("evil.txt", b"not an image", "text/plain")},
        timeout=60,
    )
    ok = r.status_code == 400
    check("POST /api/image (bad ext) -> 400", ok, f"status={r.status_code}")

    # 9. CORS 头存在
    r = get("/api")
    allow = r.headers.get("Access-Control-Allow-Origin")
    check("CORS header present", allow is not None, f"origin={allow}")

    # 汇总
    print("\n" + "=" * 50)
    print(f"结果: {_PASS} passed, {_FAIL} failed")
    print("=" * 50)
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
