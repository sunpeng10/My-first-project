# -*- coding: utf-8 -*-
"""
传播网络数据准备（第三阶段 · 第一步）
====================================

只读输入，生成第一版网络数据到 graph/ 目录：

  1. post_authors.csv          作者 UID 表（从 weibo_url 正则提取）
  2. mention_relations_raw.csv @提及关系原始表
  3. network_data_audit.json   审计统计
  4. README.md                 网络数据说明（另由人工撰写）

严禁：
  - 修改 D:\\WeiboCrawler\\data\\weibo.csv
  - 修改 weibo_metadata.csv / weibo_visual_dataset.csv
  - 把 shares/comments 数值转成 Edge
  - 猜测或伪造 username -> UID 映射
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # visual_saliency_analysis/
META_CSV = os.path.join(BASE, "data", "weibo_metadata.csv")          # 权威 post 数据（无 weibo_url）
WIBO_CSV = r"D:\WeiboCrawler\data\weibo.csv"                        # weibo_url 来源（只读）
OUT_DIR = os.path.join(BASE, "graph")
os.makedirs(OUT_DIR, exist_ok=True)

# --- 正则 ---------------------------------------------------------------
# weibo_url 形如 https://weibo.com/{UID}/{POST_ID}，提取 UID
URL_UID_RE = re.compile(r"weibo\.com/(\d+)")

# @提及：@ + 用户名（中文/字母/数字/下划线/连字符/中间点/叠字符号）
# 注意：不匹配 @ 后跟空白或标点（视为无效/残缺提及）
MENTION_RE = re.compile(r"@([\w\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\u00b7\u3005-]+)")

# 标准微博域名，用于 URL 异常检测
URL_HOST_RE = re.compile(r"^https?://(?:www\.)?weibo\.com/", re.IGNORECASE)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    meta = read_csv(META_CSV)
    weibo = read_csv(WIBO_CSV)

    url_by_id = {r["image_id"]: r.get("weibo_url", "") for r in weibo}
    text_by_id = {r["image_id"]: r["text"] for r in meta}

    # --- 1. 作者 UID 提取 -------------------------------------------------
    post_authors = []          # image_id, weibo_url, author_uid, text
    uid_fail = []              # 提取失败的记录
    url_anomaly = []           # 异常 URL
    author_posts = Counter()

    for r in meta:
        iid = r["image_id"]
        url = url_by_id.get(iid, "")
        text = text_by_id.get(iid, "")
        uid = ""

        if not url:
            url_anomaly.append({"image_id": iid, "reason": "empty_weibo_url"})
        elif not URL_HOST_RE.match(url):
            url_anomaly.append({"image_id": iid, "reason": "non_weibo_domain", "url": url})
            m = URL_UID_RE.search(url)  # 仍尝试提取纯数字段
            uid = m.group(1) if m else ""
        else:
            m = URL_UID_RE.search(url)
            uid = m.group(1) if m else ""

        if not uid:
            uid_fail.append({"image_id": iid, "url": url})

        if uid:
            author_posts[uid] += 1

        post_authors.append({
            "image_id": iid,
            "weibo_url": url,
            "author_uid": uid,
            "text": text,
        })

    # --- 2. @提及提取 -----------------------------------------------------
    mention_rows = []          # image_id, source_uid, mentioned_username, mention_count, relation
    posts_with_mention = 0
    total_mentions = 0         # 出现次数（含同一微博内重复）
    mentioned_counter = Counter()       # 被提及用户名 -> 出现次数（跨微博累计）
    mention_per_post = Counter()        # 每条微博的（去重后）提及数量

    for pa in post_authors:
        iid = pa["image_id"]
        src_uid = pa["author_uid"]
        text = pa["text"]
        found = MENTION_RE.findall(text)
        if not found:
            continue
        posts_with_mention += 1
        total_mentions += len(found)
        per_post = Counter(found)
        mention_per_post[len(per_post)] += 1
        for name, cnt in per_post.items():
            mentioned_counter[name] += cnt
            mention_rows.append({
                "image_id": iid,
                "source_uid": src_uid,
                "mentioned_username": name,
                "mention_count": cnt,
                "relation": "mention",
            })

    unique_mentioned = len(mentioned_counter)

    # --- 3. 统计汇总 -------------------------------------------------------
    author_post_dist = Counter(author_posts.values())   # 发帖数 -> 作者数
    audit = {
        "generated_by": "scripts/build_network_data.py",
        "stage": "phase3_step1_network_data_audit",
        "inputs": {
            "post_metadata": "visual_saliency_analysis/data/weibo_metadata.csv",
            "weibo_url_source": "D:/WeiboCrawler/data/weibo.csv (read-only join by image_id)",
        },
        "posts": {
            "total_posts": len(meta),
            "unique_author_uid": len(author_posts),
            "author_post_distribution": {str(k): v for k, v in sorted(author_post_dist.items())},
            "top_authors_by_post_count": [
                {"author_uid": u, "post_count": c}
                for u, c in author_posts.most_common(20)
            ],
        },
        "uid_extraction": {
            "success": len(meta) - len(uid_fail),
            "failed": len(uid_fail),
            "url_anomalies": len(url_anomaly),
            "failed_details": uid_fail,
            "url_anomaly_details": url_anomaly,
        },
        "mentions": {
            "posts_with_mention": posts_with_mention,
            "posts_without_mention": len(meta) - posts_with_mention,
            "total_mention_occurrences": total_mentions,
            "unique_mentioned_username": unique_mentioned,
            "per_post_mention_distribution": {str(k): v for k, v in sorted(mention_per_post.items())},
            "top20_mentioned": [
                {"mentioned_username": n, "count": c}
                for n, c in mentioned_counter.most_common(20)
            ],
        },
        "username_to_uid_mapping": {
            "author_has_username_field": False,
            "mentioned_username_to_uid_reliable": False,
            "reason": (
                "weibo_metadata.csv 无用户名(user_name/screen_name)字段，"
                "text 中的 @用户名 是自由文本，缺少官方 username->uid 索引，"
                "因此无法可靠地把被提及用户名映射到 UID；不得猜测或伪造。"
            ),
        },
        "outputs": {
            "post_authors": "graph/post_authors.csv",
            "mention_relations": "graph/mention_relations_raw.csv",
        },
    }

    # --- 4. 写出文件 -------------------------------------------------------
    def write_csv(path, fieldnames, rows):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    write_csv(
        os.path.join(OUT_DIR, "post_authors.csv"),
        ["image_id", "weibo_url", "author_uid", "text"],
        post_authors,
    )
    write_csv(
        os.path.join(OUT_DIR, "mention_relations_raw.csv"),
        ["image_id", "source_uid", "mentioned_username", "mention_count", "relation"],
        mention_rows,
    )
    with open(os.path.join(OUT_DIR, "network_data_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    # --- 5. 控制台摘要 -------------------------------------------------------
    print("=" * 70)
    print("[posts] total            :", len(meta))
    print("[posts] unique author_uid:", len(author_posts))
    print("[uid]   success          :", len(meta) - len(uid_fail))
    print("[uid]   failed           :", len(uid_fail))
    print("[uid]   url anomalies    :", len(url_anomaly))
    print("[mention] posts with @   :", posts_with_mention)
    print("[mention] total occurrences:", total_mentions)
    print("[mention] unique usernames:", unique_mentioned)
    print("[mention] per-post dist  :", dict(sorted(mention_per_post.items())))
    print("[mention] top20          :")
    for n, c in mentioned_counter.most_common(20):
        print(f"    {c:5d}  {n}")
    print("[uid] fail details       :", uid_fail if uid_fail else "none")
    print("[url] anomaly details    :", url_anomaly if url_anomaly else "none")
    print("=" * 70)
    print("outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
