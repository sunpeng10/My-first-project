# -*- coding: utf-8 -*-
"""
username → UID 映射 质量审计（第三阶段 · 第二步）

读取：
  - D:\\WeiboCrawler\\data\\weibo_user_mapping.csv    （补采结果 user_id,user_name,weibo_url）
  - D:\\WeiboCrawler\\data\\weibo.csv                 （500 条微博，author_uid 全量来源）
  - graph/mention_relations_raw.csv                  （53 个被提及用户名）

输出：
  - graph/user_mapping_audit.json

只做统计与精确字符串匹配，不猜测、不模糊匹配、不伪造。
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # visual_saliency_analysis/
MAPPING_CSV = r"D:\WeiboCrawler\data\weibo_user_mapping.csv"
WEIBO_CSV = r"D:\WeiboCrawler\data\weibo.csv"
MENTION_CSV = os.path.join(BASE, "graph", "mention_relations_raw.csv")
OUT_JSON = os.path.join(BASE, "graph", "user_mapping_audit.json")

UID_RE = re.compile(r"weibo\.com/(\d+)")


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    # ── 1. 补采映射 --------------------------------------------------------
    mapping = read_csv(MAPPING_CSV) if os.path.exists(MAPPING_CSV) else []
    uid2name = {}
    name2uids = defaultdict(set)
    empty_names = []
    dup_rows = []

    for r in mapping:
        uid = (r.get("user_id") or "").strip()
        name = (r.get("user_name") or "").strip()
        if not uid:
            continue
        if not name:
            empty_names.append(uid)
            continue
        if uid in uid2name and uid2name[uid] != name:
            dup_rows.append({"user_id": uid, "names": [uid2name[uid], name]})
        uid2name[uid] = name
        name2uids[name].add(uid)

    # uid → 多个不同 name（异常）
    uid_multi_name = {
        uid: sorted(set([uid2name[uid]] + [r["user_name"] for r in mapping if (r.get("user_id") or "").strip() == uid]))
        for uid in uid2name
        if len(set(r["user_name"] for r in mapping if (r.get("user_id") or "").strip() == uid)) > 1
    }
    # name → 多个 uid（异常）
    name_multi_uid = {name: sorted(uids) for name, uids in name2uids.items() if len(uids) > 1}

    # ── 2. 500 条微博全量覆盖 -------------------------------------------------
    posts = read_csv(WEIBO_CSV)
    all_uids = set()
    uid_to_posts = defaultdict(list)
    for r in posts:
        url = (r.get("weibo_url") or "").strip()
        m = UID_RE.search(url)
        if not m:
            continue
        uid = m.group(1)
        all_uids.add(uid)
        uid_to_posts[uid].append(r.get("image_id", ""))

    mapped_uids = set(uid2name.keys()) & all_uids
    unmapped_uids = all_uids - set(uid2name.keys())

    # 每条微博是否已有 author username
    posts_covered = sum(
        1 for r in posts
        if UID_RE.search(r.get("weibo_url", "") or "")
        and UID_RE.search(r.get("weibo_url", "") or "").group(1) in uid2name
    )
    posts_total = len(posts)

    # ── 3. mentioned_username 匹配 ------------------------------------------
    mentions = read_csv(MENTION_CSV) if os.path.exists(MENTION_CSV) else []
    mentioned_uniques = sorted({(r.get("mentioned_username") or "").strip() for r in mentions if (r.get("mentioned_username") or "").strip()})

    matched = []   # 能精确映射到 uid 的提及
    unmatched = []  # 无法映射
    for mname in mentioned_uniques:
        uids = name2uids.get(mname, set())
        if len(uids) == 1:
            matched.append({"mentioned_username": mname, "user_id": list(uids)[0]})
        elif len(uids) > 1:
            # 同名多 uid：视为不可可靠映射（不猜测）
            unmatched.append({"mentioned_username": mname, "reason": "ambiguous_multiple_uid", "uids": sorted(uids)})
        else:
            unmatched.append({"mentioned_username": mname, "reason": "not_in_collected_authors"})

    mentioned_total = len(mentioned_uniques)
    matched_count = len(matched)
    unmatched_count = len(unmatched)
    match_rate = round(matched_count / mentioned_total, 4) if mentioned_total else None

    audit = {
        "generated_by": "scripts/audit_user_mapping.py",
        "stage": "phase3_step2_username_uid_mapping",
        "inputs": {
            "weibo_csv": "D:/WeiboCrawler/data/weibo.csv",
            "mapping_csv": "D:/WeiboCrawler/data/weibo_user_mapping.csv",
            "mention_relations": "visual_saliency_analysis/graph/mention_relations_raw.csv",
        },
        "mapping_quality": {
            "total_unique_author_uid_in_weibo_csv": len(all_uids),
            "mapped_uid_count": len(mapped_uids),
            "unmapped_uid_count": len(unmapped_uids),
            "unmapped_uid_details": sorted(unmapped_uids),
            "mapping_success_rate": round(len(mapped_uids) / len(all_uids), 4) if all_uids else None,
            "empty_username_count": len(empty_names),
            "empty_username_uids": empty_names,
            "uid_with_multiple_names": uid_multi_name,
            "name_with_multiple_uids": name_multi_uid,
        },
        "post_coverage": {
            "total_posts": posts_total,
            "posts_with_author_username": posts_covered,
            "posts_without_author_username": posts_total - posts_covered,
        },
        "mentioned_username_match": {
            "mentioned_total": mentioned_total,
            "matched_mentions": matched_count,
            "unmatched_mentions": unmatched_count,
            "match_rate": match_rate,
            "matched_details": matched,
            "unmatched_details": unmatched,
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print("=" * 70)
    print("[mapping] total unique uid :", len(all_uids))
    print("[mapping] mapped uid       :", len(mapped_uids))
    print("[mapping] unmapped uid     :", len(unmapped_uids))
    print("[mapping] success rate     :", audit["mapping_quality"]["mapping_success_rate"])
    print("[mapping] empty username   :", len(empty_names))
    print("[mapping] uid multi-name   :", len(uid_multi_name))
    print("[mapping] name multi-uid   :", len(name_multi_uid))
    print("[posts]   covered          :", posts_covered, "/", posts_total)
    print("[mention] mentioned_total  :", mentioned_total)
    print("[mention] matched          :", matched_count)
    print("[mention] unmatched        :", unmatched_count)
    print("[mention] match_rate       :", match_rate)
    print("=" * 70)
    if unmatched:
        print("unmatched mentioned usernames:")
        for u in unmatched:
            print("   -", u.get("mentioned_username"), "|", u.get("reason"), u.get("uids", ""))
    print("audit written to:", OUT_JSON)


if __name__ == "__main__":
    main()
