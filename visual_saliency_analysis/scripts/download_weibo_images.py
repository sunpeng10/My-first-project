"""
download_weibo_images.py — 微博图片批量下载工具

功能：
    1. 读取 data/weibo_collection.csv 中的 image_url 字段
    2. 逐条下载图片，保存至 data/images/weibo/
    3. 按 image_id 命名，如 wb_0001.jpg
    4. 自动跳过已存在的图片，支持断点续传
    5. 下载失败时记录原因，不中断后续任务

使用方式：
    # 前提：在 weibo_collection.csv 中填写 image_url 列
    python visual_saliency_analysis/scripts/download_weibo_images.py

    # 可选：限制下载数量（测试）
    python visual_saliency_analysis/scripts/download_weibo_images.py --limit 10

依赖：
    pip install pandas requests pillow
"""

import os
import sys
import argparse
import time
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_BASE_DIR = os.path.join(_PROJECT_ROOT, 'visual_saliency_analysis')

CSV_PATH    = os.path.join(_BASE_DIR, 'data', 'weibo_collection.csv')       # 采集模板
IMG_OUT_DIR = os.path.join(_BASE_DIR, 'data', 'images', 'weibo')            # 图片保存目录

# 下载超时（秒）
TIMEOUT = 30
# 请求间隔（秒），避免被封
DELAY = 1.0
# User-Agent，模拟浏览器
HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}


# ===========================================================================
# 工具函数
# ===========================================================================

def sanitize_filename(filename: str) -> str:
    """移除文件名中的非法字符。"""
    invalid_chars = '<>:"/\\|?*'
    for c in invalid_chars:
        filename = filename.replace(c, '_')
    return filename


def download_image(image_url: str, save_path: str, image_id: str) -> tuple[bool, str]:
    """
    从 URL 下载图片并保存到指定路径。

    Args:
        image_url: 图片的远程 URL
        save_path: 本地保存路径（含扩展名）
        image_id:  图片编号，用于日志

    Returns:
        (success, message): 是否成功 + 说明信息
    """
    # 跳过已存在的文件
    if os.path.exists(save_path):
        return True, "已存在，跳过"

    try:
        resp = requests.get(image_url, headers=HEADERS, timeout=TIMEOUT)

        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"

        content_type = resp.headers.get('Content-Type', '')
        if 'image' not in content_type and len(resp.content) < 1024:
            return False, f"非图片响应 ({content_type})"

        # 验证是有效图片
        img = Image.open(BytesIO(resp.content))
        img.verify()

        # 保存（重新打开，因为 verify() 后不能 save）
        img = Image.open(BytesIO(resp.content))
        # 统一转为 RGB（处理 RGBA / P 模式）
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        img.save(save_path)

        return True, f"OK ({os.path.getsize(save_path) / 1024:.1f} KB)"

    except requests.exceptions.Timeout:
        return False, "超时"
    except requests.exceptions.ConnectionError:
        return False, "连接失败"
    except Exception as e:
        return False, str(e)[:80]


# ===========================================================================
# 主流程
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="微博图片批量下载工具")
    parser.add_argument('--limit', type=int, default=0,
                        help='最大下载数量（0=全部）')
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. 读取 CSV
    # -----------------------------------------------------------------------
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] 采集模板未找到: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, dtype=str).fillna('')

    if 'image_url' not in df.columns:
        print("[ERROR] CSV 缺少 image_url 字段，请检查模板格式。")
        sys.exit(1)

    # 过滤出有 image_url 的行
    df_todo = df[df['image_url'].str.strip() != ''].copy()
    total = len(df_todo)

    if total == 0:
        print("[WARN] CSV 中没有任何 image_url 数据，请先填写下载链接。")
        sys.exit(0)

    if args.limit > 0:
        df_todo = df_todo.head(args.limit)
        total = len(df_todo)

    print(f"[INFO] 待下载: {total} 张图片")
    print(f"[INFO] 保存目录: {IMG_OUT_DIR}")

    # 确保输出目录存在
    os.makedirs(IMG_OUT_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # 2. 逐条下载
    # -----------------------------------------------------------------------
    success_count = 0
    skip_count = 0
    fail_list = []

    for idx, row in df_todo.iterrows():
        image_id  = row['image_id'].strip()
        image_url = row['image_url'].strip()

        if not image_id:
            fail_list.append((idx, '', 'image_id 为空'))
            continue

        # 确定文件扩展名
        if '.' in image_url.split('/')[-1]:
            ext = os.path.splitext(image_url.split('/')[-1])[1].lower()
            # 去掉 URL 参数
            ext = ext.split('?')[0]
            if ext not in ('.jpg', '.jpeg', '.png', '.gif', '.webp'):
                ext = '.jpg'
        else:
            ext = '.jpg'

        filename = sanitize_filename(f"{image_id}{ext}")
        save_path = os.path.join(IMG_OUT_DIR, filename)

        # 下载
        print(f"  [{idx + 1}/{len(df)}] {image_id} ... ", end='', flush=True)
        ok, msg = download_image(image_url, save_path, image_id)
        print(msg)

        if ok:
            if "跳过" in msg:
                skip_count += 1
            else:
                success_count += 1
        else:
            fail_list.append((idx + 1, image_id, msg))

        # 请求间隔
        if idx < len(df_todo) - 1:
            time.sleep(DELAY)

    # -----------------------------------------------------------------------
    # 3. 汇总
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"[DONE] 下载完成: 成功 {success_count}, 跳过 {skip_count}, 失败 {len(fail_list)}")
    print(f"[INFO] 图片存放: {IMG_OUT_DIR}")

    if fail_list:
        print(f"\n[FAIL] 失败列表:")
        for row_idx, img_id, reason in fail_list:
            print(f"  - 第 {row_idx} 行, {img_id}: {reason}")
