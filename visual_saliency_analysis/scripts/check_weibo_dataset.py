"""
check_weibo_dataset.py — 微博数据集完整性检查工具

功能：
    1. 统计 CSV 中的记录总数
    2. 统计已下载的图片数量
    3. 检查缺失图片（CSV 有记录但无对应图片）
    4. 检查孤儿图片（图片存在但 CSV 无记录）
    5. 验证 image_id 命名一致性

使用方式：
    python visual_saliency_analysis/scripts/check_weibo_dataset.py

依赖：
    pip install pandas
"""

import os
import sys
import glob
import pandas as pd

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_BASE_DIR = os.path.join(_PROJECT_ROOT, 'visual_saliency_analysis')

CSV_PATH    = os.path.join(_BASE_DIR, 'data', 'weibo_collection.csv')
IMG_DIR     = os.path.join(_BASE_DIR, 'data', 'images', 'weibo')
IMAGE_EXTS  = ('.jpg', '.jpeg', '.png', '.gif', '.webp')


# ===========================================================================
# 主流程
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  微博数据集完整性检查")
    print("=" * 60)

    issues = []

    # -----------------------------------------------------------------------
    # 1. CSV 检查
    # -----------------------------------------------------------------------
    print(f"\n[1] CSV 文件检查")
    print(f"   路径: {CSV_PATH}")

    if not os.path.exists(CSV_PATH):
        print(f"   [FAIL] CSV 不存在！")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, dtype=str).fillna('')
    csv_count = len(df)
    print(f"   记录总数: {csv_count}")

    # 必填字段检查
    required_fields = ['image_id', 'image_url']
    for field in required_fields:
        if field not in df.columns:
            issues.append(f"CSV 缺少必填字段: {field}")
            print(f"   [FAIL] 缺少必填字段: {field}")
        else:
            # 非空计数
            filled = df[field].str.strip().replace('', pd.NA).notna().sum()
            print(f"   {field}: {filled}/{csv_count} 已填写")

    # image_id 重复检查
    if 'image_id' in df.columns:
        ids = df['image_id'].str.strip()
        dup_ids = ids[ids.duplicated()].unique()
        if len(dup_ids) > 0:
            issues.append(f"{len(dup_ids)} 个重复 image_id: {', '.join(dup_ids)}")
            print(f"   [WARN] 重复 image_id: {', '.join(dup_ids)}")
        else:
            print(f"   image_id 无重复: OK")

    # 获取所有 CSV 中的 image_id（非空）
    csv_ids = set(df['image_id'].str.strip().tolist()) - {''}

    # -----------------------------------------------------------------------
    # 2. 图片文件检查
    # -----------------------------------------------------------------------
    print(f"\n[2] 图片文件检查")
    print(f"   目录: {IMG_DIR}")

    if not os.path.isdir(IMG_DIR):
        img_files = []
        print(f"   [WARN] 图片目录不存在，将自动创建。")
        os.makedirs(IMG_DIR, exist_ok=True)
    else:
        img_files = []
        for ext in IMAGE_EXTS:
            img_files.extend(glob.glob(os.path.join(IMG_DIR, f'*{ext}')))

    img_count = len(img_files)
    print(f"   已下载图片: {img_count} 张")

    for f in sorted(img_files):
        basename = os.path.basename(f)
        size_kb = os.path.getsize(f) / 1024
        print(f"     {basename}  ({size_kb:.1f} KB)")

    # 提取图片文件名中的 image_id（去掉扩展名）
    disk_img_ids = set()
    for f in img_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        disk_img_ids.add(stem)

    # -----------------------------------------------------------------------
    # 3. 交叉比对
    # -----------------------------------------------------------------------
    print(f"\n[3] 交叉比对")

    # CSV 有、图片缺失
    missing_images = csv_ids - disk_img_ids
    if missing_images:
        print(f"   [WARN] CSV 中 {len(missing_images)} 条记录无对应图片:")
        for mid in sorted(missing_images):
            row = df[df['image_id'].str.strip() == mid]
            has_url = row['image_url'].str.strip().values[0] if 'image_url' in df.columns else ''
            url_status = '[有 image_url, 需下载]' if has_url else '[无 image_url]'
            print(f"         {mid}  {url_status}")
            issues.append(f"缺失图片: {mid} ({url_status})")
    else:
        print(f"   CSV → 图片: 全部匹配")

    # 图片存在、CSV 无记录（孤儿文件）
    orphan_images = disk_img_ids - csv_ids
    if orphan_images:
        print(f"   [WARN] {len(orphan_images)} 张孤儿图片（CSV 中无记录）:")
        for oid in sorted(orphan_images):
            print(f"         {oid}")
            issues.append(f"孤儿图片: {oid}")
    else:
        print(f"   孤儿图片: 无")

    # -----------------------------------------------------------------------
    # 4. 汇总
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  检查汇总")
    print(f"{'=' * 60}")
    print(f"  CSV 记录:        {csv_count}")
    print(f"  已下载图片:       {img_count}")
    print(f"  缺失图片:         {len(missing_images)}")
    print(f"  孤儿图片:         {len(orphan_images)}")
    print(f"  发现问题:         {len(issues)}")

    if issues:
        print(f"\n  问题详情:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print(f"\n  数据集状态: 健康 ✓")
