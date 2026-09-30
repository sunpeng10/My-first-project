"""
merge_visual_propagation_dataset.py — 微博视觉传播力数据集合并工具

功能：
    1. 根据 image_id 合并 weibo_metadata.csv（互动指标）与 visual_features.csv（视觉特征）
    2. 输出合并后的完整数据集: results/weibo_visual_dataset.csv
    3. 输出合并报告: outputs/dataset_merge_report.txt

运行命令（从项目根目录执行）：
    python visual_saliency_analysis/scripts/merge_visual_propagation_dataset.py

依赖：
    pip install pandas
"""

import os
import sys
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_BASE_DIR = os.path.join(_PROJECT_ROOT, 'visual_saliency_analysis')

META_CSV    = os.path.join(_BASE_DIR, 'data', 'weibo_metadata.csv')          # 互动指标
FEATURE_CSV = os.path.join(_BASE_DIR, 'features', 'visual_features.csv')     # 视觉特征
OUTPUT_CSV  = os.path.join(_BASE_DIR, 'results', 'weibo_visual_dataset.csv') # 合并输出
REPORT_TXT  = os.path.join(_BASE_DIR, 'outputs', 'dataset_merge_report.txt') # 合并报告


# ===========================================================================
# 工具函数
# ===========================================================================

def load_csv(path: str, name: str) -> pd.DataFrame:
    """加载 CSV 并验证 image_id 列存在。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] {name} 不存在: {path}")
    df = pd.read_csv(path, dtype=str).fillna('')
    if 'image_id' not in df.columns:
        raise ValueError(f"[ERROR] {name} 缺少 image_id 列。")
    return df


def validate_dataset(df_meta: pd.DataFrame, df_feat: pd.DataFrame, report_lines: list):
    """验证两个数据集的一致性。"""
    ids_meta = set(df_meta['image_id'].str.strip())
    ids_feat = set(df_feat['image_id'].str.strip())

    # 数量
    report_lines.append(f"weibo_metadata.csv:          {len(df_meta)} 条")
    report_lines.append(f"visual_features.csv:         {len(df_feat)} 条")
    report_lines.append(f"image_id 是否一致:           {'是' if len(df_meta) == len(df_feat) else '否 — 数量不同!'}")

    # 缺失检查
    missing_in_feat = ids_meta - ids_feat
    missing_in_meta = ids_feat - ids_meta
    report_lines.append(f"metadata 有但 features 缺失: {len(missing_in_feat)}")
    report_lines.append(f"features 有但 metadata 缺失: {len(missing_in_meta)}")

    # 重复检查
    dup_meta = df_meta['image_id'][df_meta['image_id'].duplicated()]
    dup_feat = df_feat['image_id'][df_feat['image_id'].duplicated()]
    report_lines.append(f"metadata 重复 image_id:       {len(dup_meta)}")
    report_lines.append(f"features 重复 image_id:       {len(dup_feat)}")

    # 打印缺失详情
    if missing_in_feat:
        report_lines.append(f"\n  metadata 中缺失于 features 的 ID ({len(missing_in_feat)}):")
        for wid in sorted(missing_in_feat)[:20]:
            report_lines.append(f"    - {wid}")
        if len(missing_in_feat) > 20:
            report_lines.append(f"    ... 共 {len(missing_in_feat)} 个")

    if missing_in_meta:
        report_lines.append(f"\n  features 中缺失于 metadata 的 ID ({len(missing_in_meta)}):")
        for wid in sorted(missing_in_meta)[:20]:
            report_lines.append(f"    - {wid}")

    if dup_meta.any():
        report_lines.append(f"\n  metadata 重复 ID: {', '.join(dup_meta.tolist())}")
    if dup_feat.any():
        report_lines.append(f"\n  features 重复 ID: {', '.join(dup_feat.tolist())}")


# ===========================================================================
# 主流程
# ===========================================================================

if __name__ == '__main__':
    report = []
    report.append(f"微博视觉传播力数据集 — 合并报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 55)
    report.append("")

    # -----------------------------------------------------------------------
    # 1. 加载两个 CSV
    # -----------------------------------------------------------------------
    print("[INFO] 加载 CSV...")
    try:
        df_meta  = load_csv(META_CSV, 'weibo_metadata.csv')
        df_feat  = load_csv(FEATURE_CSV, 'visual_features.csv')
    except Exception as e:
        print(e)
        sys.exit(1)

    # 统一 image_id 格式（去空格）
    df_meta['image_id'] = df_meta['image_id'].str.strip()
    df_feat['image_id'] = df_feat['image_id'].str.strip()

    # -----------------------------------------------------------------------
    # 2. 验证
    # -----------------------------------------------------------------------
    print("[INFO] 验证数据集...")
    report.append("【数据验证】")
    validate_dataset(df_meta, df_feat, report)

    # -----------------------------------------------------------------------
    # 3. 合并
    # -----------------------------------------------------------------------
    print("[INFO] 合并数据集...")
    report.append("")
    report.append("【合并结果】")

    # 将 visual_features 的数值列转为 float
    for col in df_feat.columns:
        if col != 'image_id':
            df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce')

    # inner join on image_id
    merged = df_meta.merge(df_feat, on='image_id', how='inner')

    report.append(f"合并方式:                   INNER JOIN (取交集)")
    report.append(f"合并后记录数:               {len(merged)}")
    report.append(f"合并后字段数:               {len(merged.columns)}")
    report.append(f"字段列表:                   {', '.join(merged.columns)}")

    # 最终字段顺序
    final_columns = [
        'image_id', 'text',
        'likes', 'comments', 'shares',
        'saliency_area_ratio', 'mean_saliency', 'saliency_std',
        'saliency_entropy', 'center_bias', 'component_count',
        'largest_component_ratio',
    ]
    merged = merged[final_columns]

    # -----------------------------------------------------------------------
    # 4. 保存
    # -----------------------------------------------------------------------
    print("[INFO] 保存结果...")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    report.append(f"输出文件:                   {OUTPUT_CSV}")

    # 数值摘要
    report.append("")
    report.append("【数值字段统计】")
    num_cols = ['saliency_area_ratio', 'mean_saliency', 'saliency_std',
                'saliency_entropy', 'center_bias', 'component_count', 'largest_component_ratio']
    for col in num_cols:
        s = merged[col]
        report.append(f"  {col:30s}  mean={s.mean():.4f}  std={s.std():.4f}  min={s.min():.4f}  max={s.max():.4f}")

    # 保存报告
    os.makedirs(os.path.dirname(REPORT_TXT), exist_ok=True)
    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\n[DONE] 合并完成")
    print(f"  metadata:     {len(df_meta)} 条")
    print(f"  features:     {len(df_feat)} 条")
    print(f"  合并后:       {len(merged)} 条 × {len(merged.columns)} 字段")
    print(f"  输出:         {OUTPUT_CSV}")
    print(f"  报告:         {REPORT_TXT}")

    # 预览前 3 行
    print(f"\n  [前3行预览]")
    # Hide long text column for cleaner output
    preview = merged.head(3).copy()
    if 'text' in preview.columns:
        preview['text'] = preview['text'].str[:30].str.encode('ascii', 'replace').str.decode('ascii') + '...'
    print(preview.to_string(index=False))
