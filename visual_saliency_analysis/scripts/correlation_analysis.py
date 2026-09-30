"""
correlation_analysis.py — 视觉特征与传播指标的相关性分析

功能：
    1. 读取 weibo_visual_dataset.csv（含传播原始值 + 视觉特征）
    2. 内部计算对数变换 (likes_log / comments_log / shares_log / engagement_score)
    3. 计算 Pearson 和 Spearman 相关系数 (视觉特征 vs 传播指标)
    4. 输出: results/correlation_analysis.csv
    5. 生成: results/correlation_heatmap.png (相关性热力图)
    6. 输出: outputs/statistics/significant_correlations.txt (显著结果排序)

运行命令（从项目根目录执行）：
    python visual_saliency_analysis/scripts/correlation_analysis.py

依赖：
    pip install pandas numpy scipy matplotlib seaborn
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# matplotlib 配置
# ---------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_BASE_DIR = os.path.join(_PROJECT_ROOT, 'visual_saliency_analysis')

INPUT_CSV     = os.path.join(_BASE_DIR, 'results', 'weibo_visual_dataset.csv')
OUTPUT_CSV    = os.path.join(_BASE_DIR, 'results', 'correlation_analysis.csv')
HEATMAP_PNG   = os.path.join(_BASE_DIR, 'results', 'correlation_heatmap.png')
SIGNIFICANT_TXT = os.path.join(_BASE_DIR, 'outputs', 'statistics', 'significant_correlations.txt')

# ---------------------------------------------------------------------------
# 指标定义
# ---------------------------------------------------------------------------
# 视觉特征 (7 个)
VISUAL_FEATURES = [
    'saliency_area_ratio',
    'mean_saliency',
    'saliency_std',
    'saliency_entropy',
    'center_bias',
    'component_count',
    'largest_component_ratio',
]

# 传播指标 — 原始值 (3 个) + 对数变换 (3 个) + 综合评分 (1 个)
PROPAGATION_RAW = ['likes', 'comments', 'shares']
PROPAGATION_LOG = ['likes_log', 'comments_log', 'shares_log']
PROPAGATION_TARGETS = PROPAGATION_LOG + ['engagement_score']

# 中文标签映射
LABEL_MAP = {
    # 视觉特征
    'saliency_area_ratio':      '显著性面积比',
    'mean_saliency':            '平均显著性',
    'saliency_std':             '显著性标准差',
    'saliency_entropy':         '显著性熵',
    'center_bias':              '中心偏置',
    'component_count':          '连通分量数',
    'largest_component_ratio':  '最大分量比',
    # 传播指标
    'likes_log':                '点赞(log)',
    'comments_log':             '评论(log)',
    'shares_log':               '转发(log)',
    'engagement_score':         '综合传播力',
}


# ===========================================================================
# 主流程
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  视觉特征 vs 传播指标 — 相关性分析")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # -------------------------------------------------------------------
    # 1. 加载数据
    # -------------------------------------------------------------------
    print(f"\n[1/5] 加载数据: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] 文件不存在: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
    print(f"      记录数: {len(df)}, 字段数: {len(df.columns)}")

    # 确保数值类型
    for col in PROPAGATION_RAW + VISUAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 检查必需列
    missing_raw = [c for c in PROPAGATION_RAW if c not in df.columns]
    missing_vis = [c for c in VISUAL_FEATURES if c not in df.columns]
    if missing_raw:
        print(f"[ERROR] 缺少传播指标列: {missing_raw}")
        sys.exit(1)
    if missing_vis:
        print(f"[ERROR] 缺少视觉特征列: {missing_vis}")
        sys.exit(1)

    # -------------------------------------------------------------------
    # 2. 计算对数变换 & engagement_score
    # -------------------------------------------------------------------
    print(f"\n[2/5] 计算传播对数变换 + engagement_score...")
    for col in PROPAGATION_RAW:
        log_col = f'{col}_log'
        df[log_col] = np.log1p(df[col].clip(lower=0))

    df['engagement_score'] = (
        df['likes_log'] + 2 * df['comments_log'] + 3 * df['shares_log']
    )

    # 去除含 NaN 的行（如有）
    before = len(df)
    df = df.dropna(subset=VISUAL_FEATURES + PROPAGATION_TARGETS)
    if len(df) < before:
        print(f"      去除含 NaN 行: {before} -> {len(df)}")

    print(f"      有效样本: {len(df)}")

    # -------------------------------------------------------------------
    # 3. 计算 Pearson & Spearman 相关系数
    # -------------------------------------------------------------------
    print(f"\n[3/5] 计算相关系数...")

    records = []
    for feat in VISUAL_FEATURES:
        for tgt in PROPAGATION_TARGETS:
            x = df[feat].values
            y = df[tgt].values

            # 去除双方任一方为 NaN 的样本
            mask = np.isfinite(x) & np.isfinite(y)
            x_clean = x[mask]
            y_clean = y[mask]

            if len(x_clean) < 3:
                records.append({
                    'feature': feat,
                    'target': tgt,
                    'pearson_r': np.nan,
                    'pearson_p': np.nan,
                    'spearman_r': np.nan,
                    'spearman_p': np.nan,
                    'n': len(x_clean),
                })
                continue

            # Pearson
            r_pearson, p_pearson = stats.pearsonr(x_clean, y_clean)

            # Spearman
            r_spearman, p_spearman = stats.spearmanr(x_clean, y_clean)

            records.append({
                'feature': feat,
                'target': tgt,
                'pearson_r':  round(r_pearson, 6),
                'pearson_p':  round(p_pearson, 6),
                'spearman_r': round(r_spearman, 6),
                'spearman_p': round(p_spearman, 6),
                'n': len(x_clean),
            })

    corr_df = pd.DataFrame(records)

    # -------------------------------------------------------------------
    # 4. 保存结果 CSV
    # -------------------------------------------------------------------
    print(f"\n[4/5] 保存结果...")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    corr_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"      输出: {OUTPUT_CSV}  ({len(corr_df)} 行 = 7 视觉特征 × 4 传播指标)")

    # -------------------------------------------------------------------
    # 5. 生成显著结果报告（按 |spearman_r| 降序）
    # -------------------------------------------------------------------
    sig_df = corr_df.copy()
    sig_df['abs_spearman_r'] = sig_df['spearman_r'].abs()
    sig_df = sig_df.sort_values('abs_spearman_r', ascending=False)

    os.makedirs(os.path.dirname(SIGNIFICANT_TXT), exist_ok=True)
    with open(SIGNIFICANT_TXT, 'w', encoding='utf-8') as f:
        f.write("视觉特征 vs 传播指标 — 显著相关性排序\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"排序依据: |Spearman r| 降序\n")
        f.write(f"样本量: {len(df)}\n")
        f.write("=" * 75 + "\n\n")

        f.write(f"{'排名':<5} {'视觉特征':<25} {'传播指标':<20} "
                f"{'Spearman r':>10} {'p值':>10} {'Pearson r':>10} {'显著性':<8}\n")
        f.write("-" * 75 + "\n")

        for rank, (_, row) in enumerate(sig_df.iterrows(), 1):
            sig_mark = ''
            if row['spearman_p'] < 0.001:
                sig_mark = '***'
            elif row['spearman_p'] < 0.01:
                sig_mark = '**'
            elif row['spearman_p'] < 0.05:
                sig_mark = '*'

            feat_label = LABEL_MAP.get(row['feature'], row['feature'])
            tgt_label  = LABEL_MAP.get(row['target'], row['target'])

            f.write(f"{rank:<5} {feat_label:<25} {tgt_label:<20} "
                    f"{row['spearman_r']:>10.4f} {row['spearman_p']:>10.4f} "
                    f"{row['pearson_r']:>10.4f} {sig_mark:<8}\n")

        # 汇总
        n_sig_05  = (sig_df['spearman_p'] < 0.05).sum()
        n_sig_01  = (sig_df['spearman_p'] < 0.01).sum()
        n_sig_001 = (sig_df['spearman_p'] < 0.001).sum()
        f.write(f"\n{'─' * 75}\n")
        f.write(f"汇总:\n")
        f.write(f"  总测试数:            {len(sig_df)}\n")
        f.write(f"  p < 0.05  (*):       {n_sig_05}\n")
        f.write(f"  p < 0.01  (**):      {n_sig_01}\n")
        f.write(f"  p < 0.001 (***):     {n_sig_001}\n")

    print(f"      显著结果报告: {SIGNIFICANT_TXT}")

    # -------------------------------------------------------------------
    # 6. 终端输出摘要
    # -------------------------------------------------------------------
    print(f"\n{'─' * 75}")
    print(f"  显著相关性排序 (Top 10 |Spearman r|)")
    print(f"{'─' * 75}")
    for rank, (_, row) in enumerate(sig_df.head(10).iterrows(), 1):
        sig_mark = '***' if row['spearman_p'] < 0.001 else \
                   '**'  if row['spearman_p'] < 0.01  else \
                   '*'   if row['spearman_p'] < 0.05  else ''
        feat_label = LABEL_MAP.get(row['feature'], row['feature'])
        tgt_label  = LABEL_MAP.get(row['target'], row['target'])
        print(f"  {rank:2}. {feat_label:<20} × {tgt_label:<16}  "
              f"Spearman r={row['spearman_r']:+.4f}  p={row['spearman_p']:.4f}  {sig_mark}")

    # -------------------------------------------------------------------
    # 7. 绘制相关性热力图
    # -------------------------------------------------------------------
    print(f"\n[5/5] 绘制相关性热力图...")

    # 构建热力图矩阵: rows=视觉特征, cols=传播指标
    heatmap_data = corr_df.pivot_table(
        index='feature', columns='target', values='spearman_r'
    )
    # 按指定顺序排列
    heatmap_data = heatmap_data.reindex(
        index=VISUAL_FEATURES, columns=PROPAGATION_TARGETS
    )

    # 显著性标注矩阵
    pvt_p = corr_df.pivot_table(
        index='feature', columns='target', values='spearman_p'
    )
    pvt_p = pvt_p.reindex(index=VISUAL_FEATURES, columns=PROPAGATION_TARGETS)

    # 构建标注文本
    annot = heatmap_data.astype(object).copy()
    for fi in VISUAL_FEATURES:
        for ti in PROPAGATION_TARGETS:
            val = heatmap_data.loc[fi, ti]
            pval = pvt_p.loc[fi, ti]
            if pd.isna(val):
                annot.loc[fi, ti] = ''
            else:
                stars = '***' if pval < 0.001 else \
                        '**'  if pval < 0.01  else \
                        '*'   if pval < 0.05  else ''
                annot.loc[fi, ti] = f'{val:.3f}\n{stars}' if stars else f'{val:.3f}'

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))

    # 用中文标签重命名行列
    y_labels = [LABEL_MAP.get(c, c) for c in heatmap_data.index]
    x_labels = [LABEL_MAP.get(c, c) for c in heatmap_data.columns]

    sns.heatmap(
        heatmap_data.values,
        annot=annot.values,
        fmt='',
        cmap='RdBu_r',
        center=0,
        vmin=-0.5,
        vmax=0.5,
        xticklabels=x_labels,
        yticklabels=y_labels,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Spearman r', 'shrink': 0.8},
        ax=ax,
        annot_kws={'fontsize': 9},
    )

    ax.set_title('视觉特征 × 传播指标 Spearman 相关性热力图\n'
                 f'(n={len(df)}, * p<0.05, ** p<0.01, *** p<0.001)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel('传播指标', fontsize=11)
    ax.set_ylabel('视觉特征', fontsize=11)
    ax.tick_params(axis='both', labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    fig.tight_layout()
    fig.savefig(HEATMAP_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"      热力图: {HEATMAP_PNG}")

    # -------------------------------------------------------------------
    # DONE
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  相关性分析完成")
    print(f"  CSV:      {OUTPUT_CSV}")
    print(f"  热力图:   {HEATMAP_PNG}")
    print(f"  显著报告: {SIGNIFICANT_TXT}")
    print(f"{'=' * 60}")
