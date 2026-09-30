"""
analyze_propagation_distribution.py — 微博传播指标分布分析

功能：
    1. 读取 weibo_visual_dataset.csv，提取 likes / comments / shares 传播指标
    2. 计算描述性统计: mean, median, std, min, max, 分位数(25%/50%/75%/90%/95%)
    3. 输出统计表: outputs/statistics/propagation_statistics.csv
    4. 绘制分布图（直方图 + 箱线图）: outputs/statistics/

运行命令（从项目根目录执行）：
    python visual_saliency_analysis/scripts/analyze_propagation_distribution.py

依赖：
    pip install pandas matplotlib
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------------------------------------------------------
# matplotlib 中文配置
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

INPUT_CSV  = os.path.join(_BASE_DIR, 'results', 'weibo_visual_dataset.csv')
OUTPUT_DIR = os.path.join(_BASE_DIR, 'outputs', 'statistics')
STATS_CSV  = os.path.join(OUTPUT_DIR, 'propagation_statistics.csv')

# 传播指标列名
PROPAGATION_COLS = ['likes', 'comments', 'shares']

# 中文标签映射
LABEL_MAP = {
    'likes':    '点赞 (Likes)',
    'comments': '评论 (Comments)',
    'shares':   '转发 (Shares)',
}


# ===========================================================================
# 统计计算
# ===========================================================================

def compute_statistics(series: pd.Series) -> dict:
    """计算单个传播指标的描述性统计。"""
    vals = series.dropna().astype(float)
    return {
        'count':   len(vals),
        'missing': int(series.isna().sum()),
        'mean':    vals.mean(),
        'median':  vals.median(),
        'std':     vals.std(),
        'min':     vals.min(),
        'max':     vals.max(),
        'p25':     vals.quantile(0.25),
        'p50':     vals.quantile(0.50),
        'p75':     vals.quantile(0.75),
        'p90':     vals.quantile(0.90),
        'p95':     vals.quantile(0.95),
        'skewness': vals.skew(),
        'kurtosis': vals.kurtosis(),
    }


def build_stats_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """对传播指标列逐一计算统计，返回汇总 DataFrame。"""
    records = []
    for col in PROPAGATION_COLS:
        if col not in df.columns:
            print(f"[WARN] 列 '{col}' 不存在，跳过。")
            continue
        stats = compute_statistics(df[col])
        stats['metric'] = col
        records.append(stats)

    stats_df = pd.DataFrame(records)
    # 调整列顺序
    col_order = ['metric', 'count', 'missing', 'mean', 'median', 'std',
                 'min', 'max', 'p25', 'p50', 'p75', 'p90', 'p95',
                 'skewness', 'kurtosis']
    stats_df = stats_df[[c for c in col_order if c in stats_df.columns]]
    return stats_df


# ===========================================================================
# 绘图
# ===========================================================================

def plot_histogram(series: pd.Series, col: str, output_path: str):
    """绘制单个指标的直方图（含 KDE 密度曲线）。"""
    vals = series.dropna().astype(float)

    fig, ax = plt.subplots(figsize=(10, 5))

    # 直方图
    ax.hist(vals, bins=40, color='#4A90D9', edgecolor='white', alpha=0.85,
            density=True, label='频率分布')

    # KDE 密度曲线
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(vals)
        x_range = np.linspace(vals.min(), vals.max(), 500)
        ax.plot(x_range, kde(x_range), color='#E05A3D', linewidth=2, label='密度曲线')
    except Exception:
        pass  # KDE 可能因数据问题失败，不影响主图

    # 标注线
    ax.axvline(vals.mean(), color='#F5A623', linestyle='--', linewidth=1.5,
               label=f'均值 = {vals.mean():.1f}')
    ax.axvline(vals.median(), color='#7ED321', linestyle='--', linewidth=1.5,
               label=f'中位数 = {vals.median():.1f}')

    ax.set_xlabel(LABEL_MAP.get(col, col), fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title(f'{LABEL_MAP.get(col, col)} 分布 (n={len(vals)})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {output_path}")


def plot_boxplot(df: pd.DataFrame, output_path: str):
    """绘制三个传播指标的并排箱线图。"""
    vals_list = []
    labels = []
    for col in PROPAGATION_COLS:
        if col in df.columns:
            vals = df[col].dropna().astype(float)
            vals_list.append(vals)
            labels.append(LABEL_MAP.get(col, col))

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(vals_list, patch_artist=True, widths=0.5,
                    medianprops={'color': 'black', 'linewidth': 1.5},
                    flierprops={'marker': 'o', 'markerfacecolor': 'red',
                                'markersize': 3, 'alpha': 0.4})

    colors = ['#4A90D9', '#7ED321', '#F5A623']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('数量', fontsize=12)
    ax.set_title('微博传播指标箱线图对比', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {output_path}")


def plot_log_histograms(df: pd.DataFrame, output_dir: str):
    """绘制对数变换后的分布图（处理长尾分布）。"""
    for col in PROPAGATION_COLS:
        if col not in df.columns:
            continue
        vals = df[col].dropna().astype(float)
        # 对数值（log1p 处理 0 值）
        log_vals = np.log1p(vals)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左：原始分布
        ax = axes[0]
        ax.hist(vals, bins=40, color='#4A90D9', edgecolor='white', alpha=0.85)
        ax.axvline(vals.mean(), color='#F5A623', linestyle='--', linewidth=1.5, label=f'均值={vals.mean():.0f}')
        ax.axvline(vals.median(), color='#7ED321', linestyle='--', linewidth=1.5, label=f'中位数={vals.median():.0f}')
        ax.set_xlabel(LABEL_MAP.get(col, col), fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.set_title('原始分布', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # 右：对数分布
        ax = axes[1]
        ax.hist(log_vals, bins=40, color='#7ED321', edgecolor='white', alpha=0.85)
        ax.axvline(log_vals.mean(), color='#F5A623', linestyle='--', linewidth=1.5,
                   label=f'log均值={log_vals.mean():.2f}')
        ax.axvline(log_vals.median(), color='#4A90D9', linestyle='--', linewidth=1.5,
                   label=f'log中位数={log_vals.median():.2f}')
        ax.set_xlabel(f'log(1 + {LABEL_MAP.get(col, col)})', fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.set_title('对数变换后分布', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        fig.suptitle(f'{LABEL_MAP.get(col, col)} 分布对比', fontsize=14, fontweight='bold')
        fig.tight_layout()

        fname = f'propagation_{col}_log_hist.png'
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [SAVED] {path}")


# ===========================================================================
# 主流程
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  微博传播指标分布分析")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. 加载数据
    # -----------------------------------------------------------------------
    print(f"\n[INFO] 加载数据: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] 文件不存在: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
    print(f"       记录数: {len(df)}, 字段数: {len(df.columns)}")

    # 确保传播指标列为数值类型
    for col in PROPAGATION_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # -----------------------------------------------------------------------
    # 2. 计算统计
    # -----------------------------------------------------------------------
    print(f"\n[INFO] 计算传播指标统计...")
    stats_df = build_stats_dataframe(df)

    print(f"\n{'─' * 70}")
    print(stats_df.to_string(index=False))
    print(f"{'─' * 70}")

    # -----------------------------------------------------------------------
    # 3. 保存统计表
    # -----------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stats_df.to_csv(STATS_CSV, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] 统计表已保存: {STATS_CSV}")

    # -----------------------------------------------------------------------
    # 4. 绘图
    # -----------------------------------------------------------------------
    print(f"\n[INFO] 绘制分布图...")

    # 4a. 每个指标的直方图（含密度曲线）
    for col in PROPAGATION_COLS:
        if col in df.columns:
            fname = f'propagation_{col}_hist.png'
            path = os.path.join(OUTPUT_DIR, fname)
            plot_histogram(df[col], col, path)

    # 4b. 并排箱线图
    box_path = os.path.join(OUTPUT_DIR, 'propagation_boxplot.png')
    plot_boxplot(df, box_path)

    # 4c. 对数变换直方图（处理长尾）
    plot_log_histograms(df, OUTPUT_DIR)

    # -----------------------------------------------------------------------
    # 5. 汇总报告
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  分析完成")
    print(f"  统计表: {STATS_CSV}")
    print(f"  图表目录: {OUTPUT_DIR}/")
    print(f"  生成文件:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    - {f}")
    print(f"{'=' * 60}")
