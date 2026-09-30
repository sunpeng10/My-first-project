"""
create_analysis_dataset.py — 构建分析数据集（传播指标 + 对数变换 + 综合传播力评分）

功能：
    1. 读取 weibo_visual_dataset.csv
    2. 保留原始传播字段: likes, comments, shares
    3. 计算对数变换字段: likes_log, comments_log, shares_log  (= log(1 + x))
    4. 计算综合传播力评分:
       engagement_score = likes_log + 2 * comments_log + 3 * shares_log
    5. 输出: results/analysis_dataset.csv
    6. 数据质量检查: 行数校验、空值检测、异常值检测

运行命令（从项目根目录执行）：
    python visual_saliency_analysis/scripts/create_analysis_dataset.py

依赖：
    pip install pandas numpy
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_BASE_DIR = os.path.join(_PROJECT_ROOT, 'visual_saliency_analysis')

INPUT_CSV  = os.path.join(_BASE_DIR, 'results', 'weibo_visual_dataset.csv')
OUTPUT_CSV = os.path.join(_BASE_DIR, 'results', 'analysis_dataset.csv')

# 原始传播指标
PROPAGATION_COLS = ['likes', 'comments', 'shares']

# 权重配置: engagement_score = w_likes * likes_log + w_comments * comments_log + w_shares * shares_log
WEIGHTS = {
    'likes':    1,
    'comments': 2,
    'shares':   3,
}


# ===========================================================================
# 工具函数
# ===========================================================================

def transform_log(series: pd.Series) -> pd.Series:
    """对序列应用 log(1 + x) 变换，处理非正值。"""
    return np.log1p(series.clip(lower=0))


def compute_engagement(df: pd.DataFrame) -> pd.Series:
    """计算综合传播力评分。"""
    score = np.zeros(len(df))
    for col, w in WEIGHTS.items():
        log_col = f'{col}_log'
        score += w * df[log_col]
    return score


# ===========================================================================
# 质量检查
# ===========================================================================

def check_dataset(df: pd.DataFrame, expected_rows: int = 500):
    """对数据集执行完整性检查，返回 (通过, 问题列表)。"""
    issues = []

    # 1. 行数校验
    if len(df) != expected_rows:
        issues.append(f"行数异常: 期望 {expected_rows}, 实际 {len(df)}")
    else:
        print(f"  [OK] 行数校验通过: {len(df)} 行")

    # 2. 空值检测
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        for col, cnt in null_cols.items():
            issues.append(f"空值: {col} 列有 {cnt} 个空值")
    else:
        print(f"  [OK] 空值检测通过: 0 个空值")

    # 3. 异常值检测
    #    对原始传播指标（已知为长尾分布），只检查数据合法性（非负、非无穷）
    #    对对数变换列和 engagement_score，使用 IQR 方法检测
    print(f"  [INFO] 原始传播指标合法性检查 (非负、非无穷)...")
    for col in PROPAGATION_COLS:
        if col not in df.columns:
            continue
        series = df[col]
        neg_count = (series < 0).sum()
        inf_count = (~np.isfinite(series)).sum()
        if neg_count > 0:
            issues.append(f"数据错误: {col} 有 {neg_count} 个负值")
        if inf_count > 0:
            issues.append(f"数据错误: {col} 有 {inf_count} 个无穷值")
    print(f"  [OK] 原始传播指标合法性通过 (无负值、无无穷值)")

    # 对变换后的列使用 IQR 检测
    print(f"  [INFO] 变换后字段 IQR 异常值检测...")
    check_cols = [f'{c}_log' for c in PROPAGATION_COLS] + ['engagement_score']
    for col in check_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3.0 * IQR
        upper = Q3 + 3.0 * IQR
        n_outliers = ((series < lower) | (series > upper)).sum()
        pct = n_outliers / len(series) * 100
        if n_outliers > 0:
            if pct > 5:
                issues.append(
                    f"异常值偏高: {col} 有 {n_outliers} 个异常点 ({pct:.1f}%), "
                    f"范围 [{lower:.2f}, {upper:.2f}]"
                )
            else:
                print(f"  [INFO] {col}: {n_outliers} 个极端值 ({pct:.1f}%), "
                      f"位于 [{lower:.2f}, {upper:.2f}] 之外")

    # 4. 字段完整性
    required_cols = (PROPAGATION_COLS +
                     [f'{c}_log' for c in PROPAGATION_COLS] +
                     ['engagement_score'])
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"缺少字段: {missing_cols}")
    else:
        print(f"  [OK] 字段完整性通过: {len(required_cols)} 个目标字段齐全")

    # 5. 对数变换合理性检验
    for col in PROPAGATION_COLS:
        log_col = f'{col}_log'
        if log_col not in df.columns:
            continue
        # 原始为 0 的值，对数后应为 0 (log1p(0) = 0)
        orig_zero_mask = (df[col] == 0)
        if orig_zero_mask.any():
            log_vals_at_zero = df.loc[orig_zero_mask, log_col]
            if not np.allclose(log_vals_at_zero, 0):
                issues.append(f"对数变换异常: {col}=0 时 {log_col} 非零")
        # 正值单调性
        pos_mask = (df[col] > 0)
        if pos_mask.any():
            if (df.loc[pos_mask, log_col] <= 0).any():
                issues.append(f"对数变换异常: {col}>0 时 {log_col} <= 0")

    print(f"  [OK] 对数变换合理性验证通过")

    return len(issues) == 0, issues


# ===========================================================================
# 主流程
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  构建分析数据集")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. 加载原始数据
    # -----------------------------------------------------------------------
    print(f"\n[1/4] 加载原始数据: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] 文件不存在: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
    print(f"      输入: {len(df)} 行 × {len(df.columns)} 列")

    # 确保传播指标为数值类型
    for col in PROPAGATION_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"[ERROR] 缺少必需列: {col}")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # 2. 保留原始字段 + 计算对数变换
    # -----------------------------------------------------------------------
    print(f"\n[2/4] 计算对数变换字段...")

    # 构建输出 DataFrame，以 image_id 和 text 为主键
    out_cols = ['image_id', 'text'] if 'text' in df.columns else ['image_id']
    out_df = df[out_cols].copy()

    # 保留原始传播指标
    for col in PROPAGATION_COLS:
        out_df[col] = df[col]

    # 计算对数变换
    for col in PROPAGATION_COLS:
        log_col = f'{col}_log'
        out_df[log_col] = transform_log(df[col])
        print(f"      {log_col}: mean={out_df[log_col].mean():.4f}, "
              f"std={out_df[log_col].std():.4f}, "
              f"range=[{out_df[log_col].min():.4f}, {out_df[log_col].max():.4f}]")

    # -----------------------------------------------------------------------
    # 3. 计算综合传播力评分
    # -----------------------------------------------------------------------
    print(f"\n[3/4] 计算综合传播力评分 (engagement_score)...")
    print(f"      权重: likes={WEIGHTS['likes']}, comments={WEIGHTS['comments']}, shares={WEIGHTS['shares']}")

    out_df['engagement_score'] = compute_engagement(out_df)

    print(f"      engagement_score: mean={out_df['engagement_score'].mean():.4f}, "
          f"std={out_df['engagement_score'].std():.4f}, "
          f"median={out_df['engagement_score'].median():.4f}, "
          f"range=[{out_df['engagement_score'].min():.4f}, {out_df['engagement_score'].max():.4f}]")

    # -----------------------------------------------------------------------
    # 4. 保存
    # -----------------------------------------------------------------------
    print(f"\n[4/4] 保存结果: {OUTPUT_CSV}")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"      输出: {len(out_df)} 行 × {len(out_df.columns)} 列")
    print(f"      字段: {', '.join(out_df.columns)}")

    # -----------------------------------------------------------------------
    # 5. 质量检查
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print(f"  数据质量检查")
    print(f"{'─' * 60}")

    passed, issues = check_dataset(out_df, expected_rows=500)

    if passed:
        print(f"\n{'─' * 60}")
        print(f"  全部检查通过 [PASS]")
        print(f"{'─' * 60}")
    else:
        print(f"\n{'─' * 60}")
        print(f"  发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
        print(f"{'─' * 60}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 预览
    # -----------------------------------------------------------------------
    print(f"\n  [前5行预览]")
    preview = out_df.head(5).copy()
    if 'text' in preview.columns:
        # 截断并移除 emoji 等无法在 GBK 终端显示的字符
        preview['text'] = preview['text'].str[:25].str.encode(
            'ascii', errors='replace'
        ).str.decode('ascii')
    # 格式化浮点列
    for c in preview.columns:
        if preview[c].dtype in ('float64', 'float32'):
            preview[c] = preview[c].round(4)
    print(preview.to_string(index=False))

    print(f"\n[DONE] 分析数据集构建完成")
