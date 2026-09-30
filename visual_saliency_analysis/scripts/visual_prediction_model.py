"""
visual_prediction_model.py — 视觉特征预测传播效果实验

功能：
    1. 读取 weibo_visual_dataset.csv，使用 7 个视觉特征预测 engagement_score
    2. 80/20 train/test 划分 (random_state=42)
    3. 训练 RandomForestRegressor（XGBoost 可选）
    4. 评估: MAE, RMSE, R²
    5. 输出: results/model_results.csv
    6. 输出: results/feature_importance.csv + results/feature_importance.png

运行命令（从项目根目录执行）：
    python visual_saliency_analysis/scripts/visual_prediction_model.py

依赖：
    pip install pandas numpy scikit-learn matplotlib
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

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

INPUT_CSV        = os.path.join(_BASE_DIR, 'results', 'weibo_visual_dataset.csv')
MODEL_RESULTS    = os.path.join(_BASE_DIR, 'results', 'model_results.csv')
FEAT_IMPORT_CSV  = os.path.join(_BASE_DIR, 'results', 'feature_importance.csv')
FEAT_IMPORT_PNG  = os.path.join(_BASE_DIR, 'results', 'feature_importance.png')

# ---------------------------------------------------------------------------
# 特征 & 目标定义
# ---------------------------------------------------------------------------
FEATURES = [
    'saliency_area_ratio',
    'mean_saliency',
    'saliency_std',
    'saliency_entropy',
    'center_bias',
    'component_count',
    'largest_component_ratio',
]

TARGET = 'engagement_score'

# 中文标签
FEATURE_LABELS = {
    'saliency_area_ratio':     '显著性面积比',
    'mean_saliency':           '平均显著性',
    'saliency_std':            '显著性标准差',
    'saliency_entropy':        '显著性熵',
    'center_bias':             '中心偏置',
    'component_count':         '连通分量数',
    'largest_component_ratio': '最大分量比',
}

# ---------------------------------------------------------------------------
# 模型参数
# ---------------------------------------------------------------------------
TEST_SIZE   = 0.2
RANDOM_SEED = 42
N_ESTIMATORS = 200
MAX_DEPTH    = 10


# ===========================================================================
# 主流程
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  视觉特征 → 传播效果 预测实验")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # -------------------------------------------------------------------
    # 1. 加载数据
    # -------------------------------------------------------------------
    print(f"\n[1/5] 加载数据: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
    print(f"      记录数: {len(df)}")

    # 数值化
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 如果 engagement_score 不存在，则计算
    if TARGET not in df.columns:
        print(f"      engagement_score 不存在，自动计算...")
        raw_prop = ['likes', 'comments', 'shares']
        for col in raw_prop:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['likes_log']    = np.log1p(df['likes'].clip(lower=0))
        df['comments_log'] = np.log1p(df['comments'].clip(lower=0))
        df['shares_log']   = np.log1p(df['shares'].clip(lower=0))
        df[TARGET] = df['likes_log'] + 2 * df['comments_log'] + 3 * df['shares_log']

    # 去除含 NaN 的行
    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET])
    if len(df) < before:
        print(f"      去除含 NaN 行: {before} -> {len(df)}")

    X = df[FEATURES].values
    y = df[TARGET].values
    print(f"      特征数: {len(FEATURES)}, 目标: {TARGET}")
    print(f"      有效样本: {len(df)}")

    # 统计摘要
    print(f"\n      特征统计:")
    for f in FEATURES:
        s = df[f]
        print(f"        {FEATURE_LABELS.get(f, f):<16s}  mean={s.mean():.4f}  std={s.std():.4f}  "
              f"range=[{s.min():.4f}, {s.max():.4f}]")

    # -------------------------------------------------------------------
    # 2. 数据划分
    # -------------------------------------------------------------------
    print(f"\n[2/5] 数据划分: {100*(1-TEST_SIZE):.0f}% train / {100*TEST_SIZE:.0f}% test  (random_state={RANDOM_SEED})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"      训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # -------------------------------------------------------------------
    # 3. 训练模型
    # -------------------------------------------------------------------
    print(f"\n[3/5] 训练模型...")

    models = {}

    # ---- RandomForest ----
    print(f"      RandomForestRegressor (n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH})")
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf

    # ---- XGBoost (可选) ----
    try:
        import xgboost as xgb
        print(f"      XGBRegressor (n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH//2})")
        xgb_model = xgb.XGBRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH // 2,
            learning_rate=0.05,
            random_state=RANDOM_SEED,
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        models['XGBoost'] = xgb_model
    except ImportError:
        print(f"      [SKIP] xgboost 未安装，只运行 RandomForest")

    # -------------------------------------------------------------------
    # 4. 评估
    # -------------------------------------------------------------------
    print(f"\n[4/5] 评估模型...")

    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        results.append({
            'model': name,
            'MAE':   round(mae, 4),
            'RMSE':  round(rmse, 4),
            'R2':    round(r2, 4),
        })

        print(f"      {name:<16s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

    # 保存模型结果
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(MODEL_RESULTS), exist_ok=True)
    results_df.to_csv(MODEL_RESULTS, index=False, encoding='utf-8-sig')
    print(f"\n      结果已保存: {MODEL_RESULTS}")

    # -------------------------------------------------------------------
    # 5. 特征重要性
    # -------------------------------------------------------------------
    print(f"\n[5/5] 特征重要性分析...")

    # 使用 RandomForest 的特征重要性
    rf_importances = rf.feature_importances_

    # 按重要性降序排列
    feat_imp_df = pd.DataFrame({
        'feature':    FEATURES,
        'importance': rf_importances,
    })
    feat_imp_df = feat_imp_df.sort_values('importance', ascending=False).reset_index(drop=True)

    # 添加中文标签
    feat_imp_df['feature_label'] = feat_imp_df['feature'].map(FEATURE_LABELS)

    # 保存 CSV
    feat_imp_df[['feature', 'importance']].to_csv(
        FEAT_IMPORT_CSV, index=False, encoding='utf-8-sig'
    )
    print(f"      特征重要性: {FEAT_IMPORT_CSV}")

    # 终端输出
    print(f"\n      RandomForest 特征重要性 (降序):")
    for _, row in feat_imp_df.iterrows():
        bar = '█' * int(row['importance'] * 100)
        print(f"        {row['feature_label']:<16s}  {row['importance']:.4f}  {bar}")

    # 绘制特征重要性条形图
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feat_imp_df)))
    bars = ax.barh(
        range(len(feat_imp_df)),
        feat_imp_df['importance'].values,
        color=colors[::-1],  # 反转使最深色在顶部
        edgecolor='white',
        height=0.6,
    )

    ax.set_yticks(range(len(feat_imp_df)))
    ax.set_yticklabels(feat_imp_df['feature_label'].values, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('视觉特征对传播力 (engagement_score) 的预测重要性\n'
                 f'(RandomForest, n={len(df)})',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 在条形末端标注数值
    for bar, val in zip(bars, feat_imp_df['importance'].values):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9, color='#333333')

    ax.set_xlim(0, feat_imp_df['importance'].max() * 1.3)

    fig.tight_layout()
    fig.savefig(FEAT_IMPORT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"      特征重要性图: {FEAT_IMPORT_PNG}")

    # -------------------------------------------------------------------
    # DONE
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  实验完成")
    print(f"  模型结果:     {MODEL_RESULTS}")
    print(f"  特征重要性:   {FEAT_IMPORT_CSV}")
    print(f"  重要性图:     {FEAT_IMPORT_PNG}")
    print(f"{'=' * 60}")
