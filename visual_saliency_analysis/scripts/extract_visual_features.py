"""
extract_visual_features.py — 微博 500 张图片视觉显著性指标批量提取

功能：
    1. 读取 outputs/saliency_maps/ 中所有 BASNet 生成的 Saliency Map
    2. 逐张提取 7 项视觉显著性数值特征
    3. 汇总输出为 features/visual_features.csv

提取指标：
    - saliency_area_ratio        : 显著像素占比 (阈值 0.5)
    - mean_saliency              : 显著图平均强度
    - saliency_entropy           : 视觉注意力信息熵 (归一化)
    - center_bias                : 显著区域质心与中心距离 (归一化)
    - component_count            : 显著连通域数量
    - largest_component_ratio    : 最大连通域面积占比
    - saliency_std               : 显著图像素标准差

运行命令（从项目根目录执行）：
    python visual_saliency_analysis/scripts/extract_visual_features.py

依赖：
    pip install numpy pandas opencv-python pillow
"""

import os
import sys
import glob
import time

import numpy as np
import pandas as pd
import cv2
from PIL import Image

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_BASE_DIR = os.path.join(_PROJECT_ROOT, 'visual_saliency_analysis')

SALIENCY_DIR = os.path.join(_BASE_DIR, 'outputs', 'saliency_maps')     # Saliency Map 输入
OUTPUT_CSV   = os.path.join(_BASE_DIR, 'features', 'visual_features.csv')  # 特征输出

SALIENCY_THRESHOLD = 0.5  # 显著区域二值化阈值


# ===========================================================================
# 特征提取函数
# ===========================================================================

def load_saliency_map(filepath: str) -> np.ndarray:
    """加载显著性图，转为 [0, 1] 灰度 numpy 数组。"""
    img = Image.open(filepath).convert('L')
    return np.array(img, dtype=np.float32) / 255.0


def extract_saliency_area_ratio(sal_map: np.ndarray, threshold: float = SALIENCY_THRESHOLD) -> float:
    """显著像素占比 (显著强度 > threshold)。"""
    salient = sal_map > threshold
    return round(float(np.sum(salient) / salient.size), 6)


def extract_mean_saliency(sal_map: np.ndarray) -> float:
    """显著图平均强度。"""
    return round(float(np.mean(sal_map)), 6)


def extract_saliency_std(sal_map: np.ndarray) -> float:
    """显著图像素标准差。"""
    return round(float(np.std(sal_map)), 6)


def extract_saliency_entropy(sal_map: np.ndarray) -> float:
    """归一化信息熵：H(p) / log2(N)，衡量注意力分布均匀度。"""
    total = np.sum(sal_map)
    if total < 1e-12:
        return 0.0
    prob = sal_map.flatten() / total
    nonzero = prob[prob > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero))
    return round(float(entropy / np.log2(sal_map.size)), 6)


def extract_center_bias(sal_map: np.ndarray, threshold: float = SALIENCY_THRESHOLD) -> float:
    """显著区域质心与图片中心的归一化距离。无显著区域返回 -1。"""
    H, W = sal_map.shape
    mask = (sal_map > threshold).astype(np.float32)
    total = np.sum(mask)
    if total < 1:
        return -1.0
    yy, xx = np.indices((H, W))
    yc = np.sum(yy * mask) / total
    xc = np.sum(xx * mask) / total
    dist = np.sqrt((xc - (W - 1) / 2.0) ** 2 + (yc - (H - 1) / 2.0) ** 2)
    return round(float(dist / (np.sqrt(H ** 2 + W ** 2) / 2.0)), 6)


def extract_salient_components(sal_map: np.ndarray, threshold: float = SALIENCY_THRESHOLD) -> int:
    """显著连通域数量（OpenCV connectedComponents）。"""
    binary = (sal_map > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    num_labels, _ = cv2.connectedComponents(cleaned, connectivity=8)
    return max(0, num_labels - 1)


def extract_largest_component_ratio(sal_map: np.ndarray, threshold: float = SALIENCY_THRESHOLD) -> float:
    """最大连通域面积占比。"""
    H, W = sal_map.shape
    binary = (sal_map > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels <= 1:
        return 0.0
    largest = np.max(stats[1:, cv2.CC_STAT_AREA])
    return round(float(largest / (H * W)), 6)


# ===========================================================================
# 主流程
# ===========================================================================

if __name__ == '__main__':
    t0 = time.time()

    # -----------------------------------------------------------------------
    # 1. 匹配 wb_xxxx.png
    # -----------------------------------------------------------------------
    saliency_files = sorted(glob.glob(os.path.join(SALIENCY_DIR, 'wb_*.png')))

    total_expected = len(saliency_files)
    print(f"[INFO] 找到 {total_expected} 张 Saliency Map (wb_*.png)")

    if total_expected == 0:
        print("[ERROR] 未找到任何微博 Saliency Map，请先运行 BASNet 推理。")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 2. 逐张提取特征
    # -----------------------------------------------------------------------
    records = []
    missing = []

    for i, fpath in enumerate(saliency_files):
        image_id = os.path.splitext(os.path.basename(fpath))[0]

        if not os.path.exists(fpath):
            missing.append(image_id)
            continue

        sal_map = load_saliency_map(fpath)

        records.append({
            'image_id':                image_id,
            'saliency_area_ratio':     extract_saliency_area_ratio(sal_map),
            'mean_saliency':           extract_mean_saliency(sal_map),
            'saliency_std':            extract_saliency_std(sal_map),
            'saliency_entropy':        extract_saliency_entropy(sal_map),
            'center_bias':             extract_center_bias(sal_map),
            'component_count':         extract_salient_components(sal_map),
            'largest_component_ratio': extract_largest_component_ratio(sal_map),
        })

        # 进度
        if (i + 1) % 100 == 0 or (i + 1) == total_expected:
            print(f"  进度: [{i + 1}/{total_expected}]")

    # -----------------------------------------------------------------------
    # 3. 保存 CSV
    # -----------------------------------------------------------------------
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # -----------------------------------------------------------------------
    # 4. 统计输出
    # -----------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  特征提取完成")
    print(f"{'='*55}")
    print(f"  Saliency Map 总数:   {total_expected}")
    print(f"  成功提取:            {len(records)}")
    print(f"  缺失:                {len(missing)}")
    print(f"  耗时:                {elapsed:.0f}s")
    print(f"  输出:                {OUTPUT_CSV}")
    print(f"  字段数:              7")
    print(f"  数据行数:            {len(df)}")

    if missing:
        print(f"\n  缺失文件:")
        for m in missing:
            print(f"    - {m}")

    # 展示前 5 行
    print(f"\n  [前 5 行预览]")
    print(df.head(5).to_string(index=False))

    # 统计摘要
    print(f"\n  [特征统计摘要]")
    for col in df.columns[1:]:
        vals = df[col]
        print(f"  {col:30s}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}")
