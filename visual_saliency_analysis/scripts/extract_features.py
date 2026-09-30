"""
extract_features.py — 从 BASNet 生成的显著性图中提取数值特征

功能：
    1. 读取 outputs/saliency_maps/ 中所有 Saliency Map (.png)
    2. 逐张提取 5 项视觉显著性指标
    3. 汇总输出为 features/visual_features.csv

提取的指标：
    - saliency_area_ratio        : 显著像素占比 (阈值 > 0.5)
    - mean_saliency              : 显著图平均强度
    - saliency_std               : 显著图像素标准差
    - saliency_entropy           : 视觉注意力分布的信息熵（归一化）
    - center_bias                : 显著区域质心与图片中心的归一化距离
    - component_count            : 显著连通域数量 (OpenCV)
    - largest_component_ratio    : 最大连通域面积占整张图片比例

运行命令（从项目根目录执行）：
    python visual_saliency_analysis/scripts/extract_features.py

依赖：
    pip install numpy pandas opencv-python pillow
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import cv2
from PIL import Image

# ---------------------------------------------------------------------------
# 路径配置（基于脚本位置自动推导项目根目录）
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_BASE_DIR = os.path.join(_PROJECT_ROOT, 'visual_saliency_analysis')

SALIENCY_DIR = os.path.join(_BASE_DIR, 'outputs', 'saliency_maps')   # Saliency Map 目录
OUTPUT_CSV   = os.path.join(_BASE_DIR, 'features', 'visual_features.csv')  # 输出 CSV

# 显著区域二值化阈值（像素值归一化到 [0,1] 后）
SALIENCY_THRESHOLD = 0.5


# ===========================================================================
# 特征提取函数
# ===========================================================================

def load_saliency_map(filepath: str) -> np.ndarray:
    """
    加载显著性图，转为 [0, 1] 范围的灰度 numpy 数组。

    Args:
        filepath: PNG 文件路径

    Returns:
        shape (H, W), dtype float32, 范围 [0, 1]
    """
    img = Image.open(filepath).convert('L')       # 转为灰度图
    arr = np.array(img, dtype=np.float32) / 255.0  # 归一化到 [0, 1]
    return arr


def extract_saliency_area_ratio(sal_map: np.ndarray, threshold: float = SALIENCY_THRESHOLD) -> float:
    """
    计算显著像素占比 —— 显著强度超过阈值的像素数 / 总像素数。

    Args:
        sal_map:   显著性图 (H, W)，范围 [0, 1]
        threshold: 二值化阈值

    Returns:
        显著像素占比 (0.0 ~ 1.0)
    """
    salient_mask = sal_map > threshold
    ratio = np.sum(salient_mask) / salient_mask.size
    return round(float(ratio), 6)


def extract_mean_saliency(sal_map: np.ndarray) -> float:
    """
    计算显著图的平均强度。

    Args:
        sal_map: 显著性图 (H, W)，范围 [0, 1]

    Returns:
        平均显著值
    """
    return round(float(np.mean(sal_map)), 6)


def extract_saliency_std(sal_map: np.ndarray) -> float:
    """
    计算显著性图的像素标准差，衡量显著强度的离散程度。

    - 值越大：显著区域与背景的对比越强烈（强信号 vs 暗背景）
    - 值越小：整体显著强度趋近均匀

    Args:
        sal_map: 显著性图 (H, W)，范围 [0, 1]

    Returns:
        像素标准差 (0.0 ~ 0.5 典型范围)
    """
    return round(float(np.std(sal_map)), 6)


def extract_saliency_entropy(sal_map: np.ndarray) -> float:
    """
    计算显著性图的信息熵，衡量视觉注意力分布的均匀程度。

    步骤：
        1. 将显著图归一化为概率分布：p_i = v_i / sum(v_i)
        2. 计算香农熵：H = -sum(p_i * log2(p_i))，仅对 p_i > 0 求和
        3. 除以理论最大熵 log2(N) 做归一化，使结果在 [0, 1] 范围

    - 值越接近 1：注意力均匀散布（复杂、分散的场景）
    - 值越接近 0：注意力高度集中（单一显著目标）

    Args:
        sal_map: 显著性图 (H, W)，范围 [0, 1]

    Returns:
        归一化信息熵 (0.0 ~ 1.0)。若全为零则返回 0.0。
    """
    total = np.sum(sal_map)
    if total < 1e-12:
        return 0.0  # 全黑图，无信息

    # 转为概率分布
    prob = sal_map.flatten() / total

    # 仅对非零概率计算熵
    nonzero_prob = prob[prob > 0]
    entropy = -np.sum(nonzero_prob * np.log2(nonzero_prob))

    # 归一化：除以理论最大熵 log2(N)
    N = sal_map.size
    max_entropy = np.log2(N)
    normalized_entropy = entropy / max_entropy

    return round(float(normalized_entropy), 6)


def extract_largest_component_ratio(sal_map: np.ndarray, threshold: float = SALIENCY_THRESHOLD) -> float:
    """
    计算最大显著连通域面积占整张图片的比例。

    步骤：
        1. 二值化显著图 (threshold)
        2. 形态学开运算去除噪点
        3. 连通域分析 (connectedComponentsWithStats)
        4. 取最大连通域面积（排除背景 label 0）
        5. 除以总像素数

    - 值越接近 1：存在一个大的连续显著区域
    - 值越接近 0：显著区域分散为碎片

    Args:
        sal_map:   显著性图 (H, W)，范围 [0, 1]
        threshold: 二值化阈值

    Returns:
        最大连通域面积占比 (0.0 ~ 1.0)。无显著区域则返回 0.0。
    """
    H, W = sal_map.shape
    total_pixels = H * W

    # 二值化
    binary = (sal_map > threshold).astype(np.uint8) * 255

    # 形态学开运算，去除孤立噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 连通域分析（带统计信息）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    if num_labels <= 1:
        return 0.0  # 仅有背景，无显著区域

    # 排除背景 (label 0)，取最大面积
    # stats[i, cv2.CC_STAT_AREA] 为第 i 个连通域的面积
    areas = stats[1:, cv2.CC_STAT_AREA]  # 跳过 label 0（背景）
    largest_area = np.max(areas) if len(areas) > 0 else 0

    ratio = largest_area / total_pixels
    return round(float(ratio), 6)


def extract_center_bias(sal_map: np.ndarray, threshold: float = SALIENCY_THRESHOLD) -> float:
    """
    计算显著区域质心与图片几何中心的归一化欧氏距离。

    - 值越小，说明显著区域越靠近图片中心。
    - 0 表示质心与中心完全重合。
    - 归一化：距离除以图片对角线长度的一半，使值在 [0, ~1] 范围。

    Args:
        sal_map:   显著性图 (H, W)，范围 [0, 1]
        threshold: 二值化阈值

    Returns:
        归一化中心偏移距离。若没有显著像素则返回 -1。
    """
    H, W = sal_map.shape
    salient_mask = (sal_map > threshold).astype(np.float32)

    total = np.sum(salient_mask)
    if total < 1:
        return -1.0  # 无显著区域

    # 计算显著区域质心 (yc, xc)
    y_indices, x_indices = np.indices((H, W))
    yc = np.sum(y_indices * salient_mask) / total
    xc = np.sum(x_indices * salient_mask) / total

    # 图片几何中心
    y_center = (H - 1) / 2.0
    x_center = (W - 1) / 2.0

    # 欧氏距离，除以对角线长度的一半做归一化
    distance = np.sqrt((xc - x_center) ** 2 + (yc - y_center) ** 2)
    diagonal_half = np.sqrt(H ** 2 + W ** 2) / 2.0

    normalized_bias = distance / diagonal_half
    return round(float(normalized_bias), 6)


def extract_salient_components(sal_map: np.ndarray, threshold: float = SALIENCY_THRESHOLD) -> int:
    """
    使用 OpenCV 计算显著区域的连通域数量。

    步骤：
        1. 二值化显著图 (threshold)
        2. 形态学开运算去除噪点
        3. 查找连通域 (connectedComponents)

    Args:
        sal_map:   显著性图 (H, W)，范围 [0, 1]
        threshold: 二值化阈值

    Returns:
        连通域数量（不含背景）
    """
    # 二值化
    binary = (sal_map > threshold).astype(np.uint8) * 255

    # 形态学开运算：先腐蚀再膨胀，去除孤立噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 连通域分析
    num_labels, labels = cv2.connectedComponents(cleaned, connectivity=8)

    # num_labels 包含背景 (label 0)，因此显著区域数 = num_labels - 1
    components = max(0, num_labels - 1)
    return components


# ===========================================================================
# 主流程
# ===========================================================================

if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # 1. 查找所有 Saliency Map
    # -----------------------------------------------------------------------
    saliency_files = sorted(glob.glob(os.path.join(SALIENCY_DIR, '*.png')))

    if len(saliency_files) == 0:
        print(f"[ERROR] 未在 {SALIENCY_DIR} 中找到任何 Saliency Map。")
        print(f"        请先运行 generate_saliency.py 生成显著性图。")
        sys.exit(1)

    print(f"[INFO] 找到 {len(saliency_files)} 张 Saliency Map。")

    # -----------------------------------------------------------------------
    # 2. 逐张提取特征
    # -----------------------------------------------------------------------
    records = []

    for fpath in saliency_files:
        # 从文件名提取 image_id（如 "01.png" → "01"）
        image_id = os.path.splitext(os.path.basename(fpath))[0]

        print(f"  处理: {image_id}")

        # 加载显著性图
        sal_map = load_saliency_map(fpath)

        # 提取七项指标
        area_ratio         = extract_saliency_area_ratio(sal_map)
        mean_sal           = extract_mean_saliency(sal_map)
        sal_std            = extract_saliency_std(sal_map)
        entropy            = extract_saliency_entropy(sal_map)
        center_bias        = extract_center_bias(sal_map)
        n_components       = extract_salient_components(sal_map)
        largest_comp_ratio = extract_largest_component_ratio(sal_map)

        records.append({
            'image_id':                 image_id,
            'saliency_area_ratio':      area_ratio,
            'mean_saliency':            mean_sal,
            'saliency_std':             sal_std,
            'saliency_entropy':         entropy,
            'center_bias':              center_bias,
            'component_count':          n_components,
            'largest_component_ratio':  largest_comp_ratio,
        })

        print(f"        面积占比={area_ratio:.4f}, 均值={mean_sal:.4f}, "
              f"标准差={sal_std:.4f}, 熵={entropy:.4f}, 中心偏移={center_bias:.4f}, "
              f"连通域数={n_components}, 最大域占比={largest_comp_ratio:.4f}")

    # -----------------------------------------------------------------------
    # 3. 保存为 CSV
    # -----------------------------------------------------------------------
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print(f"\n[DONE] 特征已保存至: {OUTPUT_CSV}")
    print(f"       共计 {len(df)} 行, {len(df.columns)} 列")
    print(f"\n{df.to_string(index=False)}")
