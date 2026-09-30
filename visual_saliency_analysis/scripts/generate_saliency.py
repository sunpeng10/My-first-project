"""
generate_saliency.py — 批量生成社交媒体图片的视觉显著性图 (Saliency Map)

功能：
    1. 加载已训练的 BASNet 模型 (basnet.pth)
    2. 批量读取 data/images/ 中的所有图片 (.jpg / .jpeg / .png)
    3. 为每张图片生成对应的 Saliency Map
    4. 保存至 outputs/saliency_maps/，保留原始文件名

运行命令（从项目根目录执行）：
    python visual_saliency_analysis/scripts/generate_saliency.py

依赖：
    - BASNet 模型权重: saved_models/basnet_bsi/basnet.pth
    - 输入图片目录: visual_saliency_analysis/data/images/
"""

import os
import sys
import glob

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage import io

# ---------------------------------------------------------------------------
# 将项目根目录加入 sys.path，以便导入 model 和 data_loader 模块
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _PROJECT_ROOT)

from data_loader import RescaleT, ToTensorLab, SalObjDataset   # noqa: E402
from model import BASNet                                        # noqa: E402


# ===========================================================================
# 工具函数
# ===========================================================================

def normPRED(d: torch.Tensor) -> torch.Tensor:
    """
    将预测结果归一化到 [0, 1] 区间。

    Args:
        d: 模型输出的显著性预测张量

    Returns:
        归一化后的张量
    """
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def save_output(image_path: str, pred: torch.Tensor, output_dir: str) -> None:
    """
    将 Saliency Map 保存为 PNG 文件，尺寸与原图保持一致。

    Args:
        image_path: 原始图片的完整路径
        pred:        模型输出的预测张量 (H, W)
        output_dir:  输出目录路径
    """
    # 转为 numpy 并缩放到 [0, 255]
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')

    # 读取原图尺寸，将显著性图 resize 回原图大小
    basename = os.path.basename(image_path)
    image = io.imread(image_path)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    # 去掉扩展名，保留文件名主体（支持多 "." 文件名如 image.001.jpg）
    parts = basename.split(".")
    stem = parts[0]
    for i in range(1, len(parts) - 1):
        stem = stem + "." + parts[i]

    output_path = os.path.join(output_dir, stem + '.png')
    imo.save(output_path)


# ===========================================================================
# 主流程
# ===========================================================================

if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # 1. 路径配置
    # -----------------------------------------------------------------------
    BASE_DIR = os.path.join(_PROJECT_ROOT, 'visual_saliency_analysis')

    image_dir = os.path.join(BASE_DIR, 'data', 'images')              # 输入图片目录
    output_dir = os.path.join(BASE_DIR, 'outputs', 'saliency_maps')   # 显著性图输出目录
    model_path = os.path.join(_PROJECT_ROOT, 'saved_models', 'basnet_bsi', 'basnet.pth')

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 2. 收集输入图片（支持 .jpg / .jpeg / .png）
    # -----------------------------------------------------------------------
    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')
    img_path_list = []
    for ext in SUPPORTED_EXTENSIONS:
        img_path_list.extend(glob.glob(os.path.join(image_dir, '*' + ext)))
    img_path_list.sort()

    if len(img_path_list) == 0:
        print(f"[ERROR] 未在 {image_dir} 中找到任何图片。")
        print(f"        支持的格式: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    print(f"[INFO] 找到 {len(img_path_list)} 张图片待处理。")

    # -----------------------------------------------------------------------
    # 3. 构建 DataLoader（复用 BASNet 预处理流水线）
    # -----------------------------------------------------------------------
    test_dataset = SalObjDataset(
        img_name_list=img_path_list,
        lbl_name_list=[],  # 推理阶段无需标签
        transform=transforms.Compose([
            RescaleT(256),
            ToTensorLab(flag=0),  # flag=0: RGB 色彩空间标准化
        ])
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    # -----------------------------------------------------------------------
    # 4. 加载 BASNet 模型
    # -----------------------------------------------------------------------
    print("[INFO] 加载 BASNet 模型...")
    if not os.path.exists(model_path):
        print(f"[ERROR] 模型权重未找到: {model_path}")
        print(f"        请先下载 basnet.pth 到 saved_models/basnet_bsi/ 目录。")
        sys.exit(1)

    net = BASNet(3, 1)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    print(f"[INFO] 模型加载完成，设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # -----------------------------------------------------------------------
    # 5. 批量推理
    # -----------------------------------------------------------------------
    print("[INFO] 开始推理...")
    for i, data in enumerate(test_dataloader):
        img_path = img_path_list[i]
        img_name = os.path.basename(img_path)
        print(f"  [{i + 1}/{len(img_path_list)}] {img_name}")

        # 准备输入张量
        inputs = data['image'].type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        # 前向传播：BASNet 输出 8 个层级，d1 为最终融合预测
        d1, d2, d3, d4, d5, d6, d7, d8 = net(inputs)

        # 取第一个通道作为显著性图，并归一化
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # 保存结果
        save_output(img_path, pred, output_dir)

        del d1, d2, d3, d4, d5, d6, d7, d8

    # -----------------------------------------------------------------------
    # 6. 完成
    # -----------------------------------------------------------------------
    print(f"\n[DONE] {len(img_path_list)} 张 Saliency Map 已保存至:")
    print(f"       {output_dir}")
