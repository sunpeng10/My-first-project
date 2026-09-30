"""
图像视觉显著性服务：BASNet 推理 + 7 项视觉显著性特征。

- 复用原项目 `model.BASNet` + `data_loader` 预处理流水线（RescaleT/ToTensorLab），
  以及 `extract_visual_features` 的 7 项特征函数，不重新训练、不重新实现。
- 模型在进程内只加载一次。
- 输出显著性图保存为 PNG，通过 /api/image/saliency/<filename> 访问。
"""
import io
import logging
import sys
import threading

import numpy as np
import torch
from PIL import Image

from . import config
from .utils import get_device, make_output_name

logger = logging.getLogger(__name__)

BASNET_INPUT_SIZE = 256


def norm_pred(d: torch.Tensor) -> torch.Tensor:
    """将预测张量归一化到 [0, 1]（与原项目 normPRED 一致）。"""
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi + 1e-8)


class ImageService:
    def __init__(self):
        self.net = None
        self.rescale = None
        self.to_tensor_lab = None
        self.feature_funcs = None
        self.device = get_device()
        self.loaded = False
        self.load_error = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 模型加载（进程内一次）
    # ------------------------------------------------------------------
    def load(self):
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            try:
                # 复用原项目 BASNet 模型与数据预处理
                if config.PROJECT_ROOT not in sys.path:
                    sys.path.insert(0, config.PROJECT_ROOT)
                from data_loader import RescaleT, ToTensorLab  # noqa: E402
                from model import BASNet  # noqa: E402

                # 复用原项目 7 项显著性特征函数
                scripts_dir = config.PROJECT_ROOT + "/visual_saliency_analysis/scripts"
                if scripts_dir not in sys.path:
                    sys.path.insert(0, scripts_dir)
                import extract_visual_features as evf  # noqa: E402

                self.net = BASNet(3, 1)
                self.net.load_state_dict(
                    torch.load(config.BASNET_MODEL_PATH, map_location="cpu")
                )
                self.net.to(self.device)
                self.net.eval()

                self.rescale = RescaleT(BASNET_INPUT_SIZE)
                self.to_tensor_lab = ToTensorLab(flag=0)
                self.feature_funcs = {
                    "saliency_area_ratio": evf.extract_saliency_area_ratio,
                    "mean_saliency": evf.extract_mean_saliency,
                    "saliency_std": evf.extract_saliency_std,
                    "saliency_entropy": evf.extract_saliency_entropy,
                    "center_bias": evf.extract_center_bias,
                    "component_count": evf.extract_salient_components,
                    "largest_component_ratio": evf.extract_largest_component_ratio,
                }
                self.loaded = True
                logger.info(
                    "image model loaded: %s (%s)", config.BASNET_MODEL_PATH, self.device
                )
            except Exception as exc:  # noqa: BLE001
                self.load_error = str(exc)
                self.loaded = False
                logger.exception("failed to load image model")

    def _ensure_loaded(self):
        if not self.loaded:
            self.load()
        if not self.loaded:
            raise RuntimeError(f"image model not loaded: {self.load_error}")

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def analyze(self, image_bytes: bytes, original_filename: str) -> dict:
        self._ensure_loaded()

        # 1. 解码图片，拿到原始尺寸（Pillow 支持 jpg/png/webp）
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.load()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"cannot decode image: {exc}")

        width, height = pil_img.size
        img_np = np.array(pil_img.convert("RGB"))  # (H, W, 3) uint8
        if img_np.size == 0 or img_np.max() == 0:
            raise RuntimeError("image is empty or fully black")

        # 2. 复用原项目预处理：sample dict -> RescaleT(256) -> ToTensorLab(flag=0)
        label = np.zeros((height, width, 1), dtype=np.float32)
        sample = {"image": img_np, "label": label}
        sample = self.rescale(sample)
        sample = self.to_tensor_lab(sample)

        inputs = sample["image"].unsqueeze(0).float().to(self.device)  # (1,3,256,256)

        # 3. BASNet 前向（d1 为最终融合预测）
        with torch.no_grad():
            d1, d2, d3, d4, d5, d6, d7, d8 = self.net(inputs)
        del d2, d3, d4, d5, d6, d7, d8

        pred = norm_pred(d1[:, 0, :, :]).squeeze().cpu().numpy()  # [0,1] (256,256)

        # 4. resize 回原图尺寸并保存显著性图（与原项目 save_output 一致）
        sal_img = Image.fromarray((pred * 255).astype(np.uint8), "L")
        sal_img = sal_img.resize((width, height), Image.BILINEAR)

        out_name = make_output_name("png")
        out_path = f"{config.OUTPUT_DIR}/{out_name}"
        sal_img.convert("RGB").save(out_path)

        # 5. 计算 7 项特征（与原项目 extract_visual_features 一致）
        sal_np = np.array(sal_img, dtype=np.float32) / 255.0
        saliency = {
            name: fn(sal_np) for name, fn in self.feature_funcs.items()
        }

        return {
            "filename": original_filename,
            "saliency": saliency,
            "saliency_map_url": f"/api/image/saliency/{out_name}",
        }
