# 多模态微博舆情分析系统

基于微博评论数据，集成三大分析能力并提供 Web 界面：

| 能力 | 技术 | 说明 |
|------|------|------|
| 文本情感分析 | RoBERTa + Captum（IG / Saliency） | 逐字显著性高亮 |
| 图像视觉显著性 | BASNet (CVPR 2019) | 显著性图 + 7 项视觉指标 |
| 传播网络核心节点 | 两层 GAT | 评论网络核心传播节点识别 |

- 后端（Flask API）：见 [`backend/README.md`](backend/README.md)
- 前端（Vue 3 + Vite）：见 [`frontend/README.md`](frontend/README.md)
- GAT 训练 / 推理：`gat/` + `scripts/run_gat.py`
- 视觉显著性分析流水线：见 [`visual_saliency_analysis/README.md`](visual_saliency_analysis/README.md)

## 快速启动

```bash
# 1. 后端（在项目根目录下）
python -m backend.app          # 监听 http://127.0.0.1:5000

# 2. 前端（另开终端）
cd frontend
npm install
npm run dev                    # http://localhost:5173
```

## 运行依赖（clone 后需自行准备）

| 依赖 | 位置 | 获取方式 |
|------|------|----------|
| BASNet 权重 `basnet.pth`（约 348MB，未入库） | `saved_models/basnet_bsi/basnet.pth` | [GoogleDrive](https://drive.google.com/open?id=1s52ek_4YTDRt_EOkx1FS53u-vJa0c4nu)（或百度网盘，提取码见下方 BASNet 说明） |
| RoBERTa 情感模型 | 外部项目 `D:\vscode python files\sentiment_baseline\models` | 独立仓库，只读加载；路径可用环境变量 `SENTIMENT_ROOT` 覆盖 |
| 微博原图 | `D:\WeiboCrawler\images\` | 采集项目产出，用于节点详情展示原图 |

> 这三项依赖不在本仓库中：`basnet.pth` 体积超过 GitHub 单文件 100MB 限制故未提交，
> 后两者属于独立的微博采集 / 情感分析项目。缺少时后端仍能启动，
> 但 `/api/image`、`/api/text`、节点详情原图等功能会报 `not_loaded`。

---

# BASNet (New Version May 2nd, 2021)

'[Boundary-Aware Segmentation Network for
Mobile and Web Applications](https://arxiv.org/pdf/2101.04704.pdf)', Xuebin Qin, Deng-Ping Fan, Chenyang Huang, Cyril Diagne, Zichen Zhang,
Adria Cabeza Sant’Anna, Albert Suarez, Martin Jagersand, and Ling Shao. 

## Salient Object Detection(SOD) Qualitative Comparison
![SOD Qualitative Comparison](figures/sod_qual_comp.PNG)

## Salient Objects in Clutter(SOC) Qualitative Comparison
![SOC Qualitative Comparison](figures/soc_qual_comp.PNG)

## Camouflaged Object Detection(COD) Qualitative Comparison
![COD Qualitative Comparison](figures/cod_qual_comp.PNG)

## Predicted maps of SOD, SOC and COD datasets

[SOD Results will come soon!]() \
[SOC Results will come soon!]() \
[COD Results](https://drive.google.com/file/d/12jijUPpdOe7k2O1YcLbkJHyXCJb3MRMN/view?usp=sharing)



# BASNet (CVPR 2019)
Code for CVPR 2019 paper '[*BASNet: Boundary-Aware Salient Object Detection*](http://openaccess.thecvf.com/content_CVPR_2019/html/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.html) [code](https://github.com/NathanUA/BASNet)', [Xuebin Qin](https://webdocs.cs.ualberta.ca/~xuebin/), [Zichen Zhang](https://webdocs.cs.ualberta.ca/~zichen2/), [Chenyang Huang](https://chenyangh.com/), [Chao Gao](https://cgao3.github.io/), [Masood Dehghan](https://sites.google.com/view/masooddehghan) and [Martin Jagersand](https://webdocs.cs.ualberta.ca/~jag/). 

__Contact__: xuebin[at]ualberta[dot]ca

## (2020-May-09) NEWS! Our new Salient Object Detection model (U^2-Net), which is just accepted by Pattern Recognition, is available now!
[U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://github.com/NathanUA/U-2-Net)

## Evaluation
[Evaluation Code](https://github.com/NathanUA/Binary-Segmentation-Evaluation-Tool)

## Required libraries

Python 3.6  
numpy 1.15.2  
scikit-image 0.14.0  
PIL 5.2.0  
PyTorch 0.4.0  
torchvision 0.2.1  
glob  

The SSIM loss is adapted from [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py).  

## Usage
1. Clone this repo
```
git clone https://github.com/NathanUA/BASNet.git
```
2. Download the pre-trained model basnet.pth from [GoogleDrive](https://drive.google.com/open?id=1s52ek_4YTDRt_EOkx1FS53u-vJa0c4nu) or [baidu](https://pan.baidu.com/s/1PrsBdepwrkMWPLSW22FhAg) extraction code: 6phq, and put it into the dirctory 'saved_models/basnet_bsi/'

3.  Cd to the directory 'BASNet', run the training or inference process by command: ```python basnet_train.py```
or ```python basnet_test.py``` respectively.  

 We also provide the predicted saliency maps ([GoogleDrive](https://drive.google.com/file/d/1K9y9HpupXT0RJ4U4OizJ_Uk5byUyCupK/view?usp=sharing),[Baidu](https://pan.baidu.com/s/1FJKVO_9YrP7Iaz7WT6Xdhg)) for datasets SOD, ECSSD, DUT-OMRON, PASCAL-S, HKU-IS and DUTS-TE.

## Architecture

![BASNet architecture](figures/architecture.png)


## Quantitative Comparison

![Quantitative Comparison](figures/quan.png)

## Qualitative Comparison

![Qualitative Comparison](figures/qual.png)


## Citation
```
@article{DBLP:journals/corr/abs-2101-04704,
  author    = {Xuebin Qin and
               Deng{-}Ping Fan and
               Chenyang Huang and
               Cyril Diagne and
               Zichen Zhang and
               Adri{\`{a}} Cabeza Sant'Anna and
               Albert Su{\`{a}}rez and
               Martin J{\"{a}}gersand and
               Ling Shao},
  title     = {Boundary-Aware Segmentation Network for Mobile and Web Applications},
  journal   = {CoRR},
  volume    = {abs/2101.04704},
  year      = {2021},
  url       = {https://arxiv.org/abs/2101.04704},
  archivePrefix = {arXiv},
}
```

## Citation
```
@InProceedings{Qin_2019_CVPR,
author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Gao, Chao and Dehghan, Masood and Jagersand, Martin},
title = {BASNet: Boundary-Aware Salient Object Detection},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
