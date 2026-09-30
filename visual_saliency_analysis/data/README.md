# data/ — 数据目录说明

## 目录结构

```
data/
├── images/                  # 图片存储
│   ├── *.jpg                # 通用测试图片（直接放 images/ 下）
│   └── weibo/               # 微博采集图片（images/weibo/）
│       └── wb_0001.jpg      # 命名规则: wb_xxxx.jpg
├── metadata.csv             # 通用元数据（早期测试用）
├── weibo_metadata.csv       # 微博社交媒体元数据
└── README.md                # 本文件
```

## weibo_metadata.csv — 微博元数据

### 用途

存储从微博平台采集的社交媒体图片对应的互动指标，用于后续与视觉显著性特征进行相关性分析。

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `image_id` | string | 图片唯一编号，格式 `wb_xxxx`（如 `wb_0001`） |
| `text` | string | 微博正文文本内容 |
| `likes` | integer | 点赞数量 |
| `comments` | integer | 评论数量 |
| `shares` | integer | 转发数量 |

### image_id 与图片文件对应规则

| image_id | 图片文件路径 |
|------|------|
| `wb_0001` | `data/images/weibo/wb_0001.jpg` |
| `wb_0002` | `data/images/weibo/wb_0002.jpg` |
| ... | ... |

> 支持扩展名：`.jpg`、`.jpeg`、`.png`。`generate_saliency.py` 会自动匹配三种格式。

### 示例数据

```csv
image_id,text,likes,comments,shares
wb_0001,今天的日落太美了,1523,89,210
wb_0002,周末探店打卡,3408,156,430
wb_0003,分享一组猫咪日常,8921,234,1200
```

### 分析流程

```
data/images/weibo/xxx.jpg           →  generate_saliency.py
outputs/saliency_maps/xxx.png       →  extract_features.py
features/visual_features.csv        ┐
data/weibo_metadata.csv             ┤  →  correlation_analysis.py
                                     ┘      results/
```

## metadata.csv — 通用元数据（保留）

早期测试阶段使用的通用元数据文件，与项目原有 test_data 目录配合使用。
