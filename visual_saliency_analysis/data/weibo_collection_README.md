# weibo_collection.csv — 微博数据采集模板

## 用途

作为微博社交媒体数据采集的标准模板。每条记录对应一条微博图文，用于后续批量采集、人工标注后导入 `weibo_metadata.csv` 参与显著性—传播力相关性分析。

## 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image_id` | string | ✅ | 图片唯一编号，格式 `wb_xxxx`（如 `wb_0001`） |
| `weibo_url` | string | ✅ | 微博原文链接，用于数据溯源和验证 |
| `text` | string | | 微博正文文本内容 |
| `likes` | integer | | 点赞数量 |
| `comments` | integer | | 评论数量 |
| `shares` | integer | | 转发数量 |
| `category` | string | | 内容分类标签，如：美食、旅行、萌宠、科技、时尚 |
| `image_file` | string | | 下载到本地的图片文件名，如 `wb_0001.jpg` |

## image_id 与文件对应规则

```
image_id:  wb_0001
          │
          ├── data/images/weibo/wb_0001.jpg     ← 下载的图片
          ├── outputs/saliency_maps/wb_0001.png  ← BASNet 生成的显著图
          └── features/visual_features.csv       ← 提取的视觉特征（按 image_id 关联）
```

## 字段填写流程

```
阶段 1: 筛选微博
    → image_id, weibo_url, category

阶段 2: 采集数据
    → text, likes, comments, shares

阶段 3: 下载图片
    → image_file

阶段 4: 导入分析
    → 将填好的数据复制到 weibo_metadata.csv
```

## 示例数据

```csv
image_id,weibo_url,text,likes,comments,shares,category,image_file
wb_0001,https://weibo.com/xxx/12345,今天的日落太美了🌇,1523,89,210,旅行,wb_0001.jpg
wb_0002,https://weibo.com/xxx/12346,周末探店打卡☕,3408,156,430,美食,wb_0002.jpg
wb_0003,https://weibo.com/xxx/12347,分享一组猫咪日常🐱,8921,234,1200,萌宠,wb_0003.jpg
```
