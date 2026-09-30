# username → UID 映射 数据说明（第三阶段 · 第二步）

为 500 条微博补齐作者显示名，建立可靠的 username ↔ UID 映射，为闭合 mention network 做准备。

## 数据来源

| 文件 | 路径 | 角色 |
|------|------|------|
| 微博原文 | `D:\WeiboCrawler\data\weibo.csv` | 500 条微博 + `weibo_url`（只读） |
| **映射结果** | `D:\WeiboCrawler\data\weibo_user_mapping.csv` | 403 行 `user_id, user_name, weibo_url` |
| 提及关系 | `visual_saliency_analysis/graph/mention_relations_raw.csv` | 53 个被提及用户名 |

## user_id 来源

`user_id` = 微博作者 UID，从 `weibo_url`（`https://weibo.com/{UID}/{POST_ID}`）正则提取，**非猜测**。500 条微博 → 403 个唯一 UID。

## user_name 来源

`user_name` = 作者**显示名**，通过 Selenium 访问作者主页 `https://weibo.com/u/{UID}`，读取页面 `<title>`（实测 2026-08 格式：`@{用户名} 的个人主页`）解析得到。**逐条访问、真实采集，非猜测、非字符串相似度匹配。**

## 映射方法

1. 修改 `D:\WeiboCrawler\crawler\parser.py`，新增 `_extract_user_id` / `_extract_user_name`（基于搜索卡片 DOM 的 `a.name` / `[nick-name]` 属性，与主页 title 互为印证），**未破坏原有 9 个字段**。
2. 新增增量补采脚本 `D:\WeiboCrawler\backfill_user_mapping.py`：
   - 读取 500 个 `weibo_url` → 去重为 403 个唯一 UID
   - 逐 UID 访问主页 → 解析 title 提取 username（多策略兜底）
   - **可断点续跑**、每条立即 flush、仅写新文件（绝不改动 `weibo.csv`）
3. 结果按 UID 去重，**一个 user_id 对应一个 user_name**。

## 成功率

| 指标 | 结果 |
|------|------|
| 总微博数 | 500 |
| 唯一 author UID | 403 |
| **成功获取 username 的微博** | **500 / 500（100%）** |
| **成功建立映射的 UID** | **403 / 403（100%）** |
| username 为空 | 0 |
| 一个 UID 对应多个 username | 0 |
| 一个 username 对应多个 UID | 0 |
| URL / UID 提取异常 | 0 |

## 无法匹配的原因（mentioned 部分）

53 个被提及用户名中，只有 **6 个**（11.32%）能精确匹配到本数据集的 403 个作者：

| 类型 | 数量 | 原因 |
|------|------|------|
| 匹配成功 | 6 | 被提及者本身是本数据集作者（如 `中国新闻周刊`、`于适工作室`、`Lila-阿敏`、`coser小梦酱`、`玉玊则不汏`、`CristianoRonaldo`） |
| 未匹配 | 47 | 被提及者是**外部账号**（名人/媒体/品牌，如 `GDRAGON_OFFICIAL`、`赵丽颖`、`微博明星`、`丁禹兮`），不在本数据集 403 个作者内 |

未匹配 ≠ 采集失败：它们对应的是真实存在的微博账号，只是**不在我们采集的 500 条微博作者集合里**。

## 数据质量限制

1. **显示名不唯一**：微博昵称允许重名。本映射保证的是「本数据集内」name ↔ UID 一一对应（403 对，零冲突），但**不代表全局唯一**——数据集外可能存在同名账号。故匹配仅对数据集内可靠。
2. **@提及匹配是精确字符串匹配**：只做 `mentioned_username == user_name` 的完全相等判断，**未做任何相似度/模糊匹配**（符合约束）。
3. **不伪造**：47 个未匹配提及者，未猜测其 UID、未造边。

## 结论：mention network 闭合程度

- **作者侧已 100% 闭合**：403 个 UID 全部有 username。
- **提及侧闭合 6/53（11.32%）**：仅 6 个被提及者能落到 UID。
- 第一版 **UID→UID** mention network 目前只覆盖 6 条可靠边；其余 47 个提及者若要闭合，需**补采这些外部账号的 uid**（按其用户名/主页 URL 反查，需登录态），仍不得猜测。
