<template>
  <section class="panel node-detail">
    <div class="panel__header">
      <span class="dot"></span>
      <h3>节点详情</h3>
      <span v-if="node && node.username" class="node-name">{{ node.username }}</span>
      <button class="close-btn" @click="$emit('close')">✕</button>
    </div>

    <div class="panel__body">
      <div v-if="loading" class="status status--loading">
        <span class="spinner"></span> 正在加载节点详情…
      </div>
      <div v-else-if="error" class="status status--error">{{ error }}</div>

      <template v-else-if="node">
        <!-- 字段展示（仅显示后端真实返回的字段） -->
        <div class="fields">
          <div v-for="f in fields" :key="f.key" class="field">
            <div class="field__label">{{ f.label }}</div>
            <div class="field__value" :class="{ mono: f.mono }">{{ f.value }}</div>
          </div>
        </div>

        <!-- Attention 关系 -->
        <div class="attention">
          <div class="attention__title">
            该节点相关 Attention 边
            <span v-if="attention" class="attention__count mono">共 {{ attention.count }} 条</span>
          </div>

          <div v-if="attentionLoading" class="status status--loading">
            <span class="spinner"></span> 正在加载 Attention 关系…
          </div>
          <div v-else-if="attentionError" class="status status--error">
            {{ attentionError }}
          </div>
          <div v-else-if="!attention || !attentionEdges.length" class="status status--empty">
            无 Attention 边数据
          </div>
          <div v-else class="table-wrap">
            <table class="table">
              <thead>
                <tr>
                  <th>Source</th>
                  <th>Target</th>
                  <th>Layer</th>
                  <th>Head</th>
                  <th>Attention Weight</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(e, i) in attentionEdges" :key="i">
                  <td>
                    <div class="cell-user">{{ displayName(e.source_uid) }}</div>
                  </td>
                  <td>
                    <div class="cell-user">{{ displayName(e.target_uid) }}</div>
                  </td>
                  <td class="mono">{{ e.layer }}</td>
                  <td class="mono">{{ e.head }}</td>
                  <td class="mono">{{ Number(e.attention_weight).toFixed(4) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- 该作者微博：文案 + 图片 + 情感/显著性分析 -->
        <div class="posts">
          <div class="posts__title">
            该作者微博
            <span v-if="authoredPosts.length" class="posts__count mono">共 {{ authoredPosts.length }} 条</span>
          </div>

          <div v-if="!authoredPosts.length" class="status status--empty">
            该用户无原创微博
          </div>

          <div v-for="post in authoredPosts" :key="post.image_id" class="post-card">
            <div class="post-card__meta">
              <span class="mono">{{ post.image_id }}</span>
              <a
                v-if="post.weibo_url"
                :href="post.weibo_url"
                target="_blank"
                rel="noopener"
                class="post-card__link"
              >微博原文 ↗</a>
            </div>

            <div class="post-card__text">{{ post.text }}</div>

            <button
              class="analyze-btn"
              :disabled="textLoading[post.image_id] || !post.text"
              @click="analyzeTextOfPost(post)"
            >
              {{ textLoading[post.image_id] ? '情感分析中…' : '情感分析' }}
            </button>
            <div v-if="textError[post.image_id]" class="status status--error">{{ textError[post.image_id] }}</div>
            <SaliencyText v-if="textResult[post.image_id]" :result="textResult[post.image_id]" class="post-card__result" />

            <template v-if="post.has_image">
              <img
                class="post-card__image"
                :src="postImageUrl(post.image_id)"
                :alt="post.image_id"
                @error="onImgError"
              />

              <button
                class="analyze-btn"
                :disabled="imgLoading[post.image_id]"
                @click="analyzeImageOfPost(post)"
              >
                {{ imgLoading[post.image_id] ? '显著性检测中…' : '显著性检测' }}
              </button>
              <div v-if="imgError[post.image_id]" class="status status--error">{{ imgError[post.image_id] }}</div>

              <template v-if="imgResult[post.image_id]">
                <div class="post-card__saliency">
                  <SaliencyImage :src="postImageUrl(post.image_id)" title="原图" />
                  <SaliencyImage :src="saliencyUrlOf(imgResult[post.image_id])" title="显著性图" />
                </div>
                <div class="post-card__metrics">
                  <StatCard
                    v-for="m in saliencyMetrics(imgResult[post.image_id])"
                    :key="m.key"
                    :label="m.label"
                    :value="m.value"
                  />
                </div>
              </template>
            </template>
          </div>
        </div>
      </template>
    </div>
  </section>
</template>

<script setup>
import { computed, reactive } from 'vue'
import { api, resolveUrl, ApiError } from '../api'
import SaliencyText from './SaliencyText.vue'
import SaliencyImage from './SaliencyImage.vue'
import StatCard from './StatCard.vue'

const props = defineProps({
  node: { type: Object, default: null },
  attention: { type: Object, default: null },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
  attentionLoading: { type: Boolean, default: false },
  attentionError: { type: String, default: '' },
  usernameMap: { type: Object, default: () => ({}) },
})

defineEmits(['close'])

const fmtPr = (v) => {
  const n = Number(v)
  if (n === 0) return '0'
  return n < 0.001 ? n.toExponential(3) : n.toFixed(6)
}

const FIELD_DEFS = [
  { key: 'uid', label: 'UID', mono: true },
  { key: 'gat_score', label: 'GAT Score', mono: true, fmt: (v) => Number(v).toFixed(4) },
  { key: 'predicted_label', label: 'Predicted Label', mono: true },
  { key: 'pseudo_label', label: 'Pseudo Label', mono: true },
  { key: 'core_score', label: 'Core Score', mono: true, fmt: (v) => Number(v).toFixed(4) },
  { key: 'pagerank', label: 'PageRank', mono: true, fmt: fmtPr },
  { key: 'in_degree', label: 'In Degree', mono: true },
  { key: 'out_degree', label: 'Out Degree', mono: true },
  { key: 'total_degree', label: 'Total Degree', mono: true },
  { key: 'weighted_in_degree', label: 'Weighted In Degree', mono: true },
  { key: 'weighted_out_degree', label: 'Weighted Out Degree', mono: true },
  { key: 'posts_commented', label: 'Posts Commented', mono: true },
  { key: 'posts_authored', label: 'Posts Authored', mono: true },
  { key: 'comment_count', label: 'Comment Count', mono: true },
  { key: 'component_size', label: 'Component Size', mono: true },
]

const fields = computed(() => {
  const n = props.node || {}
  return FIELD_DEFS.filter((f) => n[f.key] != null).map((f) => ({
    key: f.key,
    label: f.label,
    mono: f.mono,
    value: f.fmt ? f.fmt(n[f.key]) : String(n[f.key]),
  }))
})

const attentionEdges = computed(() => {
  const edges = (props.attention && props.attention.edges) || []
  return [...edges]
    .sort((a, b) => Number(b.attention_weight) - Number(a.attention_weight))
    .slice(0, 20)
})

function displayName(uid) {
  const u = String(uid)
  const name = props.usernameMap[u]
  return name ? `${name}` : u
}

// ------------------------------------------------------------------
// 该作者微博：文案 + 图片 + 情感/显著性分析
// ------------------------------------------------------------------
const authoredPosts = computed(() => {
  const posts = props.node && Array.isArray(props.node.authored_posts)
    ? props.node.authored_posts
    : []
  return posts
})

const textResult = reactive({})   // image_id -> 情感分析结果
const textLoading = reactive({})  // image_id -> bool
const textError = reactive({})    // image_id -> string
const imgResult = reactive({})    // image_id -> 显著性结果
const imgLoading = reactive({})   // image_id -> bool
const imgError = reactive({})     // image_id -> string

function postImageUrl(imageId) {
  return api.postImageUrl(imageId)
}

function onImgError(e) {
  e.target.style.display = 'none'
}

async function analyzeTextOfPost(post) {
  const id = post.image_id
  if (!post.text || !post.text.trim()) return
  textLoading[id] = true
  textError[id] = ''
  delete textResult[id]
  try {
    textResult[id] = await api.analyzeText(post.text.trim(), 'ig')
  } catch (e) {
    textError[id] = e instanceof ApiError ? e.message : '文本分析失败'
  } finally {
    textLoading[id] = false
  }
}

async function analyzeImageOfPost(post) {
  const id = post.image_id
  imgLoading[id] = true
  imgError[id] = ''
  delete imgResult[id]
  try {
    imgResult[id] = await api.analyzePostImage(id)
  } catch (e) {
    imgError[id] = e instanceof ApiError ? e.message : '图片分析失败'
  } finally {
    imgLoading[id] = false
  }
}

function saliencyUrlOf(r) {
  return r ? resolveUrl(r.saliency_map_url) : ''
}

function saliencyMetrics(r) {
  if (!r || !r.saliency) return []
  const s = r.saliency
  const pct = (v) => `${(v * 100).toFixed(2)}%`
  const num = (v) => v.toFixed(4)
  const defs = [
    { key: 'saliency_area_ratio', label: 'Saliency Area', fmt: (v) => pct(v) },
    { key: 'mean_saliency', label: 'Mean Saliency', fmt: (v) => num(v) },
    { key: 'saliency_std', label: 'Saliency Std', fmt: (v) => num(v) },
    { key: 'saliency_entropy', label: 'Entropy', fmt: (v) => num(v) },
    { key: 'center_bias', label: 'Center Bias', fmt: (v) => (v < 0 ? '无显著区域' : num(v)) },
    { key: 'component_count', label: 'Components', fmt: (v) => String(Math.round(v)) },
    { key: 'largest_component_ratio', label: 'Largest Component', fmt: (v) => pct(v) },
  ]
  return defs
    .filter((d) => s[d.key] != null)
    .map((d) => ({ key: d.key, label: d.label, value: d.fmt(s[d.key]) }))
}
</script>

<style scoped>
.node-name {
  font-weight: 600;
  color: var(--accent);
  margin-left: 6px;
}

.close-btn {
  margin-left: auto;
  border: none;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 14px;
}

.close-btn:hover {
  color: var(--danger);
}

.fields {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 10px;
  margin-bottom: 18px;
}

.field {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 10px 12px;
}

.field__label {
  font-size: 11px;
  color: var(--text-muted);
  margin-bottom: 4px;
}

.field__value {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
}

.attention__title {
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.attention__count {
  font-size: 11px;
  color: var(--text-muted);
}

.table-wrap {
  max-height: 360px;
  overflow-y: auto;
}

.cell-user {
  max-width: 220px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 该作者微博 */
.posts {
  margin-top: 22px;
  border-top: 1px solid var(--border);
  padding-top: 16px;
}

.posts__title {
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.posts__count {
  font-size: 11px;
  color: var(--text-muted);
}

.post-card {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  margin-bottom: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.post-card__meta {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  color: var(--text-muted);
}

.post-card__link {
  font-size: 12px;
  color: var(--accent);
  text-decoration: none;
}

.post-card__link:hover {
  text-decoration: underline;
}

.post-card__text {
  font-size: 14px;
  line-height: 1.6;
  color: var(--text-primary);
  word-break: break-word;
  white-space: pre-wrap;
}

.post-card__image {
  max-width: 100%;
  max-height: 260px;
  object-fit: contain;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  display: block;
  background: var(--bg-base);
}

.post-card__saliency {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.post-card__metrics {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 8px;
}

.post-card__result {
  margin-top: 4px;
}

.analyze-btn {
  align-self: flex-start;
  padding: 6px 14px;
  border: 1px solid var(--accent);
  border-radius: 999px;
  background: rgba(34, 211, 238, 0.08);
  color: var(--accent);
  font-size: 12px;
  cursor: pointer;
}

.analyze-btn:hover:not(:disabled) {
  background: rgba(34, 211, 238, 0.18);
}

.analyze-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
