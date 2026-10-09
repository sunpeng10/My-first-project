<template>
  <div class="saliency-text">
    <!-- 预测结果 -->
    <div class="result">
      <div class="result__label" :class="labelClass">
        <span class="result__badge">预测情感</span>
        <span class="result__value">{{ labelText }}</span>
      </div>
      <div class="result__conf">
        <span class="result__badge">置信度</span>
        <span class="result__value mono">{{ confidenceText }}</span>
      </div>
      <div class="result__method mono">{{ methodText }}</div>
    </div>

    <!-- 显著性高亮（字级 / 词级可切换） -->
    <div class="section-title">
      <span>情感显著性</span>
      <div class="granularity">
        <button
          class="granularity__btn"
          :class="{ 'granularity__btn--on': granularity === 'char' }"
          @click="granularity = 'char'"
        >字</button>
        <button
          class="granularity__btn"
          :class="{ 'granularity__btn--on': granularity === 'word' }"
          :disabled="!hasWordLevel"
          :title="hasWordLevel ? '' : '后端未安装 jieba，无词级数据'"
          @click="granularity = 'word'"
        >词</button>
      </div>
    </div>
    <div class="text-box">
      <span
        v-for="(seg, i) in segments"
        :key="i"
        class="tok"
        :class="{ 'tok--hl': seg.score != null }"
        :style="seg.score != null ? hlStyle(seg.score) : null"
        :title="seg.score != null ? titleText(seg) : undefined"
        >{{ seg.text }}</span
      >
    </div>

    <!-- 图例 -->
    <div class="legend">
      <span class="legend__item">
        <i class="legend__swatch" :style="hlStyle(0.9)"></i>高显著性
      </span>
      <span class="legend__item">
        <i class="legend__swatch" :style="hlStyle(0.5)"></i>中等
      </span>
      <span class="legend__item">
        <i class="legend__swatch" :style="hlStyle(0.15)"></i>低
      </span>
      <span class="legend__note">背景强度 = 显著性分数（基于后端 char_range）</span>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  result: { type: Object, required: true },
})

// 显著性展示粒度：词级（jieba）优先，缺数据时退回字级
const granularity = ref('word')

const hasWordLevel = computed(
  () => Array.isArray(props.result.words) && props.result.words.length > 0
)

const LABELS = {
  negative: '负面',
  positive: '正面',
}

const labelText = computed(() => LABELS[props.result.label] || props.result.label || '—')
const labelClass = computed(() => `label--${props.result.label || 'unknown'}`)
const confidenceText = computed(() => {
  const c = props.result.confidence
  if (c == null) return '—'
  return `${(c * 100).toFixed(2)}%`
})
const methodText = computed(() =>
  (props.result.method || 'ig').toUpperCase()
)

/**
 * 依据后端返回的 char_range 将原文切分为片段。
 * 字级（tokens）与词级（words，后端 jieba 分词聚合）共用同一 char_range 结构，
 * 前端不重新计算边界、不引入分词库。
 * char_range 为 [start, end)（tokenizer return_offsets_mapping）。
 */
const segments = computed(() => {
  const text = props.result.text || ''
  const tokens = Array.isArray(props.result.tokens)
    ? props.result.tokens
    : (Array.isArray(props.result.words) ? props.result.words : [])
  const words = Array.isArray(props.result.words) ? props.result.words : []
  // 词级数据缺失时（后端未装 jieba）自动退回字级
  const source = granularity.value === 'word' && words.length ? words : tokens
  const sorted = [...source]
    .filter((w) => Array.isArray(w.char_range) && w.char_range.length === 2)
    .sort((a, b) => a.char_range[0] - b.char_range[0])

  const segs = []
  let cursor = 0
  for (const w of sorted) {
    const [s, e] = w.char_range
    if (s > cursor) {
      segs.push({ text: text.slice(cursor, s), score: null, word: null })
    }
    segs.push({
      text: text.slice(s, e),
      score: typeof w.score === 'number' ? w.score : null,
      scoreRaw: typeof w.score_raw === 'number' ? w.score_raw : null,
      word: w.word,
    })
    cursor = e
  }
  if (cursor < text.length) {
    segs.push({ text: text.slice(cursor), score: null, word: null })
  }
  if (segs.length === 0) {
    segs.push({ text, score: null, word: null })
  }
  return segs
})

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v))
}

/** 显著性分数 -> 背景高亮样式（连续强度）。 */
function hlStyle(score) {
  const s = clamp(score, 0, 1)
  const alpha = 0.12 + 0.78 * s
  return { backgroundColor: `rgba(244, 63, 94, ${alpha.toFixed(3)})` }
}

function titleText(seg) {
  const parts = [`score: ${seg.score.toFixed(4)}`]
  if (seg.scoreRaw != null) parts.push(`score_raw: ${seg.scoreRaw.toFixed(4)}`)
  if (seg.word != null) {
    parts.push(`${granularity.value === 'word' ? '词' : '字'}: ${seg.word}`)
  }
  return parts.join('\n')
}
</script>

<style scoped>
.result {
  display: flex;
  align-items: center;
  gap: 18px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}

.result__label,
.result__conf {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.result__badge {
  font-size: 12px;
  color: var(--text-muted);
}

.result__value {
  font-size: 20px;
  font-weight: 700;
}

.label--negative .result__value {
  color: var(--danger);
}
.label--positive .result__value {
  color: var(--success);
}
.label--unknown .result__value {
  color: var(--text-primary);
}

.result__conf .result__value {
  color: var(--accent);
}

.result__method {
  margin-left: auto;
  font-size: 11px;
  color: var(--text-muted);
  border: 1px solid var(--border-strong);
  padding: 2px 8px;
  border-radius: 999px;
}

.section-title {
  font-size: 12px;
  color: var(--text-muted);
  letter-spacing: 1px;
  text-transform: uppercase;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.granularity {
  display: inline-flex;
  gap: 4px;
}

.granularity__btn {
  border: 1px solid var(--border-strong);
  background: transparent;
  color: var(--text-secondary);
  font-size: 11px;
  padding: 2px 9px;
  border-radius: 999px;
  cursor: pointer;
  letter-spacing: 0;
}

.granularity__btn--on {
  border-color: var(--accent);
  color: var(--accent);
  background: rgba(34, 211, 238, 0.08);
}

.granularity__btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.text-box {
  padding: 14px;
  background: var(--bg-base);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  line-height: 2.3;
  font-size: 16px;
  word-break: break-all;
}

.tok {
  border-radius: 3px;
  padding: 2px 1px;
  transition: background-color 0.12s ease;
}

.tok--hl:hover {
  outline: 1px solid var(--accent);
}

.legend {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-secondary);
}

.legend__item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.legend__swatch {
  width: 14px;
  height: 14px;
  border-radius: 3px;
  display: inline-block;
  border: 1px solid rgba(244, 63, 94, 0.5);
}

.legend__note {
  color: var(--text-muted);
  font-size: 11px;
}
</style>
