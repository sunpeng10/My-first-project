<template>
  <section class="panel">
    <div class="panel__header">
      <span class="dot"></span>
      <h3>文本情感分析</h3>
      <span class="sub">RoBERTa · Captum</span>
    </div>

    <div class="panel__body">
      <textarea
        v-model="text"
        placeholder="请输入要分析的文本…"
        maxlength="5000"
      ></textarea>

      <div class="controls">
        <div class="method-group">
          <span class="method-label">归因方法</span>
          <label class="radio" :class="{ 'radio--on': method === 'ig' }">
            <input v-model="method" type="radio" value="ig" />
            <span>IG</span>
          </label>
          <label class="radio" :class="{ 'radio--on': method === 'saliency' }">
            <input v-model="method" type="radio" value="saliency" />
            <span>Saliency</span>
          </label>
        </div>
        <button class="btn btn--primary" :disabled="loading || !text.trim()" @click="run">
          {{ loading ? '分析中…' : '开始文本分析' }}
        </button>
      </div>

      <div v-if="loading" class="status status--loading">
        <span class="spinner"></span> 正在调用模型，分析中…
      </div>
      <div v-else-if="error" class="status status--error">{{ error }}</div>
      <div v-else-if="!result" class="status status--empty">
        输入文本后点击「开始文本分析」
      </div>

      <SaliencyText v-if="result" :result="result" class="result-block" />
    </div>
  </section>
</template>

<script setup>
import { ref } from 'vue'
import { api, ApiError } from '../api'
import SaliencyText from './SaliencyText.vue'

const text = ref('这个手机太差了')
const method = ref('ig')
const loading = ref(false)
const error = ref('')
const result = ref(null)

async function run() {
  if (!text.value.trim()) return
  loading.value = true
  error.value = ''
  result.value = null
  try {
    result.value = await api.analyzeText(text.value.trim(), method.value)
  } catch (e) {
    error.value = e instanceof ApiError ? e.message : '文本分析失败'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.controls {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin: 12px 0 14px;
  flex-wrap: wrap;
}

.method-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.method-label {
  font-size: 12px;
  color: var(--text-muted);
  margin-right: 4px;
}

.radio {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 6px 12px;
  border: 1px solid var(--border-strong);
  border-radius: 999px;
  font-size: 12px;
  cursor: pointer;
  color: var(--text-secondary);
}

.radio input {
  display: none;
}

.radio--on {
  border-color: var(--accent);
  color: var(--accent);
  background: rgba(34, 211, 238, 0.08);
}

.result-block {
  margin-top: 16px;
}
</style>
