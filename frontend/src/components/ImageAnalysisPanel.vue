<template>
  <section class="panel">
    <div class="panel__header">
      <span class="dot"></span>
      <h3>图像显著性分析</h3>
      <span class="sub">BASNet</span>
    </div>

    <div class="panel__body">
      <!-- 上传区 -->
      <div
        class="dropzone"
        :class="{ 'dropzone--drag': dragging }"
        @click="pick"
        @dragover.prevent="dragging = true"
        @dragleave.prevent="dragging = false"
        @drop.prevent="onDrop"
      >
        <input
          ref="fileInput"
          type="file"
          accept=".jpg,.jpeg,.png,.webp"
          class="hidden-input"
          @change="onFileChange"
        />
        <div v-if="!fileName" class="dropzone__empty">
          <span class="dropzone__icon">⇪</span>
          <span>点击选择或拖拽图片到此处</span>
          <span class="dropzone__hint">支持 jpg / jpeg / png / webp，最大 10MB</span>
        </div>
        <div v-else class="dropzone__file mono">{{ fileName }}</div>
      </div>

      <div v-if="loading" class="status status--loading" style="margin-top: 12px">
        <span class="spinner"></span> BASNet 推理处理中…
      </div>
      <div v-else-if="error" class="status status--error" style="margin-top: 12px">
        {{ error }}
      </div>

      <!-- 结果 -->
      <template v-if="result">
        <div class="images">
          <SaliencyImage :src="previewUrl" title="原图" />
          <SaliencyImage
            :src="saliencyUrl"
            title="BASNet 显著性图"
          />
        </div>

        <div class="metrics-title">视觉显著性指标（7 项）</div>
        <div class="metrics">
          <StatCard
            v-for="m in metricRows"
            :key="m.key"
            :label="m.label"
            :value="m.value"
            :hint="m.hint"
          />
        </div>
      </template>
    </div>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'
import { api, resolveUrl, ApiError } from '../api'
import StatCard from './StatCard.vue'
import SaliencyImage from './SaliencyImage.vue'

const fileInput = ref(null)
const dragging = ref(false)
const fileName = ref('')
const previewUrl = ref('')
const loading = ref(false)
const error = ref('')
const result = ref(null)

let objectUrl = null

function revokePreview() {
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl)
    objectUrl = null
  }
}
onBeforeUnmount(revokePreview)

const ALLOWED = ['jpg', 'jpeg', 'png', 'webp']

function pick() {
  fileInput.value && fileInput.value.click()
}

function onDrop(e) {
  dragging.value = false
  const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]
  if (file) processFile(file)
}

function onFileChange(e) {
  const file = e.target.files && e.target.files[0]
  if (file) processFile(file)
  e.target.value = ''
}

function processFile(file) {
  const ext = (file.name.split('.').pop() || '').toLowerCase()
  if (!ALLOWED.includes(ext)) {
    error.value = `不支持的图片类型 .${ext}（允许：jpg / jpeg / png / webp）`
    return
  }
  fileName.value = file.name
  error.value = ''
  result.value = null
  revokePreview()
  objectUrl = URL.createObjectURL(file)
  previewUrl.value = objectUrl

  upload(file)
}

async function upload(file) {
  loading.value = true
  try {
    result.value = await api.analyzeImage(file)
  } catch (e) {
    error.value = e instanceof ApiError ? e.message : '图片分析失败'
  } finally {
    loading.value = false
  }
}

const saliencyUrl = computed(() =>
  result.value ? resolveUrl(result.value.saliency_map_url) : ''
)

const metricRows = computed(() => {
  if (!result.value || !result.value.saliency) return []
  const s = result.value.saliency
  const pct = (v) => `${(v * 100).toFixed(2)}%`
  const num = (v) => v.toFixed(4)
  const defs = [
    { key: 'saliency_area_ratio', label: 'Saliency Area', fmt: (v) => pct(v) },
    { key: 'mean_saliency', label: 'Mean Saliency', fmt: (v) => num(v) },
    { key: 'saliency_std', label: 'Saliency Std', fmt: (v) => num(v) },
    { key: 'saliency_entropy', label: 'Entropy', fmt: (v) => num(v) },
    {
      key: 'center_bias',
      label: 'Center Bias',
      fmt: (v) => (v < 0 ? '无显著区域' : num(v)),
    },
    { key: 'component_count', label: 'Components', fmt: (v) => String(Math.round(v)) },
    { key: 'largest_component_ratio', label: 'Largest Component', fmt: (v) => pct(v) },
  ]
  return defs
    .filter((d) => s[d.key] != null)
    .map((d) => ({ key: d.key, label: d.label, value: d.fmt(s[d.key]) }))
})
</script>

<style scoped>
.dropzone {
  border: 1px dashed var(--border-strong);
  border-radius: var(--radius-sm);
  padding: 22px 16px;
  text-align: center;
  cursor: pointer;
  transition: all 0.15s ease;
  background: var(--bg-base);
}

.dropzone:hover,
.dropzone--drag {
  border-color: var(--accent);
  background: rgba(34, 211, 238, 0.04);
}

.dropzone__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  color: var(--text-secondary);
  font-size: 13px;
}

.dropzone__icon {
  font-size: 22px;
  color: var(--accent);
}

.dropzone__hint {
  font-size: 11px;
  color: var(--text-muted);
}

.dropzone__file {
  color: var(--accent);
  font-size: 13px;
}

.hidden-input {
  display: none;
}

.images {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin-top: 16px;
}

.metrics-title {
  margin: 16px 0 10px;
  font-size: 12px;
  color: var(--text-muted);
  letter-spacing: 1px;
  text-transform: uppercase;
}

.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 10px;
}
</style>
