<template>
  <figure class="saliency-image">
    <figcaption class="saliency-image__caption">
      <span class="dot"></span>{{ title }}
    </figcaption>
    <div class="saliency-image__frame">
      <img
        v-if="src"
        :src="src"
        :alt="title"
        @error="onError"
        @load="onLoad"
      />
      <div v-else class="saliency-image__placeholder">—</div>
    </div>
  </figure>
</template>

<script setup>
import { ref } from 'vue'

defineProps({
  src: { type: String, default: '' },
  title: { type: String, default: '' },
})

const failed = ref(false)

function onError() {
  failed.value = true
}
function onLoad() {
  failed.value = false
}
</script>

<style scoped>
.saliency-image {
  margin: 0;
}

.saliency-image__caption {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 8px;
  letter-spacing: 0.5px;
}

.saliency-image__caption .dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent);
}

.saliency-image__frame {
  background: var(--bg-base);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  aspect-ratio: 4 / 3;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.saliency-image__frame img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  display: block;
}

.saliency-image__placeholder {
  color: var(--text-muted);
  font-size: 28px;
}
</style>
