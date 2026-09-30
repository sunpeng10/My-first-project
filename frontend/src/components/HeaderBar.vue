<template>
  <header class="header">
    <div class="header__left">
      <h1 class="header__title">多模态社交媒体舆情分析系统</h1>
      <div class="header__subtitle">Text · Vision · Propagation Network</div>
    </div>
    <div class="header__right">
      <div class="health" :class="`health--${status.level}`">
        <span class="health__dot"></span>
        <span class="health__text">{{ status.text }}</span>
      </div>
    </div>
  </header>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  health: { type: Object, default: null },
  healthError: { type: String, default: '' },
})

const MODEL_KEYS = ['text_model', 'image_model', 'gat_graph']

const status = computed(() => {
  if (props.healthError) {
    return { level: 'offline', text: '后端离线' }
  }
  if (!props.health) {
    return { level: 'connecting', text: '连接中…' }
  }
  const allLoaded = MODEL_KEYS.every((k) => props.health[k] === 'loaded')
  if (allLoaded) {
    return { level: 'online', text: '系统在线' }
  }
  return { level: 'partial', text: '部分服务不可用' }
})
</script>

<style scoped>
.header {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 28px;
  background: rgba(10, 14, 23, 0.82);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border);
}

.header__title {
  margin: 0;
  font-size: 20px;
  font-weight: 700;
  letter-spacing: 1px;
  background: linear-gradient(90deg, #e5e7eb 0%, #22d3ee 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.header__subtitle {
  margin-top: 2px;
  font-size: 12px;
  color: var(--text-muted);
  letter-spacing: 2px;
  font-family: var(--mono);
}

.health {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 7px 14px;
  border: 1px solid var(--border-strong);
  border-radius: 999px;
  font-size: 13px;
  font-weight: 500;
}

.health__dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  flex: none;
}

.health--online {
  color: var(--success);
  border-color: rgba(52, 211, 153, 0.4);
}
.health--online .health__dot {
  background: var(--success);
  box-shadow: 0 0 10px var(--success);
}

.health--partial {
  color: var(--warning);
  border-color: rgba(245, 158, 11, 0.4);
}
.health--partial .health__dot {
  background: var(--warning);
  box-shadow: 0 0 10px var(--warning);
}

.health--offline {
  color: var(--danger);
  border-color: rgba(244, 63, 94, 0.4);
}
.health--offline .health__dot {
  background: var(--danger);
  box-shadow: 0 0 10px var(--danger);
}

.health--connecting {
  color: var(--text-muted);
}
.health--connecting .health__dot {
  background: var(--text-muted);
}
</style>
