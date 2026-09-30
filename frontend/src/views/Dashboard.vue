<template>
  <div class="page">
    <HeaderBar :health="health" :health-error="healthError" />

    <main class="dashboard">
      <!-- 第一行：文本 + 图像 -->
      <div class="row row--top">
        <TextAnalysisPanel />
        <ImageAnalysisPanel />
      </div>

      <!-- 第二行：传播网络 + 核心节点 Top -->
      <div class="row row--net">
        <NetworkGraph
          :graph="graph"
          :highlight-uids="highlightUids"
          :loading="graphLoading"
          :error="graphError"
          @node-click="selectNode"
        />
        <TopNodesPanel
          :nodes="topNodes"
          :loading="topLoading"
          :error="topError"
          :limit="topLimit"
          :selected-uid="selectedUid"
          @change-limit="onChangeLimit"
          @select="selectNode"
        />
      </div>

      <!-- ECharts 统计 -->
      <ChartsPanel :nodes="topNodes" />

      <!-- 节点详情（点击后显示） -->
      <NodeDetailPanel
        v-if="selectedUid"
        :node="nodeDetail"
        :attention="attention"
        :loading="nodeLoading"
        :error="nodeError"
        :attention-loading="attentionLoading"
        :attention-error="attentionError"
        :username-map="usernameMap"
        @close="closeNode"
      />
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { api, ApiError } from '../api'
import HeaderBar from '../components/HeaderBar.vue'
import TextAnalysisPanel from '../components/TextAnalysisPanel.vue'
import ImageAnalysisPanel from '../components/ImageAnalysisPanel.vue'
import NetworkGraph from '../components/NetworkGraph.vue'
import TopNodesPanel from '../components/TopNodesPanel.vue'
import ChartsPanel from '../components/ChartsPanel.vue'
import NodeDetailPanel from '../components/NodeDetailPanel.vue'

// 后端健康状态
const health = ref(null)
const healthError = ref('')

// 传播网络
const graph = ref(null)
const graphLoading = ref(false)
const graphError = ref('')

// 核心节点
const topNodes = ref([])
const topLimit = ref(20)
const topLoading = ref(false)
const topError = ref('')

// 选中节点
const selectedUid = ref(null)
const nodeDetail = ref(null)
const nodeLoading = ref(false)
const nodeError = ref('')
const attention = ref(null)
const attentionLoading = ref(false)
const attentionError = ref('')

const highlightUids = computed(() => topNodes.value.map((n) => String(n.uid)))

const usernameMap = computed(() => {
  const m = {}
  const g = graph.value
  if (g && Array.isArray(g.nodes)) {
    for (const n of g.nodes) {
      const uid = String(n.id ?? n.uid ?? '')
      if (uid && n.username) m[uid] = n.username
    }
  }
  return m
})

async function checkHealth() {
  healthError.value = ''
  try {
    health.value = await api.checkHealth()
  } catch (e) {
    health.value = null
    healthError.value = e instanceof ApiError ? e.message : '无法连接后端'
  }
}

async function loadGraph() {
  graphLoading.value = true
  graphError.value = ''
  try {
    graph.value = await api.getGraph()
  } catch (e) {
    graphError.value = e instanceof ApiError ? e.message : '加载传播网络失败'
  } finally {
    graphLoading.value = false
  }
}

async function loadTopNodes(limit) {
  topLoading.value = true
  topError.value = ''
  try {
    const data = await api.getTopNodes(limit)
    topNodes.value = (data && data.nodes) || []
  } catch (e) {
    topError.value = e instanceof ApiError ? e.message : '加载核心节点失败'
  } finally {
    topLoading.value = false
  }
}

function onChangeLimit(limit) {
  if (limit === topLimit.value) return
  topLimit.value = limit
  loadTopNodes(limit)
}

async function selectNode(uid) {
  const u = String(uid)
  selectedUid.value = u
  nodeDetail.value = null
  attention.value = null
  nodeError.value = ''
  attentionError.value = ''

  nodeLoading.value = true
  attentionLoading.value = true

  const [n, a] = await Promise.allSettled([
    api.getNode(u),
    api.getAttention(u, 1000),
  ])

  nodeLoading.value = false
  attentionLoading.value = false

  if (n.status === 'fulfilled') {
    nodeDetail.value = n.value
  } else {
    nodeError.value = n.reason instanceof ApiError ? n.reason.message : '加载节点详情失败'
  }

  if (a.status === 'fulfilled') {
    attention.value = a.value
  } else {
    attentionError.value =
      a.reason instanceof ApiError ? a.reason.message : '加载 Attention 失败'
  }
}

function closeNode() {
  selectedUid.value = null
  nodeDetail.value = null
  attention.value = null
}

onMounted(() => {
  checkHealth()
  loadGraph()
  loadTopNodes(topLimit.value)
})
</script>

<style scoped>
.page {
  min-height: 100vh;
}

.dashboard {
  max-width: 1720px;
  margin: 0 auto;
  padding: 24px 28px 48px;
  display: flex;
  flex-direction: column;
  gap: 22px;
}

.row {
  display: grid;
  gap: 22px;
}

.row--top {
  grid-template-columns: 1fr 1fr;
}

.row--net {
  grid-template-columns: 3fr 2fr;
  align-items: stretch;
}

@media (max-width: 1100px) {
  .row--top,
  .row--net {
    grid-template-columns: 1fr;
  }
}
</style>
