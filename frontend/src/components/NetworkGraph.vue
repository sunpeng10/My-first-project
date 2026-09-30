<template>
  <section class="panel network-panel">
    <div class="panel__header">
      <span class="dot"></span>
      <h3>传播网络</h3>
      <span class="sub">{{ stats }}</span>
    </div>

    <div class="network-body">
      <div ref="container" class="network-container"></div>
      <div v-if="loading" class="network-overlay">
        <span class="spinner"></span> 正在加载传播网络…
      </div>
      <div v-else-if="error" class="network-overlay network-overlay--error">
        {{ error }}
      </div>
    </div>

    <div class="network-note">
      传播网络基于每条微博最多 20 条一级评论构建；节点大小为 GAT score 映射，颜色越暖代表 GAT score 越高。点击节点查看详情。
    </div>
  </section>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { DataSet, Network } from 'vis-network/standalone'
import 'vis-network/styles/vis-network.css'

const props = defineProps({
  graph: { type: Object, default: null },
  highlightUids: { type: Array, default: () => [] },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
})

const emit = defineEmits(['node-click'])

const container = ref(null)
let network = null
let nodesDs = null
let graphNodes = [] // 原始节点信息（含 uid/username/score 等），供 label 更新

const highlightSet = computed(
  () => new Set((props.highlightUids || []).map((u) => String(u)))
)

const stats = computed(() => {
  const g = props.graph
  if (!g) return ''
  const n = Array.isArray(g.nodes) ? g.nodes.length : 0
  const e = Array.isArray(g.edges) ? g.edges.length : 0
  return `${n} nodes / ${e} edges`
})

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v))
}

/** GAT score -> 颜色（低分偏蓝，高分偏暖红）。 */
function scoreColor(score) {
  const t = clamp(Number(score) || 0, 0, 1)
  const hue = 217 - (217 - 350) * t
  return `hsl(${hue.toFixed(0)}, 80%, 58%)`
}

/** GAT score -> 节点直径（平方根压缩，避免差异过于夸张）。 */
function nodeSize(score) {
  const t = clamp(Number(score) || 0, 0, 1)
  return 4 + 24 * Math.sqrt(t)
}

/** 依据 highlightSet 更新节点 label（原地更新，不重跑物理模拟）。 */
function applyLabels() {
  if (!nodesDs || !graphNodes.length) return
  const set = highlightSet.value
  const updates = graphNodes.map((n) => ({
    id: n.uid,
    label: set.has(n.uid) ? n.username || n.uid : '',
  }))
  nodesDs.update(updates)
}

function buildNetwork() {
  const g = props.graph
  if (!g || !container.value) return
  if (network) {
    network.destroy()
    network = null
  }

  const nodeArr = Array.isArray(g.nodes) ? g.nodes : []
  const edgeArr = Array.isArray(g.edges) ? g.edges : []

  graphNodes = nodeArr.map((n) => ({
    uid: String(n.id ?? n.uid ?? ''),
    username: n.username || String(n.id ?? n.uid ?? ''),
    score: Number(n.score ?? n.gat_score ?? 0),
  }))

  // 边宽：优先 attention，其次 weight，按 min-max 归一化。
  let minV = Infinity
  let maxV = -Infinity
  const edgeVals = edgeArr.map((e) => {
    const v =
      e.attention != null ? Number(e.attention) : Number(e.weight != null ? e.weight : 1)
    return Number.isFinite(v) ? v : 1
  })
  for (const v of edgeVals) {
    if (v < minV) minV = v
    if (v > maxV) maxV = v
  }
  const range = maxV - minV || 1

  nodesDs = new DataSet(
    nodeArr.map((n) => {
      const uid = String(n.id ?? n.uid ?? '')
      const score = Number(n.score ?? n.gat_score ?? 0)
      const pr = n.pagerank != null ? Number(n.pagerank) : null
      const deg =
        n.degree != null
          ? n.degree
          : n.total_degree != null
            ? n.total_degree
            : null
      const username = n.username || uid
      return {
        id: uid,
        label: '',
        title:
          `${username}<br/>GAT score: ${Number(score).toFixed(4)}<br/>` +
          `PageRank: ${pr != null ? pr.toExponential(3) : '—'}<br/>` +
          `Degree: ${deg != null ? deg : '—'}`,
        size: nodeSize(score),
        color: scoreColor(score),
        font: {
          size: 14,
          color: '#e5e7eb',
          face: 'system-ui',
          strokeWidth: 3,
          strokeColor: '#0a0e17',
        },
      }
    })
  )

  const edgesDs = new DataSet(
    edgeArr.map((e, i) => {
      const v = edgeVals[i]
      const att = e.attention != null ? Number(e.attention) : null
      const w = e.weight != null ? Number(e.weight) : 1
      const width = 0.3 + 3.2 * ((v - minV) / range)
      return {
        from: String(e.source),
        to: String(e.target),
        width,
        title: att != null ? `attention: ${att.toFixed(4)}` : `weight: ${w}`,
      }
    })
  )

  const options = {
    nodes: {
      shape: 'dot',
      borderWidth: 1,
      borderWidthSelected: 2,
    },
    edges: {
      color: { color: 'rgba(148, 163, 184, 0.22)', highlight: '#22d3ee', hover: '#22d3ee' },
      smooth: { enabled: true, type: 'continuous' },
      selectionWidth: 1.5,
    },
    physics: {
      enabled: true,
      stabilization: { enabled: true, iterations: 160, updateInterval: 20 },
      barnesHut: {
        gravitationalConstant: -8000,
        centralGravity: 0.3,
        springLength: 120,
        springConstant: 0.04,
        damping: 0.4,
      },
    },
    interaction: {
      hover: true,
      tooltipDelay: 120,
      dragNodes: true,
      dragView: true,
      zoomView: true,
      navigationButtons: false,
    },
  }

  network = new Network(container.value, { nodes: nodesDs, edges: edgesDs }, options)

  network.on('click', (params) => {
    if (params.nodes && params.nodes.length > 0) {
      emit('node-click', String(params.nodes[0]))
    }
  })

  // 稳定后冻结物理模拟，避免持续抖动。
  network.once('stabilizationIterationsDone', () => {
    if (network) network.setOptions({ physics: { enabled: false } })
  })

  applyLabels()
}

onMounted(() => nextTick(buildNetwork))
watch(
  () => props.graph,
  () => nextTick(buildNetwork)
)
watch(highlightSet, () => applyLabels())

onBeforeUnmount(() => {
  if (network) {
    network.destroy()
    network = null
  }
  nodesDs = null
  graphNodes = []
})
</script>

<style scoped>
.network-panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.network-body {
  position: relative;
  flex: 1;
  min-height: 480px;
}

.network-container {
  position: absolute;
  inset: 0;
}

.network-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: var(--accent);
  background: rgba(10, 14, 23, 0.7);
  font-size: 13px;
}

.network-overlay--error {
  color: var(--danger);
}

.network-note {
  padding: 10px 18px;
  border-top: 1px solid var(--border);
  font-size: 11px;
  color: var(--text-muted);
  line-height: 1.6;
}
</style>
