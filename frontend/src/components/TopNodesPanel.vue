<template>
  <section class="panel">
    <div class="panel__header">
      <span class="dot"></span>
      <h3>核心节点 Top {{ limit }}</h3>
      <div class="limit-toggle">
        <button
          class="limit-btn"
          :class="{ 'limit-btn--on': limit === 10 }"
          @click="$emit('change-limit', 10)"
        >
          Top 10
        </button>
        <button
          class="limit-btn"
          :class="{ 'limit-btn--on': limit === 20 }"
          @click="$emit('change-limit', 20)"
        >
          Top 20
        </button>
      </div>
    </div>

    <div class="panel__body">
      <div v-if="loading" class="status status--loading">
        <span class="spinner"></span> 正在加载核心节点…
      </div>
      <div v-else-if="error" class="status status--error">{{ error }}</div>
      <div v-else-if="!nodes.length" class="status status--empty">暂无数据</div>

      <div v-else class="table-wrap">
        <table class="table">
          <thead>
            <tr>
              <th>#</th>
              <th>用户名</th>
              <th>UID</th>
              <th>GAT Score</th>
              <th>PageRank</th>
              <th>Degree</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="n in nodes"
              :key="String(n.uid)"
              class="clickable"
              :class="{ 'row--active': String(n.uid) === String(selectedUid) }"
              @click="$emit('select', String(n.uid))"
            >
              <td class="mono">{{ n.rank }}</td>
              <td class="username">{{ n.username }}</td>
              <td class="mono muted">{{ n.uid }}</td>
              <td class="mono">{{ fmtGat(n.gat_score) }}</td>
              <td class="mono muted">{{ fmtPr(n.pagerank) }}</td>
              <td class="mono">{{ n.total_degree ?? n.degree ?? '—' }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="note">
        节点按 PageRank 降序排名；GAT score 因 pseudo-label 退化而饱和（≈1.0），故不作为排序键。
      </div>
    </div>
  </section>
</template>

<script setup>
defineProps({
  nodes: { type: Array, default: () => [] },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
  limit: { type: Number, default: 20 },
  selectedUid: { type: [String, Number], default: null },
})

defineEmits(['change-limit', 'select'])

function fmtGat(v) {
  if (v == null) return '—'
  return Number(v).toFixed(4)
}

function fmtPr(v) {
  if (v == null) return '—'
  const n = Number(v)
  if (n === 0) return '0'
  return n < 0.001 ? n.toExponential(3) : n.toFixed(6)
}
</script>

<style scoped>
.limit-toggle {
  margin-left: auto;
  display: flex;
  gap: 4px;
}

.limit-btn {
  border: 1px solid var(--border-strong);
  background: transparent;
  color: var(--text-secondary);
  font-size: 11px;
  padding: 3px 9px;
  border-radius: 999px;
  cursor: pointer;
}

.limit-btn--on {
  border-color: var(--accent);
  color: var(--accent);
  background: rgba(34, 211, 238, 0.08);
}

.table-wrap {
  max-height: 420px;
  overflow-y: auto;
}

.username {
  color: var(--text-primary);
  max-width: 140px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.muted {
  color: var(--text-muted);
}

.row--active {
  background: rgba(34, 211, 238, 0.08);
}

.note {
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
  font-size: 11px;
  color: var(--text-muted);
  line-height: 1.6;
}
</style>
