<template>
  <section class="panel">
    <div class="panel__header">
      <span class="dot"></span>
      <h3>核心节点统计</h3>
      <span class="sub">按 PageRank 排名</span>
    </div>

    <div class="panel__body charts-grid">
      <div class="chart-card">
        <div class="chart-card__title">核心节点 Top 10 — PageRank</div>
        <div ref="barEl" class="chart"></div>
      </div>
      <div class="chart-card">
        <div class="chart-card__title">GAT Score / PageRank 对比</div>
        <div ref="compareEl" class="chart"></div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  nodes: { type: Array, default: () => [] },
})

const barEl = ref(null)
const compareEl = ref(null)
let barChart = null
let compareChart = null

const AXIS_COLOR = '#9aa7bd'
const SPLIT_COLOR = '#1f2a44'
const GRID = { left: 8, right: 20, top: 10, bottom: 8, containLabel: true }

function top10() {
  return (props.nodes || []).slice(0, 10)
}

function renderBar() {
  if (!barEl.value) return
  const data = top10()
  const names = data.map((n) => n.username || String(n.uid))
  const scores = data.map((n) => Number(n.pagerank ?? 0))
  barChart = barChart || echarts.init(barEl.value)
  barChart.setOption({
    grid: { ...GRID, right: 40 },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      valueFormatter: (v) => Number(v).toExponential(3),
    },
    xAxis: {
      type: 'value',
      axisLabel: { color: AXIS_COLOR, formatter: (v) => v.toExponential(1) },
      splitLine: { lineStyle: { color: SPLIT_COLOR } },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: names,
      axisLabel: { color: '#e5e7eb', width: 120, overflow: 'truncate' },
      axisLine: { lineStyle: { color: SPLIT_COLOR } },
    },
    series: [
      {
        type: 'bar',
        data: scores,
        barMaxWidth: 14,
        itemStyle: {
          borderRadius: [0, 4, 4, 0],
          color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
            { offset: 0, color: '#3b82f6' },
            { offset: 1, color: '#22d3ee' },
          ]),
        },
      },
    ],
  })
}

function renderCompare() {
  if (!compareEl.value) return
  const data = top10()
  const xs = data.map((n) => String(n.rank ?? ''))
  const gats = data.map((n) => Number(n.gat_score ?? 0))
  const prs = data.map((n) => Number(n.pagerank ?? 0))
  compareChart = compareChart || echarts.init(compareEl.value)
  compareChart.setOption({
    grid: { ...GRID, right: 8, left: 8, bottom: 8 },
    tooltip: {
      trigger: 'axis',
      formatter(params) {
        const idx = params[0]?.dataIndex ?? 0
        const n = data[idx]
        const name = n ? n.username || String(n.uid) : ''
        const lines = params.map(
          (p) =>
            `${p.marker} ${p.seriesName}: ${
              p.seriesName === 'GAT Score'
                ? Number(p.value).toFixed(4)
                : Number(p.value).toExponential(3)
            }`
        )
        return `${name}<br/>${lines.join('<br/>')}`
      },
    },
    legend: {
      data: ['GAT Score', 'PageRank'],
      textStyle: { color: AXIS_COLOR },
      top: 0,
    },
    xAxis: {
      type: 'category',
      data: xs,
      name: 'Rank',
      axisLabel: { color: AXIS_COLOR },
      axisLine: { lineStyle: { color: SPLIT_COLOR } },
    },
    yAxis: [
      {
        type: 'value',
        name: 'GAT Score',
        max: 1,
        axisLabel: { color: AXIS_COLOR },
        splitLine: { lineStyle: { color: SPLIT_COLOR } },
        nameTextStyle: { color: AXIS_COLOR },
      },
      {
        type: 'value',
        name: 'PageRank',
        axisLabel: { color: AXIS_COLOR, formatter: (v) => v.toExponential(1) },
        splitLine: { show: false },
        nameTextStyle: { color: AXIS_COLOR },
      },
    ],
    series: [
      {
        name: 'GAT Score',
        type: 'bar',
        data: gats,
        barMaxWidth: 16,
        itemStyle: { color: '#22d3ee', borderRadius: [3, 3, 0, 0] },
      },
      {
        name: 'PageRank',
        type: 'line',
        yAxisIndex: 1,
        data: prs,
        smooth: true,
        symbol: 'circle',
        symbolSize: 7,
        itemStyle: { color: '#f43f5e' },
        lineStyle: { color: '#f43f5e', width: 2 },
      },
    ],
  })
}

function renderAll() {
  renderBar()
  renderCompare()
}

function onResize() {
  barChart && barChart.resize()
  compareChart && compareChart.resize()
}

onMounted(() => {
  renderAll()
  window.addEventListener('resize', onResize)
})

watch(
  () => props.nodes,
  () => renderAll()
)

onBeforeUnmount(() => {
  window.removeEventListener('resize', onResize)
  barChart && barChart.dispose()
  compareChart && compareChart.dispose()
  barChart = null
  compareChart = null
})
</script>

<style scoped>
.charts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
}

.chart-card__title {
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 6px;
  letter-spacing: 0.5px;
}

.chart {
  width: 100%;
  height: 260px;
}

@media (max-width: 1200px) {
  .charts-grid {
    grid-template-columns: 1fr;
  }
}
</style>
