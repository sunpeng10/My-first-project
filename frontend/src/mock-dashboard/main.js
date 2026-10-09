import * as echarts from 'echarts'
import './styles.css'
import {
  dashboardMockData,
  getNetworkNodes,
  getNetworkStats,
  getTopCoreNodes,
} from './mock-data.js'

const ui = {
  textStatus: document.querySelector('#text-status'),
  textAnalyze: document.querySelector('#text-analyze'),
  sentimentLabel: document.querySelector('#sentiment-label'),
  confidenceValue: document.querySelector('#confidence-value'),
  igHighlights: document.querySelector('#ig-highlights'),
  imageStatus: document.querySelector('#image-status'),
  imageAnalyze: document.querySelector('#image-analyze'),
  imageUpload: document.querySelector('#image-upload'),
  imageHint: document.querySelector('#image-hint'),
  originalImage: document.querySelector('#original-image'),
  saliencyImage: document.querySelector('#saliency-image'),
  saliencyArea: document.querySelector('#saliency-area'),
  saliencyMean: document.querySelector('#saliency-mean'),
  saliencyMax: document.querySelector('#saliency-max'),
  networkNodeCount: document.querySelector('#network-node-count'),
  networkEdgeCount: document.querySelector('#network-edge-count'),
  networkCoreCount: document.querySelector('#network-core-count'),
  networkNodeSelector: document.querySelector('#network-node-selector'),
  networkChart: document.querySelector('#network-chart'),
  coreNodeList: document.querySelector('#core-node-list'),
  detailState: document.querySelector('#detail-state'),
  detailName: document.querySelector('#detail-name'),
  detailId: document.querySelector('#detail-id'),
  detailScore: document.querySelector('#detail-score'),
  detailDegree: document.querySelector('#detail-degree'),
  detailPageRank: document.querySelector('#detail-pagerank'),
  detailKcore: document.querySelector('#detail-kcore'),
  detailInteractions: document.querySelector('#detail-interactions'),
}

let networkChart
let selectedNodeId
let uploadedObjectUrl

function formatPercent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`
}

function setStatus(element, text, state = 'idle') {
  element.textContent = text
  element.className = `status-pill status-pill--${state}`
}

function buildIgToken(word) {
  const token = document.createElement('button')
  const direction = word.score_raw >= 0 ? 'positive' : 'negative'
  const intensity = Math.max(0.14, Math.min(0.92, word.score))

  token.type = 'button'
  token.className = `ig-token ig-token--${direction}`
  token.style.setProperty('--token-intensity', intensity.toFixed(2))
  token.textContent = word.word
  token.setAttribute(
    'aria-label',
    `${word.word}，${direction === 'positive' ? '正' : '负'}贡献，归因分数 ${word.score.toFixed(3)}，原始分数 ${word.score_raw.toFixed(3)}，token ${word.token}`,
  )

  const tooltip = document.createElement('span')
  tooltip.className = 'ig-tooltip'
  tooltip.setAttribute('aria-hidden', 'true')
  tooltip.innerHTML = [
    `<strong>${word.word}</strong>`,
    `<span>score: ${word.score.toFixed(3)}</span>`,
    `<span>raw score: ${word.score_raw.toFixed(3)}</span>`,
    `<span>token: ${word.token}</span>`,
  ].join('')
  token.append(tooltip)

  return token
}

function renderTextResult() {
  const { text } = dashboardMockData
  ui.sentimentLabel.textContent = text.labelName
  ui.sentimentLabel.dataset.sentiment = text.label
  ui.confidenceValue.textContent = text.confidence.toFixed(4)
  ui.igHighlights.replaceChildren(...text.words.map(buildIgToken))
}

function renderImageResult() {
  const { image } = dashboardMockData
  ui.originalImage.src = uploadedObjectUrl || image.originalImage
  ui.saliencyImage.src = image.saliencyImage
  ui.saliencyArea.textContent = formatPercent(image.metrics.saliencyArea)
  ui.saliencyMean.textContent = image.metrics.mean.toFixed(2)
  ui.saliencyMax.textContent = image.metrics.max.toFixed(2)
}

function renderNetworkStats() {
  const stats = getNetworkStats(dashboardMockData.network)
  ui.networkNodeCount.textContent = stats.nodeCount
  ui.networkEdgeCount.textContent = stats.edgeCount
  ui.networkCoreCount.textContent = stats.coreNodeCount
}

function getDisplayNodes() {
  return getNetworkNodes(dashboardMockData.network)
}

function getDisplayNodeById(nodeId) {
  return getDisplayNodes().find((node) => node.id === nodeId)
}

function renderNodeDetail(node) {
  selectedNodeId = node.id
  ui.detailState.textContent = node.isCore ? '核心节点' : '互动节点'
  ui.detailName.textContent = `${node.name} · ${node.role}`
  ui.detailId.textContent = node.id
  ui.detailScore.textContent = node.gatScore.toFixed(2)
  ui.detailDegree.textContent = String(node.degree)
  ui.detailPageRank.textContent = node.pagerank.toFixed(3)
  ui.detailKcore.textContent = String(node.kcore)
  ui.detailInteractions.textContent = `${node.commentCount} / ${node.replyCount}`
  ui.networkNodeSelector.value = node.id
  renderCoreNodeList()
}

function renderNodeSelector() {
  const options = getDisplayNodes().map((node) => {
    const option = document.createElement('option')
    option.value = node.id
    option.textContent = `${node.name} · ${node.role} · GAT ${node.gatScore.toFixed(2)}`
    return option
  })
  ui.networkNodeSelector.replaceChildren(...options)
}

function selectNetworkNode(node) {
  renderNodeDetail(node)

  if (!networkChart) return
  const dataIndex = getDisplayNodes().findIndex((item) => item.id === node.id)
  if (dataIndex < 0) return

  networkChart.dispatchAction({ type: 'downplay', seriesIndex: 0 })
  networkChart.dispatchAction({ type: 'highlight', seriesIndex: 0, dataIndex })
  networkChart.dispatchAction({ type: 'focusNodeAdjacency', seriesIndex: 0, dataIndex })
}

function renderCoreNodeList() {
  const topNodes = getTopCoreNodes(dashboardMockData.network)
  const rows = topNodes.map((node, index) => {
    const row = document.createElement('button')
    row.type = 'button'
    row.className = 'core-node-row'
    row.classList.toggle('core-node-row--selected', node.id === selectedNodeId)
    row.innerHTML = `
      <span class="core-node-rank">${String(index + 1).padStart(2, '0')}</span>
      <span class="core-node-name">${node.name}</span>
      <span class="core-node-score">${node.gatScore.toFixed(2)}</span>
    `
    row.addEventListener('click', () => {
      selectNetworkNode(node)
    })
    return row
  })

  ui.coreNodeList.replaceChildren(...rows)
}

function graphNode(node) {
  const coreColor = '#5b57d9'
  const regularColor = '#2563a8'

  return {
    ...node,
    value: node.gatScore,
    symbolSize: 28 + node.gatScore * 35,
    draggable: true,
    itemStyle: {
      color: node.isCore ? coreColor : regularColor,
      borderColor: node.isCore ? '#2f2a9a' : '#d9ecff',
      borderWidth: node.isCore ? 3 : 2,
    },
    label: {
      show: node.isCore,
      formatter: '{b}',
      position: 'bottom',
      color: '#17223b',
      fontSize: 11,
      fontWeight: 700,
    },
  }
}

function graphTooltip(params) {
  if (params.dataType === 'edge') {
    return `<div class="chart-tooltip"><strong>互动关系</strong><br />${params.data.source} → ${params.data.target}<br />类型：${params.data.type === 'reply' ? '回复' : '评论'}</div>`
  }

  const node = params.data
  return [
    '<div class="chart-tooltip">',
    `<strong>${node.name}</strong>`,
    `<span>用户 ID：${node.id}</span>`,
    `<span>GAT Score：${node.gatScore.toFixed(2)}</span>`,
    `<span>Degree：${node.degree}</span>`,
    `<span>PageRank：${node.pagerank.toFixed(3)}</span>`,
    `<span>K-core：${node.kcore}</span>`,
    `<span>评论／回复：${node.commentCount}／${node.replyCount}</span>`,
    '</div>',
  ].join('')
}

function createNetworkOption() {
  return {
    animationDuration: 500,
    tooltip: {
      trigger: 'item',
      confine: true,
      backgroundColor: 'rgba(20, 31, 55, 0.96)',
      borderWidth: 0,
      textStyle: { color: '#f7f9ff', fontSize: 12 },
      formatter: graphTooltip,
    },
    series: [
      {
        type: 'graph',
        layout: 'force',
        data: getDisplayNodes().map(graphNode),
        links: dashboardMockData.network.edges.map((edge) => ({
          ...edge,
          lineStyle: {
            color: edge.type === 'reply' ? '#9b78db' : '#72a8d8',
            width: edge.type === 'reply' ? 2 : 1.4,
            opacity: 0.72,
            curveness: edge.type === 'reply' ? 0.12 : 0,
          },
        })),
        roam: true,
        draggable: true,
        edgeSymbol: ['none', 'arrow'],
        edgeSymbolSize: [0, 8],
        emphasis: {
          focus: 'adjacency',
          lineStyle: { width: 3, opacity: 1 },
        },
        force: {
          repulsion: 260,
          edgeLength: [80, 150],
          gravity: 0.12,
        },
      },
    ],
  }
}

function initNetworkChart() {
  networkChart = echarts.init(ui.networkChart, undefined, { renderer: 'canvas' })
  networkChart.setOption(createNetworkOption())
  networkChart.on('click', (params) => {
    if (params.dataType === 'node') {
      selectNetworkNode(params.data)
    }
  })
}

function withMockLoading(button, statusElement, complete) {
  button.disabled = true
  setStatus(statusElement, '正在分析', 'loading')

  window.setTimeout(() => {
    complete()
    setStatus(statusElement, '分析完成 · Mock', 'complete')
    button.disabled = false
  }, 420)
}

function bindEvents() {
  ui.textAnalyze.addEventListener('click', () => {
    withMockLoading(ui.textAnalyze, ui.textStatus, renderTextResult)
  })

  ui.imageAnalyze.addEventListener('click', () => {
    withMockLoading(ui.imageAnalyze, ui.imageStatus, renderImageResult)
  })

  ui.imageUpload.addEventListener('change', () => {
    const [file] = ui.imageUpload.files
    if (!file) return

    if (uploadedObjectUrl) URL.revokeObjectURL(uploadedObjectUrl)
    uploadedObjectUrl = URL.createObjectURL(file)
    ui.originalImage.src = uploadedObjectUrl
    ui.originalImage.alt = `本地预览：${file.name}`
    ui.imageHint.textContent = `已本地预览「${file.name}」。显著性图和指标仍为明确标记的 Mock 结果。`
    setStatus(ui.imageStatus, '待分析', 'idle')
  })

  ui.networkNodeSelector.addEventListener('change', () => {
    const node = getDisplayNodeById(ui.networkNodeSelector.value)
    if (node) selectNetworkNode(node)
  })

  window.addEventListener('resize', () => networkChart.resize())
  window.addEventListener('beforeunload', () => {
    if (uploadedObjectUrl) URL.revokeObjectURL(uploadedObjectUrl)
    networkChart.dispose()
  })
}

function initializeDashboard() {
  renderTextResult()
  renderImageResult()
  renderNetworkStats()
  renderNodeSelector()
  initNetworkChart()
  selectNetworkNode(getTopCoreNodes(dashboardMockData.network, 1)[0])
  setStatus(ui.textStatus, 'Mock 数据已就绪', 'idle')
  setStatus(ui.imageStatus, 'Mock 数据已就绪', 'idle')
  bindEvents()
}

initializeDashboard()
