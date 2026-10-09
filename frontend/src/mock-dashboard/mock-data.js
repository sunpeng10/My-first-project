export const MOCK_MODE = true

function createSvgDataUrl(markup) {
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(markup)}`
}

const originalImage = createSvgDataUrl(`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 600" role="img" aria-label="Mock scenic social-media image">
    <defs>
      <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#b6ddff" />
        <stop offset="100%" stop-color="#edf8ff" />
      </linearGradient>
      <linearGradient id="mountain" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#466e83" />
        <stop offset="100%" stop-color="#243f50" />
      </linearGradient>
    </defs>
    <rect width="960" height="600" fill="url(#sky)" />
    <circle cx="760" cy="115" r="50" fill="#ffe49d" opacity="0.9" />
    <path d="M0 355 L175 165 L326 332 L456 195 L632 365 L775 155 L960 340 V600 H0Z" fill="url(#mountain)" />
    <path d="M0 400 C180 350 280 430 450 395 C630 358 770 430 960 375 V600 H0Z" fill="#6ea7a0" />
    <path d="M0 470 C190 430 325 515 500 463 C677 410 800 500 960 450 V600 H0Z" fill="#3f7880" />
    <path d="M0 520 C220 460 355 550 550 485 C710 435 840 510 960 485 V600 H0Z" fill="#195660" />
    <g fill="#f7f1db" opacity="0.95">
      <circle cx="382" cy="414" r="12" /><circle cx="420" cy="425" r="10" /><circle cx="458" cy="407" r="11" />
      <circle cx="500" cy="435" r="12" /><circle cx="540" cy="410" r="10" /><circle cx="579" cy="426" r="12" />
    </g>
    <text x="48" y="70" font-family="Arial, sans-serif" font-size="29" font-weight="700" fill="#234351">DEMO SCENIC POST</text>
  </svg>
`)

const saliencyImage = createSvgDataUrl(`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 600" role="img" aria-label="Mock saliency heatmap">
    <defs>
      <linearGradient id="base" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#172b4d" />
        <stop offset="100%" stop-color="#385a73" />
      </linearGradient>
      <radialGradient id="hotA"><stop offset="0%" stop-color="#fff7aa"/><stop offset="28%" stop-color="#ffbd49"/><stop offset="58%" stop-color="#f34f5b" stop-opacity="0.75"/><stop offset="100%" stop-color="#f34f5b" stop-opacity="0"/></radialGradient>
      <radialGradient id="hotB"><stop offset="0%" stop-color="#e8f8ff"/><stop offset="25%" stop-color="#6cd8ff"/><stop offset="58%" stop-color="#3d67dd" stop-opacity="0.7"/><stop offset="100%" stop-color="#3d67dd" stop-opacity="0"/></radialGradient>
    </defs>
    <rect width="960" height="600" fill="url(#base)" />
    <path d="M0 355 L175 165 L326 332 L456 195 L632 365 L775 155 L960 340 V600 H0Z" fill="#294f68" opacity="0.8" />
    <path d="M0 450 C180 375 310 510 500 415 C685 342 790 500 960 425 V600 H0Z" fill="#2c586d" opacity="0.7" />
    <ellipse cx="480" cy="420" rx="252" ry="185" fill="url(#hotA)" />
    <ellipse cx="762" cy="118" rx="160" ry="145" fill="url(#hotB)" />
    <text x="48" y="70" font-family="Arial, sans-serif" font-size="29" font-weight="700" fill="#e9f5ff">DEMO SALIENCY MAP</text>
  </svg>
`)

const networkNodes = [
  {
    id: 'u-author',
    name: '热点作者',
    role: '原帖作者',
    gatScore: 0.91,
    isCore: true,
  },
  {
    id: 'u-lexi',
    name: '数据观察员',
    role: '活跃评论者',
    gatScore: 0.84,
    isCore: true,
  },
  {
    id: 'u-nova',
    name: '旅行笔记',
    role: '回复参与者',
    gatScore: 0.79,
    isCore: true,
  },
  {
    id: 'u-kite',
    name: '城市漫游者',
    role: '评论者',
    gatScore: 0.61,
    isCore: false,
  },
  {
    id: 'u-river',
    name: '影像研究所',
    role: '回复参与者',
    gatScore: 0.58,
    isCore: false,
  },
  {
    id: 'u-echo',
    name: '山野清风',
    role: '评论者',
    gatScore: 0.55,
    isCore: false,
  },
  {
    id: 'u-orbit',
    name: '周末出发',
    role: '评论者',
    gatScore: 0.47,
    isCore: false,
  },
  {
    id: 'u-sage',
    name: '微光记录',
    role: '回复参与者',
    gatScore: 0.44,
    isCore: false,
  },
  {
    id: 'u-moss',
    name: '看展的人',
    role: '评论者',
    gatScore: 0.36,
    isCore: false,
  },
  {
    id: 'u-lumen',
    name: '旅途信号',
    role: '评论者',
    gatScore: 0.32,
    isCore: false,
  },
]

const networkEdges = [
  { source: 'u-lexi', target: 'u-author', type: 'comment' },
  { source: 'u-nova', target: 'u-author', type: 'comment' },
  { source: 'u-kite', target: 'u-author', type: 'comment' },
  { source: 'u-river', target: 'u-author', type: 'comment' },
  { source: 'u-echo', target: 'u-author', type: 'comment' },
  { source: 'u-orbit', target: 'u-lexi', type: 'reply' },
  { source: 'u-sage', target: 'u-lexi', type: 'reply' },
  { source: 'u-moss', target: 'u-nova', type: 'reply' },
  { source: 'u-lumen', target: 'u-nova', type: 'reply' },
  { source: 'u-river', target: 'u-lexi', type: 'reply' },
  { source: 'u-echo', target: 'u-nova', type: 'reply' },
  { source: 'u-nova', target: 'u-lexi', type: 'reply' },
]

export const dashboardMockData = {
  source: 'mock',
  text: {
    text: '这个景区风景确实不错，就是人实在太多了。',
    label: 'positive',
    labelName: '正向',
    confidence: 0.9234,
    words: [
      { word: '这个', score: 0.08, score_raw: 0.012, token: '这个' },
      { word: '景区', score: 0.19, score_raw: 0.037, token: '景区' },
      { word: '风景', score: 0.76, score_raw: 0.156, token: '风景' },
      { word: '确实', score: 0.24, score_raw: 0.041, token: '确实' },
      { word: '不错', score: 1, score_raw: 0.208, token: '不错' },
      { word: '就是', score: 0.12, score_raw: -0.022, token: '就是' },
      { word: '人', score: 0.33, score_raw: -0.066, token: '人' },
      { word: '实在', score: 0.48, score_raw: -0.102, token: '实在' },
      { word: '太多了', score: 0.88, score_raw: -0.183, token: '太多了' },
    ],
  },
  image: {
    originalImage,
    saliencyImage,
    metrics: {
      saliencyArea: 0.384,
      mean: 0.61,
      max: 0.97,
    },
  },
  network: {
    nodes: networkNodes,
    edges: networkEdges,
  },
}

function getInteractionMetrics(network) {
  const metrics = new Map(
    network.nodes.map(({ id }) => [
      id,
      { degree: 0, commentCount: 0, replyCount: 0 },
    ]),
  )

  for (const edge of network.edges) {
    for (const nodeId of [edge.source, edge.target]) {
      const nodeMetrics = metrics.get(nodeId)
      if (!nodeMetrics) continue

      nodeMetrics.degree += 1
      if (edge.type === 'comment') nodeMetrics.commentCount += 1
      if (edge.type === 'reply') nodeMetrics.replyCount += 1
    }
  }

  return metrics
}

function getPageRanks(network, iterations = 30, damping = 0.85) {
  const nodeIds = network.nodes.map(({ id }) => id)
  const nodeCount = nodeIds.length
  if (!nodeCount) return new Map()

  const outgoingCounts = new Map(nodeIds.map((id) => [id, 0]))
  for (const edge of network.edges) {
    if (outgoingCounts.has(edge.source) && outgoingCounts.has(edge.target)) {
      outgoingCounts.set(edge.source, outgoingCounts.get(edge.source) + 1)
    }
  }

  let ranks = new Map(nodeIds.map((id) => [id, 1 / nodeCount]))
  for (let step = 0; step < iterations; step += 1) {
    const danglingRank = nodeIds.reduce(
      (total, id) => total + (outgoingCounts.get(id) ? 0 : ranks.get(id)),
      0,
    )
    const baseRank = (1 - damping) / nodeCount + (damping * danglingRank) / nodeCount
    const nextRanks = new Map(nodeIds.map((id) => [id, baseRank]))

    for (const edge of network.edges) {
      const outDegree = outgoingCounts.get(edge.source)
      if (!outDegree || !nextRanks.has(edge.target)) continue

      const contribution = (damping * ranks.get(edge.source)) / outDegree
      nextRanks.set(edge.target, nextRanks.get(edge.target) + contribution)
    }
    ranks = nextRanks
  }

  return ranks
}

function getKCoreValues(network) {
  const nodeIds = network.nodes.map(({ id }) => id)
  const adjacency = new Map(nodeIds.map((id) => [id, new Set()]))

  for (const edge of network.edges) {
    if (!adjacency.has(edge.source) || !adjacency.has(edge.target)) continue
    adjacency.get(edge.source).add(edge.target)
    adjacency.get(edge.target).add(edge.source)
  }

  const coreValues = new Map(nodeIds.map((id) => [id, 0]))
  const maxDegree = Math.max(0, ...[...adjacency.values()].map((neighbors) => neighbors.size))
  for (let k = 1; k <= maxDegree; k += 1) {
    const remaining = new Set(nodeIds)
    let removedNode
    do {
      removedNode = false
      for (const nodeId of [...remaining]) {
        const retainedNeighbors = [...adjacency.get(nodeId)].filter((id) => remaining.has(id))
        if (retainedNeighbors.length < k) {
          remaining.delete(nodeId)
          removedNode = true
        }
      }
    } while (removedNode)

    for (const nodeId of remaining) coreValues.set(nodeId, k)
  }

  return coreValues
}

export function getNetworkNodes(network) {
  const interactionMetrics = getInteractionMetrics(network)
  const pageRanks = getPageRanks(network)
  const kCoreValues = getKCoreValues(network)

  return network.nodes.map((node) => ({
    ...node,
    ...interactionMetrics.get(node.id),
    pagerank: Number(pageRanks.get(node.id).toFixed(3)),
    kcore: kCoreValues.get(node.id),
  }))
}

export function getNetworkStats(network) {
  const nodeCount = network.nodes.length
  const edgeCount = network.edges.length
  const coreNodeCount = network.nodes.filter((node) => node.isCore).length
  const nodes = getNetworkNodes(network)
  const averageDegree = nodeCount
    ? nodes.reduce((total, node) => total + node.degree, 0) / nodeCount
    : 0

  return {
    nodeCount,
    edgeCount,
    coreNodeCount,
    averageDegree: Number(averageDegree.toFixed(1)),
  }
}

export function getTopCoreNodes(network, limit = 4) {
  return getNetworkNodes(network)
    .filter((node) => node.isCore)
    .sort((left, right) => right.gatScore - left.gatScore)
    .slice(0, limit)
}
