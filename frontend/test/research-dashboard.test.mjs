import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import os from 'node:os'
import test from 'node:test'
import { fileURLToPath, pathToFileURL } from 'node:url'
import path from 'node:path'
import { build, createServer } from 'vite'

const here = path.dirname(fileURLToPath(import.meta.url))
const frontendRoot = path.resolve(here, '..')
const entryPath = path.join(frontendRoot, 'research-dashboard.html')
const dataPath = path.join(frontendRoot, 'src', 'mock-dashboard', 'mock-data.js')
const mainPath = path.join(frontendRoot, 'src', 'mock-dashboard', 'main.js')
const cssPath = path.join(frontendRoot, 'src', 'mock-dashboard', 'styles.css')
const viteConfigPath = path.join(frontendRoot, 'vite.config.js')
const packagePath = path.join(frontendRoot, 'package.json')

test('static dashboard entry and required modules exist', () => {
  for (const file of [entryPath, dataPath, mainPath, cssPath]) {
    assert.ok(existsSync(file), `Missing required dashboard resource: ${file}`)
  }
})

test('central mock data is visibly marked, complete, and graph-consistent', async () => {
  assert.ok(existsSync(dataPath), `Missing mock data module: ${dataPath}`)
  const module = await import(pathToFileURL(dataPath).href)

  assert.equal(module.MOCK_MODE, true)
  assert.equal(module.dashboardMockData.source, 'mock')
  assert.ok(module.dashboardMockData.text.words.length > 0)
  assert.ok(module.dashboardMockData.image.metrics.saliencyArea > 0)
  assert.ok(module.dashboardMockData.network.nodes.length > 0)
  assert.ok(module.dashboardMockData.network.edges.length > 0)
  assert.ok(module.getNetworkStats(module.dashboardMockData.network).coreNodeCount > 0)

  const nodes = module.getNetworkNodes(module.dashboardMockData.network)
  const totalDegree = nodes.reduce((total, node) => total + node.degree, 0)
  assert.equal(totalDegree, module.dashboardMockData.network.edges.length * 2)

  for (const node of nodes) {
    assert.equal(node.degree, node.commentCount + node.replyCount)
    assert.ok(node.pagerank > 0)
    assert.ok(node.kcore >= 0)
  }
})

test('entrypoint loads the direct DOM controller and exposes all required regions', () => {
  assert.ok(existsSync(entryPath), `Missing dashboard entrypoint: ${entryPath}`)
  const html = readFileSync(entryPath, 'utf8')

  for (const marker of [
    'text-input',
    'sentiment-result',
    'ig-highlights',
    'image-upload',
    'original-image',
    'saliency-image',
    'network-chart',
    'network-node-selector',
    'core-node-detail',
  ]) {
    assert.match(html, new RegExp(`id=["']${marker}["']`))
  }

  assert.match(html, /src="\/src\/mock-dashboard\/main\.js"/)
  assert.match(html, /for="network-node-selector"/)
  assert.match(html, /network-chart-description/)
  assert.doesNotMatch(html, /id="network-chart" role="img"/)
})

test('controller uses ECharts graph with drag, zoom, hover, keyboard node selection, and no HTTP client', () => {
  assert.ok(existsSync(mainPath), `Missing controller module: ${mainPath}`)
  const source = readFileSync(mainPath, 'utf8')

  assert.match(source, /import \* as echarts from 'echarts'/)
  assert.match(source, /type:\s*'graph'/)
  assert.match(source, /roam:\s*true/)
  assert.match(source, /draggable:\s*true/)
  assert.match(source, /networkNodeSelector/)
  assert.match(source, /selectNetworkNode/)
  assert.match(source, /getNetworkNodes/)
  assert.match(source, /token\.type = 'button'/)
  assert.doesNotMatch(source, /fetch\s*\(|axios\.|XMLHttpRequest/)
})

test('package test script runs the Node test suite', () => {
  const packageJson = JSON.parse(readFileSync(packagePath, 'utf8'))
  assert.equal(packageJson.scripts.test, 'node --test')
})

test('package dev:research script opens the standalone Dashboard', () => {
  const packageJson = JSON.parse(readFileSync(packagePath, 'utf8'))
  assert.equal(packageJson.scripts['dev:research'], 'vite --open /research-dashboard.html')
})

test('Vite development server serves the static Dashboard entrypoint', async () => {
  const server = await createServer({
    root: frontendRoot,
    configFile: false,
    logLevel: 'silent',
    server: { host: '127.0.0.1', port: 4173, strictPort: false },
  })
  try {
    await server.listen()
    const address = server.httpServer.address()
    assert.ok(address && typeof address !== 'string')
    const baseUrl = `http://127.0.0.1:${address.port}`
    const pageResponse = await fetch(`${baseUrl}/research-dashboard.html`)
    const pageHtml = await pageResponse.text()

    assert.equal(pageResponse.status, 200)
    assert.match(pageHtml, /\/src\/mock-dashboard\/main\.js/)
  } finally {
    await server.close()
  }
})

test('production Vite build emits the static Dashboard entrypoint and bundled asset', async () => {
  const config = readFileSync(viteConfigPath, 'utf8')

  assert.match(config, /rollupOptions/)
  assert.match(config, /research-dashboard\.html/)

  const outDir = await mkdtemp(path.join(os.tmpdir(), 'basnet-dashboard-build-'))
  try {
    await build({
      root: frontendRoot,
      configFile: viteConfigPath,
      logLevel: 'silent',
      build: { outDir, emptyOutDir: true },
    })

    const outputHtml = await readFile(path.join(outDir, 'research-dashboard.html'), 'utf8')
    assert.match(outputHtml, /assets\/researchDashboard-[^"']+\.js/)
  } finally {
    await rm(outDir, { recursive: true, force: true })
  }
})
