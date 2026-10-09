# Mock Research Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone, static mock research Dashboard under the existing Vite frontend without changing any RoBERTa, Captum, BASNet, or GAT model code.

**Architecture:** Preserve the existing Vue application and add a second Vite-served HTML entry point. A single JavaScript data module is the only mock-data source; a DOM controller renders text, image, metrics, core-node details, and an ECharts force graph from that module. No API calls are made in this phase.

**Tech Stack:** HTML, CSS, browser JavaScript, ECharts already installed in `frontend/node_modules`, Node built-in test runner, Vite development server.

---

### Task 1: Define the static-dashboard contract with a failing test

**Files:**
- Create: `D:\BASNet\frontend\test\research-dashboard.test.mjs`
- Test: `D:\BASNet\frontend\test\research-dashboard.test.mjs`

- [x] **Step 1: Write the failing resource and data-contract test**

```js
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import test from 'node:test'
import { fileURLToPath, pathToFileURL } from 'node:url'
import path from 'node:path'

const here = path.dirname(fileURLToPath(import.meta.url))
const frontendRoot = path.resolve(here, '..')
const entryPath = path.join(frontendRoot, 'research-dashboard.html')
const dataPath = path.join(frontendRoot, 'src', 'mock-dashboard', 'mock-data.js')
const mainPath = path.join(frontendRoot, 'src', 'mock-dashboard', 'main.js')
const cssPath = path.join(frontendRoot, 'src', 'mock-dashboard', 'styles.css')

test('static dashboard entry and required modules exist', () => {
  for (const file of [entryPath, dataPath, mainPath, cssPath]) {
    assert.ok(existsSync(file), `Missing required dashboard resource: ${file}`)
  }
})

test('central mock data is visibly marked and contains every analysis domain', async () => {
  assert.ok(existsSync(dataPath), `Missing mock data module: ${dataPath}`)
  const module = await import(pathToFileURL(dataPath).href)
  assert.equal(module.MOCK_MODE, true)
  assert.equal(module.dashboardMockData.source, 'mock')
  assert.ok(module.dashboardMockData.text.words.length > 0)
  assert.ok(module.dashboardMockData.image.metrics.saliencyArea > 0)
  assert.ok(module.dashboardMockData.network.nodes.length > 0)
  assert.ok(module.dashboardMockData.network.edges.length > 0)
  assert.ok(module.getNetworkStats(module.dashboardMockData.network).coreNodeCount > 0)
})

test('entrypoint loads the direct DOM controller and exposes all required regions', () => {
  assert.ok(existsSync(entryPath), `Missing dashboard entrypoint: ${entryPath}`)
  const html = readFileSync(entryPath, 'utf8')
  for (const marker of ['text-input', 'sentiment-result', 'ig-highlights', 'image-upload', 'original-image', 'saliency-image', 'network-chart', 'core-node-detail']) {
    assert.match(html, new RegExp(`id=["']${marker}["']`))
  }
  assert.match(html, /src="\/src\/mock-dashboard\/main\.js"/)
})

test('controller uses ECharts graph with drag, zoom, hover, and no HTTP client', () => {
  assert.ok(existsSync(mainPath), `Missing controller module: ${mainPath}`)
  const source = readFileSync(mainPath, 'utf8')
  assert.match(source, /import \* as echarts from 'echarts'/)
  assert.match(source, /type:\s*'graph'/)
  assert.match(source, /roam:\s*true/)
  assert.match(source, /draggable:\s*true/)
  assert.doesNotMatch(source, /fetch\s*\(|axios\.|XMLHttpRequest/)
})
```

- [x] **Step 2: Run the test and verify the expected RED result**

Run: `node --test test/research-dashboard.test.mjs` from `D:\BASNet\frontend`

Expected: `FAIL` with `Missing required dashboard resource` because the static entry and modules do not yet exist.

### Task 2: Add the isolated static mock Dashboard

**Files:**
- Create: `D:\BASNet\frontend\research-dashboard.html`
- Create: `D:\BASNet\frontend\src\mock-dashboard\mock-data.js`
- Create: `D:\BASNet\frontend\src\mock-dashboard\main.js`
- Create: `D:\BASNet\frontend\src\mock-dashboard\styles.css`
- Modify: `D:\BASNet\frontend\vite.config.js`

- [x] **Step 1: Add `research-dashboard.html` as a Vite-served, framework-free entrypoint**

The document must contain a header, visible mock-mode notice, text card, image card, interactive-network card, Nodes/Edges/Core Nodes stat cards, and a core-node detail area. It must use these stable IDs: `text-input`, `sentiment-result`, `ig-highlights`, `image-upload`, `original-image`, `saliency-image`, `network-chart`, and `core-node-detail`.

- [x] **Step 2: Add `mock-data.js` as the only demo-data source**

Export exactly `MOCK_MODE = true`, `dashboardMockData`, `getNetworkStats(network)`, and `getTopCoreNodes(network, limit)`. Store the sample text, signed IG scores, mock source/saliency SVG data URLs, saliency metrics, nodes, edges, and core-node metadata here. `dashboardMockData.source` must equal `"mock"`.

- [x] **Step 3: Add `main.js` as a direct-DOM controller**

Import the central data module and `echarts`. Render signed IG spans from `score_raw`; render uploaded-file previews only through `URL.createObjectURL`; render all analysis output from mock data; show a short loading state when either analysis button is clicked; update core-node details on ECharts `click`; dispose/recreate chart safely on resize. Do not make `fetch`, Axios, or model calls.

- [x] **Step 4: Add `styles.css` with responsive, light research-Dashboard styling**

Use a two-column `45% / 55%` desktop layout, white cards, restrained blue/purple accents, clear status colors, responsive single-column behavior at narrow widths, and distinct positive/negative IG token treatments.

- [x] **Step 5: Register the extra HTML entry in `vite.config.js`**

Import `fileURLToPath` and `URL` from `node:url`, then add `build.rollupOptions.input` entries for both `index.html` and `research-dashboard.html`. This preserves the existing Vue entry and causes `npm run build` to emit the static Dashboard page as well.

- [x] **Step 6: Run the contract test and verify GREEN**

Run: `node --test test/research-dashboard.test.mjs` from `D:\BASNet\frontend`

Expected: all seven regression tests pass.

### Task 3: Build and browser-verify the static page

**Files:**
- Verify: `D:\BASNet\frontend\research-dashboard.html`
- Verify: `D:\BASNet\frontend\src\mock-dashboard\main.js`
- Verify: `D:\BASNet\frontend\src\mock-dashboard\styles.css`

- [x] **Step 1: Run the existing Vite production build**

Run: `npm run build` from `D:\BASNet\frontend`

Expected: Vite emits a successful production build with no unresolved resource or ECharts import errors.

- [x] **Step 2: Start Vite and load the new entrypoint**

Run: `npm run dev -- --host 127.0.0.1` from `D:\BASNet\frontend`.

Open: `http://127.0.0.1:5173/research-dashboard.html`.

Expected: the page loads without backend startup because it has no HTTP calls.

- [x] **Step 3: Browser acceptance check**

Verify the header says `社交媒体热点舆情分析系统`; mock-mode disclosure is visible; text input, confidence, signed IG tooltip, upload preview, original/saliency panels, three image metrics, ECharts graph, three network stats, and core-node detail all render. Hover a graph node, drag it, zoom it, and click it; confirm the detail panel updates and the browser console has no resource errors.
