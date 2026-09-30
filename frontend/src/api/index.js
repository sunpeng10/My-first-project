import axios from 'axios'

/**
 * 统一 API 封装。
 * 后端统一响应结构（见 backend/utils.py）：
 *   成功：{ "success": true,  "data": ... }
 *   失败：{ "success": false, "error": { "code", "message" } }
 * 这里在拦截器中解包：成功直接返回内层 data，失败抛出 ApiError。
 */

const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000'

export class ApiError extends Error {
  constructor(code, message, status) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
  }
}

const http = axios.create({
  baseURL,
  timeout: 180000, // 模型首次加载 / IG 归因可能较慢
})

http.interceptors.response.use(
  (res) => {
    const body = res.data
    // 后端失败也会以非 2xx 状态返回，理论上不会走到这里；
    // 防御性检查 success 标志。
    if (body && body.success === false) {
      const err = body.error || {}
      return Promise.reject(
        new ApiError(err.code || 'UNKNOWN', err.message || '请求失败', res.status)
      )
    }
    return body && typeof body === 'object' && 'data' in body ? body.data : body
  },
  (err) => {
    const detail = err.response && err.response.data
    const code = (detail && detail.error && detail.error.code) || 'NETWORK_ERROR'
    const message =
      (detail && detail.error && detail.error.message) ||
      (err.code === 'ECONNABORTED'
        ? '请求超时'
        : err.message === 'Network Error'
          ? '无法连接 Flask 后端，请确认已运行：python -m backend.app'
          : err.message) ||
      '请求失败'
    return Promise.reject(new ApiError(code, message, err.response && err.response.status))
  }
)

/** 把后端返回的相对路径（如 /api/image/saliency/xxx.png）拼成完整 URL。 */
export function resolveUrl(path) {
  if (!path) return ''
  if (/^https?:\/\//i.test(path)) return path
  return baseURL.replace(/\/$/, '') + path
}

export const api = {
  /** GET /api/health */
  checkHealth: () => http.get('/api/health'),

  /** POST /api/text */
  analyzeText: (text, method = 'ig') => http.post('/api/text', { text, method }),

  /** POST /api/image（multipart/form-data，字段名 file） */
  analyzeImage: (file) => {
    const form = new FormData()
    form.append('file', file)
    return http.post('/api/image', form)
  },

  /** GET /api/graph */
  getGraph: () => http.get('/api/graph'),

  /** GET /api/top-nodes?limit=N */
  getTopNodes: (limit = 20) => http.get('/api/top-nodes', { params: { limit } }),

  /** GET /api/node/<uid> */
  getNode: (uid) => http.get(`/api/node/${encodeURIComponent(uid)}`),

  /** GET /api/attention?uid=xxx&limit=N */
  getAttention: (uid, limit = 1000) =>
    http.get('/api/attention', { params: { uid, limit } }),

  /** GET /api/post-image/<image_id>（微博本地原图 URL） */
  postImageUrl: (imageId) => `${baseURL}/api/post-image/${encodeURIComponent(imageId)}`,

  /** POST /api/post-image/<image_id>/saliency（对本地微博图片做显著性检测） */
  analyzePostImage: (imageId) =>
    http.post(`/api/post-image/${encodeURIComponent(imageId)}/saliency`),
}

export { http, baseURL }
