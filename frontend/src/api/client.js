/**
 * All API communication goes through this module.
 * No component imports axios directly.
 * Base URL from VITE_API_BASE_URL env var, defaulting to http://localhost:8000.
 */
import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
})

/**
 * Upload a PDF or image file.
 * @param {File} file
 * @returns {Promise<{doc_id, page_count, clusters}>}
 */
export async function uploadDocument(file) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/documents/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

/**
 * Fetch all pages for a document with cluster and status metadata.
 * @param {string} docId
 * @returns {Promise<Array<{page_id, page_num, cluster_id, similarity_to_representative,
 *   status, has_ground_truth, has_settings, thumbnail_url}>>}
 */
export async function getDocumentPages(docId) {
  const { data } = await api.get(`/documents/${docId}/pages`)
  return data
}

/**
 * Request a live preprocessing preview for a page.
 * @param {string} pageId
 * @param {{clahe: number, denoise: number, deskew_threshold: number, binarization: string}} settings
 * @returns {Promise<{preview_image_b64, settings_applied, preview_id, processing_time_ms}>}
 */
export async function previewPage(pageId, settings) {
  const { data } = await api.post(`/pages/${pageId}/preview`, settings)
  return data
}

/**
 * Fetch saved ground truth for a page.
 * Returns null if none has been saved yet (404 treated as empty).
 * @param {string} pageId
 * @returns {Promise<{page_id, text, submitted_at} | null>}
 */
export async function getGroundTruth(pageId) {
  try {
    const { data } = await api.get(`/pages/${pageId}/ground-truth`)
    return data
  } catch (err) {
    if (err?.response?.status === 404) return null
    throw err
  }
}

/**
 * Save Arabic ground truth text for a page.
 * @param {string} pageId
 * @param {string} text
 * @returns {Promise<{page_id, saved_at}>}
 */
export async function submitGroundTruth(pageId, text) {
  const { data } = await api.post(`/pages/${pageId}/ground-truth`, {
    page_id: pageId,
    text,
  })
  return data
}

/**
 * Propagate preprocessing settings to every page in a cluster.
 * @param {string} docId
 * @param {string} clusterId
 * @param {{clahe: number, denoise: number, deskew_threshold: number, binarization: string}} settings
 * @returns {Promise<{pages_updated, cluster_id}>}
 */
export async function applyClusterSettings(docId, clusterId, settings) {
  const { data } = await api.post(`/documents/${docId}/apply-cluster-settings`, {
    cluster_id: clusterId,
    settings,
  })
  return data
}

/**
 * Return the URL to load a raw page image.
 * Used directly as an <img src> — no fetch needed.
 * @param {string} pageId
 * @returns {string}
 */
export function getPageImageUrl(pageId) {
  return `${BASE_URL}/pages/${pageId}/image`
}

/**
 * Run OCR on a page using its saved preprocessing settings.
 * @param {string} pageId
 * @returns {Promise<{page_id, word_count, decisions, tokens, processed_at}>}
 */
export async function runPageOCR(pageId) {
  const { data } = await api.post(`/pages/${pageId}/ocr`)
  return data
}

/**
 * Fetch a previously saved OCR result for a page.
 * Returns null if no OCR has been run yet (404 treated as empty).
 * @param {string} pageId
 * @returns {Promise<{page_id, word_count, decisions, tokens, processed_at} | null>}
 */
export async function getPageOCR(pageId) {
  try {
    const { data } = await api.get(`/pages/${pageId}/ocr`)
    return data
  } catch (err) {
    if (err?.response?.status === 404) return null
    throw err
  }
}
