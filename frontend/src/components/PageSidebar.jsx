import { useState, useEffect, useMemo } from 'react'
import { getDocumentPages, getPageImageUrl } from '../api/client'

// 8-color deterministic palette — same cluster_id always maps to same entry
const PALETTE = [
  { border: '#3b82f6', bg: '#eff6ff', text: '#1d4ed8' }, // blue
  { border: '#10b981', bg: '#ecfdf5', text: '#065f46' }, // emerald
  { border: '#f59e0b', bg: '#fffbeb', text: '#92400e' }, // amber
  { border: '#f43f5e', bg: '#fff1f2', text: '#be123c' }, // rose
  { border: '#8b5cf6', bg: '#f5f3ff', text: '#5b21b6' }, // violet
  { border: '#06b6d4', bg: '#ecfeff', text: '#155e75' }, // cyan
  { border: '#f97316', bg: '#fff7ed', text: '#9a3412' }, // orange
  { border: '#ec4899', bg: '#fdf2f8', text: '#9d174d' }, // pink
]

function clusterColor(clusterId) {
  if (!clusterId) return PALETTE[0]
  const idx =
    Array.from(clusterId).reduce((s, c) => s + c.charCodeAt(0), 0) % PALETTE.length
  return PALETTE[idx]
}

/**
 * PageSidebar — scrollable list of page thumbnails.
 *
 * Calls onPageSelect(pageId, clusterId). When no page is selected on
 * initial load the first page is auto-selected.
 */
export default function PageSidebar({ docId, selectedPageId, onPageSelect }) {
  const [pages, setPages] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filterClusterId, setFilterClusterId] = useState(null)

  async function load() {
    setLoading(true)
    setError(null)
    try {
      const data = await getDocumentPages(docId)
      setPages(data)
      // Auto-select first page if nothing is selected yet
      if (!selectedPageId && data.length > 0) {
        onPageSelect(data[0].page_id, data[0].cluster_id)
      }
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || 'Failed to load pages')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [docId]) // eslint-disable-line react-hooks/exhaustive-deps

  // Derive stable short labels for each cluster based on order of appearance
  const clusterLabels = useMemo(() => {
    const map = {}
    let n = 0
    pages.forEach((p) => {
      if (!(p.cluster_id in map)) map[p.cluster_id] = `C${++n}`
    })
    return map
  }, [pages])

  const displayed = filterClusterId
    ? pages.filter((p) => p.cluster_id === filterClusterId)
    : pages

  if (loading) return <LoadingSkeleton />
  if (error) return <ErrorState message={error} onRetry={load} />

  return (
    <div className="flex h-full w-[200px] flex-shrink-0 flex-col overflow-hidden border-r border-gray-200 bg-white">
      {filterClusterId && (
        <div className="flex items-center justify-between border-b border-gray-100 bg-gray-50 px-3 py-1.5">
          <span className="text-xs text-gray-500">
            {clusterLabels[filterClusterId]} only
          </span>
          <button
            className="text-xs font-medium text-indigo-600 hover:underline"
            onClick={() => setFilterClusterId(null)}
          >
            Show all
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto">
        {displayed.map((page) => (
          <PageThumb
            key={page.page_id}
            page={page}
            label={clusterLabels[page.cluster_id] ?? '?'}
            selected={page.page_id === selectedPageId}
            activeFilter={filterClusterId}
            onSelect={() => onPageSelect(page.page_id, page.cluster_id)}
            onFilterToggle={() =>
              setFilterClusterId((prev) =>
                prev === page.cluster_id ? null : page.cluster_id
              )
            }
          />
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// PageThumb
// ---------------------------------------------------------------------------

function PageThumb({ page, label, selected, activeFilter, onSelect, onFilterToggle }) {
  const color = clusterColor(page.cluster_id)
  const isRep = page.similarity_to_representative >= 1.0
  const simText = isRep
    ? 'rep'
    : `${Math.round(page.similarity_to_representative * 100)}%`
  const [imgError, setImgError] = useState(false)

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={(e) => e.key === 'Enter' && onSelect()}
      className={[
        'relative cursor-pointer border-b border-gray-100 outline-none transition-colors',
        selected
          ? 'ring-2 ring-inset ring-indigo-500 bg-indigo-50'
          : 'hover:bg-gray-50',
      ].join(' ')}
      style={{ borderLeft: `4px solid ${color.border}` }}
    >
      {/* Thumbnail image */}
      <div className="relative h-[120px] w-full overflow-hidden bg-gray-100">
        {imgError ? (
          <div className="flex h-full items-center justify-center text-gray-400">
            <ImagePlaceholderIcon />
          </div>
        ) : (
          <img
            src={getPageImageUrl(page.page_id)}
            alt={`Page ${page.page_num + 1}`}
            className="h-full w-full object-cover"
            onError={() => setImgError(true)}
          />
        )}

        {/* Cluster badge — top left, clickable to toggle filter */}
        <button
          onClick={(e) => { e.stopPropagation(); onFilterToggle() }}
          className={[
            'absolute left-1 top-1 rounded px-1.5 py-0.5 text-[10px] font-semibold',
            'leading-none transition-opacity',
            activeFilter && activeFilter !== page.cluster_id ? 'opacity-40' : 'opacity-100',
          ].join(' ')}
          style={{ background: color.bg, color: color.text }}
          title="Click to filter this cluster"
        >
          {label}
        </button>

        {/* Similarity badge — top right */}
        <span
          className={[
            'absolute right-1 top-1 rounded px-1.5 py-0.5 text-[10px] leading-none',
            isRep
              ? 'bg-indigo-600 text-white'
              : 'bg-black/50 text-white',
          ].join(' ')}
        >
          {simText}
        </span>

        {/* Status icons — bottom right */}
        {(page.has_ground_truth || page.has_settings) && (
          <div className="absolute bottom-1 right-1 flex gap-0.5">
            {page.has_ground_truth && (
              <StatusDot color="bg-green-500" title="Ground truth saved" />
            )}
            {page.has_settings && (
              <StatusDot color="bg-blue-500" title="Settings applied" />
            )}
          </div>
        )}
      </div>

      {/* Page number */}
      <p className="px-2 py-1 text-xs text-gray-500">
        Page {page.page_num + 1}
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusDot({ color, title }) {
  return (
    <span
      className={`flex h-4 w-4 items-center justify-center rounded-full text-[9px] font-bold text-white ${color}`}
      title={title}
    >
      ✓
    </span>
  )
}

function LoadingSkeleton() {
  return (
    <div className="flex h-full w-[200px] flex-shrink-0 flex-col gap-2 overflow-hidden border-r border-gray-200 bg-white p-2">
      {[1, 2, 3, 4].map((n) => (
        <div key={n} className="animate-pulse rounded-lg bg-gray-200" style={{ height: 148 }} />
      ))}
    </div>
  )
}

function ErrorState({ message, onRetry }) {
  return (
    <div className="flex h-full w-[200px] flex-shrink-0 flex-col items-center justify-center gap-3 border-r border-gray-200 bg-white p-4 text-center">
      <p className="text-xs text-red-600">{message}</p>
      <button
        onClick={onRetry}
        className="rounded bg-red-100 px-3 py-1.5 text-xs font-medium text-red-700 hover:bg-red-200"
      >
        Retry
      </button>
    </div>
  )
}

function ImagePlaceholderIcon() {
  return (
    <svg className="h-8 w-8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5M21 3.75H3A2.25 2.25 0 001.5 6v12A2.25 2.25 0 003.75 21h16.5A2.25 2.25 0 0022.5 18V6A2.25 2.25 0 0020.25 3.75H21z" />
    </svg>
  )
}
