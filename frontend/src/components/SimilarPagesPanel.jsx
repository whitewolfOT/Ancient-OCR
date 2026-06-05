import { useState, useEffect } from 'react'
import { getDocumentPages, getPageImageUrl } from '../api/client'

/**
 * SimilarPagesPanel — lists other pages in the same cluster, ranked by similarity.
 *
 * Fetches on docId/clusterId change. Re-filters on currentPageId change without
 * a network round-trip (cheap render-time exclude).
 */
export default function SimilarPagesPanel({ docId, currentPageId, clusterId, onPageSelect }) {
  // clusterPages holds ALL pages in the cluster (currentPageId excluded at render time)
  const [clusterPages, setClusterPages] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!docId || !clusterId) {
      setClusterPages([])
      return
    }

    let cancelled = false
    setLoading(true)

    getDocumentPages(docId)
      .then((all) => {
        if (cancelled) return
        const similar = all
          .filter((p) => p.cluster_id === clusterId)
          .sort((a, b) => b.similarity_to_representative - a.similarity_to_representative)
        setClusterPages(similar)
      })
      .catch(() => { if (!cancelled) setClusterPages([]) })
      .finally(() => { if (!cancelled) setLoading(false) })

    return () => { cancelled = true }
  }, [docId, clusterId]) // currentPageId excluded — filter is render-time only

  const displayed = clusterPages.filter((p) => p.page_id !== currentPageId)

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="flex-none border-b border-gray-200 px-3 py-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">
          Similar Pages
        </span>
      </div>

      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Spinner />
          </div>
        ) : !clusterId ? (
          <p className="px-3 py-4 text-xs text-gray-400">Select a page</p>
        ) : displayed.length === 0 ? (
          <p className="px-3 py-4 text-xs text-gray-400">
            No similar pages in this cluster
          </p>
        ) : (
          displayed.map((page) => (
            <SimilarThumb
              key={page.page_id}
              page={page}
              onSelect={() => onPageSelect?.(page.page_id, page.cluster_id)}
            />
          ))
        )}
      </div>
    </div>
  )
}

function SimilarThumb({ page, onSelect }) {
  const [imgError, setImgError] = useState(false)
  const simPct = Math.round(page.similarity_to_representative * 100)

  return (
    <button
      onClick={onSelect}
      className="flex w-full items-center gap-2 border-b border-gray-100 px-2 py-1.5 text-left transition-colors hover:bg-gray-50"
    >
      {/* Thumbnail */}
      <div className="h-12 w-10 flex-shrink-0 overflow-hidden rounded bg-gray-100">
        {imgError ? (
          <div className="flex h-full items-center justify-center text-gray-300">
            <PlaceholderIcon />
          </div>
        ) : (
          <img
            src={getPageImageUrl(page.page_id)}
            alt={`Page ${page.page_num + 1}`}
            className="h-full w-full object-cover"
            onError={() => setImgError(true)}
          />
        )}
      </div>

      {/* Text */}
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-gray-700">Page {page.page_num + 1}</p>
        <p className="text-[10px] text-gray-400">{simPct}% similar</p>
      </div>
    </button>
  )
}

function Spinner() {
  return (
    <svg className="h-4 w-4 animate-spin text-gray-400" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}

function PlaceholderIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5M21 3.75H3A2.25 2.25 0 001.5 6v12A2.25 2.25 0 003.75 21h16.5A2.25 2.25 0 0022.5 18V6A2.25 2.25 0 0020.25 3.75H21z" />
    </svg>
  )
}
