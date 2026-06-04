import { useState, useEffect } from 'react'
import { getPageImageUrl } from '../api/client'

/**
 * PageViewer — displays the selected page image.
 *
 * - No preprocessedImageB64 → raw page image from API
 * - preprocessedImageB64 set → shows preprocessed; toggle button for before/after
 * - Loading spinner while the image loads (reset on every pageId change)
 */
export default function PageViewer({ pageId, preprocessedImageB64 }) {
  // true = show preprocessed (when available); false = show raw
  const [showPreprocessed, setShowPreprocessed] = useState(true)
  const [imgLoaded, setImgLoaded] = useState(false)

  // When the selected page changes: reset view state
  useEffect(() => {
    setShowPreprocessed(true)
    setImgLoaded(false)
  }, [pageId])

  // When a new preprocessed image arrives, switch to showing it
  useEffect(() => {
    if (preprocessedImageB64) {
      setShowPreprocessed(true)
      setImgLoaded(false)
    }
  }, [preprocessedImageB64])

  if (!pageId) {
    return (
      <div className="flex flex-1 items-center justify-center bg-gray-100 text-sm text-gray-400">
        Select a page
      </div>
    )
  }

  const hasPreview = Boolean(preprocessedImageB64)
  const src =
    hasPreview && showPreprocessed
      ? `data:image/jpeg;base64,${preprocessedImageB64}`
      : getPageImageUrl(pageId)

  return (
    <div className="relative flex flex-1 items-center justify-center overflow-hidden bg-gray-100">
      {/* Loading indicator */}
      {!imgLoaded && (
        <div className="absolute inset-0 flex items-center justify-center">
          <Spinner className="h-8 w-8 text-gray-400" />
        </div>
      )}

      {/* key={src} forces img remount on source change, reliably firing onLoad */}
      <img
        key={src}
        src={src}
        alt="Page"
        className={[
          'max-h-full max-w-full object-contain transition-opacity duration-200',
          imgLoaded ? 'opacity-100' : 'opacity-0',
        ].join(' ')}
        onLoad={() => setImgLoaded(true)}
        onError={() => setImgLoaded(true)}
      />

      {/* Before/after toggle — only when preprocessed image exists */}
      {hasPreview && (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2">
          <div className="flex overflow-hidden rounded-full border border-gray-300 bg-white text-xs font-medium shadow-sm">
            <button
              className={[
                'px-3 py-1.5 transition-colors',
                !showPreprocessed
                  ? 'bg-gray-800 text-white'
                  : 'text-gray-600 hover:bg-gray-50',
              ].join(' ')}
              onClick={() => { setShowPreprocessed(false); setImgLoaded(false) }}
            >
              Raw
            </button>
            <button
              className={[
                'px-3 py-1.5 transition-colors',
                showPreprocessed
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-600 hover:bg-gray-50',
              ].join(' ')}
              onClick={() => { setShowPreprocessed(true); setImgLoaded(false) }}
            >
              Preprocessed
            </button>
          </div>
        </div>
      )}

      {/* Active view label */}
      {hasPreview && (
        <span className={[
          'absolute right-2 top-2 rounded-full px-2 py-0.5 text-[10px] font-semibold',
          showPreprocessed
            ? 'bg-indigo-100 text-indigo-700'
            : 'bg-gray-200 text-gray-600',
        ].join(' ')}>
          {showPreprocessed ? 'Preprocessed' : 'Raw'}
        </span>
      )}
    </div>
  )
}

function Spinner({ className }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}

