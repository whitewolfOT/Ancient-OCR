import { useState, useEffect, useRef } from 'react'
import { getPageImageUrl } from '../api/client'

// Decision → fill RGBA and stroke hex
const DECISION_COLORS = {
  accept:           { fill: 'rgba(34,197,94,0.25)',   stroke: '#22c55e' },
  accept_with_note: { fill: 'rgba(251,191,36,0.25)',  stroke: '#fbbf24' },
  uncertain:        { fill: 'rgba(249,115,22,0.25)',  stroke: '#f97316' },
  review_required:  { fill: 'rgba(239,68,68,0.25)',   stroke: '#ef4444' },
}

/**
 * PageViewer — displays the selected page image.
 *
 * Props:
 *   pageId               — selected page ID
 *   preprocessedImageB64 — b64 JPEG from preview endpoint; null = show raw
 *   ocrTokens            — [{text, bbox:[x1,y1,x2,y2], confidence, decision, sources}] | null
 *   onRequestOCR         — () => void — called when "Run OCR" is clicked
 */
export default function PageViewer({ pageId, preprocessedImageB64, ocrTokens, onRequestOCR }) {
  const [showPreprocessed, setShowPreprocessed] = useState(true)
  const [imgLoaded, setImgLoaded] = useState(false)
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [imgNaturalSize, setImgNaturalSize] = useState(null)   // {w, h}
  const [imgDisplaySize, setImgDisplaySize] = useState(null)   // {w, h}
  const imgRef = useRef(null)

  useEffect(() => {
    setShowPreprocessed(true)
    setImgLoaded(false)
    setImgNaturalSize(null)
    setImgDisplaySize(null)
  }, [pageId])

  useEffect(() => {
    if (preprocessedImageB64) {
      setShowPreprocessed(true)
      setImgLoaded(false)
    }
  }, [preprocessedImageB64])

  function handleImgLoad(e) {
    const el = e.currentTarget
    setImgNaturalSize({ w: el.naturalWidth, h: el.naturalHeight })
    setImgDisplaySize({ w: el.clientWidth, h: el.clientHeight })
    setImgLoaded(true)
  }

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

  const hasTokens = Boolean(ocrTokens && ocrTokens.length > 0)
  const scaleX = imgNaturalSize && imgDisplaySize ? imgDisplaySize.w / imgNaturalSize.w : 1
  const scaleY = imgNaturalSize && imgDisplaySize ? imgDisplaySize.h / imgNaturalSize.h : 1

  return (
    <div className="relative flex flex-1 items-center justify-center overflow-hidden bg-gray-100">
      {/* Loading indicator */}
      {!imgLoaded && (
        <div className="absolute inset-0 flex items-center justify-center">
          <Spinner className="h-8 w-8 text-gray-400" />
        </div>
      )}

      {/* Page image */}
      <img
        key={src}
        ref={imgRef}
        src={src}
        alt="Page"
        className={[
          'max-h-full max-w-full object-contain transition-opacity duration-200',
          imgLoaded ? 'opacity-100' : 'opacity-0',
        ].join(' ')}
        onLoad={handleImgLoad}
        onError={(e) => { setImgLoaded(true) }}
      />

      {/* Heatmap SVG overlay — absolutely positioned over the image */}
      {hasTokens && showHeatmap && imgLoaded && imgDisplaySize && (
        <svg
          className="pointer-events-none absolute"
          style={{
            width: imgDisplaySize.w,
            height: imgDisplaySize.h,
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
          }}
        >
          {ocrTokens.map((token, i) => {
            const [x1, y1, x2, y2] = token.bbox
            const dx = x1 * scaleX
            const dy = y1 * scaleY
            const dw = (x2 - x1) * scaleX
            const dh = (y2 - y1) * scaleY
            const colors = DECISION_COLORS[token.decision] || DECISION_COLORS.uncertain
            return (
              <g key={i}>
                <title>{token.text} — {Math.round(token.confidence * 100)}% — {token.decision}</title>
                <rect
                  x={dx} y={dy} width={dw} height={dh}
                  fill={colors.fill}
                  stroke={colors.stroke}
                  strokeWidth="1"
                  rx="1"
                />
              </g>
            )
          })}
        </svg>
      )}

      {/* Bottom toolbar */}
      <div className="absolute bottom-3 left-1/2 flex -translate-x-1/2 items-center gap-2">
        {/* Before/after toggle */}
        {hasPreview && (
          <div className="flex overflow-hidden rounded-full border border-gray-300 bg-white text-xs font-medium shadow-sm">
            <button
              className={[
                'px-3 py-1.5 transition-colors',
                !showPreprocessed ? 'bg-gray-800 text-white' : 'text-gray-600 hover:bg-gray-50',
              ].join(' ')}
              onClick={() => { setShowPreprocessed(false); setImgLoaded(false) }}
            >
              Raw
            </button>
            <button
              className={[
                'px-3 py-1.5 transition-colors',
                showPreprocessed ? 'bg-indigo-600 text-white' : 'text-gray-600 hover:bg-gray-50',
              ].join(' ')}
              onClick={() => { setShowPreprocessed(true); setImgLoaded(false) }}
            >
              Preprocessed
            </button>
          </div>
        )}

        {/* Heatmap toggle — only when tokens are loaded */}
        {hasTokens && (
          <button
            onClick={() => setShowHeatmap((v) => !v)}
            className={[
              'rounded-full border px-3 py-1.5 text-xs font-medium shadow-sm transition-colors',
              showHeatmap
                ? 'border-indigo-400 bg-indigo-600 text-white'
                : 'border-gray-300 bg-white text-gray-600 hover:bg-gray-50',
            ].join(' ')}
          >
            {showHeatmap ? 'Hide heatmap' : 'Show heatmap'}
          </button>
        )}

        {/* Run OCR button */}
        {!hasTokens && (
          <button
            onClick={onRequestOCR}
            disabled={!onRequestOCR}
            className="rounded-full border border-indigo-300 bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white shadow-sm transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Run OCR
          </button>
        )}
      </div>

      {/* Active view label (top-right) */}
      {hasPreview && (
        <span className={[
          'absolute right-2 top-2 rounded-full px-2 py-0.5 text-[10px] font-semibold',
          showPreprocessed ? 'bg-indigo-100 text-indigo-700' : 'bg-gray-200 text-gray-600',
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
