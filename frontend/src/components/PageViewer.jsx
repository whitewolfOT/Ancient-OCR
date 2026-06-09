import { useState, useEffect, useRef } from 'react'
import { getPageImageUrl, getGroundTruth } from '../api/client'

const DECISION_COLORS = {
  accept:           { fill: 'rgba(34,197,94,0.25)',   stroke: '#22c55e' },
  accept_with_note: { fill: 'rgba(251,191,36,0.25)',  stroke: '#fbbf24' },
  uncertain:        { fill: 'rgba(249,115,22,0.25)',  stroke: '#f97316' },
  review_required:  { fill: 'rgba(239,68,68,0.25)',   stroke: '#ef4444' },
}

/**
 * PageViewer — displays page image with optional OCR heatmap, plus a Compare tab.
 *
 * Props:
 *   pageId               — selected page ID
 *   preprocessedImageB64 — b64 JPEG from preview endpoint; null = show raw
 *   ocrTokens            — [{text, bbox, confidence, decision, sources}] | null
 *   onRequestOCR         — () => void
 */
export default function PageViewer({ pageId, preprocessedImageB64, ocrTokens, onRequestOCR }) {
  const [activeTab, setActiveTab] = useState('image')   // 'image' | 'compare'
  const [showPreprocessed, setShowPreprocessed] = useState(true)
  const [imgLoaded, setImgLoaded] = useState(false)
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [imgNaturalSize, setImgNaturalSize] = useState(null)
  const [imgDisplaySize, setImgDisplaySize] = useState(null)
  const imgRef = useRef(null)

  useEffect(() => {
    setShowPreprocessed(true); setImgLoaded(false)
    setImgNaturalSize(null); setImgDisplaySize(null)
    setActiveTab('image')
  }, [pageId])

  useEffect(() => {
    if (preprocessedImageB64) { setShowPreprocessed(true); setImgLoaded(false) }
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
    <div className="relative flex flex-1 flex-col overflow-hidden bg-gray-100">
      {/* Tab bar */}
      <div className="flex shrink-0 border-b border-gray-200 bg-white">
        {['image', 'compare'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            disabled={tab === 'compare' && !hasTokens}
            className={[
              'px-4 py-2 text-xs font-medium capitalize transition-colors',
              activeTab === tab
                ? 'border-b-2 border-indigo-600 text-indigo-700'
                : 'text-gray-500 hover:text-gray-700 disabled:cursor-not-allowed disabled:text-gray-300',
            ].join(' ')}
          >
            {tab === 'compare' ? 'Compare' : 'Image'}
          </button>
        ))}
      </div>

      {/* ── Image tab ───────────────────────────────────────────── */}
      {activeTab === 'image' && (
        <div className="relative flex flex-1 items-center justify-center overflow-hidden">
          {!imgLoaded && (
            <div className="absolute inset-0 flex items-center justify-center">
              <Spinner className="h-8 w-8 text-gray-400" />
            </div>
          )}

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
            onError={() => setImgLoaded(true)}
          />

          {hasTokens && showHeatmap && imgLoaded && imgDisplaySize && (
            <svg
              className="pointer-events-none absolute"
              style={{
                width: imgDisplaySize.w, height: imgDisplaySize.h,
                top: '50%', left: '50%',
                transform: 'translate(-50%, -50%)',
              }}
            >
              {ocrTokens.map((token, i) => {
                const [x1, y1, x2, y2] = token.bbox
                const colors = DECISION_COLORS[token.decision] || DECISION_COLORS.uncertain
                return (
                  <g key={i}>
                    <title>{token.text} — {Math.round(token.confidence * 100)}% — {token.decision}</title>
                    <rect
                      x={x1 * scaleX} y={y1 * scaleY}
                      width={(x2 - x1) * scaleX} height={(y2 - y1) * scaleY}
                      fill={colors.fill} stroke={colors.stroke} strokeWidth="1" rx="1"
                    />
                  </g>
                )
              })}
            </svg>
          )}

          {/* Bottom toolbar */}
          <div className="absolute bottom-3 left-1/2 flex -translate-x-1/2 items-center gap-2">
            {hasPreview && (
              <div className="flex overflow-hidden rounded-full border border-gray-300 bg-white text-xs font-medium shadow-sm">
                <button
                  className={['px-3 py-1.5 transition-colors',
                    !showPreprocessed ? 'bg-gray-800 text-white' : 'text-gray-600 hover:bg-gray-50'].join(' ')}
                  onClick={() => { setShowPreprocessed(false); setImgLoaded(false) }}
                >Raw</button>
                <button
                  className={['px-3 py-1.5 transition-colors',
                    showPreprocessed ? 'bg-indigo-600 text-white' : 'text-gray-600 hover:bg-gray-50'].join(' ')}
                  onClick={() => { setShowPreprocessed(true); setImgLoaded(false) }}
                >Preprocessed</button>
              </div>
            )}
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
            {!hasTokens && (
              <button
                onClick={onRequestOCR}
                disabled={!onRequestOCR}
                className="rounded-full border border-indigo-300 bg-indigo-600 px-3 py-1.5
                           text-xs font-medium text-white shadow-sm transition-colors
                           hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Run OCR
              </button>
            )}
          </div>

          {hasPreview && (
            <span className={[
              'absolute right-2 top-2 rounded-full px-2 py-0.5 text-[10px] font-semibold',
              showPreprocessed ? 'bg-indigo-100 text-indigo-700' : 'bg-gray-200 text-gray-600',
            ].join(' ')}>
              {showPreprocessed ? 'Preprocessed' : 'Raw'}
            </span>
          )}
        </div>
      )}

      {/* ── Compare tab ─────────────────────────────────────────── */}
      {activeTab === 'compare' && (
        <ComparePane pageId={pageId} ocrTokens={ocrTokens} />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// ComparePane — OCR text vs saved ground truth with CER/WER
// ---------------------------------------------------------------------------

function ComparePane({ pageId, ocrTokens }) {
  const [gt, setGt] = useState(null)         // { text } | null
  const [gtLoading, setGtLoading] = useState(false)

  useEffect(() => {
    if (!pageId) { setGt(null); return }
    setGtLoading(true)
    getGroundTruth(pageId)
      .then(setGt)
      .catch(() => setGt(null))
      .finally(() => setGtLoading(false))
  }, [pageId])

  const ocrText = ocrTokens ? ocrTokens.map((t) => t.text).join(' ') : ''
  const gtText  = gt?.text ?? ''

  const cer = gtText ? _cer(gtText, ocrText) : null
  const wer = gtText ? _wer(gtText, ocrText) : null

  return (
    <div className="flex flex-1 flex-col overflow-hidden p-3 gap-3">
      {/* Metrics row */}
      {cer !== null && (
        <div className="flex gap-4 text-xs">
          <Metric label="CER" value={`${(cer * 100).toFixed(1)}%`}
            color={cer < 0.1 ? 'text-green-600' : cer < 0.3 ? 'text-amber-600' : 'text-red-600'} />
          <Metric label="WER" value={`${(wer * 100).toFixed(1)}%`}
            color={wer < 0.1 ? 'text-green-600' : wer < 0.3 ? 'text-amber-600' : 'text-red-600'} />
          <span className="text-gray-400">{ocrTokens?.length ?? 0} tokens</span>
        </div>
      )}

      {/* Side-by-side panels */}
      <div className="flex flex-1 gap-3 overflow-hidden min-h-0">
        {/* OCR output */}
        <TextPane label="OCR Output" text={ocrText} rtl dir="rtl" />

        {/* Ground truth */}
        <TextPane
          label="Ground Truth"
          text={gtLoading ? '…' : (gtText || 'No ground truth saved')}
          dim={!gtText}
          dir="rtl"
        />
      </div>

      {/* Diff tokens if both present */}
      {ocrText && gtText && (
        <DiffView ocr={ocrText} gt={gtText} />
      )}
    </div>
  )
}

function Metric({ label, value, color }) {
  return (
    <div className="flex items-baseline gap-1">
      <span className="text-gray-500 uppercase tracking-wide text-[10px]">{label}</span>
      <span className={`font-semibold ${color}`}>{value}</span>
    </div>
  )
}

function TextPane({ label, text, dim, dir }) {
  return (
    <div className="flex flex-1 flex-col overflow-hidden rounded border border-gray-200 bg-white">
      <div className="shrink-0 border-b border-gray-100 px-2 py-1 text-[10px] font-semibold
                      uppercase tracking-wide text-gray-400">
        {label}
      </div>
      <div
        dir={dir}
        className={[
          'flex-1 overflow-y-auto p-2 text-sm leading-relaxed',
          dim ? 'text-gray-400 italic' : 'text-gray-800',
        ].join(' ')}
        style={{ fontFamily: '"Scheherazade New", "Amiri", serif', fontSize: '1rem' }}
      >
        {text}
      </div>
    </div>
  )
}

function DiffView({ ocr, gt }) {
  const ocrWords = ocr.split(/\s+/).filter(Boolean)
  const gtWords  = gt.split(/\s+/).filter(Boolean)

  // Simple word-level diff: mark words missing in OCR vs extra
  const gtSet  = new Set(gtWords)
  const ocrSet = new Set(ocrWords)
  const missing = gtWords.filter((w) => !ocrSet.has(w))
  const extra   = ocrWords.filter((w) => !gtSet.has(w))

  if (missing.length === 0 && extra.length === 0) return null

  return (
    <div className="shrink-0 rounded border border-amber-100 bg-amber-50 p-2 text-[11px]">
      {missing.length > 0 && (
        <p className="text-red-600">
          <span className="font-semibold">Missing in OCR:</span>{' '}
          <span dir="rtl">{missing.slice(0, 10).join(' ')}{missing.length > 10 ? ' …' : ''}</span>
        </p>
      )}
      {extra.length > 0 && (
        <p className="text-amber-700">
          <span className="font-semibold">Extra in OCR:</span>{' '}
          <span dir="rtl">{extra.slice(0, 10).join(' ')}{extra.length > 10 ? ' …' : ''}</span>
        </p>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Client-side CER / WER (mirrors training/feedback_store.py logic)
// ---------------------------------------------------------------------------

function _editDist(a, b) {
  const n = a.length, m = b.length
  let dp = Array.from({ length: m + 1 }, (_, i) => i)
  for (let i = 1; i <= n; i++) {
    const prev = dp.slice()
    dp[0] = i
    for (let j = 1; j <= m; j++) {
      dp[j] = a[i - 1] === b[j - 1]
        ? prev[j - 1]
        : 1 + Math.min(prev[j - 1], prev[j], dp[j - 1])
    }
  }
  return dp[m]
}

function _cer(ref, hyp) {
  if (!ref.length) return hyp.length ? 1 : 0
  return Math.min(1, _editDist([...ref], [...hyp]) / ref.length)
}

function _wer(ref, hyp) {
  const r = ref.split(/\s+/).filter(Boolean)
  const h = hyp.split(/\s+/).filter(Boolean)
  if (!r.length) return h.length ? 1 : 0
  return Math.min(1, _editDist(r, h) / r.length)
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
