import { useState, useEffect, useRef } from 'react'
import {
  getGroundTruth, submitGroundTruth,
  getLineGroundTruth, submitLineGroundTruth,
} from '../api/client'

/**
 * GroundTruthPanel — RTL textarea for Arabic ground truth.
 *
 * Two views:
 *   Full  — single textarea (original behaviour)
 *   Lines — per-line editing with OCR token count hint per line
 *
 * Props:
 *   pageId      — selected page ID
 *   ocrTokens   — [{text, bbox, confidence, decision}] | null  (for line hints)
 *   onSubmitted — () => void
 */
export default function GroundTruthPanel({ pageId, ocrTokens, onSubmitted }) {
  const [view, setView] = useState('full')          // 'full' | 'lines'
  const [text, setText] = useState('')
  const [savedText, setSavedText] = useState(null)
  const [lastSavedAt, setLastSavedAt] = useState(null)
  const [status, setStatus] = useState(null)        // null | 'saving' | 'saved' | 'error'
  const [loading, setLoading] = useState(false)
  // Line view state
  const [lineTexts, setLineTexts] = useState([])    // string[]
  const [savedLines, setSavedLines] = useState(null)
  const timerRef = useRef(null)

  useEffect(() => {
    if (!pageId) {
      setText(''); setSavedText(null); setLastSavedAt(null)
      setStatus(null); setLineTexts([]); setSavedLines(null)
      return
    }

    let cancelled = false
    setLoading(true); setStatus(null)

    Promise.all([
      getGroundTruth(pageId).catch(() => null),
      getLineGroundTruth(pageId).catch(() => null),
    ]).then(([gt, lineGt]) => {
      if (cancelled) return
      if (gt) {
        setText(gt.text); setSavedText(gt.text); setLastSavedAt(gt.submitted_at)
      } else {
        setText(''); setSavedText(null); setLastSavedAt(null)
      }
      if (lineGt) {
        const lines = lineGt.lines.map((l) => l.text)
        setLineTexts(lines); setSavedLines(lines)
      } else if (gt) {
        const lines = gt.text.split('\n')
        setLineTexts(lines); setSavedLines(null)
      } else {
        setLineTexts([]); setSavedLines(null)
      }
    }).finally(() => {
      if (!cancelled) setLoading(false)
    })

    return () => {
      cancelled = true; clearTimeout(timerRef.current)
    }
  }, [pageId])

  // Sync full text → line texts when switching to Lines view
  function switchToLines() {
    if (text && lineTexts.length === 0) setLineTexts(text.split('\n').filter(Boolean))
    setView('lines')
  }

  // Sync line texts → full text when switching back
  function switchToFull() {
    if (lineTexts.length > 0) setText(lineTexts.join('\n'))
    setView('full')
  }

  async function handleSubmitFull() {
    if (!pageId || !text.trim() || text === savedText) return
    clearTimeout(timerRef.current)
    setStatus('saving')
    try {
      const result = await submitGroundTruth(pageId, text)
      setSavedText(text); setLastSavedAt(result.saved_at); setStatus('saved')
      timerRef.current = setTimeout(() => setStatus(null), 2000)
      onSubmitted?.()
    } catch {
      setStatus('error'); timerRef.current = setTimeout(() => setStatus(null), 2500)
    }
  }

  async function handleSubmitLines() {
    const filtered = lineTexts.filter((l) => l.trim())
    if (!pageId || filtered.length === 0) return
    clearTimeout(timerRef.current)
    setStatus('saving')
    try {
      await submitLineGroundTruth(pageId, filtered)
      await submitGroundTruth(pageId, filtered.join('\n'))
      setSavedLines(filtered); setSavedText(filtered.join('\n'))
      setStatus('saved'); timerRef.current = setTimeout(() => setStatus(null), 2000)
      onSubmitted?.()
    } catch {
      setStatus('error'); timerRef.current = setTimeout(() => setStatus(null), 2500)
    }
  }

  function updateLine(idx, val) {
    setLineTexts((prev) => prev.map((t, i) => (i === idx ? val : t)))
  }

  function addLine() { setLineTexts((prev) => [...prev, '']) }

  function removeLine(idx) {
    setLineTexts((prev) => prev.filter((_, i) => i !== idx))
  }

  const isUnchangedFull = text === savedText
  const submitDisabled = !pageId || loading || status === 'saving'

  return (
    <div className="flex flex-col border-b border-gray-200 bg-white p-3">
      {/* Header */}
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">
          Ground Truth
        </span>
        <div className="flex items-center gap-1">
          {lastSavedAt && (
            <span className="mr-1 text-[10px] text-gray-400" title={lastSavedAt}>
              {_relativeTime(lastSavedAt)}
            </span>
          )}
          {/* View toggle */}
          {pageId && (
            <div className="flex overflow-hidden rounded border border-gray-200 text-[10px] font-medium">
              <button
                onClick={switchToFull}
                className={[
                  'px-2 py-0.5 transition-colors',
                  view === 'full' ? 'bg-indigo-600 text-white' : 'text-gray-500 hover:bg-gray-50',
                ].join(' ')}
              >
                Full
              </button>
              <button
                onClick={switchToLines}
                className={[
                  'px-2 py-0.5 transition-colors',
                  view === 'lines' ? 'bg-indigo-600 text-white' : 'text-gray-500 hover:bg-gray-50',
                ].join(' ')}
              >
                Lines
              </button>
            </div>
          )}
        </div>
      </div>

      {!pageId ? (
        <p className="text-xs text-gray-400">Select a page</p>
      ) : view === 'full' ? (
        <FullView
          text={text} loading={loading}
          onChange={setText}
          onSubmit={handleSubmitFull}
          disabled={submitDisabled || isUnchangedFull || !text.trim()}
          status={status}
        />
      ) : (
        <LinesView
          lines={lineTexts} loading={loading}
          onChange={updateLine} onAdd={addLine} onRemove={removeLine}
          onSubmit={handleSubmitLines}
          disabled={submitDisabled || lineTexts.filter((l) => l.trim()).length === 0}
          status={status}
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sub-views
// ---------------------------------------------------------------------------

function FullView({ text, loading, onChange, onSubmit, disabled, status }) {
  return (
    <>
      <div className="relative">
        <textarea
          dir="rtl" lang="ar" rows={8}
          value={text}
          onChange={(e) => onChange(e.target.value)}
          disabled={loading}
          placeholder="اكتب النص الصحيح هنا..."
          className={[
            'w-full resize-none rounded border border-gray-300 p-2 text-right',
            'font-arabic text-base leading-relaxed text-gray-900',
            'placeholder:text-gray-400 focus:outline-none focus:ring-1 focus:ring-indigo-400',
            loading ? 'opacity-50' : '',
          ].join(' ')}
          style={{ fontSize: '1.0625rem', fontFamily: '"Scheherazade New", "Amiri", serif' }}
        />
        <span className="absolute bottom-1.5 left-2 select-none text-[10px] text-gray-300">
          {text.length}
        </span>
      </div>
      <SubmitButton disabled={disabled} status={status} onClick={onSubmit} />
    </>
  )
}

function LinesView({ lines, loading, onChange, onAdd, onRemove, onSubmit, disabled, status }) {
  return (
    <>
      <div className="flex max-h-60 flex-col gap-1.5 overflow-y-auto pr-0.5">
        {lines.length === 0 && (
          <p className="text-[11px] text-gray-400">No lines yet — add one below</p>
        )}
        {lines.map((line, idx) => (
          <div key={idx} className="flex items-center gap-1">
            <span className="w-5 shrink-0 text-right text-[10px] text-gray-300 select-none">
              {idx + 1}
            </span>
            <input
              dir="rtl" lang="ar" type="text"
              value={line}
              onChange={(e) => onChange(idx, e.target.value)}
              disabled={loading}
              className="flex-1 rounded border border-gray-200 px-2 py-1 text-right text-sm
                         text-gray-900 focus:outline-none focus:ring-1 focus:ring-indigo-300"
              style={{ fontFamily: '"Scheherazade New", "Amiri", serif' }}
            />
            <button
              onClick={() => onRemove(idx)}
              className="shrink-0 text-gray-300 hover:text-red-400 text-xs px-0.5"
              title="Remove line"
            >
              ×
            </button>
          </div>
        ))}
      </div>
      <button
        onClick={onAdd}
        className="mt-1.5 w-full rounded border border-dashed border-gray-300 py-1
                   text-[11px] text-gray-400 hover:border-indigo-300 hover:text-indigo-500"
      >
        + line
      </button>
      <SubmitButton disabled={disabled} status={status} onClick={onSubmit} />
    </>
  )
}

function SubmitButton({ disabled, status, onClick }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={[
        'mt-2 w-full rounded-lg py-1.5 text-xs font-medium transition-colors',
        status === 'saved'  ? 'bg-green-100 text-green-700'
          : status === 'error' ? 'bg-red-100 text-red-700'
          : disabled         ? 'cursor-not-allowed bg-gray-100 text-gray-400'
          :                    'bg-indigo-600 text-white hover:bg-indigo-700',
      ].join(' ')}
    >
      {status === 'saving' ? 'Saving…'
        : status === 'saved' ? 'Saved ✓'
        : status === 'error' ? 'Save failed'
        : 'Save ground truth'}
    </button>
  )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _relativeTime(isoString) {
  try {
    const diff = Date.now() - new Date(isoString).getTime()
    const mins = Math.floor(diff / 60_000)
    if (mins < 1) return 'just now'
    if (mins < 60) return `${mins}m ago`
    const hrs = Math.floor(mins / 60)
    if (hrs < 24) return `${hrs}h ago`
    return `${Math.floor(hrs / 24)}d ago`
  } catch { return '' }
}
