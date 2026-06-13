import { useState, useEffect, useRef, useCallback } from 'react'

const PAGES = ['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg']
const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function lineImgUrl(pageId, idx) {
  return `${API}/api/lines/${encodeURIComponent(pageId)}/image/line_${String(idx).padStart(3,'0')}.png`
}
function pageImgUrl(pageId) {
  return `${API}/images/${pageId}`
}

// ── Progress bar ─────────────────────────────────────────────────────────────
function ProgressBar({ done, total }) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0
  return (
    <div className="flex items-center gap-2 text-xs text-gray-600">
      <div className="h-2 w-32 overflow-hidden rounded-full bg-gray-200">
        <div
          className="h-full rounded-full bg-green-500 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span>{done}/{total} done</span>
    </div>
  )
}

// ── Confidence bar ────────────────────────────────────────────────────────────
function ConfBar({ value }) {
  const pct = Math.round((value || 0) * 100)
  const color = pct >= 90 ? 'bg-green-500' : pct >= 70 ? 'bg-yellow-400' : 'bg-red-400'
  return (
    <div className="flex items-center gap-2 text-xs text-gray-500">
      <span>Confidence: {pct}%</span>
      <div className="h-1.5 w-24 overflow-hidden rounded-full bg-gray-200">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export default function LineReviewView({ onBack }) {
  const [currentPage, setCurrentPage] = useState(PAGES[0])
  const [lines, setLines]             = useState([])
  const [currentIdx, setCurrentIdx]   = useState(0)
  const [inputText, setInputText]     = useState('')
  const [loading, setLoading]         = useState(false)
  const [saving, setSaving]           = useState(false)
  const [imgScale, setImgScale]       = useState({ x: 1, y: 1 })

  const textareaRef  = useRef(null)
  const pageImgRef   = useRef(null)
  const linesRef     = useRef(lines)
  const idxRef       = useRef(currentIdx)
  const inputRef_val = useRef(inputText)
  const savingRef    = useRef(saving)

  linesRef.current     = lines
  idxRef.current       = currentIdx
  inputRef_val.current = inputText
  savingRef.current    = saving

  const currentLine = lines[currentIdx] ?? null

  // ── Load lines for page ──────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setLines([])
    fetch(`${API}/api/lines/${encodeURIComponent(currentPage)}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(data => {
        if (cancelled) return
        const ls = data.lines || []
        setLines(ls)
        const first = ls.findIndex(l => l.status === 'pending')
        setCurrentIdx(first >= 0 ? first : 0)
      })
      .catch(() => { if (!cancelled) setLines([]) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [currentPage])

  // ── Sync textarea when current line changes ──────────────────────────────
  useEffect(() => {
    if (!currentLine) return
    const text = currentLine.corrected_text ?? currentLine.ocr_text ?? ''
    setInputText(text)
    // defer focus so the value is flushed first
    requestAnimationFrame(() => {
      if (textareaRef.current) {
        textareaRef.current.focus()
        textareaRef.current.select()
      }
    })
  }, [currentIdx, lines.length]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Recompute scale when page image loads / window resizes ───────────────
  function computeScale() {
    const el = pageImgRef.current
    if (!el || !el.naturalWidth) return
    setImgScale({
      x: el.clientWidth  / el.naturalWidth,
      y: el.clientHeight / el.naturalHeight,
    })
  }
  useEffect(() => {
    window.addEventListener('resize', computeScale)
    return () => window.removeEventListener('resize', computeScale)
  }, [])

  // ── Save + advance ────────────────────────────────────────────────────────
  const saveAndAdvance = useCallback(async (text, status) => {
    const ls   = linesRef.current
    const idx  = idxRef.current
    const line = ls[idx]
    if (!line || savingRef.current) return
    setSaving(true)
    try {
      const r = await fetch(
        `${API}/api/lines/${encodeURIComponent(currentPage)}/${line.index}/correction`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ corrected_text: text, status }),
        }
      )
      if (!r.ok) return
      // Update local state
      setLines(prev => prev.map(l =>
        l.index === line.index ? { ...l, corrected_text: text, status } : l
      ))
      // Advance: prefer next pending after current
      const updated = ls.map(l =>
        l.index === line.index ? { ...l, status } : l
      )
      const next = updated.findIndex((l, i) => i > idx && l.status === 'pending')
      setCurrentIdx(next >= 0 ? next : Math.min(idx + 1, ls.length - 1))
    } finally {
      setSaving(false)
    }
  }, [currentPage])

  // ── Keyboard handler ──────────────────────────────────────────────────────
  useEffect(() => {
    function onKey(e) {
      const inTA = document.activeElement === textareaRef.current
      if (inTA) {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault()
          saveAndAdvance(inputRef_val.current, 'corrected')
        } else if (e.key === 'Tab') {
          e.preventDefault()
          saveAndAdvance(inputRef_val.current, 'skipped')
        }
      } else {
        // Navigation shortcuts when textarea not focused
        if (e.key === 'ArrowLeft')  { e.preventDefault(); setCurrentIdx(i => Math.max(0, i - 1)) }
        if (e.key === 'ArrowRight') { e.preventDefault(); setCurrentIdx(i => Math.min(linesRef.current.length - 1, i + 1)) }
        if (e.key === 'u' || e.key === 'U') {
          e.preventDefault()
          saveAndAdvance('', 'skipped')
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [saveAndAdvance])

  // ── Derived stats ─────────────────────────────────────────────────────────
  const totalCorrected = lines.filter(l => l.status === 'corrected').length
  const totalDone      = lines.filter(l => l.status !== 'pending').length

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-gray-50">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex flex-none items-center justify-between border-b border-gray-200 bg-white px-4 py-2">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="rounded px-2 py-1 text-sm text-gray-600 hover:bg-gray-100"
          >
            ← Back
          </button>

          {/* Page selector */}
          <select
            value={currentPage}
            onChange={e => setCurrentPage(e.target.value)}
            className="rounded border border-gray-300 bg-white px-2 py-1 text-sm"
          >
            {PAGES.map(p => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>

          {/* Line counter */}
          <span className="text-sm text-gray-600">
            Line {currentIdx + 1} / {lines.length}
          </span>

          <ProgressBar done={totalDone} total={lines.length} />
        </div>

        <div className="flex items-center gap-2">
          {/* Prev / Skip / Next */}
          <button
            onClick={() => setCurrentIdx(i => Math.max(0, i - 1))}
            className="rounded border border-gray-200 px-2 py-1 text-xs hover:bg-gray-100"
            title="Previous line (←)"
          >← Prev</button>
          <button
            onClick={() => saveAndAdvance(inputText, 'skipped')}
            className="rounded border border-gray-200 px-2 py-1 text-xs hover:bg-gray-100"
            title="Skip this line (Tab)"
          >Skip</button>
          <button
            onClick={() => setCurrentIdx(i => Math.min(lines.length - 1, i + 1))}
            className="rounded border border-gray-200 px-2 py-1 text-xs hover:bg-gray-100"
            title="Next line (→)"
          >Next →</button>

          {/* Export */}
          <a
            href={`${API}/api/corrections/export`}
            download="training_data.zip"
            className="rounded bg-green-600 px-3 py-1 text-xs font-medium text-white hover:bg-green-700"
          >
            ↓ Export {totalCorrected > 0 ? `${totalCorrected} pairs` : 'training data'}
          </a>
        </div>
      </div>

      {/* ── Body: left page panel + right correction panel ─────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── Left: full page with line overlays ──────────────────────── */}
        <div className="relative flex w-1/2 flex-col overflow-hidden border-r border-gray-200 bg-white">
          {loading ? (
            <div className="flex flex-1 items-center justify-center text-gray-400 text-sm">
              Loading…
            </div>
          ) : (
            <div className="relative flex-1 overflow-auto">
              <img
                ref={pageImgRef}
                src={pageImgUrl(currentPage)}
                alt={`Page ${currentPage}`}
                className="block w-full"
                onLoad={computeScale}
              />
              {/* SVG line overlays */}
              {lines.length > 0 && (
                <svg
                  className="pointer-events-none absolute inset-0"
                  style={{ width: '100%', height: '100%' }}
                >
                  {lines.map((ln, i) => {
                    const [bx, by, bw, bh] = ln.bbox || [0, 0, 0, 0]
                    const sx = imgScale.x, sy = imgScale.y
                    const isCurrent = i === currentIdx
                    return (
                      <rect
                        key={ln.index}
                        x={bx * sx} y={by * sy}
                        width={bw * sx} height={bh * sy}
                        fill={isCurrent ? 'rgba(59,130,246,0.25)' : 'transparent'}
                        stroke={isCurrent ? '#3b82f6' : '#9ca3af'}
                        strokeWidth={isCurrent ? 2 : 1}
                        style={{ cursor: 'pointer', pointerEvents: 'all' }}
                        onClick={() => setCurrentIdx(i)}
                      />
                    )
                  })}
                </svg>
              )}
            </div>
          )}
        </div>

        {/* ── Right: line crop + editor ────────────────────────────────── */}
        <div className="flex w-1/2 flex-col overflow-hidden">
          {!currentLine ? (
            <div className="flex flex-1 items-center justify-center text-gray-400 text-sm">
              {loading ? 'Loading…' : lines.length === 0 ? 'No lines found for this page.' : 'Select a line.'}
            </div>
          ) : (
            <div className="flex flex-1 flex-col gap-4 overflow-y-auto p-5">

              {/* Line crop image — 3× zoom via CSS scale */}
              <div className="overflow-x-auto rounded border border-gray-200 bg-white p-2">
                <img
                  src={lineImgUrl(currentPage, currentLine.index)}
                  alt={`Line ${currentLine.index}`}
                  className="block"
                  style={{ imageRendering: 'pixelated', transform: 'scale(3)', transformOrigin: 'left top' }}
                  onError={e => { e.target.style.display = 'none' }}
                />
              </div>

              <ConfBar value={currentLine.confidence} />

              {/* Editable OCR text */}
              <div>
                <label className="mb-1 block text-xs font-medium text-gray-500 uppercase tracking-wide">
                  Correct the text
                </label>
                <textarea
                  ref={textareaRef}
                  value={inputText}
                  onChange={e => setInputText(e.target.value)}
                  dir="rtl"
                  lang="ar"
                  rows={3}
                  className="w-full resize-none rounded border border-gray-300 p-3 text-right focus:border-blue-500 focus:outline-none"
                  style={{ fontFamily: "'Amiri', 'Scheherazade New', 'Noto Naskh Arabic', serif", fontSize: '1.4rem', lineHeight: 1.8 }}
                  placeholder="…"
                />
              </div>

              {/* Action buttons */}
              <div className="flex flex-col gap-2">
                <button
                  onClick={() => saveAndAdvance(inputText, 'corrected')}
                  disabled={saving}
                  className="flex items-center justify-center gap-2 rounded bg-blue-600 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
                >
                  {saving ? '…' : '✓ Save & Next'} <kbd className="rounded bg-blue-700 px-1 text-xs">Enter</kbd>
                </button>
                <button
                  onClick={() => saveAndAdvance(inputText, 'skipped')}
                  disabled={saving}
                  className="flex items-center justify-center gap-2 rounded border border-gray-300 py-2 text-sm text-gray-600 hover:bg-gray-50 disabled:opacity-50"
                >
                  → Skip <kbd className="rounded bg-gray-100 px-1 text-xs">Tab</kbd>
                </button>
                <button
                  onClick={() => saveAndAdvance('', 'skipped')}
                  disabled={saving}
                  className="flex items-center justify-center gap-2 rounded border border-red-200 py-2 text-sm text-red-500 hover:bg-red-50 disabled:opacity-50"
                >
                  ✗ Mark unreadable <kbd className="rounded bg-red-50 px-1 text-xs">U</kbd>
                </button>
              </div>

              {/* Status badge */}
              {currentLine.status !== 'pending' && (
                <div className={`rounded px-3 py-1 text-center text-xs font-medium ${
                  currentLine.status === 'corrected'
                    ? 'bg-green-50 text-green-700'
                    : 'bg-yellow-50 text-yellow-700'
                }`}>
                  {currentLine.status === 'corrected' ? '✓ Corrected' : '→ Skipped'}
                </div>
              )}

              {/* Keyboard legend */}
              <div className="mt-auto rounded bg-gray-50 p-3 text-xs text-gray-400 leading-6">
                <span className="font-medium text-gray-500">Keyboard: </span>
                Enter save &amp; next · Tab skip · U unreadable · ← → navigate (outside textarea)
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
