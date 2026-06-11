import { useState, useEffect, useRef, useCallback } from 'react'

const ARABIC_FONT = "'Amiri', 'Traditional Arabic', serif"
const STORAGE_KEY = 'ancient-ocr-corrections'

// ── Helpers ───────────────────────────────────────────────────────────────────

function confChipColor(c) {
  if (c >= 0.9) return 'bg-green-50 text-green-800 border-green-200'
  if (c >= 0.7) return 'bg-yellow-50 text-yellow-800 border-yellow-200'
  return 'bg-red-50 text-red-800 border-red-200'
}

function confCharColor(c) {
  // Interpolate: red(0) → yellow(0.7) → green(0.9+)
  if (c >= 0.9) return '#16a34a'   // green-600
  if (c >= 0.7) return '#ca8a04'   // yellow-600
  if (c >= 0.5) return '#ea580c'   // orange-600
  return '#dc2626'                  // red-600
}

function confBarBg(c) {
  if (c >= 0.9) return '#bbf7d0'
  if (c >= 0.7) return '#fef08a'
  return '#fecaca'
}

function loadCorrections() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}') } catch { return {} }
}

function saveCorrections(corr) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(corr))
}

function statusRingClass(status) {
  if (status === 'accepted')  return 'ring-2 ring-green-500 ring-offset-1'
  if (status === 'rejected')  return 'ring-2 ring-red-500 ring-offset-1'
  if (status === 'corrected') return 'ring-2 ring-blue-500 ring-offset-1'
  return ''
}

function pageStats(pageTokens, pageCorr) {
  let accepted = 0, rejected = 0, corrected = 0
  pageTokens.forEach((_, i) => {
    const s = pageCorr?.[i]?.status
    if (s === 'accepted')  accepted++
    else if (s === 'rejected')  rejected++
    else if (s === 'corrected') corrected++
  })
  return { accepted, rejected, corrected, reviewed: accepted + rejected + corrected }
}

// ── CharBar ───────────────────────────────────────────────────────────────────

function CharBar({ text, charConfs }) {
  const chars = [...text]  // Unicode-aware split
  return (
    <div className="flex flex-wrap gap-1" dir="rtl">
      {chars.map((ch, i) => {
        const c = charConfs?.[i] ?? null
        return (
          <div key={i} className="flex flex-col items-center">
            <span
              className="rounded px-1 py-0.5 text-base font-medium"
              style={{
                fontFamily: ARABIC_FONT,
                background: c !== null ? confBarBg(c) : '#f3f4f6',
                color: c !== null ? confCharColor(c) : '#6b7280',
                minWidth: '1.5rem',
                textAlign: 'center',
              }}
            >
              {ch}
            </span>
            {c !== null && (
              <div className="mt-0.5 h-1 w-full rounded-full bg-gray-200" style={{ minWidth: '1.5rem' }}>
                <div
                  className="h-1 rounded-full"
                  style={{ width: `${Math.round(c * 100)}%`, background: confCharColor(c) }}
                />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ── TokenDetailPanel ──────────────────────────────────────────────────────────

function TokenDetailPanel({ token, tokenIdx, pageId, correction, onCorrect, onNext, onPrev }) {
  const [selectedCandidateIdx, setSelectedCandidateIdx] = useState(0)
  const [editText, setEditText] = useState(token.text)
  const inputRef = useRef(null)

  // Reset when token changes
  useEffect(() => {
    setSelectedCandidateIdx(0)
    setEditText(token.text)
  }, [tokenIdx, token.text])

  const candidates = token.candidates?.length ? token.candidates : [{ text: token.text, confidence: token.confidence }]

  function handleCandidateSelect(idx) {
    setSelectedCandidateIdx(idx)
    setEditText(candidates[idx].text)
  }

  function handleAccept() {
    onCorrect(tokenIdx, { status: 'accepted', correctedText: editText, timestamp: new Date().toISOString() })
  }

  function handleReject() {
    onCorrect(tokenIdx, { status: 'rejected', correctedText: token.text, timestamp: new Date().toISOString() })
  }

  function handleSaveEdit() {
    if (!editText.trim()) return
    onCorrect(tokenIdx, { status: 'corrected', correctedText: editText, timestamp: new Date().toISOString() })
  }

  // Keyboard shortcuts
  useEffect(() => {
    function onKey(e) {
      // Don't intercept when typing in the edit input
      if (document.activeElement === inputRef.current) {
        if (e.key === 'Enter') { e.preventDefault(); handleSaveEdit() }
        return
      }
      if (e.key === 'a' || e.key === 'A') { e.preventDefault(); handleAccept() }
      else if (e.key === 'x' || e.key === 'X') { e.preventDefault(); handleReject() }
      else if (e.key === 'ArrowRight' || e.key === ']') { e.preventDefault(); onPrev() }
      else if (e.key === 'ArrowLeft'  || e.key === '[') { e.preventDefault(); onNext() }
      else if (e.key >= '1' && e.key <= '5') {
        const idx = parseInt(e.key) - 1
        if (idx < candidates.length) handleCandidateSelect(idx)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const currentStatus = correction?.status ?? null

  return (
    <aside className="flex w-72 flex-shrink-0 flex-col overflow-y-auto border-l border-gray-200 bg-white">
      <div className="border-b border-gray-200 px-4 py-2">
        <p className="text-[10px] font-semibold uppercase tracking-wide text-gray-400">Selected token #{tokenIdx + 1}</p>
      </div>

      <div className="flex-1 space-y-4 p-4">
        {/* Large Arabic display */}
        <div className="rounded-lg border border-gray-100 bg-gray-50 p-3 text-center">
          <p
            className="text-4xl font-bold text-gray-900 leading-loose"
            dir="rtl"
            lang="ar"
            style={{ fontFamily: ARABIC_FONT }}
          >
            {correction?.correctedText ?? token.text}
          </p>
          {currentStatus && (
            <span className={[
              'mt-1 inline-block rounded-full px-2 py-0.5 text-[10px] font-semibold',
              currentStatus === 'accepted'  ? 'bg-green-100 text-green-700' :
              currentStatus === 'rejected'  ? 'bg-red-100 text-red-700' :
                                              'bg-blue-100 text-blue-700',
            ].join(' ')}>
              {currentStatus}
            </span>
          )}
        </div>

        {/* Character confidence */}
        <div>
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-gray-400">Character confidence</p>
          {token.char_confidences?.length ? (
            <CharBar text={token.text} charConfs={token.char_confidences} />
          ) : (
            <div className="rounded border border-gray-100 bg-gray-50 px-3 py-2">
              <p className="text-center font-medium text-gray-700" dir="rtl" style={{ fontFamily: ARABIC_FONT }}>
                {token.text}
              </p>
              <div className="mt-1 h-1.5 w-full rounded-full bg-gray-200">
                <div
                  className="h-1.5 rounded-full"
                  style={{ width: `${Math.round(token.confidence * 100)}%`, background: confCharColor(token.confidence) }}
                />
              </div>
            </div>
          )}
          <p className="mt-1 text-right text-xs text-gray-400">{(token.confidence * 100).toFixed(1)}% avg</p>
        </div>

        {/* Candidates */}
        <div>
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-gray-400">Candidates</p>
          <div className="space-y-1">
            {candidates.map((c, i) => (
              <label
                key={i}
                className={[
                  'flex cursor-pointer items-center gap-2 rounded border px-3 py-2 text-sm transition-colors',
                  selectedCandidateIdx === i
                    ? 'border-indigo-300 bg-indigo-50'
                    : 'border-gray-100 bg-gray-50 hover:bg-gray-100',
                ].join(' ')}
              >
                <input
                  type="radio"
                  name={`cand-${tokenIdx}`}
                  checked={selectedCandidateIdx === i}
                  onChange={() => handleCandidateSelect(i)}
                  className="accent-indigo-600"
                />
                <span dir="rtl" lang="ar" style={{ fontFamily: ARABIC_FONT, flex: 1 }}>{c.text}</span>
                <span className={[
                  'ml-auto rounded px-1.5 py-0.5 text-[10px] font-semibold',
                  c.confidence >= 0.9 ? 'bg-green-100 text-green-700' :
                  c.confidence >= 0.7 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700',
                ].join(' ')}>
                  {(c.confidence * 100).toFixed(1)}%
                </span>
                {i < 5 && <span className="text-[10px] text-gray-300">{i + 1}</span>}
              </label>
            ))}
          </div>
        </div>

        {/* Manual edit */}
        <div>
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-gray-400">Manual edit</p>
          <input
            ref={inputRef}
            type="text"
            dir="rtl"
            lang="ar"
            value={editText}
            onChange={(e) => setEditText(e.target.value)}
            className="w-full rounded border border-gray-200 bg-white px-3 py-2 text-lg text-gray-900 focus:border-indigo-400 focus:outline-none focus:ring-1 focus:ring-indigo-300"
            style={{ fontFamily: ARABIC_FONT }}
            placeholder="Edit Arabic text…"
          />
        </div>

        {/* Action buttons */}
        <div className="flex gap-2">
          <button
            onClick={handleAccept}
            className={[
              'flex-1 rounded px-2 py-1.5 text-xs font-semibold transition-colors',
              currentStatus === 'accepted'
                ? 'bg-green-600 text-white'
                : 'border border-green-200 bg-green-50 text-green-700 hover:bg-green-100',
            ].join(' ')}
          >
            ✓ Accept <span className="opacity-50">(A)</span>
          </button>
          <button
            onClick={handleReject}
            className={[
              'flex-1 rounded px-2 py-1.5 text-xs font-semibold transition-colors',
              currentStatus === 'rejected'
                ? 'bg-red-600 text-white'
                : 'border border-red-200 bg-red-50 text-red-700 hover:bg-red-100',
            ].join(' ')}
          >
            ✗ Reject <span className="opacity-50">(X)</span>
          </button>
        </div>
        <button
          onClick={handleSaveEdit}
          className={[
            'w-full rounded px-2 py-1.5 text-xs font-semibold transition-colors',
            currentStatus === 'corrected'
              ? 'bg-blue-600 text-white'
              : 'border border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100',
          ].join(' ')}
        >
          💾 Save edit <span className="opacity-50">(Enter)</span>
        </button>

        {/* Navigation */}
        <div className="flex gap-2">
          <button onClick={onPrev} className="flex-1 rounded border border-gray-200 bg-gray-50 px-2 py-1.5 text-xs text-gray-600 hover:bg-gray-100">
            ← Prev <span className="opacity-50">([)</span>
          </button>
          <button onClick={onNext} className="flex-1 rounded border border-gray-200 bg-gray-50 px-2 py-1.5 text-xs text-gray-600 hover:bg-gray-100">
            Next → <span className="opacity-50">(])</span>
          </button>
        </div>

        {/* Meta */}
        <div className="space-y-0.5 border-t border-gray-100 pt-3 text-xs text-gray-400">
          <p>Source: {token.source}</p>
          <p>BBox: {token.bbox?.join(', ') ?? '—'}</p>
        </div>
      </div>
    </aside>
  )
}

// ── Export helper ─────────────────────────────────────────────────────────────

function exportCorrections(pages, corrections) {
  const out = { exported_at: new Date().toISOString(), pages: {} }
  pages.forEach((page) => {
    const pageCorr = corrections[page.filename] ?? {}
    const tokens = page.tokens ?? []
    const stats = pageStats(tokens, pageCorr)
    const corrList = Object.entries(pageCorr).map(([idxStr, c]) => {
      const idx = parseInt(idxStr)
      const tok = tokens[idx]
      return {
        token_index: idx,
        original_text: tok?.text ?? '',
        corrected_text: c.correctedText,
        status: c.status,
        confidence: tok?.confidence ?? 0,
        timestamp: c.timestamp,
      }
    }).sort((a, b) => a.token_index - b.token_index)

    out.pages[page.filename] = {
      corrections: corrList,
      total_tokens: page.total_tokens,
      reviewed: stats.reviewed,
      accepted: stats.accepted,
      rejected: stats.rejected,
      corrected: stats.corrected,
    }
  })

  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `ocr-corrections-${new Date().toISOString().slice(0, 10)}.json`
  a.click()
  URL.revokeObjectURL(url)
}

// ── ReviewTab ─────────────────────────────────────────────────────────────────

export default function ReviewTab({ onBack }) {
  const [pages, setPages] = useState(null)
  const [error, setError] = useState(null)
  const [selectedPageIdx, setSelectedPageIdx] = useState(0)
  const [selectedTokenIdx, setSelectedTokenIdx] = useState(null)
  const [corrections, setCorrections] = useState(loadCorrections)

  useEffect(() => {
    fetch('/results.json')
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then((d) => setPages(Array.isArray(d) ? d : (d.pages || [])))
      .catch((e) => setError(e.message))
  }, [])

  const applyCorrection = useCallback((tokenIdx, entry) => {
    if (!pages) return
    const pageId = pages[selectedPageIdx].filename
    setCorrections((prev) => {
      const updated = {
        ...prev,
        [pageId]: { ...(prev[pageId] ?? {}), [tokenIdx]: entry },
      }
      saveCorrections(updated)
      return updated
    })
  }, [pages, selectedPageIdx])

  const goNext = useCallback(() => {
    if (!pages) return
    const tokens = pages[selectedPageIdx]?.tokens ?? []
    setSelectedTokenIdx((i) => (i === null ? 0 : Math.min(i + 1, tokens.length - 1)))
  }, [pages, selectedPageIdx])

  const goPrev = useCallback(() => {
    setSelectedTokenIdx((i) => (i === null ? 0 : Math.max(i - 1, 0)))
  }, [])

  if (error) return (
    <div className="flex h-full items-center justify-center text-red-600">
      Failed to load results.json: {error}
    </div>
  )
  if (!pages) return (
    <div className="flex h-full items-center justify-center text-gray-400">Loading results…</div>
  )

  const selectedPage = pages[selectedPageIdx]
  const tokens = selectedPage?.tokens ?? []
  const pageId = selectedPage?.filename ?? ''
  const pageCorr = corrections[pageId] ?? {}
  const stats = pageStats(tokens, pageCorr)

  return (
    <div className="flex h-full overflow-hidden">

      {/* ── Left: page list ──────────────────────────────────────────────── */}
      <aside className="flex w-52 flex-shrink-0 flex-col overflow-hidden border-r border-gray-200 bg-white">
        <div className="flex items-center justify-between border-b border-gray-200 px-3 py-2">
          <span className="text-[10px] font-semibold uppercase tracking-wide text-gray-500">Pages</span>
          <button onClick={onBack} className="text-xs text-indigo-600 hover:underline">← Back</button>
        </div>
        <ul className="flex-1 overflow-y-auto">
          {pages.map((page, idx) => {
            const pc = corrections[page.filename] ?? {}
            const s = pageStats(page.tokens ?? [], pc)
            const isSelected = idx === selectedPageIdx
            return (
              <li key={page.filename}>
                <button
                  onClick={() => { setSelectedPageIdx(idx); setSelectedTokenIdx(null) }}
                  className={[
                    'w-full border-b border-gray-100 px-3 py-2.5 text-left hover:bg-gray-50',
                    isSelected ? 'border-l-2 border-l-indigo-500 bg-indigo-50' : '',
                  ].join(' ')}
                >
                  <div className="truncate text-sm font-medium text-gray-800">{page.filename}</div>
                  <div className="mt-0.5 flex items-center gap-2 text-xs text-gray-500">
                    <span>{page.total_tokens}t</span>
                    <span className={[
                      'rounded px-1 py-0.5 text-[10px] font-semibold',
                      page.mean_confidence >= 0.9 ? 'bg-green-100 text-green-700'
                      : page.mean_confidence >= 0.7 ? 'bg-yellow-100 text-yellow-700'
                      : 'bg-red-100 text-red-700',
                    ].join(' ')}>
                      {(page.mean_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  {s.reviewed > 0 && (
                    <div className="mt-0.5 flex gap-1 text-[10px]">
                      {s.accepted > 0  && <span className="text-green-600">✓{s.accepted}</span>}
                      {s.rejected > 0  && <span className="text-red-600">✗{s.rejected}</span>}
                      {s.corrected > 0 && <span className="text-blue-600">✎{s.corrected}</span>}
                    </div>
                  )}
                </button>
              </li>
            )
          })}
        </ul>
      </aside>

      {/* ── Center: token grid ───────────────────────────────────────────── */}
      <main className="flex flex-1 flex-col overflow-hidden">
        <div className="flex flex-none items-center gap-3 border-b border-gray-200 bg-white px-4 py-2">
          <span className="font-medium text-gray-800">{pageId}</span>
          <span className="text-sm text-gray-400">{tokens.length} tokens</span>
          <div className="flex gap-2 text-xs">
            {stats.accepted  > 0 && <span className="text-green-600">✓ {stats.accepted}</span>}
            {stats.rejected  > 0 && <span className="text-red-600">✗ {stats.rejected}</span>}
            {stats.corrected > 0 && <span className="text-blue-600">✎ {stats.corrected}</span>}
            <span className="text-gray-400">{stats.reviewed}/{tokens.length} reviewed</span>
          </div>
          <button
            onClick={() => exportCorrections(pages, corrections)}
            className="ml-auto rounded border border-gray-200 bg-white px-3 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50"
          >
            ↓ Export corrections
          </button>
        </div>

        <div className="flex flex-1 overflow-hidden">
          {/* token chips */}
          <div className="flex-1 overflow-y-auto p-4">
            <div className="flex flex-wrap gap-2" dir="rtl">
              {tokens.map((tok, idx) => {
                const corr = pageCorr[idx]
                const isSelected = selectedTokenIdx === idx
                return (
                  <button
                    key={idx}
                    onClick={() => setSelectedTokenIdx(isSelected ? null : idx)}
                    className={[
                      'rounded border px-2 py-1 text-sm font-medium transition-all',
                      confChipColor(tok.confidence),
                      isSelected ? 'ring-2 ring-indigo-400 ring-offset-1' : 'hover:opacity-80',
                      statusRingClass(corr?.status),
                    ].filter(Boolean).join(' ')}
                    style={{ fontFamily: ARABIC_FONT }}
                    dir="rtl"
                  >
                    {corr?.correctedText ?? tok.text}
                  </button>
                )
              })}
            </div>
          </div>

          {/* detail panel */}
          {selectedTokenIdx !== null && tokens[selectedTokenIdx] && (
            <TokenDetailPanel
              token={tokens[selectedTokenIdx]}
              tokenIdx={selectedTokenIdx}
              pageId={pageId}
              correction={pageCorr[selectedTokenIdx] ?? null}
              onCorrect={applyCorrection}
              onNext={goNext}
              onPrev={goPrev}
            />
          )}
        </div>
      </main>
    </div>
  )
}
