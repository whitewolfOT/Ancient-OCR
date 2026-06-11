import { useState, useEffect } from 'react'

function confColor(c) {
  if (c >= 0.9) return 'bg-green-50 text-green-800 border-green-200'
  if (c >= 0.7) return 'bg-yellow-50 text-yellow-800 border-yellow-200'
  return 'bg-red-50 text-red-800 border-red-200'
}

function confBar(c) {
  const pct = Math.round(c * 100)
  const color = c >= 0.9 ? 'bg-green-500' : c >= 0.7 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div className="mt-1 h-1.5 w-full rounded-full bg-gray-200">
      <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${pct}%` }} />
    </div>
  )
}

function loadMarks() {
  try { return JSON.parse(localStorage.getItem('ocr-review-marks') || '{}') } catch { return {} }
}

export default function ReviewTab({ onBack }) {
  const [pages, setPages] = useState(null)
  const [error, setError] = useState(null)
  const [selectedPage, setSelectedPage] = useState(null)
  const [selectedToken, setSelectedToken] = useState(null) // { token, idx }
  const [marks, setMarks] = useState(loadMarks)

  useEffect(() => {
    fetch('/results.json')
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then((d) => {
        const list = Array.isArray(d) ? d : (d.pages || [])
        setPages(list)
        if (list.length) setSelectedPage(list[0])
      })
      .catch((e) => setError(e.message))
  }, [])

  function mark(verdict) {
    if (!selectedToken || !selectedPage) return
    const key = `${selectedPage.filename}:${selectedToken.idx}`
    const updated = { ...marks, [key]: verdict }
    setMarks(updated)
    localStorage.setItem('ocr-review-marks', JSON.stringify(updated))
  }

  function tokenMark(filename, idx) {
    return marks[`${filename}:${idx}`] ?? null
  }

  function markedCount(filename, tokens) {
    return tokens.filter((_, i) => marks[`${filename}:${i}`]).length
  }

  // ── Loading / error states ────────────────────────────────────────────────
  if (error) return (
    <div className="flex h-full items-center justify-center text-red-600">
      Failed to load results.json: {error}
    </div>
  )
  if (!pages) return (
    <div className="flex h-full items-center justify-center text-gray-400">
      Loading results…
    </div>
  )

  const tokens = selectedPage?.tokens ?? []
  const currentMark = selectedToken ? tokenMark(selectedPage.filename, selectedToken.idx) : null

  return (
    <div className="flex h-full overflow-hidden">

      {/* ── Left: page list ─────────────────────────────────────────────── */}
      <aside className="flex w-52 flex-shrink-0 flex-col overflow-hidden border-r border-gray-200 bg-white">
        <div className="flex items-center justify-between border-b border-gray-200 px-3 py-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">Pages</span>
          <button
            onClick={onBack}
            className="text-xs text-indigo-600 hover:underline"
          >
            ← Back
          </button>
        </div>
        <ul className="flex-1 overflow-y-auto">
          {pages.map((page) => {
            const reviewed = markedCount(page.filename, page.tokens)
            const isSelected = selectedPage?.filename === page.filename
            return (
              <li key={page.filename}>
                <button
                  onClick={() => { setSelectedPage(page); setSelectedToken(null) }}
                  className={[
                    'w-full px-3 py-2.5 text-left hover:bg-gray-50 border-b border-gray-100',
                    isSelected ? 'bg-indigo-50 border-l-2 border-l-indigo-500' : '',
                  ].join(' ')}
                >
                  <div className="truncate text-sm font-medium text-gray-800">{page.filename}</div>
                  <div className="mt-0.5 flex items-center gap-2 text-xs text-gray-500">
                    <span>{page.total_tokens} tokens</span>
                    <span
                      className={[
                        'rounded px-1 py-0.5 text-[10px] font-semibold',
                        page.mean_confidence >= 0.9 ? 'bg-green-100 text-green-700'
                        : page.mean_confidence >= 0.7 ? 'bg-yellow-100 text-yellow-700'
                        : 'bg-red-100 text-red-700',
                      ].join(' ')}
                    >
                      {(page.mean_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  {reviewed > 0 && (
                    <div className="mt-0.5 text-[10px] text-indigo-500">{reviewed} reviewed</div>
                  )}
                </button>
              </li>
            )
          })}
        </ul>
      </aside>

      {/* ── Center: token grid ───────────────────────────────────────────── */}
      <main className="flex flex-1 flex-col overflow-hidden">
        {selectedPage && (
          <div className="flex items-center gap-3 border-b border-gray-200 bg-white px-4 py-2">
            <span className="font-medium text-gray-800">{selectedPage.filename}</span>
            <span className="text-sm text-gray-400">{selectedPage.total_tokens} tokens</span>
            <span className="text-sm text-gray-400">
              mean conf {(selectedPage.mean_confidence * 100).toFixed(1)}%
            </span>
            <span className="ml-auto text-xs text-gray-400">
              {markedCount(selectedPage.filename, tokens)} / {tokens.length} reviewed
            </span>
          </div>
        )}

        <div className="flex flex-1 overflow-hidden">
          {/* token chips */}
          <div className="flex-1 overflow-y-auto p-4">
            <div
              className="flex flex-wrap gap-2"
              dir="rtl"
            >
              {tokens.map((tok, idx) => {
                const m = tokenMark(selectedPage.filename, idx)
                const isSelected = selectedToken?.idx === idx
                return (
                  <button
                    key={idx}
                    onClick={() => setSelectedToken(isSelected ? null : { token: tok, idx })}
                    className={[
                      'rounded border px-2 py-1 text-sm font-medium transition-all',
                      confColor(tok.confidence),
                      isSelected ? 'ring-2 ring-indigo-400 ring-offset-1' : 'hover:opacity-80',
                      m === 'correct'   ? 'ring-2 ring-green-500 ring-offset-1' : '',
                      m === 'incorrect' ? 'ring-2 ring-red-500 ring-offset-1' : '',
                    ].join(' ')}
                  >
                    {tok.text}
                  </button>
                )
              })}
            </div>
          </div>

          {/* token detail panel */}
          {selectedToken && (
            <aside className="w-56 flex-shrink-0 overflow-y-auto border-l border-gray-200 bg-white p-4">
              <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-400">Token</p>
              <p
                className="mb-3 break-words text-2xl font-bold text-gray-900"
                dir="rtl"
                style={{ fontFamily: "'Scheherazade New', 'Amiri', serif" }}
              >
                {selectedToken.token.text}
              </p>

              <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-400">
                Confidence
              </p>
              <p className="text-sm font-medium text-gray-700">
                {(selectedToken.token.confidence * 100).toFixed(1)}%
              </p>
              {confBar(selectedToken.token.confidence)}

              <p className="mb-1 mt-3 text-xs font-semibold uppercase tracking-wide text-gray-400">
                Source
              </p>
              <p className="text-sm text-gray-600">{selectedToken.token.source}</p>

              <p className="mb-1 mt-3 text-xs font-semibold uppercase tracking-wide text-gray-400">
                BBox
              </p>
              <p className="text-xs text-gray-500">
                {selectedToken.token.bbox?.join(', ') ?? '—'}
              </p>

              <div className="mt-4 flex gap-2">
                <button
                  onClick={() => mark('correct')}
                  className={[
                    'flex-1 rounded px-2 py-1.5 text-xs font-semibold transition-colors',
                    currentMark === 'correct'
                      ? 'bg-green-600 text-white'
                      : 'bg-green-50 text-green-700 hover:bg-green-100 border border-green-200',
                  ].join(' ')}
                >
                  ✓ Correct
                </button>
                <button
                  onClick={() => mark('incorrect')}
                  className={[
                    'flex-1 rounded px-2 py-1.5 text-xs font-semibold transition-colors',
                    currentMark === 'incorrect'
                      ? 'bg-red-600 text-white'
                      : 'bg-red-50 text-red-700 hover:bg-red-100 border border-red-200',
                  ].join(' ')}
                >
                  ✗ Wrong
                </button>
              </div>
              {currentMark && (
                <button
                  onClick={() => mark(null)}
                  className="mt-2 w-full rounded bg-gray-50 px-2 py-1.5 text-xs text-gray-500 hover:bg-gray-100 border border-gray-200"
                >
                  Clear mark
                </button>
              )}
            </aside>
          )}
        </div>
      </main>
    </div>
  )
}
