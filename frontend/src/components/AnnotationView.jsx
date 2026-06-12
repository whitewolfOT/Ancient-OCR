import { useState, useEffect, useRef, useCallback } from 'react'

const ARABIC_FONT = "'Amiri', 'Traditional Arabic', serif"
const PAGES = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg',
               '6.jpg', '7.jpg', '8.jpg', '9.jpg']
const ZOOM = 4
const INK_COLOR = '#ef4444'
const INK_WIDTH = 2

// ── Drawing canvas (single canvas: bg image + strokes) ────────────────────────

function DrawCanvas({ imgEl, token, canvasRef }) {
  const isDrawing = useRef(false)
  const lastPos = useRef(null)

  // Draw background image into canvas when token or image changes
  const drawBg = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !imgEl || !imgEl.complete || !token?.bbox) return
    const [x, y, w, h] = token.bbox
    if (w <= 0 || h <= 0) return
    const cw = Math.max(w * ZOOM, 80)
    const ch = Math.max(h * ZOOM, 40)
    canvas.width = cw
    canvas.height = ch
    canvas.getContext('2d').drawImage(imgEl, x, y, w, h, 0, 0, cw, ch)
  }, [imgEl, token, canvasRef])

  useEffect(() => { drawBg() }, [drawBg])

  function pos(e) {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const sx = canvas.width / rect.width
    const sy = canvas.height / rect.height
    return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy }
  }

  function onDown(e) {
    isDrawing.current = true
    lastPos.current = pos(e)
  }

  function onMove(e) {
    if (!isDrawing.current) return
    const p = pos(e)
    const ctx = canvasRef.current.getContext('2d')
    ctx.strokeStyle = INK_COLOR
    ctx.lineWidth = INK_WIDTH
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath()
    ctx.moveTo(lastPos.current.x, lastPos.current.y)
    ctx.lineTo(p.x, p.y)
    ctx.stroke()
    lastPos.current = p
  }

  function onUp() {
    isDrawing.current = false
    lastPos.current = null
  }

  return { onDown, onMove, onUp, drawBg }
}

// ── AnnotationView ────────────────────────────────────────────────────────────

export default function AnnotationView({ onBack }) {
  const [pages, setPages]               = useState(null)      // { filename: pageData }
  const [pageId, setPageId]             = useState(PAGES[0])
  const [tokenIdx, setTokenIdx]         = useState(0)
  const [label, setLabel]               = useState('')
  const [savedTotal, setSavedTotal]     = useState(0)
  const [savedPerPage, setSavedPerPage] = useState({})
  const [apiAvailable, setApiAvailable] = useState(true)
  const [saveFlash, setSaveFlash]       = useState(false)
  const [displayScale, setDisplayScale] = useState(1)

  const hiddenImgRef  = useRef(null)   // full-res, hidden, used for canvas.drawImage
  const bgCanvasRef   = useRef(null)   // reference crop (no strokes)
  const drawCanvasRef = useRef(null)   // interactive canvas (image + strokes)
  const labelRef      = useRef(null)

  // ── Load results.json ─────────────────────────────────────────────────────

  useEffect(() => {
    fetch('/results.json')
      .then(r => r.json())
      .then(d => {
        const list = Array.isArray(d) ? d : (d.pages || [])
        const map = {}
        list.forEach(p => { map[p.filename] = p })
        setPages(map)
      })
      .catch(() => {})
  }, [])

  // ── Load saved count ──────────────────────────────────────────────────────

  useEffect(() => {
    fetch('/api/training-pairs')
      .then(r => r.json())
      .then(d => {
        setSavedTotal(d.total ?? 0)
        const perPage = {}
        ;(d.pairs ?? []).forEach(p => {
          perPage[p.page] = (perPage[p.page] ?? 0) + 1
        })
        setSavedPerPage(perPage)
      })
      .catch(() => setApiAvailable(false))
  }, [])

  // ── Current token ─────────────────────────────────────────────────────────

  const tokens = pages?.[pageId]?.tokens ?? []
  const token  = tokens[tokenIdx] ?? null

  // ── Draw helpers ──────────────────────────────────────────────────────────

  function drawIntoCanvas(canvas, tok) {
    const img = hiddenImgRef.current
    if (!canvas || !img?.complete || !tok?.bbox) return
    const [x, y, w, h] = tok.bbox
    if (w <= 0 || h <= 0) return
    const cw = Math.max(w * ZOOM, 80)
    const ch = Math.max(h * ZOOM, 40)
    canvas.width = cw
    canvas.height = ch
    canvas.getContext('2d').drawImage(img, x, y, w, h, 0, 0, cw, ch)
  }

  function refreshCanvases() {
    drawIntoCanvas(bgCanvasRef.current, token)
    drawIntoCanvas(drawCanvasRef.current, token)
  }

  function clearStrokes() {
    drawIntoCanvas(drawCanvasRef.current, token)
  }

  // Redraw when token changes
  useEffect(() => {
    setLabel(token?.text ?? '')
    refreshCanvases()
  }, [tokenIdx, pageId]) // eslint-disable-line

  // Reset when page changes
  useEffect(() => {
    setTokenIdx(0)
  }, [pageId])

  function onHiddenImgLoad() {
    refreshCanvases()
  }

  function onPageImgLoad(e) {
    setDisplayScale(e.target.clientWidth / e.target.naturalWidth)
  }

  // ── Drawing ───────────────────────────────────────────────────────────────

  const isDrawing = useRef(false)
  const lastPos   = useRef(null)

  function canvasPos(e) {
    const canvas = drawCanvasRef.current
    const rect = canvas.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top)  * (canvas.height / rect.height),
    }
  }

  function onMouseDown(e) {
    isDrawing.current = true
    lastPos.current = canvasPos(e)
  }

  function onMouseMove(e) {
    if (!isDrawing.current) return
    const p = canvasPos(e)
    const ctx = drawCanvasRef.current.getContext('2d')
    ctx.strokeStyle = INK_COLOR
    ctx.lineWidth   = INK_WIDTH
    ctx.lineCap     = 'round'
    ctx.lineJoin    = 'round'
    ctx.beginPath()
    ctx.moveTo(lastPos.current.x, lastPos.current.y)
    ctx.lineTo(p.x, p.y)
    ctx.stroke()
    lastPos.current = p
  }

  function onMouseUp() {
    isDrawing.current = false
    lastPos.current   = null
  }

  // ── Save ──────────────────────────────────────────────────────────────────

  const goNext = useCallback(() => {
    setTokenIdx(i => Math.min(i + 1, tokens.length - 1))
  }, [tokens.length])

  const goPrev = useCallback(() => {
    setTokenIdx(i => Math.max(i - 1, 0))
  }, [])

  const savePair = useCallback(async () => {
    if (!token || !label.trim()) return

    // Composite: bg image + strokes
    const bg  = bgCanvasRef.current
    const drw = drawCanvasRef.current
    if (!bg || !drw) return

    const off = document.createElement('canvas')
    off.width  = bg.width
    off.height = bg.height
    const ctx = off.getContext('2d')
    ctx.drawImage(bg, 0, 0)
    ctx.drawImage(drw, 0, 0)

    const patchB64 = off.toDataURL('image/png').split(',')[1]
    const form = new FormData()
    form.append('page_id',       pageId)
    form.append('token_index',   tokenIdx)
    form.append('label',         label.trim())
    form.append('patch_b64',     patchB64)
    form.append('original_bbox', JSON.stringify(token.bbox))

    try {
      const r = await fetch('/api/training-pairs', { method: 'POST', body: form })
      if (r.ok) {
        const d = await r.json()
        setSavedTotal(d.total_pairs)
        setSavedPerPage(prev => ({ ...prev, [pageId]: (prev[pageId] ?? 0) + 1 }))
        setSaveFlash(true)
        setTimeout(() => setSaveFlash(false), 900)
        goNext()
      }
    } catch {
      setApiAvailable(false)
    }
  }, [token, label, pageId, tokenIdx, goNext])

  // ── Keyboard shortcuts ────────────────────────────────────────────────────

  useEffect(() => {
    function onKey(e) {
      if (document.activeElement === labelRef.current) {
        if (e.key === 'Enter') { e.preventDefault(); savePair() }
        return
      }
      if (e.key === 'Enter' || e.key === 's' || e.key === 'S') {
        e.preventDefault(); savePair()
      } else if (e.key === ' ') {
        e.preventDefault(); goNext()
      } else if (e.key === 'ArrowRight') {
        e.preventDefault(); goPrev()
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault(); goNext()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [savePair, goNext, goPrev])

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-full flex-col overflow-hidden bg-gray-900 text-gray-100">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex flex-none flex-wrap items-center gap-2 border-b border-gray-700 bg-gray-800 px-4 py-2 text-sm">
        <button onClick={onBack} className="text-gray-400 hover:text-white">← Back</button>

        <select
          value={pageId}
          onChange={e => setPageId(e.target.value)}
          className="rounded bg-gray-700 px-2 py-1 text-sm text-white"
        >
          {PAGES.map(p => (
            <option key={p} value={p}>
              {p}{savedPerPage[p] ? ` · ${savedPerPage[p]} saved` : ''}
            </option>
          ))}
        </select>

        <span className="text-gray-400 tabular-nums">
          {tokenIdx + 1} / {tokens.length}
        </span>
        <button onClick={goPrev} disabled={tokenIdx === 0}
          className="rounded bg-gray-700 px-2 py-1 hover:bg-gray-600 disabled:opacity-30 text-xs">
          ‹
        </button>
        <button onClick={goNext} disabled={tokenIdx >= tokens.length - 1}
          className="rounded bg-gray-700 px-2 py-1 hover:bg-gray-600 disabled:opacity-30 text-xs">
          ›
        </button>

        <div className="ml-auto flex items-center gap-3">
          {!apiAvailable && (
            <span className="text-xs text-yellow-400">API unavailable — pairs will not be saved</span>
          )}
          {saveFlash && (
            <span className="font-semibold text-green-400">✓ Saved</span>
          )}
          <span className="text-gray-400 text-xs">{savedTotal} pairs saved</span>
          <button
            onClick={() => window.open('/api/training-pairs/export', '_blank')}
            className="rounded bg-gray-700 px-3 py-1 text-xs hover:bg-gray-600"
          >
            ↓ Export pairs
          </button>
        </div>
      </div>

      {/* ── Body ───────────────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left: manuscript page + bbox overlays */}
        <div className="relative flex-1 overflow-auto bg-gray-950 p-3">
          {/* Hidden full-res image for canvas.drawImage */}
          <img
            ref={hiddenImgRef}
            src={`/images/${pageId}`}
            onLoad={onHiddenImgLoad}
            className="hidden"
            crossOrigin="anonymous"
            alt=""
          />

          <div className="relative inline-block">
            <img
              src={`/images/${pageId}`}
              onLoad={onPageImgLoad}
              className="block max-w-full select-none"
              alt={pageId}
              draggable={false}
            />
            {/* SVG bbox overlays — sized to match the displayed img */}
            <svg
              className="pointer-events-none absolute inset-0 h-full w-full"
            >
              {tokens.map((tok, i) => {
                if (!tok.bbox) return null
                const [bx, by, bw, bh] = tok.bbox
                const s = displayScale
                const sel = i === tokenIdx
                return (
                  <rect
                    key={i}
                    x={bx * s} y={by * s}
                    width={Math.max(bw * s, 4)} height={Math.max(bh * s, 4)}
                    fill={sel ? 'rgba(99,102,241,0.25)' : 'rgba(250,204,21,0.10)'}
                    stroke={sel ? '#818cf8' : '#fbbf24'}
                    strokeWidth={sel ? 2 : 0.5}
                    rx={1}
                    className="pointer-events-auto cursor-pointer"
                    onClick={() => setTokenIdx(i)}
                  />
                )
              })}
            </svg>
          </div>
        </div>

        {/* Right: detail panel */}
        <aside className="flex w-80 flex-shrink-0 flex-col gap-4 overflow-y-auto border-l border-gray-700 bg-gray-800 p-4">
          {!token ? (
            <p className="text-sm text-gray-400">
              {tokens.length === 0 ? 'Loading tokens…' : 'Click a word on the page to select it.'}
            </p>
          ) : (
            <>
              {/* Reference crop */}
              <div>
                <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-gray-400">
                  Word crop (4×)
                </p>
                <div className="overflow-hidden rounded border border-gray-600 bg-white">
                  <canvas
                    ref={bgCanvasRef}
                    className="block w-full"
                    style={{ imageRendering: 'pixelated' }}
                  />
                </div>
              </div>

              {/* Drawing canvas */}
              <div>
                <div className="mb-1 flex items-center justify-between">
                  <p className="text-[10px] font-semibold uppercase tracking-wide text-gray-400">
                    Draw correction
                  </p>
                  <button
                    onClick={clearStrokes}
                    className="text-[10px] text-gray-400 hover:text-white"
                  >
                    Clear
                  </button>
                </div>
                <div className="overflow-hidden rounded border border-gray-600 bg-white">
                  <canvas
                    ref={drawCanvasRef}
                    className="block w-full cursor-crosshair"
                    style={{ imageRendering: 'pixelated', touchAction: 'none' }}
                    onMouseDown={onMouseDown}
                    onMouseMove={onMouseMove}
                    onMouseUp={onMouseUp}
                    onMouseLeave={onMouseUp}
                  />
                </div>
              </div>

              {/* Label */}
              <div>
                <div className="mb-1 flex items-center justify-between">
                  <p className="text-[10px] font-semibold uppercase tracking-wide text-gray-400">
                    Label
                  </p>
                  <button
                    onClick={() => setLabel(token.text)}
                    className="text-[10px] text-indigo-400 hover:text-indigo-300"
                  >
                    Use OCR ↑
                  </button>
                </div>
                <p className="mb-1.5 text-[11px] text-gray-500">
                  OCR:{' '}
                  <span dir="rtl" style={{ fontFamily: ARABIC_FONT }}>{token.text}</span>
                  {' '}({(token.confidence * 100).toFixed(0)}%)
                </p>
                <input
                  ref={labelRef}
                  type="text"
                  dir="rtl"
                  lang="ar"
                  value={label}
                  onChange={e => setLabel(e.target.value)}
                  className="w-full rounded border border-gray-600 bg-gray-700 px-3 py-2 text-lg text-white placeholder-gray-500 focus:border-indigo-400 focus:outline-none"
                  style={{ fontFamily: ARABIC_FONT }}
                  placeholder="Arabic text…"
                />
              </div>

              {/* Action buttons */}
              <div className="flex gap-2">
                <button
                  onClick={savePair}
                  disabled={!label.trim()}
                  className="flex-1 rounded bg-indigo-600 px-3 py-2 text-sm font-semibold hover:bg-indigo-500 disabled:opacity-40"
                >
                  💾 Save <span className="text-xs opacity-50">(S)</span>
                </button>
                <button
                  onClick={goNext}
                  className="flex-1 rounded bg-gray-700 px-3 py-2 text-sm font-semibold hover:bg-gray-600"
                >
                  Skip <span className="text-xs opacity-50">(Space)</span>
                </button>
              </div>

              {/* Shortcut legend */}
              <div className="space-y-0.5 text-[10px] text-gray-500">
                <p>← / → : prev / next token</p>
                <p>S / Enter : save + advance</p>
                <p>Space : skip</p>
              </div>
            </>
          )}
        </aside>
      </div>
    </div>
  )
}
