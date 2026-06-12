import { useState, useEffect, useRef, useCallback } from 'react'

const ARABIC_FONT = "'Amiri', 'Traditional Arabic', serif"
const PAGES = ['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg']
const ZOOM = 4
const INK_COLOR = '#ef4444'
const INK_WIDTH = 2

function effectiveBbox(tok, imgH) {
  if (!tok?.bbox) return [0, 0, 0, 0]
  const [x, y, w, h] = tok.bbox
  if (h >= 10) return [x, y, w, h]
  // Expand: y-20 to y+40, clamped to image height
  const ny = Math.max(0, y - 20)
  const nh = Math.min(60, (imgH ?? 9999) - ny)
  return [x, ny, Math.max(w, 4), nh]
}

export default function AnnotationView({ onBack }) {
  const [pages, setPages]               = useState(null)
  const [pageId, setPageId]             = useState(PAGES[0])
  const [tokenIdx, setTokenIdx]         = useState(0)
  const [label, setLabel]               = useState('')
  const [savedTotal, setSavedTotal]     = useState(0)
  const [savedPerPage, setSavedPerPage] = useState({})
  const [apiAvailable, setApiAvailable] = useState(true)
  const [saveFlash, setSaveFlash]       = useState(false)
  const [displayScale, setDisplayScale] = useState(1)
  const [canvasSize, setCanvasSize]     = useState({ w: 1, h: 1 })
  const [imgNaturalH, setImgNaturalH]   = useState(9999)

  const hiddenImgRef  = useRef(null)   // full-res hidden, for canvas drawImage
  const drawCanvasRef = useRef(null)   // full-page drawing canvas (overlay)
  const bgCanvasRef   = useRef(null)   // right panel: word crop reference
  const labelRef      = useRef(null)

  const isDrawing    = useRef(false)
  const pendingClick = useRef(false)
  const dragStart    = useRef({ x: 0, y: 0 })
  const lastPos      = useRef(null)

  // ── Data loading ──────────────────────────────────────────────────────────

  useEffect(() => {
    fetch('/results.json').then(r => r.json()).then(d => {
      const list = Array.isArray(d) ? d : (d.pages || [])
      const map = {}
      list.forEach(p => { map[p.filename] = p })
      setPages(map)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    fetch('/api/training-pairs').then(r => r.json()).then(d => {
      setSavedTotal(d.total ?? 0)
      const pp = {}
      ;(d.pairs ?? []).forEach(p => { pp[p.page] = (pp[p.page] ?? 0) + 1 })
      setSavedPerPage(pp)
    }).catch(() => setApiAvailable(false))
  }, [])

  // ── Token helpers ─────────────────────────────────────────────────────────

  const tokens = pages?.[pageId]?.tokens ?? []
  const token  = tokens[tokenIdx] ?? null

  useEffect(() => { setTokenIdx(0) }, [pageId])

  useEffect(() => {
    setLabel(token?.text ?? '')
    drawCropRef()
  }, [tokenIdx, pageId]) // eslint-disable-line

  // ── Word crop (right panel) ───────────────────────────────────────────────

  function drawCropRef() {
    const canvas = bgCanvasRef.current
    const img    = hiddenImgRef.current
    if (!canvas || !img?.complete || !token) return
    const [x, y, w, h] = effectiveBbox(token, imgNaturalH)
    if (w <= 0 || h <= 0) return
    canvas.width  = Math.max(w * ZOOM, 80)
    canvas.height = Math.max(h * ZOOM, 40)
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, x, y, w, h, 0, 0, canvas.width, canvas.height)
    // Red rectangle outline indicating the selected word
    ctx.strokeStyle = '#ef4444'
    ctx.lineWidth = 3
    ctx.strokeRect(1.5, 1.5, canvas.width - 3, canvas.height - 3)
  }

  function onHiddenImgLoad() {
    const img = hiddenImgRef.current
    if (img) setImgNaturalH(img.naturalHeight)
    drawCropRef()
  }

  function onPageImgLoad(e) {
    const img  = e.target
    const rect = img.getBoundingClientRect()
    const s    = rect.width / img.naturalWidth
    setDisplayScale(s)
    setCanvasSize({ w: Math.round(rect.width), h: Math.round(rect.height) })
    setImgNaturalH(img.naturalHeight)
  }

  // ── Drawing (full-page canvas overlay) ───────────────────────────────────

  function canvasPos(e) {
    const canvas = drawCanvasRef.current
    const rect   = canvas.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left) * (canvas.width  / rect.width),
      y: (e.clientY - rect.top)  * (canvas.height / rect.height),
    }
  }

  function onMouseDown(e) {
    pendingClick.current = true
    dragStart.current    = { x: e.clientX, y: e.clientY }
    lastPos.current      = canvasPos(e)
  }

  function onMouseMove(e) {
    if (!pendingClick.current && !isDrawing.current) return
    const dx = e.clientX - dragStart.current.x
    const dy = e.clientY - dragStart.current.y
    // Promote click → draw once cursor moves > 3px
    if (pendingClick.current && Math.sqrt(dx * dx + dy * dy) > 3) {
      pendingClick.current = false
      isDrawing.current    = true
    }
    if (!isDrawing.current) return
    const p   = canvasPos(e)
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

  function onMouseUp(e) {
    if (pendingClick.current) {
      // Hit-test: find the bbox under click position → select token
      const p  = canvasPos(e)
      const ox = p.x / displayScale
      const oy = p.y / displayScale
      for (let i = 0; i < tokens.length; i++) {
        const [bx, by, bw, bh] = effectiveBbox(tokens[i], imgNaturalH)
        if (ox >= bx && ox <= bx + bw && oy >= by && oy <= by + bh) {
          setTokenIdx(i)
          break
        }
      }
    }
    pendingClick.current = false
    isDrawing.current    = false
    lastPos.current      = null
  }

  function clearStrokes() {
    const canvas = drawCanvasRef.current
    if (!canvas) return
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
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
    const img     = hiddenImgRef.current
    const strokes = drawCanvasRef.current
    if (!img?.complete || !strokes) return

    const [x, y, w, h] = effectiveBbox(token, imgNaturalH)
    if (w <= 0) return

    // Composite: original image crop + strokes cropped to same region
    const cw  = Math.max(w * ZOOM, 80)
    const ch  = Math.max(h * ZOOM, 40)
    const off = document.createElement('canvas')
    off.width = cw; off.height = ch
    const ctx = off.getContext('2d')
    ctx.drawImage(img, x, y, w, h, 0, 0, cw, ch)
    const ds  = displayScale
    ctx.drawImage(strokes, x * ds, y * ds, w * ds, h * ds, 0, 0, cw, ch)

    const patchB64 = off.toDataURL('image/png').split(',')[1]
    const form     = new FormData()
    form.append('page_id',       pageId)
    form.append('token_index',   tokenIdx)
    form.append('label',         label.trim())
    form.append('patch_b64',     patchB64)
    form.append('original_bbox', JSON.stringify([x, y, w, h]))

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
  }, [token, label, pageId, tokenIdx, displayScale, imgNaturalH, goNext])

  // ── Keyboard ──────────────────────────────────────────────────────────────

  useEffect(() => {
    function onKey(e) {
      if (document.activeElement === labelRef.current) {
        if (e.key === 'Enter') { e.preventDefault(); savePair() }
        return
      }
      if (e.key === 'Enter' || e.key === 's' || e.key === 'S') { e.preventDefault(); savePair() }
      else if (e.key === ' ')          { e.preventDefault(); goNext() }
      else if (e.key === 'ArrowRight') { e.preventDefault(); goPrev() }
      else if (e.key === 'ArrowLeft')  { e.preventDefault(); goNext() }
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

        <span className="tabular-nums text-gray-400">{tokenIdx + 1} / {tokens.length}</span>
        <button onClick={goPrev} disabled={tokenIdx === 0}
          className="rounded bg-gray-700 px-2 py-1 text-xs hover:bg-gray-600 disabled:opacity-30">‹</button>
        <button onClick={goNext} disabled={tokenIdx >= tokens.length - 1}
          className="rounded bg-gray-700 px-2 py-1 text-xs hover:bg-gray-600 disabled:opacity-30">›</button>

        <span className="text-xs text-gray-500">Click word to select · Drag to draw</span>

        <div className="ml-auto flex items-center gap-3">
          {!apiAvailable && (
            <span className="text-xs text-yellow-400">API unavailable — pairs will not be saved</span>
          )}
          {saveFlash && <span className="font-semibold text-green-400">✓ Saved</span>}
          <span className="text-xs text-gray-400">{savedTotal} pairs saved</span>
          <button
            onClick={clearStrokes}
            className="rounded bg-gray-700 px-3 py-1 text-xs hover:bg-gray-600"
          >Clear strokes</button>
          <button
            onClick={() => window.open('/api/training-pairs/export', '_blank')}
            className="rounded bg-gray-700 px-3 py-1 text-xs hover:bg-gray-600"
          >↓ Export</button>
        </div>
      </div>

      {/* ── Body ───────────────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left: manuscript image + SVG bbox overlays + drawing canvas */}
        <div className="relative flex-1 overflow-auto bg-gray-950 p-3">
          {/* Hidden full-res image for canvas compositing */}
          <img
            ref={hiddenImgRef}
            src={`/images/${pageId}`}
            onLoad={onHiddenImgLoad}
            className="hidden"
            crossOrigin="anonymous"
            alt=""
          />

          <div className="relative inline-block select-none">
            {/* Manuscript image */}
            <img
              src={`/images/${pageId}`}
              onLoad={onPageImgLoad}
              className="block max-w-full"
              alt={pageId}
              draggable={false}
            />

            {/* SVG: bbox overlays (below canvas, visual only) */}
            <svg
              className="pointer-events-none absolute"
              style={{ top: 0, left: 0, width: canvasSize.w, height: canvasSize.h }}
            >
              {tokens.map((tok, i) => {
                if (!tok.bbox) return null
                const [bx, by, bw, bh] = effectiveBbox(tok, imgNaturalH)
                const s   = displayScale
                const sel = i === tokenIdx
                return (
                  <rect
                    key={i}
                    x={bx * s} y={by * s}
                    width={Math.max(bw * s, 4)} height={Math.max(bh * s, 4)}
                    fill={sel ? 'rgba(239,68,68,0.20)' : 'rgba(250,204,21,0.08)'}
                    stroke={sel ? '#ef4444' : '#fbbf24'}
                    strokeWidth={sel ? 2.5 : 0.5}
                    rx={1}
                  />
                )
              })}
            </svg>

            {/* Drawing canvas: transparent overlay, captures all mouse events */}
            <canvas
              ref={drawCanvasRef}
              width={canvasSize.w}
              height={canvasSize.h}
              className="absolute cursor-crosshair"
              style={{ top: 0, left: 0 }}
              onMouseDown={onMouseDown}
              onMouseMove={onMouseMove}
              onMouseUp={onMouseUp}
              onMouseLeave={onMouseUp}
            />
          </div>
        </div>

        {/* Right: word crop + label + buttons */}
        <aside className="flex w-72 flex-shrink-0 flex-col gap-4 overflow-y-auto border-l border-gray-700 bg-gray-800 p-4">
          {!token ? (
            <p className="text-sm text-gray-400">
              {tokens.length === 0 ? 'Loading tokens…' : 'Click a word on the manuscript to select it.'}
            </p>
          ) : (
            <>
              {/* Word crop reference at 4× with red outline */}
              <div>
                <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-gray-400">
                  Word crop (4×)
                </p>
                <div className="overflow-hidden rounded bg-white">
                  <canvas
                    ref={bgCanvasRef}
                    className="block w-full"
                    style={{ imageRendering: 'pixelated' }}
                  />
                </div>
              </div>

              {/* Label input */}
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

              {/* Shortcuts */}
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
