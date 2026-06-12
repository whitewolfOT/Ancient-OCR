import { useState, useEffect, useRef, useCallback, useMemo } from 'react'

const ARABIC_FONT = "'Amiri', 'Traditional Arabic', serif"
const PAGES = ['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg']
const ZOOM = 4
const INK_COLOR  = '#ef4444'
const INK_WIDTH  = 3
const HANDLE_R   = 5   // visual radius (display px)
const HANDLE_HIT = 9   // hit detection radius (display px)

const MODE_SELECT = 'select'
const MODE_ADDBOX = 'addbox'
const MODE_LASSO  = 'lasso'

function effectiveBbox(tok, imgH) {
  if (!tok?.bbox) return [0, 0, 0, 0]
  const [x, y, w, h] = tok.bbox
  if (h >= 10) return [x, y, w, h]
  const ny = Math.max(0, y - 30)
  const nh = Math.min(80, (imgH ?? 9999) - ny)
  return [x, ny, Math.max(w, 4), nh]
}

function rectOverlap(ax, ay, aw, ah, bx, by, bw, bh) {
  const ox = Math.max(0, Math.min(ax + aw, bx + bw) - Math.max(ax, bx))
  const oy = Math.max(0, Math.min(ay + ah, by + bh) - Math.max(ay, by))
  return ox * oy
}

// 8 handles in display-space from original-coord bbox
function getHandles(tok, imgH, sx, sy) {
  const [bx, by, bw, bh] = effectiveBbox(tok, imgH)
  const x = bx * sx, y = by * sy, w = bw * sx, h = bh * sy
  return [
    { id: 'nw', x,        y        }, { id: 'n', x: x+w/2, y        },
    { id: 'ne', x: x+w,   y        }, { id: 'e', x: x+w,   y: y+h/2 },
    { id: 'se', x: x+w,   y: y+h   }, { id: 's', x: x+w/2, y: y+h   },
    { id: 'sw', x,        y: y+h   }, { id: 'w', x,        y: y+h/2 },
  ]
}

// dx/dy in original image coordinates
function applyHandleDrag(orig, handle, dx, dy) {
  let [x, y, w, h] = orig
  const mw = (v) => Math.max(4, v)
  switch (handle) {
    case 'move': return [x+dx,    y+dy,    w,       h      ]
    case 'nw':   return [x+dx,    y+dy,    mw(w-dx), mw(h-dy)]
    case 'n':    return [x,       y+dy,    w,       mw(h-dy)]
    case 'ne':   return [x,       y+dy,    mw(w+dx), mw(h-dy)]
    case 'e':    return [x,       y,       mw(w+dx), h      ]
    case 'se':   return [x,       y,       mw(w+dx), mw(h+dy)]
    case 's':    return [x,       y,       w,       mw(h+dy)]
    case 'sw':   return [x+dx,    y,       mw(w-dx), mw(h+dy)]
    case 'w':    return [x+dx,    y,       mw(w-dx), h      ]
    default:     return orig
  }
}

// RTL reading order: top-to-bottom (y asc), right-to-left within line (x desc)
function rtlSort(a, b) {
  const [ax, ay] = a.bbox ?? [0, 0]
  const [bx, by] = b.bbox ?? [0, 0]
  const la = Math.round(ay / 20), lb = Math.round(by / 20)
  if (la !== lb) return la - lb
  return bx - ax
}

let _uid = 0
function uid() { return ++_uid }

export default function AnnotationView({ onBack }) {
  const [pages, setPages]               = useState(null)
  const [pageId, setPageId]             = useState(PAGES[0])
  const [tokenIdx, setTokenIdx]         = useState(0)
  const [label, setLabel]               = useState('')
  const [savedTotal, setSavedTotal]     = useState(0)
  const [savedPerPage, setSavedPerPage] = useState({})
  const [apiAvailable, setApiAvailable] = useState(true)
  const [saveFlash, setSaveFlash]       = useState(false)
  const [scale, setScale]               = useState({ x: 1, y: 1 })
  const [canvasSize, setCanvasSize]     = useState({ w: 1, h: 1 })
  const [imgNaturalH, setImgNaturalH]   = useState(9999)
  const [mode, setMode]                 = useState(MODE_SELECT)
  const [manualTokens, setManualTokens] = useState([])
  const [bboxOverrides, setBboxOverrides] = useState({})   // baseIdx → [x,y,w,h]
  const [deletedBaseIds, setDeletedBaseIds] = useState(new Set())
  const [, forceUpdate]                 = useState(0)

  const hiddenImgRef     = useRef(null)
  const pageImgRef       = useRef(null)
  const drawCanvasRef    = useRef(null)
  const bgCanvasRef      = useRef(null)
  const labelRef         = useRef(null)
  const isDrawing        = useRef(false)
  const pendingClick     = useRef(false)
  const dragStart        = useRef({ x: 0, y: 0 })
  const lastPos          = useRef(null)
  const strokesRef       = useRef([])
  const redoRef          = useRef([])
  const currentStroke    = useRef(null)
  const lassoPath        = useRef([])
  const boxStart         = useRef(null)
  const dragState        = useRef(null)   // {handle, startX, startY, origBbox, tokId, isBase}
  const pendingSelectId  = useRef(null)   // uid of newly added token to select after sort

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

  useEffect(() => {
    function updateScale() {
      const img = pageImgRef.current
      if (!img || !img.naturalWidth) return
      const rect = img.getBoundingClientRect()
      setScale({ x: rect.width / img.naturalWidth, y: rect.height / img.naturalHeight })
      setCanvasSize({ w: Math.round(rect.width), h: Math.round(rect.height) })
    }
    window.addEventListener('resize', updateScale)
    return () => window.removeEventListener('resize', updateScale)
  }, [])

  // ── Token list: sorted RTL, bbox overrides applied, deleted filtered ───────

  const baseTokens = pages?.[pageId]?.tokens ?? []

  const allTokens = useMemo(() => {
    const base = baseTokens
      .map((t, i) => ({ ...t, _baseIdx: i, bbox: bboxOverrides[i] ?? t.bbox }))
      .filter((_, i) => !deletedBaseIds.has(i))
    return [...base, ...manualTokens].sort(rtlSort)
  }, [baseTokens, manualTokens, bboxOverrides, deletedBaseIds])

  const token = allTokens[tokenIdx] ?? null

  // After addbox/lasso creates a new token, find its sorted index and select it
  useEffect(() => {
    if (pendingSelectId.current == null) return
    const idx = allTokens.findIndex(t => t._uid === pendingSelectId.current)
    if (idx >= 0) {
      setTokenIdx(idx)
      setLabel('')
      pendingSelectId.current = null
    }
  }, [allTokens])

  useEffect(() => {
    setTokenIdx(0)
    setManualTokens([])
    setBboxOverrides({})
    setDeletedBaseIds(new Set())
    pendingSelectId.current = null
    strokesRef.current = []
    redoRef.current    = []
    const c = drawCanvasRef.current
    if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height)
    forceUpdate(n => n + 1)
  }, [pageId])

  // 1. DEBUG — log crop coordinates on selection change
  useEffect(() => {
    const img  = hiddenImgRef.current
    const pimg = pageImgRef.current
    if (token) {
      console.log('[crop-debug]', {
        originalWidth:   img?.naturalWidth,
        originalHeight:  img?.naturalHeight,
        displayedWidth:  pimg ? Math.round(pimg.getBoundingClientRect().width)  : null,
        displayedHeight: pimg ? Math.round(pimg.getBoundingClientRect().height) : null,
        'scale.x': +scale.x.toFixed(4),
        'scale.y': +scale.y.toFixed(4),
        bboxRaw:       token.bbox,
        bboxEffective: effectiveBbox(token, imgNaturalH),
      })
    }
    setLabel(token?.text ?? '')
    drawCropRef()
  }, [tokenIdx, pageId]) // eslint-disable-line

  // ── Canvas redraw ─────────────────────────────────────────────────────────

  const redrawCanvas = useCallback(() => {
    const canvas = drawCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.globalCompositeOperation = 'source-over'
    for (const stroke of strokesRef.current) {
      if (!stroke.points || stroke.points.length < 2) continue
      ctx.strokeStyle = stroke.color
      ctx.lineWidth   = stroke.width
      ctx.lineCap     = 'round'
      ctx.lineJoin    = 'round'
      ctx.beginPath()
      ctx.moveTo(stroke.points[0].x, stroke.points[0].y)
      for (const pt of stroke.points.slice(1)) ctx.lineTo(pt.x, pt.y)
      ctx.stroke()
    }
  }, [])

  // ── Word crop ─────────────────────────────────────────────────────────────

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
    ctx.strokeStyle = '#ef4444'
    ctx.lineWidth   = 3
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
    setScale({ x: rect.width / img.naturalWidth, y: rect.height / img.naturalHeight })
    setCanvasSize({ w: Math.round(rect.width), h: Math.round(rect.height) })
    setImgNaturalH(img.naturalHeight)
  }

  // ── Undo / Redo / Clear strokes ───────────────────────────────────────────

  const undo = useCallback(() => {
    if (!strokesRef.current.length) return
    redoRef.current.push(strokesRef.current.pop())
    redrawCanvas(); forceUpdate(n => n + 1)
  }, [redrawCanvas])

  const redo = useCallback(() => {
    if (!redoRef.current.length) return
    strokesRef.current.push(redoRef.current.pop())
    redrawCanvas(); forceUpdate(n => n + 1)
  }, [redrawCanvas])

  // 4. Clear strokes only — does NOT touch token boxes
  const clearStrokes = useCallback(() => {
    strokesRef.current = []
    redoRef.current    = []
    const c = drawCanvasRef.current
    if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height)
    forceUpdate(n => n + 1)
  }, [])

  // ── Delete token ──────────────────────────────────────────────────────────

  // 6. Delete selected token
  const deleteToken = useCallback(() => {
    if (!token) return
    if (token._baseIdx !== undefined) {
      setDeletedBaseIds(prev => new Set([...prev, token._baseIdx]))
    } else {
      const id = token._uid
      setManualTokens(prev => prev.filter(t => t._uid !== id))
    }
    setTokenIdx(i => Math.max(0, i - 1))
  }, [token])

  // ── Update token bbox (drag/resize) ───────────────────────────────────────

  const updateTokenBbox = useCallback((tokId, isBase, newBbox) => {
    const clamped = [
      Math.max(0, Math.round(newBbox[0])),
      Math.max(0, Math.round(newBbox[1])),
      Math.max(4, Math.round(newBbox[2])),
      Math.max(4, Math.round(newBbox[3])),
    ]
    if (isBase) {
      setBboxOverrides(prev => ({ ...prev, [tokId]: clamped }))
    } else {
      setManualTokens(prev => prev.map(t => t._uid === tokId ? { ...t, bbox: clamped } : t))
    }
  }, [])

  // ── Canvas position helper ────────────────────────────────────────────────

  function canvasPos(e) {
    const c    = drawCanvasRef.current
    const rect = c.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left) * (c.width  / rect.width),
      y: (e.clientY - rect.top)  * (c.height / rect.height),
    }
  }

  // ── Mouse handlers ────────────────────────────────────────────────────────

  function onMouseDown(e) {
    const pos = canvasPos(e)
    dragStart.current = { x: e.clientX, y: e.clientY }
    lastPos.current   = pos

    if (mode === MODE_SELECT) {
      // 3. Check resize handles first
      if (token) {
        const handles = getHandles(token, imgNaturalH, scale.x, scale.y)
        for (const h of handles) {
          if (Math.abs(pos.x - h.x) <= HANDLE_HIT && Math.abs(pos.y - h.y) <= HANDLE_HIT) {
            dragState.current = {
              handle:   h.id,
              startX:   pos.x,
              startY:   pos.y,
              origBbox: [...effectiveBbox(token, imgNaturalH)],
              tokId:    token._baseIdx ?? token._uid,
              isBase:   token._baseIdx !== undefined,
            }
            return
          }
        }
        // Check inside bbox for move
        const [bx, by, bw, bh] = effectiveBbox(token, imgNaturalH)
        const ox = pos.x / scale.x, oy = pos.y / scale.y
        if (ox >= bx && ox <= bx + bw && oy >= by && oy <= by + bh) {
          dragState.current = {
            handle:   'move',
            startX:   pos.x,
            startY:   pos.y,
            origBbox: [bx, by, bw, bh],
            tokId:    token._baseIdx ?? token._uid,
            isBase:   token._baseIdx !== undefined,
          }
          return
        }
      }
      pendingClick.current = true
    } else if (mode === MODE_ADDBOX) {
      isDrawing.current = true
      boxStart.current  = pos
    } else if (mode === MODE_LASSO) {
      isDrawing.current = true
      lassoPath.current = [pos]
    }
  }

  function onMouseMove(e) {
    const pos = canvasPos(e)
    const dx  = e.clientX - dragStart.current.x
    const dy  = e.clientY - dragStart.current.y

    // 3. Handle drag / resize
    if (dragState.current) {
      const ds  = dragState.current
      const ddx = (pos.x - ds.startX) / scale.x
      const ddy = (pos.y - ds.startY) / scale.y
      updateTokenBbox(ds.tokId, ds.isBase, applyHandleDrag(ds.origBbox, ds.handle, ddx, ddy))
      return
    }

    if (mode === MODE_SELECT) {
      if (pendingClick.current && Math.sqrt(dx * dx + dy * dy) > 3) {
        pendingClick.current  = false
        isDrawing.current     = true
        currentStroke.current = [lastPos.current]
      }
      if (isDrawing.current) {
        currentStroke.current?.push(pos)
        const ctx = drawCanvasRef.current?.getContext('2d')
        if (ctx && currentStroke.current?.length >= 2) {
          const pts  = currentStroke.current
          const prev = pts[pts.length - 2]
          ctx.globalCompositeOperation = 'source-over'
          ctx.strokeStyle = INK_COLOR
          ctx.lineWidth   = INK_WIDTH
          ctx.lineCap     = 'round'
          ctx.lineJoin    = 'round'
          ctx.beginPath()
          ctx.moveTo(prev.x, prev.y)
          ctx.lineTo(pos.x, pos.y)
          ctx.stroke()
        }
      }
    } else if (mode === MODE_ADDBOX && isDrawing.current) {
      redrawCanvas()
      const ctx = drawCanvasRef.current?.getContext('2d')
      if (ctx && boxStart.current) {
        ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 2
        ctx.setLineDash([5, 4])
        ctx.strokeRect(boxStart.current.x, boxStart.current.y,
          pos.x - boxStart.current.x, pos.y - boxStart.current.y)
        ctx.setLineDash([])
      }
    } else if (mode === MODE_LASSO && isDrawing.current) {
      lassoPath.current.push(pos)
      redrawCanvas()
      const ctx = drawCanvasRef.current?.getContext('2d')
      if (ctx && lassoPath.current.length > 1) {
        ctx.strokeStyle = '#a855f7'; ctx.lineWidth = 2
        ctx.setLineDash([3, 3])
        ctx.beginPath()
        ctx.moveTo(lassoPath.current[0].x, lassoPath.current[0].y)
        for (const pt of lassoPath.current.slice(1)) ctx.lineTo(pt.x, pt.y)
        ctx.stroke()
        ctx.setLineDash([])
      }
    }
    lastPos.current = pos
  }

  function onMouseUp(e) {
    if (dragState.current) { dragState.current = null; return }

    if (mode === MODE_SELECT) {
      if (pendingClick.current) {
        const p  = canvasPos(e)
        const ox = p.x / scale.x, oy = p.y / scale.y
        for (let i = 0; i < allTokens.length; i++) {
          const [bx, by, bw, bh] = effectiveBbox(allTokens[i], imgNaturalH)
          if (ox >= bx && ox <= bx + bw && oy >= by && oy <= by + bh) {
            setTokenIdx(i); break
          }
        }
      } else if (isDrawing.current && currentStroke.current?.length >= 2) {
        strokesRef.current.push({ color: INK_COLOR, width: INK_WIDTH, points: currentStroke.current })
        redoRef.current = []; currentStroke.current = null
        forceUpdate(n => n + 1)
      }
    } else if (mode === MODE_ADDBOX && isDrawing.current) {
      const pos = canvasPos(e), bst = boxStart.current
      if (bst) {
        const rx = Math.min(bst.x, pos.x) / scale.x, ry = Math.min(bst.y, pos.y) / scale.y
        const rw = Math.abs(pos.x - bst.x) / scale.x, rh = Math.abs(pos.y - bst.y) / scale.y
        if (rw > 4 && rh > 4) {
          const id = uid()
          // 2. Store ID for RTL-sorted position lookup after re-render
          pendingSelectId.current = id
          setManualTokens(prev => [...prev, {
            _uid: id, text: '', confidence: 0, source: 'manual',
            bbox: [Math.round(rx), Math.round(ry), Math.round(rw), Math.round(rh)]
          }])
        }
        boxStart.current = null; redrawCanvas()
      }
    } else if (mode === MODE_LASSO && isDrawing.current) {
      const pts = lassoPath.current
      if (pts.length > 3) {
        const xs = pts.map(p => p.x / scale.x), ys = pts.map(p => p.y / scale.y)
        const lx = Math.min(...xs), ly = Math.min(...ys)
        const lw = Math.max(...xs) - lx, lh = Math.max(...ys) - ly
        let bestIdx = -1, bestArea = 0
        for (let i = 0; i < allTokens.length; i++) {
          const [bx, by, bw, bh] = effectiveBbox(allTokens[i], imgNaturalH)
          const area = rectOverlap(lx, ly, lw, lh, bx, by, bw, bh)
          if (area > bestArea) { bestArea = area; bestIdx = i }
        }
        if (bestIdx >= 0 && bestArea > 0) {
          setTokenIdx(bestIdx)
        } else if (lw > 4 && lh > 4) {
          const id = uid()
          pendingSelectId.current = id
          setManualTokens(prev => [...prev, {
            _uid: id, text: '', confidence: 0, source: 'manual',
            bbox: [Math.round(lx), Math.round(ly), Math.round(lw), Math.round(lh)]
          }])
        }
      }
      lassoPath.current = []; redrawCanvas()
    }
    pendingClick.current = false; isDrawing.current = false; lastPos.current = null
  }

  // ── Navigation + Save ─────────────────────────────────────────────────────

  const goNext = useCallback(() => {
    setTokenIdx(i => Math.min(i + 1, allTokens.length - 1))
  }, [allTokens.length])

  const goPrev = useCallback(() => {
    setTokenIdx(i => Math.max(i - 1, 0))
  }, [])

  const savePair = useCallback(async () => {
    const currentLabel = (labelRef.current?.value ?? '').trim()
    // 5. Debug: log what is being saved
    console.log('[save]', { label: currentLabel, tokenIdx, pageId, bbox: token ? effectiveBbox(token, imgNaturalH) : null })
    if (!token || !currentLabel) return
    const img     = hiddenImgRef.current
    const strokes = drawCanvasRef.current
    if (!img?.complete || !strokes) return

    const [x, y, w, h] = effectiveBbox(token, imgNaturalH)
    if (w <= 0) return

    const cw  = Math.max(w * ZOOM, 80), ch = Math.max(h * ZOOM, 40)
    const off = document.createElement('canvas')
    off.width = cw; off.height = ch
    const ctx = off.getContext('2d')
    ctx.drawImage(img, x, y, w, h, 0, 0, cw, ch)
    ctx.drawImage(strokes, x * scale.x, y * scale.y, w * scale.x, h * scale.y, 0, 0, cw, ch)

    const patchB64 = off.toDataURL('image/png').split(',')[1]
    const form = new FormData()
    form.append('page_id',       pageId)
    form.append('token_index',   tokenIdx)
    form.append('label',         currentLabel)
    form.append('patch_b64',     patchB64)
    form.append('original_bbox', JSON.stringify([x, y, w, h]))

    try {
      const r = await fetch('/api/training-pairs', { method: 'POST', body: form })
      if (r.ok) {
        const d = await r.json()
        setSavedTotal(d.total_pairs)
        setSavedPerPage(prev => ({ ...prev, [pageId]: (prev[pageId] ?? 0) + 1 }))
        setSaveFlash(true); setTimeout(() => setSaveFlash(false), 900)
        goNext()
      }
    } catch { setApiAvailable(false) }
  }, [token, pageId, tokenIdx, scale, imgNaturalH, goNext])

  // ── Keyboard ──────────────────────────────────────────────────────────────

  useEffect(() => {
    function onKey(e) {
      if (e.ctrlKey && e.key === 'z' && !e.shiftKey) { e.preventDefault(); undo(); return }
      if ((e.ctrlKey && e.shiftKey && e.key === 'z') || (e.ctrlKey && e.key === 'y')) {
        e.preventDefault(); redo(); return
      }
      if (document.activeElement === labelRef.current) {
        if (e.key === 'Enter') { e.preventDefault(); savePair() }
        return
      }
      // 6. Delete key
      if (e.key === 'Delete') { e.preventDefault(); deleteToken(); return }
      if (e.key === 'Enter' || e.key === 's' || e.key === 'S') { e.preventDefault(); savePair() }
      else if (e.key === ' ')          { e.preventDefault(); goNext() }
      else if (e.key === 'ArrowRight') { e.preventDefault(); goPrev() }
      else if (e.key === 'ArrowLeft')  { e.preventDefault(); goNext() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [savePair, goNext, goPrev, undo, redo, deleteToken])

  // ── Render ────────────────────────────────────────────────────────────────

  const cursorClass = mode === MODE_ADDBOX ? 'cursor-crosshair'
                    : mode === MODE_LASSO  ? 'cursor-cell'
                    : 'cursor-default'

  const modeLabel = { [MODE_SELECT]: 'Select', [MODE_ADDBOX]: 'Add Box', [MODE_LASSO]: 'Lasso' }

  return (
    <div className="flex h-full flex-col overflow-hidden bg-gray-900 text-gray-100">

      {/* Header */}
      <div className="flex flex-none flex-wrap items-center gap-2 border-b border-gray-700 bg-gray-800 px-4 py-2 text-sm">
        <button onClick={onBack} className="text-gray-400 hover:text-white">← Back</button>

        <select value={pageId} onChange={e => setPageId(e.target.value)}
          className="rounded bg-gray-700 px-2 py-1 text-sm text-white">
          {PAGES.map(p => (
            <option key={p} value={p}>{p}{savedPerPage[p] ? ` · ${savedPerPage[p]} saved` : ''}</option>
          ))}
        </select>

        <span className="tabular-nums text-gray-400">{tokenIdx + 1} / {allTokens.length}</span>
        <button onClick={goPrev} disabled={tokenIdx === 0}
          className="rounded bg-gray-700 px-2 py-1 text-xs hover:bg-gray-600 disabled:opacity-30">‹</button>
        <button onClick={goNext} disabled={tokenIdx >= allTokens.length - 1}
          className="rounded bg-gray-700 px-2 py-1 text-xs hover:bg-gray-600 disabled:opacity-30">›</button>

        {/* Mode toggle */}
        <div className="flex overflow-hidden rounded border border-gray-600">
          {[MODE_SELECT, MODE_ADDBOX, MODE_LASSO].map(m => (
            <button key={m} onClick={() => setMode(m)}
              className={`px-2 py-1 text-xs ${mode === m ? 'bg-indigo-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}>
              {modeLabel[m]}
            </button>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-2">
          {!apiAvailable && <span className="text-xs text-yellow-400">API unavailable</span>}
          {saveFlash && <span className="font-semibold text-green-400">✓ Saved</span>}
          <span className="text-xs text-gray-400">{savedTotal} pairs</span>
          <button onClick={undo} disabled={strokesRef.current.length === 0}
            className="rounded bg-gray-700 px-2 py-1 text-xs hover:bg-gray-600 disabled:opacity-30"
            title="Undo stroke (Ctrl+Z)">↩ Undo</button>
          <button onClick={redo} disabled={redoRef.current.length === 0}
            className="rounded bg-gray-700 px-2 py-1 text-xs hover:bg-gray-600 disabled:opacity-30"
            title="Redo stroke (Ctrl+Shift+Z)">↪ Redo</button>
          {/* 4. Clear strokes only — explicit label so it's clear this does NOT remove boxes */}
          <button onClick={clearStrokes}
            className="rounded bg-gray-700 px-2 py-1 text-xs hover:bg-gray-600"
            title="Clear ink strokes only — does not remove token boxes">Clear strokes</button>
          <button onClick={() => window.open('/api/training-pairs/export', '_blank')}
            className="rounded bg-gray-700 px-2 py-1 text-xs hover:bg-gray-600">↓ Export</button>
        </div>
      </div>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left: manuscript + overlays */}
        <div className="relative flex-1 overflow-auto bg-gray-950 p-3">
          <img ref={hiddenImgRef} src={`/images/${pageId}`} onLoad={onHiddenImgLoad}
            className="hidden" crossOrigin="anonymous" alt="" />

          <div className="relative inline-block select-none">
            <img ref={pageImgRef} src={`/images/${pageId}`} onLoad={onPageImgLoad}
              className="block max-w-full" alt={pageId} draggable={false} />

            {/* SVG bbox overlays + resize handles for selected token */}
            <svg className="pointer-events-none absolute"
              style={{ top: 0, left: 0, width: canvasSize.w, height: canvasSize.h }}>
              {allTokens.map((tok, i) => {
                if (!tok.bbox) return null
                const [bx, by, bw, bh] = effectiveBbox(tok, imgNaturalH)
                const sel      = i === tokenIdx
                const isManual = tok.source === 'manual'
                const rx = bx * scale.x, ry = by * scale.y
                const rw = Math.max(bw * scale.x, 4), rh = Math.max(bh * scale.y, 4)
                return (
                  <g key={i}>
                    <rect x={rx} y={ry} width={rw} height={rh}
                      fill={sel ? 'rgba(239,68,68,0.20)' : isManual ? 'rgba(34,197,94,0.08)' : 'rgba(250,204,21,0.08)'}
                      stroke={sel ? '#ef4444' : isManual ? '#22c55e' : '#fbbf24'}
                      strokeWidth={sel ? 2.5 : 0.5} rx={1} />
                    {/* 3. Resize handles for selected token */}
                    {sel && getHandles(tok, imgNaturalH, scale.x, scale.y).map(h => (
                      <circle key={h.id} cx={h.x} cy={h.y} r={HANDLE_R}
                        fill="#fff" stroke="#ef4444" strokeWidth={2} />
                    ))}
                  </g>
                )
              })}
            </svg>

            {/* Drawing canvas */}
            <canvas ref={drawCanvasRef} width={canvasSize.w} height={canvasSize.h}
              className={`absolute ${cursorClass}`} style={{ top: 0, left: 0 }}
              onMouseDown={onMouseDown} onMouseMove={onMouseMove}
              onMouseUp={onMouseUp} onMouseLeave={onMouseUp} />
          </div>
        </div>

        {/* Right panel */}
        <aside className="flex w-72 flex-shrink-0 flex-col gap-4 overflow-y-auto border-l border-gray-700 bg-gray-800 p-4">
          {!token ? (
            <p className="text-sm text-gray-400">
              {allTokens.length === 0 ? 'Loading tokens…' : 'Click a word on the manuscript to select it.'}
            </p>
          ) : (
            <>
              {/* Word crop */}
              <div>
                <div className="mb-1 flex items-center justify-between">
                  <p className="text-[10px] font-semibold uppercase tracking-wide text-gray-400">
                    Word crop (4×){token.source === 'manual' ? ' · manual' : ''}
                  </p>
                  {/* 6. Delete button */}
                  <button onClick={deleteToken}
                    className="rounded bg-red-900 px-2 py-0.5 text-[10px] text-red-300 hover:bg-red-800"
                    title="Delete token (Del key)">✕ Delete</button>
                </div>
                <div className="overflow-hidden rounded bg-white">
                  <canvas ref={bgCanvasRef} className="block w-full"
                    style={{ imageRendering: 'pixelated' }} />
                </div>
              </div>

              {/* Label */}
              <div>
                <div className="mb-1 flex items-center justify-between">
                  <p className="text-[10px] font-semibold uppercase tracking-wide text-gray-400">Label</p>
                  {token.text && (
                    <button onClick={() => setLabel(token.text)}
                      className="text-[10px] text-indigo-400 hover:text-indigo-300">Use OCR ↑</button>
                  )}
                </div>
                {token.text && (
                  <p className="mb-1.5 text-[11px] text-gray-500">
                    OCR: <span dir="rtl" style={{ fontFamily: ARABIC_FONT }}>{token.text}</span>
                    {' '}({(token.confidence * 100).toFixed(0)}%)
                  </p>
                )}
                <input ref={labelRef} type="text" dir="rtl" lang="ar"
                  value={label} onChange={e => setLabel(e.target.value)}
                  className="w-full rounded border border-gray-600 bg-gray-700 px-3 py-2 text-lg text-white placeholder-gray-500 focus:border-indigo-400 focus:outline-none"
                  style={{ fontFamily: ARABIC_FONT }} placeholder="Arabic text…" />
              </div>

              {/* Action buttons */}
              <div className="flex gap-2">
                <button onClick={savePair} disabled={!label.trim()}
                  className="flex-1 rounded bg-indigo-600 px-3 py-2 text-sm font-semibold hover:bg-indigo-500 disabled:opacity-40">
                  💾 Save <span className="text-xs opacity-50">(S)</span>
                </button>
                <button onClick={goNext}
                  className="flex-1 rounded bg-gray-700 px-3 py-2 text-sm font-semibold hover:bg-gray-600">
                  Skip <span className="text-xs opacity-50">(Space)</span>
                </button>
              </div>

              {/* Shortcuts */}
              <div className="space-y-0.5 text-[10px] text-gray-500">
                <p>← / → : prev / next token</p>
                <p>S / Enter : save + advance · Space : skip</p>
                <p>Del : delete selected token</p>
                <p>Ctrl+Z / Ctrl+Shift+Z : undo / redo strokes</p>
                <p>Select: drag box or handle to move / resize</p>
              </div>
            </>
          )}
        </aside>
      </div>
    </div>
  )
}
