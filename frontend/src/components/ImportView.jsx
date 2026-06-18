import { useState, useEffect, useRef, useCallback } from 'react'
import { listProfiles } from '../api/client'

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const ZOOM = 3
const HANDLE_HIT = 9

// ── Geometry helpers (all in ORIGINAL image coordinates unless noted) ──────

function rectFromBbox([x, y, w, h]) {
  return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
}

function getCornerHandles(bbox, sx, sy) {
  const [x, y, w, h] = bbox
  return [
    { id: 'nw', x: x * sx,       y: y * sy       },
    { id: 'ne', x: (x + w) * sx, y: y * sy       },
    { id: 'se', x: (x + w) * sx, y: (y + h) * sy },
    { id: 'sw', x: x * sx,       y: (y + h) * sy },
  ]
}

function applyCornerDrag(orig, handle, dx, dy) {
  const [x, y, w, h] = orig
  const mw = (v) => Math.max(8, v)
  switch (handle) {
    case 'move': return [x + dx, y + dy, w, h]
    case 'nw':   return [x + dx, y + dy, mw(w - dx), mw(h - dy)]
    case 'ne':   return [x,      y + dy, mw(w + dx), mw(h - dy)]
    case 'se':   return [x,      y,      mw(w + dx), mw(h + dy)]
    case 'sw':   return [x + dx, y,      mw(w - dx), mw(h + dy)]
    default:     return orig
  }
}

function pageImageUrl(sessionId, pageId, preprocessed, profile) {
  const params = new URLSearchParams({ preprocessed: preprocessed ? 'true' : 'false' })
  if (preprocessed) params.set('profile', profile)
  return `${API}/api/import/page-image/${sessionId}/${pageId}?${params.toString()}`
}

function statusIcon(status) {
  if (status === 'accepted') return { icon: '✓', color: 'text-green-600' }
  if (status === 'segmented') return { icon: '◑', color: 'text-yellow-600' }
  return { icon: '○', color: 'text-gray-400' }
}

function lineColor(line, isSelected) {
  if (isSelected) return { fill: 'rgba(59,130,246,0.20)', stroke: '#3b82f6' }
  if (line._status === 'accepted') return { fill: 'rgba(34,197,94,0.15)', stroke: '#22c55e' }
  if (line._status === 'rejected') return { fill: 'rgba(239,68,68,0.10)', stroke: '#ef4444' }
  return { fill: 'rgba(156,163,175,0.10)', stroke: '#9ca3af' }
}

export default function ImportView({ onBack, onNavigateLineReview }) {
  const [session, setSession]           = useState(null) // { session_id, page_count, page_ids }
  const [pageStatus, setPageStatus]      = useState({})   // page_id -> 'none'|'segmented'|'accepted'
  const [currentPageId, setCurrentPageId] = useState(null)
  const [viewMode, setViewMode]          = useState('raw') // 'raw' | 'preprocessed'
  const [profiles, setProfiles]          = useState([])
  const [profileName, setProfileName]    = useState('default')
  const [lines, setLines]                = useState([])
  const [selectedLineId, setSelectedLineId] = useState(null)
  const [mode, setMode]                  = useState('select') // 'select' | 'addbox'
  const [uploading, setUploading]        = useState(false)
  const [segmenting, setSegmenting]      = useState(false)
  const [accepting, setAccepting]        = useState(false)
  const [toast, setToast]                = useState(null)   // { text, kind, action }
  const [scale, setScale]                = useState({ x: 1, y: 1 })
  const [canvasSize, setCanvasSize]      = useState({ w: 1, h: 1 })
  const [addBoxDraft, setAddBoxDraft]    = useState(null)

  const fileInputRef     = useRef(null)
  const pageImgRef       = useRef(null)
  const hiddenImgRef     = useRef(null)
  const overlayRef       = useRef(null)
  const cropCanvasRef    = useRef(null)
  const dragStateRef     = useRef(null)
  const addBoxStartRef   = useRef(null)
  const pendingClickRef  = useRef(false)

  // ── Load profile list once ────────────────────────────────────────────────
  useEffect(() => {
    listProfiles().then(setProfiles).catch(() => setProfiles(['default']))
  }, [])

  // ── Reset selection when the current page changes (render-time adjustment,
  //    not an effect — see https://react.dev/learn/you-might-not-need-an-effect) ──
  const [prevPageId, setPrevPageId] = useState(currentPageId)
  if (currentPageId !== prevPageId) {
    setPrevPageId(currentPageId)
    setSelectedLineId(null)
  }

  // ── Load saved segmentation when the current page changes ──────────────────
  useEffect(() => {
    if (!session || !currentPageId) { setLines([]); return }
    let cancelled = false
    fetch(`${API}/api/import/segments/${session.session_id}/${currentPageId}`)
      .then(r => (r.ok ? r.json() : null))
      .then(data => {
        if (cancelled) return
        if (data) {
          setLines(data.lines.map(l => ({ ...l, _status: 'pending' })))
          setPageStatus(prev => ({
            ...prev,
            [currentPageId]: prev[currentPageId] === 'accepted' ? 'accepted' : 'segmented',
          }))
        } else {
          setLines([])
        }
      })
      .catch(() => { if (!cancelled) setLines([]) })
    return () => { cancelled = true }
  }, [session, currentPageId])

  // ── Toast auto-dismiss ───────────────────────────────────────────────────
  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 6000)
    return () => clearTimeout(t)
  }, [toast])

  // ── Upload ────────────────────────────────────────────────────────────────
  async function handleFilesSelected(e) {
    const files = Array.from(e.target.files || [])
    e.target.value = ''
    if (files.length === 0) return
    setUploading(true)
    try {
      const isPdf = files.length === 1 && files[0].name.toLowerCase().endsWith('.pdf')
      let data
      if (isPdf) {
        const form = new FormData()
        form.append('file', files[0])
        form.append('profile_name', profileName)
        const r = await fetch(`${API}/api/import/upload-pdf`, { method: 'POST', body: form })
        if (!r.ok) throw new Error('upload-pdf failed')
        data = await r.json()
      } else {
        const form = new FormData()
        files.forEach(f => form.append('files', f))
        if (session?.session_id) form.append('session_id', session.session_id)
        const r = await fetch(`${API}/api/import/upload-images`, { method: 'POST', body: form })
        if (!r.ok) throw new Error('upload-images failed')
        data = await r.json()
      }
      setSession(prev => {
        if (prev && prev.session_id === data.session_id) {
          return { ...prev, page_count: data.page_count, page_ids: [...prev.page_ids, ...data.page_ids] }
        }
        return data
      })
      if (data.page_ids.length > 0) setCurrentPageId(data.page_ids[0])
    } catch {
      setToast({ text: 'Upload failed', kind: 'error' })
    } finally {
      setUploading(false)
    }
  }

  // ── Segment ──────────────────────────────────────────────────────────────
  async function handleSegment() {
    if (!session || !currentPageId || segmenting) return
    setSegmenting(true)
    try {
      const r = await fetch(
        `${API}/api/import/segment/${session.session_id}/${currentPageId}?profile=${encodeURIComponent(profileName)}`,
        { method: 'POST' }
      )
      if (!r.ok) throw new Error('segment failed')
      const data = await r.json()
      setLines(data.lines.map(l => ({ ...l, _status: 'pending' })))
      setPageStatus(prev => ({ ...prev, [currentPageId]: 'segmented' }))
      setSelectedLineId(null)
    } catch {
      setToast({ text: 'Segmentation failed', kind: 'error' })
    } finally {
      setSegmenting(false)
    }
  }

  // ── Accept ───────────────────────────────────────────────────────────────
  const handleAccept = useCallback(async (linesToSend) => {
    if (!session || !currentPageId || !linesToSend.length || accepting) return
    setAccepting(true)
    try {
      const payload = {
        lines: linesToSend.map(({ id, bbox, baseline, boundary }) => ({ id, bbox, baseline, boundary })),
        profile_name: profileName,
      }
      const r = await fetch(`${API}/api/import/accept-lines/${session.session_id}/${currentPageId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!r.ok) throw new Error('accept failed')
      const data = await r.json()
      setPageStatus(prev => ({ ...prev, [currentPageId]: 'accepted' }))
      setToast({ text: `${data.saved_lines} lines queued in Line Review`, kind: 'success', action: 'line-review' })
    } catch {
      setToast({ text: 'Accept failed', kind: 'error' })
    } finally {
      setAccepting(false)
    }
  }, [session, currentPageId, profileName, accepting])

  const acceptedCount = lines.filter(l => l._status === 'accepted').length

  // ── Scale tracking ───────────────────────────────────────────────────────
  function onPageImgLoad() {
    const img = pageImgRef.current
    if (!img || !img.naturalWidth) return
    const rect = img.getBoundingClientRect()
    setScale({ x: rect.width / img.naturalWidth, y: rect.height / img.naturalHeight })
    setCanvasSize({ w: Math.round(rect.width), h: Math.round(rect.height) })
  }
  useEffect(() => {
    window.addEventListener('resize', onPageImgLoad)
    return () => window.removeEventListener('resize', onPageImgLoad)
  }, [])

  // ── Line crop preview (3×) ───────────────────────────────────────────────
  const drawLineCrop = useCallback(() => {
    const canvas = cropCanvasRef.current
    const img = hiddenImgRef.current
    const line = lines.find(l => l.id === selectedLineId)
    if (!canvas || !img?.complete || !line) return
    const [x, y, w, h] = line.bbox
    if (w <= 0 || h <= 0) return
    canvas.width = Math.max(w * ZOOM, 80)
    canvas.height = Math.max(h * ZOOM, 40)
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, x, y, w, h, 0, 0, canvas.width, canvas.height)
  }, [lines, selectedLineId])

  useEffect(() => { drawLineCrop() }, [drawLineCrop])

  // ── Overlay mouse handling (select / resize / move / add-box) ──────────────
  function overlayPos(e) {
    const rect = overlayRef.current.getBoundingClientRect()
    return { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }

  function onOverlayMouseDown(e) {
    const pos = overlayPos(e)
    if (mode === 'addbox') {
      addBoxStartRef.current = pos
      setAddBoxDraft({ x: pos.x, y: pos.y, w: 0, h: 0 })
      return
    }
    const sel = lines.find(l => l.id === selectedLineId)
    if (sel) {
      const handles = getCornerHandles(sel.bbox, scale.x, scale.y)
      for (const h of handles) {
        if (Math.abs(pos.x - h.x) <= HANDLE_HIT && Math.abs(pos.y - h.y) <= HANDLE_HIT) {
          dragStateRef.current = {
            handle: h.id,
            startOrigX: pos.x / scale.x,
            startOrigY: pos.y / scale.y,
            origBbox: [...sel.bbox],
            lineId: sel.id,
          }
          return
        }
      }
      const [bx, by, bw, bh] = sel.bbox
      const ox = pos.x / scale.x, oy = pos.y / scale.y
      if (ox >= bx && ox <= bx + bw && oy >= by && oy <= by + bh) {
        dragStateRef.current = {
          handle: 'move',
          startOrigX: ox, startOrigY: oy,
          origBbox: [bx, by, bw, bh],
          lineId: sel.id,
        }
        return
      }
    }
    pendingClickRef.current = true
  }

  function onOverlayMouseMove(e) {
    const pos = overlayPos(e)
    if (mode === 'addbox' && addBoxStartRef.current) {
      const st = addBoxStartRef.current
      setAddBoxDraft({
        x: Math.min(st.x, pos.x), y: Math.min(st.y, pos.y),
        w: Math.abs(pos.x - st.x), h: Math.abs(pos.y - st.y),
      })
      return
    }
    if (dragStateRef.current) {
      const ds = dragStateRef.current
      const dx = pos.x / scale.x - ds.startOrigX
      const dy = pos.y / scale.y - ds.startOrigY
      const newBbox = applyCornerDrag(ds.origBbox, ds.handle, dx, dy).map(v => Math.round(v))
      setLines(prev => prev.map(l => (
        l.id === ds.lineId ? { ...l, bbox: newBbox, boundary: rectFromBbox(newBbox) } : l
      )))
    }
  }

  function onOverlayMouseUp(e) {
    if (mode === 'addbox' && addBoxStartRef.current) {
      const draft = addBoxDraft
      addBoxStartRef.current = null
      setAddBoxDraft(null)
      if (draft && draft.w > 6 && draft.h > 6) {
        const ox = draft.x / scale.x, oy = draft.y / scale.y
        const ow = draft.w / scale.x, oh = draft.h / scale.y
        const bbox = [Math.round(ox), Math.round(oy), Math.round(ow), Math.round(oh)]
        const newLine = {
          id: `manual_${Date.now()}`,
          bbox,
          boundary: rectFromBbox(bbox),
          baseline: [[bbox[0], bbox[1] + bbox[3]], [bbox[0] + bbox[2], bbox[1] + bbox[3]]],
          _status: 'pending',
        }
        setLines(prev => [...prev, newLine])
        setSelectedLineId(newLine.id)
      }
      setMode('select')
      return
    }
    if (dragStateRef.current) { dragStateRef.current = null; return }
    if (pendingClickRef.current) {
      pendingClickRef.current = false
      const pos = overlayPos(e)
      const ox = pos.x / scale.x, oy = pos.y / scale.y
      const hit = lines.find(l => {
        const [bx, by, bw, bh] = l.bbox
        return ox >= bx && ox <= bx + bw && oy >= by && oy <= by + bh
      })
      setSelectedLineId(hit ? hit.id : null)
    }
  }

  // ── Delete key ───────────────────────────────────────────────────────────
  useEffect(() => {
    function onKey(e) {
      if (e.key === 'Delete' && selectedLineId) {
        setLines(prev => prev.filter(l => l.id !== selectedLineId))
        setSelectedLineId(null)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selectedLineId])

  const selectedLine = lines.find(l => l.id === selectedLineId) ?? null
  const imgSrc = currentPageId ? pageImageUrl(session.session_id, currentPageId, viewMode === 'preprocessed', profileName) : null

  return (
    <div className="flex h-full flex-col overflow-hidden bg-gray-50">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex flex-none items-center justify-between border-b border-gray-200 bg-white px-4 py-2">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="rounded px-2 py-1 text-sm text-gray-600 hover:bg-gray-100">
            ← Back
          </button>
          <h1 className="text-sm font-semibold text-gray-700">Import Manuscript Pages</h1>
        </div>
        <div>
          <input
            ref={fileInputRef} type="file" multiple accept=".pdf,image/*"
            className="hidden" onChange={handleFilesSelected}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="rounded bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
          >
            {uploading ? 'Uploading & extracting…' : '📥 Upload PDF/Images'}
          </button>
        </div>
      </div>

      {/* ── Toast ──────────────────────────────────────────────────────── */}
      {toast && (
        <div className={`flex items-center justify-between px-4 py-1.5 text-xs ${
          toast.kind === 'error' ? 'bg-red-50 text-red-700' : 'bg-green-50 text-green-700'
        }`}>
          <span>{toast.text}</span>
          {toast.action === 'line-review' && (
            <button
              onClick={() => onNavigateLineReview?.()}
              className="font-semibold underline hover:no-underline"
            >
              View in Line Review →
            </button>
          )}
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        {/* ── Left: page list ──────────────────────────────────────────── */}
        <div className="w-52 flex-shrink-0 overflow-y-auto border-r border-gray-200 bg-white">
          {!session ? (
            <p className="p-3 text-xs text-gray-400">Upload a PDF or images to begin.</p>
          ) : (
            session.page_ids.map(pid => {
              const { icon, color } = statusIcon(pageStatus[pid] || 'none')
              const isCurrent = pid === currentPageId
              return (
                <button
                  key={pid}
                  onClick={() => setCurrentPageId(pid)}
                  className={`flex w-full items-center gap-2 px-3 py-2 text-left text-xs ${
                    isCurrent ? 'bg-indigo-50 font-semibold text-indigo-700' : 'text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  <span className={color}>{icon}</span>
                  {pid}
                </button>
              )
            })
          )}
        </div>

        {/* ── Center: page view ────────────────────────────────────────── */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {!currentPageId ? (
            <div className="flex flex-1 items-center justify-center text-sm text-gray-400">
              No page loaded yet.
            </div>
          ) : (
            <>
              <div className="flex flex-none flex-wrap items-center gap-2 border-b border-gray-200 bg-white px-3 py-2 text-xs">
                <div className="flex overflow-hidden rounded border border-gray-300">
                  {['raw', 'preprocessed'].map(m => (
                    <button
                      key={m}
                      onClick={() => setViewMode(m)}
                      className={`px-2 py-1 ${viewMode === m ? 'bg-indigo-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'}`}
                    >
                      {m === 'raw' ? 'Raw' : 'Preprocessed'}
                    </button>
                  ))}
                </div>
                <select
                  value={profileName}
                  onChange={e => setProfileName(e.target.value)}
                  className="rounded border border-gray-300 px-2 py-1"
                >
                  {(profiles.length ? profiles : ['default']).map(p => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>

                <div className="ml-auto flex gap-2">
                  <button
                    onClick={handleSegment}
                    disabled={segmenting}
                    className="rounded bg-gray-700 px-2.5 py-1 text-white hover:bg-gray-600 disabled:opacity-50"
                  >
                    {segmenting ? 'Segmenting…' : 'Auto-segment page'}
                  </button>
                  <button
                    onClick={() => setMode(m => (m === 'addbox' ? 'select' : 'addbox'))}
                    className={`rounded px-2.5 py-1 ${mode === 'addbox' ? 'bg-green-600 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}
                  >
                    {mode === 'addbox' ? 'Drawing… (drag on image)' : '+ Add line box'}
                  </button>
                  <button
                    onClick={() => handleAccept(lines.filter(l => l._status !== 'rejected'))}
                    disabled={accepting || lines.length === 0}
                    className="rounded bg-blue-600 px-2.5 py-1 text-white hover:bg-blue-700 disabled:opacity-50"
                  >
                    Accept all
                  </button>
                  <button
                    onClick={() => handleAccept(lines.filter(l => l._status === 'accepted'))}
                    disabled={accepting || acceptedCount === 0}
                    className="rounded bg-blue-100 px-2.5 py-1 text-blue-700 hover:bg-blue-200 disabled:opacity-50"
                  >
                    Accept selected ({acceptedCount})
                  </button>
                </div>
              </div>

              <div className="relative flex-1 overflow-auto bg-gray-950 p-3">
                <img
                  ref={hiddenImgRef} src={imgSrc} className="hidden" crossOrigin="anonymous" alt=""
                  onLoad={drawLineCrop}
                />
                <div className="relative inline-block select-none">
                  <img
                    ref={pageImgRef} src={imgSrc} alt={currentPageId}
                    className="block max-w-full" draggable={false} onLoad={onPageImgLoad}
                  />
                  <svg
                    className="pointer-events-none absolute"
                    style={{ top: 0, left: 0, width: canvasSize.w, height: canvasSize.h }}
                  >
                    {lines.map(line => {
                      const [bx, by, bw, bh] = line.bbox
                      const isSelected = line.id === selectedLineId
                      const { fill, stroke } = lineColor(line, isSelected)
                      const rx = bx * scale.x, ry = by * scale.y
                      const rw = Math.max(bw * scale.x, 4), rh = Math.max(bh * scale.y, 4)
                      return (
                        <g key={line.id}>
                          <rect x={rx} y={ry} width={rw} height={rh} fill={fill} stroke={stroke} strokeWidth={isSelected ? 2.5 : 1.5} />
                          {isSelected && getCornerHandles(line.bbox, scale.x, scale.y).map(h => (
                            <circle key={h.id} cx={h.x} cy={h.y} r={5} fill="#fff" stroke="#3b82f6" strokeWidth={2} />
                          ))}
                        </g>
                      )
                    })}
                    {addBoxDraft && (
                      <rect
                        x={addBoxDraft.x} y={addBoxDraft.y} width={addBoxDraft.w} height={addBoxDraft.h}
                        fill="none" stroke="#22c55e" strokeWidth={2} strokeDasharray="5,4"
                      />
                    )}
                  </svg>
                  <div
                    ref={overlayRef}
                    className="absolute"
                    style={{
                      top: 0, left: 0, width: canvasSize.w, height: canvasSize.h,
                      cursor: mode === 'addbox' ? 'crosshair' : 'default',
                    }}
                    onMouseDown={onOverlayMouseDown}
                    onMouseMove={onOverlayMouseMove}
                    onMouseUp={onOverlayMouseUp}
                    onMouseLeave={onOverlayMouseUp}
                  />
                </div>
              </div>
            </>
          )}
        </div>

        {/* ── Right: selected line panel ───────────────────────────────── */}
        <aside className="flex w-72 flex-shrink-0 flex-col gap-3 overflow-y-auto border-l border-gray-200 bg-white p-4">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-gray-500">Selected line</h2>
          {!selectedLine ? (
            <p className="text-sm text-gray-400">Click a line on the page to select it.</p>
          ) : (
            <>
              <div className="overflow-x-auto rounded border border-gray-200 bg-white p-2">
                <canvas ref={cropCanvasRef} className="block" style={{ imageRendering: 'pixelated' }} />
              </div>
              <p className="text-[10px] text-gray-400">
                bbox: [{selectedLine.bbox.join(', ')}]
              </p>
              <div className="flex gap-2">
                <button
                  onClick={() => setLines(prev => prev.map(l => l.id === selectedLineId ? { ...l, _status: 'accepted' } : l))}
                  className="flex-1 rounded bg-green-600 px-2 py-1.5 text-xs font-medium text-white hover:bg-green-700"
                >
                  ✓ Accept
                </button>
                <button
                  onClick={() => setLines(prev => prev.map(l => l.id === selectedLineId ? { ...l, _status: 'rejected' } : l))}
                  className="flex-1 rounded border border-red-200 px-2 py-1.5 text-xs font-medium text-red-600 hover:bg-red-50"
                >
                  ✗ Reject
                </button>
              </div>
              <button
                onClick={() => { setLines(prev => prev.filter(l => l.id !== selectedLineId)); setSelectedLineId(null) }}
                className="rounded border border-gray-200 px-2 py-1.5 text-xs text-gray-500 hover:bg-gray-50"
              >
                🗑 Delete (Del)
              </button>
            </>
          )}

          <div className="mt-auto space-y-0.5 text-[10px] text-gray-400">
            <p>Drag a corner handle to resize · drag inside to move</p>
            <p>Del : delete selected line</p>
            <p>+ Add line box, then drag on the image to draw</p>
          </div>
        </aside>
      </div>
    </div>
  )
}
