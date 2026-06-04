import { useState, useEffect, useRef } from 'react'
import useDebounce from '../hooks/useDebounce'
import { previewPage, applyClusterSettings } from '../api/client'

const DEFAULTS = {
  clahe: 2.0,
  denoise: 3,
  deskew_threshold: 0.5,
  binarization: 'Adaptive',
}

/**
 * PreprocessingControls
 *
 * Debounces slider changes (300ms) then calls POST /pages/{pageId}/preview.
 * Verifies the response belongs to the most recently fired request before
 * calling onPreviewReady — stale responses are silently discarded.
 */
export default function PreprocessingControls({
  pageId,
  docId,
  clusterId,
  onPreviewReady,
}) {
  const [settings, setSettings] = useState(DEFAULTS)
  const [processing, setProcessing] = useState(false)
  const [processingMs, setProcessingMs] = useState(null)
  const [applyStatus, setApplyStatus] = useState(null) // null | 'applying' | 'done' | 'error'
  const [hasPreview, setHasPreview] = useState(false)

  // Monotonic counter — increments on every debounced call.
  // Response is only acted on when its captured counter matches the current one.
  const reqCounter = useRef(0)

  const debouncedSettings = useDebounce(settings, 300)

  // Reset hasPreview when page changes (require new preview for new page)
  useEffect(() => {
    setHasPreview(false)
    setProcessingMs(null)
    setApplyStatus(null)
  }, [pageId])

  // Fire preview call after debounce settles
  useEffect(() => {
    if (!pageId) return

    const myCount = ++reqCounter.current
    setProcessing(true)

    previewPage(pageId, debouncedSettings)
      .then((result) => {
        // Discard if a newer request has already been fired
        if (reqCounter.current !== myCount) return
        setProcessingMs(result.processing_time_ms)
        setHasPreview(true)
        setProcessing(false)
        onPreviewReady?.(
          result.preview_image_b64,
          result.settings_applied,
          result.preview_id,
        )
      })
      .catch(() => {
        if (reqCounter.current !== myCount) return
        setProcessing(false)
      })
  }, [debouncedSettings, pageId]) // eslint-disable-line react-hooks/exhaustive-deps

  async function handleApplyToCluster() {
    if (!hasPreview || !clusterId) return
    setApplyStatus('applying')
    try {
      await applyClusterSettings(docId, clusterId, settings)
      setApplyStatus('done')
      setTimeout(() => setApplyStatus(null), 2000)
    } catch {
      setApplyStatus('error')
      setTimeout(() => setApplyStatus(null), 2500)
    }
  }

  function update(key, value) {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <div className="flex-none border-t border-gray-200 bg-white px-4 py-3">
      <div className="flex items-start gap-6">
        {/* Sliders grid */}
        <div className="flex flex-1 flex-wrap gap-x-6 gap-y-3">
          {/* CLAHE */}
          <SliderField
            label="CLAHE"
            value={settings.clahe}
            min={1.0}
            max={8.0}
            step={0.1}
            display={settings.clahe.toFixed(1)}
            onChange={(v) => update('clahe', parseFloat(v))}
          />

          {/* Denoise — odd values only (step=2 starting at 1) */}
          <SliderField
            label="Denoise"
            value={settings.denoise}
            min={1}
            max={15}
            step={2}
            display={String(settings.denoise)}
            onChange={(v) => update('denoise', parseInt(v, 10))}
          />

          {/* Deskew threshold */}
          <SliderField
            label="Deskew"
            value={settings.deskew_threshold}
            min={0}
            max={5}
            step={0.5}
            display={`${settings.deskew_threshold}°`}
            onChange={(v) => update('deskew_threshold', parseFloat(v))}
          />

          {/* Binarization select */}
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-gray-600">Binarization</label>
            <select
              value={settings.binarization}
              onChange={(e) => update('binarization', e.target.value)}
              className="rounded border border-gray-300 bg-white px-2 py-1 text-xs text-gray-800
                         focus:outline-none focus:ring-1 focus:ring-indigo-400"
            >
              <option value="Adaptive">Adaptive</option>
              <option value="Global">Global</option>
              <option value="OTSU">OTSU</option>
            </select>
          </div>
        </div>

        {/* Right column: status + apply button */}
        <div className="flex flex-col items-end gap-2 self-end">
          {/* Processing indicator / timing */}
          <div className="h-4 text-right">
            {processing ? (
              <span className="text-xs text-gray-400">Processing…</span>
            ) : processingMs !== null ? (
              <span className="text-xs text-gray-400">{processingMs}ms</span>
            ) : null}
          </div>

          {/* Apply to cluster button */}
          <button
            onClick={handleApplyToCluster}
            disabled={!hasPreview || applyStatus === 'applying' || !clusterId}
            className={[
              'rounded-lg px-3 py-1.5 text-xs font-medium transition-colors',
              applyStatus === 'done'
                ? 'bg-green-100 text-green-700'
                : applyStatus === 'error'
                ? 'bg-red-100 text-red-700'
                : !hasPreview || !clusterId
                ? 'cursor-not-allowed bg-gray-100 text-gray-400'
                : 'bg-indigo-600 text-white hover:bg-indigo-700',
            ].join(' ')}
          >
            {applyStatus === 'applying'
              ? 'Applying…'
              : applyStatus === 'done'
              ? 'Applied ✓'
              : applyStatus === 'error'
              ? 'Failed'
              : 'Apply to cluster'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// SliderField
// ---------------------------------------------------------------------------

function SliderField({ label, value, min, max, step, display, onChange }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between gap-3">
        <label className="text-xs font-medium text-gray-600">{label}</label>
        <span className="min-w-[36px] text-right text-xs font-mono text-gray-800">
          {display}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-28 accent-indigo-600"
      />
    </div>
  )
}

