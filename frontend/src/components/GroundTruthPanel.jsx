import { useState, useEffect, useRef } from 'react'
import { getGroundTruth, submitGroundTruth } from '../api/client'

/**
 * GroundTruthPanel — RTL textarea for entering correct Arabic transcription.
 *
 * On pageId change: loads any saved ground truth and pre-populates.
 * Submit is disabled when text is empty or unchanged from the loaded value.
 * Calls onSubmitted() after a successful save.
 */
export default function GroundTruthPanel({ pageId, onSubmitted }) {
  const [text, setText] = useState('')
  const [savedText, setSavedText] = useState(null)   // null = never saved; '' or string = saved value
  const [lastSavedAt, setLastSavedAt] = useState(null)
  const [status, setStatus] = useState(null)         // null | 'saving' | 'saved' | 'error'
  const [loading, setLoading] = useState(false)
  const timerRef = useRef(null)

  useEffect(() => {
    if (!pageId) {
      setText('')
      setSavedText(null)
      setLastSavedAt(null)
      setStatus(null)
      return
    }

    let cancelled = false
    setLoading(true)
    setStatus(null)

    getGroundTruth(pageId)
      .then((gt) => {
        if (cancelled) return
        if (gt) {
          setText(gt.text)
          setSavedText(gt.text)
          setLastSavedAt(gt.submitted_at)
        } else {
          setText('')
          setSavedText(null)
          setLastSavedAt(null)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setText('')
          setSavedText(null)
          setLastSavedAt(null)
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
      clearTimeout(timerRef.current)
    }
  }, [pageId])

  async function handleSubmit() {
    if (!pageId || !text.trim() || text === savedText) return
    clearTimeout(timerRef.current)
    setStatus('saving')
    try {
      const result = await submitGroundTruth(pageId, text)
      setSavedText(text)
      setLastSavedAt(result.saved_at)
      setStatus('saved')
      timerRef.current = setTimeout(() => setStatus(null), 2000)
      onSubmitted?.()
    } catch {
      setStatus('error')
      timerRef.current = setTimeout(() => setStatus(null), 2500)
    }
  }

  const isUnchanged = text === savedText
  const isEmpty = !text.trim()
  const submitDisabled = !pageId || isEmpty || isUnchanged || status === 'saving' || loading

  return (
    <div className="flex flex-col border-b border-gray-200 bg-white p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">
          Ground Truth
        </span>
        {lastSavedAt && (
          <span className="text-[10px] text-gray-400" title={lastSavedAt}>
            Saved {_relativeTime(lastSavedAt)}
          </span>
        )}
      </div>

      {!pageId ? (
        <p className="text-xs text-gray-400">Select a page</p>
      ) : (
        <>
          <div className="relative">
            <textarea
              dir="rtl"
              lang="ar"
              rows={8}
              value={text}
              onChange={(e) => setText(e.target.value)}
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
            {/* Character count */}
            <span className="absolute bottom-1.5 left-2 text-[10px] text-gray-300 select-none">
              {text.length}
            </span>
          </div>

          <button
            onClick={handleSubmit}
            disabled={submitDisabled}
            className={[
              'mt-2 w-full rounded-lg py-1.5 text-xs font-medium transition-colors',
              status === 'saved'
                ? 'bg-green-100 text-green-700'
                : status === 'error'
                ? 'bg-red-100 text-red-700'
                : submitDisabled
                ? 'cursor-not-allowed bg-gray-100 text-gray-400'
                : 'bg-indigo-600 text-white hover:bg-indigo-700',
            ].join(' ')}
          >
            {status === 'saving'
              ? 'Saving…'
              : status === 'saved'
              ? 'Saved ✓'
              : status === 'error'
              ? 'Save failed'
              : 'Save ground truth'}
          </button>
        </>
      )}
    </div>
  )
}

function _relativeTime(isoString) {
  try {
    const diff = Date.now() - new Date(isoString).getTime()
    const mins = Math.floor(diff / 60_000)
    if (mins < 1) return 'just now'
    if (mins < 60) return `${mins}m ago`
    const hrs = Math.floor(mins / 60)
    if (hrs < 24) return `${hrs}h ago`
    return `${Math.floor(hrs / 24)}d ago`
  } catch {
    return ''
  }
}
