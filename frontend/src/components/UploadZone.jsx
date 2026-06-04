import { useState, useRef, useCallback } from 'react'
import { uploadDocument } from '../api/client'

const ACCEPTED_TYPES = new Set([
  'application/pdf',
  'image/png',
  'image/jpeg',
  'image/tiff',
])
const ACCEPTED_EXTENSIONS = new Set(['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif'])

function isAccepted(file) {
  if (ACCEPTED_TYPES.has(file.type)) return true
  const ext = '.' + file.name.split('.').pop().toLowerCase()
  return ACCEPTED_EXTENSIONS.has(ext)
}

// Visual state machine: 'idle' | 'dragover' | 'uploading' | 'error'
export default function UploadZone({ onSuccess }) {
  const [phase, setPhase] = useState('idle')
  const [error, setError] = useState(null)
  const inputRef = useRef(null)
  const dragCounter = useRef(0)   // track nested drag-enter/leave pairs

  const handleFile = useCallback(async (file) => {
    if (!file) return

    if (!isAccepted(file)) {
      setError(`"${file.name}" is not supported. Please upload a PDF, PNG, JPG, or TIFF.`)
      setPhase('error')
      return
    }

    setPhase('uploading')
    setError(null)

    try {
      const result = await uploadDocument(file)
      setPhase('idle')
      onSuccess?.(result.doc_id, result.clusters)
    } catch (err) {
      const message =
        err?.response?.data?.detail ||
        err?.message ||
        'Upload failed. Please try again.'
      setError(message)
      setPhase('error')
    }
  }, [onSuccess])

  // --- drag handlers ---

  const onDragEnter = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounter.current += 1
    if (phase !== 'uploading') setPhase('dragover')
  }, [phase])

  const onDragLeave = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounter.current -= 1
    if (dragCounter.current === 0 && phase !== 'uploading') setPhase('idle')
  }, [phase])

  const onDragOver = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounter.current = 0
    if (phase === 'uploading') return
    const file = e.dataTransfer.files?.[0]
    handleFile(file)
  }, [phase, handleFile])

  // --- click-to-browse ---

  const onInputChange = useCallback((e) => {
    const file = e.target.files?.[0]
    // Reset so the same file can be re-selected after an error
    e.target.value = ''
    handleFile(file)
  }, [handleFile])

  const onZoneClick = useCallback(() => {
    if (phase !== 'uploading') inputRef.current?.click()
  }, [phase])

  // --- derived Tailwind classes ---

  const borderClass = {
    idle:      'border-gray-300 hover:border-indigo-400',
    dragover:  'border-indigo-500 bg-indigo-50',
    uploading: 'border-gray-300 cursor-default',
    error:     'border-red-400 bg-red-50',
  }[phase]

  const cursorClass = phase === 'uploading' ? 'cursor-default' : 'cursor-pointer'

  return (
    <div
      role="button"
      tabIndex={phase === 'uploading' ? -1 : 0}
      aria-label="Upload zone"
      onClick={onZoneClick}
      onKeyDown={(e) => e.key === 'Enter' && onZoneClick()}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
      className={[
        'flex flex-col items-center justify-center gap-4',
        'rounded-2xl border-2 border-dashed',
        'px-10 py-16 text-center transition-colors duration-150',
        borderClass,
        cursorClass,
        'select-none outline-none focus-visible:ring-2 focus-visible:ring-indigo-500',
      ].join(' ')}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.png,.jpg,.jpeg,.tiff,.tif"
        className="hidden"
        onChange={onInputChange}
      />

      {phase === 'uploading' ? (
        <UploadingState />
      ) : phase === 'error' ? (
        <ErrorState message={error} onRetry={() => { setPhase('idle'); setError(null) }} />
      ) : (
        <IdleState dragover={phase === 'dragover'} />
      )}
    </div>
  )
}

function IdleState({ dragover }) {
  return (
    <>
      <UploadIcon
        className={[
          'h-12 w-12 transition-colors duration-150',
          dragover ? 'text-indigo-500' : 'text-gray-400',
        ].join(' ')}
      />
      <div>
        <p className={[
          'text-lg font-medium transition-colors duration-150',
          dragover ? 'text-indigo-700' : 'text-gray-700',
        ].join(' ')}>
          Drop a PDF or image here
        </p>
        <p className="mt-1 text-sm text-gray-500">
          or <span className="font-medium text-indigo-600">click to browse</span>
        </p>
      </div>
      <p className="text-xs text-gray-400">PDF · PNG · JPG · TIFF</p>
    </>
  )
}

function UploadingState() {
  return (
    <>
      <Spinner className="h-10 w-10 text-indigo-500" />
      <div>
        <p className="text-lg font-medium text-gray-700">
          Uploading and clustering pages…
        </p>
        <p className="mt-1 text-sm text-gray-500">
          Computing perceptual hashes and grouping similar pages
        </p>
      </div>
    </>
  )
}

function ErrorState({ message, onRetry }) {
  return (
    <>
      <ErrorIcon className="h-10 w-10 text-red-400" />
      <div>
        <p className="text-lg font-medium text-red-700">Upload failed</p>
        <p className="mt-1 max-w-sm text-sm text-red-600">{message}</p>
      </div>
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); onRetry() }}
        className="rounded-lg bg-red-100 px-4 py-2 text-sm font-medium text-red-700
                   hover:bg-red-200 focus-visible:outline-none focus-visible:ring-2
                   focus-visible:ring-red-500 transition-colors"
      >
        Try again
      </button>
    </>
  )
}

// --- inline SVG icons (no external dep) ---

function UploadIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
    </svg>
  )
}

function ErrorIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
    </svg>
  )
}

function Spinner({ className }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}
