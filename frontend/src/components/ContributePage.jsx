import { useState, useEffect, useRef, useCallback } from 'react'

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const ARABIC_FONT = "'Amiri', 'Scheherazade New', 'Noto Naskh Arabic', serif"
const SESSION_KEY = 'ancient_ocr_contribute_session'

function getSessionToken() {
  let token = localStorage.getItem(SESSION_KEY)
  if (!token) {
    token = `web_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`
    localStorage.setItem(SESSION_KEY, token)
  }
  return token
}

export default function ContributePage() {
  const [sessionToken] = useState(getSessionToken)
  const [word, setWord] = useState(null)       // current ContributeWordResponse
  const [loading, setLoading] = useState(true)
  const [transcription, setTranscription] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const loadNextWord = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const r = await fetch(`${API}/api/contribute/next-word?session_token=${encodeURIComponent(sessionToken)}`)
      if (!r.ok) throw new Error('Failed to load next word')
      const data = await r.json()
      setWord(data)
      setTranscription(data.ocr_guess || '')
    } catch {
      setError('Could not reach the server. Please try again.')
    } finally {
      setLoading(false)
    }
  }, [sessionToken])

  useEffect(() => { loadNextWord() }, [loadNextWord])

  useEffect(() => {
    if (!loading && word && !word.done && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [loading, word])

  async function submit(skipped) {
    if (!word || !word.word_id || submitting) return
    setSubmitting(true)
    setError(null)
    try {
      const r = await fetch(`${API}/api/contribute/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          word_id: word.word_id,
          transcription: skipped ? '' : transcription,
          skipped,
          session_token: sessionToken,
        }),
      })
      if (!r.ok) throw new Error('Submit failed')
      await loadNextWord()
    } catch {
      setError('Submit failed. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Enter') {
      e.preventDefault()
      submit(false)
    }
  }

  const accuracyText = word && word.contributor_accuracy != null
    ? `${Math.round(word.contributor_accuracy * 100)}%`
    : '—'

  return (
    <div className="flex min-h-screen flex-col items-center bg-gray-50 px-4 py-8">
      <div className="w-full max-w-md">
        <h1 className="text-center text-xl font-bold text-gray-800">
          Ancient Arabic OCR — Help Transcribe
        </h1>
        <p className="mt-1 text-center text-sm text-gray-500">
          You've contributed: {word?.contributor_words_submitted ?? 0} words&nbsp;&nbsp;
          Accuracy: {accuracyText}
        </p>

        {error && (
          <div className="mt-4 rounded border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </div>
        )}

        {loading && (
          <div className="mt-10 text-center text-gray-400">Loading…</div>
        )}

        {!loading && word && word.done && (
          <div className="mt-10 rounded-lg border border-green-200 bg-green-50 p-6 text-center">
            <p className="text-lg font-semibold text-green-700">All caught up!</p>
            <p className="mt-1 text-sm text-green-600">
              There are no words waiting for transcription right now. Thank you for contributing.
            </p>
          </div>
        )}

        {!loading && word && !word.done && (
          <div className="mt-6 rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
            <div className="flex items-center justify-center rounded border border-gray-200 bg-gray-100 p-3" style={{ minHeight: 120 }}>
              {word.image_b64 ? (
                <img
                  src={`data:image/png;base64,${word.image_b64}`}
                  alt="Word to transcribe"
                  style={{ transform: 'scale(4)', maxHeight: 80 }}
                  className="select-none"
                  draggable={false}
                />
              ) : (
                <span className="text-sm text-gray-400">No image available</span>
              )}
            </div>

            {(word.context_before || word.context_after) && (
              <p className="mt-2 text-center text-xs text-gray-400" dir="rtl" style={{ fontFamily: ARABIC_FONT }}>
                {word.context_before} <span className="text-gray-300">[ ? ]</span> {word.context_after}
              </p>
            )}

            <label className="mt-4 block text-sm font-medium text-gray-700">
              What does this word say?
            </label>
            <input
              ref={inputRef}
              type="text"
              dir="rtl"
              lang="ar"
              inputMode="text"
              autoCapitalize="off"
              autoCorrect="off"
              spellCheck={false}
              value={transcription}
              onChange={(e) => setTranscription(e.target.value)}
              onKeyDown={onKeyDown}
              disabled={submitting}
              className="mt-1 w-full rounded border border-gray-300 px-3 py-3 text-xl focus:border-blue-400 focus:outline-none"
              style={{ fontFamily: ARABIC_FONT }}
              placeholder="اكتب الكلمة هنا"
            />

            <div className="mt-4 flex gap-2">
              <button
                onClick={() => submit(false)}
                disabled={submitting || !transcription.trim()}
                className="flex-1 rounded bg-blue-600 px-4 py-3 font-medium text-white disabled:opacity-50"
              >
                Submit (Enter)
              </button>
              <button
                onClick={() => submit(true)}
                disabled={submitting}
                className="rounded border border-gray-300 px-4 py-3 text-sm text-gray-600 disabled:opacity-50"
              >
                Can't read — Skip
              </button>
            </div>

            <p className="mt-3 text-center text-xs text-gray-400">
              Word {word.queue_position} of {word.queue_total} in queue
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
