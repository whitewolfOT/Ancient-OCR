import { useState, useEffect } from 'react'
import { listProfiles, suggestProfileForPage } from '../api/client'

/**
 * ProfileSelector — dropdown of OCR profiles + Suggest button.
 *
 * Props:
 *   pageId        — current page ID (needed for Suggest)
 *   profileName   — currently selected profile name
 *   onProfileChange — (name: string) => void
 */
export default function ProfileSelector({ pageId, profileName, onProfileChange }) {
  const [profiles, setProfiles] = useState([])
  const [suggesting, setSuggesting] = useState(false)
  const [suggestResult, setSuggestResult] = useState(null)  // {suggested_profile, confidence} | null
  const [error, setError] = useState(null)

  useEffect(() => {
    listProfiles()
      .then(setProfiles)
      .catch(() => setProfiles(['default']))
  }, [])

  // Clear suggestion when page changes
  useEffect(() => {
    setSuggestResult(null)
    setError(null)
  }, [pageId])

  async function handleSuggest() {
    if (!pageId || suggesting) return
    setSuggesting(true)
    setError(null)
    setSuggestResult(null)
    try {
      const result = await suggestProfileForPage(pageId)
      setSuggestResult(result)
      onProfileChange?.(result.suggested_profile)
    } catch {
      setError('Suggestion failed')
    } finally {
      setSuggesting(false)
    }
  }

  const confidencePct = suggestResult
    ? Math.round(suggestResult.confidence * 100)
    : null

  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-semibold uppercase tracking-wide text-gray-500">
        OCR Profile
      </label>

      <div className="flex items-center gap-2">
        {/* Profile dropdown */}
        <select
          value={profileName || 'default'}
          onChange={(e) => onProfileChange?.(e.target.value)}
          className="flex-1 rounded border border-gray-300 bg-white px-2 py-1.5 text-xs
                     text-gray-800 focus:outline-none focus:ring-1 focus:ring-indigo-400"
        >
          {profiles.length === 0 && (
            <option value="default">default</option>
          )}
          {profiles.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>

        {/* Suggest button */}
        <button
          onClick={handleSuggest}
          disabled={!pageId || suggesting}
          title="Suggest a profile based on this page's image characteristics"
          className={[
            'rounded border px-2.5 py-1.5 text-xs font-medium transition-colors',
            suggesting
              ? 'cursor-wait border-gray-200 bg-gray-50 text-gray-400'
              : !pageId
              ? 'cursor-not-allowed border-gray-200 bg-gray-50 text-gray-300'
              : 'border-indigo-300 bg-indigo-50 text-indigo-700 hover:bg-indigo-100',
          ].join(' ')}
        >
          {suggesting ? '…' : 'Suggest'}
        </button>
      </div>

      {/* Suggestion feedback */}
      {suggestResult && (
        <p className="text-[10px] text-indigo-600">
          Suggested <strong>{suggestResult.suggested_profile}</strong>
          {confidencePct !== null && ` (${confidencePct}% confidence)`}
        </p>
      )}
      {error && (
        <p className="text-[10px] text-red-500">{error}</p>
      )}
    </div>
  )
}
