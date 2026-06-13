import { useState } from 'react'
import UploadZone from './components/UploadZone'
import PageSidebar from './components/PageSidebar'
import PageViewer from './components/PageViewer'
import PreprocessingControls from './components/PreprocessingControls'
import GroundTruthPanel from './components/GroundTruthPanel'
import WorkflowBar from './components/WorkflowBar'
import SimilarPagesPanel from './components/SimilarPagesPanel'
import ProfileSelector from './components/ProfileSelector'
import ReviewTab from './components/ReviewTab'
import AnnotationView from './components/AnnotationView'
import LineReviewView from './components/LineReviewView'
import { runPageOCR } from './api/client'

export default function App() {
  const [doc, setDoc] = useState(null)               // { docId, clusters } | null
  const [selectedPageId, setSelectedPageId] = useState(null)
  const [selectedClusterId, setSelectedClusterId] = useState(null)
  const [preprocessedB64, setPreprocessedB64] = useState(null)
  const [ocrTokens, setOcrTokens] = useState(null)   // null | token[]
  const [ocrRunning, setOcrRunning] = useState(false)
  const [profileName, setProfileName] = useState('default')
  const [activeView, setActiveView] = useState('workspace') // 'workspace' | 'review' | 'annotate' | 'line-review'
  // Incrementing this remounts PageSidebar, re-fetching page list with updated status icons
  const [sidebarRefreshKey, setSidebarRefreshKey] = useState(0)

  function handleUploadSuccess(docId, clusters) {
    setDoc({ docId, clusters })
    setSelectedPageId(null)
    setSelectedClusterId(null)
    setPreprocessedB64(null)
    setOcrTokens(null)
  }

  function handlePageSelect(pageId, clusterId) {
    setSelectedPageId(pageId)
    setSelectedClusterId(clusterId)
    setPreprocessedB64(null)
    setOcrTokens(null)        // clear stale OCR when navigating pages
  }

  function handlePreviewReady(b64) {
    setPreprocessedB64(b64)
  }

  async function handleRequestOCR() {
    if (!selectedPageId || ocrRunning) return
    setOcrRunning(true)
    try {
      const result = await runPageOCR(selectedPageId)
      setOcrTokens(result.tokens)
      setSidebarRefreshKey((k) => k + 1)   // refresh sidebar to show ocr_done status
    } catch (err) {
      console.error('OCR failed:', err)
    } finally {
      setOcrRunning(false)
    }
  }

  // ── Line review view ─────────────────────────────────────────────────────
  if (activeView === 'line-review') {
    return (
      <div className="flex h-screen flex-col overflow-hidden">
        <LineReviewView onBack={() => setActiveView('workspace')} />
      </div>
    )
  }

  // ── Annotate view ────────────────────────────────────────────────────────
  if (activeView === 'annotate') {
    return (
      <div className="flex h-screen flex-col overflow-hidden">
        <AnnotationView onBack={() => setActiveView('workspace')} />
      </div>
    )
  }

  // ── Review view (always available, no doc required) ─────────────────────
  if (activeView === 'review') {
    return (
      <div className="flex h-screen flex-col overflow-hidden bg-gray-50">
        <div className="flex-none border-b border-gray-200 bg-white px-4 py-2">
          <h1 className="text-sm font-semibold text-gray-700">OCR Results Review</h1>
        </div>
        <div className="flex-1 overflow-hidden">
          <ReviewTab onBack={() => setActiveView('workspace')} />
        </div>
      </div>
    )
  }

  // ── Upload screen ─────────────────────────────────────────────────────────
  if (!doc) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gray-50 p-8">
        <div className="w-full max-w-lg">
          <h1 className="mb-2 text-center text-2xl font-semibold text-gray-800">
            Ancient OCR Calibration
          </h1>
          <p className="mb-8 text-center text-sm text-gray-500">
            Upload a document to begin preprocessing calibration
          </p>
          <UploadZone onSuccess={handleUploadSuccess} />
          <div className="mt-4 flex justify-center gap-6">
            <button
              onClick={() => setActiveView('review')}
              className="text-sm text-indigo-600 hover:underline"
            >
              View OCR results →
            </button>
            <button
              onClick={() => setActiveView('annotate')}
              className="text-sm text-indigo-600 hover:underline"
            >
              ✏️ Annotate →
            </button>
            <button
              onClick={() => setActiveView('line-review')}
              className="text-sm text-indigo-600 hover:underline"
            >
              📝 Correct Lines →
            </button>
          </div>
        </div>
      </div>
    )
  }

  // ── Document workspace ───────────────────────────────────────────────────
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-gray-50">
      <div className="flex flex-none items-center border-b border-gray-200 bg-white">
        <div className="flex-1">
          <WorkflowBar hasOcrResults={ocrTokens !== null} />
        </div>
        <div className="mr-3 flex gap-2">
          <button
            onClick={() => setActiveView('review')}
            className="rounded bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700 hover:bg-indigo-100"
          >
            Review Results
          </button>
          <button
            onClick={() => setActiveView('annotate')}
            className="rounded bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700 hover:bg-indigo-100"
          >
            ✏️ Annotate
          </button>
          <button
            onClick={() => setActiveView('line-review')}
            className="rounded bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700 hover:bg-indigo-100"
          >
            📝 Correct Lines
          </button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Left: scrollable page list — key remounts on GT save or OCR done */}
        <PageSidebar
          key={sidebarRefreshKey}
          docId={doc.docId}
          selectedPageId={selectedPageId}
          onPageSelect={handlePageSelect}
        />

        {/* Center: viewer stacked above controls */}
        <div className="flex flex-1 flex-col overflow-hidden">
          <PageViewer
            pageId={selectedPageId}
            preprocessedImageB64={preprocessedB64}
            ocrTokens={ocrRunning ? null : ocrTokens}
            onRequestOCR={handleRequestOCR}
          />
          <PreprocessingControls
            pageId={selectedPageId}
            docId={doc.docId}
            clusterId={selectedClusterId}
            onPreviewReady={handlePreviewReady}
          />
        </div>

        {/* Right panel — hidden below xl breakpoint */}
        <div className="hidden w-64 flex-shrink-0 border-l border-gray-200 bg-white xl:flex xl:flex-col">
          <div className="border-b border-gray-200 p-3">
            <ProfileSelector
              pageId={selectedPageId}
              profileName={profileName}
              onProfileChange={setProfileName}
            />
          </div>
          <GroundTruthPanel
            pageId={selectedPageId}
            ocrTokens={ocrTokens}
            onSubmitted={() => setSidebarRefreshKey((k) => k + 1)}
          />
          <SimilarPagesPanel
            docId={doc.docId}
            currentPageId={selectedPageId}
            clusterId={selectedClusterId}
            onPageSelect={handlePageSelect}
          />
        </div>
      </div>
    </div>
  )
}
