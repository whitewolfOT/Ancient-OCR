import { useState } from 'react'
import UploadZone from './components/UploadZone'
import PageSidebar from './components/PageSidebar'
import PageViewer from './components/PageViewer'
import PreprocessingControls from './components/PreprocessingControls'
import GroundTruthPanel from './components/GroundTruthPanel'
import WorkflowBar from './components/WorkflowBar'
import SimilarPagesPanel from './components/SimilarPagesPanel'

export default function App() {
  const [doc, setDoc] = useState(null)               // { docId, clusters } | null
  const [selectedPageId, setSelectedPageId] = useState(null)
  const [selectedClusterId, setSelectedClusterId] = useState(null)
  const [preprocessedB64, setPreprocessedB64] = useState(null)
  // Incrementing this remounts PageSidebar, re-fetching page list with updated status icons
  const [sidebarRefreshKey, setSidebarRefreshKey] = useState(0)

  function handleUploadSuccess(docId, clusters) {
    setDoc({ docId, clusters })
    setSelectedPageId(null)
    setSelectedClusterId(null)
    setPreprocessedB64(null)
  }

  // PageSidebar calls onPageSelect(pageId, clusterId)
  function handlePageSelect(pageId, clusterId) {
    setSelectedPageId(pageId)
    setSelectedClusterId(clusterId)
    setPreprocessedB64(null)   // clear stale preview when navigating pages
  }

  function handlePreviewReady(b64 /*, settings, previewId */) {
    setPreprocessedB64(b64)
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
        </div>
      </div>
    )
  }

  // ── Document workspace ───────────────────────────────────────────────────
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-gray-50">
      <WorkflowBar currentStep="calibrate" />

      <div className="flex flex-1 overflow-hidden">
        {/* Left: scrollable page list — key remounts on GT save to refresh status icons */}
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
          <GroundTruthPanel
            pageId={selectedPageId}
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
