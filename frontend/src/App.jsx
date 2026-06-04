import { useState } from 'react'
import UploadZone from './components/UploadZone'
import PageSidebar from './components/PageSidebar'
import PageViewer from './components/PageViewer'
import PreprocessingControls from './components/PreprocessingControls'
import GroundTruthPanel from './components/GroundTruthPanel'
import WorkflowBar from './components/WorkflowBar'
import SimilarPagesPanel from './components/SimilarPagesPanel'

export default function App() {
  // null = upload screen; {docId, clusters} = document workspace
  const [doc, setDoc] = useState(null)

  function handleUploadSuccess(docId, clusters) {
    setDoc({ docId, clusters })
  }

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

  // Placeholder workspace — replaced component by component in Phase 3
  return (
    <div className="flex min-h-screen flex-col bg-gray-50 p-8">
      <div className="mb-6 rounded-xl border border-gray-200 bg-white px-6 py-4 shadow-sm">
        <p className="text-sm text-gray-500">Document loaded</p>
        <p className="font-mono text-sm text-gray-800">{doc.docId}</p>
        <p className="mt-1 text-sm text-gray-600">
          {doc.clusters.length} cluster{doc.clusters.length !== 1 ? 's' : ''} ·{' '}
          {doc.clusters.reduce((s, c) => s + c.page_count, 0)} page
          {doc.clusters.reduce((s, c) => s + c.page_count, 0) !== 1 ? 's' : ''}
        </p>
        <button
          className="mt-3 text-xs text-indigo-600 hover:underline"
          onClick={() => setDoc(null)}
        >
          ← Upload another document
        </button>
      </div>

      {/* WorkflowBar, PageSidebar, PageViewer etc. will slot in here */}
    </div>
  )
}

