const STEPS = [
  { id: 'upload',    label: 'Upload' },
  { id: 'cluster',   label: 'Cluster' },
  { id: 'calibrate', label: 'Calibrate' },
  { id: 'ocr',       label: 'OCR' },
  { id: 'review',    label: 'Review' },
  { id: 'export',    label: 'Export' },
]

export default function WorkflowBar({ currentStep: currentStepProp, hasOcrResults = false }) {
  const currentStep = currentStepProp ?? (hasOcrResults ? 'review' : 'calibrate')
  const currentIdx = STEPS.findIndex((s) => s.id === currentStep)

  return (
    <div className="flex-none bg-white px-4 py-2">
      <ol className="flex items-center">
        {STEPS.map((step, idx) => {
          const isDone    = idx < currentIdx
          const isCurrent = idx === currentIdx
          const isLast    = idx === STEPS.length - 1

          return (
            <li key={step.id} className="flex items-center">
              <div className="flex items-center gap-1.5">
                <span
                  className={[
                    'flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold leading-none',
                    isDone    ? 'bg-indigo-600 text-white'
                    : isCurrent ? 'bg-indigo-600 text-white ring-2 ring-indigo-200 ring-offset-1'
                    :             'bg-gray-200 text-gray-400',
                  ].join(' ')}
                >
                  {isDone ? '✓' : idx + 1}
                </span>
                <span
                  className={[
                    'text-xs font-medium',
                    isDone    ? 'text-indigo-600'
                    : isCurrent ? 'text-indigo-700'
                    :             'text-gray-400',
                  ].join(' ')}
                >
                  {step.label}
                </span>
              </div>

              {!isLast && (
                <div
                  className={[
                    'mx-3 h-px w-6 flex-shrink-0',
                    isDone ? 'bg-indigo-300' : 'bg-gray-200',
                  ].join(' ')}
                />
              )}
            </li>
          )
        })}
      </ol>
    </div>
  )
}
