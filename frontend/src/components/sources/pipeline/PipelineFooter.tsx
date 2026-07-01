'use client'

import { useRouter } from 'next/navigation'
import { Button } from '@/components/ui/button'

export type PipelinePhase = 'config' | 'processing' | 'complete' | 'error'

interface PipelineFooterProps {
  phase: PipelinePhase
  activeTab: number
  /** The last config-phase step; Submit shows here instead of Next. */
  lastConfigStep: number
  isStepValid: boolean
  isSubmitting: boolean
  sourceId?: string
  onBack: () => void
  onNext: () => void
  onSubmit: () => void
  onReset: () => void
}

export function PipelineFooter({
  phase,
  activeTab,
  lastConfigStep,
  isStepValid,
  isSubmitting,
  sourceId,
  onBack,
  onNext,
  onSubmit,
  onReset,
}: PipelineFooterProps) {
  const router = useRouter()

  // Complete phase
  if (phase === 'complete') {
    return (
      <div className="flex justify-between items-center px-6 py-4 border-t border-border bg-muted/30">
        <div />
        <div className="flex gap-2">
          <Button variant="outline" onClick={onReset}>
            Create Another
          </Button>
          {sourceId && (
            <Button onClick={() => router.push(`/sources/${sourceId}`)}>
              View Source
            </Button>
          )}
        </div>
      </div>
    )
  }

  // Processing phase
  if (phase === 'processing') {
    return (
      <div className="flex justify-between items-center px-6 py-4 border-t border-border bg-muted/30">
        <div />
        <Button
          variant="outline"
          onClick={() => router.push('/sources')}
        >
          View in Background
        </Button>
      </div>
    )
  }

  // Config phase
  const isLastConfigStep = activeTab === lastConfigStep
  return (
    <div className="flex justify-between items-center px-6 py-4 border-t border-border bg-muted/30">
      <Button
        variant="outline"
        onClick={() => router.push('/sources')}
      >
        Cancel
      </Button>

      <div className="flex gap-2">
        {activeTab > 1 && (
          <Button variant="outline" onClick={onBack}>
            Back
          </Button>
        )}

        {!isLastConfigStep && (
          <Button
            onClick={onNext}
            disabled={!isStepValid}
          >
            Next
          </Button>
        )}

        {isLastConfigStep && (
          <Button
            onClick={onSubmit}
            disabled={!isStepValid || isSubmitting}
            className="min-w-[120px]"
          >
            {isSubmitting ? 'Submitting...' : 'Submit'}
          </Button>
        )}
      </div>
    </div>
  )
}
