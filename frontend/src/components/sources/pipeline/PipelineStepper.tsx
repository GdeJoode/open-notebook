'use client'

import { cn } from '@/lib/utils'
import { Check, X, Minus } from 'lucide-react'

export type StepStatus = 'pending' | 'active' | 'completed' | 'running' | 'failed' | 'skipped' | 'locked'

export interface PipelineStep {
  id: number
  label: string
  phase: 'config' | 'pipeline'
  status: StepStatus
}

interface PipelineStepperProps {
  steps: PipelineStep[]
  activeTab: number
  onStepClick: (stepId: number) => void
}

export function PipelineStepper({ steps, activeTab, onStepClick }: PipelineStepperProps) {
  const configSteps = steps.filter(s => s.phase === 'config')
  const pipelineSteps = steps.filter(s => s.phase === 'pipeline')

  return (
    <div className="px-6 py-4 border-b border-border bg-muted/30 overflow-x-auto">
      <div className="flex items-center min-w-max gap-1">
        {/* Config phase */}
        <div className="flex items-center gap-1">
          {configSteps.map((step, index) => (
            <div key={step.id} className="flex items-center">
              <StepIndicator
                step={step}
                isActive={activeTab === step.id}
                onClick={() => onStepClick(step.id)}
              />
              {index < configSteps.length - 1 && (
                <StepConnector status={step.status} />
              )}
            </div>
          ))}
        </div>

        {/* Phase divider */}
        <div className="mx-3 h-8 w-px bg-border" />

        {/* Pipeline phase */}
        <div className="flex items-center gap-1">
          {pipelineSteps.map((step, index) => (
            <div key={step.id} className="flex items-center">
              <StepIndicator
                step={step}
                isActive={activeTab === step.id}
                onClick={() => onStepClick(step.id)}
              />
              {index < pipelineSteps.length - 1 && (
                <StepConnector status={step.status} />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function StepIndicator({
  step,
  isActive,
  onClick,
}: {
  step: PipelineStep
  isActive: boolean
  onClick: () => void
}) {
  const isClickable = step.status !== 'locked'
  const isRunning = step.status === 'running'

  return (
    <button
      type="button"
      onClick={isClickable ? onClick : undefined}
      disabled={!isClickable}
      className={cn(
        'flex items-center gap-2 px-3 py-2 rounded-md transition-colors text-sm whitespace-nowrap',
        isClickable ? 'cursor-pointer' : 'cursor-not-allowed',
        isActive && 'bg-background shadow-sm border border-border',
        !isActive && isClickable && 'hover:bg-background/50',
        !isClickable && 'opacity-50'
      )}
    >
      <StepCircle status={step.status} isActive={isActive} isRunning={isRunning} />
      <span
        className={cn(
          'font-medium',
          isActive ? 'text-foreground' : 'text-muted-foreground',
          step.status === 'completed' && 'text-green-600 dark:text-green-400',
          step.status === 'failed' && 'text-destructive'
        )}
      >
        {step.label}
      </span>
    </button>
  )
}

function StepCircle({
  status,
  isActive,
  isRunning,
}: {
  status: StepStatus
  isActive: boolean
  isRunning: boolean
}) {
  const baseClasses = 'flex items-center justify-center w-6 h-6 rounded-full border-2 text-xs font-medium transition-colors'

  if (status === 'completed') {
    return (
      <div className={cn(baseClasses, 'bg-green-600 border-green-600 text-white')}>
        <Check className="h-3.5 w-3.5" />
      </div>
    )
  }

  if (status === 'failed') {
    return (
      <div className={cn(baseClasses, 'bg-destructive border-destructive text-destructive-foreground')}>
        <X className="h-3.5 w-3.5" />
      </div>
    )
  }

  if (status === 'skipped') {
    return (
      <div className={cn(baseClasses, 'border-muted-foreground/40 text-muted-foreground')}>
        <Minus className="h-3.5 w-3.5" />
      </div>
    )
  }

  if (isRunning) {
    return (
      <div className={cn(baseClasses, 'border-primary bg-primary/10 text-primary')}>
        <div className="h-2.5 w-2.5 rounded-full bg-primary animate-pulse" />
      </div>
    )
  }

  if (isActive) {
    return (
      <div className={cn(baseClasses, 'border-primary text-primary bg-primary/10')}>
        <div className="h-2 w-2 rounded-full bg-primary" />
      </div>
    )
  }

  // locked / pending
  return (
    <div className={cn(baseClasses, 'border-muted-foreground/30 text-muted-foreground/50 bg-muted')}>
      <div className="h-1.5 w-1.5 rounded-full bg-muted-foreground/30" />
    </div>
  )
}

function StepConnector({ status }: { status: StepStatus }) {
  return (
    <div
      className={cn(
        'w-6 h-0.5 mx-0.5',
        status === 'completed' ? 'bg-green-600 dark:bg-green-400' : 'bg-border'
      )}
    />
  )
}
