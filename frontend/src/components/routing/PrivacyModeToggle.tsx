'use client'

import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { PrivacyValue } from '@/lib/types/model-routing'
import {
  canInherit,
  PrivacyScope,
  resolveInherited,
} from '@/lib/utils/privacy-mode'
import { Cloud, Lock } from 'lucide-react'

interface PrivacyModeToggleProps {
  scope: PrivacyScope
  /** Human label for the scope (e.g. "Global default", "This notebook"). */
  scopeLabel: string
  value: PrivacyValue
  onChange: (value: PrivacyValue) => void
  /** Context used to compute the "inherits: <resolved>" hint. */
  globalDefault?: 'cloud' | 'private'
  notebookValue?: 'cloud' | 'private' | null
  disabled?: boolean
}

/**
 * Layered privacy selector reused at global / notebook / document scope.
 *
 * Renders an accessible radio group (Radix RadioGroup — Tab to focus, arrow
 * keys to move, Space/Enter to select) of Cloud / Private (+ Inherit for
 * notebook/document scopes). When set to Inherit it shows the resolved
 * effective value so the user knows what will actually run. AC4.
 */
export function PrivacyModeToggle({
  scope,
  scopeLabel,
  value,
  onChange,
  globalDefault = 'cloud',
  notebookValue = null,
  disabled = false,
}: PrivacyModeToggleProps) {
  const showInherit = canInherit(scope)
  const inheritedValue = resolveInherited(scope, {
    globalDefault,
    notebookValue,
  })

  const options: { value: PrivacyValue; label: string; icon: typeof Cloud }[] = [
    { value: 'cloud', label: 'Cloud', icon: Cloud },
    { value: 'private', label: 'Private', icon: Lock },
  ]
  if (showInherit) {
    options.push({ value: 'inherit', label: 'Inherit', icon: Cloud })
  }

  const groupId = `privacy-${scope}`

  return (
    <div className="space-y-2">
      <Label htmlFor={groupId} className="text-sm font-medium">
        {scopeLabel}
      </Label>
      <RadioGroup
        id={groupId}
        value={value}
        onValueChange={(v) => onChange(v as PrivacyValue)}
        disabled={disabled}
        aria-label={`Privacy mode for ${scopeLabel}`}
        className="flex flex-wrap gap-4"
      >
        {options.map((opt) => {
          const Icon = opt.icon
          const itemId = `${groupId}-${opt.value}`
          return (
            <div key={opt.value} className="flex items-center gap-2">
              <RadioGroupItem
                value={opt.value}
                id={itemId}
                aria-label={`${scopeLabel}: ${opt.label}`}
              />
              <Label
                htmlFor={itemId}
                className="flex items-center gap-1 text-sm font-normal cursor-pointer"
              >
                <Icon className="h-3.5 w-3.5" aria-hidden="true" />
                {opt.label}
              </Label>
            </div>
          )
        })}
      </RadioGroup>
      {value === 'inherit' && showInherit && (
        <p className="text-xs text-muted-foreground" data-testid="inherit-hint">
          inherits: {inheritedValue}
        </p>
      )}
    </div>
  )
}
