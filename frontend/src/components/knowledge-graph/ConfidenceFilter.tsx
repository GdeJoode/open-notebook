'use client'

import * as React from 'react'

import { Slider } from '@/components/ui/slider'

const STORAGE_KEY = 'kg_confidence_threshold'

/**
 * Read the persisted threshold from localStorage on the client.
 *
 * Returns `0` on the server (where `localStorage` is unavailable) and
 * also when the stored value is missing or malformed. Centralised so
 * the hook below and any caller that needs the initial value before
 * mount share the same parse logic.
 */
function readStoredThreshold(): number {
  if (typeof window === 'undefined') return 0
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (raw === null) return 0
    const parsed = parseFloat(raw)
    if (!Number.isFinite(parsed)) return 0
    if (parsed < 0) return 0
    if (parsed > 1) return 1
    return parsed
  } catch {
    return 0
  }
}

export interface ConfidenceFilterProps {
  /**
   * Notified whenever the slider value changes. The parent owns the
   * canonical threshold used to filter entity tiles; this component only
   * persists it.
   */
  onChange: (value: number) => void
  /** Optional className applied to the outer wrapper. */
  className?: string
}

/**
 * Slider that lets the user hide low-confidence entities.
 *
 * Range 0.0 – 1.0, step 0.05. The value is persisted to localStorage
 * under `kg_confidence_threshold` and restored on mount, so the choice
 * survives reloads and tab restores.
 */
export function ConfidenceFilter({ onChange, className }: ConfidenceFilterProps) {
  // SSR-safe initial state: always `0` on the server so the markup
  // matches between server- and client-render. The effect below
  // upgrades to the persisted value after hydration and fires
  // `onChange` so the page picks it up.
  const [value, setValue] = React.useState<number>(0)
  const [hydrated, setHydrated] = React.useState(false)

  React.useEffect(() => {
    const initial = readStoredThreshold()
    setValue(initial)
    setHydrated(true)
    if (initial > 0) {
      // Only nudge the parent if the persisted value is non-default —
      // saves an unnecessary re-render on first paint.
      onChange(initial)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleChange = React.useCallback(
    (next: number[]) => {
      const v = next[0] ?? 0
      setValue(v)
      onChange(v)
      try {
        window.localStorage.setItem(STORAGE_KEY, String(v))
      } catch {
        // Swallow — private-mode browsers reject writes. The filter
        // still works for the current session; it just won't persist.
      }
    },
    [onChange],
  )

  return (
    <div
      className={`flex items-center gap-3 ${className ?? ''}`}
      data-testid="confidence-filter"
      data-hydrated={hydrated ? 'true' : 'false'}
    >
      <label
        htmlFor="kg-confidence-filter"
        className="text-sm text-muted-foreground whitespace-nowrap"
      >
        Hide entities below: {value.toFixed(2)}
      </label>
      <Slider
        id="kg-confidence-filter"
        data-testid="confidence-filter-slider"
        aria-label="Hide entities below confidence threshold"
        min={0}
        max={1}
        step={0.05}
        value={[value]}
        onValueChange={handleChange}
        className="w-48"
      />
    </div>
  )
}
