'use client'

import { useState } from 'react'

import type { OverlayCreateInput } from '@/lib/api/entity-resolution'
import { isValidOverlay } from '@/lib/utils/entity-resolution'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Link2, Scissors } from 'lucide-react'

interface OverlayEditorProps {
  /** The active notebook, if the hub is scoped to one. Enables notebook scope. */
  notebookId?: string
  notebookLabel?: string
  onSubmit: (rule: OverlayCreateInput) => void
  disabled?: boolean
}

type Kind = 'merge' | 'split'
type Scope = 'global' | 'notebook'

/**
 * Manual force-merge / force-split rule editor (the human escape hatch over the
 * automated matcher). Force-split is an absolute veto — a pair is never proposed
 * again. Rules apply at global or per-notebook scope (most-specific wins
 * server-side).
 *
 * Reuses the J.5 `PrivacyModeToggle` Radix-RadioGroup scope-toggle pattern: Tab
 * to focus, arrow keys to move, Space/Enter to select, with an `aria-label` per
 * option. AC5.
 */
export function OverlayEditor({
  notebookId,
  notebookLabel,
  onSubmit,
  disabled = false,
}: OverlayEditorProps) {
  const [kind, setKind] = useState<Kind>('split')
  const [scope, setScope] = useState<Scope>(notebookId ? 'notebook' : 'global')
  const [nameA, setNameA] = useState('')
  const [nameB, setNameB] = useState('')
  const [entityType, setEntityType] = useState('')

  const rule: OverlayCreateInput = {
    kind,
    name_a: nameA,
    name_b: nameB,
    entity_type: entityType,
    scope,
    notebook_id: scope === 'notebook' ? (notebookId ?? null) : null,
  }
  const valid = isValidOverlay(rule)

  const kindOptions: { value: Kind; label: string; icon: typeof Link2 }[] = [
    { value: 'split', label: 'Force split (keep apart)', icon: Scissors },
    { value: 'merge', label: 'Force merge (treat as same)', icon: Link2 },
  ]

  const scopeOptions: { value: Scope; label: string }[] = [
    { value: 'global', label: 'Global (everywhere)' },
  ]
  if (notebookId) {
    scopeOptions.push({
      value: 'notebook',
      label: notebookLabel ? `This notebook (${notebookLabel})` : 'This notebook',
    })
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Manual override</CardTitle>
        <CardDescription>
          Force two surface forms to always merge, or to never merge. A force
          split is an absolute veto — the pair is never proposed again.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="overlay-name-a" className="text-sm">
              Surface form A
            </Label>
            <Input
              id="overlay-name-a"
              value={nameA}
              onChange={(e) => setNameA(e.target.value)}
              disabled={disabled}
              placeholder="e.g. BZK"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="overlay-name-b" className="text-sm">
              Surface form B
            </Label>
            <Input
              id="overlay-name-b"
              value={nameB}
              onChange={(e) => setNameB(e.target.value)}
              disabled={disabled}
              placeholder="e.g. Financiën"
            />
          </div>
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="overlay-type" className="text-sm">
            Entity type{' '}
            <span className="text-muted-foreground font-normal">(optional)</span>
          </Label>
          <Input
            id="overlay-type"
            value={entityType}
            onChange={(e) => setEntityType(e.target.value)}
            disabled={disabled}
            placeholder="organization"
          />
        </div>

        <div className="space-y-2">
          <Label className="text-sm font-medium">Rule</Label>
          <RadioGroup
            value={kind}
            onValueChange={(v) => setKind(v as Kind)}
            disabled={disabled}
            aria-label="Override rule kind"
            className="flex flex-col gap-2"
          >
            {kindOptions.map((opt) => {
              const Icon = opt.icon
              const id = `overlay-kind-${opt.value}`
              return (
                <div key={opt.value} className="flex items-center gap-2">
                  <RadioGroupItem
                    value={opt.value}
                    id={id}
                    aria-label={opt.label}
                  />
                  <Label
                    htmlFor={id}
                    className="flex items-center gap-1.5 text-sm font-normal cursor-pointer"
                  >
                    <Icon className="h-3.5 w-3.5" aria-hidden="true" />
                    {opt.label}
                  </Label>
                </div>
              )
            })}
          </RadioGroup>
        </div>

        <div className="space-y-2">
          <Label className="text-sm font-medium">Scope</Label>
          <RadioGroup
            value={scope}
            onValueChange={(v) => setScope(v as Scope)}
            disabled={disabled}
            aria-label="Override scope"
            className="flex flex-wrap gap-4"
          >
            {scopeOptions.map((opt) => {
              const id = `overlay-scope-${opt.value}`
              return (
                <div key={opt.value} className="flex items-center gap-2">
                  <RadioGroupItem
                    value={opt.value}
                    id={id}
                    aria-label={`Scope: ${opt.label}`}
                  />
                  <Label
                    htmlFor={id}
                    className="text-sm font-normal cursor-pointer"
                  >
                    {opt.label}
                  </Label>
                </div>
              )
            })}
          </RadioGroup>
        </div>

        <div className="flex justify-end">
          <Button
            size="sm"
            disabled={disabled || !valid}
            onClick={() => onSubmit(rule)}
          >
            Add rule
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
