/**
 * Pure helpers for the layered privacy toggle (Track J.5).
 *
 * Privacy has three scopes — global default, per-notebook, per-document —
 * with most-specific-wins and a sticky `private`. The UI lets notebook/document
 * say "inherit"; this module maps the UI value to/from the wire and resolves the
 * inherited effective value for the "inherits: <resolved>" hint.
 *
 * Backed by the J.3 resolver semantics (see
 * `apps/app-main/src/app_main/services/model_routing/privacy_resolver.py`):
 * document `private == true` is sticky PRIVATE; otherwise notebook overrides
 * global; otherwise the global default applies.
 */

import { PrivacyValue } from '@/lib/types/model-routing'

export type PrivacyScope = 'global' | 'notebook' | 'document'

/** The on-the-wire value: global is always concrete; others may be null=inherit. */
export type WirePrivacy = 'cloud' | 'private' | null

/** UI value -> wire value. `inherit` (notebook/document only) becomes null. */
export function toWire(value: PrivacyValue): WirePrivacy {
  return value === 'inherit' ? null : value
}

/** Wire value -> UI value. null becomes `inherit`. */
export function fromWire(value: WirePrivacy | undefined): PrivacyValue {
  return value == null ? 'inherit' : value
}

/**
 * Resolve the effective ('cloud' | 'private') value a scope inherits when set to
 * `inherit`, given the more-general scopes. Used only for the hint text; the
 * authoritative resolution is server-side.
 */
export function resolveInherited(
  scope: PrivacyScope,
  context: { globalDefault: 'cloud' | 'private'; notebookValue?: WirePrivacy },
): 'cloud' | 'private' {
  if (scope === 'global') return context.globalDefault
  if (scope === 'notebook') return context.globalDefault
  // document inherits notebook when set, else global.
  return context.notebookValue ?? context.globalDefault
}

/** Whether a scope is allowed to choose "inherit" (notebook/document only). */
export function canInherit(scope: PrivacyScope): boolean {
  return scope !== 'global'
}
