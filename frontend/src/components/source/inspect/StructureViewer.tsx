'use client'

import { Network } from 'lucide-react'

/**
 * Structure tab of the inspect workspace — intentional placeholder (I.D-4).
 *
 * The real document-structure graph (DoclingDocument tree / reading-order
 * graph rendered with Sigma.js) is Phase I.F. This component reserves the tab
 * slot so the tab strip is complete now; I.F replaces it with
 * <StructureGraphView> per the plan.
 */
export function StructureViewer() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center">
      <Network className="h-8 w-8 text-muted-foreground" />
      <div>
        <p className="text-sm font-medium">Structure graph</p>
        <p className="mt-1 text-xs text-muted-foreground">
          The document structure graph is coming in a later phase (I.F).
        </p>
      </div>
    </div>
  )
}
