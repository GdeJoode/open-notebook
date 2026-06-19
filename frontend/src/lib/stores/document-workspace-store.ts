import { create } from 'zustand'
import { persist } from 'zustand/middleware'

/**
 * Panel sizes are stored as percentages that sum to 100. They are clamped to
 * [MIN_PANEL_PCT, MAX_PANEL_PCT] before persisting so a user can never drag a
 * pane to a degenerate width that the layout can't recover from on reload.
 *
 * react-resizable-panels already enforces per-panel min/max at drag time; the
 * clamp here is the defensive second line for values arriving from persisted
 * storage (e.g. an older schema) or programmatic setters.
 */
export const MIN_PANEL_PCT = 10
export const MAX_PANEL_PCT = 80

export interface PanelSizes {
  left: number
  middle: number
  right: number
}

export const DEFAULT_PANEL_SIZES: PanelSizes = {
  left: 22,
  middle: 56,
  right: 22,
}

function clamp(value: number): number {
  if (Number.isNaN(value)) return MIN_PANEL_PCT
  return Math.min(MAX_PANEL_PCT, Math.max(MIN_PANEL_PCT, value))
}

interface DocumentWorkspaceState {
  activePage: number
  activeChunkId: string | null
  panelSizes: PanelSizes
  setActivePage: (page: number) => void
  setActiveChunkId: (chunkId: string | null) => void
  /**
   * Persist the three pane sizes. Each is individually clamped to
   * [MIN_PANEL_PCT, MAX_PANEL_PCT]; the trio is NOT renormalized to 100 here
   * because react-resizable-panels owns the live layout math and emits values
   * that already sum to 100. The clamp only guards against out-of-range input.
   */
  setPanelSizes: (sizes: PanelSizes) => void
  reset: () => void
}

export const useDocumentWorkspaceStore = create<DocumentWorkspaceState>()(
  persist(
    (set) => ({
      activePage: 1,
      activeChunkId: null,
      panelSizes: { ...DEFAULT_PANEL_SIZES },
      setActivePage: (page) => set({ activePage: Math.max(1, Math.floor(page)) }),
      setActiveChunkId: (chunkId) => set({ activeChunkId: chunkId }),
      setPanelSizes: (sizes) =>
        set({
          panelSizes: {
            left: clamp(sizes.left),
            middle: clamp(sizes.middle),
            right: clamp(sizes.right),
          },
        }),
      reset: () =>
        set({
          activePage: 1,
          activeChunkId: null,
          panelSizes: { ...DEFAULT_PANEL_SIZES },
        }),
    }),
    {
      name: 'document-workspace-storage',
      // Only panel sizes need to survive a full reload. Active page / chunk are
      // navigation-scoped and intentionally reset per source visit.
      partialize: (state) => ({ panelSizes: state.panelSizes }),
    }
  )
)
