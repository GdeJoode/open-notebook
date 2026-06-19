import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  DEFAULT_PANEL_SIZES,
  MAX_PANEL_PCT,
  MIN_PANEL_PCT,
  useDocumentWorkspaceStore,
} from './document-workspace-store'

/**
 * Unit tests for the document-workspace store reducer (Phase I.B AC3).
 *
 * Covers: set/reset of sizes, clamping to [MIN_PANEL_PCT, MAX_PANEL_PCT], and
 * the persist contract (only panelSizes is written to storage; navigation
 * state is not).
 */

const STORAGE_KEY = 'document-workspace-storage'

function freshStore() {
  // Reset to defaults between tests; the store is a module-level singleton.
  useDocumentWorkspaceStore.getState().reset()
}

describe('document-workspace-store', () => {
  beforeEach(() => {
    localStorage.clear()
    freshStore()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('starts with the default panel sizes', () => {
    expect(useDocumentWorkspaceStore.getState().panelSizes).toEqual(
      DEFAULT_PANEL_SIZES
    )
  })

  it('setPanelSizes stores in-range values verbatim', () => {
    useDocumentWorkspaceStore
      .getState()
      .setPanelSizes({ left: 30, middle: 40, right: 30 })
    expect(useDocumentWorkspaceStore.getState().panelSizes).toEqual({
      left: 30,
      middle: 40,
      right: 30,
    })
  })

  it('clamps panel sizes below MIN_PANEL_PCT up to the floor', () => {
    useDocumentWorkspaceStore
      .getState()
      .setPanelSizes({ left: 2, middle: 90, right: 8 })
    const { panelSizes } = useDocumentWorkspaceStore.getState()
    expect(panelSizes.left).toBe(MIN_PANEL_PCT)
    expect(panelSizes.right).toBe(MIN_PANEL_PCT)
  })

  it('clamps panel sizes above MAX_PANEL_PCT down to the ceiling', () => {
    useDocumentWorkspaceStore
      .getState()
      .setPanelSizes({ left: 95, middle: 3, right: 2 })
    const { panelSizes } = useDocumentWorkspaceStore.getState()
    expect(panelSizes.left).toBe(MAX_PANEL_PCT)
  })

  it('coerces NaN sizes to the floor', () => {
    useDocumentWorkspaceStore
      .getState()
      .setPanelSizes({ left: Number.NaN, middle: 50, right: 30 })
    expect(useDocumentWorkspaceStore.getState().panelSizes.left).toBe(
      MIN_PANEL_PCT
    )
  })

  it('setActivePage floors and never goes below 1', () => {
    useDocumentWorkspaceStore.getState().setActivePage(4.9)
    expect(useDocumentWorkspaceStore.getState().activePage).toBe(4)
    useDocumentWorkspaceStore.getState().setActivePage(-3)
    expect(useDocumentWorkspaceStore.getState().activePage).toBe(1)
  })

  it('setActiveChunkId round-trips a value and clears with null', () => {
    useDocumentWorkspaceStore.getState().setActiveChunkId('chunk:42')
    expect(useDocumentWorkspaceStore.getState().activeChunkId).toBe('chunk:42')
    useDocumentWorkspaceStore.getState().setActiveChunkId(null)
    expect(useDocumentWorkspaceStore.getState().activeChunkId).toBeNull()
  })

  it('reset restores defaults', () => {
    const s = useDocumentWorkspaceStore.getState()
    s.setPanelSizes({ left: 40, middle: 30, right: 30 })
    s.setActivePage(7)
    s.setActiveChunkId('chunk:9')
    useDocumentWorkspaceStore.getState().reset()
    const after = useDocumentWorkspaceStore.getState()
    expect(after.panelSizes).toEqual(DEFAULT_PANEL_SIZES)
    expect(after.activePage).toBe(1)
    expect(after.activeChunkId).toBeNull()
  })

  it('persists only panelSizes to storage (navigation state excluded)', () => {
    const s = useDocumentWorkspaceStore.getState()
    s.setPanelSizes({ left: 35, middle: 35, right: 30 })
    s.setActivePage(5)
    s.setActiveChunkId('chunk:1')

    const raw = localStorage.getItem(STORAGE_KEY)
    expect(raw).not.toBeNull()
    const parsed = JSON.parse(raw as string)
    expect(parsed.state.panelSizes).toEqual({
      left: 35,
      middle: 35,
      right: 30,
    })
    // activePage / activeChunkId must NOT be serialized.
    expect(parsed.state.activePage).toBeUndefined()
    expect(parsed.state.activeChunkId).toBeUndefined()
  })
})
