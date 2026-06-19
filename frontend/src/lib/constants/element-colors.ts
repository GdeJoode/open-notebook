/**
 * Element-type color scheme for PDF bbox overlays and the LayersBar.
 *
 * Single source of truth shared by the embed-mode Chunks viewer
 * (`PdfChunkViewer`) and the inspect workspace's `LayersBar`. The palette
 * mirrors Docling Studio's `elementColors.ts` (MIT) — kept in sync here as the
 * canonical in-repo copy since the upstream DS module is not vendored.
 */

export const ELEMENT_COLORS: Record<string, string> = {
  title: '#EF4444',
  section_header: '#F97316',
  text: '#3B82F6',
  paragraph: '#3B82F6',
  table: '#8B5CF6',
  picture: '#22C55E',
  list_item: '#06B6D4',
  list: '#06B6D4',
  formula: '#EC4899',
  code: '#14B8A6',
  caption: '#EAB308',
  heading: '#F97316',
  page_header: '#9CA3AF',
  page_footer: '#9CA3AF',
  footnote: '#9CA3AF',
}

export const ELEMENT_TYPES = Object.keys(ELEMENT_COLORS)

export const DEFAULT_COLOR = '#6B7280'

/** Normalize an element-type label to its color-map key (lowercase, `_`-joined). */
export function elementTypeKey(elementType: string): string {
  return elementType.toLowerCase().replace(/\s+/g, '_')
}

export function getElementColor(elementType: string): string {
  return ELEMENT_COLORS[elementTypeKey(elementType)] ?? DEFAULT_COLOR
}
