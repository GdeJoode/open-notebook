import { describe, expect, it } from 'vitest'
import {
  IMAGE_PAGE_SIZE,
  clampPage,
  pageCount,
  pageSlice,
} from './image-pagination'

/**
 * Unit tests for the image-gallery pagination helpers (I.D-4 AC3).
 *
 * The load-bearing guarantee is that a single page never exposes more than
 * IMAGE_PAGE_SIZE images, which is how the gallery bounds concurrent image
 * requests to ≤6.
 */
describe('image-pagination (I.D-4)', () => {
  it('caps the in-flight bound at 6', () => {
    expect(IMAGE_PAGE_SIZE).toBe(6)
  })

  it('a page slice never exceeds the page size', () => {
    const items = Array.from({ length: 25 }, (_, i) => i)
    for (let page = 0; page < pageCount(items.length); page++) {
      expect(pageSlice(items, page).length).toBeLessThanOrEqual(IMAGE_PAGE_SIZE)
    }
  })

  it('computes page count by ceiling division', () => {
    expect(pageCount(0)).toBe(0)
    expect(pageCount(1)).toBe(1)
    expect(pageCount(6)).toBe(1)
    expect(pageCount(7)).toBe(2)
    expect(pageCount(13)).toBe(3)
  })

  it('clamps page index into range', () => {
    expect(clampPage(-5, 13)).toBe(0)
    expect(clampPage(99, 13)).toBe(2)
    expect(clampPage(0, 0)).toBe(0)
  })

  it('returns the correct window for a middle page', () => {
    const items = Array.from({ length: 13 }, (_, i) => i)
    expect(pageSlice(items, 1)).toEqual([6, 7, 8, 9, 10, 11])
    expect(pageSlice(items, 2)).toEqual([12])
  })

  it('returns an empty slice for an empty collection', () => {
    expect(pageSlice([], 0)).toEqual([])
    expect(pageCount(0)).toBe(0)
  })
})
