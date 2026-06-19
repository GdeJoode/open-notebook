/**
 * Pure pagination helpers for the inspect-workspace image gallery (I.D-4).
 *
 * The gallery bounds how many images are requested at once to satisfy the AC
 * "≤6 in flight": it renders one page of `IMAGE_PAGE_SIZE` images at a time, so
 * the browser never fetches more than a page-worth simultaneously. Keeping the
 * slicing logic here (framework-free) lets the node-env vitest suite assert the
 * bound directly without rendering a component.
 */

/** Max images requested per page — the in-flight upper bound (AC: ≤6). */
export const IMAGE_PAGE_SIZE = 6

/** Total number of pages for a given image count. */
export function pageCount(total: number, pageSize: number = IMAGE_PAGE_SIZE): number {
  if (total <= 0 || pageSize <= 0) return 0
  return Math.ceil(total / pageSize)
}

/** Clamp a (zero-based) page index into the valid range for `total` items. */
export function clampPage(
  page: number,
  total: number,
  pageSize: number = IMAGE_PAGE_SIZE
): number {
  const pages = pageCount(total, pageSize)
  if (pages === 0) return 0
  return Math.min(Math.max(0, page), pages - 1)
}

/**
 * Slice `items` to the (zero-based) page. The returned slice never exceeds
 * `pageSize` elements, which is what bounds concurrent image requests.
 */
export function pageSlice<T>(
  items: readonly T[],
  page: number,
  pageSize: number = IMAGE_PAGE_SIZE
): T[] {
  if (pageSize <= 0) return []
  const safePage = clampPage(page, items.length, pageSize)
  const start = safePage * pageSize
  return items.slice(start, start + pageSize)
}
