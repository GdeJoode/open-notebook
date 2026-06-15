/**
 * Pure helpers for parsing the ``Content-Disposition`` HTTP header.
 *
 * Extracted from ``use-obsidian-export.ts`` so unit tests can import
 * the parser in a Node-only context (Playwright spec files cannot
 * resolve ``'use client'`` modules pulling in React Query).
 *
 * Keep this module dependency-free -- it is imported by hooks, tests,
 * and potentially future export buttons.
 */

/**
 * Strict-but-permissive ``Content-Disposition`` filename parser.
 *
 * Handles the three header forms HTTP commonly produces:
 *
 *   1. ``attachment; filename="my-file.zip"``   (RFC 6266 quoted)
 *   2. ``attachment; filename=my-file.zip``      (unquoted token)
 *   3. ``attachment; filename*=UTF-8''my-file.zip`` (RFC 5987 percent-encoded)
 *
 * The RFC 5987 form takes precedence when both ``filename`` and
 * ``filename*`` are present (per RFC 6266 §4.3) -- it's the only one
 * that can carry non-ASCII characters.
 *
 * Returns ``null`` for:
 *   - missing header
 *   - header with no ``filename``/``filename*`` parameter
 *   - parameter value that's empty after decoding/unquoting
 *
 * The dialog falls back to a default name when the parser returns
 * ``null`` so the user always gets *something* downloadable.
 */
export function parseFilenameFromContentDisposition(
  header: string | undefined | null,
): string | null {
  if (!header) {
    return null
  }

  // RFC 5987 first -- precedence rule from RFC 6266 §4.3. The format
  // is ``filename*=<charset>'<lang>'<percent-encoded-value>``. We only
  // honour UTF-8; other charsets are rare on the modern web and a
  // strict implementation would require a TextDecoder.
  const rfc5987 =
    /filename\*\s*=\s*UTF-8''([^;]+)/i.exec(header)
  if (rfc5987) {
    try {
      const decoded = decodeURIComponent(rfc5987[1].trim())
      if (decoded) return decoded
    } catch {
      // Malformed percent-encoding -- fall through to the next form.
    }
  }

  // Quoted form: ``filename="..."``. The capture is everything
  // between the first pair of double quotes -- escaping isn't
  // standardised in RFC 6266 so we don't try to unescape.
  const quoted = /filename\s*=\s*"([^"]+)"/i.exec(header)
  if (quoted) {
    return quoted[1].trim() || null
  }

  // Token form: ``filename=value`` where value is a token (no spaces,
  // no quotes) bounded by ``;`` or end-of-string.
  const token = /filename\s*=\s*([^;]+)/i.exec(header)
  if (token) {
    const value = token[1].trim()
    return value || null
  }

  return null
}
