/**
 * Unit-test spec for ``parseFilenameFromContentDisposition``.
 *
 * Lives in the Playwright suite because the project has no other
 * frontend unit-test runner. The parser is pure (no DOM, no React)
 * so we don't need the Playwright ``page`` fixture -- the test just
 * imports the function and asserts return values.
 *
 * Coverage matrix (matches plan §D.1c "minimum 4 cases"):
 *
 *   1. RFC 6266 quoted form -> filename.
 *   2. Token form (unquoted, no quotes) -> filename.
 *   3. Header with semicolons before/after the filename -> filename.
 *   4. Missing / empty / no-parameter header -> null.
 *
 * Plus the mental-inversion regression: RFC 5987 ``filename*=UTF-8''``
 * form must decode percent-escapes correctly. If the parser were
 * simplified to a single ``/filename="([^"]+)"/`` regex it would
 * return ``null`` here -- so this assertion catches that failure
 * mode at the unit level.
 */

import { expect, test } from '@playwright/test'

import { parseFilenameFromContentDisposition } from '../../src/lib/utils/content-disposition'

test.describe('parseFilenameFromContentDisposition', () => {
  test('extracts the standard quoted form', () => {
    expect(
      parseFilenameFromContentDisposition(
        'attachment; filename="my-export.zip"',
      ),
    ).toBe('my-export.zip')
  })

  test('extracts the unquoted token form', () => {
    expect(
      parseFilenameFromContentDisposition('attachment; filename=my-export.zip'),
    ).toBe('my-export.zip')
  })

  test('handles extra parameters before and after the filename', () => {
    // Real-world header with multiple parameters. The parser must
    // bound the token capture at the next ``;`` so the trailing
    // ``modification-date`` doesn't bleed into the filename.
    expect(
      parseFilenameFromContentDisposition(
        'attachment; size=1024; filename="my export.zip"; modification-date="Wed, 12 Feb 2026 16:29:51 -0500"',
      ),
    ).toBe('my export.zip')
  })

  test('returns null for missing or unparseable headers', () => {
    expect(parseFilenameFromContentDisposition(undefined)).toBeNull()
    expect(parseFilenameFromContentDisposition(null)).toBeNull()
    expect(parseFilenameFromContentDisposition('')).toBeNull()
    expect(parseFilenameFromContentDisposition('inline')).toBeNull()
    // Header with no filename parameter at all.
    expect(
      parseFilenameFromContentDisposition('attachment; size=1024'),
    ).toBeNull()
  })

  /**
   * Mental-inversion regression check (REQUIRED by plan §D.1c).
   *
   * The RFC 5987 ``filename*=UTF-8''<percent-encoded>`` form is the
   * only one that carries non-ASCII characters. A naive parser that
   * only matched ``filename="..."`` would return ``null`` here even
   * though the header is well-formed. This test fails fast on the
   * simplification.
   *
   * The decoded form should be the original UTF-8 filename, including
   * spaces and the accented vowel.
   */
  test('decodes the RFC 5987 percent-encoded form', () => {
    expect(
      parseFilenameFromContentDisposition(
        "attachment; filename*=UTF-8''my-file.zip",
      ),
    ).toBe('my-file.zip')

    // Non-ASCII path: ``café résumé.zip`` percent-encoded as UTF-8.
    expect(
      parseFilenameFromContentDisposition(
        "attachment; filename*=UTF-8''caf%C3%A9%20r%C3%A9sum%C3%A9.zip",
      ),
    ).toBe('café résumé.zip')
  })

  test('prefers RFC 5987 form when both filename and filename* are present', () => {
    // RFC 6266 §4.3 says clients SHOULD pick filename* when both are
    // present. The hook relies on this so a UTF-8 filename isn't
    // silently demoted to its ASCII fallback.
    expect(
      parseFilenameFromContentDisposition(
        "attachment; filename=\"fallback.zip\"; filename*=UTF-8''real-file.zip",
      ),
    ).toBe('real-file.zip')
  })

  test('falls back to other forms when RFC 5987 value is malformed', () => {
    // %ZZ is invalid percent-encoding; decodeURIComponent throws.
    // The parser must catch and fall through to the next form so the
    // user still gets *something* downloadable.
    expect(
      parseFilenameFromContentDisposition(
        "attachment; filename*=UTF-8''%ZZ; filename=\"safe.zip\"",
      ),
    ).toBe('safe.zip')
  })
})
