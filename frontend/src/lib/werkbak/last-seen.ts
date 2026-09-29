/**
 * The Werkbak's "last visit", stored per browser (WB.1).
 *
 * Holds the `generated_at` of the report the owner last marked as seen; the
 * next read sends it as `since`. Storage can be unavailable (private mode,
 * blocked); every access is guarded and the page works without it.
 */

const KEY = 'werkbak.lastSeen'
const EVENT = 'werkbak:last-seen'

export function getLastSeen(): string | null {
  try {
    return window.localStorage.getItem(KEY)
  } catch {
    return null
  }
}

export function setLastSeen(value: string): void {
  try {
    window.localStorage.setItem(KEY, value)
  } catch {
    // Not stored: the next visit shows the default window, nothing breaks.
  }
  try {
    window.dispatchEvent(new Event(EVENT))
  } catch {
    // no window
  }
}

/** For `useSyncExternalStore`: other tabs (storage) and this tab (EVENT). */
export function subscribeLastSeen(onChange: () => void): () => void {
  window.addEventListener('storage', onChange)
  window.addEventListener(EVENT, onChange)
  return () => {
    window.removeEventListener('storage', onChange)
    window.removeEventListener(EVENT, onChange)
  }
}
