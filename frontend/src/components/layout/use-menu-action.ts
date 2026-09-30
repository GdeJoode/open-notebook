import { useRef } from 'react'

/**
 * Runs a menu item's action once its (modal) dropdown menu has closed and
 * handed focus back to its trigger.
 *
 * A Radix Dialog or Popover opened from `onSelect` mounts while the menu is
 * still closing: the two layers race over `pointer-events: none` on <body>
 * (left set after the dialog closes: the page stops responding) and over
 * focus (a popover sees the menu's focus return as a move outside and
 * closes). Deferring the open to the menu's `onCloseAutoFocus` avoids both,
 * and the dialog then returns focus to the trigger when it closes.
 */
export function useMenuAction() {
  const pending = useRef<(() => void) | null>(null)
  return {
    /** For an item's `onSelect`. */
    defer: (action: () => void) => () => {
      pending.current = action
    },
    /** For the menu content's `onCloseAutoFocus`. */
    onCloseAutoFocus: () => {
      const run = pending.current
      pending.current = null
      run?.()
    },
  }
}
