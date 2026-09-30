import * as React from "react"

type AutoFocusHandler = (event: Event) => void

/** Records, as the layer mounts, what had focus before it opened. */
function CaptureFocus({ into }: { into: React.RefObject<HTMLElement | null> }) {
  // Read while rendering: before the layer's content commits and an
  // `autoFocus` field inside it takes focus.
  const [before] = React.useState(() =>
    typeof document === "undefined" ? null : (document.activeElement as HTMLElement | null)
  )
  React.useLayoutEffect(() => {
    into.current = before
  }, [into, before])
  return null
}

/**
 * Radix dialogs give focus back to their own Trigger only. One opened
 * without it (from a menu through `useMenuAction`, from a card, from a
 * button inside another component) left focus on <body>: inside an open
 * drawer, outside the layer the user is in. This gives focus back to
 * whatever had it when the layer opened, unless the caller handles it.
 * Render `capture` inside the portal, before the content.
 */
export function useReturnFocus(onCloseAutoFocus?: AutoFocusHandler) {
  const returnTo = React.useRef<HTMLElement | null>(null)
  return {
    capture: <CaptureFocus into={returnTo} />,
    onCloseAutoFocus: (event: Event) => {
      onCloseAutoFocus?.(event)
      const target = returnTo.current
      if (event.defaultPrevented || !target?.isConnected || target === document.body) return
      event.preventDefault()
      target.focus()
    },
  }
}
