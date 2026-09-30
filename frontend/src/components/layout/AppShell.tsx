'use client'

import { useEffect, useState } from 'react'
import { usePathname } from 'next/navigation'
import { Menu } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Sheet, SheetContent, SheetDescription, SheetTitle, SheetTrigger } from '@/components/ui/sheet'
import { useMinWidth } from '@/lib/hooks/use-media-query'
import { AppSidebar } from './AppSidebar'
import { NAV_SECTIONS, activeNavHref } from './nav-items'

interface AppShellProps {
  children: React.ReactNode
}

/** The entry the page belongs to, as the phone top bar's title. */
function sectionTitle(pathname: string): string {
  const href = activeNavHref(pathname)
  return NAV_SECTIONS.flatMap((s) => s.items).find((i) => i.href === href)?.name ?? 'Noesis'
}

export function AppShell({ children }: AppShellProps) {
  // WB.5: below md the sidebar is a drawer, opened from a top bar.
  const isMd = useMinWidth('md')
  const pathname = usePathname()
  const [drawerOpen, setDrawerOpen] = useState(false)

  // Any navigation closes the drawer (Create › Source, a link on the page).
  useEffect(() => {
    setDrawerOpen(false)
  }, [pathname])

  return (
    // The content stays the second child either way, so a resize across md
    // does not remount the page.
    <div className="flex h-screen flex-col overflow-hidden md:flex-row">
      {isMd ? (
        <AppSidebar />
      ) : (
        <header
          className="flex h-14 shrink-0 items-center gap-2 border-b bg-sidebar px-2"
          data-testid="mobile-top-bar"
        >
          <Sheet open={drawerOpen} onOpenChange={setDrawerOpen}>
            {/* The Trigger: the drawer hands focus back to it on close. */}
            <SheetTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="size-10"
                aria-label="Open navigation"
                data-testid="mobile-nav-trigger"
              >
                <Menu className="size-5" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-72 gap-0 p-0" data-testid="mobile-nav-drawer">
              <SheetTitle className="sr-only">Navigation</SheetTitle>
              <SheetDescription className="sr-only">The app&apos;s pages</SheetDescription>
              {/* A link to the page already shown changes no path: close here too. */}
              <AppSidebar drawer onNavigate={() => setDrawerOpen(false)} />
            </SheetContent>
          </Sheet>
          <span className="truncate font-semibold" data-testid="mobile-top-bar-title">
            {sectionTitle(pathname)}
          </span>
        </header>
      )}
      {/* min-w-0: a page wider than the room left scrolls or clips inside
          main; it never squeezes the sidebar. */}
      <div className="flex-1 flex flex-col min-h-0 min-w-0">
        <main className="flex-1 flex flex-col min-h-0 overflow-hidden">
          {children}
        </main>
      </div>
    </div>
  )
}
