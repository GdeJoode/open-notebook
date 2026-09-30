'use client'

import { AppSidebar } from './AppSidebar'

interface AppShellProps {
  children: React.ReactNode
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="flex h-screen overflow-hidden">
      <AppSidebar />
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
