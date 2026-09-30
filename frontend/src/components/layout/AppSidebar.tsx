'use client'

import { useState } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { usePathname, useRouter } from 'next/navigation'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { useAuth } from '@/lib/hooks/use-auth'
import { useAttention } from '@/lib/hooks/use-attention'
import { useSidebarStore } from '@/lib/stores/sidebar-store'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { ThemeToggle } from '@/components/common/ThemeToggle'
import { CreateNotebookDialog } from '@/components/notebooks/CreateNotebookDialog'
import { GeneratePodcastDialog } from '@/components/podcasts/GeneratePodcastDialog'
import { Separator } from '@/components/ui/separator'
import { Badge } from '@/components/ui/badge'
import {
  Book,
  Mic,
  LogOut,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Ellipsis,
  Menu,
  FileText,
  Plus,
} from 'lucide-react'
import { NAV_SECTIONS, TRAY_HREF, activeNavHref, type NavItem, type NavSection } from './nav-items'

const MORE_OPEN_KEY = 'sidebar.moreOpen'

function readMoreOpen(): boolean {
  try {
    return window.localStorage.getItem(MORE_OPEN_KEY) === 'true'
  } catch {
    return false
  }
}

function writeMoreOpen(open: boolean): void {
  try {
    window.localStorage.setItem(MORE_OPEN_KEY, String(open))
  } catch {
    // Storage unavailable: the group still toggles for this page's life.
  }
}

type CreateTarget = 'source' | 'notebook' | 'podcast'

export function AppSidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const { logout } = useAuth()
  const { isCollapsed, toggleCollapse } = useSidebarStore()

  const [createMenuOpen, setCreateMenuOpen] = useState(false)
  const [notebookDialogOpen, setNotebookDialogOpen] = useState(false)
  const [podcastDialogOpen, setPodcastDialogOpen] = useState(false)
  const [moreOpen, setMoreOpen] = useState(readMoreOpen)
  const activeHref = activeNavHref(pathname)

  const toggleMore = () => {
    setMoreOpen((open) => {
      writeMoreOpen(!open)
      return !open
    })
  }

  // WB.1: The Tray's total, and per entry the `counted` of the groups
  // whose `nav` it is (each thing to act on once). Partial ("12+") when a
  // group behind the number failed or timed out.
  const { data: attention } = useAttention()
  const badgeFor = (href: string): { count: number; partial: boolean } => {
    if (!attention) return { count: 0, partial: false }
    if (href === TRAY_HREF) return { count: attention.total, partial: attention.partial }
    const groups = attention.groups.filter((g) => g.nav === href)
    return {
      count: groups.reduce((sum, g) => sum + g.counted, 0),
      partial: groups.some((g) => g.needs_action && !!g.error),
    }
  }

  const renderItem = (item: NavItem) => {
    const isActive = item.href === activeHref
    const { count, partial } = badgeFor(item.href)
    const button = (
      <Button
        variant={isActive ? 'secondary' : 'ghost'}
        className={cn(
          'w-full gap-3 text-sidebar-foreground',
          isActive && 'bg-sidebar-accent text-sidebar-accent-foreground',
          isCollapsed ? 'relative justify-center px-2' : 'justify-start'
        )}
      >
        <item.icon className="h-4 w-4" />
        {!isCollapsed && <span>{item.name}</span>}
        {(count > 0 || partial) && (
          <Badge
            className={cn(
              'tabular-nums',
              isCollapsed ? 'absolute -right-1 -top-1 px-1' : 'ml-auto'
            )}
            data-testid={`nav-badge-${item.name}`}
          >
            {count}
            {partial && '+'}
            <span className="sr-only">
              {' need you'}
              {partial && ', some groups did not load'}
            </span>
          </Badge>
        )}
      </Button>
    )

    if (isCollapsed) {
      return (
        <Tooltip key={item.name}>
          <TooltipTrigger asChild>
            <Link href={item.href} aria-current={isActive ? 'page' : undefined}>
              {button}
            </Link>
          </TooltipTrigger>
          <TooltipContent side="right">
            {count > 0 || partial ? `${item.name} (${count}${partial ? '+' : ''})` : item.name}
          </TooltipContent>
        </Tooltip>
      )
    }

    return (
      <Link key={item.name} href={item.href} aria-current={isActive ? 'page' : undefined}>
        {button}
      </Link>
    )
  }

  // WB-D4: features outside the product scope, folded away by default.
  const renderCollapsible = (section: NavSection) => {
    const listId = `nav-section-${section.title}`
    const toggle = (
      <Button
        variant="ghost"
        onClick={toggleMore}
        aria-expanded={moreOpen}
        aria-controls={listId}
        aria-label={isCollapsed ? section.title : undefined}
        className={cn(
          'w-full gap-3 text-sidebar-foreground/80',
          isCollapsed ? 'justify-center px-2' : 'justify-start'
        )}
        data-testid="nav-more-toggle"
      >
        {isCollapsed ? (
          <Ellipsis className="h-4 w-4" />
        ) : (
          <>
            {moreOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            <span>{section.title}</span>
          </>
        )}
      </Button>
    )
    return (
      <>
        {isCollapsed ? (
          <Tooltip>
            <TooltipTrigger asChild>{toggle}</TooltipTrigger>
            <TooltipContent side="right">{section.title}</TooltipContent>
          </Tooltip>
        ) : (
          toggle
        )}
        <div id={listId} className="space-y-1" hidden={!moreOpen}>
          {moreOpen && section.items.map(renderItem)}
        </div>
      </>
    )
  }

  const handleCreateSelection = (target: CreateTarget) => {
    setCreateMenuOpen(false)

    if (target === 'source') {
      router.push('/sources/new')
    } else if (target === 'notebook') {
      setNotebookDialogOpen(true)
    } else if (target === 'podcast') {
      setPodcastDialogOpen(true)
    }
  }

  return (
    <TooltipProvider delayDuration={0}>
      <div
        className={cn(
          'app-sidebar flex h-full flex-col bg-sidebar border-sidebar-border border-r transition-all duration-300',
          isCollapsed ? 'w-16' : 'w-64'
        )}
      >
        <div
          className={cn(
            'flex h-16 shrink-0 items-center group',
            isCollapsed ? 'justify-center px-2' : 'justify-between px-4'
          )}
        >
          {isCollapsed ? (
            <div className="relative flex items-center justify-center w-full">
              <Image
                src="/logo.svg"
                alt="Noesis"
                width={32}
                height={32}
                className="transition-opacity group-hover:opacity-0"
              />
              <Button
                variant="ghost"
                size="sm"
                onClick={toggleCollapse}
                className="absolute text-sidebar-foreground hover:bg-sidebar-accent opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <Menu className="h-4 w-4" />
              </Button>
            </div>
          ) : (
            <>
              <div className="flex items-center gap-2">
                <Image src="/logo.svg" alt="Noesis" width={32} height={32} />
                <span className="text-base font-medium text-sidebar-foreground">
                  Noesis
                </span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={toggleCollapse}
                className="text-sidebar-foreground hover:bg-sidebar-accent"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
            </>
          )}
        </div>

        {/* WB.4: the nav scrolls inside the sidebar, so Theme and Sign Out stay
            on screen on a 768 px tall laptop. */}
        <nav
          className={cn(
            'min-h-0 flex-1 space-y-1 overflow-y-auto py-3',
            isCollapsed ? 'px-2' : 'px-3'
          )}
        >
          <div
            className={cn(
              'mb-3',
              isCollapsed ? 'px-0' : 'px-3'
            )}
          >
            <DropdownMenu open={createMenuOpen} onOpenChange={setCreateMenuOpen}>
              {isCollapsed ? (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <DropdownMenuTrigger asChild>
                      <Button
                        onClick={() => setCreateMenuOpen(true)}
                        variant="default"
                        size="sm"
                        className="w-full justify-center px-2 bg-primary hover:bg-primary/90 text-primary-foreground border-0"
                        aria-label="Create"
                      >
                        <Plus className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                  </TooltipTrigger>
                  <TooltipContent side="right">Create</TooltipContent>
                </Tooltip>
              ) : (
                <DropdownMenuTrigger asChild>
                  <Button
                    onClick={() => setCreateMenuOpen(true)}
                    variant="default"
                    size="sm"
                    className="w-full justify-start bg-primary hover:bg-primary/90 text-primary-foreground border-0"
                  >
                    <Plus className="h-4 w-4 mr-2" />
                    Create
                  </Button>
                </DropdownMenuTrigger>
              )}

              <DropdownMenuContent
                align={isCollapsed ? 'end' : 'start'}
                side={isCollapsed ? 'right' : 'bottom'}
                className="w-48"
              >
                <DropdownMenuItem
                  onSelect={(event) => {
                    event.preventDefault()
                    handleCreateSelection('source')
                  }}
                  className="gap-2"
                >
                  <FileText className="h-4 w-4" />
                  Source
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={(event) => {
                    event.preventDefault()
                    handleCreateSelection('notebook')
                  }}
                  className="gap-2"
                >
                  <Book className="h-4 w-4" />
                  Notebook
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={(event) => {
                    event.preventDefault()
                    handleCreateSelection('podcast')
                  }}
                  className="gap-2"
                >
                  <Mic className="h-4 w-4" />
                  Podcast
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          {NAV_SECTIONS.map((section, index) => (
            <div key={section.title ?? index}>
              {index > 0 && <Separator className="my-2" />}
              <div className="space-y-1">
                {section.collapsible
                  ? renderCollapsible(section)
                  : (
                    <>
                      {section.title && !isCollapsed && (
                        <h3 className="mb-1 px-3 text-xs font-semibold uppercase tracking-wider text-sidebar-foreground/60">
                          {section.title}
                        </h3>
                      )}
                      {section.items.map(renderItem)}
                    </>
                  )}
              </div>
            </div>
          ))}
        </nav>

        <div
          className={cn(
            'shrink-0 border-t border-sidebar-border p-3 space-y-2',
            isCollapsed && 'px-2'
          )}
        >
          <div
            className={cn(
              'flex',
              isCollapsed ? 'justify-center' : 'justify-start'
            )}
          >
            {isCollapsed ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <div>
                    <ThemeToggle iconOnly />
                  </div>
                </TooltipTrigger>
                <TooltipContent side="right">Theme</TooltipContent>
              </Tooltip>
            ) : (
              <ThemeToggle />
            )}
          </div>

          {isCollapsed ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  className="w-full justify-center"
                  onClick={logout}
                >
                  <LogOut className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">Sign Out</TooltipContent>
            </Tooltip>
          ) : (
            <Button
              variant="outline"
              className="w-full justify-start gap-3"
              onClick={logout}
            >
              <LogOut className="h-4 w-4" />
              Sign Out
            </Button>
          )}
        </div>
      </div>

      <CreateNotebookDialog
        open={notebookDialogOpen}
        onOpenChange={setNotebookDialogOpen}
      />
      <GeneratePodcastDialog
        open={podcastDialogOpen}
        onOpenChange={setPodcastDialogOpen}
      />
    </TooltipProvider>
  )
}
