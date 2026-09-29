import {
  AlignLeft,
  Bot,
  Book,
  ClipboardList,
  FileText,
  GitMerge,
  Inbox,
  ListChecks,
  Mic,
  Network,
  Replace,
  Route,
  Search,
  Settings,
  Share2,
  Shuffle,
  type LucideIcon,
} from 'lucide-react'

/**
 * The sidebar in pipeline order (WB.2): what needs you, the sources, their
 * review, the knowledge they produce, the collections, the settings, and the
 * features outside the product scope under "More" (WB-D4).
 *
 * An entry's `href` is also the `nav` value of the `/api/attention` groups
 * whose badge it shows (`attention_service.py` names these hrefs).
 */
export interface NavItem {
  name: string
  href: string
  icon: LucideIcon
}

export interface NavSection {
  /** No title: the entries sit directly under The Tray. */
  title?: string
  items: NavItem[]
  /** Folded away by default; open or closed per browser. */
  collapsible?: boolean
}

export const TRAY_HREF = '/'

/**
 * The proposals are a queue per notebook, accepted on each notebook's Schema
 * tab. The Tray's "Schema proposals" card already lists every notebook with
 * proposals, each linked to its tab; the entry opens that card.
 */
export const SCHEMA_PROPOSALS_HREF = '/#schema_proposals'

/**
 * The retype panel retypes under every active ontology, whichever one it is
 * opened on; `schema_core` ships with the app. The same link as the Tray's
 * "Retype pending" card (`RETYPE_LINK` in `attention_service.py`).
 */
export const RETYPE_HREF = '/ontologies?name=schema_core&tab=edit&retype=1'

export const NAV_SECTIONS: NavSection[] = [
  {
    items: [
      { name: 'The Tray', href: TRAY_HREF, icon: Inbox },
      { name: 'Sources', href: '/sources', icon: FileText },
    ],
  },
  {
    title: 'Review',
    items: [
      { name: 'Schema proposals', href: SCHEMA_PROPOSALS_HREF, icon: ListChecks },
      { name: 'Triage', href: '/knowledge-graph/triage', icon: ClipboardList },
      { name: 'Duplicates', href: '/knowledge-graph/resolution', icon: GitMerge },
      { name: 'Retype', href: RETYPE_HREF, icon: Replace },
    ],
  },
  {
    title: 'Knowledge',
    items: [
      { name: 'Graph', href: '/knowledge-graph', icon: Share2 },
      { name: 'Ontologies', href: '/ontologies', icon: Network },
      { name: 'Search', href: '/search', icon: Search },
    ],
  },
  {
    title: 'Collections',
    items: [{ name: 'Notebooks', href: '/notebooks', icon: Book }],
  },
  {
    title: 'Settings',
    items: [
      { name: 'Models', href: '/models', icon: Bot },
      { name: 'Model routing', href: '/settings/model-routing', icon: Route },
      { name: 'Settings', href: '/settings', icon: Settings },
    ],
  },
  {
    title: 'More',
    collapsible: true,
    items: [
      { name: 'Podcasts', href: '/podcasts', icon: Mic },
      { name: 'Transformations', href: '/transformations', icon: Shuffle },
      { name: 'Summaries', href: '/summaries', icon: AlignLeft },
    ],
  },
]

const pathOf = (href: string) => href.split(/[?#]/)[0]

/**
 * The one entry to highlight for `pathname`: the entry whose path is the
 * longest prefix of it (so `/knowledge-graph/triage` is Triage, not Graph),
 * and on a tie the one without a query or anchor (on `/ontologies`,
 * Ontologies rather than Retype; on `/`, The Tray rather than Schema
 * proposals).
 */
export function activeNavHref(pathname: string): string | undefined {
  let best: NavItem | undefined
  for (const item of NAV_SECTIONS.flatMap((s) => s.items)) {
    const path = pathOf(item.href)
    const matches =
      path === '/' ? pathname === '/' : pathname === path || pathname.startsWith(`${path}/`)
    if (!matches) continue
    if (
      !best ||
      path.length > pathOf(best.href).length ||
      (path.length === pathOf(best.href).length && best.href !== pathOf(best.href))
    ) {
      best = item
    }
  }
  return best?.href
}
