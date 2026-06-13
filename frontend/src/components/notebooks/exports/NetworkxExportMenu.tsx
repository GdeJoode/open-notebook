'use client'

import { useState } from 'react'
import { Download, ChevronDown, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { useToast } from '@/lib/hooks/use-toast'

/**
 * NetworkX 7-format export dropdown (Phase D.3).
 *
 * Sits on the Schema tab next to the TTL download button (B.2b). Each
 * menu item POSTs to `/api/notebooks/{id}/export-networkx` with the
 * matching `format`, receives the blob, and stages a browser download.
 *
 * Filter defaults
 * ===============
 *
 * The D.0 `ExportFilter` defaults to `min_connections=5` /
 * `min_confidence=0.9` for a "publication-grade" subset. The popover
 * that lets users tune the sliders lands in D.1c; until then, this
 * component sends `min_connections=0` (per-format requests) so an
 * analyst clicking GraphML on a fresh notebook always gets *something*
 * back rather than an empty file from the high default. Confidence
 * defaults stay at 0.9 to keep noise out of the export.
 *
 * Accessibility
 * =============
 *
 * Radix `DropdownMenu` handles roving focus + Escape-to-close + Enter
 * to activate. Each menu item carries `data-testid` matching its format
 * literal so e2e specs can drive the flow.
 */
interface NetworkxExportMenuProps {
  notebookId: string
}

type NetworkxFormat =
  | 'graphml'
  | 'gexf'
  | 'gml'
  | 'json-tree'
  | 'edge-list'
  | 'adjacency-list'
  | 'pickle'

interface FormatOption {
  format: NetworkxFormat
  label: string
  description: string
  defaultFilename: string
}

// Single source of truth for format-specific UI metadata. Order matches
// the dropdown render order; descriptions are short tooltips users see
// in the menu.
const FORMAT_OPTIONS: FormatOption[] = [
  {
    format: 'graphml',
    label: 'GraphML',
    description: 'XML graph format (Gephi, Cytoscape).',
    defaultFilename: 'graph.graphml',
  },
  {
    format: 'gexf',
    label: 'GEXF',
    description: 'Gephi-native XML format.',
    defaultFilename: 'graph.gexf',
  },
  {
    format: 'gml',
    label: 'GML',
    description: 'Plain-text graph modelling language.',
    defaultFilename: 'graph.gml',
  },
  {
    format: 'json-tree',
    label: 'JSON tree',
    description: 'JSON tree/node-link representation.',
    defaultFilename: 'graph.json',
  },
  {
    format: 'edge-list',
    label: 'Edge list',
    description: 'Plain-text edge list (one edge per line).',
    defaultFilename: 'graph.edges',
  },
  {
    format: 'adjacency-list',
    label: 'Adjacency list',
    description: 'Plain-text adjacency list.',
    defaultFilename: 'graph.adj',
  },
  {
    format: 'pickle',
    label: 'Pickle',
    description: 'Python pickle (lossless, NetworkX-only).',
    defaultFilename: 'graph.pkl',
  },
]

export function NetworkxExportMenu({ notebookId }: NetworkxExportMenuProps) {
  // We track the format currently downloading so the trigger can show
  // a spinner. A single boolean would do, but tracking the literal lets
  // a future revision disable only the active item while leaving the
  // others clickable (queue semantics).
  const [activeFormat, setActiveFormat] = useState<NetworkxFormat | null>(null)
  const { toast } = useToast()

  const handleSelect = async (option: FormatOption) => {
    setActiveFormat(option.format)
    try {
      // Import the shared client lazily so the bearer-token interceptor
      // fires automatically -- same trick as the B.2b TTL button.
      const apiClient = (await import('@/lib/api/client')).default
      const response = await apiClient.post<Blob>(
        `/notebooks/${encodeURIComponent(notebookId)}/export-networkx`,
        {
          format: option.format,
          // Minimal filter: keep confidence at the V1 default but let
          // isolated nodes through so the export is never empty on a
          // notebook the user just opened. D.1c will expose the sliders.
          filter: { min_connections: 0, min_confidence: 0.9 },
        },
        { responseType: 'blob' },
      )

      const blob = response.data
      // Axios normalises response headers to lowercase, but route.fulfill
      // in some Playwright setups passes them through with the original
      // casing. Check both to keep mocks and real responses happy.
      const headers = (response.headers ?? {}) as Record<string, string>
      const disposition =
        headers['content-disposition'] ?? headers['Content-Disposition']
      const filename = parseFilename(disposition, option.defaultFilename)

      const objectUrl = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = objectUrl
      anchor.download = filename
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      URL.revokeObjectURL(objectUrl)
    } catch (err) {
      console.error(`Failed to download NetworkX ${option.format}`, err)
      toast({
        title: 'Export failed',
        description: `Could not export graph as ${option.label}. Please try again.`,
        variant: 'destructive',
      })
    } finally {
      setActiveFormat(null)
    }
  }

  const isDownloading = activeFormat !== null

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={isDownloading}
          aria-label="Export NetworkX graph"
          data-testid="networkx-export-trigger"
        >
          {isDownloading ? (
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          ) : (
            <Download className="h-4 w-4" aria-hidden="true" />
          )}
          <span>{isDownloading ? 'Exporting…' : 'Export NetworkX'}</span>
          <ChevronDown className="h-4 w-4" aria-hidden="true" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="w-64"
        data-testid="networkx-export-menu"
      >
        <DropdownMenuLabel>NetworkX format</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {FORMAT_OPTIONS.map((option) => (
          <DropdownMenuItem
            key={option.format}
            disabled={isDownloading}
            data-testid={`networkx-format-${option.format}`}
            onSelect={(event) => {
              // Radix calls preventDefault from the parent menu after
              // onSelect -- we still need to call it here so the menu
              // closes synchronously after the user click.
              event.preventDefault()
              void handleSelect(option)
            }}
            className="flex-col items-start gap-0.5"
          >
            <span className="text-sm font-medium">{option.label}</span>
            <span className="text-xs text-muted-foreground">
              {option.description}
            </span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/**
 * Extract `filename="..."` from a Content-Disposition header.
 *
 * Mirrors the B.2b TTL button helper but uses a per-format default so
 * the user never gets a bare `attachment` filename.
 */
function parseFilename(
  contentDisposition: string | undefined,
  fallback: string,
): string {
  if (contentDisposition) {
    const match = /filename="([^"]+)"/.exec(contentDisposition)
    if (match) {
      return match[1]
    }
  }
  return fallback
}
