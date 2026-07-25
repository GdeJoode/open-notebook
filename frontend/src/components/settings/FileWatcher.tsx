'use client'

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import {
  FolderInput,
  Loader2,
  CheckCircle,
  XCircle,
  CircleSlash,
} from 'lucide-react'
import { useWatcherStatus } from '@/lib/hooks/use-watcher'

function formatWhen(value: string): string {
  const d = new Date(value)
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString()
}

export function FileWatcher() {
  const { data, isLoading, isError } = useWatcherStatus()

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <Loader2
            className="h-6 w-6 animate-spin text-muted-foreground"
            role="status"
            aria-label="Loading file-watcher status"
          />
        </CardContent>
      </Card>
    )
  }

  if (isError || !data) {
    return (
      <Card>
        <CardContent
          className="py-12 text-center text-sm text-destructive"
          data-testid="watcher-error"
        >
          Could not load the file-watcher status.
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <FolderInput className="h-5 w-5 text-muted-foreground" />
            <div>
              <CardTitle className="text-base">Inbox File-Watcher</CardTitle>
              <CardDescription>
                Auto-ingests files dropped into a watched inbox folder. Configured
                via environment (read at startup); this panel is read-only.
              </CardDescription>
            </div>
          </div>
          {data.enabled ? (
            <Badge className="gap-1 bg-green-600 text-white" data-testid="watcher-enabled">
              <CheckCircle className="h-3 w-3" />
              Enabled
            </Badge>
          ) : (
            <Badge variant="outline" className="gap-1" data-testid="watcher-disabled">
              <CircleSlash className="h-3 w-3" />
              Disabled
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {!data.enabled && (
          <Alert>
            <AlertDescription className="text-sm">
              The watcher is off. Set <code className="font-mono">FILE_WATCHER_ENABLED=true</code>{' '}
              (and restart) to enable auto-ingest.
            </AlertDescription>
          </Alert>
        )}

        <div className="grid gap-3 sm:grid-cols-2">
          <div className="rounded-md border p-3">
            <div className="text-xs text-muted-foreground mb-1">Watched paths</div>
            {data.paths.length ? (
              <ul className="space-y-0.5" data-testid="watcher-paths">
                {data.paths.map((p) => (
                  <li key={p} className="truncate font-mono text-sm">
                    {p}
                  </li>
                ))}
              </ul>
            ) : (
              <div className="text-sm text-muted-foreground">None</div>
            )}
          </div>
          <div className="rounded-md border p-3">
            <div className="text-xs text-muted-foreground mb-1">Default notebook</div>
            <div className="font-mono text-sm">
              {data.default_notebook_id ?? '— (none)'}
            </div>
            <div className="mt-2 text-xs text-muted-foreground">Debounce</div>
            <div className="text-sm">{data.debounce_seconds}s</div>
          </div>
        </div>

        <div>
          <div className="mb-2 text-sm font-medium">Recent activity</div>
          {data.recent.length === 0 ? (
            <div
              className="rounded-md border py-6 text-center text-sm text-muted-foreground"
              data-testid="watcher-activity-empty"
            >
              No files processed yet.
            </div>
          ) : (
            <div
              className="max-h-64 overflow-y-auto rounded-md border"
              data-testid="watcher-activity"
            >
              <table className="w-full text-sm">
                <thead className="text-left text-xs text-muted-foreground">
                  <tr>
                    <th className="p-2">File</th>
                    <th className="p-2">Status</th>
                    <th className="p-2">When</th>
                  </tr>
                </thead>
                <tbody>
                  {data.recent.map((a, i) => (
                    <tr key={`${a.name}-${i}`} className="border-t">
                      <td className="p-2 font-mono truncate max-w-[16rem]">
                        {a.name}
                      </td>
                      <td className="p-2">
                        {a.status === 'processed' ? (
                          <span className="inline-flex items-center gap-1 text-green-600">
                            <CheckCircle className="h-3 w-3" />
                            processed
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-destructive">
                            <XCircle className="h-3 w-3" />
                            errored
                          </span>
                        )}
                      </td>
                      <td className="p-2 text-muted-foreground">
                        {formatWhen(a.when)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
