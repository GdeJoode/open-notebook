'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import {
  BookMarked,
  RefreshCw,
  Loader2,
  CheckCircle,
  XCircle,
  Unplug,
  Clock,
  ArrowUpDown,
} from 'lucide-react'
import {
  useZoteroStatus,
  useZoteroConnect,
  useZoteroDisconnect,
  useZoteroSync,
} from '@/lib/hooks/use-zotero'
import type { ZoteroConnectConfig } from '@/lib/api/zotero'

export function ZoteroSettings() {
  const { data: status, isLoading } = useZoteroStatus()
  const connect = useZoteroConnect()
  const disconnect = useZoteroDisconnect()
  const sync = useZoteroSync()

  // Connection form state
  const [libraryId, setLibraryId] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [libraryType, setLibraryType] = useState('user')
  const [defaultCollection, setDefaultCollection] = useState('')
  const [autoPush, setAutoPush] = useState(false)
  const [syncInterval, setSyncInterval] = useState(15)

  const handleConnect = () => {
    const config: ZoteroConnectConfig = {
      library_id: libraryId,
      api_key: apiKey,
      library_type: libraryType,
      default_collection: defaultCollection || undefined,
      auto_push: autoPush,
      sync_interval_minutes: syncInterval,
    }
    connect.mutate(config)
  }

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    )
  }

  // Connected state
  if (status?.connected) {
    return (
      <div className="space-y-6">
        {/* Connection Status */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <BookMarked className="h-5 w-5 text-red-700" />
                <div>
                  <CardTitle className="text-base">Zotero Connected</CardTitle>
                  <CardDescription>
                    Library type: {status.library_type}
                    {status.default_collection && ` · Collection: ${status.default_collection}`}
                  </CardDescription>
                </div>
              </div>
              <Badge variant="default" className="bg-green-600 gap-1">
                <CheckCircle className="h-3 w-3" />
                Connected
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Sync status */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="border rounded-md p-3 text-center">
                <div className="text-xs text-muted-foreground mb-1">Library Version</div>
                <div className="text-lg font-semibold">{status.library_version ?? 0}</div>
              </div>
              <div className="border rounded-md p-3 text-center">
                <div className="text-xs text-muted-foreground mb-1">Sync Status</div>
                <div className="text-sm font-medium capitalize">{status.sync_status ?? 'idle'}</div>
              </div>
              <div className="border rounded-md p-3 text-center">
                <div className="text-xs text-muted-foreground mb-1">Auto Push</div>
                <div className="text-sm font-medium">{status.auto_push ? 'On' : 'Off'}</div>
              </div>
              <div className="border rounded-md p-3 text-center">
                <div className="text-xs text-muted-foreground mb-1">Last Sync</div>
                <div className="text-sm font-medium">
                  {status.last_sync
                    ? new Date(status.last_sync).toLocaleString()
                    : 'Never'}
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-3 pt-2">
              <Button
                onClick={() => sync.mutate()}
                disabled={sync.isPending}
                className="gap-2"
              >
                {sync.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <ArrowUpDown className="h-4 w-4" />
                )}
                {sync.isPending ? 'Syncing...' : 'Sync Now'}
              </Button>
              <Button
                variant="outline"
                onClick={() => disconnect.mutate()}
                disabled={disconnect.isPending}
                className="gap-2 text-destructive hover:text-destructive"
              >
                <Unplug className="h-4 w-4" />
                Disconnect
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Sync Info */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">How Sync Works</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-muted-foreground space-y-2">
              <li className="flex gap-2">
                <span className="text-primary font-medium">Push:</span>
                Use &quot;Push to Zotero&quot; on any source to create a Zotero item with the PDF uploaded to Zotero cloud storage.
              </li>
              <li className="flex gap-2">
                <span className="text-primary font-medium">Sync:</span>
                &quot;Sync Now&quot; performs bidirectional sync — pushes new local sources and pulls changes from Zotero.
              </li>
              <li className="flex gap-2">
                <span className="text-primary font-medium">Duplicates:</span>
                Before pushing, the system checks for existing items by DOI, ISBN, and title+author matching.
              </li>
              <li className="flex gap-2">
                <span className="text-primary font-medium">Files:</span>
                PDFs are uploaded to Zotero cloud storage and sync across all your devices.
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>
    )
  }

  // Not connected — show setup form
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <BookMarked className="h-5 w-5 text-muted-foreground" />
          <div>
            <CardTitle className="text-base">Connect to Zotero</CardTitle>
            <CardDescription>
              Sync your sources with Zotero for reference management. Files are uploaded to Zotero cloud storage.
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <Alert>
          <AlertDescription className="text-sm">
            To get your credentials, go to{' '}
            <a
              href="https://www.zotero.org/settings/keys"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary underline"
            >
              zotero.org/settings/keys
            </a>
            . Create a new API key with read/write access. Your Library ID is shown on the same page.
          </AlertDescription>
        </Alert>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="library-id">Library ID</Label>
            <Input
              id="library-id"
              value={libraryId}
              onChange={(e) => setLibraryId(e.target.value)}
              placeholder="e.g. 12345678"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="library-type">Library Type</Label>
            <select
              id="library-type"
              value={libraryType}
              onChange={(e) => setLibraryType(e.target.value)}
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            >
              <option value="user">Personal Library</option>
              <option value="group">Group Library</option>
            </select>
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="api-key">API Key</Label>
          <Input
            id="api-key"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Enter your Zotero API key"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="collection">Default Collection (optional)</Label>
          <Input
            id="collection"
            value={defaultCollection}
            onChange={(e) => setDefaultCollection(e.target.value)}
            placeholder="e.g. Open Notebook Imports"
          />
          <p className="text-xs text-muted-foreground">
            New items will be added to this collection. Created automatically if it doesn&apos;t exist.
          </p>
        </div>

        <div className="flex items-center justify-between border-t pt-4">
          <div className="flex items-center gap-2">
            <Switch
              checked={autoPush}
              onCheckedChange={setAutoPush}
            />
            <Label className="text-sm">Auto-push new sources to Zotero</Label>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <Label className="text-sm">Sync every</Label>
            <Input
              type="number"
              className="w-16 h-8 text-sm"
              value={syncInterval}
              onChange={(e) => setSyncInterval(parseInt(e.target.value) || 15)}
              min={1}
              max={1440}
            />
            <span className="text-sm text-muted-foreground">min</span>
          </div>
        </div>

        {connect.isError && (
          <Alert variant="destructive">
            <XCircle className="h-4 w-4" />
            <AlertDescription>
              {(connect.error as Error)?.message || 'Connection failed'}
            </AlertDescription>
          </Alert>
        )}

        <Button
          onClick={handleConnect}
          disabled={!libraryId || !apiKey || connect.isPending}
          className="w-full gap-2"
        >
          {connect.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <BookMarked className="h-4 w-4" />
          )}
          {connect.isPending ? 'Connecting...' : 'Connect to Zotero'}
        </Button>
      </CardContent>
    </Card>
  )
}
