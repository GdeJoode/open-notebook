'use client'

import { useState } from 'react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import {
  KeyRound,
  Loader2,
  Plus,
  Copy,
  Check,
  Trash2,
  ScrollText,
  AlertTriangle,
  XCircle,
} from 'lucide-react'
import {
  useAgentKeys,
  useMintAgentKey,
  useRevokeAgentKey,
  useKeyAuditLog,
} from '@/lib/hooks/use-agent-keys'
import type {
  AgentKey,
  AgentKeyCreated,
  AgentPermission,
} from '@/lib/api/agent-keys'

const PERMISSION_VARIANT: Record<AgentPermission, string> = {
  read: 'bg-slate-500',
  write: 'bg-blue-600',
  admin: 'bg-amber-600',
}

function formatWhen(value?: string | null): string {
  if (!value) return '—'
  const d = new Date(value)
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString()
}

// ---------------------------------------------------------------------------
// Generate-key dialog: mint → show the plaintext ONCE → clear on close.
// ---------------------------------------------------------------------------

function GenerateKeyDialog() {
  const mint = useMintAgentKey()
  const [open, setOpen] = useState(false)
  const [agentId, setAgentId] = useState('')
  const [permission, setPermission] = useState<AgentPermission>('read')
  const [label, setLabel] = useState('')
  const [rpm, setRpm] = useState('')
  const [minted, setMinted] = useState<AgentKeyCreated | null>(null)
  const [copied, setCopied] = useState(false)

  const reset = () => {
    setAgentId('')
    setPermission('read')
    setLabel('')
    setRpm('')
    setMinted(null)
    setCopied(false)
    mint.reset()
  }

  const onOpenChange = (next: boolean) => {
    setOpen(next)
    if (!next) reset() // closing drops the plaintext — never shown again
  }

  const handleGenerate = () => {
    mint.mutate(
      {
        agent_id: agentId.trim(),
        permission,
        label: label.trim() || undefined,
        rate_limit_rpm: rpm ? parseInt(rpm, 10) : undefined,
      },
      { onSuccess: (created) => setMinted(created) },
    )
  }

  const handleCopy = async () => {
    if (!minted) return
    try {
      await navigator.clipboard?.writeText(minted.key)
      setCopied(true)
    } catch {
      setCopied(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button className="gap-2" data-testid="api-keys-generate">
          <Plus className="h-4 w-4" />
          Generate key
        </Button>
      </DialogTrigger>
      <DialogContent aria-label="Generate API key">
        {minted ? (
          // ---- show-once reveal --------------------------------------------
          <div data-testid="api-key-reveal">
            <DialogHeader>
              <DialogTitle>API key created</DialogTitle>
              <DialogDescription>
                Copy it now — for security it is shown only once and cannot be
                retrieved again.
              </DialogDescription>
            </DialogHeader>
            <Alert className="my-4">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                Store this key somewhere safe. You won&apos;t see it again after
                closing this dialog.
              </AlertDescription>
            </Alert>
            <div className="flex items-center gap-2">
              <code
                className="flex-1 truncate rounded-md border bg-muted px-3 py-2 font-mono text-sm"
                data-testid="api-key-plaintext"
              >
                {minted.key}
              </code>
              <Button
                variant="outline"
                size="icon"
                onClick={handleCopy}
                aria-label="Copy API key"
                data-testid="api-key-copy"
              >
                {copied ? (
                  <Check className="h-4 w-4 text-green-600" />
                ) : (
                  <Copy className="h-4 w-4" />
                )}
              </Button>
            </div>
            <DialogFooter className="mt-4">
              <Button onClick={() => onOpenChange(false)}>Done</Button>
            </DialogFooter>
          </div>
        ) : (
          // ---- mint form ---------------------------------------------------
          <div>
            <DialogHeader>
              <DialogTitle>Generate an API key</DialogTitle>
              <DialogDescription>
                Mint a key for an external agent. Scope: read &lt; write &lt; admin.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="agent-id">Agent ID</Label>
                <Input
                  id="agent-id"
                  value={agentId}
                  onChange={(e) => setAgentId(e.target.value)}
                  placeholder="e.g. research-bot"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="permission">Permission</Label>
                  <select
                    id="permission"
                    value={permission}
                    onChange={(e) =>
                      setPermission(e.target.value as AgentPermission)
                    }
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  >
                    <option value="read">read</option>
                    <option value="write">write</option>
                    <option value="admin">admin</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="rpm">Rate limit (rpm, optional)</Label>
                  <Input
                    id="rpm"
                    type="number"
                    value={rpm}
                    onChange={(e) => setRpm(e.target.value)}
                    placeholder="default"
                    min={1}
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="label">Label (optional)</Label>
                <Input
                  id="label"
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                  placeholder="e.g. CI ingestion bot"
                />
              </div>
              {mint.isError && (
                <Alert variant="destructive">
                  <XCircle className="h-4 w-4" />
                  <AlertDescription>
                    {(mint.error as Error)?.message || 'Could not mint key'}
                  </AlertDescription>
                </Alert>
              )}
            </div>
            <DialogFooter>
              <Button
                onClick={handleGenerate}
                disabled={!agentId.trim() || mint.isPending}
                className="gap-2"
                data-testid="api-key-generate-submit"
              >
                {mint.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <KeyRound className="h-4 w-4" />
                )}
                {mint.isPending ? 'Generating…' : 'Generate key'}
              </Button>
            </DialogFooter>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}

// ---------------------------------------------------------------------------
// Per-key audit-log drawer (a dialog listing the agent's recent calls).
// ---------------------------------------------------------------------------

function AuditLogDialog({
  keyId,
  onOpenChange,
}: {
  keyId: string | null
  onOpenChange: (open: boolean) => void
}) {
  const { data, isLoading, isError } = useKeyAuditLog(keyId)

  return (
    <Dialog open={!!keyId} onOpenChange={onOpenChange}>
      <DialogContent aria-label="API key audit log">
        <DialogHeader>
          <DialogTitle>Audit log</DialogTitle>
          <DialogDescription>
            Recent API calls for this key&apos;s agent (newest first).
          </DialogDescription>
        </DialogHeader>
        <div className="max-h-[50vh] overflow-y-auto" data-testid="audit-entries">
          {isLoading ? (
            <div
              className="flex items-center justify-center py-8"
              role="status"
              aria-label="Loading audit log"
            >
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : isError ? (
            <p className="py-6 text-center text-sm text-destructive">
              Could not load the audit log.
            </p>
          ) : !data || data.entries.length === 0 ? (
            <p
              className="py-6 text-center text-sm text-muted-foreground"
              data-testid="audit-empty"
            >
              No calls recorded yet.
            </p>
          ) : (
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-muted-foreground">
                <tr>
                  <th className="py-1 pr-2">Method</th>
                  <th className="py-1 pr-2">Path</th>
                  <th className="py-1 pr-2">Status</th>
                  <th className="py-1">When</th>
                </tr>
              </thead>
              <tbody>
                {data.entries.map((e, i) => (
                  <tr key={i} className="border-t">
                    <td className="py-1 pr-2 font-mono">{e.method}</td>
                    <td className="py-1 pr-2 font-mono truncate max-w-[16rem]">
                      {e.path}
                    </td>
                    <td className="py-1 pr-2">{e.status}</td>
                    <td className="py-1 text-muted-foreground">
                      {formatWhen(e.ts)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

// ---------------------------------------------------------------------------
// One key row.
// ---------------------------------------------------------------------------

function KeyRow({
  agentKey,
  onViewAudit,
}: {
  agentKey: AgentKey
  onViewAudit: (id: string) => void
}) {
  const revoke = useRevokeAgentKey()

  return (
    <div
      className="flex items-center gap-3 border-t py-3"
      data-testid={`api-key-row-${agentKey.id}`}
    >
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="font-medium">{agentKey.agent_id}</span>
          <Badge
            className={`${PERMISSION_VARIANT[agentKey.permission]} text-white`}
          >
            {agentKey.permission}
          </Badge>
          {agentKey.revoked && (
            <Badge variant="outline" className="text-destructive">
              revoked
            </Badge>
          )}
        </div>
        <div className="mt-0.5 text-xs text-muted-foreground">
          <code className="font-mono">{agentKey.key_prefix}…</code>
          {agentKey.label ? ` · ${agentKey.label}` : ''} · created{' '}
          {formatWhen(agentKey.created)} · last used{' '}
          {formatWhen(agentKey.last_used_at)}
        </div>
      </div>

      <Button
        variant="ghost"
        size="sm"
        className="gap-2"
        onClick={() => onViewAudit(agentKey.id)}
        aria-label={`View audit log for ${agentKey.agent_id}`}
        data-testid={`audit-${agentKey.id}`}
      >
        <ScrollText className="h-4 w-4" />
        Audit log
      </Button>

      {!agentKey.revoked && (
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="gap-2 text-destructive hover:text-destructive"
              aria-label={`Revoke key for ${agentKey.agent_id}`}
              data-testid={`revoke-${agentKey.id}`}
            >
              <Trash2 className="h-4 w-4" />
              Revoke
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Revoke this API key?</AlertDialogTitle>
              <AlertDialogDescription>
                The key for <strong>{agentKey.agent_id}</strong> (
                <code className="font-mono">{agentKey.key_prefix}…</code>) will
                stop authenticating immediately. This cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                onClick={() => revoke.mutate(agentKey.id)}
                data-testid={`revoke-confirm-${agentKey.id}`}
              >
                Revoke
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// The settings tab.
// ---------------------------------------------------------------------------

export function ApiKeys() {
  const { data: keys, isLoading, isError } = useAgentKeys()
  const [auditKeyId, setAuditKeyId] = useState<string | null>(null)

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <Loader2
            className="h-6 w-6 animate-spin text-muted-foreground"
            role="status"
            aria-label="Loading API keys"
          />
        </CardContent>
      </Card>
    )
  }

  if (isError) {
    return (
      <Card>
        <CardContent
          className="py-12 text-center text-sm text-destructive"
          data-testid="api-keys-error"
        >
          Could not load API keys. Please retry.
        </CardContent>
      </Card>
    )
  }

  const list = keys ?? []

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <KeyRound className="h-5 w-5 text-muted-foreground" />
            <div>
              <CardTitle className="text-base">API Keys</CardTitle>
              <CardDescription>
                Mint and revoke keys that external agents use (via{' '}
                <code className="font-mono">X-API-Key</code>) to call the agent
                API. The plaintext is shown only once at creation.
              </CardDescription>
            </div>
          </div>
          <GenerateKeyDialog />
        </div>
      </CardHeader>
      <CardContent>
        {list.length === 0 ? (
          <div
            className="py-10 text-center text-sm text-muted-foreground"
            data-testid="api-keys-empty"
          >
            No API keys yet. Generate one to let an agent call the API.
          </div>
        ) : (
          <div data-testid="api-keys-list">
            {list.map((k) => (
              <KeyRow key={k.id} agentKey={k} onViewAudit={setAuditKeyId} />
            ))}
          </div>
        )}
      </CardContent>

      <AuditLogDialog
        keyId={auditKeyId}
        onOpenChange={(open) => !open && setAuditKeyId(null)}
      />
    </Card>
  )
}
