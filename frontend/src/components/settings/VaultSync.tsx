'use client'

import { useState, useCallback } from 'react'
import { RefreshCw, Download, Upload, Check, X, BookOpen } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import {
  usePendingImports,
  usePendingExports,
  useVaultSync,
  useApplyImports,
  useWriteExports,
} from '@/lib/hooks/use-vault'
import type { VaultImportItem, VaultExportItem } from '@/lib/api/vault'

export function VaultSync() {
  const { data: pendingImports = [], isLoading: loadingImports } = usePendingImports()
  const { data: pendingExports = [], isLoading: loadingExports } = usePendingExports()
  const syncMutation = useVaultSync()
  const applyMutation = useApplyImports()
  const writeMutation = useWriteExports()

  const [selectedImports, setSelectedImports] = useState<Set<string>>(new Set())
  const [selectedExports, setSelectedExports] = useState<Set<string>>(new Set())

  const toggleImport = useCallback((id: string) => {
    setSelectedImports(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const toggleExport = useCallback((id: string) => {
    setSelectedExports(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const selectAllImports = useCallback(() => {
    setSelectedImports(new Set(pendingImports.map(i => i.id)))
  }, [pendingImports])

  const selectAllExports = useCallback(() => {
    setSelectedExports(new Set(pendingExports.map(e => e.id)))
  }, [pendingExports])

  const handleApplyImports = useCallback(() => {
    const allIds = pendingImports.map(i => i.id)
    const acceptedIds = allIds.filter(id => selectedImports.has(id))
    const rejectedIds = allIds.filter(id => !selectedImports.has(id))
    applyMutation.mutate({ acceptedIds, rejectedIds })
    setSelectedImports(new Set())
  }, [pendingImports, selectedImports, applyMutation])

  const handleWriteExports = useCallback(() => {
    const allIds = pendingExports.map(e => e.id)
    const acceptedIds = allIds.filter(id => selectedExports.has(id))
    const rejectedIds = allIds.filter(id => !selectedExports.has(id))
    writeMutation.mutate({ acceptedIds, rejectedIds })
    setSelectedExports(new Set())
  }, [pendingExports, selectedExports, writeMutation])

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <BookOpen className="h-5 w-5" />
          Vault Sync
        </CardTitle>
        <Button
          variant="outline"
          size="sm"
          onClick={() => syncMutation.mutate()}
          disabled={syncMutation.isPending}
        >
          {syncMutation.isPending ? (
            <LoadingSpinner />
          ) : (
            <RefreshCw className="h-4 w-4 mr-2" />
          )}
          Sync Now
        </Button>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="imports">
          <TabsList>
            <TabsTrigger value="imports" className="gap-2">
              <Download className="h-3.5 w-3.5" />
              Import from Vault
              {pendingImports.length > 0 && (
                <Badge variant="secondary" className="ml-1 text-xs">
                  {pendingImports.length}
                </Badge>
              )}
            </TabsTrigger>
            <TabsTrigger value="exports" className="gap-2">
              <Upload className="h-3.5 w-3.5" />
              Export to Vault
              {pendingExports.length > 0 && (
                <Badge variant="secondary" className="ml-1 text-xs">
                  {pendingExports.length}
                </Badge>
              )}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="imports" className="mt-4">
            {loadingImports ? (
              <div className="flex justify-center py-8"><LoadingSpinner /></div>
            ) : pendingImports.length === 0 ? (
              <p className="text-sm text-muted-foreground py-4">
                No pending imports. Click "Sync Now" to scan your vault.
              </p>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Button variant="ghost" size="sm" onClick={selectAllImports}>
                    Select all
                  </Button>
                  <Button
                    size="sm"
                    onClick={handleApplyImports}
                    disabled={applyMutation.isPending}
                  >
                    {applyMutation.isPending ? <LoadingSpinner /> : <Check className="h-4 w-4 mr-1" />}
                    Apply ({selectedImports.size} accepted, {pendingImports.length - selectedImports.size} rejected)
                  </Button>
                </div>
                <div className="rounded-md border max-h-80 overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-muted/80 backdrop-blur">
                      <tr className="border-b">
                        <th className="w-10 px-3 py-2"></th>
                        <th className="text-left px-3 py-2 font-medium">Entity</th>
                        <th className="text-left px-3 py-2 font-medium">Aliases</th>
                        <th className="text-left px-3 py-2 font-medium">Type</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pendingImports.map((item: VaultImportItem) => (
                        <tr
                          key={item.id}
                          className="border-b cursor-pointer hover:bg-muted/50"
                          onClick={() => toggleImport(item.id)}
                        >
                          <td className="px-3 py-2">
                            <input
                              type="checkbox"
                              checked={selectedImports.has(item.id)}
                              onChange={() => toggleImport(item.id)}
                              className="rounded"
                            />
                          </td>
                          <td className="px-3 py-2 font-medium">{item.canonical_name}</td>
                          <td className="px-3 py-2 text-muted-foreground">
                            {item.aliases.slice(0, 3).join(', ')}
                            {item.aliases.length > 3 && ` +${item.aliases.length - 3}`}
                          </td>
                          <td className="px-3 py-2">
                            <Badge variant="outline" className="text-xs">{item.entity_type}</Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </TabsContent>

          <TabsContent value="exports" className="mt-4">
            {loadingExports ? (
              <div className="flex justify-center py-8"><LoadingSpinner /></div>
            ) : pendingExports.length === 0 ? (
              <p className="text-sm text-muted-foreground py-4">
                No entities pending export. Run entity extraction to discover new entities.
              </p>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Button variant="ghost" size="sm" onClick={selectAllExports}>
                    Select all
                  </Button>
                  <Button
                    size="sm"
                    onClick={handleWriteExports}
                    disabled={writeMutation.isPending}
                  >
                    {writeMutation.isPending ? <LoadingSpinner /> : <Upload className="h-4 w-4 mr-1" />}
                    Write to Vault ({selectedExports.size} selected)
                  </Button>
                </div>
                <div className="rounded-md border max-h-80 overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-muted/80 backdrop-blur">
                      <tr className="border-b">
                        <th className="w-10 px-3 py-2"></th>
                        <th className="text-left px-3 py-2 font-medium">Entity</th>
                        <th className="text-left px-3 py-2 font-medium">Type</th>
                        <th className="text-left px-3 py-2 font-medium">Sources</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pendingExports.map((item: VaultExportItem) => (
                        <tr
                          key={item.id}
                          className="border-b cursor-pointer hover:bg-muted/50"
                          onClick={() => toggleExport(item.id)}
                        >
                          <td className="px-3 py-2">
                            <input
                              type="checkbox"
                              checked={selectedExports.has(item.id)}
                              onChange={() => toggleExport(item.id)}
                              className="rounded"
                            />
                          </td>
                          <td className="px-3 py-2 font-medium">{item.canonical_name}</td>
                          <td className="px-3 py-2">
                            <Badge variant="outline" className="text-xs">{item.entity_type}</Badge>
                          </td>
                          <td className="px-3 py-2 text-muted-foreground text-xs">
                            {item.source_ids.length} source{item.source_ids.length !== 1 ? 's' : ''}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
