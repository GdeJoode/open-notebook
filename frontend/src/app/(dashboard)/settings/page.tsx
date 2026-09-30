'use client'

import { SettingsForm } from './components/SettingsForm'
import { SystemInfo } from '@/components/settings/SystemInfo'
import { RebuildEmbeddings } from '@/components/settings/RebuildEmbeddings'
import { VaultSync } from '@/components/settings/VaultSync'
import { ZoteroSettings } from '@/components/settings/ZoteroSettings'
import { ApiKeys } from '@/components/settings/ApiKeys'
import { FileWatcher } from '@/components/settings/FileWatcher'
import { useSettings } from '@/lib/hooks/use-settings'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { RefreshCw } from 'lucide-react'
import { useSearchParams } from 'next/navigation'
import { PageBody } from '@/components/layout/PageBody'
import { PageHeader } from '@/components/layout/PageHeader'

const TABS = ['general', 'vault', 'zotero', 'api-keys', 'watcher', 'advanced']

export default function SettingsPage() {
  const { refetch } = useSettings()
  // `?tab=advanced`: the embedding-model change dialog sends the owner to
  // Rebuild Embeddings.
  const tab = useSearchParams().get('tab')

  return (
    <PageBody width="form">
      <PageHeader
        title="Settings"
        actions={
          <Button variant="outline" size="sm" onClick={() => refetch()} aria-label="Refresh settings">
            <RefreshCw className="h-4 w-4" />
          </Button>
        }
      />
      <Tabs defaultValue={tab && TABS.includes(tab) ? tab : 'general'}>
        <TabsList>
          <TabsTrigger value="general">General</TabsTrigger>
          <TabsTrigger value="vault">Vault</TabsTrigger>
          <TabsTrigger value="zotero">Zotero</TabsTrigger>
          <TabsTrigger value="api-keys">API Keys</TabsTrigger>
          <TabsTrigger value="watcher">Watcher</TabsTrigger>
          <TabsTrigger value="advanced">Advanced</TabsTrigger>
        </TabsList>
        <TabsContent value="general" className="mt-6">
          <SettingsForm />
        </TabsContent>
        <TabsContent value="vault" className="mt-6">
          <VaultSync />
        </TabsContent>
        <TabsContent value="zotero" className="mt-6">
          <ZoteroSettings />
        </TabsContent>
        <TabsContent value="api-keys" className="mt-6">
          <ApiKeys />
        </TabsContent>
        <TabsContent value="watcher" className="mt-6">
          <FileWatcher />
        </TabsContent>
        <TabsContent value="advanced" className="mt-6 space-y-6">
          <SystemInfo />
          <RebuildEmbeddings />
        </TabsContent>
      </Tabs>
    </PageBody>
  )
}
