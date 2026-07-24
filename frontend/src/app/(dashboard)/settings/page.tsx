'use client'

import { AppShell } from '@/components/layout/AppShell'
import { SettingsForm } from './components/SettingsForm'
import { SystemInfo } from '../advanced/components/SystemInfo'
import { RebuildEmbeddings } from '../advanced/components/RebuildEmbeddings'
import { VaultSync } from '@/components/settings/VaultSync'
import { ZoteroSettings } from '@/components/settings/ZoteroSettings'
import { ApiKeys } from '@/components/settings/ApiKeys'
import { useSettings } from '@/lib/hooks/use-settings'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { RefreshCw } from 'lucide-react'

export default function SettingsPage() {
  const { refetch } = useSettings()

  return (
    <AppShell>
      <div className="flex-1 overflow-y-auto">
        <div className="p-6">
          <div className="max-w-4xl">
            <div className="flex items-center gap-4 mb-6">
              <h1 className="text-2xl font-bold">Settings</h1>
              <Button variant="outline" size="sm" onClick={() => refetch()}>
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
            <Tabs defaultValue="general">
              <TabsList>
                <TabsTrigger value="general">General</TabsTrigger>
                <TabsTrigger value="vault">Vault</TabsTrigger>
                <TabsTrigger value="zotero">Zotero</TabsTrigger>
                <TabsTrigger value="api-keys">API Keys</TabsTrigger>
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
              <TabsContent value="advanced" className="mt-6 space-y-6">
                <SystemInfo />
                <RebuildEmbeddings />
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </AppShell>
  )
}
