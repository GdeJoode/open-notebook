'use client'

import { useState } from 'react'
import { AppShell } from '@/components/layout/AppShell'
import { EmptyState } from '@/components/common/EmptyState'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { useOntologies, useOntology } from '@/lib/hooks/use-ontologies'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import { Network, ArrowLeft, ArrowRight } from 'lucide-react'
import type { OntologySummary } from '@/lib/api/ontologies'

export default function OntologiesPage() {
  const [selectedOntology, setSelectedOntology] = useState<string | null>(null)

  return (
    <AppShell>
      <div className="flex-1 overflow-y-auto">
        <div className="p-6">
          <div className="max-w-6xl">
            {selectedOntology ? (
              <OntologyDetail
                name={selectedOntology}
                onBack={() => setSelectedOntology(null)}
              />
            ) : (
              <OntologyList onSelect={setSelectedOntology} />
            )}
          </div>
        </div>
      </div>
    </AppShell>
  )
}

function OntologyList({ onSelect }: { onSelect: (name: string) => void }) {
  const { data: ontologies, isLoading, error } = useOntologies()

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <LoadingSpinner />
      </div>
    )
  }

  if (error) {
    return <p className="text-red-500">Failed to load ontologies</p>
  }

  if (!ontologies || ontologies.length === 0) {
    return (
      <>
        <h1 className="text-2xl font-bold mb-6">Ontologies</h1>
        <EmptyState
          icon={Network}
          title="No ontologies found"
          description="No ontology definitions are available yet."
        />
      </>
    )
  }

  return (
    <>
      <h1 className="text-2xl font-bold mb-6">Ontologies</h1>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {ontologies.map((o: OntologySummary) => (
          <Card
            key={o.name}
            className="p-5 cursor-pointer hover:border-primary/50 transition-colors"
            onClick={() => onSelect(o.name)}
          >
            <div className="flex items-start justify-between mb-3">
              <h3 className="font-semibold text-lg">{o.name}</h3>
              <Badge variant="outline">{o.version}</Badge>
            </div>
            {o.description && (
              <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                {o.description}
              </p>
            )}
            <div className="flex gap-3 text-xs text-muted-foreground">
              <span>{o.entity_type_count} entity types</span>
              <span>{o.relationship_type_count} relationships</span>
              <span>{o.concept_count} concepts</span>
            </div>
          </Card>
        ))}
      </div>
    </>
  )
}

function OntologyDetail({
  name,
  onBack,
}: {
  name: string
  onBack: () => void
}) {
  const { data: ontology, isLoading, error } = useOntology(name)

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <LoadingSpinner />
      </div>
    )
  }

  if (error || !ontology) {
    return <p className="text-red-500">Failed to load ontology</p>
  }

  return (
    <>
      <div className="flex items-center gap-3 mb-6">
        <Button variant="ghost" size="sm" onClick={onBack}>
          <ArrowLeft className="h-4 w-4 mr-1" />
          Back
        </Button>
        <h1 className="text-2xl font-bold">{ontology.name}</h1>
        <Badge variant="outline">{ontology.version}</Badge>
      </div>

      {ontology.description && (
        <p className="text-muted-foreground mb-6">{ontology.description}</p>
      )}

      <Tabs defaultValue="entity-types">
        <TabsList>
          <TabsTrigger value="entity-types">
            Entity Types ({ontology.entity_types.length})
          </TabsTrigger>
          <TabsTrigger value="relationships">
            Relationships ({ontology.relationship_types.length})
          </TabsTrigger>
          <TabsTrigger value="concepts">
            Concepts ({ontology.concepts.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="entity-types" className="mt-4">
          <Accordion type="multiple" className="space-y-2">
            {ontology.entity_types.map((et) => (
              <AccordionItem key={et.name} value={et.name} className="border rounded-lg px-4">
                <AccordionTrigger className="hover:no-underline">
                  <div className="flex items-center gap-3">
                    <span className="font-medium">{et.name}</span>
                    {et.parent_type && (
                      <Badge variant="secondary" className="text-xs">
                        extends {et.parent_type}
                      </Badge>
                    )}
                    <span className="text-xs text-muted-foreground">
                      {et.properties.length} properties
                    </span>
                  </div>
                </AccordionTrigger>
                <AccordionContent>
                  {et.description && (
                    <p className="text-sm text-muted-foreground mb-3">{et.description}</p>
                  )}

                  {et.extraction_hints && et.extraction_hints.length > 0 && (
                    <div className="mb-3">
                      <span className="text-xs font-medium">Extraction hints:</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {et.extraction_hints.map((h) => (
                          <Badge key={h} variant="outline" className="text-xs">
                            {h}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {et.examples && et.examples.length > 0 && (
                    <div className="mb-3">
                      <span className="text-xs font-medium">Examples:</span>
                      <span className="text-xs text-muted-foreground ml-1">
                        {et.examples.join(', ')}
                      </span>
                    </div>
                  )}

                  {et.properties.length > 0 && (
                    <div className="border rounded-md overflow-hidden">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b bg-muted/50">
                            <th className="text-left p-2 font-medium">Property</th>
                            <th className="text-left p-2 font-medium">Type</th>
                            <th className="text-left p-2 font-medium">Required</th>
                            <th className="text-left p-2 font-medium">Description</th>
                          </tr>
                        </thead>
                        <tbody>
                          {et.properties.map((p) => (
                            <tr key={p.name} className="border-b last:border-0">
                              <td className="p-2 font-mono text-xs">{p.name}</td>
                              <td className="p-2">
                                <Badge variant="secondary" className="text-xs">
                                  {p.data_type}
                                </Badge>
                              </td>
                              <td className="p-2">
                                {p.required && (
                                  <Badge variant="destructive" className="text-xs">
                                    required
                                  </Badge>
                                )}
                              </td>
                              <td className="p-2 text-muted-foreground text-xs">
                                {p.description}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </TabsContent>

        <TabsContent value="relationships" className="mt-4">
          <div className="space-y-3">
            {ontology.relationship_types.map((rt) => (
              <Card key={rt.name} className="p-4">
                <div className="flex items-center gap-3 mb-2">
                  <span className="font-medium">{rt.name}</span>
                  <Badge variant="outline" className="text-xs">
                    {rt.cardinality}
                  </Badge>
                  {rt.symmetric && (
                    <Badge variant="secondary" className="text-xs">symmetric</Badge>
                  )}
                  {rt.transitive && (
                    <Badge variant="secondary" className="text-xs">transitive</Badge>
                  )}
                </div>
                {rt.description && (
                  <p className="text-sm text-muted-foreground mb-2">{rt.description}</p>
                )}
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span className="font-medium">{rt.domain.join(', ')}</span>
                  <ArrowRight className="h-3 w-3" />
                  <span className="font-medium">{rt.range.join(', ')}</span>
                </div>
                {rt.inverse && (
                  <div className="text-xs text-muted-foreground mt-1">
                    Inverse: {rt.inverse}
                  </div>
                )}
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="concepts" className="mt-4">
          <div className="space-y-3">
            {ontology.concepts.map((c) => (
              <Card key={c.name} className="p-4">
                <h4 className="font-medium mb-1">{c.name}</h4>
                {c.description && (
                  <p className="text-sm text-muted-foreground mb-2">{c.description}</p>
                )}
                {c.related_concepts && c.related_concepts.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {c.related_concepts.map((rc) => (
                      <Badge key={rc} variant="outline" className="text-xs">
                        {rc}
                      </Badge>
                    ))}
                  </div>
                )}
              </Card>
            ))}
            {ontology.concepts.length === 0 && (
              <p className="text-sm text-muted-foreground text-center py-8">
                No concepts defined in this ontology.
              </p>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </>
  )
}
