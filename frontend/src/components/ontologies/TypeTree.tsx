'use client'

/**
 * OE.3: the ontology's entity types as a nested list; an OWN type can be
 * dragged onto another type (its new parent) or onto "No parent". Inherited
 * types (labelled with the ontology that defines them) and placeholders
 * (parents defined nowhere, e.g. a schema.org base) are targets only. Every
 * drop is an edit of the DRAFT — nothing takes effect before a publish — and
 * the server's gate re-validates it.
 *
 * Keyboard (owner: "Ctrl-X, Ctrl-V is fine"): focus an own type's name,
 * Ctrl+X (Cmd+X) to cut it, focus the new parent's name — or "No parent" — and
 * Ctrl+V to move it there; Escape cancels. The drag handle also works with the
 * keyboard (Space, arrow keys, Space). Both end in the same draft edit.
 */

import {
  DndContext,
  KeyboardSensor,
  MeasuringStrategy,
  PointerSensor,
  useDraggable,
  useDroppable,
  useSensor,
  useSensors,
  type DragEndEvent,
  type DragStartEvent,
} from '@dnd-kit/core'
import { GripVertical } from 'lucide-react'
import { useState } from 'react'

import type { StoredEntityType } from '@/lib/api/ontologies'
import {
  ROOT_ZONE,
  canDrop,
  childrenOf,
  dropEdit,
  zoneId,
  type TypeTree as Tree,
} from '@/lib/utils/ontology-tree'
import { cn } from '@/lib/utils'

interface TypeTreeProps {
  tree: Tree
  own: Record<string, StoredEntityType>
  selected: string | null
  onSelect: (name: string) => void
  onReparent: (type: StoredEntityType) => void
  /** Blocking findings per type name (lower case), shown next to the type. */
  findingsByType?: Map<string, number>
  disabled?: boolean
}

export function TypeTree({
  tree, own, selected, onSelect, onReparent, findingsByType, disabled,
}: TypeTreeProps) {
  const [dragging, setDragging] = useState<string | null>(null)
  const [cut, setCut] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor),
  )

  const apply = (dragged: string, zone: string | null) => {
    const edit = dropEdit(tree, own, dragged, zone)
    if (edit) {
      setNotice(null)
      onReparent(edit)
    } else if (zone) {
      setNotice(`${dragged} cannot move there (itself, one of its own subtypes, or its current parent).`)
    }
  }
  const onDragStart = (event: DragStartEvent) => setDragging(String(event.active.id))
  const onDragEnd = (event: DragEndEvent) => {
    setDragging(null)
    if (event.over) apply(String(event.active.id), String(event.over.id))
  }
  const onKeys = (name: string | null) => (event: React.KeyboardEvent) => {
    const mod = event.ctrlKey || event.metaKey
    if (mod && event.key.toLowerCase() === 'x' && name) {
      event.preventDefault()
      if (disabled) return
      if (tree.nodes.get(name.toLowerCase())?.own) {
        setCut(name)
        setNotice(null)
      } else {
        setNotice(`${name} belongs to another ontology and cannot be moved here.`)
      }
    } else if (mod && event.key.toLowerCase() === 'v' && cut) {
      event.preventDefault()
      if (disabled) return
      const moving = cut
      setCut(null)
      apply(moving, name === null ? ROOT_ZONE : zoneId(name))
    } else if (event.key === 'Escape' && cut) {
      setCut(null)
    }
  }
  const moving = dragging ?? cut

  return (
    <DndContext sensors={sensors} onDragStart={onDragStart} onDragEnd={onDragEnd}
      onDragCancel={() => setDragging(null)}
      // Drop zones are enabled only once a drag starts (they depend on what is
      // dragged), after dnd-kit's start-of-drag measuring: measure while
      // dragging, or the keyboard sensor finds no target.
      measuring={{ droppable: { strategy: MeasuringStrategy.Always } }}>
      <div className="space-y-2" data-testid="ontology-type-tree">
        <RootZone active={moving !== null && canDrop(tree, moving, null)} onKeyDown={onKeys(null)} />
        {(cut || notice) && (
          <p className="text-xs text-muted-foreground" role="status" data-testid="ontology-cut-state">
            {cut ? `${cut} cut — focus its new parent and press Ctrl+V, or Escape to cancel` : notice}
          </p>
        )}
        <ul aria-label="Entity types" className="space-y-0.5">
          {tree.roots.map((name) => (
            <Node key={name} name={name} tree={tree} depth={0} moving={moving}
              selected={selected} onSelect={onSelect} disabled={disabled}
              cut={cut} onKeys={onKeys} findingsByType={findingsByType} />
          ))}
        </ul>
      </div>
    </DndContext>
  )
}

function RootZone({ active, onKeyDown }: {
  active: boolean
  onKeyDown: (event: React.KeyboardEvent) => void
}) {
  const { setNodeRef, isOver } = useDroppable({ id: ROOT_ZONE, disabled: !active })
  return (
    <div ref={setNodeRef} data-testid="ontology-root-zone" tabIndex={0} onKeyDown={onKeyDown}
      role="button" aria-label="No parent — paste a cut type here to make it a root"
      className={cn('rounded-md border border-dashed px-3 py-2 text-xs text-muted-foreground',
        active ? 'border-primary/60' : 'opacity-50', isOver && active && 'bg-primary/10')}>
      No parent — drop or paste here to make a type a root
    </div>
  )
}

interface NodeProps {
  name: string
  tree: Tree
  depth: number
  moving: string | null
  selected: string | null
  onSelect: (name: string) => void
  disabled?: boolean
  cut: string | null
  onKeys: (name: string | null) => (event: React.KeyboardEvent) => void
  findingsByType?: Map<string, number>
}

function Node(props: NodeProps) {
  const { name, tree, depth, moving, selected, onSelect, disabled, cut, onKeys, findingsByType } = props
  const node = tree.nodes.get(name.toLowerCase())
  const accepts = moving !== null && canDrop(tree, moving, name)
  const drop = useDroppable({ id: zoneId(name), disabled: !accepts })
  const drag = useDraggable({ id: name, disabled: disabled || !node?.own })
  if (!node) return null
  const kids = childrenOf(tree, name)
  const blocking = findingsByType?.get(name.toLowerCase()) ?? 0

  return (
    <li>
      <div
        ref={drop.setNodeRef}
        data-testid={`ontology-type-${name}`}
        style={{ paddingLeft: depth * 16 }}
        className={cn(
          'flex items-center gap-1 rounded px-1 py-0.5 text-sm',
          selected === name && 'bg-muted',
          accepts && 'outline outline-1 outline-dashed outline-primary/50',
          drop.isOver && accepts && 'bg-primary/10',
          (drag.isDragging || cut === name) && 'opacity-40',
        )}
      >
        {node.own ? (
          <button
            ref={drag.setNodeRef}
            {...drag.listeners}
            {...drag.attributes}
            type="button"
            aria-label={`Move ${name}`}
            className="cursor-grab text-muted-foreground hover:text-foreground disabled:cursor-default"
            disabled={disabled}
          >
            <GripVertical className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        ) : (
          <span className="w-3.5" aria-hidden="true" />
        )}
        <button type="button" onClick={() => onSelect(name)} onKeyDown={onKeys(name)}
          aria-current={selected === name ? 'true' : undefined}
          className={cn('text-left', !node.own && 'text-muted-foreground',
            node.placeholder && 'italic')}>
          {name}
        </button>
        {node.definedIn && (
          <span className="text-[10px] text-muted-foreground">from {node.definedIn}</span>
        )}
        {node.placeholder && <span className="text-[10px] text-muted-foreground">not defined</span>}
        {node.inCycle && (
          <span className="text-[10px] text-destructive" data-testid={`ontology-cycle-${name}`}>
            parent cycle
          </span>
        )}
        {blocking > 0 && (
          <span className="rounded bg-destructive/10 px-1 text-[10px] text-destructive"
            data-testid={`ontology-type-findings-${name}`}>
            {blocking} blocking
          </span>
        )}
      </div>
      {kids.length > 0 && (
        <ul className="space-y-0.5">
          {kids.map((child) => (
            <Node key={child} {...props} name={child} depth={depth + 1} />
          ))}
        </ul>
      )}
    </li>
  )
}
