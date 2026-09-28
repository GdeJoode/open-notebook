'use client'

/**
 * OE.3: the ontology's entity types as a tree; an OWN type can be dragged onto
 * another type (its new parent) or onto "No parent". Inherited and placeholder
 * types are drop targets only. Every drop is an edit of the DRAFT — nothing
 * takes effect before a publish — and the server's gate re-validates it.
 *
 * Keyboard: focus a type's handle, Space to pick it up, arrow keys to move,
 * Space to drop, Escape to cancel (dnd-kit's keyboard sensor).
 */

import {
  DndContext,
  KeyboardSensor,
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
  disabled?: boolean
}

export function TypeTree({ tree, own, selected, onSelect, onReparent, disabled }: TypeTreeProps) {
  const [dragging, setDragging] = useState<string | null>(null)
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor),
  )

  const onDragStart = (event: DragStartEvent) => setDragging(String(event.active.id))
  const onDragEnd = (event: DragEndEvent) => {
    const edit = dropEdit(tree, own, String(event.active.id), event.over ? String(event.over.id) : null)
    setDragging(null)
    if (edit) onReparent(edit)
  }

  return (
    <DndContext sensors={sensors} onDragStart={onDragStart} onDragEnd={onDragEnd}
      onDragCancel={() => setDragging(null)}>
      <div className="space-y-2" data-testid="ontology-type-tree">
        <RootZone active={dragging !== null && canDrop(tree, dragging, null)} />
        <ul role="tree" aria-label="Entity types" className="space-y-0.5">
          {tree.roots.map((name) => (
            <Node key={name} name={name} tree={tree} depth={0} dragging={dragging}
              selected={selected} onSelect={onSelect} disabled={disabled} />
          ))}
        </ul>
      </div>
    </DndContext>
  )
}

function RootZone({ active }: { active: boolean }) {
  const { setNodeRef, isOver } = useDroppable({ id: ROOT_ZONE, disabled: !active })
  return (
    <div ref={setNodeRef} data-testid="ontology-root-zone"
      className={cn('rounded-md border border-dashed px-3 py-2 text-xs text-muted-foreground',
        active ? 'border-primary/60' : 'opacity-50', isOver && active && 'bg-primary/10')}>
      No parent — drop here to make a type a root
    </div>
  )
}

interface NodeProps {
  name: string
  tree: Tree
  depth: number
  dragging: string | null
  selected: string | null
  onSelect: (name: string) => void
  disabled?: boolean
}

function Node({ name, tree, depth, dragging, selected, onSelect, disabled }: NodeProps) {
  const node = tree.nodes.get(name.toLowerCase())
  const accepts = dragging !== null && canDrop(tree, dragging, name)
  const drop = useDroppable({ id: zoneId(name), disabled: !accepts })
  const drag = useDraggable({ id: name, disabled: disabled || !node?.own })
  if (!node) return null
  const kids = childrenOf(tree, name)

  return (
    <li role="treeitem" aria-selected={selected === name} aria-expanded={kids.length ? true : undefined}>
      <div
        ref={drop.setNodeRef}
        data-testid={`ontology-type-${name}`}
        style={{ paddingLeft: depth * 16 }}
        className={cn(
          'flex items-center gap-1 rounded px-1 py-0.5 text-sm',
          selected === name && 'bg-muted',
          accepts && 'outline outline-1 outline-dashed outline-primary/50',
          drop.isOver && accepts && 'bg-primary/10',
          drag.isDragging && 'opacity-40',
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
        <button type="button" onClick={() => onSelect(name)}
          className={cn('text-left', !node.own && 'text-muted-foreground',
            node.placeholder && 'italic')}>
          {name}
        </button>
        {!node.own && !node.placeholder && (
          <span className="text-[10px] text-muted-foreground">inherited</span>
        )}
        {node.placeholder && <span className="text-[10px] text-muted-foreground">base</span>}
      </div>
      {kids.length > 0 && (
        <ul role="group" className="space-y-0.5">
          {kids.map((child) => (
            <Node key={child} name={child} tree={tree} depth={depth + 1} dragging={dragging}
              selected={selected} onSelect={onSelect} disabled={disabled} />
          ))}
        </ul>
      )}
    </li>
  )
}
