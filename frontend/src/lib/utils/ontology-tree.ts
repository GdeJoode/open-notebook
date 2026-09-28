/**
 * OE.3: the entity-type tree the ontology editor drags types around in.
 *
 * Two kinds of type appear: the ontology's OWN types (from its draft — the
 * only ones an edit here may change, so the only draggable ones) and the types
 * it INHERITS (from the ontology it `extends` — drop targets only). A parent
 * named but defined nowhere (`GovernmentService`, a schema.org base the bridge
 * maps by name) becomes a placeholder node, so a type can be dropped under it.
 *
 * Names are matched case-insensitively, like the backend's `_find_definition`.
 */

import type { StoredEntityType } from '@/lib/api/ontologies'

export interface TypeNode {
  name: string
  parent: string | null
  own: boolean
  placeholder: boolean
}

export interface TypeTree {
  nodes: Map<string, TypeNode>
  children: Map<string, string[]>
  roots: string[]
}

const key = (name: string) => name.trim().toLowerCase()

export function buildTypeTree(
  own: Record<string, StoredEntityType>,
  inherited: { name: string; parent_type?: string | null }[],
): TypeTree {
  const nodes = new Map<string, TypeNode>()
  for (const t of inherited) {
    nodes.set(key(t.name), { name: t.name, parent: t.parent_type ?? null, own: false, placeholder: false })
  }
  // An own type with an inherited name redefines it: the own one wins.
  for (const t of Object.values(own)) {
    nodes.set(key(t.name), { name: t.name, parent: t.parent_type ?? null, own: true, placeholder: false })
  }
  for (const node of [...nodes.values()]) {
    if (node.parent && !nodes.has(key(node.parent))) {
      nodes.set(key(node.parent), { name: node.parent, parent: null, own: false, placeholder: true })
    }
  }
  const children = new Map<string, string[]>()
  const roots: string[] = []
  for (const node of nodes.values()) {
    if (node.parent && nodes.has(key(node.parent)) && key(node.parent) !== key(node.name)) {
      const k = key(node.parent)
      children.set(k, [...(children.get(k) ?? []), node.name])
    } else {
      roots.push(node.name)
    }
  }
  const byName = (a: string, b: string) => a.localeCompare(b)
  roots.sort(byName)
  for (const list of children.values()) list.sort(byName)
  return { nodes, children, roots }
}

export function childrenOf(tree: TypeTree, name: string): string[] {
  return tree.children.get(key(name)) ?? []
}

/** Every type below `name` (a cycle in the data cannot loop this). */
export function descendantsOf(tree: TypeTree, name: string): Set<string> {
  const out = new Set<string>()
  const todo = [...childrenOf(tree, name)]
  while (todo.length) {
    const next = todo.pop() as string
    if (out.has(key(next))) continue
    out.add(key(next))
    todo.push(...childrenOf(tree, next))
  }
  return out
}

/** A drop of `dragged` onto `target` (null = "no parent") the editor accepts:
 *  an own type, onto anything but itself, its own descendants (a cycle) or
 *  the parent it already has (a no-op). The server's gate still validates
 *  what the new parent does to typing. */
export function canDrop(tree: TypeTree, dragged: string, target: string | null): boolean {
  const node = tree.nodes.get(key(dragged))
  if (!node || !node.own) return false
  if (target === null) return node.parent !== null
  if (key(target) === key(dragged)) return false
  if (!tree.nodes.has(key(target))) return false
  if (descendantsOf(tree, dragged).has(key(target))) return false
  return key(node.parent ?? '') !== key(target)
}

/** The type with a new parent, every other key kept — the edit API replaces a
 *  type whole, so what is not sent is dropped. `null` removes the parent. */
export function reparented(type: StoredEntityType, target: string | null): StoredEntityType {
  const next: StoredEntityType = { ...type }
  if (target === null) delete next.parent_type
  else next.parent_type = target
  return next
}

/** Drop-zone id for "no parent" (a root). Every other zone is `type:<name>`. */
export const ROOT_ZONE = 'root'

export function zoneId(typeName: string): string {
  return `type:${typeName}`
}

/** What a finished drag changes: the dragged own type with its new parent, or
 *  `null` when the drop is not an edit (dropped outside, or refused). */
export function dropEdit(
  tree: TypeTree,
  own: Record<string, StoredEntityType>,
  dragged: string,
  zone: string | null,
): StoredEntityType | null {
  if (!zone) return null
  const target = zone === ROOT_ZONE ? null : zone.startsWith('type:') ? zone.slice(5) : undefined
  if (target === undefined || !canDrop(tree, dragged, target)) return null
  const type = Object.values(own).find((t) => key(t.name) === key(dragged))
  return type ? reparented(type, target) : null
}
