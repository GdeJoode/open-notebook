/**
 * OE.3: the entity-type tree the ontology editor drags types around in.
 *
 * Two kinds of type appear: the ontology's OWN types (from its draft — the
 * only ones an edit here may change, so the only draggable ones) and the types
 * it INHERITS, each labelled with the ontology that defines it (from every
 * ontology it `extends` or `includes` — drop targets only). A parent named but
 * defined nowhere (`GovernmentService`, a schema.org base the bridge maps by
 * name) becomes a placeholder node, so a type can be dropped under it.
 *
 * Names — and parents given by an ALIAS — are matched case-insensitively, like
 * the backend's `_find_definition`. A parent cycle in the data (the free-text
 * parent field can make one; the gate blocks its publish) is broken for display
 * — one member becomes a root marked `inCycle` — so every type stays reachable
 * and the cycle can be repaired.
 */

import type { StoredEntityType } from '@/lib/api/ontologies'

export interface TypeNode {
  name: string
  parent: string | null
  own: boolean
  placeholder: boolean
  /** The ontology an inherited type comes from. */
  definedIn: string | null
  aliases: string[]
  inCycle: boolean
}

export interface TypeTree {
  nodes: Map<string, TypeNode>
  children: Map<string, string[]>
  roots: string[]
}

export interface InheritedType {
  name: string
  parent_type?: string | null
  aliases?: string[] | null
  /** The ontology that defines it: introduces it, or redefines it. */
  definedIn: string
}

/** An ontology as the browse API returns it: inheritance already merged. */
export interface MergedOntology {
  extends?: string | null
  includes?: string[] | null
  entity_types: { name: string; parent_type?: string | null; aliases?: string[] | null }[]
}

/** The ontologies an ontology inherits from, as the registry merges them:
 *  `extends`, then each `includes` — and `includes` only with `extends`. */
export function parentsOf(o: { extends?: string | null; includes?: string[] | null }): string[] {
  if (!o.extends) return []
  return [o.extends, ...(o.includes ?? [])].filter((p, i, all) => all.indexOf(p) === i)
}

/** Every ontology reachable through `parentsOf`, nearest first; `missing`
 *  lists the ones not loaded yet (the caller loads them and asks again). */
export function inheritanceClosure(
  direct: string[],
  loaded: Record<string, MergedOntology | undefined>,
): { order: string[]; missing: string[] } {
  const order: string[] = []
  const missing: string[] = []
  const queue = [...direct]
  while (queue.length) {
    const name = queue.shift() as string
    if (order.includes(name) || missing.includes(name)) continue
    const o = loaded[name]
    if (!o) {
      missing.push(name)
      continue
    }
    order.push(name)
    queue.push(...parentsOf(o))
  }
  return { order, missing }
}

/**
 * The types an ontology inherits, each labelled with the ontology that
 * DEFINES it (review OE.3 major 2: the browse API returns a parent's MERGED
 * chain, so labelling by the direct parent named `instruments` for a type
 * `schema_core` defines). An ontology defines a type when its merged set has
 * it and none of ITS parents has it the same way — it introduced or
 * redefined it. The definition shown is the one this ontology sees: the
 * nearest direct parent's.
 */
export function inheritedTypes(
  direct: string[],
  loaded: Record<string, MergedOntology | undefined>,
): InheritedType[] {
  const { order } = inheritanceClosure(direct, loaded)
  const typeIn = (name: string, ontology: string) =>
    loaded[ontology]?.entity_types.find((t) => key(t.name) === key(name))
  const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)
  const definer = (name: string): string | undefined =>
    order.find((o) => {
      const here = typeIn(name, o)
      if (!here) return false
      return parentsOf(loaded[o] as MergedOntology).every((p) => {
        const there = typeIn(name, p)
        return !there || !same(there, here)
      })
    })
  const out: InheritedType[] = []
  const seen = new Set<string>()
  for (const parent of direct) {
    for (const t of loaded[parent]?.entity_types ?? []) {
      if (seen.has(key(t.name))) continue
      seen.add(key(t.name))
      out.push({ ...t, definedIn: definer(t.name) ?? parent })
    }
  }
  return out
}

const key = (name: string) => name.trim().toLowerCase()

export function buildTypeTree(
  own: Record<string, StoredEntityType>,
  inherited: InheritedType[],
): TypeTree {
  const nodes = new Map<string, TypeNode>()
  const put = (name: string, node: Omit<TypeNode, 'name' | 'inCycle'>) =>
    nodes.set(key(name), { name, inCycle: false, ...node })
  for (const t of inherited) {
    if (!nodes.has(key(t.name))) {
      put(t.name, { parent: t.parent_type ?? null, own: false, placeholder: false,
        definedIn: t.definedIn, aliases: t.aliases ?? [] })
    }
  }
  // An own type with an inherited name redefines it: the own one wins.
  for (const t of Object.values(own)) {
    put(t.name, { parent: t.parent_type ?? null, own: true, placeholder: false,
      definedIn: null, aliases: t.aliases ?? [] })
  }
  const byAlias = new Map<string, string>()
  for (const [k, node] of nodes) for (const a of node.aliases) byAlias.set(key(a), k)
  const resolve = (name: string) => (nodes.has(key(name)) ? key(name) : byAlias.get(key(name)))
  for (const node of [...nodes.values()]) {
    if (node.parent && resolve(node.parent) === undefined) {
      put(node.parent, { parent: null, own: false, placeholder: true, definedIn: null, aliases: [] })
    }
  }

  const children = new Map<string, string[]>()
  const roots: string[] = []
  const parentKey = new Map<string, string>()
  for (const [k, node] of nodes) {
    const p = node.parent ? resolve(node.parent) : undefined
    if (p !== undefined && p !== k) {
      parentKey.set(k, p)
      children.set(p, [...(children.get(p) ?? []), node.name])
    } else {
      roots.push(node.name)
    }
  }

  // Break parent cycles for display: whatever the roots do not reach.
  const reached = new Set<string>()
  const walk = (name: string) => {
    const todo = [key(name)]
    while (todo.length) {
      const k = todo.pop() as string
      if (reached.has(k)) continue
      reached.add(k)
      for (const child of children.get(k) ?? []) todo.push(key(child))
    }
  }
  roots.forEach(walk)
  for (const start of [...nodes.keys()].sort()) {
    if (reached.has(start)) continue
    // Follow the parents up to the first repeat: that node is ON the cycle. A
    // node merely hanging below the cycle keeps its real parent.
    const seen = new Set<string>()
    let k = start
    while (!seen.has(k)) {
      seen.add(k)
      k = parentKey.get(k) as string
    }
    const node = nodes.get(k) as TypeNode
    const p = parentKey.get(k) as string
    children.set(p, (children.get(p) ?? []).filter((c) => key(c) !== k))
    parentKey.delete(k)
    node.inCycle = true
    roots.push(node.name)
    walk(node.name)
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

/** What a finished drag or paste changes: the dragged own type with its new
 *  parent, or `null` when it is not an edit (dropped outside, or refused). */
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

/** A name already taken — case-insensitively, like the backend — by an own
 *  type (adding it would REPLACE that type: the edit API creates or replaces). */
export function nameTaken(own: Record<string, { name: string }>, name: string): boolean {
  return Object.values(own).some((t) => key(t.name) === key(name))
}
