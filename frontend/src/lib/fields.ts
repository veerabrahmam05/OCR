import type { SourceRef } from "@/lib/types"

export interface FieldRow {
  path: string
  value: unknown
}

function isFreeForm(data: Record<string, unknown>): data is { fields: { name: string; value: unknown }[] } {
  const fields = data.fields
  return (
    Object.keys(data).length === 1 &&
    Array.isArray(fields) &&
    fields.every((f) => typeof f === "object" && f !== null && "name" in f)
  )
}

/** Every non-null scalar value with its JSON path (mirrors backend linking.paths.leaf_paths). */
export function flattenData(data: Record<string, unknown>): FieldRow[] {
  if (isFreeForm(data)) {
    return data.fields.filter((f) => f.value != null).map((f) => ({ path: f.name, value: f.value }))
  }
  const rows: FieldRow[] = []
  const walk = (node: unknown, path: string) => {
    if (Array.isArray(node)) node.forEach((v, i) => walk(v, `${path}[${i}]`))
    else if (node !== null && typeof node === "object")
      Object.entries(node).forEach(([k, v]) => walk(v, path ? `${path}.${k}` : k))
    else if (node != null) rows.push({ path, value: node })
  }
  walk(data, "")
  return rows
}

function covers(sourcePath: string, leaf: string): boolean {
  return leaf === sourcePath || leaf.startsWith(`${sourcePath}.`) || leaf.startsWith(`${sourcePath}[`)
}

/** The source for a field: an exact match, else the most specific parent (e.g. line_items[0]). */
export function sourceFor(path: string, sources: SourceRef[]): SourceRef | undefined {
  return (
    sources.find((s) => s.field === path) ??
    sources
      .filter((s) => covers(s.field, path))
      .sort((a, b) => b.field.length - a.field.length)[0]
  )
}

export function formatValue(value: unknown): string {
  if (typeof value === "number") return value.toLocaleString(undefined, { maximumFractionDigits: 6 })
  if (typeof value === "boolean") return value ? "yes" : "no"
  return String(value)
}
