"use client"

import { AlertTriangleIcon } from "lucide-react"

import { SOURCE_STATUS, SourceDot } from "@/components/status-badges"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { flattenData, formatValue, sourceFor } from "@/lib/fields"
import type { ExtractionResult, SourceRef, SourceStatus } from "@/lib/types"
import { cn } from "@/lib/utils"

interface Props {
  result: ExtractionResult
  selected: SourceRef | null
  onSelect: (source: SourceRef) => void
}

export function DataPanel({ result, selected, onSelect }: Props) {
  const rows = flattenData(result.data)
  const counts = countStatuses(result)

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {Object.entries(counts).map(([status, n]) => (
          <Tooltip key={status}>
            <TooltipTrigger render={<span className="flex cursor-default items-center gap-1.5" />}>
              <SourceDot status={status as SourceStatus | "unsourced"} />
              {n} {SOURCE_STATUS[status as SourceStatus | "unsourced"].label.toLowerCase()}
            </TooltipTrigger>
            <TooltipContent>{SOURCE_STATUS[status as SourceStatus | "unsourced"].hint}</TooltipContent>
          </Tooltip>
        ))}
      </div>

      {result.conflicts.length > 0 && (
        <Alert>
          <AlertTriangleIcon />
          <AlertTitle>Different values in different parts of the document</AlertTitle>
          <AlertDescription>
            <ul className="list-disc pl-4">
              {result.conflicts.map((c) => (
                <li key={c.field}>
                  <code>{c.field}</code>:{" "}
                  {c.values.map((v, i) => `${formatValue(v)} (pages ${c.pages[i]?.join(", ") || "?"})`).join(" vs ")}
                  . The first value was kept.
                </li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {result.warnings.length > 0 && (
        <Alert>
          <AlertTriangleIcon />
          <AlertTitle>Warnings</AlertTitle>
          <AlertDescription>
            <ul className="list-disc pl-4">
              {result.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {rows.length === 0 ? (
        <p className="text-sm text-muted-foreground">The model found no values for this request.</p>
      ) : (
        <div className="divide-y rounded-lg border">
          {rows.map((row) => {
            const source = sourceFor(row.path, result.sources)
            const status: SourceStatus | "unsourced" = source?.verification?.status ?? "unsourced"
            const isSelected = !!source && source === selected
            const valueMissing = source?.field === row.path && source.verification?.value_found === false
            return (
              <button
                key={row.path}
                disabled={!source?.verification?.boxes.length}
                onClick={() => source && onSelect(source)}
                className={cn(
                  "grid w-full grid-cols-[auto_minmax(0,2fr)_minmax(0,3fr)_auto] items-center gap-3 px-3 py-2 text-left text-sm transition-colors",
                  "enabled:hover:bg-muted/60 disabled:cursor-default",
                  isSelected && "bg-muted",
                )}
              >
                <SourceDot status={status} />
                <span className="truncate font-mono text-xs text-muted-foreground" title={row.path}>
                  {row.path}
                </span>
                <span className="truncate" title={formatValue(row.value)}>
                  {formatValue(row.value)}
                </span>
                {valueMissing ? (
                  <Tooltip>
                    <TooltipTrigger render={<span />}>
                      <AlertTriangleIcon className="size-4 text-amber-500" />
                    </TooltipTrigger>
                    <TooltipContent>
                      The value does not appear in the quoted page text: check it on the page.
                    </TooltipContent>
                  </Tooltip>
                ) : (
                  <span className="size-4" />
                )}
              </button>
            )
          })}
        </div>
      )}

      {selected?.verification && <SourceDetail source={selected} />}

      <p className="text-xs text-muted-foreground">
        {result.provider} · {result.model} · {result.chunks} call{result.chunks > 1 ? "s" : ""} ·{" "}
        {result.input_tokens.toLocaleString()} in / {result.output_tokens.toLocaleString()} out tokens
      </p>
    </div>
  )
}

function SourceDetail({ source }: { source: SourceRef }) {
  const v = source.verification!
  return (
    <div className="rounded-lg border bg-muted/30 p-3 text-sm">
      <div className="mb-2 flex items-center gap-2">
        <SourceDot status={v.status} />
        <span className="font-medium">{SOURCE_STATUS[v.status].label}</span>
        {v.status === "fuzzy" && <span className="text-muted-foreground">similarity {(v.score * 100).toFixed(0)}%</span>}
        <span className="text-muted-foreground">
          · page {v.page ?? source.page}
          {v.status === "wrong_page" && ` (model cited page ${source.page})`}
        </span>
      </div>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
        <dt className="text-muted-foreground">Model quote</dt>
        <dd className="font-mono text-xs">{source.quote}</dd>
        {v.matched_text && (
          <>
            <dt className="text-muted-foreground">Page text</dt>
            <dd className="font-mono text-xs">{v.matched_text}</dd>
          </>
        )}
      </dl>
    </div>
  )
}

function countStatuses(result: ExtractionResult): Partial<Record<SourceStatus | "unsourced", number>> {
  const counts: Partial<Record<SourceStatus | "unsourced", number>> = {}
  for (const s of result.sources) {
    const k = s.verification?.status
    if (k) counts[k] = (counts[k] ?? 0) + 1
  }
  if (result.unsourced_fields.length) counts.unsourced = result.unsourced_fields.length
  return counts
}
