import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import type { JobStatus, PageContent, SourceStatus } from "@/lib/types"

const JOB_STATUS: Record<JobStatus, { label: string; className: string }> = {
  queued: { label: "Queued", className: "bg-muted text-muted-foreground" },
  classifying: { label: "Classifying", className: "bg-sky-500/15 text-sky-700 dark:text-sky-300" },
  extracting_text: { label: "Reading text", className: "bg-sky-500/15 text-sky-700 dark:text-sky-300" },
  extracting_data: { label: "Extracting data", className: "bg-violet-500/15 text-violet-700 dark:text-violet-300" },
  linking: { label: "Verifying sources", className: "bg-violet-500/15 text-violet-700 dark:text-violet-300" },
  done: { label: "Done", className: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300" },
  failed: { label: "Failed", className: "bg-destructive/15 text-destructive" },
}

export function JobStatusBadge({ status }: { status: JobStatus }) {
  const s = JOB_STATUS[status]
  return <Badge className={cn("border-transparent", s.className)}>{s.label}</Badge>
}

export const SOURCE_STATUS: Record<SourceStatus | "unsourced", { label: string; dot: string; box: string; hint: string }> = {
  verified: {
    label: "Verified",
    dot: "bg-emerald-500",
    box: "border-emerald-500 bg-emerald-400/20",
    hint: "Quote found exactly on the cited page.",
  },
  fuzzy: {
    label: "Fuzzy",
    dot: "bg-amber-500",
    box: "border-amber-500 bg-amber-400/20",
    hint: "Close match on the cited page (OCR noise or small differences).",
  },
  wrong_page: {
    label: "Wrong page",
    dot: "bg-orange-500",
    box: "border-orange-500 bg-orange-400/20",
    hint: "Quote found, but on a different page than the model cited.",
  },
  not_found: {
    label: "Not found",
    dot: "bg-red-500",
    box: "border-red-500 bg-red-400/20",
    hint: "Quote not found in the document: the value may be invented.",
  },
  unknown_field: {
    label: "Unknown field",
    dot: "bg-zinc-400",
    box: "border-zinc-400 bg-zinc-400/20",
    hint: "The source points at a field that is not in the data.",
  },
  unsourced: {
    label: "No source",
    dot: "bg-zinc-300 dark:bg-zinc-600",
    box: "",
    hint: "The model gave no source for this value.",
  },
}

export function SourceDot({ status }: { status: SourceStatus | "unsourced" }) {
  return <span className={cn("inline-block size-2 shrink-0 rounded-full", SOURCE_STATUS[status].dot)} />
}

const PAGE_TYPE: Record<string, string> = {
  digital: "bg-blue-500/15 text-blue-700 dark:text-blue-300",
  scanned: "bg-amber-500/15 text-amber-700 dark:text-amber-300",
  empty: "bg-muted text-muted-foreground",
}

const METHOD_LABEL: Record<string, string> = { text_layer: "Text layer", ocr: "OCR", none: "No text" }

export function PageBadges({ page }: { page: PageContent }) {
  const c = page.classification
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Badge className={cn("border-transparent capitalize", PAGE_TYPE[c.page_type])}>{c.page_type}</Badge>
      <Badge variant="outline">
        {METHOD_LABEL[page.method]}
        {page.ocr_engine ? ` · ${page.ocr_engine}` : ""}
      </Badge>
      {c.quality_score !== null && (
        <Badge variant="outline" className={c.quality_score < c.threshold ? "text-destructive" : ""}>
          quality {c.quality_score.toFixed(2)} / {c.threshold.toFixed(2)}
        </Badge>
      )}
      <span className="text-xs text-muted-foreground">{c.reason.replaceAll("_", " ")}</span>
    </div>
  )
}
