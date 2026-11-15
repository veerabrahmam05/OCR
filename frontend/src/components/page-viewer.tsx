"use client"

import { ChevronLeftIcon, ChevronRightIcon } from "lucide-react"
import { useState } from "react"

import { PageBadges, SOURCE_STATUS } from "@/components/status-badges"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { api } from "@/lib/api"
import type { BBox, PageContent, SourceRef } from "@/lib/types"
import { cn } from "@/lib/utils"

const DPI = 150

interface Props {
  jobId: string
  pages: PageContent[]
  page: number
  onPageChange: (page: number) => void
  imagesReady: boolean
  sources: SourceRef[]
  selected: SourceRef | null
  onSelect: (source: SourceRef) => void
}

export function PageViewer({ jobId, pages, page, onPageChange, imagesReady, sources, selected, onSelect }: Props) {
  const current = pages.find((p) => p.page_number === page) ?? pages[0]
  if (!current) return null

  // Every located source on this page; the selected one is drawn on top.
  const onPage = sources.filter((s) => s.verification?.page === current.page_number && s.verification.boxes.length)
  const ordered = selected ? [...onPage.filter((s) => s !== selected), ...onPage.filter((s) => s === selected)] : onPage

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <PageBadges page={current} />
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="icon-sm"
            aria-label="Previous page"
            disabled={page <= 1}
            onClick={() => onPageChange(page - 1)}
          >
            <ChevronLeftIcon />
          </Button>
          <span className="min-w-20 text-center text-sm tabular-nums">
            Page {current.page_number} / {pages.length}
          </span>
          <Button
            variant="outline"
            size="icon-sm"
            aria-label="Next page"
            disabled={page >= pages.length}
            onClick={() => onPageChange(page + 1)}
          >
            <ChevronRightIcon />
          </Button>
        </div>
      </div>

      {pages.length > 1 && pages.length <= 40 && (
        <div className="flex flex-wrap gap-1">
          {pages.map((p) => (
            <button
              key={p.page_number}
              onClick={() => onPageChange(p.page_number)}
              title={`Page ${p.page_number}: ${p.classification.page_type}, ${p.method}`}
              className={cn(
                "h-6 min-w-6 rounded-md border px-1 text-xs tabular-nums transition-colors",
                p.method === "ocr" ? "border-amber-500/40" : "border-border",
                p.page_number === current.page_number ? "bg-primary text-primary-foreground" : "hover:bg-muted",
              )}
            >
              {p.page_number}
            </button>
          ))}
        </div>
      )}

      <div
        className="relative w-full overflow-hidden rounded-lg border bg-white shadow-sm"
        style={{ aspectRatio: `${current.width} / ${current.height}` }}
      >
        {imagesReady ? (
          <PageImage
            key={current.page_number}
            src={api.pageImageUrl(jobId, current.page_number, DPI)}
            alt={`Page ${current.page_number}`}
          />
        ) : (
          <Skeleton className="absolute inset-0 rounded-none" />
        )}
        {ordered.map((s, i) =>
          s.verification!.boxes.map((box, j) => (
            <Highlight
              key={`${i}-${j}`}
              box={box}
              page={current}
              className={SOURCE_STATUS[s.verification!.status].box}
              active={s === selected}
              title={`${s.field}: ${s.quote}`}
              onClick={() => onSelect(s)}
            />
          )),
        )}
      </div>
    </div>
  )
}

function PageImage({ src, alt }: { src: string; alt: string }) {
  const [loaded, setLoaded] = useState(false)
  return (
    <>
      {!loaded && <Skeleton className="absolute inset-0 rounded-none" />}
      {/* Plain <img>: the API serves page renders whose pixels map 1:1 to PDF points × dpi/72. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={alt} className="absolute inset-0 size-full" onLoad={() => setLoaded(true)} draggable={false} />
    </>
  )
}

function Highlight({
  box,
  page,
  className,
  active,
  title,
  onClick,
}: {
  box: BBox
  page: PageContent
  className: string
  active: boolean
  title: string
  onClick: () => void
}) {
  const [x0, y0, x1, y1] = box
  const pad = 1.5 // points
  return (
    <button
      title={title}
      onClick={onClick}
      className={cn(
        "absolute rounded-sm border transition-all",
        className,
        active ? "z-10 border-2 opacity-100 ring-4 ring-primary/25" : "opacity-50 hover:opacity-90",
      )}
      style={{
        left: `${((x0 - pad) / page.width) * 100}%`,
        top: `${((y0 - pad) / page.height) * 100}%`,
        width: `${((x1 - x0 + 2 * pad) / page.width) * 100}%`,
        height: `${((y1 - y0 + 2 * pad) / page.height) * 100}%`,
      }}
    />
  )
}
