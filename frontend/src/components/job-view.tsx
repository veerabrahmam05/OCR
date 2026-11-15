"use client"

import { DownloadIcon } from "lucide-react"
import Link from "next/link"
import { use, useState } from "react"

import { DataPanel } from "@/components/data-panel"
import { PageViewer } from "@/components/page-viewer"
import { JobStatusBadge } from "@/components/status-badges"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button, buttonVariants } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { usePolling } from "@/hooks/use-polling"
import { ApiError, api } from "@/lib/api"
import { FINISHED, TEXT_READY, type JobDetail, type JobStatus, type SourceRef } from "@/lib/types"

const STAGES: JobStatus[] = ["queued", "classifying", "extracting_text", "extracting_data", "linking", "done"]

export function JobView({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params)
  const { data: job, error } = usePolling(
    () => api.job(id),
    1000,
    (j) => !j || !FINISHED.includes(j.status),
    [id],
  )
  const [page, setPage] = useState(1)
  const [selected, setSelected] = useState<SourceRef | null>(null)

  if (error && !job) {
    const gone = error instanceof ApiError && error.status === 404
    return (
      <Alert variant="destructive">
        <AlertTitle>{gone ? "Job not found" : "Could not load the job"}</AlertTitle>
        <AlertDescription>
          {gone ? "Jobs are kept in server memory only; the backend may have restarted." : error.message}{" "}
          <Link href="/" className="underline">
            Start a new extraction
          </Link>
        </AlertDescription>
      </Alert>
    )
  }
  if (!job) return <Skeleton className="h-96 w-full rounded-xl" />

  const select = (s: SourceRef) => {
    setSelected(s)
    setPage(s.verification?.page ?? s.page)
  }
  const pages = job.document?.pages ?? []
  const current = pages.find((p) => p.page_number === page)

  return (
    <div className="flex flex-col gap-4">
      <Header job={job} />

      {job.status === "failed" && (
        <Alert variant="destructive">
          <AlertTitle>Extraction failed</AlertTitle>
          <AlertDescription className="break-words">{job.error}</AlertDescription>
        </Alert>
      )}

      {!job.document ? (
        <Card>
          <CardContent className="flex flex-col gap-3 py-8">
            <p className="text-sm text-muted-foreground">Classifying pages and reading text…</p>
            <Progress value={(STAGES.indexOf(job.status) / (STAGES.length - 1)) * 100} />
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardContent>
              <PageViewer
                jobId={job.id}
                pages={pages}
                page={page}
                onPageChange={setPage}
                imagesReady={TEXT_READY.includes(job.status) || (job.status === "failed" && !!job.document)}
                sources={job.result?.sources ?? []}
                selected={selected}
                onSelect={select}
              />
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <Tabs defaultValue="data">
                <TabsList>
                  <TabsTrigger value="data">Data</TabsTrigger>
                  <TabsTrigger value="markdown">Page text</TabsTrigger>
                  <TabsTrigger value="json">JSON</TabsTrigger>
                </TabsList>

                <TabsContent value="data" className="pt-3">
                  {job.result ? (
                    <DataPanel result={job.result} selected={selected} onSelect={select} />
                  ) : !job.options.extract_data ? (
                    <p className="text-sm text-muted-foreground">
                      Text-only job: no data was extracted. See the page text tab.
                    </p>
                  ) : job.status === "failed" ? (
                    <p className="text-sm text-muted-foreground">No data: the LLM step failed.</p>
                  ) : (
                    <div className="flex flex-col gap-3">
                      <p className="text-sm text-muted-foreground">
                        {job.status === "linking" ? "Verifying sources…" : "Extracting data with the LLM…"}
                      </p>
                      <Progress value={(STAGES.indexOf(job.status) / (STAGES.length - 1)) * 100} />
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="markdown" className="pt-3">
                  <ScrollArea className="h-[70vh] rounded-lg border bg-muted/30">
                    <pre className="p-3 font-mono text-xs whitespace-pre-wrap">
                      {current?.markdown.trim() || "(no text on this page)"}
                    </pre>
                  </ScrollArea>
                </TabsContent>

                <TabsContent value="json" className="pt-3">
                  {job.result ? (
                    <div className="flex flex-col gap-2">
                      <Button variant="outline" size="sm" className="self-end" onClick={() => download(job)}>
                        <DownloadIcon /> Download JSON
                      </Button>
                      <ScrollArea className="h-[65vh] rounded-lg border bg-muted/30">
                        <pre className="p-3 font-mono text-xs">{JSON.stringify(job.result.data, null, 2)}</pre>
                      </ScrollArea>
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">No extracted data.</p>
                  )}
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}

function Header({ job }: { job: JobDetail }) {
  const running = !FINISHED.includes(job.status)
  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex min-w-0 items-center gap-3">
        <h1 className="truncate text-xl font-semibold">{job.filename}</h1>
        <JobStatusBadge status={job.status} />
        {job.page_count !== null && (
          <span className="text-sm text-muted-foreground">
            {job.page_count} page{job.page_count === 1 ? "" : "s"}
          </span>
        )}
      </div>
      <div className="flex items-center gap-3">
        {running && <Progress className="w-40" value={(STAGES.indexOf(job.status) / (STAGES.length - 1)) * 100} />}
        <Link href="/" className={buttonVariants({ variant: "outline", size: "sm" })}>
          New extraction
        </Link>
      </div>
    </div>
  )
}

function download(job: JobDetail) {
  const blob = new Blob([JSON.stringify(job.result, null, 2)], { type: "application/json" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = `${job.filename.replace(/\.pdf$/i, "")}.extraction.json`
  a.click()
  URL.revokeObjectURL(url)
}
