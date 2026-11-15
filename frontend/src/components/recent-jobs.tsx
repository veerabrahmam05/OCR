"use client"

import { Trash2Icon } from "lucide-react"
import Link from "next/link"
import { toast } from "sonner"

import { JobStatusBadge } from "@/components/status-badges"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { usePolling } from "@/hooks/use-polling"
import { api } from "@/lib/api"
import { FINISHED } from "@/lib/types"

export function RecentJobs() {
  const { data: jobs, error, refresh } = usePolling(api.jobs, 3000)

  async function remove(id: string) {
    try {
      await api.deleteJob(id)
      refresh()
    } catch (e) {
      toast.error((e as Error).message)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent jobs</CardTitle>
        <CardDescription>Kept in server memory only: restarting the backend clears this list.</CardDescription>
      </CardHeader>
      <CardContent>
        {error && <p className="text-sm text-destructive">{error.message}</p>}
        {!error && jobs?.length === 0 && <p className="text-sm text-muted-foreground">No jobs yet.</p>}
        {jobs && jobs.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>File</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Pages</TableHead>
                <TableHead className="text-right">Started</TableHead>
                <TableHead className="w-10" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.map((job) => (
                <TableRow key={job.id}>
                  <TableCell className="max-w-56 truncate font-medium">
                    <Link href={`/jobs/${job.id}`} className="hover:underline">
                      {job.filename}
                    </Link>
                  </TableCell>
                  <TableCell>
                    <JobStatusBadge status={job.status} />
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{job.page_count ?? "–"}</TableCell>
                  <TableCell className="text-right text-muted-foreground tabular-nums">
                    {new Date(job.created_at).toLocaleTimeString()}
                  </TableCell>
                  <TableCell>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      aria-label="Delete job"
                      disabled={!FINISHED.includes(job.status)}
                      onClick={() => remove(job.id)}
                    >
                      <Trash2Icon />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}
