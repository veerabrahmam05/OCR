import { Suspense } from "react"

import { JobView } from "@/components/job-view"
import { Skeleton } from "@/components/ui/skeleton"

// Cache Components: the job id is URL data with no generateStaticParams, so reading it
// must happen inside a Suspense boundary (JobView unwraps `params` with use()).
export default function JobPage({ params }: PageProps<"/jobs/[id]">) {
  return (
    <Suspense fallback={<Skeleton className="h-96 w-full rounded-xl" />}>
      <JobView params={params} />
    </Suspense>
  )
}
