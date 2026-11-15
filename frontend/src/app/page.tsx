import { RecentJobs } from "@/components/recent-jobs"
import { UploadForm } from "@/components/upload-form"

export default function Home() {
  return (
    <div className="grid items-start gap-6 lg:grid-cols-5">
      <div className="lg:col-span-3">
        <UploadForm />
      </div>
      <div className="lg:col-span-2">
        <RecentJobs />
      </div>
    </div>
  )
}
