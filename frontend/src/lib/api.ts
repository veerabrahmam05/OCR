import type { JobDetail, JobOptions, JobSummary, SettingsView } from "@/lib/types"

// Inlined at build time (NEXT_PUBLIC_*). Set it in frontend/.env.local.
export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "")

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message)
  }
}

type ErrorDetail = string | { loc?: (string | number)[]; msg: string }[] | undefined

function formatDetail(detail: ErrorDetail): string | undefined {
  if (typeof detail === "string") return detail
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        const field = (d.loc ?? []).filter((p) => typeof p === "string").join(".")
        return field ? `${field}: ${d.msg}` : d.msg
      })
      .join("; ")
  }
  return undefined
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init })
  } catch {
    throw new ApiError(0, `Cannot reach the API at ${API_URL}. Is the backend running?`)
  }
  if (!response.ok) {
    let detail: ErrorDetail
    try {
      detail = (await response.json()).detail
    } catch {
      // non-JSON error body
    }
    throw new ApiError(response.status, formatDetail(detail) ?? response.statusText)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const api = {
  settings: () => request<SettingsView>("/settings"),
  jobs: () => request<JobSummary[]>("/jobs"),
  job: (id: string) => request<JobDetail>(`/jobs/${id}`),
  deleteJob: (id: string) => request<void>(`/jobs/${id}`, { method: "DELETE" }),
  createJob: (file: File, options: JobOptions) => {
    const body = new FormData()
    body.append("file", file)
    body.append("options", JSON.stringify(options))
    return request<JobSummary>("/jobs", { method: "POST", body })
  },
  pageImageUrl: (id: string, page: number, dpi: number) =>
    `${API_URL}/jobs/${id}/pages/${page}/image?dpi=${dpi}`,
}
