// Mirrors the FastAPI response models in backend/src/ocr_pipeline/{api,models}.

export type BBox = [number, number, number, number] // x0, y0, x1, y1 in PDF points

export type JobStatus =
  | "queued"
  | "classifying"
  | "extracting_text"
  | "extracting_data"
  | "linking"
  | "done"
  | "failed"

export type PageType = "digital" | "scanned" | "empty"
export type ExtractionMethod = "text_layer" | "ocr" | "none"
export type SourceStatus = "verified" | "fuzzy" | "wrong_page" | "not_found" | "unknown_field"
export type Provider = "anthropic" | "openai" | "ollama"

export interface PageClassification {
  page_number: number
  page_type: PageType
  needs_ocr: boolean
  reason: string
  quality_score: number | null
  threshold: number
  score_components: Record<string, number>
}

export interface ContentBlock {
  kind: string
  bbox: BBox
  start: number
  end: number
  text: string
}

export interface PageContent {
  page_number: number
  width: number
  height: number
  method: ExtractionMethod
  ocr_engine: string | null
  markdown: string
  blocks: ContentBlock[]
  classification: PageClassification
}

export interface ExtractedDocument {
  source: string
  page_count: number
  pages: PageContent[]
}

export interface SourceVerification {
  status: SourceStatus
  score: number
  page: number | null
  matched_text: string | null
  boxes: BBox[]
  value: unknown
  value_found: boolean | null
}

export interface SourceRef {
  field: string
  page: number
  quote: string
  verification: SourceVerification | null
}

export interface FieldConflict {
  field: string
  values: unknown[]
  pages: number[][]
}

export interface ExtractionResult {
  data: Record<string, unknown>
  sources: SourceRef[]
  unsourced_fields: string[]
  conflicts: FieldConflict[]
  warnings: string[]
  provider: string
  model: string
  chunks: number
  input_tokens: number
  output_tokens: number
}

export interface JobOptions {
  schema_name?: string | null
  json_schema?: Record<string, unknown> | null
  instructions?: string | null
  extract_data?: boolean
  provider?: Provider | null
  model?: string | null
  effort?: string | null
  quality_threshold?: number | null
  table_format?: "markdown" | "html" | null
}

export interface JobSummary {
  id: string
  filename: string
  status: JobStatus
  error: string | null
  created_at: string
  updated_at: string
  page_count: number | null
}

export interface JobDetail extends JobSummary {
  options: JobOptions
  document: ExtractedDocument | null
  result: ExtractionResult | null
}

export interface ProviderInfo {
  name: Provider
  default_model: string
  configured: boolean
}

export interface SettingsView {
  providers: ProviderInfo[]
  default_provider: Provider
  efforts: string[]
  default_effort: string
  quality_threshold: number
  table_formats: ("markdown" | "html")[]
  default_table_format: "markdown" | "html"
  schemas: string[]
  max_upload_mb: number
  page_image_max_dpi: number
}

export const FINISHED: JobStatus[] = ["done", "failed"]
export const TEXT_READY: JobStatus[] = ["extracting_data", "linking", "done"]
