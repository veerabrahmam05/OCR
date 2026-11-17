"use client"

import { FileTextIcon, Loader2Icon, UploadIcon, XIcon } from "lucide-react"
import { useRouter } from "next/navigation"
import { useRef, useState } from "react"
import { toast } from "sonner"

import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { usePolling } from "@/hooks/use-polling"
import { api } from "@/lib/api"
import type { DateOrder, JobOptions, Provider, SettingsView } from "@/lib/types"
import { cn } from "@/lib/utils"

type Mode = "builtin" | "custom" | "instructions"

const PROVIDER_LABEL: Record<Provider, string> = { anthropic: "Claude", openai: "OpenAI", ollama: "Ollama (local)" }

const DATE_ORDER_LABEL: Record<DateOrder, string> = {
  auto: "Auto (no hint)",
  DMY: "DD/MM/YYYY",
  MDY: "MM/DD/YYYY",
  YMD: "YYYY/MM/DD",
}

export function UploadForm() {
  const { data: settings, error } = usePolling(api.settings, 0, () => false)
  if (error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>{error.message}</AlertDescription>
      </Alert>
    )
  }
  if (!settings) return <Skeleton className="h-[560px] w-full rounded-xl" />
  return <Form settings={settings} />
}

function Form({ settings }: { settings: SettingsView }) {
  const router = useRouter()
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  const [extractData, setExtractData] = useState(true)
  const [mode, setMode] = useState<Mode>("builtin")
  const [schemaName, setSchemaName] = useState(settings.schemas[0] ?? "")
  const [customSchema, setCustomSchema] = useState("")
  const [instructions, setInstructions] = useState("")

  const [provider, setProvider] = useState<Provider>(settings.default_provider)
  const [model, setModel] = useState("")
  const [effort, setEffort] = useState(settings.default_effort)
  const [threshold, setThreshold] = useState(settings.quality_threshold)
  const [tableFormat, setTableFormat] = useState(settings.default_table_format)
  const [dateOrder, setDateOrder] = useState<DateOrder>(settings.default_date_order)

  const providerInfo = settings.providers.find((p) => p.name === provider)
  const maxBytes = settings.max_upload_mb * 1024 * 1024

  function pick(f: File | undefined) {
    if (!f) return
    if (f.type !== "application/pdf" && !f.name.toLowerCase().endsWith(".pdf")) {
      toast.error("Please choose a PDF file.")
      return
    }
    if (f.size > maxBytes) {
      toast.error(`File is larger than ${settings.max_upload_mb} MB.`)
      return
    }
    setFile(f)
  }

  function buildOptions(): JobOptions | string {
    const opts: JobOptions = {
      extract_data: extractData,
      quality_threshold: threshold,
      table_format: tableFormat,
    }
    if (!extractData) return opts
    if (mode === "builtin") opts.schema_name = schemaName
    if (mode === "custom") {
      try {
        const parsed = JSON.parse(customSchema)
        if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return "The schema must be a JSON object."
        opts.json_schema = parsed
      } catch {
        return "The custom schema is not valid JSON."
      }
    }
    if (instructions.trim()) opts.instructions = instructions.trim()
    if (mode === "instructions" && !opts.instructions) return "Write instructions describing what to extract."
    opts.date_order = dateOrder
    opts.provider = provider
    if (model.trim()) opts.model = model.trim()
    if (provider === "anthropic") opts.effort = effort
    return opts
  }

  async function submit() {
    if (!file) return
    const opts = buildOptions()
    if (typeof opts === "string") {
      toast.error(opts)
      return
    }
    setSubmitting(true)
    try {
      const job = await api.createJob(file, opts)
      router.push(`/jobs/${job.id}`)
    } catch (e) {
      toast.error((e as Error).message)
      setSubmitting(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>New extraction</CardTitle>
        <CardDescription>
          Upload a PDF. Digital pages are read directly; scanned pages go through OCR. Then an LLM extracts the
          data you ask for, with a verified source for every value.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-6">
        {/* File */}
        <div
          role="button"
          tabIndex={0}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            pick(e.dataTransfer.files[0])
          }}
          className={cn(
            "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-8 text-center transition-colors outline-none focus-visible:ring-3 focus-visible:ring-ring/50",
            dragging ? "border-primary bg-primary/5" : "border-border hover:bg-muted/50",
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf,.pdf"
            className="hidden"
            onChange={(e) => pick(e.target.files?.[0])}
          />
          {file ? (
            <div className="flex items-center gap-2 text-sm">
              <FileTextIcon className="size-5 text-muted-foreground" />
              <span className="font-medium">{file.name}</span>
              <span className="text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label="Remove file"
                onClick={(e) => {
                  e.stopPropagation()
                  setFile(null)
                  if (inputRef.current) inputRef.current.value = ""
                }}
              >
                <XIcon />
              </Button>
            </div>
          ) : (
            <>
              <UploadIcon className="size-6 text-muted-foreground" />
              <p className="text-sm">
                <span className="font-medium">Drop a PDF here</span> or click to browse
              </p>
              <p className="text-xs text-muted-foreground">Up to {settings.max_upload_mb} MB</p>
            </>
          )}
        </div>

        {/* What to extract */}
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between gap-4">
            <Label htmlFor="extract-data" className="text-base">
              Extract data with an LLM
            </Label>
            <Switch id="extract-data" checked={extractData} onCheckedChange={setExtractData} />
          </div>
          {!extractData && (
            <p className="text-sm text-muted-foreground">
              Text only: classify pages, run OCR where needed and return Markdown. No LLM call, no API key needed.
            </p>
          )}
          {extractData && (
            <Tabs value={mode} onValueChange={(v) => setMode(v as Mode)}>
              <TabsList className="w-full">
                <TabsTrigger value="builtin">Built-in schema</TabsTrigger>
                <TabsTrigger value="custom">Custom schema</TabsTrigger>
                <TabsTrigger value="instructions">Instructions only</TabsTrigger>
              </TabsList>
              <TabsContent value="builtin" className="pt-2">
                <Select
                  value={schemaName}
                  onValueChange={(v) => v && setSchemaName(v)}
                  items={Object.fromEntries(settings.schemas.map((s) => [s, s[0].toUpperCase() + s.slice(1)]))}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {settings.schemas.map((s) => (
                      <SelectItem key={s} value={s} className="capitalize">
                        {s}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </TabsContent>
              <TabsContent value="custom" className="pt-2">
                <Textarea
                  value={customSchema}
                  onChange={(e) => setCustomSchema(e.target.value)}
                  placeholder={'{\n  "type": "object",\n  "properties": {\n    "po_number": {"type": "string"},\n    "total": {"type": "number"}\n  }\n}'}
                  className="min-h-40 font-mono text-xs"
                  spellCheck={false}
                />
              </TabsContent>
              <TabsContent value="instructions" className="pt-2">
                <p className="text-sm text-muted-foreground">
                  The model returns a list of named values based on the instructions below.
                </p>
              </TabsContent>
            </Tabs>
          )}
          {extractData && (
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="instructions">
                Instructions {mode !== "instructions" && <span className="text-muted-foreground">(optional)</span>}
              </Label>
              <Textarea
                id="instructions"
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
                placeholder="e.g. Extract the PO number, the delivery address and every line item. Amounts are in EUR."
                className="min-h-20"
              />
            </div>
          )}
        </div>

        <Separator />

        {/* Model */}
        {extractData && (
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="flex flex-col gap-1.5">
              <Label>Provider</Label>
              <Select
                value={provider}
                onValueChange={(v) => v && setProvider(v as Provider)}
                items={PROVIDER_LABEL}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {settings.providers.map((p) => (
                    <SelectItem key={p.name} value={p.name}>
                      {PROVIDER_LABEL[p.name]}
                      {!p.configured && <span className="text-muted-foreground"> (needs model)</span>}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="model">Model</Label>
              <Input
                id="model"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder={providerInfo?.default_model || "e.g. qwen2.5:7b"}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Effort</Label>
              <Select
                value={effort}
                onValueChange={(v) => v && setEffort(v)}
                disabled={provider !== "anthropic"}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {settings.efforts.map((e) => (
                    <SelectItem key={e} value={e}>
                      {e}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-1.5 sm:col-span-3">
              <Label>Dates in document</Label>
              <Select
                value={dateOrder}
                onValueChange={(v) => v && setDateOrder(v as DateOrder)}
                items={DATE_ORDER_LABEL}
              >
                <SelectTrigger className="w-full sm:w-1/3">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {settings.date_orders.map((d) => (
                    <SelectItem key={d} value={d}>
                      {DATE_ORDER_LABEL[d]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                How numeric dates are written, e.g. whether 12/11/2026 is 12 November or 11 December. Tells the
                model and makes the date check strict.
              </p>
            </div>
            {provider === "ollama" && !providerInfo?.configured && !model.trim() && (
              <p className="text-xs text-destructive sm:col-span-3">
                No Ollama model is configured on the server: enter one (e.g. qwen2.5:7b).
              </p>
            )}
          </div>
        )}

        {/* Text extraction */}
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="flex flex-col gap-2.5 sm:col-span-2">
            <div className="flex items-center justify-between">
              <Label>OCR quality threshold</Label>
              <span className="text-sm tabular-nums text-muted-foreground">{threshold.toFixed(2)}</span>
            </div>
            <Slider
              value={[threshold]}
              onValueChange={(v) => setThreshold(Array.isArray(v) ? v[0] : v)}
              min={0}
              max={1}
              step={0.05}
            />
            <p className="text-xs text-muted-foreground">
              Pages whose text layer scores below this are re-read with OCR. Higher = OCR more often.
            </p>
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>Table format</Label>
            <Select value={tableFormat} onValueChange={(v) => v && setTableFormat(v)}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {settings.table_formats.map((t) => (
                  <SelectItem key={t} value={t}>
                    {t}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <Button size="lg" onClick={submit} disabled={!file || submitting} className="self-end">
          {submitting && <Loader2Icon className="animate-spin" />}
          {submitting ? "Uploading…" : "Start extraction"}
        </Button>
      </CardContent>
    </Card>
  )
}
