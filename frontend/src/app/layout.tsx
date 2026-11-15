import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import Link from "next/link"

import { Toaster } from "@/components/ui/sonner"
import { TooltipProvider } from "@/components/ui/tooltip"
import "./globals.css"

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] })
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] })

export const metadata: Metadata = {
  title: "OCR Extract",
  description: "Classify, OCR and extract structured data from PDFs, with verified sources.",
}

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col bg-background text-foreground">
        <TooltipProvider>
          <header className="border-b">
            <div className="mx-auto flex h-14 max-w-7xl items-center px-4">
              <Link href="/" className="font-semibold tracking-tight">
                OCR Extract
              </Link>
            </div>
          </header>
          <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">{children}</main>
          <Toaster richColors position="top-right" />
        </TooltipProvider>
      </body>
    </html>
  )
}
