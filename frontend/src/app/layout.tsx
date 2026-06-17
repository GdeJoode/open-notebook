import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import { QueryProvider } from "@/components/providers/QueryProvider";
import { ThemeProvider } from "@/components/providers/ThemeProvider";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";
import { ConnectionGuard } from "@/components/common/ConnectionGuard";
import { themeScript } from "@/lib/theme-script";

// Track I.A — Docling Studio visual identity adoption.
//
// Inter is the UI font for the entire app. IBM Plex Mono is RESERVED for
// numeric metadata (page pills, token counts, bbox coordinates) but is NOT
// loaded in I.A: nothing consumes `.mono-num` until I.E ships the inspector
// polish that applies it to `token-count`, `page-pill`, `bbox-coords`. Loading
// a ~65KB WOFF2 with zero consumers is bundle-waste. The `.mono-num` utility
// in globals.css falls back to the system monospace stack until I.E lands the
// IBM Plex Mono load + consumer wiring together.
//
// Inter is loaded via next/font/google with `display: 'swap'` so the browser
// uses a fallback while the WOFF2 streams in (zero CLS, no FOIT). The `latin`
// subset keeps the payload small.
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Noesis",
  description: "Privacy-focused research and knowledge management",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // Expose Inter's CSS variable on <html> so descendants -- including
    // portaled Radix content rendered outside <body> -- can resolve it.
    // `--font-mono-numeric` is intentionally unset here; the `.mono-num`
    // utility class falls back to ui-monospace until I.E loads IBM Plex Mono.
    <html
      lang="en"
      suppressHydrationWarning
      className={inter.variable}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      {/* Inter is applied to <body> via the next/font className so the
          actual font-face is resolved. The Tailwind `font-sans` utility also
          works because --font-sans is forwarded to --font-inter in
          globals.css. */}
      <body className={inter.className}>
        <ErrorBoundary>
          <ThemeProvider>
            <QueryProvider>
              <ConnectionGuard>
                {children}
                <Toaster />
              </ConnectionGuard>
            </QueryProvider>
          </ThemeProvider>
        </ErrorBoundary>
      </body>
    </html>
  );
}
