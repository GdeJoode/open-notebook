import type { Metadata } from "next";
import { Inter, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import { QueryProvider } from "@/components/providers/QueryProvider";
import { ThemeProvider } from "@/components/providers/ThemeProvider";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";
import { ConnectionGuard } from "@/components/common/ConnectionGuard";
import { themeScript } from "@/lib/theme-script";

// Track I.A / I.E — Docling Studio visual identity adoption.
//
// Inter is the UI font for the entire app. IBM Plex Mono is the numeric font:
// it backs the `.mono-num` utility used on the inspector's numeric readouts
// (page indicator, element-type counts, bbox coordinates). I.A deferred this
// load until its first consumer landed; I.E wires both together so the WOFF2
// is only shipped now that something reads `--font-mono-numeric`.
//
// Both are loaded via next/font/google with `display: 'swap'` so the browser
// uses a fallback while the WOFF2 streams in (zero CLS, no FOIT). The `latin`
// subset keeps the payload small; IBM Plex Mono ships only the 400/500 weights
// the numeric readouts use.
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

// Weight 400 only: `.mono-num` renders tabular numerics at the normal weight;
// no consumer requests medium (500), so shipping it would be dead font payload
// against the track's <30KB bundle budget (plan §I.A AC5).
const ibmPlexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400"],
  variable: "--font-mono-numeric",
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
    // Expose both font CSS variables on <html> so descendants -- including
    // portaled Radix content rendered outside <body> -- can resolve them.
    // `--font-mono-numeric` binds IBM Plex Mono, which `.mono-num` (and the
    // Tailwind `font-mono` utility) resolve to in globals.css.
    <html
      lang="en"
      suppressHydrationWarning
      className={`${inter.variable} ${ibmPlexMono.variable}`}
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
