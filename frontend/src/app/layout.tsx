import type { Metadata } from "next";
import { Inter, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import { QueryProvider } from "@/components/providers/QueryProvider";
import { ThemeProvider } from "@/components/providers/ThemeProvider";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";
import { ConnectionGuard } from "@/components/common/ConnectionGuard";
import { themeScript } from "@/lib/theme-script";

// Track I.A — Docling Studio visual identity adoption.
//
// Inter is the UI font for the entire app; IBM Plex Mono is reserved for
// numeric metadata (page pills, token counts, bbox coordinates — applied in
// phase I.E via the `.mono-num` utility defined in globals.css).
//
// Both fonts are loaded via next/font/google with `display: 'swap'` so the
// browser uses a fallback while the WOFF2 streams in (zero CLS, no FOIT). The
// `latin` subset keeps the payload small; the plan bounds total font delta at
// < 30KB gzipped.
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  // IBM Plex Mono ships several weights; 400 + 500 cover regular numeric
  // metadata and emphasised values (e.g. an active page pill). Sticking to
  // two weights keeps the bundle small.
  weight: ["400", "500"],
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
    <html
      lang="en"
      suppressHydrationWarning
      className={`${inter.variable} ${plexMono.variable}`}
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
