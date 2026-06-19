'use client'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSanitize from 'rehype-sanitize'

interface MarkdownViewerProps {
  /** The source's extracted markdown (`source.full_text`). */
  content: string | null | undefined
}

/**
 * Markdown tab of the inspect workspace (I.D-4).
 *
 * Renders the source's extracted markdown. react-markdown v10 does not render
 * raw embedded HTML unless `rehype-raw` is added (which we deliberately do
 * NOT), so markup is safe by default; `rehype-sanitize` is layered on as
 * explicit defense-in-depth per the AC, stripping any disallowed tags/attrs
 * (scripts, event handlers, javascript: URLs) should the upstream behavior
 * ever change. GFM tables/strikethrough match the rest of the app's markdown.
 */
export function MarkdownViewer({ content }: MarkdownViewerProps) {
  if (!content) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <p className="text-center text-sm text-muted-foreground">
          No markdown content available for this source.
        </p>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto bg-card p-4">
      <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:text-foreground prose-p:text-muted-foreground prose-p:leading-relaxed prose-li:text-muted-foreground">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeSanitize]}
        >
          {content}
        </ReactMarkdown>
      </div>
    </div>
  )
}
