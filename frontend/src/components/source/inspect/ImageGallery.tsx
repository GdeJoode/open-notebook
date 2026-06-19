'use client'

import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { sourcesApi } from '@/lib/api/sources'
import { getApiUrl } from '@/lib/config'
import {
  IMAGE_PAGE_SIZE,
  clampPage,
  pageCount,
  pageSlice,
} from '@/lib/utils/image-pagination'

interface ImageGalleryProps {
  sourceId: string
}

/**
 * Images tab of the inspect workspace (I.D-4).
 *
 * Renders the source's extracted images in a paginated grid. Two mechanisms
 * bound concurrent image requests to satisfy the AC ("≤6 in flight"):
 *   1. Only one page of IMAGE_PAGE_SIZE (=6) images is mounted at a time, so at
 *      most 6 <img> requests can start together.
 *   2. Each <img> uses `loading="lazy"` so off-screen images defer further.
 *
 * When a source has no extracted images (the common case — image filenames are
 * not persisted on chunks, only enumerable from the output directory) the tab
 * shows a clean empty state rather than erroring.
 */
export function ImageGallery({ sourceId }: ImageGalleryProps) {
  const [page, setPage] = useState(0)
  const [apiUrl, setApiUrl] = useState<string | null>(null)

  // Image src URLs are absolute (served by the backend, not Next), so resolve
  // the API base once rather than awaiting getImageUrl() per image.
  useEffect(() => {
    let cancelled = false
    getApiUrl().then((url) => {
      if (!cancelled) setApiUrl(url)
    })
    return () => {
      cancelled = true
    }
  }, [])

  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ['source-images', sourceId],
    queryFn: () => sourcesApi.getImages(sourceId),
  })

  const images = useMemo(() => data?.images ?? [], [data])
  const totalPages = pageCount(images.length)
  const safePage = clampPage(page, images.length)
  const visible = useMemo(
    () => pageSlice(images, safePage),
    [images, safePage]
  )

  // Reset to the first page whenever the underlying list changes.
  useEffect(() => {
    setPage(0)
  }, [images.length])

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <LoadingSpinner />
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <Alert variant="destructive" className="max-w-sm">
          <AlertTitle>Failed to load images</AlertTitle>
          <AlertDescription>
            Could not retrieve extracted images for this source.{' '}
            <button
              type="button"
              onClick={() => refetch()}
              className="underline underline-offset-2"
            >
              Retry
            </button>
          </AlertDescription>
        </Alert>
      </div>
    )
  }

  if (images.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <p className="text-center text-sm text-muted-foreground">
          No extracted images available for this source.
        </p>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col bg-card">
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        <div className="grid grid-cols-2 gap-3">
          {visible.map((filename) => (
            <figure
              key={filename}
              className="overflow-hidden rounded-md border bg-muted/30"
            >
              {apiUrl ? (
                <img
                  src={`${apiUrl}/api/sources/${sourceId}/images/${filename}`}
                  alt={filename}
                  loading="lazy"
                  className="h-32 w-full object-contain"
                />
              ) : (
                <div className="flex h-32 w-full items-center justify-center">
                  <LoadingSpinner />
                </div>
              )}
              <figcaption className="truncate px-2 py-1 text-[10px] text-muted-foreground mono-num">
                {filename}
              </figcaption>
            </figure>
          ))}
        </div>
      </div>

      {totalPages > 1 && (
        <div className="flex flex-shrink-0 items-center justify-between border-t px-3 py-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={safePage === 0}
            aria-label="Previous image page"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-xs text-muted-foreground mono-num">
            {safePage + 1} / {totalPages}
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={safePage >= totalPages - 1}
            aria-label="Next image page"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      )}

      <p className="sr-only">
        Showing up to {IMAGE_PAGE_SIZE} images per page.
      </p>
    </div>
  )
}
