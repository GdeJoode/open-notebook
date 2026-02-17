'use client'

import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogFooter,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogCancel,
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'
import { FilesIcon, PackageIcon } from 'lucide-react'

export type BatchMode = 'individual' | 'single'

interface BatchModeDialogProps {
  open: boolean
  fileCount: number
  onSelect: (mode: BatchMode) => void
  onCancel: () => void
}

export function BatchModeDialog({ open, fileCount, onSelect, onCancel }: BatchModeDialogProps) {
  return (
    <AlertDialog open={open} onOpenChange={(o) => { if (!o) onCancel() }}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Multiple Files Detected</AlertDialogTitle>
          <AlertDialogDescription>
            You selected {fileCount} files. How should they be processed?
          </AlertDialogDescription>
        </AlertDialogHeader>

        <div className="space-y-3 py-2">
          <button
            type="button"
            className="w-full text-left rounded-lg border p-4 hover:bg-accent transition-colors"
            onClick={() => onSelect('individual')}
          >
            <div className="flex items-center gap-3">
              <FilesIcon className="h-5 w-5 text-primary shrink-0" />
              <div>
                <p className="font-medium">Individual Sources</p>
                <p className="text-sm text-muted-foreground">
                  Each file becomes its own source with separate processing
                </p>
              </div>
            </div>
          </button>

          <button
            type="button"
            disabled
            className="w-full text-left rounded-lg border p-4 opacity-50 cursor-not-allowed"
          >
            <div className="flex items-center gap-3">
              <PackageIcon className="h-5 w-5 shrink-0" />
              <div>
                <p className="font-medium">
                  Single Batch{' '}
                  <span className="text-xs text-muted-foreground">(coming soon)</span>
                </p>
                <p className="text-sm text-muted-foreground">
                  Files are combined into one source
                </p>
              </div>
            </div>
          </button>
        </div>

        <AlertDialogFooter>
          <AlertDialogCancel onClick={onCancel}>Cancel</AlertDialogCancel>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
