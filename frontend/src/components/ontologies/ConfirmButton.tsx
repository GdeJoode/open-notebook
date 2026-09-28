'use client'

/**
 * OE.3: a button that asks before it acts — for the editor's actions a click
 * too many would regret (discarding a draft, rolling back the ontology every
 * notebook types with, deleting a type).
 */

import type { ComponentProps, ReactNode } from 'react'

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'

interface ConfirmButtonProps extends Omit<ComponentProps<typeof Button>, 'onClick'> {
  title: string
  description: ReactNode
  confirmLabel: string
  onConfirm: () => void
}

export function ConfirmButton({
  title, description, confirmLabel, onConfirm, children, ...button
}: ConfirmButtonProps) {
  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button {...button}>{children}</Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          <AlertDialogDescription>{description}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction onClick={onConfirm} data-testid="confirm-action">
            {confirmLabel}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
