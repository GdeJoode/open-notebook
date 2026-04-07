'use client'

import { useAuth } from '@/lib/hooks/use-auth'
import { useRouter, usePathname } from 'next/navigation'
import { useEffect, useRef, useState } from 'react'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { ErrorBoundary } from '@/components/common/ErrorBoundary'
import { ModalProvider } from '@/components/providers/ModalProvider'
import { useModelStatus } from '@/lib/hooks/use-models'

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const { isAuthenticated, isLoading } = useAuth()
  const router = useRouter()
  const pathname = usePathname()
  const [hasCheckedAuth, setHasCheckedAuth] = useState(false)
  const modelStatusChecked = useRef(false)

  // Only run model status check once after auth is confirmed
  const { data: modelStatus, isLoading: statusLoading } = useModelStatus(
    isAuthenticated && hasCheckedAuth && !modelStatusChecked.current
  )

  useEffect(() => {
    // Mark that we've completed the initial auth check
    if (!isLoading) {
      setHasCheckedAuth(true)

      // Redirect to login if not authenticated
      if (!isAuthenticated) {
        // Store the current path to redirect back after login
        const currentPath = window.location.pathname + window.location.search
        sessionStorage.setItem('redirectAfterLogin', currentPath)
        router.push('/login')
      }
    }
  }, [isAuthenticated, isLoading, router])

  // Redirect to models page if configuration is invalid
  useEffect(() => {
    if (modelStatus && !modelStatusChecked.current) {
      modelStatusChecked.current = true
      if (!modelStatus.valid && !pathname.startsWith('/models')) {
        router.push('/models?setup_required=true')
      }
    }
  }, [modelStatus, pathname, router])

  // Show loading spinner during initial auth check or while loading
  if (isLoading || !hasCheckedAuth) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner />
      </div>
    )
  }

  // Don't render anything if not authenticated (during redirect)
  if (!isAuthenticated) {
    return null
  }

  // Show loading while checking model status (only on first load, not on models page)
  if (statusLoading && !modelStatusChecked.current && !pathname.startsWith('/models')) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner />
      </div>
    )
  }

  return (
    <ErrorBoundary>
      {children}
      <ModalProvider />
    </ErrorBoundary>
  )
}