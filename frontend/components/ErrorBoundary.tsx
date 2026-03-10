import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertCircle } from 'lucide-react'
import { Button } from './ui/button'

interface Props {
  children: ReactNode
  /** Optional label for the fallback (e.g. "Something went wrong") */
  fallbackTitle?: string
}

interface State {
  error: Error | null
  errorInfo: ErrorInfo | null
}

/**
 * Catches React render errors and shows the message (and stack) on screen
 * so we're not blind when the app crashes (e.g. without DevTools).
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, errorInfo: null }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.setState({ errorInfo })
    if (typeof window !== 'undefined' && window.electronAPI?.sendRendererLog) {
      window.electronAPI.sendRendererLog('error', '[ErrorBoundary]', error?.message, error?.stack, errorInfo.componentStack)
    }
  }

  render(): ReactNode {
    const { error, errorInfo } = this.state
    if (!error) return this.props.children

    const title = this.props.fallbackTitle ?? 'Something went wrong'
    const message = error.message ?? String(error)
    const stack = error.stack ?? ''
    const componentStack = errorInfo?.componentStack ?? ''

    return (
      <div className="fixed inset-0 z-[100] flex items-center justify-center bg-zinc-950 p-4">
        <div className="w-full max-w-2xl max-h-[90vh] overflow-auto rounded-xl border border-red-900/50 bg-zinc-900 p-5 text-zinc-100 shadow-xl">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-6 w-6 shrink-0 text-red-400" />
            <div className="min-w-0 flex-1">
              <h2 className="text-lg font-semibold text-red-300">{title}</h2>
              <p className="mt-2 font-mono text-sm text-zinc-300 break-words">{message}</p>
              {stack && (
                <pre className="mt-3 overflow-x-auto rounded bg-zinc-800 p-3 text-xs text-zinc-400 whitespace-pre-wrap break-words">
                  {stack}
                </pre>
              )}
              {componentStack && (
                <details className="mt-2">
                  <summary className="cursor-pointer text-xs text-zinc-500 hover:text-zinc-400">Component stack</summary>
                  <pre className="mt-1 overflow-x-auto rounded bg-zinc-800 p-2 text-xs text-zinc-500 whitespace-pre-wrap">
                    {componentStack}
                  </pre>
                </details>
              )}
              <div className="mt-4 flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => this.setState({ error: null, errorInfo: null })}
                >
                  Dismiss
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => window.location.reload()}
                >
                  Reload app
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }
}
