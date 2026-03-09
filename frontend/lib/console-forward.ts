/**
 * Forward renderer console.log / info / warn / error to the Electron main process
 * so they appear in the session log and terminal (embedded console) for troubleshooting.
 * Only runs when window.electronAPI.sendRendererLog is available (Electron app).
 */
export function initConsoleForward(): void {
  const api = (window as Window & { electronAPI?: { sendRendererLog?: (level: 'log' | 'info' | 'warn' | 'error', ...args: unknown[]) => void } }).electronAPI
  const send = api?.sendRendererLog
  if (!send) return

  const forward = (level: 'log' | 'info' | 'warn' | 'error') => {
    const original = console[level]
    if (typeof original !== 'function') return
    console[level] = function (...args: unknown[]) {
      try {
        send(level, ...args)
      } catch {
        // ignore IPC errors
      }
      return original.apply(console, args)
    }
  }

  forward('log')
  forward('info')
  forward('warn')
  forward('error')
}
