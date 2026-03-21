/**
 * Backend URL and auth token (from Electron). Use for all HTTP and WebSocket calls.
 * When token is empty, backend auth is disabled (e.g. dev).
 */

export async function getBackend(): Promise<{ url: string; token: string }> {
  return window.electronAPI.getBackend()
}

export async function backendFetch(path: string, options: RequestInit = {}): Promise<Response> {
  const { url, token } = await getBackend()
  const headers = new Headers(options.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return fetch(`${url}${path}`, { ...options, headers })
}

/** Build WebSocket URL with optional token query param for auth. */
export function backendWsUrl(baseUrl: string, path: string, token: string): string {
  const wsUrl = baseUrl.replace(/^http/, 'ws') + path
  if (!token) return wsUrl
  return `${wsUrl}${path.includes('?') ? '&' : '?'}token=${encodeURIComponent(token)}`
}
