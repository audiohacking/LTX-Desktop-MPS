import { app } from 'electron'
import path from 'path'
import os from 'os'

// Use 8010 to avoid colliding with ComfyUI (often 8000 or 8188)
export const PYTHON_PORT = 8010
export const BACKEND_BASE_URL = `http://localhost:${PYTHON_PORT}`
export const isDev = !app.isPackaged

// Get directory - works in both CJS and ESM contexts
export function getCurrentDir(): string {
  // In bundled output, use app.getAppPath()
  if (!isDev) {
    return path.dirname(app.getPath('exe'))
  }
  // In development, use process.cwd() which is the project root
  return process.cwd()
}

export function getAllowedRoots(): string[] {
  const roots = [
    getCurrentDir(),
    app.getPath('userData'),
    app.getPath('downloads'),
    os.tmpdir(),
  ]
  if (!isDev && process.resourcesPath) {
    roots.push(process.resourcesPath)
  }
  return roots
}
