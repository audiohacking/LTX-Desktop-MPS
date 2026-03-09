import { ipcMain } from 'electron'
import fs from 'fs'
import { getLogDir, getCurrentLogFilename } from '../logging-management'
import { logger, writeLog } from '../logger'

const VALID_LOG_LEVELS = new Set(['INFO', 'WARNING', 'ERROR', 'DEBUG'])

function formatRendererArgs(args: unknown[]): string {
  return args
    .map((a) => (typeof a === 'object' && a !== null ? JSON.stringify(a) : String(a)))
    .join(' ')
}

export function registerLogHandlers(): void {
  ipcMain.on('renderer-console', (_event, level: string, args: unknown[]) => {
    const message = formatRendererArgs(Array.isArray(args) ? args : [args])
    const tag = '[Renderer]'
    if (level === 'error') {
      logger.error(`${tag} ${message}`)
      writeLog('ERROR', `${tag} ${message}`)
    } else if (level === 'warn') {
      logger.warn(`${tag} ${message}`)
      writeLog('WARNING', `${tag} ${message}`)
    } else {
      logger.info(`${tag} ${message}`)
      writeLog('INFO', `${tag} ${message}`)
    }
  })

  ipcMain.handle('write-log', async (_event, level: string, message: string) => {
    const upperLevel = String(level).toUpperCase()
    if (!VALID_LOG_LEVELS.has(upperLevel)) return
    writeLog(upperLevel as 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG', 'Renderer', String(message))
  })
  ipcMain.handle('get-logs', async () => {
    try {
      const logPath = getCurrentLogFilename()
      if (fs.existsSync(logPath)) {
        const content = fs.readFileSync(logPath, 'utf-8')
        const allLines = content.split('\n')
        const lines = allLines.slice(-200).map(l => l.trimEnd())
        return { logPath, lines }
      }
      return { logPath, lines: [] }
    } catch (error) {
      logger.error(`Error getting logs: ${error}`)
      return { logPath: '', lines: [], error: String(error) }
    }
  })

  ipcMain.handle('get-log-path', async () => {
    const logPath = getCurrentLogFilename()
    const logDir = getLogDir()
    return { logPath, logDir }
  })

  ipcMain.handle('open-log-folder', async () => {
    const logDir = getLogDir()
    if (fs.existsSync(logDir)) {
      const { shell } = await import('electron')
      shell.openPath(logDir)
      return true
    }
    return false
  })
}
