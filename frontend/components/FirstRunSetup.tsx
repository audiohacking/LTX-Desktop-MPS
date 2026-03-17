import { useState, useEffect } from 'react'
import { logger } from '../lib/logger'
import './FirstRunSetup.css'

interface LaunchGateProps {
  licenseOnly?: boolean
  showLicenseStep?: boolean
  onComplete: () => Promise<void>
  onAcceptLicense?: () => Promise<void>
}

type Step = 'license' | 'location' | 'installing' | 'complete'

interface DownloadProgress {
  status: 'idle' | 'downloading' | 'complete' | 'error'
  currentFile: string
  currentFileProgress: number
  totalProgress: number
  downloadedBytes: number
  totalBytes: number
  filesCompleted: number
  totalFiles: number
  error: string | null
  speedMbps: number
}

// Fun loading messages
const INSTALL_MESSAGES = [
  "Downloading model weights...",
  "Teaching AI to dream in 4K...",
  "Loading neural pathways...",
  "Calibrating inference engine...",
  "Almost there...",
  "Unpacking the magic...",
  "Configuring parameters...",
  "Finalizing installation..."
]


export function LaunchGate({
  licenseOnly,
  showLicenseStep = true,
  onComplete,
  onAcceptLicense,
}: LaunchGateProps) {
  const [currentStep, setCurrentStep] = useState<Step>(showLicenseStep ? 'license' : 'location')
  const [installPath, setInstallPath] = useState('')
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [installMessage, setInstallMessage] = useState(INSTALL_MESSAGES[0])
  const [availableSpace, setAvailableSpace] = useState('...')
  const [videoPath, setVideoPath] = useState('/splash/splash.mp4')
  const [ltxApiKey, setLtxApiKey] = useState('')
  const [backendUrl, setBackendUrl] = useState<string | null>(null)
  const [licenseAccepted, setLicenseAccepted] = useState(false)
  const [licenseText, setLicenseText] = useState<string | null>(null)
  const [licenseError, setLicenseError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [isActionPending, setIsActionPending] = useState(false)
  const [haveModelsChecking, setHaveModelsChecking] = useState(false)
  const [haveModelsMessage, setHaveModelsMessage] = useState<string | null>(null)

  // Format bytes to human readable
  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`
  }

  // Format time remaining
  const formatTimeRemaining = (seconds: number): string => {
    if (!seconds || !isFinite(seconds) || seconds <= 0) return '--'
    if (seconds < 60) return `${Math.round(seconds)}s`
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`
    return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`
  }

  // Calculate ETA based on speed and remaining bytes
  const getTimeRemaining = (): string => {
    if (!downloadProgress || downloadProgress.speedMbps <= 0) return '--'
    const remainingBytes = downloadProgress.totalBytes - downloadProgress.downloadedBytes
    if (remainingBytes <= 0) return '--'
    const speedBytesPerSec = downloadProgress.speedMbps * 1024 * 1024
    const secondsRemaining = remainingBytes / speedBytesPerSec
    return formatTimeRemaining(secondsRemaining)
  }

  // Fetch license text
  const fetchLicense = async () => {
    setLicenseError(null)
    setLicenseText(null)
    try {
      const text = await window.electronAPI.fetchLicenseText()
      setLicenseText(text)
    } catch (e) {
      setLicenseError(e instanceof Error ? e.message : 'Failed to fetch license text.')
    }
  }

  // Initialize
  useEffect(() => {
    const init = async () => {
      try {
        // Get video path for production (unpacked from asar)
        try {
          const resourcePath = await window.electronAPI.getResourcePath?.()
          if (resourcePath) {
            setVideoPath(`file://${resourcePath}/app.asar.unpacked/dist/splash/splash.mp4`)
          }
        } catch {
          // Dev mode: use relative path
          setVideoPath('/splash/splash.mp4')
        }

        // Get models path from backend
        try {
          const url = await window.electronAPI.getBackendUrl()
          setBackendUrl(url)
          const response = await fetch(`${url}/api/models/status`)
          if (response.ok) {
            const data = await response.json()
            if (data.models_path) {
              setInstallPath(data.models_path)
            }
          }
        } catch (e) {
          logger.error(`Failed to get models path: ${e}`)
        }

        // TODO: Get actual available space
        setAvailableSpace('1.8 TB')
      } catch (e) {
        logger.error(`Init error: ${e}`)
      }
    }
    init()
    if (showLicenseStep) {
      void fetchLicense()
    }
  }, [showLicenseStep])

  // Cycle install messages
  useEffect(() => {
    if (currentStep !== 'installing') return
    let index = 0
    const interval = setInterval(() => {
      index = (index + 1) % INSTALL_MESSAGES.length
      setInstallMessage(INSTALL_MESSAGES[index])
    }, 4000)
    return () => clearInterval(interval)
  }, [currentStep])

  // Poll download progress during installation
  useEffect(() => {
    if (currentStep !== 'installing' || !backendUrl) return

    const pollProgress = async () => {
      try {
        const response = await fetch(`${backendUrl}/api/models/download/progress`)
        if (response.ok) {
          const progress = await response.json()
          setDownloadProgress(progress)

          if (progress.status === 'error') {
            setDownloadError(progress.error || 'Download failed.')
          } else if (progress.status === 'complete') {
            setTimeout(() => setCurrentStep('complete'), 600)
          }
        }
      } catch (e) {
        logger.error(`Progress poll error: ${e}`)
      }
    }

    pollProgress()
    const interval = setInterval(pollProgress, 300)
    return () => clearInterval(interval)
  }, [currentStep, backendUrl])

  // Start installation
  const startInstallation = async () => {
    if (!backendUrl) return
    setCurrentStep('installing')
    try {
      // If API key is provided, save it to settings first and skip text encoder download
      if (ltxApiKey.trim()) {
        try {
          await fetch(`${backendUrl}/api/settings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ltxApiKey: ltxApiKey.trim() }),
          })
        } catch (e) {
          logger.error(`Failed to save API key: ${e}`)
        }
      }

      // Start download - skip text encoder if API key is provided
      await fetch(`${backendUrl}/api/models/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ skipTextEncoder: !!ltxApiKey.trim() }),
      })
    } catch (e) {
      logger.error(`Download start error: ${e}`)
      setDownloadError(e instanceof Error ? e.message : 'Failed to start model download.')
    }
  }

  const retryInstallation = () => {
    setDownloadError(null)
    startInstallation()
  }

  // Handle next button
  const handleNext = async () => {
    setActionError(null)
    if (currentStep === 'license') {
      if (!licenseAccepted) return
      setIsActionPending(true)
      try {
        if (onAcceptLicense) {
          await onAcceptLicense()
        }
        if (licenseOnly) {
          await onComplete()
          return
        }
        setCurrentStep('location')
      } catch (e) {
        setActionError(e instanceof Error ? e.message : 'Failed to accept license.')
      } finally {
        setIsActionPending(false)
      }
      return
    }
    if (currentStep === 'location') {
      startInstallation()
      return
    }
    if (currentStep === 'complete') {
      await handleFinish()
    }
  }

  const handleFinish = async () => {
    setActionError(null)
    setIsActionPending(true)
    try {
      await onComplete()
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Failed to complete setup.')
    } finally {
      setIsActionPending(false)
    }
  }

  // Get button text
  const getNextButtonText = () => {
    if (currentStep === 'license') return licenseOnly ? 'Accept' : 'Next'
    if (currentStep === 'location') return 'Install'
    if (currentStep === 'complete') return 'Finish'
    return 'Continue'
  }

  // Check if next button should be disabled
  const isNextDisabled = () => {
    if (currentStep === 'license') return !licenseAccepted || isActionPending
    if (currentStep === 'complete') return isActionPending
    return false
  }

  return (
    <div className="h-screen flex flex-col" style={{
      background: '#000000',
      fontFamily: 'Arial, Helvetica, sans-serif',
      color: '#ffffff'
    }}>
      {/* Custom Title Bar */}
      <div style={{
        height: 32,
        background: '#000000',
        display: 'flex',
        alignItems: 'center',
        paddingLeft: 80,
        borderBottom: '1px solid #1a1a1a',
        // @ts-expect-error - Electron-specific CSS property
        WebkitAppRegion: 'drag'
      }}>
      </div>

      {/* Main Container */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        overflow: 'hidden',
        minHeight: 0,
        // @ts-expect-error - Electron-specific CSS property
        WebkitAppRegion: 'no-drag'
      }}>
        {/* Header */}
        <div style={{
          padding: currentStep === 'installing' ? '12px 32px' : '16px 32px',
          borderBottom: '1px solid #1a1a1a'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            {/* Director's Desktop Logo */}
            <svg style={{ height: 24, width: 24 }} viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg" fill="none">
              <path d="M64 12C34 12 10 36 10 64c0 28 24 52 54 52 6 0 10-4 10-9 0-2.5-1-4.5-2-6-1-1.5-2-3.5-2-6 0-5 4-9 9-9h11c17 0 30-13 30-30C120 32 96 12 64 12Z" fill="#8B4513"/>
              <rect x="20" y="8" width="88" height="16" rx="4" fill="#2A2A2A"/>
              <rect x="28" y="10" width="8" height="12" rx="1" fill="white" transform="rotate(-15 32 16)"/>
              <rect x="44" y="10" width="8" height="12" rx="1" fill="white" transform="rotate(-15 48 16)"/>
              <rect x="60" y="10" width="8" height="12" rx="1" fill="white" transform="rotate(-15 64 16)"/>
              <rect x="76" y="10" width="8" height="12" rx="1" fill="white" transform="rotate(-15 80 16)"/>
              <rect x="92" y="10" width="8" height="12" rx="1" fill="white" transform="rotate(-15 96 16)"/>
              <circle cx="38" cy="42" r="8" fill="#E74C3C"/>
              <circle cx="58" cy="36" r="7" fill="#F39C12"/>
              <circle cx="78" cy="36" r="7" fill="#3498DB"/>
              <circle cx="94" cy="46" r="7" fill="#2ECC71"/>
              <circle cx="98" cy="66" r="6" fill="#9B59B6"/>
              <circle cx="34" cy="62" r="6" fill="#E67E22"/>
              <circle cx="56" cy="80" r="12" fill="#1a1a1a"/>
              <circle cx="56" cy="80" r="9" fill="#111"/>
            </svg>
            <span style={{ fontSize: 14, color: '#71717a', fontWeight: 500, letterSpacing: '0.02em', paddingTop: 2, paddingLeft: 6 }}>Director's Desktop</span>
          </div>
        </div>

        {/* Content Area */}
        <div style={{
          flex: 1,
          padding: currentStep === 'installing' ? 0 : '28px 32px',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}>
          {/* Step 1: Model License */}
          {currentStep === 'license' && (
            <div style={{ animation: 'fadeIn 0.25s ease', display: 'flex', flexDirection: 'column', overflow: 'hidden', flex: 1 }}>
              <h2 style={{
                fontFamily: "'Miriam Libre', serif",
                fontSize: 24,
                fontWeight: 700,
                marginBottom: 6
              }}>
                LTX-2 Model License
              </h2>
              <p style={{ color: '#a0a0a0', fontSize: 14, marginBottom: 16 }}>
                The LTX-2 model is subject to the following license agreement. Please review and accept before downloading.
              </p>

              <div style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
                minHeight: 0
              }}>
                <div style={{
                  flex: 1,
                  overflow: 'hidden',
                  borderRadius: 8,
                  minHeight: 0
                }}>
                  {licenseError ? (
                    <div style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      height: '100%',
                      gap: 12
                    }}>
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#f87171" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                      </svg>
                      <span style={{ color: '#f87171', fontSize: 13, textAlign: 'center' }}>{licenseError}</span>
                      <button
                        onClick={fetchLicense}
                        style={{
                          padding: '6px 20px',
                          borderRadius: 9999,
                          fontSize: 13,
                          fontWeight: 600,
                          cursor: 'pointer',
                          background: 'linear-gradient(125deg, #A98BD9, #6D28D9)',
                          border: 'none',
                          color: '#ffffff',
                        }}
                      >
                        Retry
                      </button>
                    </div>
                  ) : licenseText === null ? (
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      height: '100%',
                      gap: 10
                    }}>
                      <svg width="20" height="20" viewBox="0 0 24 24" style={{ animation: 'spin 1s linear infinite' }}>
                        <circle cx="12" cy="12" r="10" stroke="#6D28D9" strokeWidth="3" fill="none" strokeDasharray="31.4 31.4" strokeLinecap="round" />
                      </svg>
                      <span style={{ color: '#a0a0a0', fontSize: 13 }}>Loading license...</span>
                    </div>
                  ) : (
                    <div style={{
                      overflowY: 'auto',
                      height: '100%',
                      background: '#1a1a1a',
                      borderRadius: 8,
                      padding: 40
                    }}>
                      <pre style={{
                        fontFamily: "'Consolas', 'Monaco', monospace",
                        fontSize: 11,
                        lineHeight: 1.5,
                        color: '#d0d0d0',
                        margin: 0,
                        whiteSpace: 'pre-line',
                        wordWrap: 'break-word',
                        width: '100%'
                      }}>
                        {licenseText?.replace(/([^\n])\n([^\n])/g, '$1 $2')}
                      </pre>
                    </div>
                  )}
                </div>

                <label style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  marginTop: 14,
                  cursor: 'pointer',
                  fontSize: 13,
                  userSelect: 'none'
                }}>
                  <input
                    type="checkbox"
                    checked={licenseAccepted}
                    onChange={(e) => setLicenseAccepted(e.target.checked)}
                    style={{
                      width: 16,
                      height: 16,
                      accentColor: '#2B61FF',
                      cursor: 'pointer',
                      flexShrink: 0
                    }}
                  />
                  <span>I have read and agree to the LTX-2 Community License Agreement</span>
                </label>
              </div>
            </div>
          )}

          {/* Step 2: Choose Location */}
          {currentStep === 'location' && (
            <div style={{ animation: 'fadeIn 0.25s ease' }}>
              <h2 style={{
                fontFamily: "'Miriam Libre', serif",
                fontSize: 24,
                fontWeight: 700,
                marginBottom: 6
              }}>
                Choose Location
              </h2>
              <p style={{ color: '#a0a0a0', fontSize: 14, marginBottom: 24 }}>
                Select where to install the model files.
              </p>

              <div style={{
                background: '#2e3445',
                borderRadius: 12,
                padding: '14px 18px'
              }}>
                <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                  <input
                    type="text"
                    value={installPath}
                    readOnly
                    style={{
                      flex: 1,
                      background: '#1a1a1a',
                      border: '1px solid #333',
                      borderRadius: 8,
                      padding: '12px 14px',
                      color: '#ffffff',
                      fontSize: 13,
                      fontFamily: "'Consolas', 'Monaco', monospace"
                    }}
                  />
                  <button
                    onClick={async () => {
                      const dir = await window.electronAPI?.showOpenDirectoryDialog({ title: 'Select Model Installation Folder' })
                      if (dir) {
                        setInstallPath(dir)
                      }
                    }}
                    style={{
                      padding: '10px 28px',
                      borderRadius: 9999,
                      fontSize: 13,
                      fontWeight: 600,
                      cursor: 'pointer',
                      background: 'transparent',
                      border: '1px solid #444',
                      color: '#ffffff',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    Browse
                  </button>
                </div>

                <div style={{
                  display: 'flex',
                  justifyContent: 'flex-end',
                  fontSize: 12,
                  color: '#a0a0a0',
                  marginTop: 10
                }}>
                  <span>Available: <strong style={{ color: '#fff' }}>{availableSpace}</strong></span>
                </div>
              </div>

              {/* LTX API Key - Optional but saves ~25 GB download */}
              <div style={{
                marginTop: 24,
                background: '#2e3445',
                borderRadius: 12,
                padding: '14px 18px'
              }}>
                <div style={{ marginBottom: 8 }}>
                  <label style={{ fontSize: 13, fontWeight: 600, color: '#ffffff' }}>
                    LTX API Key
                    <span style={{
                      fontSize: 11,
                      color: '#A98BD9',
                      marginLeft: 8,
                      fontWeight: 400
                    }}>
                      Optional - Saves ~25 GB download
                    </span>
                  </label>
                </div>
                <input
                  type="password"
                  value={ltxApiKey}
                  onChange={(e) => setLtxApiKey(e.target.value)}
                  placeholder="Enter API key to skip text encoder download..."
                  style={{
                    width: '100%',
                    background: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: 8,
                    padding: '12px 14px',
                    color: '#ffffff',
                    fontSize: 13,
                    boxSizing: 'border-box'
                  }}
                />
                <p style={{ fontSize: 11, color: '#888', marginTop: 8 }}>
                  {ltxApiKey ? (
                    <span style={{ color: '#6D28D9' }}>
                      ✓ Text encoder download will be skipped (using API instead)
                    </span>
                  ) : (
                    'If you have an LTX API key, entering it here skips the 25 GB text encoder download. ' +
                    'The API provides faster text encoding (~1s vs 23s local).'
                  )}
                </p>
              </div>

              {/* I already have models */}
              <div style={{ marginTop: 20, paddingTop: 20, borderTop: '1px solid #333' }}>
                <p style={{ fontSize: 13, color: '#a0a0a0', marginBottom: 10 }}>
                  Already have LTX or ComfyUI models on this machine?
                </p>
                <button
                  type="button"
                  onClick={async () => {
                    if (!backendUrl || haveModelsChecking) return
                    setHaveModelsMessage(null)
                    const dir = await window.electronAPI?.showOpenDirectoryDialog({ title: 'Select folder containing models (e.g. ComfyUI models)' })
                    if (!dir) return
                    setHaveModelsChecking(true)
                    try {
                      await fetch(`${backendUrl}/api/settings`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ comfyuiModelsPath: dir }),
                      })
                      const res = await fetch(`${backendUrl}/api/models/check-path`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ comfyuiModelsPath: dir }),
                      })
                      if (!res.ok) throw new Error('Could not scan folder.')
                      const data = await res.json() as { all_downloaded?: boolean }
                      if (data.all_downloaded === true) {
                        await onComplete()
                        return
                      }
                      setHaveModelsMessage('No models found in that folder. Restart the app after choosing the correct folder.')
                    } catch (e) {
                      setHaveModelsMessage(e instanceof Error ? e.message : 'Could not scan folder.')
                    } finally {
                      setHaveModelsChecking(false)
                    }
                  }}
                  disabled={haveModelsChecking}
                  style={{
                    padding: '8px 16px',
                    borderRadius: 8,
                    fontSize: 13,
                    fontWeight: 500,
                    cursor: haveModelsChecking ? 'default' : 'pointer',
                    background: 'transparent',
                    border: '1px solid #555',
                    color: '#A98BD9',
                  }}
                >
                  {haveModelsChecking ? 'Scanning...' : 'I already have models'}
                </button>
                {haveModelsMessage && (
                  <p style={{ fontSize: 12, color: '#a0a0a0', marginTop: 10 }}>{haveModelsMessage}</p>
                )}
              </div>
            </div>
          )}

          {/* Step 3: Installing */}
          {currentStep === 'installing' && (
            <div style={{
              position: 'relative',
              height: '100%',
              animation: 'fadeIn 0.25s ease'
            }}>
              {/* Video Section - fills container but leaves room for progress */}
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 140,
                background: '#0a0a0a',
                overflow: 'hidden'
              }}>
                {/* Splash Video */}
                <video
                  key={videoPath}
                  autoPlay
                  loop
                  muted
                  playsInline
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    display: 'block'
                  }}
                >
                  <source src={videoPath} type="video/mp4" />
                </video>

                {/* Video Credit */}
                <div style={{
                  position: 'absolute',
                  bottom: 20,
                  left: 24,
                  fontFamily: "'Miriam Libre', serif",
                  fontSize: 13,
                  color: 'rgba(255,255,255,0.75)',
                  textShadow: '0 1px 4px rgba(0,0,0,0.9)',
                  zIndex: 10
                }}>
                  Generated by PongFlongo
                </div>
              </div>

              {/* Progress Section - fixed at bottom */}
              <div style={{
                position: 'absolute',
                left: 0,
                right: 0,
                bottom: 0,
                height: 140,
                background: '#0d0d0d',
                padding: '16px 24px',
                borderTop: '1px solid #2a2a2a'
              }}>
              {downloadError ? (
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '100%',
                  gap: 10,
                }}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#f87171" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                  </svg>
                  <span style={{ color: '#f87171', fontSize: 13, textAlign: 'center', maxWidth: 400 }}>{downloadError}</span>
                  <div style={{ display: 'flex', gap: 10 }}>
                    <button
                      onClick={() => { setDownloadError(null); setCurrentStep('location') }}
                      style={{
                        padding: '6px 20px',
                        borderRadius: 9999,
                        fontSize: 13,
                        fontWeight: 600,
                        cursor: 'pointer',
                        background: 'transparent',
                        border: '1px solid #444',
                        color: '#ffffff',
                      }}
                    >
                      Back
                    </button>
                    <button
                      onClick={retryInstallation}
                      style={{
                        padding: '6px 20px',
                        borderRadius: 9999,
                        fontSize: 13,
                        fontWeight: 600,
                        cursor: 'pointer',
                        background: 'linear-gradient(125deg, #A98BD9, #6D28D9)',
                        border: 'none',
                        color: '#ffffff',
                      }}
                    >
                      Retry
                    </button>
                  </div>
                </div>
              ) : (
              <>
                {/* Header row with status and percentage */}
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: 8
                }}>
                  <span style={{ fontSize: 13, fontWeight: 500 }}>
                    {(downloadProgress?.totalProgress || 0) > 85 ? 'Installing...' : 'Downloading...'}
                  </span>
                  <span style={{ fontSize: 13, color: '#A98BD9', fontWeight: 600 }}>
                    {downloadProgress?.totalProgress || 0}%
                  </span>
                </div>

                {/* Progress Bar */}
                <div style={{
                  height: 6,
                  background: '#1a1a1a',
                  borderRadius: 3,
                  overflow: 'hidden'
                }}>
                  <div style={{
                    height: '100%',
                    background: 'linear-gradient(125deg, #A98BD9, #6D28D9, #194DF9)',
                    backgroundSize: '200% 200%',
                    animation: 'gradientShift 3s ease infinite',
                    borderRadius: 3,
                    width: `${downloadProgress?.totalProgress || 0}%`,
                    transition: 'width 0.3s ease'
                  }} />
                </div>

                {/* Download stats row */}
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginTop: 10,
                  fontSize: 12,
                  color: '#a0a0a0'
                }}>
                  {/* Current file */}
                  <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {downloadProgress?.currentFile || installMessage}
                  </span>

                  {/* Speed and ETA */}
                  <div style={{ display: 'flex', gap: 16, marginLeft: 16, flexShrink: 0 }}>
                    {downloadProgress && downloadProgress.speedMbps > 0 && (
                      <span style={{ color: '#6D28D9', fontWeight: 500 }}>
                        {downloadProgress.speedMbps.toFixed(1)} MB/s
                      </span>
                    )}
                    {downloadProgress && downloadProgress.totalBytes > 0 && (
                      <span>
                        {formatBytes(downloadProgress.downloadedBytes)} / {formatBytes(downloadProgress.totalBytes)}
                      </span>
                    )}
                    {downloadProgress && downloadProgress.speedMbps > 0 && (
                      <span>
                        ETA: {getTimeRemaining()}
                      </span>
                    )}
                  </div>
                </div>

                {/* Files progress */}
                {downloadProgress && downloadProgress.totalFiles > 0 && (
                  <div style={{
                    marginTop: 6,
                    fontSize: 11,
                    color: '#666'
                  }}>
                    File {downloadProgress.filesCompleted + 1} of {downloadProgress.totalFiles}
                  </div>
                )}
              </>
              )}
              </div>
            </div>
          )}

          {/* Step 4: Complete */}
          {currentStep === 'complete' && (
            <div style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              textAlign: 'center',
              animation: 'fadeIn 0.25s ease'
            }}>
              {/* Success Icon */}
              <div style={{
                width: 72,
                height: 72,
                background: 'linear-gradient(125deg, #A98BD9, #6D28D9, #194DF9)',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: 20
              }}>
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                  <polyline points="20 6 9 17 4 12"/>
                </svg>
              </div>

              <h2 style={{
                fontFamily: "'Miriam Libre', serif",
                fontSize: 26,
                fontWeight: 700,
                marginBottom: 8
              }}>
                Ready to Create
              </h2>
              <p style={{ color: '#a0a0a0', fontSize: 14, maxWidth: 320 }}>
                Director's Desktop is ready. Start generating.
              </p>

              {/* Install Summary */}
              <div style={{
                background: '#2e3445',
                borderRadius: 12,
                padding: '16px 28px',
                marginTop: 20,
                textAlign: 'left',
                minWidth: 260
              }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  padding: '8px 0',
                  fontSize: 13
                }}>
                  <span style={{ color: '#a0a0a0' }}>Location</span>
                  <span style={{ fontWeight: 500, maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {installPath.split('\\').pop() || installPath}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{
          padding: currentStep === 'installing' ? '12px 24px' : '16px 32px',
          borderTop: '1px solid #1a1a1a',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div style={{ fontSize: 11, color: '#666' }}>© 2026 Lightricks</div>

          <div style={{ display: 'flex', gap: 10 }}>
            {/* Next/Install/Finish Button */}
            {currentStep !== 'installing' && (
              <button
                onClick={() => void handleNext()}
                disabled={isNextDisabled()}
                style={{
                  padding: '10px 28px',
                  borderRadius: 8,
                  fontSize: 13,
                  fontWeight: 700,
                  cursor: isNextDisabled() ? 'not-allowed' : 'pointer',
                  background: isNextDisabled() ? '#555' : '#2B61FF',
                  border: 'none',
                  color: '#ffffff',
                  transition: 'all 0.2s ease',
                  opacity: isNextDisabled() ? 0.6 : 1
                }}
              >
                {getNextButtonText()}
              </button>
            )}
          </div>
        </div>
        {actionError && (
          <div style={{ padding: '0 32px 12px 32px', color: '#fca5a5', fontSize: 12 }}>
            {actionError}
          </div>
        )}
      </div>

    </div>
  )
}
