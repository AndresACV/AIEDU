'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { apiClient } from '@/services/api'
import { TranscribeResponse } from '@/types/speech'

interface AudioRecorderProps {
  onTranscription?: (text: string, confidence?: number) => void
  onError?: (error: string) => void
  language?: string
  className?: string
}

interface RecordingState {
  isRecording: boolean
  isProcessing: boolean
  volume: number
  duration: number
  error?: string
}

export default function AudioRecorder({
  onTranscription,
  onError,
  language = 'en-US',
  className = ''
}: AudioRecorderProps) {
  const [state, setState] = useState<RecordingState>({
    isRecording: false,
    isProcessing: false,
    volume: 0,
    duration: 0
  })

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const animationFrameRef = useRef<number>()
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const startTimeRef = useRef<number>(0)
  const durationIntervalRef = useRef<NodeJS.Timeout>()

  // Volume visualization
  const updateVolume = useCallback(() => {
    if (!analyserRef.current) return

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
    analyserRef.current.getByteFrequencyData(dataArray)
    
    const volume = dataArray.reduce((acc, value) => acc + value, 0) / dataArray.length
    setState(prev => ({ ...prev, volume: Math.min(volume / 255 * 100, 100) }))

    if (state.isRecording) {
      animationFrameRef.current = requestAnimationFrame(updateVolume)
    }
  }, [state.isRecording])

  // Duration tracking
  const updateDuration = useCallback(() => {
    if (startTimeRef.current > 0) {
      const elapsed = (Date.now() - startTimeRef.current) / 1000
      setState(prev => ({ ...prev, duration: elapsed }))
    }
  }, [])

  // Start recording
  const startRecording = async () => {
    try {
      setState(prev => ({ ...prev, error: undefined }))

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      })

      streamRef.current = stream

      // Setup audio context for volume visualization
      audioContextRef.current = new AudioContext()
      const source = audioContextRef.current.createMediaStreamSource(stream)
      analyserRef.current = audioContextRef.current.createAnalyser()
      analyserRef.current.fftSize = 256
      source.connect(analyserRef.current)

      // Setup MediaRecorder
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })

      audioChunksRef.current = []

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorderRef.current.onstop = handleRecordingStop

      // Start recording
      mediaRecorderRef.current.start(100) // Collect data every 100ms
      startTimeRef.current = Date.now()
      
      setState(prev => ({ ...prev, isRecording: true, duration: 0 }))

      // Start volume visualization
      updateVolume()

      // Start duration tracking
      durationIntervalRef.current = setInterval(updateDuration, 100)

    } catch (error) {
      const errorMessage = `Failed to start recording: ${error instanceof Error ? error.message : 'Unknown error'}`
      setState(prev => ({ ...prev, error: errorMessage }))
      onError?.(errorMessage)
    }
  }

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && state.isRecording) {
      mediaRecorderRef.current.stop()
      setState(prev => ({ ...prev, isRecording: false }))

      // Cleanup
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current)
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close()
        audioContextRef.current = null
      }
    }
  }

  // Handle recording stop
  const handleRecordingStop = async () => {
    if (audioChunksRef.current.length === 0) {
      setState(prev => ({ ...prev, error: 'No audio data recorded' }))
      return
    }

    setState(prev => ({ ...prev, isProcessing: true }))

    try {
      // Create audio blob
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
      
      // Convert to WAV if needed (FastAPI expects WAV)
      const wavBlob = await convertToWav(audioBlob)

      // Send to speech service
      const response: TranscribeResponse = await apiClient.transcribe(wavBlob, language)

      if (response.success && response.text) {
        onTranscription?.(response.text, response.confidence)
      } else {
        const error = response.error || 'Transcription failed'
        setState(prev => ({ ...prev, error }))
        onError?.(error)
      }
    } catch (error) {
      const errorMessage = `Transcription error: ${error instanceof Error ? error.message : 'Unknown error'}`
      setState(prev => ({ ...prev, error: errorMessage }))
      onError?.(errorMessage)
    } finally {
      setState(prev => ({ ...prev, isProcessing: false }))
    }
  }

  // Convert audio blob to WAV format
  const convertToWav = async (audioBlob: Blob): Promise<Blob> => {
    // For now, return the original blob as most modern browsers support webm
    // In production, you might want to use a library like 'lamejs' for proper WAV conversion
    return audioBlob
  }

  // Format duration
  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // Volume bar visualization
  const VolumeBar = () => (
    <div className="flex items-center space-x-2">
      <div className="w-20 h-2 bg-gray-200 rounded-full overflow-hidden">
        <div 
          className="h-full bg-gradient-to-r from-green-400 to-red-500 transition-all duration-100"
          style={{ width: `${state.volume}%` }}
        />
      </div>
      <span className="text-xs text-gray-500 min-w-[3rem]">
        {Math.round(state.volume)}%
      </span>
    </div>
  )

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current)
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close()
        audioContextRef.current = null
      }
    }
  }, [])

  return (
    <div className={`bg-white rounded-lg border border-gray-200 p-4 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">Audio Recorder</h3>
        <div className="text-sm text-gray-500">
          Language: {language === 'en-US' ? 'English' : 'Spanish'}
        </div>
      </div>

      {/* Recording Controls */}
      <div className="flex items-center space-x-4 mb-4">
        <button
          onClick={state.isRecording ? stopRecording : startRecording}
          disabled={state.isProcessing}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all ${
            state.isRecording
              ? 'bg-red-500 hover:bg-red-600 text-white'
              : 'bg-blue-500 hover:bg-blue-600 text-white'
          } ${state.isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {state.isRecording ? (
            <>
              <div className="w-3 h-3 bg-white rounded-sm animate-pulse" />
              <span>Stop Recording</span>
            </>
          ) : (
            <>
              <div className="w-3 h-3 bg-white rounded-full" />
              <span>Start Recording</span>
            </>
          )}
        </button>

        {state.isProcessing && (
          <div className="flex items-center space-x-2 text-blue-600">
            <div className="animate-spin w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full" />
            <span className="text-sm">Processing...</span>
          </div>
        )}
      </div>

      {/* Recording Status */}
      {state.isRecording && (
        <div className="bg-gray-50 rounded-lg p-3 mb-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">Recording</span>
            <span className="text-sm text-gray-500">{formatDuration(state.duration)}</span>
          </div>
          <VolumeBar />
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            <span className="text-xs text-gray-500">Speak clearly into your microphone</span>
          </div>
        </div>
      )}

      {/* Error Display */}
      {state.error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-red-500 rounded-full flex-shrink-0" />
            <span className="text-sm text-red-700">{state.error}</span>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="text-xs text-gray-500 bg-gray-50 rounded-lg p-3">
        <p><strong>Tips for best results:</strong></p>
        <ul className="mt-1 space-y-1">
          <li>• Speak clearly and at normal volume</li>
          <li>• Ensure microphone permissions are granted</li>
          <li>• Record in a quiet environment</li>
          <li>• Keep recordings under 30 seconds for fastest processing</li>
        </ul>
      </div>
    </div>
  )
} 