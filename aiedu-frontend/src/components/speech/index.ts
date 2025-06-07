// Speech components
export { default as AudioRecorder } from './AudioRecorder'
export { default as TextToSpeech } from './TextToSpeech'
export { default as VoiceSelector } from './VoiceSelector'
export { default as SpeechControls } from './SpeechControls'

// Re-export speech hook
export { useSpeech } from '../../hooks/useSpeech'

// Re-export types
export type { Voice, SynthesizeRequest, SynthesizeResponse, TranscribeResponse, RecordingState } from '../../types/speech' 