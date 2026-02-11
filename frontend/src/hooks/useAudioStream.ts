import { useState, useEffect, useRef, useCallback } from 'react';

export interface Message {
  type: 'transcript' | 'agent_response' | 'thinking' | 'agent_audio' |
  'orchestrator_intro' | 'orchestrator_question' | 'orchestrator_redirect' |
  'orchestrator_interrupt' | 'orchestrator_topic_switch' | 'orchestrator_wrapup' |
  'response_evaluation';
  speaker_id?: string;
  speaker_name?: string;
  speaker_role?: string;
  text?: string;
  query?: string;
  response?: string;
  question?: string;
  message?: string;
  timestamp?: string;
  audio_data?: string;
  audio_format?: string;
  sample_rate?: number;
  turn_number?: number;
  is_redirect?: boolean;
  evaluation?: any;
  topic?: string;
  total_turns?: number;
  speaker_turns?: any;
}

export const useAudioStream = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isThinking, setIsThinking] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const currentAudioUrlRef = useRef<string | null>(null);
  const audioQueueRef = useRef<Array<{ data: string; format: string; sampleRate?: number }>>([]);
  const isRecordingPausedRef = useRef(false);
  const audioChunkCounterRef = useRef(0);

  const pauseRecording = useCallback(() => {
    // Don't actually pause - just set a flag
    // We need to keep sending audio to AWS Transcribe to prevent timeout
    if (!isRecordingPausedRef.current) {
      console.log('🎙️ Muting user input during TTS playback (still sending to Transcribe)');
      isRecordingPausedRef.current = true;
      console.log('✅ Recording muted flag set to:', isRecordingPausedRef.current);
    }
  }, []);

  const resumeRecording = useCallback(() => {
    if (isRecordingPausedRef.current) {
      console.log('🎙️ Unmuting user input after TTS playback');
      isRecordingPausedRef.current = false;
      console.log('✅ Recording unmuted flag set to:', isRecordingPausedRef.current);
    }
  }, []);

  const playNextAudio = useCallback(() => {
    if (audioQueueRef.current.length === 0) {
      console.log('📭 Audio queue empty');
      setIsPlayingAudio(false);

      // Notify backend that TTS has ended
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'tts_ended' }));
      }

      // Resume recording when no more audio to play
      resumeRecording();
      return;
    }

    const { data, format, sampleRate } = audioQueueRef.current.shift()!;
    console.log(`🎵 Playing audio from queue (${audioQueueRef.current.length} remaining)`);

    try {
      console.log('🔊 Attempting to play audio:', {
        format,
        sampleRate,
        dataLength: data.length,
        firstChars: data.substring(0, 50)
      });

      setIsPlayingAudio(true);

      // Decode base64 audio data
      const binaryString = atob(data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      console.log('✅ Decoded audio bytes:', bytes.length);

      // Determine MIME type
      let mimeType = 'audio/wav';
      if (format === 'wav') {
        mimeType = 'audio/wav';
      } else if (format === 'mp3') {
        mimeType = 'audio/mpeg';
      }

      const blob = new Blob([bytes], { type: mimeType });
      console.log('✅ Created blob:', { type: mimeType, size: blob.size });

      const url = URL.createObjectURL(blob);
      console.log('✅ Created object URL:', url);

      // Store URL for cleanup
      currentAudioUrlRef.current = url;

      const audio = new Audio(url);

      // Keep strong reference to prevent garbage collection
      currentAudioRef.current = audio;

      // Configure audio element
      audio.preload = 'auto';
      audio.volume = 1.0; // Set volume to maximum
      audio.preservesPitch = true;

      audio.onloadedmetadata = () => {
        console.log('✅ Audio metadata loaded:', {
          duration: audio.duration,
          readyState: audio.readyState
        });
      };

      audio.onloadeddata = () => {
        console.log('✅ Audio data loaded');
      };

      audio.oncanplay = () => {
        console.log('✅ Audio can play');
      };

      audio.oncanplaythrough = () => {
        console.log('✅ Audio can play through');
      };

      audio.onplay = () => {
        console.log('▶️  Audio started playing');
      };

      audio.onpause = () => {
        console.log('⏸️  Audio paused');

        // If audio was paused externally (not by user), try to resume
        if (currentAudioRef.current === audio && !audio.ended) {
          console.log('⚠️  Audio was paused externally, attempting to resume...');
          setTimeout(() => {
            if (audio && !audio.ended && audio.paused) {
              audio.play().catch(err => {
                console.error('Failed to resume paused audio:', err);
              });
            }
          }, 100);
        }
      };

      audio.ontimeupdate = () => {
        // Log progress every second
        if (Math.floor(audio.currentTime) !== Math.floor(audio.currentTime - 0.1)) {
          console.log(`⏱️  Audio progress: ${audio.currentTime.toFixed(2)}s / ${audio.duration.toFixed(2)}s`);
        }
      };

      audio.onended = () => {
        console.log('✅ Audio playback ended naturally');

        // Cleanup
        if (currentAudioUrlRef.current) {
          URL.revokeObjectURL(currentAudioUrlRef.current);
          currentAudioUrlRef.current = null;
        }
        currentAudioRef.current = null;

        // Always call playNextAudio to handle queue state and backend notification
        setTimeout(() => playNextAudio(), 100);
      };

      audio.onerror = (err) => {
        console.error('❌ Audio playback error:', err);
        console.error('Audio error details:', {
          error: audio.error,
          networkState: audio.networkState,
          readyState: audio.readyState,
          currentTime: audio.currentTime,
          duration: audio.duration
        });

        // Cleanup
        if (currentAudioUrlRef.current) {
          URL.revokeObjectURL(currentAudioUrlRef.current);
          currentAudioUrlRef.current = null;
        }
        currentAudioRef.current = null;

        // Always call playNextAudio to handle queue state and backend notification
        setTimeout(() => playNextAudio(), 100);
      };

      // Play the audio with user interaction handling
      const playPromise = audio.play();

      if (playPromise !== undefined) {
        playPromise
          .then(() => {
            console.log('✅ Audio playing successfully');
          })
          .catch(err => {
            console.error('❌ Failed to play audio:', err);
            console.error('Error details:', {
              name: err.name,
              message: err.message
            });

            // Always call playNextAudio to handle queue state and backend notification
            setTimeout(() => playNextAudio(), 100);
          });
      }

    } catch (err) {
      console.error('❌ Error processing audio:', err);

      // Always call playNextAudio to handle queue state and backend notification
      setTimeout(() => playNextAudio(), 100);
    }
  }, [resumeRecording]);

  const playAudio = useCallback((audioData: string, format: string, sampleRate?: number) => {
    const wasEmpty = audioQueueRef.current.length === 0;
    const isCurrentlyPlaying = currentAudioRef.current !== null;

    // Add to queue
    audioQueueRef.current.push({ data: audioData, format, sampleRate });
    console.log(`📥 Added audio to queue (queue size: ${audioQueueRef.current.length})`);

    // Only start playing if queue was empty AND nothing is currently playing
    if (wasEmpty && !isCurrentlyPlaying) {
      console.log('🎙️ Starting TTS playback - notifying backend');
      pauseRecording();

      // Notify backend that TTS is starting
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'tts_started' }));
      }

      playNextAudio();
    } else {
      console.log('🎵 Audio already playing, packet will be queued');
    }
  }, [playNextAudio, pauseRecording]);

  const connect = useCallback((topic?: string) => {
    const ws = new WebSocket('ws://localhost:8001/ws/orchestrated-podcast');
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      setError(null);

      // Start the orchestrated podcast
      ws.send(JSON.stringify({
        type: 'start',
        topic: topic || 'Technology and Innovation',
        max_turns: 20
      }));

      startRecording();
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      console.log('📨 Received message:', data.type, data);

      if (data.type === 'thinking') {
        setIsThinking(true);
      } else if (data.type === 'agent_audio' || data.type === 'orchestrator_audio') {
        // Handle audio playback
        console.log('🎵 Received audio message:', data.type);
        if (data.audio_data && data.audio_format) {
          playAudio(data.audio_data, data.audio_format, data.sample_rate);
        } else {
          console.error('❌ Missing audio_data or audio_format:', data);
        }
        // Don't add audio messages to the messages array
      } else if (data.type === 'response_evaluation') {
        // Log evaluation but don't show in UI
        console.log('📊 Response evaluation:', data.evaluation);
      } else {
        if (data.type === 'agent_response' || data.type === 'orchestrator_question') {
          setIsThinking(false);
        }
        setMessages((prev) => [...prev, data]);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      stopRecording();
    };

    ws.onerror = () => {
      setError('WebSocket connection error');
    };
  }, [playAudio]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    // Stop any playing audio
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    // Clear queue
    audioQueueRef.current = [];
    // Cleanup URL
    if (currentAudioUrlRef.current) {
      URL.revokeObjectURL(currentAudioUrlRef.current);
      currentAudioUrlRef.current = null;
    }
    setIsPlayingAudio(false);
  }, []);

  const startRecording = async () => {
    try {
      // AWS Transcribe recommended audio constraints
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,              // Mono audio required
          sampleRate: 16000,            // 16kHz sample rate (AWS recommended)
          echoCancellation: true,       // Reduce echo
          noiseSuppression: true,       // Reduce background noise
          autoGainControl: true,        // Normalize volume levels
        }
      });
      streamRef.current = stream;

      // Force 16kHz sample rate
      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      sourceNodeRef.current = source;

      // AWS recommended: 1024 samples per chunk for optimal latency/quality balance
      // At 16kHz, this is 64ms of audio per chunk
      const processor = audioContext.createScriptProcessor(1024, 1, 1);
      processorRef.current = processor;
      isRecordingPausedRef.current = false;

      processor.onaudioprocess = (e) => {
        // Always send audio to keep Transcribe stream alive
        // The backend will handle ignoring transcripts during TTS playback
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          const inputData = e.inputBuffer.getChannelData(0);

          // Convert Float32Array to Int16Array (16-bit PCM, little-endian)
          // AWS Transcribe expects PCM16 format
          const pcmData = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            // Clamp to [-1, 1] range and convert to 16-bit integer
            const s = Math.max(-1, Math.min(1, inputData[i]));
            pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }

          // Send as ArrayBuffer (binary data)
          wsRef.current.send(pcmData.buffer);

          // Log every 500 chunks to verify audio is being sent (reduced logging)
          audioChunkCounterRef.current++;
          if (audioChunkCounterRef.current % 500 === 0) {
            console.log(`🎤 Sent ${audioChunkCounterRef.current} audio chunks to server`);
          }
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);
    } catch (err) {
      console.error('Error accessing microphone:', err);
      setError('Microphone access denied or error');
      disconnect();
    }
  };

  const stopRecording = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    isRecordingPausedRef.current = false;
  };

  useEffect(() => {
    return () => {
      disconnect();
      stopRecording();
    };
  }, [disconnect]);

  return { isConnected, messages, isThinking, isPlayingAudio, error, connect, disconnect };
};
