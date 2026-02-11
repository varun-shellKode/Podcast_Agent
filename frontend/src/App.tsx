import React, { useEffect, useRef, useState } from 'react';
import { Mic, MicOff, Podcast, Zap, MessageCircle, AlertCircle } from 'lucide-react';
import './App.css';
import { useAudioStream } from './hooks/useAudioStream';

const App: React.FC = () => {
  const { isConnected, messages, isThinking, isPlayingAudio, error, connect, disconnect } = useAudioStream();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [topic, setTopic] = useState('AWS Serverless Architecture Best Practices');

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleStartPodcast = () => {
    if (topic.trim()) {
      connect(topic);
    }
  };

  return (
    <div className="app-container">
      <header>
        <div className="logo">
          PodcastAI <span style={{ color: 'var(--accent-color)' }}>Orchestrator</span>
        </div>
        <div className="controls">
          {!isConnected ? (
            <>
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="Enter podcast topic..."
                style={{
                  padding: '0.5rem 1rem',
                  borderRadius: '8px',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  background: 'rgba(255, 255, 255, 0.05)',
                  color: 'white',
                  marginRight: '1rem',
                  minWidth: '300px'
                }}
              />
              <button className="btn btn-primary" onClick={handleStartPodcast}>
                <Mic size={20} />
                Start Podcast
              </button>
            </>
          ) : (
            <button className="btn btn-danger" onClick={disconnect}>
              <MicOff size={20} />
              End Session
            </button>
          )}
        </div>
      </header>

      {error && (
        <div style={{ color: '#ff4d4d', textAlign: 'center', padding: '1rem', background: 'rgba(255, 77, 77, 0.1)', borderRadius: '12px' }}>
          {error}
        </div>
      )}

      <main className="main-content">
        <section className="transcript-panel">
          <div className="panel-header">
            <Podcast size={16} />
            Live Transcript
          </div>
          <div className="scroll-area" ref={scrollRef}>
            {messages.length === 0 && !isConnected && (
              <div style={{ color: 'var(--text-dim)', textAlign: 'center', marginTop: '40px' }}>
                Press "Start Podcast" and speak into your mic.
              </div>
            )}
            {messages.map((msg, idx) => {
              const isOrchestrator = msg.type?.startsWith('orchestrator_');
              const isTranscript = msg.type === 'transcript';
              const isIntervention = ['orchestrator_redirect', 'orchestrator_interrupt', 'orchestrator_topic_switch'].includes(msg.type || '');

              return (
                <div key={idx} className={`utterance ${isOrchestrator ? 'orchestrator' : ''} ${isIntervention ? 'intervention' : ''}`}>
                  <div className="speaker-label">
                    {isOrchestrator && <MessageCircle size={14} />}
                    {isIntervention && <AlertCircle size={14} />}
                    {msg.type === 'agent_response' && <Zap size={14} />}
                    {isOrchestrator ? 'Orchestrator' : msg.speaker_name || 'System'}
                    {msg.speaker_role && <span style={{opacity: 0.6, marginLeft: '0.5rem'}}>({msg.speaker_role})</span>}
                  </div>
                  <div className="text-content">
                    {msg.type === 'orchestrator_intro' && msg.text}
                    {msg.type === 'orchestrator_question' && (
                      <>
                        <strong>To {msg.speaker_name}:</strong> {msg.question}
                        {msg.turn_number && <span style={{opacity: 0.5, marginLeft: '0.5rem'}}>Turn {msg.turn_number}</span>}
                      </>
                    )}
                    {msg.type === 'orchestrator_redirect' && (
                      <span style={{color: '#ffa500'}}>🔄 {msg.message}</span>
                    )}
                    {msg.type === 'orchestrator_interrupt' && (
                      <span style={{color: '#ff6b6b'}}>✋ {msg.message}</span>
                    )}
                    {msg.type === 'orchestrator_topic_switch' && (
                      <span style={{color: '#ff6b6b'}}>🔀 {msg.message}</span>
                    )}
                    {msg.type === 'orchestrator_wrapup' && (
                      <div>
                        <div>{msg.text}</div>
                        {msg.total_turns && (
                          <div style={{marginTop: '0.5rem', opacity: 0.6, fontSize: '0.9rem'}}>
                            Total turns: {msg.total_turns}
                          </div>
                        )}
                      </div>
                    )}
                    {msg.type === 'transcript' && msg.text}
                    {msg.type === 'agent_response' && msg.response}
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="agent-panel">
          <div className={`agent-avatar ${isThinking || isPlayingAudio ? 'pulse' : ''}`}>
            <Podcast size={48} color="white" />
          </div>
          <div className="agent-status">
            {isPlayingAudio ? (
              <>
                🔊 Orchestrator is speaking
                <div className="thinking-indicator">
                  <div className="dot"></div>
                  <div className="dot"></div>
                  <div className="dot"></div>
                </div>
              </>
            ) : isThinking ? (
              <>
                Orchestrator is thinking
                <div className="thinking-indicator">
                  <div className="dot"></div>
                  <div className="dot"></div>
                  <div className="dot"></div>
                </div>
              </>
            ) : isConnected ? (
              'Listening & Managing...'
            ) : (
              'Off-air'
            )}
          </div>
          <div style={{ marginTop: '20px', color: 'var(--text-dim)', fontSize: '0.9rem' }}>
            <strong>5 Speakers:</strong>
            <div style={{ marginTop: '0.5rem', lineHeight: '1.6' }}>
              • Host<br />
              • AWS Expert<br />
              • ShellKode Developer<br />
              • Tech Generalist<br />
              • Client
            </div>
            <div style={{ marginTop: '1rem', fontSize: '0.85rem', opacity: 0.7 }}>
              Speakers will introduce themselves
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};

export default App;
