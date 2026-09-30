"use client";

import { useEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { Loader2, Send, Sparkles, X } from 'lucide-react';
import { useTranscripts } from '@/contexts/TranscriptContext';

interface Message {
  id: string;
  question: string;
  /** Text streamed so far by local providers, replaced by `answer` when done */
  partial?: string;
  answer?: string;
  error?: string;
}

const QUICK_PROMPTS = [
  'Catch me up',
  'Summarize the discussion so far',
  'What are the key decisions and action items so far?',
];

const TIME_PATTERN = /\[(\d{1,2}):(\d{2})\]/g;

interface AskAiPanelProps {
  onClose: () => void;
}

/**
 * Private Ask AI side panel for the meeting in progress, like Meet's "Ask Gemini".
 * Answers come from the live transcript and the summary provider; history lives only while the panel is mounted.
 */
export function AskAiPanel({ onClose }: AskAiPanelProps) {
  const { transcriptsRef } = useTranscripts();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    listen<{ request_id: string; text: string }>('ask-ai-token', (event) => {
      const { request_id, text } = event.payload;
      setMessages(prev => prev.map(m => (m.id === request_id ? { ...m, partial: (m.partial ?? '') + text } : m)));
    }).then((u) => {
      if (cancelled) u();
      else unlisten = u;
    });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  const ask = async (question: string) => {
    const q = question.trim();
    if (!q || busy) return;
    setInput('');
    setBusy(true);
    const id = crypto.randomUUID();
    setMessages(prev => [...prev, { id, question: q }]);
    const lines = transcriptsRef.current.map(t => ({ start: t.audio_start_time ?? null, text: t.text }));
    try {
      const answer = await invoke<string>('ask_ai_live', { requestId: id, question: q, lines });
      setMessages(prev => prev.map(m => (m.id === id ? { ...m, answer } : m)));
    } catch (e) {
      setMessages(prev => prev.map(m => (m.id === id ? { ...m, error: String(e) } : m)));
    } finally {
      setBusy(false);
    }
  };

  // Scrolls the live transcript to the segment that contains the cited time
  const jumpTo = (seconds: number) => {
    const segments = transcriptsRef.current.filter(t => t.audio_start_time !== undefined);
    let target = segments[0];
    for (const t of segments) {
      if ((t.audio_start_time ?? 0) <= seconds) target = t;
    }
    if (!target) return;
    const el = document.getElementById(`segment-${target.id}`);
    el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    el?.classList.add('bg-yellow-100');
    setTimeout(() => el?.classList.remove('bg-yellow-100'), 1500);
  };

  const renderAnswer = (text: string) => {
    const parts: React.ReactNode[] = [];
    let last = 0;
    for (const match of text.matchAll(TIME_PATTERN)) {
      const index = match.index ?? 0;
      parts.push(text.slice(last, index));
      const seconds = Number(match[1]) * 60 + Number(match[2]);
      parts.push(
        <button
          key={`${index}-${match[0]}`}
          type="button"
          onClick={() => jumpTo(seconds)}
          className="text-blue-600 hover:underline font-mono text-xs"
          title="Show in transcript"
        >
          {match[0]}
        </button>
      );
      last = index + match[0].length;
    }
    parts.push(text.slice(last));
    return parts;
  };

  return (
    <aside className="w-[360px] flex-shrink-0 h-full border-l border-gray-200 bg-white flex flex-col">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
        <div className="flex items-center gap-2 text-sm font-semibold text-gray-800">
          <Sparkles className="w-4 h-4 text-blue-600" />
          Ask AI
        </div>
        <button type="button" onClick={onClose} className="text-gray-400 hover:text-gray-600" title="Close">
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4 text-sm">
        {messages.length === 0 && (
          <p className="text-gray-500">
            Ask about the meeting so far. Only you see the answers, and they're cleared when the recording ends.
            Speaker names aren't available until after the meeting.
          </p>
        )}
        {messages.map(m => (
          <div key={m.id} className="space-y-1">
            <div className="font-medium text-gray-800">{m.question}</div>
            {m.answer !== undefined && (
              <div className="text-gray-700 whitespace-pre-wrap leading-relaxed">{renderAnswer(m.answer)}</div>
            )}
            {m.error && <div className="text-red-600">{m.error}</div>}
            {m.answer === undefined && !m.error && m.partial && (
              <div className="text-gray-700 whitespace-pre-wrap leading-relaxed">{renderAnswer(m.partial)}</div>
            )}
            {m.answer === undefined && !m.error && (
              <div className="flex items-center gap-2 text-gray-400">
                <Loader2 className="w-3 h-3 animate-spin" /> {m.partial ? 'Writing...' : 'Thinking...'}
              </div>
            )}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <div className="border-t border-gray-200 p-3 space-y-2">
        <div className="flex flex-wrap gap-1">
          {QUICK_PROMPTS.map(p => (
            <button
              key={p}
              type="button"
              disabled={busy}
              onClick={() => ask(p)}
              className="text-xs px-2 py-1 rounded-full border border-gray-200 text-gray-700 hover:bg-gray-50 disabled:opacity-50"
            >
              {p}
            </button>
          ))}
        </div>
        <form
          className="flex items-center gap-2"
          onSubmit={e => {
            e.preventDefault();
            ask(input);
          }}
        >
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask about this meeting"
            className="flex-1 px-3 py-2 border border-gray-200 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
          <button type="submit" disabled={busy || !input.trim()} className="text-blue-600 disabled:text-gray-300" title="Ask">
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </aside>
  );
}
