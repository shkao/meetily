"use client";

import { useEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import ReactMarkdown from 'react-markdown';
import { AlertCircle, ArrowUpRight, Check, Copy, History, ListChecks, Loader2, Lock, RotateCcw, Send, Sparkles, TextQuote, Users, X } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { useTranscripts } from '@/contexts/TranscriptContext';
import { toast } from 'sonner';

interface Message {
  id: string;
  question: string;
  /** Text streamed so far by local providers, replaced by `answer` when done */
  partial?: string;
  /** Progress note while the answer is prepared, such as identifying speakers */
  status?: string;
  answer?: string;
  error?: string;
}

const QUICK_PROMPTS: { text: string; icon: LucideIcon }[] = [
  { text: 'Summarize the discussion so far', icon: TextQuote },
  { text: 'What was discussed in the last two minutes?', icon: History },
  { text: 'What decisions and action items came up?', icon: ListChecks },
  { text: 'What has each speaker said?', icon: Users },
];

// A bracketed group holding one or more mm:ss times: "[03:15]", "[00:39–02:24]", or the malformed "[07:24–[09:16]"
const CITATION_GROUP = /\[([^\]\n]*?\d{1,3}:\d{2}[^\]\n]*?)\]/g;
const TIME = /(\d{1,3}):(\d{2})/g; // minutes can pass 99 in long meetings
const TIME_LINK_PREFIX = '#t-';

/** Turns cited times into markdown links that render as clickable time chips. */
function linkCitations(text: string): string {
  return text.replace(CITATION_GROUP, (_, inner: string) =>
    inner
      .replace(/\[/g, '')
      .replace(TIME, (t, mm, ss) => `[${t}](${TIME_LINK_PREFIX}${Number(mm) * 60 + Number(ss)})`)
  );
}

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

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
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const panelRef = useRef<HTMLElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
  }, [messages]);

  // Focus once on open; the recording page re-renders several times a second
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Escape closes the panel only when focus is inside it, so closing another dialog doesn't wipe the chat
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented) return;
      if (panelRef.current?.contains(document.activeElement)) onCloseRef.current();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const unlisteners: (() => void)[] = [];
    const keep = (u: () => void) => (cancelled ? u() : unlisteners.push(u));
    listen<{ request_id: string; text: string }>('ask-ai-token', (event) => {
      const { request_id, text } = event.payload;
      setMessages(prev => prev.map(m => (m.id === request_id ? { ...m, partial: (m.partial ?? '') + text } : m)));
    }).then(keep);
    listen<{ request_id: string; message: string }>('ask-ai-status', (event) => {
      const { request_id, message } = event.payload;
      setMessages(prev => prev.map(m => (m.id === request_id ? { ...m, status: message } : m)));
    }).then(keep);
    return () => {
      cancelled = true;
      unlisteners.forEach(u => u());
    };
  }, []);

  const ask = async (question: string) => {
    const q = question.trim();
    if (!q || busy) return;
    setInput('');
    setBusy(true);
    const id = crypto.randomUUID();
    setMessages(prev => [...prev, { id, question: q }]);
    const lines = transcriptsRef.current.map(t => ({
      start: t.audio_start_time ?? null,
      end: t.audio_end_time ?? null,
      text: t.text,
    }));
    try {
      const answer = await invoke<string>('ask_ai_live', { requestId: id, question: q, lines });
      setMessages(prev => prev.map(m => (m.id === id ? { ...m, answer } : m)));
    } catch (e) {
      setMessages(prev => prev.map(m => (m.id === id ? { ...m, error: String(e) } : m)));
    } finally {
      setBusy(false);
      // back to the question box, unless the user has moved on to something else meanwhile
      const active = document.activeElement;
      if (!active || active === document.body || panelRef.current?.contains(active)) inputRef.current?.focus();
    }
  };

  // Asks the live transcript to show the segment that contains the cited time
  const jumpTo = (seconds: number) => {
    const segments = transcriptsRef.current.filter(t => t.audio_start_time !== undefined);
    let target = segments[0];
    for (const t of segments) {
      // cited times are shown rounded down, like the transcript's [mm:ss]
      if (Math.floor(t.audio_start_time ?? 0) <= seconds) target = t;
    }
    if (target) window.dispatchEvent(new CustomEvent('transcript-jump', { detail: { id: target.id } }));
  };

  const copy = async (m: Message) => {
    if (!m.answer) return;
    try {
      await navigator.clipboard.writeText(m.answer);
    } catch {
      toast.error('Could not copy the answer');
      return;
    }
    setCopiedId(m.id);
    setTimeout(() => setCopiedId(id => (id === m.id ? null : id)), 1500);
  };

  const renderAnswer = (text: string) => (
    <ReactMarkdown
      components={{
        a: ({ href, children }) => {
          if (!href?.startsWith(TIME_LINK_PREFIX)) return <span>{children}</span>;
          const seconds = Number(href.slice(TIME_LINK_PREFIX.length));
          return (
            <button
              type="button"
              onClick={() => jumpTo(seconds)}
              aria-label={`Show ${children} in the transcript`}
              className="mx-0.5 inline-flex items-center rounded bg-blue-50 px-1.5 py-0.5 align-baseline text-xs font-medium tabular-nums text-blue-700 hover:bg-blue-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              {children}
            </button>
          );
        },
        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
        ul: ({ children }) => <ul className="mb-2 list-disc space-y-1 pl-4 last:mb-0">{children}</ul>,
        ol: ({ children }) => <ol className="mb-2 list-decimal space-y-1 pl-4 last:mb-0">{children}</ol>,
        strong: ({ children }) => <strong className="font-semibold text-gray-900">{children}</strong>,
      }}
    >
      {linkCitations(text)}
    </ReactMarkdown>
  );

  const suggestions = (exclude?: string) => (
    <section aria-labelledby="ask-ai-suggestions" className="space-y-2">
      <h3 id="ask-ai-suggestions" className="text-xs font-medium text-gray-500">
        {exclude ? 'Ask next' : 'Try asking'}
      </h3>
      <ul className="flex flex-col gap-2">
        {QUICK_PROMPTS.filter(p => p.text !== exclude).map(({ text, icon: Icon }) => (
          <li key={text}>
            <button
              type="button"
              disabled={busy}
              onClick={() => ask(text)}
              className="group flex min-h-11 w-full items-center gap-3 rounded-xl border border-gray-200 bg-gray-50 px-3 py-2 text-left text-sm text-gray-800 transition-[transform,background-color,border-color,box-shadow] duration-100 ease-out hover:border-gray-300 hover:bg-white hover:shadow-sm active:scale-[0.98] active:bg-gray-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:border-gray-200 disabled:hover:bg-gray-50 disabled:hover:shadow-none disabled:active:scale-100 motion-reduce:transition-colors motion-reduce:active:scale-100"
            >
              <span className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-600" aria-hidden="true">
                <Icon className="h-4 w-4" />
              </span>
              <span className="flex-1 leading-snug">{text}</span>
              <ArrowUpRight
                className="h-4 w-4 flex-shrink-0 text-gray-400 transition-colors group-hover:text-blue-600 group-disabled:group-hover:text-gray-400"
                aria-hidden="true"
              />
            </button>
          </li>
        ))}
      </ul>
    </section>
  );

  const last = messages[messages.length - 1];
  const showFollowUps = last && !busy && (last.answer !== undefined || last.error);

  return (
    <aside ref={panelRef} className="flex h-full w-[360px] flex-shrink-0 flex-col border-l border-gray-200 bg-white" aria-label="Ask AI">
      <header className="flex items-start justify-between border-b border-gray-200 px-4 py-3">
        <div>
          <h2 className="flex items-center gap-2 text-sm font-semibold text-gray-900">
            <Sparkles className="h-4 w-4 text-blue-600" aria-hidden="true" />
            Ask AI
          </h2>
          <p className="mt-0.5 flex items-center gap-1 text-xs text-gray-500">
            <Lock className="h-3 w-3" aria-hidden="true" />
            Only you can see this. Cleared when the recording ends.
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close Ask AI"
          className="-mr-1 rounded-md p-1.5 text-gray-500 hover:bg-gray-100 hover:text-gray-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      </header>

      <div className="flex-1 space-y-5 overflow-y-auto px-4 py-4 text-sm">
        {messages.length === 0 && (
          <div className="space-y-3">
            <p className="text-gray-600">
              Ask about the meeting so far. Answers cite transcript times you can click.
            </p>
            {suggestions()}
            <p className="text-xs text-gray-500">
              Questions about who said what take longer: speakers are identified from the audio and numbered, not named.
            </p>
          </div>
        )}

        {messages.map(m => {
          const pending = m.answer === undefined && !m.error;
          return (
            <article key={m.id} className="space-y-2">
              <div className="flex justify-end">
                <p className="max-w-[85%] rounded-2xl rounded-br-sm bg-gray-100 px-3 py-2 text-gray-900">{m.question}</p>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-1.5 text-xs font-medium text-gray-500">
                    <Sparkles className="h-3.5 w-3.5 text-blue-600" aria-hidden="true" />
                    AI answer
                  </span>
                  {m.answer && (
                    <button
                      type="button"
                      onClick={() => copy(m)}
                      aria-label={copiedId === m.id ? 'Copied' : 'Copy answer'}
                      className="rounded p-1 text-gray-500 hover:bg-gray-100 hover:text-gray-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                    >
                      {copiedId === m.id ? <Check className="h-3.5 w-3.5" aria-hidden="true" /> : <Copy className="h-3.5 w-3.5" aria-hidden="true" />}
                    </button>
                  )}
                </div>

                {(m.answer ?? m.partial) && (
                  <div className="leading-relaxed text-gray-800">{renderAnswer(m.answer ?? m.partial ?? '')}</div>
                )}

                {pending && (
                  <p role="status" className="flex items-center gap-2 text-xs text-gray-500">
                    <Loader2 className="h-3.5 w-3.5 animate-spin motion-reduce:animate-none" aria-hidden="true" />
                    {m.partial ? 'Writing...' : m.status ?? 'Reading the transcript...'}
                  </p>
                )}

                {m.answer !== undefined && m.status?.startsWith("Couldn't") && (
                  <p className="text-xs text-gray-500">{m.status}</p>
                )}

                {m.error && (
                  <div role="alert" className="flex items-start gap-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-red-800">
                    <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0" aria-hidden="true" />
                    <div className="space-y-1">
                      <p>{m.error}</p>
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() => ask(m.question)}
                        className="inline-flex items-center gap-1 text-xs font-medium text-red-700 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500 disabled:opacity-50"
                      >
                        <RotateCcw className="h-3 w-3" aria-hidden="true" />
                        Try again
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </article>
          );
        })}

        {showFollowUps && <div className="border-t border-gray-100 pt-4">{suggestions(last.question)}</div>}
        <div ref={endRef} />
      </div>

      <form
        className="flex items-center gap-2 border-t border-gray-200 p-3"
        onSubmit={e => {
          e.preventDefault();
          ask(input);
        }}
      >
        <label htmlFor="ask-ai-input" className="sr-only">Ask about this meeting</label>
        <input
          id="ask-ai-input"
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask about this meeting"
          autoComplete="off"
          className="h-10 flex-1 rounded-full border border-gray-300 px-4 text-sm text-gray-900 placeholder:text-gray-500 focus-visible:border-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500/30"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          aria-label="Ask"
          className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-600 text-white hover:bg-blue-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 disabled:bg-gray-200 disabled:text-gray-500"
        >
          {busy ? <Loader2 className="h-4 w-4 animate-spin motion-reduce:animate-none" aria-hidden="true" /> : <Send className="h-4 w-4" aria-hidden="true" />}
        </button>
      </form>
    </aside>
  );
}

