"use client";

import { useEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { Loader2, Users } from 'lucide-react';
import { toast } from 'sonner';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog';
import { Button } from '../ui/button';
import { Input } from '../ui/input';

interface DiarizationProgress {
  meeting_id: string;
  stage: 'downloading' | 'diarizing' | 'done' | 'skipped' | 'failed';
  percent: number | null;
  message: string;
}

interface SpeakerStatusProps {
  meetingId: string;
  hasSpeakers: boolean;
  onRefetchTranscripts?: () => Promise<void>;
}

/** Shows post-meeting speaker identification progress, and offers it for meetings without labels. */
export function SpeakerStatus({ meetingId, hasSpeakers, onRefetchTranscripts }: SpeakerStatusProps) {
  const [progress, setProgress] = useState<DiarizationProgress | null>(null);
  const refetchRef = useRef(onRefetchTranscripts);
  refetchRef.current = onRefetchTranscripts;

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    listen<DiarizationProgress>('diarization-progress', async (event) => {
      if (event.payload.meeting_id !== meetingId) return;
      setProgress(event.payload);
      if (event.payload.stage === 'done') {
        await refetchRef.current?.();
      }
    }).then((u) => {
      if (cancelled) u();
      else unlisten = u;
    });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [meetingId]);

  const running = progress?.stage === 'downloading' || progress?.stage === 'diarizing';

  const identify = async () => {
    setProgress({ meeting_id: meetingId, stage: 'diarizing', percent: null, message: 'Identifying speakers' });
    try {
      await invoke<boolean>('diarization_run', { meetingId });
    } catch {
      // the failed event carries the message
    }
  };

  if (running) {
    return (
      <div className="flex items-center gap-2 text-xs text-gray-500 mt-2">
        <Loader2 className="w-3 h-3 animate-spin" />
        <span>
          {progress.message}
          {progress.stage === 'downloading' && progress.percent !== null ? ` ${progress.percent}%` : '...'}
        </span>
      </div>
    );
  }
  if (progress?.stage === 'failed' || progress?.stage === 'skipped') {
    return (
      <div className="flex items-center gap-2 text-xs text-gray-500 mt-2">
        <span>{progress.message}</span>
        {progress.stage === 'failed' && (
          <button type="button" className="text-blue-600 hover:underline" onClick={identify}>Retry</button>
        )}
      </div>
    );
  }
  if (!hasSpeakers) {
    return (
      <div className="flex items-center gap-2 text-xs mt-2">
        <button type="button" className="flex items-center gap-1 text-blue-600 hover:underline" onClick={identify}>
          <Users className="w-3 h-3" />
          Identify speakers
        </button>
      </div>
    );
  }
  return (
    <div className="text-xs text-gray-400 mt-2">Click a speaker name to rename it.</div>
  );
}

interface RenameSpeakerDialogProps {
  meetingId: string;
  speaker: string | null;
  onClose: () => void;
  onRenamed?: () => Promise<void>;
}

/** Renames one speaker across the meeting's transcript. */
export function RenameSpeakerDialog({ meetingId, speaker, onClose, onRenamed }: RenameSpeakerDialogProps) {
  const [name, setName] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setName(speaker ?? '');
  }, [speaker]);

  const save = async () => {
    if (!speaker || !name.trim() || name.trim() === speaker) {
      onClose();
      return;
    }
    setSaving(true);
    try {
      await invoke<number>('diarization_rename_speaker', { meetingId, from: speaker, to: name.trim() });
      await onRenamed?.();
      onClose();
    } catch (e) {
      toast.error(`Could not rename speaker: ${e}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={speaker !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>Rename speaker</DialogTitle>
          <DialogDescription>Every segment labelled &quot;{speaker}&quot; in this meeting gets the new name.</DialogDescription>
        </DialogHeader>
        <Input
          autoFocus
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && save()}
          placeholder="Name"
        />
        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={saving}>Cancel</Button>
          <Button onClick={save} disabled={saving || !name.trim()}>Rename</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
