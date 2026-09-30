---
name: verify
description: Build, launch and drive the Meetily Tauri desktop app on macOS to observe a change at runtime (recording, saving, recovery, transcripts). Use when verifying a change to the Rust core or UI in the running app.
---

# Verify Meetily in the running app (macOS)

## Launch

```sh
cd frontend && RUST_LOG=info pnpm run tauri:dev > /tmp/tauri-dev.log 2>&1 &   # not clean_run.sh: it deletes node_modules
```

- The first build takes about 10 minutes. It's ready when the log shows `Successfully got N meetings`.
- `tauri dev` watches the Rust source, so `git switch` to another branch rebuilds and relaunches the app. Use that for before/after runs.
- It uses the real app data: the database and `~/Movies/meetily-recordings`. Test recordings become real meetings, so name them or list them for the user afterwards.
- Stop it with `kill` on `node scripts/tauri-auto.js dev`, `next dev -p 3118` and `target/debug/meetily`.

## Drive

- Window position and size: `osascript -e 'tell application "System Events" to tell process "meetily" to get {position, size} of window 1'`.
- Screenshot the window only: `screencapture -x -R<x>,<y>,<w>,<h> out.png`.
- Click with `cliclick c:X,Y` in screen coordinates (window origin plus offset). With the default 1100×700 window at (410,118):
  - Record, home screen: (994,738)
  - Stop, while recording: (1017,741)
  - Record, sidebar, used after Stop because the app moves to the meeting page: (441,300)
- The webview exposes no accessibility buttons, so clicking by position is the only handle.
- **A terminal overlay window (iTerm2 hotkey window) can sit above the app and swallow clicks**, even when meetily is frontmost. Hide it first: `osascript -e 'tell application "System Events" to set visible of process "iTerm2" to false'`. Restore it afterwards.
- Give the app audio through system capture with `afplay <wav>`, which it records and transcribes live.

## Observe

- Logs: `grep -E "incremental_saver|recording_saver|CALLED start_recording" /tmp/tauri-dev.log`.
- A saved meeting is `~/Movies/meetily-recordings/<Meeting ...>/` with `audio.mp4`, `metadata.json`, `transcripts.json`, and `.checkpoints/` while recording.
- Decoded length: `ffmpeg -v error -i audio.mp4 -f f32le -ac 1 -ar 48000 pipe:1 | wc -c` (divide by 192000 for seconds).
- Timing: play a known clip or wall-clock-scheduled beeps during recording, then cross-correlate against `audio.mp4`.
- Crash recovery: `kill -9` the `target/debug/meetily` process while recording, then relaunch. A "Recover Interrupted Meetings" dialog appears.

## Gotchas

- `ls`, `cp` and `rm` are aliased (eza, `-i`). In scripts, use `command ls`, `command cp -f` and `command rm -f`.
- Recording timelines run 0.6 to 0.9% fast against wall time on `main` (issue #5), so don't mistake that for a regression.
