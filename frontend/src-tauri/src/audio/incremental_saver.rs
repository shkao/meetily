use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use anyhow::{Result, anyhow};
use log::{info, warn, error};
use super::recording_state::AudioChunk;
use serde::{Serialize, Deserialize};

use super::ffmpeg::find_ffmpeg_path;

/// Seconds of audio per crash-recovery checkpoint segment
const CHECKPOINT_SECONDS: u32 = 30;
/// Final audio file, written inside .checkpoints/ while recording and moved into the meeting folder on finalize
const PARTIAL_AUDIO_FILE: &str = "audio.partial.mp4";

/// Incremental audio saver backed by one continuous AAC encoder.
///
/// A single ffmpeg process encodes the whole recording and writes two outputs through the tee muxer:
/// the final audio.mp4, and 30-second ADTS checkpoint segments (.checkpoints/audio_chunk_NNNNN.aac) for
/// crash recovery. Encoding each checkpoint separately and joining them with `concat -c copy` added
/// 1024 samples of encoder priming plus final-frame padding per checkpoint (37 ms every 30 s), so saved
/// audio drifted against capture time. One continuous stream has a single priming, which the mp4 muxer
/// records in an edit list, so audio.mp4 matches the captured samples exactly.
pub struct IncrementalAudioSaver {
    encoder: Option<Child>,
    encoder_stdin: Option<ChildStdin>,
    /// Reads the encoder's stderr while it runs, so a full pipe can never stall it; yields the output at the end
    encoder_stderr: Option<std::thread::JoinHandle<Vec<u8>>>,
    /// First encoder failure; later chunks are dropped and finalize reports it
    encoder_error: Option<String>,
    samples_written: u64,
    checkpoint_count: u32,
    checkpoints_dir: PathBuf,
    meeting_folder: PathBuf,
    sample_rate: u32,
}

impl IncrementalAudioSaver {
    /// Create a new incremental saver. The encoder starts with the first audio chunk.
    ///
    /// # Arguments
    /// * `meeting_folder` - Path to the meeting folder (contains .checkpoints/)
    /// * `sample_rate` - Sample rate of audio (typically 48000)
    pub fn new(meeting_folder: PathBuf, sample_rate: u32) -> Result<Self> {
        let checkpoints_dir = meeting_folder.join(".checkpoints");

        // Verify checkpoints directory exists
        if !checkpoints_dir.exists() {
            return Err(anyhow!("Checkpoints directory does not exist: {}", checkpoints_dir.display()));
        }

        Ok(Self {
            encoder: None,
            encoder_stdin: None,
            encoder_stderr: None,
            encoder_error: None,
            samples_written: 0,
            checkpoint_count: 0,
            checkpoints_dir,
            meeting_folder,
            sample_rate,
        })
    }

    /// Start the continuous encoder: f32 mono PCM on stdin, AAC-LC 192k out to audio.partial.mp4 and to
    /// 30-second ADTS checkpoint segments.
    fn start_encoder(&mut self) -> Result<()> {
        let ffmpeg_path = find_ffmpeg_path()
            .ok_or_else(|| anyhow!("FFmpeg not found. Please install FFmpeg to save recordings."))?;
        info!("Starting continuous audio encoder");

        let segments = self.checkpoints_dir.join("audio_chunk_%05d.aac");
        let partial = self.checkpoints_dir.join(PARTIAL_AUDIO_FILE);
        // tee escapes: '|' separates outputs, so the paths are passed as-is and must not contain '|'
        let tee_outputs = format!(
            "[f=segment:segment_time={}:segment_format=adts]{}|[f=mp4:movflags=+faststart]{}",
            CHECKPOINT_SECONDS,
            segments.display(),
            partial.display()
        );

        let mut command = Command::new(ffmpeg_path);
        command
            .args([
                // errors only: ffmpeg's progress lines would otherwise stream to stderr for the whole meeting
                "-hide_banner", "-nostats", "-loglevel", "error",
                "-f", "f32le",
                "-ar", &self.sample_rate.to_string(),
                "-ac", "1",
                "-i", "pipe:0",
                "-map", "0:a",
                "-c:a", "aac",
                "-b:a", "192k",
                "-profile:a", "aac_low",
                "-f", "tee",
                "-y",
                &tee_outputs,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        // Hide console window on Windows to prevent CMD popup during recording
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            command.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = command.spawn()?;
        self.encoder_stdin = child.stdin.take();
        self.encoder_stderr = child.stderr.take().map(drain_stderr);
        self.encoder = Some(child);
        Ok(())
    }

    /// Add an audio chunk: stream it to the encoder.
    /// Counts one checkpoint for every 30 seconds of audio written.
    pub fn add_chunk(&mut self, chunk: AudioChunk) -> Result<()> {
        if self.encoder_error.is_some() || chunk.data.is_empty() {
            return Ok(());
        }
        if self.encoder.is_none() {
            if let Err(e) = self.start_encoder() {
                self.encoder_error = Some(e.to_string());
                return Err(e);
            }
        }

        let stdin = self.encoder_stdin.as_mut().ok_or_else(|| anyhow!("Encoder input closed"))?;
        if let Err(e) = stdin.write_all(bytemuck::cast_slice(&chunk.data)) {
            let message = format!("Audio encoder stopped accepting data: {}", e);
            self.encoder_error = Some(message.clone());
            return Err(anyhow!(message));
        }

        let interval = self.sample_rate as u64 * CHECKPOINT_SECONDS as u64;
        let before = self.samples_written / interval;
        self.samples_written += chunk.data.len() as u64;
        let after = self.samples_written / interval;
        if after > before {
            self.checkpoint_count = after as u32;
            info!("Checkpoint {}: {:.2}s of audio encoded",
                  self.checkpoint_count,
                  self.samples_written as f64 / self.sample_rate as f64);
        }

        Ok(())
    }

    /// Finalize the recording: close the encoder, move audio.mp4 into place, clean up checkpoints
    ///
    /// Returns the path to the final audio.mp4 file
    pub async fn finalize(&mut self) -> Result<PathBuf> {
        info!("Finalizing incremental recording...");

        if self.samples_written == 0 {
            return Err(anyhow!("No audio checkpoints to merge - recording may have failed"));
        }

        // Closing stdin ends the stream; ffmpeg then flushes the encoder and writes the mp4 index
        drop(self.encoder_stdin.take());
        let mut encoder = self.encoder.take().ok_or_else(|| anyhow!("Audio encoder was not running"))?;
        let status = encoder.wait()?;
        let stderr_bytes = self.encoder_stderr.take().and_then(|h| h.join().ok()).unwrap_or_default();
        if let Some(e) = &self.encoder_error {
            return Err(anyhow!("Audio encoding failed during recording: {}", e));
        }
        if !status.success() {
            let stderr = String::from_utf8_lossy(&stderr_bytes);
            error!("FFmpeg encoder failed");
            return Err(anyhow!("FFmpeg encoder failed: {}", stderr));
        }

        let partial = self.checkpoints_dir.join(PARTIAL_AUDIO_FILE);
        let final_audio_path = self.meeting_folder.join("audio.mp4");
        std::fs::rename(&partial, &final_audio_path)
            .map_err(|e| anyhow!("Failed to move encoded audio into place: {}", e))?;

        // Clean up checkpoints directory
        info!("Cleaning up checkpoint segments");
        if let Err(_) = std::fs::remove_dir_all(&self.checkpoints_dir) {
            warn!("Failed to clean up checkpoints");
            // Non-fatal - user can manually delete
        }

        info!("Finalized recording ({:.2}s)", self.samples_written as f64 / self.sample_rate as f64);

        Ok(final_audio_path)
    }

    /// Get the meeting folder path
    pub fn get_meeting_folder(&self) -> &PathBuf {
        &self.meeting_folder
    }

    /// Get current checkpoint count
    pub fn get_checkpoint_count(&self) -> u32 {
        self.checkpoint_count
    }
}

/// True for crash-recovery checkpoint files: ADTS segments from the continuous encoder, or per-checkpoint
/// mp4 files written by earlier versions. The in-progress audio.partial.mp4 is not a checkpoint.
fn is_checkpoint_file(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    name.starts_with("audio_chunk_") && (name.ends_with(".aac") || name.ends_with(".mp4"))
}

/// Audio recovery status for transcript recovery feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioRecoveryStatus {
    pub status: String, // "success" | "partial" | "failed" | "none"
    pub chunk_count: u32,
    pub estimated_duration_seconds: f64,
    pub audio_file_path: Option<String>,
    pub message: String,
}

/// Recover audio from checkpoint files
/// This is called by the transcript recovery system to merge audio chunks after a crash
#[tauri::command]
pub async fn recover_audio_from_checkpoints(
    meeting_folder: String,
    _sample_rate: u32
) -> Result<AudioRecoveryStatus, String> {
    info!("Starting audio recovery");

    let folder_path = PathBuf::from(&meeting_folder);
    let checkpoints_dir = folder_path.join(".checkpoints");

    // Check if checkpoints directory exists
    if !checkpoints_dir.exists() {
        info!("No checkpoints directory found");
        return Ok(AudioRecoveryStatus {
            status: "none".to_string(),
            chunk_count: 0,
            estimated_duration_seconds: 0.0,
            audio_file_path: None,
            message: "No audio checkpoints found".to_string(),
        });
    }

    // Scan for checkpoint files
    let mut checkpoint_files: Vec<_> = std::fs::read_dir(&checkpoints_dir)
        .map_err(|e| format!("Failed to read checkpoints directory: {}", e))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| is_checkpoint_file(&entry.path()))
        .collect();

    if checkpoint_files.is_empty() {
        info!("No checkpoint files found");
        return Ok(AudioRecoveryStatus {
            status: "none".to_string(),
            chunk_count: 0,
            estimated_duration_seconds: 0.0,
            audio_file_path: None,
            message: "No audio checkpoint files found".to_string(),
        });
    }

    // Sort by filename (audio_chunk_00000.aac, audio_chunk_00001.aac, etc.)
    checkpoint_files.sort_by_key(|entry| entry.path());

    let chunk_count = checkpoint_files.len() as u32;
    let estimated_duration = (chunk_count as f64) * 30.0; // 30 seconds per chunk

    info!("Found {} checkpoint files, estimated duration: {:.2}s", chunk_count, estimated_duration);

    let output_path = folder_path.join("audio.mp4");
    let output_path_str = output_path.to_str()
        .ok_or("Invalid output path")?
        .to_string();

    let ffmpeg_path = find_ffmpeg_path()
        .ok_or_else(|| "FFmpeg not found. Please install FFmpeg to recover audio.".to_string())?;
    info!("FFmpeg selected for recovery");

    let mut command = std::process::Command::new(ffmpeg_path);

    // ADTS segments are slices of one continuous AAC stream, so joining their bytes restores that stream
    // with no gaps; remuxing it keeps the audio aligned with capture time (only the stream's single
    // 21 ms encoder priming remains at the start).
    let adts = checkpoint_files.iter().all(|e| e.path().extension().and_then(|s| s.to_str()) == Some("aac"));
    let concat_file_path = if adts {
        let joined = checkpoints_dir.join("recovered_stream.aac");
        let mut out = std::fs::File::create(&joined)
            .map_err(|e| format!("Failed to create joined stream: {}", e))?;
        for entry in &checkpoint_files {
            let mut segment = std::fs::File::open(entry.path())
                .map_err(|e| format!("Failed to open checkpoint: {}", e))?;
            std::io::copy(&mut segment, &mut out)
                .map_err(|e| format!("Failed to join checkpoints: {}", e))?;
        }
        command.args(&[
            "-i", joined.to_str().unwrap(),
            "-c", "copy",
            "-y", // Overwrite if exists
            &output_path_str
        ]);
        joined
    } else {
        // Per-checkpoint mp4 files from earlier versions: each carries its own encoder priming
        let list = checkpoints_dir.join("concat_list.txt");
        let mut concat_content = String::new();
        for entry in &checkpoint_files {
            let path = entry.path().canonicalize()
                .map_err(|e| format!("Failed to canonicalize path: {}", e))?;
            concat_content.push_str(&format!("file '{}'\n", path.display()));
        }
        std::fs::write(&list, concat_content)
            .map_err(|e| format!("Failed to write concat file: {}", e))?;
        command.args(&[
            "-f", "concat",
            "-safe", "0",
            "-i", list.to_str().unwrap(),
            "-c", "copy",
            "-y", // Overwrite if exists
            &output_path_str
        ]);
        list
    };

    // Hide console window on Windows
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    let ffmpeg_result = command.output();

    match ffmpeg_result {
        Ok(output) if output.status.success() => {
            // Clean up concat file
            let _ = std::fs::remove_file(concat_file_path);

            info!("Successfully recovered audio from {} checkpoints", chunk_count);

            Ok(AudioRecoveryStatus {
                status: "success".to_string(),
                chunk_count,
                estimated_duration_seconds: estimated_duration,
                audio_file_path: Some(output_path_str),
                message: format!("Successfully recovered {} audio chunks", chunk_count),
            })
        }
        Ok(output) => {
            let error = String::from_utf8_lossy(&output.stderr);
            error!("FFmpeg recovery failed");
            Ok(AudioRecoveryStatus {
                status: "failed".to_string(),
                chunk_count,
                estimated_duration_seconds: estimated_duration,
                audio_file_path: None,
                message: format!("FFmpeg failed: {}", error),
            })
        }
        Err(e) => {
            error!("Failed to run FFmpeg");
            Ok(AudioRecoveryStatus {
                status: "failed".to_string(),
                chunk_count,
                estimated_duration_seconds: estimated_duration,
                audio_file_path: None,
                message: format!("Failed to run FFmpeg: {}", e),
            })
        }
    }
}

/// Clean up checkpoint files after successful recording or recovery
/// This command is called by the frontend after successful save to clean up checkpoint files
#[tauri::command]
pub async fn cleanup_checkpoints(meeting_folder: String) -> Result<(), String> {
    info!("Cleaning up checkpoints");

    let folder_path = PathBuf::from(&meeting_folder);
    let checkpoints_dir = folder_path.join(".checkpoints");

    if checkpoints_dir.exists() {
        std::fs::remove_dir_all(&checkpoints_dir)
            .map_err(|e| format!("Failed to remove checkpoints directory: {}", e))?;
        info!("Successfully cleaned up checkpoints directory");
    } else {
        info!("No checkpoints directory to clean up");
    }

    Ok(())
}

/// Check if a meeting folder has audio checkpoint files
/// Returns true if .checkpoints/ directory exists and contains checkpoint segments
#[tauri::command]
pub async fn has_audio_checkpoints(meeting_folder: String) -> Result<bool, String> {
    let folder_path = PathBuf::from(&meeting_folder);
    let checkpoints_dir = folder_path.join(".checkpoints");

    // Check if checkpoints directory exists
    if !checkpoints_dir.exists() {
        return Ok(false);
    }

    // Scan for checkpoint segments
    let has_mp4_files = std::fs::read_dir(&checkpoints_dir)
        .map_err(|e| format!("Failed to read checkpoints directory: {}", e))?
        .filter_map(|entry| entry.ok())
        .any(|entry| is_checkpoint_file(&entry.path()));

    Ok(has_mp4_files)
}

/// Reads a child's stderr on a thread until it closes, keeping the last 64 KB for error messages. Without it, a
/// child that writes more than the pipe buffer blocks on stderr and stops reading its input.
fn drain_stderr(mut stderr: std::process::ChildStderr) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        use std::io::Read;
        const KEEP: usize = 64 * 1024;
        let mut kept = Vec::new();
        let mut buf = [0u8; 8192];
        while let Ok(n) = stderr.read(&mut buf) {
            if n == 0 {
                break;
            }
            kept.extend_from_slice(&buf[..n]);
            if kept.len() > KEEP {
                kept.drain(..kept.len() - KEEP);
            }
        }
        kept
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use super::super::recording_state::DeviceType;

    #[tokio::test]
    async fn test_checkpoint_creation() {
        // Create temp meeting folder
        let temp_dir = tempdir().unwrap();
        let meeting_folder = temp_dir.path().join("Test_Meeting");
        std::fs::create_dir_all(&meeting_folder).unwrap();
        std::fs::create_dir_all(meeting_folder.join(".checkpoints")).unwrap();

        let mut saver = IncrementalAudioSaver::new(
            meeting_folder.clone(),
            48000
        ).unwrap();

        // Add 60 seconds worth of audio (should create 2 checkpoints)
        for i in 0..120 {  // 120 chunks of 0.5s each
            let chunk = AudioChunk {
                data: vec![0.5f32; 24000],  // 0.5s at 48kHz
                sample_rate: 48000,
                timestamp: i as f64 * 0.5,  // timestamp in seconds
                chunk_id: i as u64,
                device_type: DeviceType::Microphone,
            };
            saver.add_chunk(chunk).unwrap();
        }

        // Verify 2 checkpoints created
        assert_eq!(saver.checkpoint_count, 2);

        // Finalize and verify merge
        let final_path = saver.finalize().await.unwrap();
        assert!(final_path.exists());

        // Verify checkpoints directory deleted
        assert!(!meeting_folder.join(".checkpoints").exists());
    }

    /// Samples ffmpeg decodes from a file at the given rate (mono f32).
    fn decoded_samples(path: &Path, sample_rate: u32) -> usize {
        let output = std::process::Command::new(find_ffmpeg_path().expect("ffmpeg required"))
            .args(["-v", "error", "-i", path.to_str().unwrap(), "-f", "f32le", "-ac", "1",
                   "-ar", &sample_rate.to_string(), "pipe:1"])
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        output.stdout.len() / 4
    }

    fn tone_chunk(i: usize, samples: usize) -> AudioChunk {
        let data = (0..samples)
            .map(|n| (((i * samples + n) as f32) * 440.0 * std::f32::consts::TAU / 48000.0).sin() * 0.3)
            .collect();
        AudioChunk {
            data,
            sample_rate: 48000,
            timestamp: i as f64 * samples as f64 / 48000.0,
            chunk_id: i as u64,
            device_type: DeviceType::Microphone,
        }
    }

    /// The saved file must hold exactly the captured samples. Per-checkpoint encoding joined with
    /// `concat -c copy` added 1792 samples (37 ms) per 30 s checkpoint.
    #[tokio::test]
    async fn test_saved_audio_matches_captured_length() {
        let temp_dir = tempdir().unwrap();
        let meeting_folder = temp_dir.path().join("Drift_Test");
        std::fs::create_dir_all(meeting_folder.join(".checkpoints")).unwrap();
        let mut saver = IncrementalAudioSaver::new(meeting_folder.clone(), 48000).unwrap();

        // 95 s in 0.5 s chunks: three full checkpoints and a partial one
        for i in 0..190 {
            saver.add_chunk(tone_chunk(i, 24000)).unwrap();
        }
        assert_eq!(saver.get_checkpoint_count(), 3);

        let final_path = saver.finalize().await.unwrap();
        let captured = 95 * 48000;
        let saved = decoded_samples(&final_path, 48000);
        // some ffmpeg builds keep the last frame's padding: allow one AAC frame (1024 samples) at the end
        assert!(
            saved >= captured && saved <= captured + 1024,
            "saved {} samples, captured {} ({:+} ms)",
            saved, captured, (saved as f64 - captured as f64) / 48.0
        );
    }

    /// After a crash the ADTS checkpoint segments rebuild the recording without per-checkpoint drift.
    #[tokio::test]
    async fn test_recovery_from_segments_after_crash() {
        let temp_dir = tempdir().unwrap();
        let meeting_folder = temp_dir.path().join("Crash_Test");
        std::fs::create_dir_all(meeting_folder.join(".checkpoints")).unwrap();
        let mut saver = IncrementalAudioSaver::new(meeting_folder.clone(), 48000).unwrap();

        // 65 s of audio, then the process dies without finalizing
        for i in 0..130 {
            saver.add_chunk(tone_chunk(i, 24000)).unwrap();
        }
        drop(saver.encoder_stdin.take());
        let mut encoder = saver.encoder.take().unwrap();
        // let the encoder write what it has received, as the OS would after the app process exits
        encoder.wait().unwrap();

        assert!(has_audio_checkpoints(meeting_folder.to_string_lossy().into_owned()).await.unwrap());
        let status = recover_audio_from_checkpoints(meeting_folder.to_string_lossy().into_owned(), 48000)
            .await
            .unwrap();
        assert_eq!(status.status, "success", "{}", status.message);
        assert_eq!(status.chunk_count, 3);

        let recovered = decoded_samples(&meeting_folder.join("audio.mp4"), 48000);
        let captured = 65 * 48000;
        // the recovered stream keeps the single 1024-sample encoder priming, and no drift per segment
        assert!(
            recovered >= captured && recovered <= captured + 2048,
            "recovered {} samples, captured {}", recovered, captured
        );
    }

    #[tokio::test]
    async fn test_partial_audio_file_is_not_a_checkpoint() {
        assert!(is_checkpoint_file(Path::new("/m/.checkpoints/audio_chunk_00001.aac")));
        assert!(is_checkpoint_file(Path::new("/m/.checkpoints/audio_chunk_001.mp4")));
        assert!(!is_checkpoint_file(Path::new("/m/.checkpoints/audio.partial.mp4")));
        assert!(!is_checkpoint_file(Path::new("/m/.checkpoints/concat_list.txt")));
    }

    #[tokio::test]
    async fn test_empty_recording() {
        let temp_dir = tempdir().unwrap();
        let meeting_folder = temp_dir.path().join("Empty_Test");
        std::fs::create_dir_all(&meeting_folder).unwrap();
        std::fs::create_dir_all(meeting_folder.join(".checkpoints")).unwrap();

        let mut saver = IncrementalAudioSaver::new(
            meeting_folder.clone(),
            48000
        ).unwrap();

        // Try to finalize without adding any chunks
        let result = saver.finalize().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No audio checkpoints"));
    }

    /// A child that writes more to stderr than a pipe holds must still finish once its stderr is drained. Before
    /// the drain, the encoder blocked like this after about 7.5 minutes of real-time progress lines and stopped
    /// reading audio, so recordings stopped growing and finalize timed out.
    #[test]
    fn drained_stderr_never_blocks_the_child() {
        let mut child = std::process::Command::new("sh")
            .args(["-c", "head -c 300000 /dev/zero | tr '\\0' x >&2; cat > /dev/null"])
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let drain = drain_stderr(child.stderr.take().unwrap());
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(&[0u8; 100_000]).unwrap();
        drop(stdin);
        let started = std::time::Instant::now();
        assert!(child.wait().unwrap().success());
        let kept = drain.join().unwrap();
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        assert_eq!(kept.len(), 64 * 1024);
        assert!(kept.iter().all(|&b| b == b'x'));
    }
}
