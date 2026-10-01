//! Speaker labels for finished meetings.
//!
//! Live transcription is unchanged. After a recording is saved, imported or retranscribed, the meeting's audio is
//! diarized with Nemotron 3 Diarization in the background and each stored transcript segment gets the speaker who
//! holds most of its speech, stored as "Speaker N" in `transcripts.speaker`. Renaming a speaker rewrites that
//! column for the meeting. Summaries read `[mm:ss] Name: text` lines built from the stored labels.

mod model;

use anyhow::{anyhow, bail, Context, Result};
use futures_util::StreamExt;
use log::{info, warn};
use model::{Diarizer, PcmFile, HOP, N_SPK, SR};
use serde::Serialize;
use sha2::{Digest, Sha256};
use sqlx::SqlitePool;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tauri::{AppHandle, Emitter, Manager, Runtime};
use tokio::io::AsyncWriteExt;

const MODEL_DIR: &str = "nemotron-3-diarization-int8";
const MODEL_BASE_URL: &str =
    "https://huggingface.co/NealCaren/Nemotron-3-Diarization-ONNX/resolve/46642c09b3bd014c5ceb19d20524fa2f1dc27dee";
const MODEL_FILES: &[(&str, u64, &str)] = &[
    ("embed.onnx", 2_102_458, "2e8796e42286abe2e2d7d44a97fab7e5a7d06b2d4cbee82a3a0f730cfff84a33"),
    ("step_int8.onnx", 103_325_385, "1f18a5aed4909677d3618c9e6ed6e3003720b6d6e766f0f8b337aa4fcb0a9d2a"),
    ("mel_filters.bin", 131_584, "bce5ec5f194a5913f6508cee5a85512e7bad2352db8fc28f5c6ff75af8b09137"),
    ("silence_embeds.bin", 2_048, "d4417b3c0eabdf7c47032fac2b5b5a7ee83d819a6ddda8fd8eaf74e2b5cc4ac7"),
];

// Segment labelling rule from the spike's label_eval.py, which gave 97.2 to 100% precision on named segments.
const ACTIVE_PROB: f32 = 0.5;
const MIN_ACTIVE: f64 = 0.2; // below this share of frames with any speaker, the segment stays unnamed
const DOMINANT: f64 = 0.7; // one speaker must hold this share of the active frames
const SECOND: f64 = 0.3; // a second speaker at this share makes the segment mixed, so unnamed

static DOWNLOAD_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
// One diarization at a time: it is CPU-heavy, and a summary started during a background run waits for its labels.
static RUN_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Emitted as `diarization-progress` so the meeting page can show what is happening.
#[derive(Clone, Serialize)]
pub struct DiarizationProgress {
    pub meeting_id: String,
    /// "downloading", "diarizing", "done", "skipped" or "failed"
    pub stage: String,
    /// Download progress 0 to 100 while downloading
    pub percent: Option<u32>,
    pub message: String,
}

/// One stored transcript segment, in audio order.
pub struct Segment {
    pub id: String,
    pub start: Option<f64>,
    pub end: Option<f64>,
    pub timestamp: String,
    pub text: String,
    pub speaker: Option<String>,
}

async fn load_segments(pool: &SqlitePool, meeting_id: &str) -> Result<Vec<Segment>> {
    let rows: Vec<(String, Option<f64>, Option<f64>, String, String, Option<String>)> = sqlx::query_as(
        "SELECT id, audio_start_time, audio_end_time, timestamp, transcript, speaker FROM transcripts
         WHERE meeting_id = ? ORDER BY audio_start_time ASC",
    )
    .bind(meeting_id)
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|(id, start, end, timestamp, text, speaker)| Segment { id, start, end, timestamp, text, speaker })
        .collect())
}

/// Diarizes the meeting's saved audio and stores "Speaker N" (or NULL) on each transcript segment.
/// Returns false when the meeting cannot be diarized (no saved audio, no segment timings).
pub async fn diarize_meeting(
    pool: &SqlitePool,
    app_data_dir: &Path,
    meeting_id: &str,
    progress: &(dyn Fn(&str, Option<u32>) + Send + Sync),
) -> Result<bool> {
    let folder: Option<String> = sqlx::query_scalar("SELECT folder_path FROM meetings WHERE id = ?")
        .bind(meeting_id)
        .fetch_optional(pool)
        .await?
        .flatten();
    let Some(audio) = folder.and_then(|f| crate::audio::retranscription::find_audio_file(Path::new(&f)).ok()) else {
        info!("Diarization skipped for {}: no saved audio", meeting_id);
        return Ok(false);
    };
    let segments = load_segments(pool, meeting_id).await?;
    if !segments.iter().any(|s| s.start.is_some() && s.end.is_some()) {
        info!("Diarization skipped for {}: transcript has no audio timings", meeting_id);
        return Ok(false);
    }

    let started = std::time::Instant::now();
    let (labels, audio_secs) = label_audio(app_data_dir, &audio, &segments, progress).await?;

    let mut tx = pool.begin().await?;
    for (seg, label) in segments.iter().zip(&labels) {
        sqlx::query("UPDATE transcripts SET speaker = ? WHERE id = ?")
            .bind(label.map(|n| format!("Speaker {}", n)))
            .bind(&seg.id)
            .execute(&mut *tx)
            .await?;
    }
    tx.commit().await?;

    info!(
        "Diarized {} ({:.0} s audio) in {:.1} s: {} of {} segments named, {} speakers",
        meeting_id,
        audio_secs,
        started.elapsed().as_secs_f64(),
        labels.iter().filter(|l| l.is_some()).count(),
        labels.len(),
        labels.iter().flatten().max().map_or(0, |n| *n),
    );
    Ok(true)
}

/// Diarizes an audio file and returns, for each segment, its speaker number (1, 2, ... by first appearance) or None
/// when silent, mixed or untimed, plus the audio length in seconds.
async fn label_audio(
    app_data_dir: &Path,
    audio: &Path,
    segments: &[Segment],
    progress: &(dyn Fn(&str, Option<u32>) + Send + Sync),
) -> Result<(Vec<Option<usize>>, f64)> {
    let model_dir = ensure_model(&app_data_dir.join("models").join(MODEL_DIR), progress).await?;
    progress("diarizing", None);
    let pcm = decode_to_pcm(audio).await?;
    let (probs, num_frames) = tokio::task::spawn_blocking(move || -> Result<_> {
        let mut source = PcmFile::open(&pcm)?;
        let out = Diarizer::load(&model_dir)?.run(&mut source)?;
        drop(pcm); // deletes the temp file
        Ok(out)
    })
    .await??;
    Ok((number_speakers(&label_segments(&probs, num_frames, segments)), (num_frames * HOP) as f64 / SR as f64))
}

/// Speaker numbers for the timed segments of a meeting still being recorded, from the audio recorded so far.
/// Fails instead of waiting when another diarization is running. Segments past the recorded audio stay None.
pub async fn label_live_segments(
    app_data_dir: &Path,
    meeting_folder: &Path,
    spans: &[(Option<f64>, Option<f64>)],
) -> Result<Vec<Option<usize>>> {
    // Don't make a live question wait minutes behind the previous meeting's diarization
    let _run = RUN_LOCK
        .try_lock()
        .map_err(|_| anyhow!("speaker identification is busy with a saved meeting"))?;
    let joined = tempfile::Builder::new().prefix("meetily_live_").suffix(".aac").tempfile()?.into_temp_path();
    let count = crate::audio::incremental_saver::join_recorded_audio(meeting_folder, &joined).map_err(|e| anyhow!(e))?;
    if count == 0 {
        bail!("No audio has been saved yet; try again in 30 seconds");
    }
    let segments: Vec<Segment> = spans
        .iter()
        .map(|&(start, end)| Segment { id: String::new(), start, end, timestamp: String::new(), text: String::new(), speaker: None })
        .collect();
    let started = std::time::Instant::now();
    let (labels, audio_secs) = label_audio(app_data_dir, &joined, &segments, &|_, _| {}).await?;
    info!(
        "Diarized live meeting ({:.0} s audio) in {:.1} s: {} of {} segments named",
        audio_secs,
        started.elapsed().as_secs_f64(),
        labels.iter().filter(|l| l.is_some()).count(),
        labels.len()
    );
    Ok(labels)
}

/// Runs `diarize_meeting` for the app, one at a time, reporting progress as `diarization-progress` events.
pub async fn diarize_for_app<R: Runtime>(app: &AppHandle<R>, meeting_id: &str) -> Result<bool> {
    let _run = RUN_LOCK.lock().await;
    let pool = app.state::<crate::state::AppState>().db_manager.pool().clone();
    let app_data_dir = app.path().app_data_dir().context("no app data dir")?;
    let emit = |stage: &str, percent: Option<u32>, message: String| {
        let _ = app.emit(
            "diarization-progress",
            DiarizationProgress { meeting_id: meeting_id.to_string(), stage: stage.to_string(), percent, message },
        );
    };
    let progress = |stage: &str, percent: Option<u32>| {
        let message = match stage {
            "downloading" => "Downloading the speaker model (105 MB, first time only)".to_string(),
            _ => "Identifying speakers".to_string(),
        };
        emit(stage, percent, message);
    };
    match diarize_meeting(&pool, &app_data_dir, meeting_id, &progress).await {
        Ok(true) => {
            emit("done", None, "Speakers identified".to_string());
            Ok(true)
        }
        Ok(false) => {
            emit("skipped", None, "No saved audio to identify speakers from".to_string());
            Ok(false)
        }
        Err(e) => {
            warn!("Diarization failed for {}: {:#}", meeting_id, e);
            emit("failed", None, format!("Could not identify speakers: {}", e));
            Err(e)
        }
    }
}

/// Starts background diarization after a meeting's transcript is saved. Failures only log and emit.
pub fn spawn_for_meeting<R: Runtime>(app: AppHandle<R>, meeting_id: String) {
    tauri::async_runtime::spawn(async move {
        let _ = diarize_for_app(&app, &meeting_id).await;
    });
}

/// Summary transcript with speaker names from the stored labels. Diarizes first when the meeting has none yet,
/// and falls back to the frontend's unlabelled text when that is impossible or fails.
pub async fn transcript_for_summary<R: Runtime>(app: &AppHandle<R>, pool: &SqlitePool, meeting_id: &str, text: String) -> String {
    let labelled = |segments: &[Segment]| segments.iter().any(|s| s.speaker.is_some());
    let mut segments = match load_segments(pool, meeting_id).await {
        Ok(s) => s,
        Err(e) => {
            warn!("Could not read transcript for speaker names: {:#}", e);
            return text;
        }
    };
    if !labelled(&segments) {
        if !matches!(diarize_for_app(app, meeting_id).await, Ok(true)) {
            return text;
        }
        segments = match load_segments(pool, meeting_id).await {
            Ok(s) => s,
            Err(_) => return text,
        };
    }
    if labelled(&segments) { format_transcript(&segments) } else { text }
}

/// Renames one speaker across a meeting. Returns the number of segments changed.
pub async fn rename_speaker(pool: &SqlitePool, meeting_id: &str, from: &str, to: &str) -> Result<u64> {
    let to = to.trim();
    if to.is_empty() {
        bail!("Speaker name cannot be empty");
    }
    let result = sqlx::query("UPDATE transcripts SET speaker = ? WHERE meeting_id = ? AND speaker = ?")
        .bind(to)
        .bind(meeting_id)
        .bind(from)
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
}

#[tauri::command]
pub async fn diarization_rename_speaker<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    from: String,
    to: String,
) -> Result<u64, String> {
    let pool = app.state::<crate::state::AppState>().db_manager.pool().clone();
    rename_speaker(&pool, &meeting_id, &from, &to).await.map_err(|e| e.to_string())
}

/// Identifies speakers on demand, for meetings saved before diarization existed or after a failed run.
#[tauri::command]
pub async fn diarization_run<R: Runtime>(app: AppHandle<R>, meeting_id: String) -> Result<bool, String> {
    diarize_for_app(&app, &meeting_id).await.map_err(|e| e.to_string())
}

/// Model speaker index for each segment, or None when the segment is silent, mixed or has no timings.
fn label_segments(probs: &[f32], num_frames: usize, segments: &[Segment]) -> Vec<Option<usize>> {
    let fps = (SR / HOP) as f64;
    segments
        .iter()
        .map(|seg| {
            let (Some(start), Some(end)) = (seg.start, seg.end) else { return None };
            let f0 = ((start * fps) as usize).min(num_frames);
            let f1 = ((end * fps) as usize).max(f0 + 1).min(num_frames);
            if f0 >= f1 {
                return None;
            }
            let mut share = [0usize; N_SPK];
            let mut active = 0usize;
            for t in f0..f1 {
                let row = &probs[t * N_SPK..(t + 1) * N_SPK];
                if row.iter().any(|&p| p > ACTIVE_PROB) {
                    active += 1;
                    for (s, &p) in row.iter().enumerate() {
                        if p > ACTIVE_PROB {
                            share[s] += 1;
                        }
                    }
                }
            }
            if (active as f64) < MIN_ACTIVE * (f1 - f0) as f64 {
                return None;
            }
            let mut order: Vec<usize> = (0..N_SPK).collect();
            order.sort_by(|&a, &b| share[b].cmp(&share[a]).then(a.cmp(&b)));
            let frac = |s: usize| share[s] as f64 / active as f64;
            (frac(order[1]) < SECOND && frac(order[0]) >= DOMINANT).then_some(order[0])
        })
        .collect()
}

/// Renames model speaker slots to 1, 2, 3... in order of first appearance in the transcript.
fn number_speakers(labels: &[Option<usize>]) -> Vec<Option<usize>> {
    let mut seen: Vec<usize> = Vec::new();
    labels
        .iter()
        .map(|l| {
            l.map(|s| match seen.iter().position(|&x| x == s) {
                Some(i) => i + 1,
                None => {
                    seen.push(s);
                    seen.len()
                }
            })
        })
        .collect()
}

/// Same line format the frontend sends (`[mm:ss] text`), with `Name: ` before named segments.
fn format_transcript(segments: &[Segment]) -> String {
    segments
        .iter()
        .map(|seg| {
            let time = match seg.start {
                Some(s) => {
                    let secs = s.max(0.0) as u64;
                    format!("[{:02}:{:02}]", secs / 60, secs % 60)
                }
                None => seg.timestamp.clone(),
            };
            match &seg.speaker {
                Some(name) => format!("{} {}: {}", time, name, seg.text),
                None => format!("{} {}", time, seg.text),
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Downloads the pinned model files on first use and checks their SHA-256.
async fn ensure_model(dir: &Path, progress: &(dyn Fn(&str, Option<u32>) + Send + Sync)) -> Result<PathBuf> {
    let _guard = DOWNLOAD_LOCK.lock().await;
    tokio::fs::create_dir_all(dir).await?;
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()?;
    for (name, size, sha) in MODEL_FILES {
        let path = dir.join(name);
        if tokio::fs::metadata(&path).await.map(|m| m.len() == *size).unwrap_or(false) {
            continue;
        }
        info!("Downloading diarization model file {} ({} bytes)", name, size);
        progress("downloading", Some(0));
        let part = dir.join(format!("{}.part", name));
        let response = client
            .get(format!("{}/{}", MODEL_BASE_URL, name))
            .send()
            .await?
            .error_for_status()?;
        let mut file = tokio::fs::File::create(&part).await?;
        let mut hasher = Sha256::new();
        let mut stream = response.bytes_stream();
        let (mut done, mut reported) = (0u64, 0u32);
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            hasher.update(&chunk);
            file.write_all(&chunk).await?;
            done += chunk.len() as u64;
            let percent = (done * 100 / size) as u32;
            if percent >= reported + 5 {
                reported = percent;
                progress("downloading", Some(percent.min(100)));
            }
        }
        file.flush().await?;
        drop(file);
        let got = format!("{:x}", hasher.finalize());
        if got != *sha {
            let _ = tokio::fs::remove_file(&part).await;
            bail!("{} failed its checksum: got {}, expected {}", name, got, sha);
        }
        tokio::fs::rename(&part, &path).await?;
    }
    Ok(dir.to_path_buf())
}

/// Decodes the recording to 16 kHz mono f32 in a temp file with the bundled ffmpeg.
async fn decode_to_pcm(audio: &Path) -> Result<tempfile::TempPath> {
    let ffmpeg = crate::audio::ffmpeg::find_ffmpeg_path().ok_or_else(|| anyhow!("ffmpeg not found"))?;
    let out = tempfile::Builder::new()
        .prefix("meetily_diarize_")
        .suffix(".f32")
        .tempfile()?
        .into_temp_path();
    let mut command = tokio::process::Command::new(ffmpeg);
    command
        .arg("-v").arg("error")
        .arg("-i").arg(audio)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-y"])
        .arg(&*out)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    #[cfg(target_os = "windows")]
    {
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let output = command.output().await.context("running ffmpeg")?;
    if !output.status.success() {
        bail!("ffmpeg failed on {}: {}", audio.display(), String::from_utf8_lossy(&output.stderr).trim());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f64, end: f64, text: &str) -> Segment {
        Segment { id: String::new(), start: Some(start), end: Some(end), timestamp: String::new(), text: text.into(), speaker: None }
    }

    // probs for `secs` seconds where `who(t)` lists the active speakers at frame t
    fn probs(secs: usize, who: impl Fn(usize) -> Vec<usize>) -> (Vec<f32>, usize) {
        let n = secs * 100;
        let mut p = vec![0.01f32; n * N_SPK];
        for t in 0..n {
            for s in who(t) {
                p[t * N_SPK + s] = 0.9;
            }
        }
        (p, n)
    }

    #[test]
    fn names_dominant_speaker_and_withholds_mixed_or_silent() {
        // speaker 3 for 0-10 s, speaker 5 for 10-20 s, silence 20-30 s
        let (p, n) = probs(30, |t| match t {
            0..=999 => vec![3],
            1000..=1999 => vec![5],
            _ => vec![],
        });
        let segs = [
            seg(1.0, 9.0, "a"),   // speaker 3 only
            seg(8.0, 12.0, "b"),  // 50/50: mixed
            seg(10.5, 19.0, "c"), // speaker 5 only
            seg(21.0, 29.0, "d"), // silent
            seg(9.0, 18.0, "e"),  // speaker 5 holds 8 of 9 s
            seg(40.0, 45.0, "f"), // past the end of the audio
        ];
        assert_eq!(
            label_segments(&p, n, &segs),
            vec![Some(3), None, Some(5), None, Some(5), None]
        );
    }

    /// Compares against the spike binary's `--probs` output on the same 16 kHz WAV.
    /// DIARIZATION_MODELS=<dir> DIARIZATION_WAV=<wav> DIARIZATION_REF_PROBS=<f32> cargo test --lib diarization -- --ignored
    #[tokio::test]
    #[ignore]
    async fn matches_spike_probabilities() {
        let env = |k: &str| PathBuf::from(std::env::var(k).unwrap_or_else(|_| panic!("{k} not set")));
        let pcm = decode_to_pcm(&env("DIARIZATION_WAV")).await.unwrap();
        let (probs, _) = Diarizer::load(&env("DIARIZATION_MODELS")).unwrap().run(&mut PcmFile::open(&pcm).unwrap()).unwrap();
        let bytes = std::fs::read(env("DIARIZATION_REF_PROBS")).unwrap();
        let reference: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        assert_eq!(probs.len(), reference.len());
        let flips = probs.iter().zip(&reference).filter(|(a, b)| (**a > 0.5) != (**b > 0.5)).count();
        let max_diff = probs.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("values {} thresholded flips {} ({:.4}%) max diff {:.2e}", probs.len(), flips, 100.0 * flips as f64 / probs.len() as f64, max_diff);
        assert!((flips as f64) < 0.001 * probs.len() as f64);
    }

    /// Downloads the model into DIARIZATION_APP_DIR and labels a meeting in a temp copy of the app database.
    /// DIARIZATION_DB=<sqlite> DIARIZATION_MEETING=<id> DIARIZATION_APP_DIR=<dir> cargo test --lib diarization -- --ignored
    #[tokio::test]
    #[ignore]
    async fn labels_a_saved_meeting() {
        let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"));
        let copy = tempfile::NamedTempFile::new().unwrap();
        std::fs::copy(env("DIARIZATION_DB"), copy.path()).unwrap();
        let pool = SqlitePool::connect(&format!("sqlite://{}", copy.path().display())).await.unwrap();
        let meeting = env("DIARIZATION_MEETING");
        let ran = diarize_meeting(&pool, Path::new(&env("DIARIZATION_APP_DIR")), &meeting, &|stage, pct| println!("{stage} {pct:?}"))
            .await
            .unwrap();
        assert!(ran, "meeting has audio and timings");
        let text = format_transcript(&load_segments(&pool, &meeting).await.unwrap());
        println!("{text}");
        assert!(text.contains("Speaker 1: "));
    }

    async fn transcripts_db(rows: &[(&str, &str, Option<&str>)]) -> SqlitePool {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query("CREATE TABLE transcripts (id TEXT, meeting_id TEXT, transcript TEXT, timestamp TEXT,
                     audio_start_time REAL, audio_end_time REAL, speaker TEXT)")
            .execute(&pool).await.unwrap();
        for (i, (meeting, text, speaker)) in rows.iter().enumerate() {
            sqlx::query("INSERT INTO transcripts VALUES (?, ?, ?, '', ?, ?, ?)")
                .bind(format!("t{i}")).bind(meeting).bind(text)
                .bind(i as f64 * 10.0).bind(i as f64 * 10.0 + 5.0).bind(speaker)
                .execute(&pool).await.unwrap();
        }
        pool
    }

    #[tokio::test]
    async fn renames_one_speaker_in_one_meeting() {
        let pool = transcripts_db(&[
            ("m1", "hi", Some("Speaker 1")),
            ("m1", "hello", Some("Speaker 2")),
            ("m1", "mm", None),
            ("m1", "bye", Some("Speaker 2")),
            ("m2", "other", Some("Speaker 2")),
        ]).await;
        assert_eq!(rename_speaker(&pool, "m1", "Speaker 2", " Priya ").await.unwrap(), 2);
        assert!(rename_speaker(&pool, "m1", "Speaker 1", "  ").await.is_err());
        assert_eq!(
            format_transcript(&load_segments(&pool, "m1").await.unwrap()),
            "[00:00] Speaker 1: hi\n[00:10] Priya: hello\n[00:20] mm\n[00:30] Priya: bye"
        );
        assert_eq!(load_segments(&pool, "m2").await.unwrap()[0].speaker.as_deref(), Some("Speaker 2"));
    }

    #[test]
    fn numbers_speakers_by_first_appearance() {
        assert_eq!(
            number_speakers(&[Some(5), None, Some(2), Some(5), Some(7)]),
            vec![Some(1), None, Some(2), Some(1), Some(3)]
        );
    }

    #[test]
    fn formats_like_the_frontend_with_names() {
        let mut segs = vec![seg(65.4, 70.0, "hello"), seg(71.0, 72.0, "hmm")];
        segs[0].speaker = Some("Speaker 1".into());
        segs.push(Segment {
            id: String::new(), start: None, end: None, timestamp: "2025-01-01T10:00:00Z".into(), text: "old".into(), speaker: None,
        });
        assert_eq!(
            format_transcript(&segs),
            "[01:05] Speaker 1: hello\n[01:11] hmm\n2025-01-01T10:00:00Z old"
        );
    }
}
