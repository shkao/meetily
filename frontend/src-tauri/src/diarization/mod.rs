//! Speaker labels for meeting minutes (issue #1).
//!
//! Live transcription is unchanged. When a summary is generated, the saved `audio.mp4` is diarized with
//! Nemotron 3 Diarization and each stored transcript segment gets the speaker who holds most of its speech.
//! The LLM then reads `[mm:ss] Speaker N: text` lines. Any failure leaves the transcript unlabelled.

mod model;

use anyhow::{anyhow, bail, Context, Result};
use futures_util::StreamExt;
use log::{info, warn};
use model::{Diarizer, PcmFile, HOP, N_SPK, SR};
use sha2::{Digest, Sha256};
use sqlx::SqlitePool;
use std::path::{Path, PathBuf};
use std::process::Stdio;
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

/// One stored transcript segment, in the order the summary text lists them.
pub struct Segment {
    pub start: Option<f64>,
    pub end: Option<f64>,
    pub timestamp: String,
    pub text: String,
}

/// Rebuilds the summary transcript with speaker names, or returns None when the meeting cannot be diarized
/// (no saved audio, no segment timings). Errors cover download, decode and inference failures.
pub async fn speaker_labelled_transcript(
    pool: &SqlitePool,
    app_data_dir: &Path,
    meeting_id: &str,
) -> Result<Option<String>> {
    let folder: Option<String> = sqlx::query_scalar("SELECT folder_path FROM meetings WHERE id = ?")
        .bind(meeting_id)
        .fetch_optional(pool)
        .await?
        .flatten();
    let Some(audio) = folder.map(|f| PathBuf::from(f).join("audio.mp4")).filter(|p| p.exists()) else {
        info!("Diarization skipped for {}: no saved audio.mp4", meeting_id);
        return Ok(None);
    };
    let rows: Vec<(Option<f64>, Option<f64>, String, String)> = sqlx::query_as(
        "SELECT audio_start_time, audio_end_time, timestamp, transcript FROM transcripts
         WHERE meeting_id = ? ORDER BY audio_start_time ASC",
    )
    .bind(meeting_id)
    .fetch_all(pool)
    .await?;
    let segments: Vec<Segment> = rows
        .into_iter()
        .map(|(start, end, timestamp, text)| Segment { start, end, timestamp, text })
        .collect();
    if !segments.iter().any(|s| s.start.is_some() && s.end.is_some()) {
        info!("Diarization skipped for {}: transcript has no audio timings", meeting_id);
        return Ok(None);
    }

    let model_dir = ensure_model(&app_data_dir.join("models").join(MODEL_DIR)).await?;
    let pcm = decode_to_pcm(&audio).await?;
    let started = std::time::Instant::now();
    let (probs, num_frames) = tokio::task::spawn_blocking(move || -> Result<_> {
        let mut source = PcmFile::open(&pcm)?;
        let out = Diarizer::load(&model_dir)?.run(&mut source)?;
        drop(pcm); // deletes the temp file
        Ok(out)
    })
    .await??;
    let labels = number_speakers(&label_segments(&probs, num_frames, &segments));
    info!(
        "Diarized {} ({:.0} s audio) in {:.1} s: {} of {} segments named, {} speakers",
        meeting_id,
        (num_frames * HOP) as f64 / SR as f64,
        started.elapsed().as_secs_f64(),
        labels.iter().filter(|l| l.is_some()).count(),
        labels.len(),
        labels.iter().flatten().max().map_or(0, |n| *n),
    );
    Ok(Some(format_transcript(&segments, &labels)))
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

/// Same line format the frontend sends (`[mm:ss] text`), with `Speaker N: ` before named segments.
fn format_transcript(segments: &[Segment], labels: &[Option<usize>]) -> String {
    segments
        .iter()
        .zip(labels)
        .map(|(seg, label)| {
            let time = match seg.start {
                Some(s) => {
                    let secs = s.max(0.0) as u64;
                    format!("[{:02}:{:02}]", secs / 60, secs % 60)
                }
                None => seg.timestamp.clone(),
            };
            match label {
                Some(n) => format!("{} Speaker {}: {}", time, n, seg.text),
                None => format!("{} {}", time, seg.text),
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Downloads the pinned model files on first use and checks their SHA-256.
async fn ensure_model(dir: &Path) -> Result<PathBuf> {
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
        let part = dir.join(format!("{}.part", name));
        let response = client
            .get(format!("{}/{}", MODEL_BASE_URL, name))
            .send()
            .await?
            .error_for_status()?;
        let mut file = tokio::fs::File::create(&part).await?;
        let mut hasher = Sha256::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            hasher.update(&chunk);
            file.write_all(&chunk).await?;
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

/// Runs diarization for the summary, falling back to the frontend's unlabelled text on any failure.
pub async fn transcript_for_summary(
    pool: &SqlitePool,
    app_data_dir: Option<&Path>,
    meeting_id: &str,
    text: String,
) -> String {
    let Some(dir) = app_data_dir else { return text };
    match speaker_labelled_transcript(pool, dir, meeting_id).await {
        Ok(Some(labelled)) => labelled,
        Ok(None) => text,
        Err(e) => {
            warn!("Diarization failed for {}, summarizing without speaker names: {:#}", meeting_id, e);
            text
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f64, end: f64, text: &str) -> Segment {
        Segment { start: Some(start), end: Some(end), timestamp: String::new(), text: text.into() }
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

    /// Downloads the model into DIARIZATION_APP_DIR and labels a meeting from a copy of the app database.
    /// DIARIZATION_DB=<sqlite> DIARIZATION_MEETING=<id> DIARIZATION_APP_DIR=<dir> cargo test --lib diarization -- --ignored
    #[tokio::test]
    #[ignore]
    async fn labels_a_saved_meeting() {
        let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"));
        let pool = SqlitePool::connect(&format!("sqlite://{}?mode=ro", env("DIARIZATION_DB"))).await.unwrap();
        let text = speaker_labelled_transcript(&pool, Path::new(&env("DIARIZATION_APP_DIR")), &env("DIARIZATION_MEETING"))
            .await
            .unwrap()
            .expect("meeting has audio and timings");
        println!("{text}");
        assert!(text.contains("Speaker 1: "));
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
        let segs = [
            seg(65.4, 70.0, "hello"),
            seg(71.0, 72.0, "hmm"),
            Segment { start: None, end: None, timestamp: "2025-01-01T10:00:00Z".into(), text: "old".into() },
        ];
        assert_eq!(
            format_transcript(&segs, &[Some(1), None, None]),
            "[01:05] Speaker 1: hello\n[01:11] hmm\n2025-01-01T10:00:00Z old"
        );
    }
}
