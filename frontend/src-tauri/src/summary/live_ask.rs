//! Ask AI during a meeting, modelled on Google Meet's "Ask Gemini".
//!
//! The panel sends the live transcript it already holds (it is not in the database until the meeting is saved)
//! and a question. The answer comes from the provider picked for summaries, so the transcript never goes to a
//! provider the user has not chosen. Transcripts that fit the model's context go in whole; longer ones are split
//! with the summary chunker, notes relevant to the question are taken from each part, and the answer comes from
//! those notes. Live transcription is unlabelled; when a question is about who said what, the audio recorded so far
//! is diarized first and lines get anonymous "Speaker N" labels.

use crate::database::repositories::setting::SettingsRepository;
use crate::summary::llm_client::{generate_streaming, generate_summary};
use crate::summary::processor::{chunk_text, clean_llm_markdown_detailed, rough_token_count};
use crate::summary::service::{LlmSettings, SummaryService};
use log::info;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, Runtime};

/// One live transcript segment as the panel holds it.
#[derive(Debug, Deserialize)]
pub struct LiveLine {
    /// Seconds from recording start
    pub start: Option<f64>,
    pub end: Option<f64>,
    pub text: String,
}

const SYSTEM_PROMPT_TEMPLATE: &str = "You answer questions about a meeting that is still in progress, using only its live \
transcript. Each line starts with its time as [mm:ss]. Rules:
- Answer briefly and directly. Use short bullet points for lists.
- End every bullet point and sentence with the [mm:ss] time of the line it comes from, exactly as it appears, for \
example: \"Budget review moved to Friday [03:15]\". Never invent times.
- {speakers}
- If the transcript doesn't cover the question, say so. Don't guess.
- Answer in the language of the question.";

const NO_SPEAKERS_RULE: &str = "The transcript has no speaker names. If the question asks who said something or what \
a named person said, say that speaker names aren't available during the meeting, then answer what you can without \
attributing it.";

const SPEAKERS_RULE: &str = "Some lines start with an anonymous label such as \"Speaker 2:\". The numbers tell \
voices apart but not who they are; lines without a label have no reliable speaker. If the question names a person, \
say that speakers are only numbered during the meeting, then answer using the numbers.";

fn system_prompt(with_speakers: bool) -> String {
    SYSTEM_PROMPT_TEMPLATE.replace("{speakers}", if with_speakers { SPEAKERS_RULE } else { NO_SPEAKERS_RULE })
}

/// Whether a question is about who said what, which needs speaker labels.
/// "Who...", "each speaker", or "did/does <someone> say/agree..." with a named subject; "what was said about X",
/// "what did we say" and "in person" don't count, since diarizing costs seconds of CPU during the meeting.
fn needs_speakers(question: &str) -> bool {
    const WHO: &[&str] = &["who", "whom", "whose", "speaker", "speakers"];
    const SUBJECTLESS: &[&str] = &["we", "i", "you", "they", "it", "anyone", "anybody", "someone", "somebody", "everyone", "people", "the", "this", "that"];
    const VERBS: &[&str] = &["say", "said", "mention", "mentioned", "think", "suggest", "suggested", "propose", "agree", "disagree", "ask", "want", "decide", "promise", "commit"];
    let words: Vec<String> = question
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(String::from)
        .collect();
    if words.iter().any(|w| WHO.contains(&w.as_str())) {
        return true;
    }
    words.windows(3).any(|w| {
        matches!(w[0].as_str(), "did" | "does" | "do")
            && !SUBJECTLESS.contains(&w[1].as_str())
            && VERBS.contains(&w[2].as_str())
    })
}

const NOTES_PROMPT: &str = "You read one part of a live meeting transcript. Each line starts with its time as \
[mm:ss]. Copy out, as short bullet points, only what is relevant to the question, keeping each point's [mm:ss] \
time. If nothing is relevant, reply with exactly: NONE";

fn format_time(seconds: f64) -> String {
    let secs = seconds.max(0.0) as u64;
    format!("[{:02}:{:02}]", secs / 60, secs % 60)
}

/// Transcript lines as `[mm:ss] text`, or `[mm:ss] Speaker N: text` where `speakers` names the line, skipping
/// empty segments.
fn format_transcript(lines: &[LiveLine], speakers: &[Option<usize>]) -> String {
    lines
        .iter()
        .enumerate()
        .filter(|(_, l)| !l.text.trim().is_empty())
        .map(|(i, l)| {
            let text = match speakers.get(i).copied().flatten() {
                Some(n) => format!("Speaker {}: {}", n, l.text.trim()),
                None => l.text.trim().to_string(),
            };
            match l.start {
                Some(s) => format!("{} {}", format_time(s), text),
                None => text,
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn question_prompt(transcript: &str, question: &str) -> String {
    format!("<transcript>\n{}\n</transcript>\n\nQuestion: {}", transcript, question)
}

async fn complete(
    client: &Client,
    settings: &LlmSettings,
    model_name: &str,
    app_data_dir: Option<&std::path::PathBuf>,
    system: &str,
    user: &str,
) -> Result<String, String> {
    let completion = generate_summary(
        client,
        &settings.provider,
        model_name,
        &settings.api_key,
        system,
        user,
        settings.ollama_endpoint.as_deref(),
        settings.custom_openai_endpoint.as_deref(),
        settings.custom_openai_max_tokens,
        settings.custom_openai_temperature,
        settings.custom_openai_top_p,
        app_data_dir,
        None,
    )
    .await?;
    Ok(clean_llm_markdown_detailed(&completion.content).markdown.trim().to_string())
}

/// Answer text as it is generated, emitted as `ask-ai-token`.
#[derive(Clone, Serialize)]
struct AnswerToken {
    request_id: String,
    text: String,
}

/// What the panel shows while the answer is being prepared, emitted as `ask-ai-status`.
#[derive(Clone, Serialize)]
struct AnswerStatus {
    request_id: String,
    message: String,
}

/// Answers a question from the live transcript with the summary provider. Questions about who said what first
/// diarize the audio recorded so far. Local providers stream the answer as `ask-ai-token` events; the returned
/// text is the final, cleaned answer.
#[tauri::command]
pub async fn ask_ai_live<R: Runtime>(
    app: AppHandle<R>,
    request_id: String,
    question: String,
    lines: Vec<LiveLine>,
) -> Result<String, String> {
    let pool = app.state::<crate::state::AppState>().db_manager.pool().clone();
    let app_data_dir = app.path().app_data_dir().ok();
    let status = |message: &str| {
        let _ = app.emit("ask-ai-status", AnswerStatus { request_id: request_id.clone(), message: message.to_string() });
    };

    let mut speakers = Vec::new();
    if needs_speakers(&question) {
        status("Identifying speakers in the audio so far...");
        let folder = crate::audio::recording_commands::get_meeting_folder_path().await?;
        let result = match (folder, &app_data_dir) {
            (Some(folder), Some(dir)) => {
                let spans: Vec<_> = lines.iter().map(|l| (l.start, l.end)).collect();
                crate::diarization::label_live_segments(dir, std::path::Path::new(&folder), &spans)
                    .await
                    .map_err(|e| e.to_string())
            }
            _ => Err("No recording in progress".to_string()),
        };
        match result {
            Ok(labels) => {
                speakers = labels;
                status("Reading the transcript...");
            }
            Err(e) => status(&format!("Couldn't identify speakers ({}); answering without them", e)),
        }
    }

    let on_token = |text: &str| {
        let _ = app.emit("ask-ai-token", AnswerToken { request_id: request_id.clone(), text: text.to_string() });
    };
    answer(&pool, app_data_dir, &question, &lines, &speakers, &on_token).await
}

async fn answer(
    pool: &sqlx::SqlitePool,
    app_data_dir: Option<std::path::PathBuf>,
    question: &str,
    lines: &[LiveLine],
    speakers: &[Option<usize>],
    on_token: &(dyn Fn(&str) + Send + Sync),
) -> Result<String, String> {
    let question = question.trim();
    if question.is_empty() {
        return Err("Type a question first".to_string());
    }
    let transcript = format_transcript(lines, speakers);
    let system = system_prompt(speakers.iter().any(|s| s.is_some()));
    if transcript.is_empty() {
        return Ok("Nothing has been transcribed yet.".to_string());
    }

    let config = SettingsRepository::get_model_config(pool)
        .await
        .map_err(|e| format!("Failed to read AI model settings: {}", e))?
        .ok_or_else(|| "Choose an AI model in Settings first".to_string())?;
    let settings = SummaryService::resolve_llm_settings(pool, &config.provider, &config.model).await?;
    let client = Client::new();

    let budget = settings.token_threshold.saturating_sub(rough_token_count(&system) + rough_token_count(question));
    let context = if rough_token_count(&transcript) < budget {
        transcript
    } else {
        // Same chunk sizing as summaries for small local contexts
        let chunks = chunk_text(&transcript, settings.token_threshold.saturating_sub(300), 100);
        info!("Ask AI: transcript over the context budget, reading {} parts", chunks.len());
        let mut notes = Vec::new();
        for chunk in &chunks {
            let part = complete(&client, &settings, &config.model, app_data_dir.as_ref(), NOTES_PROMPT, &question_prompt(chunk, question)).await?;
            if !part.is_empty() && part != "NONE" {
                notes.push(part);
            }
        }
        if notes.is_empty() {
            return Ok("The transcript so far doesn't cover that.".to_string());
        }
        notes.join("\n")
    };

    let completion = generate_streaming(
        &client,
        &settings.provider,
        &config.model,
        &settings.api_key,
        &system,
        &question_prompt(&context, question),
        settings.ollama_endpoint.as_deref(),
        settings.custom_openai_endpoint.as_deref(),
        settings.custom_openai_max_tokens,
        settings.custom_openai_temperature,
        settings.custom_openai_top_p,
        app_data_dir.as_ref(),
        on_token,
    )
    .await?;
    Ok(clean_llm_markdown_detailed(&completion.content).markdown.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_lines_with_times_and_skips_empty() {
        let lines = vec![
            LiveLine { start: Some(5.4), end: None, text: " hello ".into() },
            LiveLine { start: Some(9.0), end: None, text: "  ".into() },
            LiveLine { start: Some(125.0), end: None, text: "next item".into() },
            LiveLine { start: None, end: None, text: "untimed".into() },
        ];
        assert_eq!(format_transcript(&lines, &[]), "[00:05] hello\n[02:05] next item\nuntimed");
        assert_eq!(
            format_transcript(&lines, &[Some(2), Some(1), None, Some(1)]),
            "[00:05] Speaker 2: hello\n[02:05] next item\nSpeaker 1: untimed"
        );
    }

    #[test]
    fn prompt_asks_for_times_and_admits_missing_speakers() {
        assert!(system_prompt(false).contains("[03:15]"));
        assert!(system_prompt(false).contains("speaker names aren't available"));
        assert!(system_prompt(true).contains("only numbered during the meeting"));
        assert!(!system_prompt(true).contains("{speakers}"));
    }

    #[test]
    fn detects_questions_about_who_said_what() {
        for q in ["What did Priya say about the budget?", "Who disagreed?", "What has each speaker said?", "Whose idea was it?", "Did Priya agree?"] {
            assert!(needs_speakers(q), "{q}");
        }
        for q in [
            "Catch me up", "What are the action items so far?", "Summarize the discussion so far", "Any wholesale changes?",
            "What was said about the deadline?", "What did we say about hiring?", "Is the offsite in person?",
            "What did they decide?",
        ] {
            assert!(!needs_speakers(q), "{q}");
        }
    }

    /// Asks the configured summary provider about a saved meeting's transcript, as the live panel would.
    /// ASK_DB=<sqlite> ASK_MEETING=<id> ASK_APP_DIR=<app data dir> cargo test --lib live_ask -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn answers_from_a_real_transcript() {
        let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"));
        let pool = sqlx::SqlitePool::connect(&format!("sqlite://{}?mode=ro", env("ASK_DB"))).await.unwrap();
        let rows: Vec<(Option<f64>, Option<f64>, String)> = sqlx::query_as(
            "SELECT audio_start_time, audio_end_time, transcript FROM transcripts WHERE meeting_id = ? ORDER BY audio_start_time",
        )
        .bind(env("ASK_MEETING"))
        .fetch_all(&pool)
        .await
        .unwrap();
        let lines: Vec<LiveLine> = rows.into_iter().map(|(start, end, text)| LiveLine { start, end, text }).collect();
        // ASK_LIVE_FOLDER=<folder with .checkpoints/audio_chunk_*.aac> labels speakers as a live question would
        let speakers = match std::env::var("ASK_LIVE_FOLDER") {
            Ok(folder) => {
                let spans: Vec<_> = lines.iter().map(|l| (l.start, l.end)).collect();
                let started = std::time::Instant::now();
                let labels = crate::diarization::label_live_segments(
                    std::path::Path::new(&env("ASK_APP_DIR")),
                    std::path::Path::new(&folder),
                    &spans,
                )
                .await
                .unwrap();
                println!(
                    "diarized in {:.1} s: {} of {} lines labelled",
                    started.elapsed().as_secs_f64(),
                    labels.iter().filter(|l| l.is_some()).count(),
                    labels.len()
                );
                labels
            }
            Err(_) => Vec::new(),
        };
        for question in std::env::var("ASK_QUESTIONS").map(|q| q.split('|').map(String::from).collect::<Vec<_>>()).unwrap_or_else(|_| vec!["Catch me up".into(), "What are the key decisions and action items so far?".into(), "What did Priya say about the budget?".into()]) {
            let started = std::time::Instant::now();
            let first_token = std::sync::Mutex::new(None::<f64>);
            let streamed = std::sync::Mutex::new(String::new());
            let on_token = |text: &str| {
                first_token.lock().unwrap().get_or_insert(started.elapsed().as_secs_f64());
                streamed.lock().unwrap().push_str(text);
            };
            let reply = answer(&pool, Some(env("ASK_APP_DIR").into()), &question, &lines, &speakers, &on_token).await.unwrap();
            println!(
                "=== {question} ({:.1} s, first text at {:?} s, {} chars streamed)\n{reply}\n",
                started.elapsed().as_secs_f64(),
                first_token.lock().unwrap().map(|t| (t * 10.0).round() / 10.0),
                streamed.lock().unwrap().len()
            );
            assert!(!reply.is_empty());
        }
    }
}
