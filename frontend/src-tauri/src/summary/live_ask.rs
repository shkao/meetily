//! Ask AI during a meeting, modelled on Google Meet's "Ask Gemini".
//!
//! The panel sends the live transcript it already holds (it is not in the database until the meeting is saved)
//! and a question. The answer comes from the provider picked for summaries, so the transcript never goes to a
//! provider the user has not chosen. Transcripts that fit the model's context go in whole; longer ones are split
//! with the summary chunker, notes relevant to the question are taken from each part, and the answer comes from
//! those notes. Live transcription is unlabelled, so questions about who said what get a plain "can't tell".

use crate::database::repositories::setting::SettingsRepository;
use crate::summary::llm_client::generate_summary;
use crate::summary::processor::{chunk_text, clean_llm_markdown_detailed, rough_token_count};
use crate::summary::service::{LlmSettings, SummaryService};
use log::info;
use reqwest::Client;
use serde::Deserialize;
use tauri::{AppHandle, Manager, Runtime};

/// One live transcript segment as the panel holds it.
#[derive(Debug, Deserialize)]
pub struct LiveLine {
    /// Seconds from recording start
    pub start: Option<f64>,
    pub text: String,
}

const SYSTEM_PROMPT: &str = "You answer questions about a meeting that is still in progress, using only its live \
transcript. Each line starts with its time as [mm:ss]. Rules:
- Answer briefly and directly. Use short bullet points for lists.
- Cite the time of every fact you use, exactly as it appears, for example [03:15]. Never invent times.
- The transcript has no speaker names. If the question asks who said something or what a named person said, say \
that speaker names aren't available during the meeting, then answer what you can without attributing it.
- If the transcript doesn't cover the question, say so. Don't guess.
- Answer in the language of the question.";

const NOTES_PROMPT: &str = "You read one part of a live meeting transcript. Each line starts with its time as \
[mm:ss]. Copy out, as short bullet points, only what is relevant to the question, keeping each point's [mm:ss] \
time. If nothing is relevant, reply with exactly: NONE";

fn format_time(seconds: f64) -> String {
    let secs = seconds.max(0.0) as u64;
    format!("[{:02}:{:02}]", secs / 60, secs % 60)
}

/// Transcript lines as `[mm:ss] text`, skipping empty segments.
fn format_transcript(lines: &[LiveLine]) -> String {
    lines
        .iter()
        .filter(|l| !l.text.trim().is_empty())
        .map(|l| match l.start {
            Some(s) => format!("{} {}", format_time(s), l.text.trim()),
            None => l.text.trim().to_string(),
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

/// Answers a question from the live transcript with the summary provider.
#[tauri::command]
pub async fn ask_ai_live<R: Runtime>(
    app: AppHandle<R>,
    question: String,
    lines: Vec<LiveLine>,
) -> Result<String, String> {
    let pool = app.state::<crate::state::AppState>().db_manager.pool().clone();
    answer(&pool, app.path().app_data_dir().ok(), &question, &lines).await
}

async fn answer(
    pool: &sqlx::SqlitePool,
    app_data_dir: Option<std::path::PathBuf>,
    question: &str,
    lines: &[LiveLine],
) -> Result<String, String> {
    let question = question.trim();
    if question.is_empty() {
        return Err("Type a question first".to_string());
    }
    let transcript = format_transcript(lines);
    if transcript.is_empty() {
        return Ok("Nothing has been transcribed yet.".to_string());
    }

    let config = SettingsRepository::get_model_config(pool)
        .await
        .map_err(|e| format!("Failed to read AI model settings: {}", e))?
        .ok_or_else(|| "Choose an AI model in Settings first".to_string())?;
    let settings = SummaryService::resolve_llm_settings(pool, &config.provider, &config.model).await?;
    let client = Client::new();

    let budget = settings.token_threshold.saturating_sub(rough_token_count(SYSTEM_PROMPT) + rough_token_count(question));
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

    complete(&client, &settings, &config.model, app_data_dir.as_ref(), SYSTEM_PROMPT, &question_prompt(&context, question)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_lines_with_times_and_skips_empty() {
        let lines = vec![
            LiveLine { start: Some(5.4), text: " hello ".into() },
            LiveLine { start: Some(9.0), text: "  ".into() },
            LiveLine { start: Some(125.0), text: "next item".into() },
            LiveLine { start: None, text: "untimed".into() },
        ];
        assert_eq!(format_transcript(&lines), "[00:05] hello\n[02:05] next item\nuntimed");
    }

    #[test]
    fn prompt_asks_for_times_and_admits_missing_speakers() {
        assert!(SYSTEM_PROMPT.contains("[03:15]"));
        assert!(SYSTEM_PROMPT.contains("speaker names aren't available"));
    }

    /// Asks the configured summary provider about a saved meeting's transcript, as the live panel would.
    /// ASK_DB=<sqlite> ASK_MEETING=<id> ASK_APP_DIR=<app data dir> cargo test --lib live_ask -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn answers_from_a_real_transcript() {
        let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"));
        let pool = sqlx::SqlitePool::connect(&format!("sqlite://{}?mode=ro", env("ASK_DB"))).await.unwrap();
        let rows: Vec<(Option<f64>, String)> = sqlx::query_as(
            "SELECT audio_start_time, transcript FROM transcripts WHERE meeting_id = ? ORDER BY audio_start_time",
        )
        .bind(env("ASK_MEETING"))
        .fetch_all(&pool)
        .await
        .unwrap();
        let lines: Vec<LiveLine> = rows.into_iter().map(|(start, text)| LiveLine { start, text }).collect();
        for question in ["Catch me up", "What are the key decisions and action items so far?", "What did Priya say about the budget?"] {
            let started = std::time::Instant::now();
            let reply = answer(&pool, Some(env("ASK_APP_DIR").into()), question, &lines).await.unwrap();
            println!("=== {question} ({:.1} s)\n{reply}\n", started.elapsed().as_secs_f64());
            assert!(!reply.is_empty());
        }
    }
}
