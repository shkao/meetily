// Nemotron 3 Diarization (int8 ONNX) on the app's ort runtime.
// Ported from spikes/nemotron-diarization/src/main.rs on enhance/diarization-spike, which ports diar.js
// (nealcaren/local-interview-transcriber@b11f004), itself a port of the offline mode of transformers'
// Nemotron3DiarizationForAudioFrameClassification. ONNX export: NealCaren/Nemotron-3-Diarization-ONNX@46642c0.

use anyhow::{Context, Result};
use ndarray::Array3;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub const SR: usize = 16000;
const N_FFT: usize = 512;
const WIN: usize = 400;
pub const HOP: usize = 160;
const N_MELS: usize = 128;
const N_BINS: usize = N_FFT / 2 + 1;
const PREEMPH: f64 = 0.97;
const SUB: usize = 8;
const HIDDEN: usize = 512;
pub const N_SPK: usize = 8;
const CHUNK_LEN: usize = 340;
const RIGHT_CTX: usize = 40;
const FIFO_LEN: usize = 40;
const UPDATE_PERIOD: usize = 300;
const CACHE_LEN: usize = 264;
const SILENCE_FRAMES: usize = 1;
const PRED_THRESH: f64 = 0.25;
const LATEST_BOOST: f64 = 0.05;
const MIN_POS_RATE: f64 = 0.5;
const STRONG_RATE: f64 = 0.75;
const WEAK_RATE: f64 = 1.5;
// 4 intra-op threads beat ONNX Runtime's default on the M4 the spike measured (119x real time).
const THREADS: usize = 4;

struct MelExtractor {
    filters: Vec<f32>, // [N_MELS * N_BINS]
    lo: Vec<usize>,
    hi: Vec<usize>,
    window: Vec<f64>,
    fft: std::sync::Arc<dyn rustfft::Fft<f64>>,
}

impl MelExtractor {
    fn new(filters: Vec<f32>) -> Self {
        // symmetric hann(400), zero-padded to 512 and centered, as torch.stft does
        let mut window = vec![0.0f64; N_FFT];
        let off = (N_FFT - WIN) / 2;
        for i in 0..WIN {
            window[off + i] = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (WIN - 1) as f64).cos();
        }
        // mel filters are sparse: remember each band's nonzero bin range
        let mut lo = vec![N_BINS; N_MELS];
        let mut hi = vec![0; N_MELS];
        for m in 0..N_MELS {
            for k in 0..N_BINS {
                if filters[m * N_BINS + k] != 0.0 {
                    lo[m] = lo[m].min(k);
                    hi[m] = k + 1;
                }
            }
        }
        let fft = FftPlanner::new().plan_fft_forward(N_FFT);
        Self { filters, lo, hi, window, fft }
    }

    // Log-mel features for STFT frames [f0, f1): [(f1 - f0) * N_MELS]. `audio` holds the samples from `base` on
    // (at least mel_span(f0, f1, l)) of a recording `l` samples long, so long files need no whole-file buffer.
    fn extract(&self, audio: &[f32], base: usize, l: usize, f0: usize, f1: usize) -> Vec<f32> {
        let (base, l) = (base as i64, l as i64);
        let a = |j: i64| audio[(j - base) as usize] as f64;
        // pre-emphasized sample j (rounded to f32, as the reference stores it), 0 outside the audio
        let x = |j: i64| -> f64 {
            if j < 0 || j >= l {
                0.0
            } else if j == 0 {
                a(0)
            } else {
                (a(j) - PREEMPH * a(j - 1)) as f32 as f64
            }
        };
        let guard = 2f64.powi(-24);
        let mut buf = vec![Complex::new(0.0f64, 0.0); N_FFT];
        let mut pow = vec![0.0f64; N_BINS];
        let mut out = vec![0.0f32; (f1 - f0) * N_MELS];
        for f in f0..f1 {
            let s = (f * HOP) as i64 - (N_FFT / 2) as i64;
            for i in 0..N_FFT {
                buf[i] = Complex::new(x(s + i as i64) * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            for k in 0..N_BINS {
                pow[k] = buf[k].norm_sqr();
            }
            let o = (f - f0) * N_MELS;
            for m in 0..N_MELS {
                let row = m * N_BINS;
                let mut acc = 0.0f64;
                for k in self.lo[m]..self.hi[m] {
                    acc += self.filters[row + k] as f64 * pow[k];
                }
                out[o + m] = (acc + guard).ln() as f32;
            }
        }
        out
    }
}

// Samples [s0, s1) that STFT frames [f0, f1) read, in a recording l samples long.
fn mel_span(f0: usize, f1: usize, l: usize) -> (usize, usize) {
    let lo = (f0 * HOP) as i64 - (N_FFT / 2) as i64 - 1;
    (lo.max(0) as usize, ((f1 - 1) * HOP + N_FFT / 2).min(l))
}

/// 16 kHz mono little-endian f32 samples in a file, read span by span so an hour of audio never sits in memory.
pub struct PcmFile {
    file: std::fs::File,
    len: usize,
}

impl PcmFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let len = file.metadata()?.len() as usize / 4;
        Ok(Self { file, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    fn read(&mut self, s0: usize, s1: usize) -> Result<Vec<f32>> {
        self.file.seek(SeekFrom::Start(s0 as u64 * 4))?;
        let mut bytes = vec![0u8; (s1 - s0) * 4];
        self.file.read_exact(&mut bytes)?;
        Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    }
}

type Emb = Vec<f32>; // [HIDDEN]
type Probs = [f32; N_SPK];

// Speaker cache for batch size 1.
struct SpeakerCache {
    silence: Emb,
    embeds: Vec<Emb>,
    probs: Vec<Probs>,
    fifo: Vec<Emb>,
    is_compressed: bool,
    min_pos: usize,
    n_strong: usize,
    n_weak: usize,
}

impl SpeakerCache {
    fn new(silence: Emb) -> Self {
        let budget = (CACHE_LEN / N_SPK - SILENCE_FRAMES) as f64;
        Self {
            silence,
            embeds: Vec::new(),
            probs: Vec::new(),
            fifo: Vec::new(),
            is_compressed: false,
            min_pos: (budget * MIN_POS_RATE).floor() as usize,
            n_strong: (budget * STRONG_RATE).floor() as usize,
            n_weak: (budget * WEAK_RATE).floor() as usize,
        }
    }

    fn get_embeds(&self) -> Vec<Emb> {
        self.embeds.iter().chain(self.fifo.iter()).cloned().collect()
    }

    // input: frames fed to the encoder, probs: pooled probs per input frame
    fn update(&mut self, input: &[Emb], probs: &[Probs], num_chunk_frames: usize) {
        let nc = self.embeds.len();
        let nf = self.fifo.len();
        let chunk_start = nc + nf;
        let mut fifo_emb: Vec<Emb> = self.fifo.clone();
        fifo_emb.extend_from_slice(&input[chunk_start..chunk_start + num_chunk_frames]);
        let lf = fifo_emb.len();
        let popped = if lf > FIFO_LEN { UPDATE_PERIOD.max(lf - FIFO_LEN).min(lf) } else { 0 };
        if popped > 0 {
            let fifo_probs = &probs[nc..nc + lf];
            let mut cache_p: Vec<Probs> =
                if self.is_compressed { self.probs.clone() } else { probs[..nc].to_vec() };
            let mut cache_e: Vec<Emb> = std::mem::take(&mut self.embeds);
            cache_e.extend(fifo_emb.drain(..popped));
            cache_p.extend_from_slice(&fifo_probs[..popped]);
            if cache_e.len() > CACHE_LEN {
                let (e, p) = self.compress(&cache_e, &cache_p);
                cache_e = e;
                cache_p = p;
                self.is_compressed = true;
            }
            self.embeds = cache_e;
            self.probs = cache_p;
        }
        self.fifo = fifo_emb;
    }

    fn frame_scores(&self, probs: &[Probs]) -> Vec<[f64; N_SPK]> {
        let log_half = 0.5f64.ln();
        let mut scores = vec![[0.0f64; N_SPK]; probs.len()];
        let mut pos_count = [0usize; N_SPK];
        for (t, p) in probs.iter().enumerate() {
            let mut lc = [0.0f64; N_SPK];
            let mut sum_lc = 0.0;
            for s in 0..N_SPK {
                lc[s] = (1.0 - p[s] as f64).max(PRED_THRESH).ln();
                sum_lc += lc[s];
            }
            for s in 0..N_SPK {
                let mut v = (p[s] as f64).max(PRED_THRESH).ln() - lc[s] + sum_lc - log_half;
                if !(p[s] > 0.5) {
                    v = f64::NEG_INFINITY;
                }
                scores[t][s] = v;
                if v > 0.0 {
                    pos_count[s] += 1;
                }
            }
        }
        for (t, p) in probs.iter().enumerate() {
            for s in 0..N_SPK {
                if !(scores[t][s] > 0.0) && p[s] > 0.5 && pos_count[s] >= self.min_pos {
                    scores[t][s] = f64::NEG_INFINITY;
                }
            }
        }
        scores
    }

    fn boost(scores: &mut [[f64; N_SPK]], k: usize, amount: f64) {
        let n = scores.len();
        for s in 0..N_SPK {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| scores[b][s].total_cmp(&scores[a][s]).then(a.cmp(&b)));
            for &i in idx.iter().take(k.min(n)) {
                scores[i][s] += amount;
            }
        }
    }

    fn compress(&self, embeds: &[Emb], probs: &[Probs]) -> (Vec<Emb>, Vec<Probs>) {
        let log_half = 0.5f64.ln();
        let n = probs.len();
        let mut scores = self.frame_scores(probs);
        for row in scores.iter_mut().skip(CACHE_LEN) {
            for v in row.iter_mut() {
                *v += LATEST_BOOST;
            }
        }
        Self::boost(&mut scores, self.n_strong, -2.0 * log_half);
        Self::boost(&mut scores, self.n_weak, -log_half);
        let n_scored = n + SILENCE_FRAMES;
        // flat index = s * n_scored + t; silence rows score +inf
        let mut flat: Vec<(usize, f64)> = Vec::with_capacity(N_SPK * n_scored);
        for s in 0..N_SPK {
            for t in 0..n_scored {
                flat.push((s * n_scored + t, if t < n { scores[t][s] } else { f64::INFINITY }));
            }
        }
        flat.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let sentinel = n_scored * N_SPK;
        let mut picked: Vec<usize> = flat
            .iter()
            .take(CACHE_LEN)
            .map(|&(i, v)| if v == f64::NEG_INFINITY { sentinel } else { i })
            .collect();
        picked.sort_unstable();
        let mut out_e = Vec::with_capacity(picked.len());
        let mut out_p = Vec::with_capacity(picked.len());
        for i in picked {
            let f = if i == sentinel { n } else { (i % n_scored).min(n) };
            if f == n {
                out_e.push(self.silence.clone());
                out_p.push([0.0; N_SPK]);
            } else {
                out_e.push(embeds[f].clone());
                out_p.push(probs[f]);
            }
        }
        (out_e, out_p)
    }
}

pub struct Diarizer {
    embed: Session,
    step: Session,
    mel: MelExtractor,
    silence: Emb,
}

fn sigmoid(x: f32) -> f64 {
    1.0 / (1.0 + (-(x as f64)).exp())
}

fn read_f32_file(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

fn load_session(path: &Path) -> Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(THREADS)?
        .commit_from_file(path)
        .with_context(|| format!("loading {}", path.display()))
}

impl Diarizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        Ok(Self {
            embed: load_session(&model_dir.join("embed.onnx"))?,
            step: load_session(&model_dir.join("step_int8.onnx"))?,
            mel: MelExtractor::new(read_f32_file(&model_dir.join("mel_filters.bin"))?),
            silence: read_f32_file(&model_dir.join("silence_embeds.bin"))?,
        })
    }

    // Embeddings [e0, e1), one per 8 mel frames. Each depends only on its own 8 frames, so computing them
    // chunk by chunk gives exactly the whole-file result.
    fn embed_range(&mut self, source: &mut PcmFile, e0: usize, e1: usize, num_frames: usize) -> Result<Vec<Emb>> {
        let f0 = e0 * SUB;
        let f1 = (e1 * SUB).min(num_frames);
        let l = source.len();
        let (s0, s1) = mel_span(f0, f1, l);
        let audio = source.read(s0, s1)?;
        let feats = self.mel.extract(&audio, s0, l, f0, f1);
        let arr = Array3::from_shape_vec((1, f1 - f0, N_MELS), feats)?;
        let outputs = self.embed.run(inputs!["features" => TensorRef::from_array_view(arr.view())?])?;
        let e = outputs.get("embeds").context("embed output 'embeds' missing")?.try_extract_array::<f32>()?;
        let e = e.as_standard_layout();
        let flat = e.as_slice().context("embeds not contiguous")?;
        Ok(flat.chunks(HIDDEN).map(|c| c.to_vec()).collect())
    }

    /// Speaker probabilities (sigmoid), [num_frames * N_SPK], one row per 10 ms. Returns (probs, num_frames).
    pub fn run(&mut self, source: &mut PcmFile) -> Result<(Vec<f32>, usize)> {
        let num_frames = source.len() / HOP;
        let n_emb = num_frames.div_ceil(SUB);
        let mut cache = SpeakerCache::new(self.silence.clone());
        let mut probs_out: Vec<f32> = Vec::with_capacity(n_emb * SUB * N_SPK);
        let mut start = 0;
        while start < n_emb {
            let end = (start + CHUNK_LEN).min(n_emb);
            let n_chunk = end - start;
            let chunk = self.embed_range(source, start, (end + RIGHT_CTX).min(n_emb), num_frames)?;
            let cached = cache.get_embeds();
            let n_cached = cached.len();
            let mut input = cached;
            input.extend(chunk);
            let t_len = input.len();
            let mut buf = Vec::with_capacity(t_len * HIDDEN);
            for e in &input {
                buf.extend_from_slice(e);
            }
            let arr = Array3::from_shape_vec((1, t_len, HIDDEN), buf)?;
            let outputs = self.step.run(inputs!["embeds" => TensorRef::from_array_view(arr.view())?])?;
            let logits = outputs.get("logits").context("step output 'logits' missing")?.try_extract_array::<f32>()?;
            let logits = logits.as_standard_layout();
            let logits = logits.as_slice().context("logits not contiguous")?; // [t_len*8, 8]
            // sigmoid + average-pool to encoder rate (accumulated in f32, as the reference's Float32Array)
            let mut pooled: Vec<Probs> = Vec::with_capacity(t_len);
            for t in 0..t_len {
                let mut p = [0.0f32; N_SPK];
                for k in 0..SUB {
                    for s in 0..N_SPK {
                        p[s] = (p[s] as f64 + sigmoid(logits[(t * SUB + k) * N_SPK + s])) as f32;
                    }
                }
                for v in p.iter_mut() {
                    *v = (*v as f64 / SUB as f64) as f32;
                }
                pooled.push(p);
            }
            cache.update(&input, &pooled, n_chunk);
            let a = n_cached * SUB * N_SPK;
            let b = (n_cached + n_chunk) * SUB * N_SPK;
            probs_out.extend(logits[a..b].iter().map(|&x| sigmoid(x) as f32));
            start = end;
        }
        probs_out.truncate(num_frames * N_SPK);
        Ok((probs_out, num_frames))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_span_covers_window_and_preemphasis() {
        // frame f reads samples [f*HOP - N_FFT/2, f*HOP + N_FFT/2), plus one earlier sample for pre-emphasis
        assert_eq!(mel_span(0, 1, 16000), (0, 256));
        assert_eq!(mel_span(10, 20, 16000), (10 * HOP - N_FFT / 2 - 1, 19 * HOP + N_FFT / 2));
        // clamped to the recording
        assert_eq!(mel_span(95, 100, 16000).1, 16000);
    }

    #[test]
    fn mel_extract_is_window_independent() {
        let l = 8000;
        let audio: Vec<f32> = (0..l).map(|i| ((i as f32) * 0.037).sin() * 0.3).collect();
        let mel = MelExtractor::new(vec![1.0 / N_BINS as f32; N_MELS * N_BINS]);
        let whole = mel.extract(&audio, 0, l, 0, l / HOP);
        for (f0, f1) in [(0, 7), (7, 23), (23, l / HOP)] {
            let (s0, s1) = mel_span(f0, f1, l);
            let part = mel.extract(&audio[s0..s1], s0, l, f0, f1);
            assert_eq!(part, whole[f0 * N_MELS..f1 * N_MELS]);
        }
    }

    #[test]
    fn speaker_cache_respects_capacity_and_order() {
        let mut cache = SpeakerCache::new(vec![-1.0; HIDDEN]);
        // tag each embedding with its arrival index; speaker (i / 97) % 3 is active
        let mut next = 0usize;
        for _ in 0..8 {
            let mut input = cache.get_embeds();
            for _ in 0..CHUNK_LEN + RIGHT_CTX {
                input.push(vec![next as f32; HIDDEN]);
                next += 1;
            }
            next -= RIGHT_CTX; // right context frames open the next chunk
            let probs: Vec<Probs> = input
                .iter()
                .map(|e| {
                    let mut p = [0.02f32; N_SPK];
                    if e[0] >= 0.0 {
                        p[(e[0] as usize / 97) % 3] = 0.95;
                    }
                    p
                })
                .collect();
            cache.update(&input, &probs, CHUNK_LEN);
            assert!(cache.embeds.len() <= CACHE_LEN, "cache {} > {}", cache.embeds.len(), CACHE_LEN);
            assert!(cache.fifo.len() <= FIFO_LEN.max(UPDATE_PERIOD), "fifo {}", cache.fifo.len());
            assert_eq!(cache.embeds.len(), cache.probs.len());
        }
        assert!(cache.is_compressed);
        // all three active speakers keep frames in the compressed cache
        let kept: std::collections::BTreeSet<usize> =
            cache.embeds.iter().filter(|e| e[0] >= 0.0).map(|e| (e[0] as usize / 97) % 3).collect();
        assert_eq!(kept.len(), 3);
    }
}
