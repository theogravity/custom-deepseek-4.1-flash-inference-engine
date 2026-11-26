mod api;
mod attention;
mod chat;
mod draft;
mod draft_attention;
mod engram;
mod model;
mod moe;
mod prefix;
mod runtime;
mod swap;
mod weights;

use anyhow::{anyhow, ensure, Context, Result};
use clap::{Parser, Subcommand};
use model::Model;
use serde::Deserialize;
use std::{path::PathBuf, process::Command, time::Instant};
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(about = "Standalone DeepSeek V4.1 Flash EXL3 text inference")]
struct Args {
    #[arg(long, global = true, default_value = "/mnt/models/dsv41-flash-exl3")]
    model_dir: PathBuf,
    #[arg(long, global = true, default_value_t = 16384)]
    max_context: usize,
    #[arg(long, global = true, default_value_t = 32)]
    chunk_size: usize,
    /// Choose the checkpoint's original or code-tuned DSpark drafter.
    #[arg(long, global = true, value_parser = ["none","released","code"])]
    drafter: Option<String>,
    /// Number of speculative tokens proposed per target verification.
    #[arg(long, global = true)]
    spec_k: Option<usize>,
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    /// Generate a greedy continuation from a raw text prompt.
    Infer {
        prompt: String,
        #[arg(long, default_value_t = 16)]
        max_tokens: usize,
    },
    /// Measure first-token latency and steady single-stream decode on a repeatable raw prompt.
    Bench {
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,
        #[arg(long, default_value_t = 6)]
        samples: usize,
        #[arg(default_value = "The first twenty prime numbers are 2, 3, 5, 7,")]
        prompt: String,
    },
    /// Compare greedy output IDs to saved tokens from the serving recipe.
    Verify {
        #[arg(default_value = "tests/reference.json")]
        fixture: PathBuf,
    },
    /// Serve Anthropic and OpenAI APIs. Requires DSV41_API_KEY or LITELLM_MASTER_KEY.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8001")]
        bind: String,
    },
}
#[derive(Deserialize)]
struct Fixture {
    prompt: String,
    ids: Vec<u32>,
    output_ids: Vec<u32>,
}
fn select_cards() -> Result<()> {
    if let Ok(value) = std::env::var("CUDA_VISIBLE_DEVICES") {
        ensure!(
            value.split(',').count() == 2 && !value.split(',').any(|s| s.trim().is_empty()),
            "set CUDA_VISIBLE_DEVICES to the two inference GPU UUIDs"
        );
    } else {
        let q = Command::new("nvidia-smi")
            .args(["--query-gpu=name,uuid", "--format=csv,noheader"])
            .output()?;
        ensure!(q.status.success(), "nvidia-smi GPU inventory failed");
        let cards: Vec<String> = std::str::from_utf8(&q.stdout)?
            .lines()
            .filter_map(|l| l.split_once(','))
            .filter(|(n, _)| n.contains("RTX PRO 6000"))
            .map(|(_, u)| u.trim().to_owned())
            .collect();
        ensure!(
            cards.len() == 2,
            "set CUDA_VISIBLE_DEVICES to two inference GPU UUIDs"
        );
        std::env::set_var("CUDA_VISIBLE_DEVICES", cards.join(","));
    }
    Ok(())
}
fn main() -> Result<()> {
    use std::io::Write;
    let args = Args::parse();
    if let Some(name) = &args.drafter {
        std::env::set_var("DS4_DRAFTER", name);
    }
    if let Some(k) = args.spec_k {
        std::env::set_var("DS4_SPEC_K", k.to_string());
    }
    let key = if matches!(args.command, Action::Serve { .. }) {
        let key = std::env::var("DSV41_API_KEY")
            .or_else(|_| std::env::var("LITELLM_MASTER_KEY"))
            .context("server API key required")?;
        ensure!(!key.trim().is_empty(), "server API key is empty");
        key
    } else {
        String::new()
    };
    select_cards()?;
    let tok =
        Tokenizer::from_file(args.model_dir.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
    let t = Instant::now();
    let mut model = Model::open(
        &args.model_dir,
        &tok,
        [0, 1],
        args.max_context,
        args.chunk_size,
    )?;
    eprintln!(
        "Opened checkpoint in {:.2}s; weights upload on demand",
        t.elapsed().as_secs_f64()
    );
    match args.command {
        Action::Serve { bind } => api::serve(model, tok, &bind, key),
        Action::Infer { prompt, max_tokens } => {
            let input = tok.encode(prompt, false).map_err(|e| anyhow!("{e}"))?;
            let mut ids = Vec::new();
            let start = Instant::now();
            let n = model.generate(input.get_ids(), max_tokens, 0., |id| {
                ids.push(id);
                true
            })?;
            let text = tok.decode(&ids, false).map_err(|e| anyhow!("{e}"))?;
            writeln!(std::io::stdout(), "{text}")?;
            eprintln!(
                "{n} output tokens in {:.3}s including prefill",
                start.elapsed().as_secs_f64()
            );
            Ok(())
        }
        Action::Bench {
            prompt,
            max_tokens,
            samples,
        } => {
            ensure!(
                (1..=100).contains(&samples) && max_tokens >= 2,
                "bench needs 1–100 samples and at least two tokens per sample"
            );
            let ids = tok
                .encode(prompt.as_str(), false)
                .map_err(|e| anyhow!("{e}"))?;
            let mut rates = vec![];
            let mut records = vec![];
            let mut baseline = vec![];
            for run in 0..=samples {
                let start = Instant::now();
                let mut tokens = vec![];
                let mut times = vec![];
                model.generate(ids.get_ids(), max_tokens, 0., |id| {
                    tokens.push(id);
                    times.push(Instant::now());
                    true
                })?;
                ensure!(times.len() >= 2, "model emitted fewer than two tokens");
                if run == 0 {
                    baseline = tokens.clone();
                } else {
                    ensure!(tokens == baseline, "greedy output changed across runs");
                }
                let rate = (times.len() - 1) as f64
                    / times.last().unwrap().duration_since(times[0]).as_secs_f64();
                let ttft = times[0].duration_since(start).as_secs_f64();
                if run > 0 {
                    rates.push(rate);
                }
                records.push(serde_json::json!({"warmup":run==0,"visible_tokens":times.len(),"ttft_s":ttft,"decode_tok_s":rate}));
            }
            rates.sort_by(f64::total_cmp);
            let median = (rates[(rates.len() - 1) / 2] + rates[rates.len() / 2]) / 2.;
            let continuation = tok.decode(&baseline, false).map_err(|e| anyhow!("{e}"))?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"prompt":prompt,"prompt_tokens":ids.len(),"continuation":continuation,"runs":records,"median_decode_tok_s":median})
                )?
            );
            Ok(())
        }
        Action::Verify { fixture } => {
            let fixtures: Vec<Fixture> = serde_json::from_reader(std::fs::File::open(fixture)?)?;
            ensure!(!fixtures.is_empty(), "empty fixture");
            let mut matches = 0;
            for f in &fixtures {
                let encoded = tok
                    .encode(f.prompt.as_str(), false)
                    .map_err(|e| anyhow!("{e}"))?;
                ensure!(
                    encoded.get_ids() == f.ids,
                    "tokenizer disagrees on {:?}",
                    f.prompt
                );
                let mut got = vec![];
                let mut times = vec![];
                let start = Instant::now();
                let n = model.generate(&f.ids, f.output_ids.len(), 0., |id| {
                    got.push(id);
                    times.push(Instant::now());
                    true
                })?;
                if n > got.len() {
                    got.push(1);
                }
                let elapsed = start.elapsed().as_secs_f64();
                let speed = if times.len() > 1 {
                    (times.len() - 1) as f64
                        / times.last().unwrap().duration_since(times[0]).as_secs_f64()
                } else {
                    0.
                };
                println!(
                    "{:?} expected {:?} generated {:?} ({elapsed:.2}s, {speed:.2} decode tok/s)",
                    f.prompt, f.output_ids, got
                );
                matches += usize::from(got == f.output_ids);
            }
            ensure!(
                matches == fixtures.len(),
                "{matches}/{} exact continuations",
                fixtures.len()
            );
            Ok(())
        }
    }
}
