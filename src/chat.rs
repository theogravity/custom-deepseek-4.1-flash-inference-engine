//! DeepSeek V4.1 prompt rendering. Keep text and template control tokens
//! separate: user text spelling a role marker must encode as ordinary BPE.
use anyhow::{anyhow, bail, ensure, Context, Result};
use serde_json::{json, Map, Value};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};
use tokenizers::Tokenizer;

type EncodedText = (Box<str>, Arc<[u32]>);

/// Belongs to one serving tokenizer. Only complete ordinary-text spans enter this cache.
#[derive(Default)]
pub struct TextCache {
    entries: Mutex<VecDeque<EncodedText>>,
}

impl TextCache {
    const BUDGET: usize = 32 * 1024 * 1024;
    fn hit(&self, text: &str) -> Option<Arc<[u32]>> {
        let mut entries = self.entries.lock().unwrap();
        let index = entries.iter().rposition(|(key, _)| key.as_ref() == text)?;
        let pair = entries.remove(index)?;
        let ids = Arc::clone(&pair.1);
        entries.push_back(pair);
        Some(ids)
    }

    fn insert(&self, text: &str, ids: &[u32]) {
        let bytes = text.len() + std::mem::size_of_val(ids);
        if bytes > Self::BUDGET {
            return;
        }
        let mut entries = self.entries.lock().unwrap();
        if entries.iter().any(|(key, _)| key.as_ref() == text) {
            return;
        }
        let mut used = entries
            .iter()
            .map(|(k, v)| k.len() + std::mem::size_of_val(v.as_ref()))
            .sum::<usize>();
        while used + bytes > Self::BUDGET || entries.len() >= 64 {
            let (key, value) = entries.pop_front().unwrap();
            used -= key.len() + std::mem::size_of_val(value.as_ref());
        }
        entries.push_back((text.into(), Arc::from(ids)));
    }
}

const BOS: &str = "<｜begin▁of▁sentence｜>";
const EOS: &str = "<｜end▁of▁sentence｜>";
const USER: &str = "<｜User｜>";
const SYSTEM: &str = "<｜System｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";
const DSML: &str = "｜DSML｜";
const CALLS: &str = "<｜DSML｜ calls>";
const CALLS_END: &str = "</｜DSML｜ calls>";
const INVOKE: &str = "<｜DSML｜ invoke name=\"";
const INVOKE_END: &str = "</｜DSML｜ invoke>";
const PARAM: &str = "<｜DSML｜ parameter name=\"";
const PARAM_END: &str = "</｜DSML｜ parameter>";

#[derive(Default)]
pub struct Prompt(Vec<(String, bool)>);

impl Prompt {
    fn text(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        if let Some((text, false)) = self.0.last_mut() {
            text.push_str(s)
        } else {
            self.0.push((s.into(), false))
        }
    }
    fn control(&mut self, s: &str) {
        self.0.push((s.into(), true));
    }
    // Use only on constant template fragments, never on user-provided data.
    fn template(&mut self, mut s: &str) {
        while let Some((pos, token)) = [DSML, OPEN, CLOSE]
            .iter()
            .filter_map(|t| s.find(t).map(|i| (i, *t)))
            .min_by_key(|x| x.0)
        {
            self.text(&s[..pos]);
            self.control(token);
            s = &s[pos + token.len()..];
        }
        self.text(s)
    }
    pub fn encode(
        &self,
        full: &Tokenizer,
        plain: &Tokenizer,
        cache: Option<&TextCache>,
    ) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for (s, control) in &self.0 {
            if *control {
                ids.push(
                    full.token_to_id(s)
                        .with_context(|| format!("missing checkpoint token {s}"))?,
                );
            } else {
                let memo = cache.filter(|_| (512..=TextCache::BUDGET).contains(&s.len()));
                if let Some(hit) = memo.and_then(|m| m.hit(s)) {
                    ids.extend_from_slice(&hit);
                    continue;
                }
                let encoded = plain
                    .encode(s.as_str(), false)
                    .map_err(|e| anyhow!("{e}"))?;
                ids.extend_from_slice(encoded.get_ids());
                if let Some(m) = memo {
                    m.insert(s, encoded.get_ids());
                }
            }
        }
        Ok(ids)
    }
}

// Match Python json.dumps(ensure_ascii=False) from the checkpoint template.
fn schema_json(v: &Value) -> String {
    match v {
        Value::Object(fields) => format!(
            "{{{}}}",
            fields
                .iter()
                .map(|(k, v)| format!("{}: {}", json!(k), schema_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(schema_json).collect::<Vec<_>>().join(", ")
        ),
        other => other.to_string(),
    }
}

fn array<'a>(v: &'a Value, field: &str) -> Result<&'a [Value]> {
    v.as_array()
        .map(Vec::as_slice)
        .with_context(|| format!("{field} must be an array"))
}
fn string<'a>(v: &'a Value, field: &str) -> Result<&'a str> {
    v.as_str()
        .with_context(|| format!("{field} must be a string"))
}

fn text_content(v: &Value) -> Result<String> {
    if let Some(text) = v.as_str() {
        return Ok(text.into());
    }
    array(v, "content")?
        .iter()
        .map(|b| {
            ensure!(
                matches!(
                    b["type"].as_str(),
                    Some("text" | "input_text" | "output_text")
                ),
                "unsupported content type {}",
                b["type"]
            );
            Ok(string(&b["text"], "text")?.to_owned())
        })
        .collect::<Result<Vec<_>>>()
        .map(|parts| parts.join("\n\n"))
}

#[derive(Debug, PartialEq)]
pub struct Call {
    pub name: String,
    pub args: Value,
}
impl Call {
    fn from_history(name: &Value, args: &Value) -> Result<Self> {
        let mut args = args.clone();
        for _ in 0..2 {
            if let Value::String(s) = &args {
                args = serde_json::from_str(s).context("invalid JSON tool arguments")?
            }
        }
        ensure!(args.is_object(), "tool input must be an object");
        Ok(Self {
            name: string(name, "tool name")?.to_owned(),
            args,
        })
    }
    fn render(&self, p: &mut Prompt) {
        p.template(INVOKE);
        p.text(&self.name);
        p.text("\">\n");
        for (index, (name, val)) in self.args.as_object().unwrap().iter().enumerate() {
            if index > 0 {
                p.text("\n");
            }
            p.template(PARAM);
            p.text(name);
            if let Some(s) = val.as_str() {
                p.text("\" string=\"true\">");
                p.text(s);
            } else {
                p.text("\" string=\"false\">");
                p.text(&schema_json(val));
            }
            p.template(PARAM_END);
        }
        p.text("\n");
        p.template(INVOKE_END);
    }
}
enum Block {
    Text(String),
    Tool { id: String, text: String },
}
enum Message {
    System(String),
    User(Vec<Block>),
    Assistant {
        reason: String,
        text: String,
        calls: Vec<(String, Call)>,
    },
}
fn append_user(messages: &mut Vec<Message>, blocks: Vec<Block>) {
    if let Some(Message::User(previous)) = messages.last_mut() {
        previous.extend(blocks)
    } else {
        messages.push(Message::User(blocks))
    }
}
fn append_system(messages: &mut Vec<Message>, content: &Value) -> Result<()> {
    let text = text_content(content)?;
    let text = if text.starts_with("x-anthropic-billing-header") {
        [text.find('\n'), text.find("You are")]
            .into_iter()
            .flatten()
            .min()
            .map(|i| text[i..].trim_start())
            .unwrap_or("")
    } else {
        &text
    };
    if !text.is_empty() {
        messages.push(Message::System(text.into()));
    }
    Ok(())
}
fn normalize(req: &Value) -> Result<Vec<Message>> {
    let mut messages = Vec::new();
    if !req["system"].is_null() {
        append_system(&mut messages, &req["system"])?
    }
    for msg in array(&req["messages"], "messages")? {
        let c = &msg["content"];
        match string(&msg["role"], "role")? {
            "system" | "developer" => append_system(&mut messages, c)?,
            "user" => {
                let mut blocks = Vec::new();
                if let Some(s) = c.as_str() {
                    blocks.push(Block::Text(s.into()))
                } else {
                    for b in array(c, "user content")? {
                        let block = match b["type"].as_str() {
                            Some("text" | "input_text") => {
                                Block::Text(string(&b["text"], "text")?.into())
                            }
                            Some("tool_result") => Block::Tool {
                                id: string(&b["tool_use_id"], "tool_use_id")?.into(),
                                text: if b["content"].is_null() {
                                    String::new()
                                } else {
                                    text_content(&b["content"])?
                                },
                            },
                            _ => bail!("unsupported user content type {}", b["type"]),
                        };
                        blocks.push(block);
                    }
                }
                append_user(&mut messages, blocks);
            }
            "tool" => append_user(
                &mut messages,
                vec![Block::Tool {
                    id: msg["tool_call_id"].as_str().unwrap_or("").into(),
                    text: text_content(c)?,
                }],
            ),
            "assistant" => {
                let mut reason = msg["reasoning_content"]
                    .as_str()
                    .or(msg["reasoning"].as_str())
                    .unwrap_or("")
                    .to_owned();
                let mut calls = Vec::new();
                let mut texts = Vec::new();
                if let Some(s) = c.as_str() {
                    texts.push(s)
                } else if !c.is_null() {
                    for b in array(c, "assistant content")? {
                        match b["type"].as_str() {
                            Some("text" | "output_text") => texts.push(string(&b["text"], "text")?),
                            Some("thinking") => {
                                reason.push_str(string(&b["thinking"], "thinking")?)
                            }
                            Some("tool_use") => calls.push((
                                string(&b["id"], "tool_use id")?.into(),
                                Call::from_history(&b["name"], &b["input"])?,
                            )),
                            _ => bail!("unsupported assistant content type {}", b["type"]),
                        }
                    }
                }
                if !msg["tool_calls"].is_null() {
                    for call in array(&msg["tool_calls"], "tool_calls")? {
                        let fun = &call["function"];
                        calls.push((
                            call["id"].as_str().unwrap_or("").into(),
                            Call::from_history(&fun["name"], &fun["arguments"])?,
                        ));
                    }
                }
                messages.push(Message::Assistant {
                    reason,
                    text: texts.join("\n\n"),
                    calls,
                });
            }
            role => bail!("unsupported role {role}"),
        }
    }
    ensure!(
        !messages.is_empty(),
        "messages must have at least one entry"
    );
    Ok(messages)
}
fn tool_schemas(req: &Value) -> Result<Vec<String>> {
    let choice = &req["tool_choice"];
    if req["tools"].is_null() || choice == "none" || choice["type"] == "none" {
        return Ok(vec![]);
    }
    // The backend has no constrained tool sampler. Refuse forced selection
    // instead of silently returning unconstrained text or a different call.
    ensure!(
        choice.is_null() || choice == "auto" || choice["type"] == "auto",
        "only auto and none tool_choice are supported"
    );
    array(&req["tools"], "tools")?
        .iter()
        .map(|tool| {
            let fun = tool.get("function").unwrap_or(tool);
            let name = string(&fun["name"], "tool name")?;
            let desc = fun["description"].as_str().unwrap_or("");
            let params = fun
                .get("parameters")
                .or(fun.get("input_schema"))
                .cloned()
                .unwrap_or_else(|| json!({}));
            ensure!(params.is_object(), "tool parameters must be an object");
            Ok(schema_json(
                &json!({"name":name,"description":desc,"parameters":params}),
            ))
        })
        .collect()
}

pub fn render(req: &Value) -> Result<(Prompt, bool, bool)> {
    let args = &req["chat_template_kwargs"];
    let toggle = args
        .get("thinking")
        .or(args.get("enable_thinking"))
        .and_then(Value::as_bool);
    let effort = args
        .get("reasoning_effort")
        .or(req.get("reasoning_effort"))
        .or(req["output_config"].get("effort"));
    let thinking =
        toggle.unwrap_or(req["thinking"]["type"] != "disabled") && effort != Some(&json!("none"));
    let budget = match effort {
        None | Some(Value::Null) => 75,
        Some(v) => match v.as_str() {
            Some("minimal" | "low") => 50,
            Some("medium" | "high" | "xhigh" | "none") => 75,
            Some("max" | "ultra") => 100,
            _ => v
                .as_u64()
                .filter(|n| (1..=100).contains(n))
                .context("unsupported reasoning effort")?,
        },
    };
    let tools = tool_schemas(req)?;
    let response_format = match req["response_format"]["type"].as_str() {
        Some("json_object") => Some(json!({"type":"json_object"})),
        Some("json_schema") => Some(
            req["response_format"]["json_schema"]
                .get("schema")
                .context("missing response schema")?
                .clone(),
        ),
        None | Some("text") => None,
        _ => bail!("unsupported response format"),
    };
    let mut messages = normalize(req)?;
    if (!tools.is_empty() || response_format.is_some())
        && !matches!(messages.first(), Some(Message::System(_)))
    {
        messages.insert(0, Message::System(String::new()));
    }
    let keep_all = !tools.is_empty() || args["drop_thinking"] == false;
    let last_user = messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(i, m)| match m {
            Message::User(blocks) if blocks.iter().any(|b| matches!(b, Block::Text(_))) => Some(i),
            Message::System(_) if i > 0 => Some(i),
            _ => None,
        });
    let mut p = Prompt::default();
    p.control(BOS);
    let mut call_ids: Vec<&str> = vec![];
    for (i, m) in messages.iter().enumerate() {
        if i == 0 && (thinking || matches!(m, Message::System(_))) {
            p.control(SYSTEM)
        }
        if i == 0 && thinking {
            p.text(&format!("Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"));
        }
        let keep = thinking && (keep_all || last_user.is_none_or(|u| i >= u));
        match m {
            Message::System(s) => {
                if i > 0 {
                    p.control(SYSTEM);
                }
                p.text(s);
                if i == 0 && !tools.is_empty() {
                    let (start, end) = include_str!("tool_prompt.txt")
                        .split_once("{SCHEMAS}")
                        .unwrap();
                    p.text("\n\n");
                    p.template(start);
                    p.text(&tools.join("\n"));
                    p.template(end);
                }
                if i == 0 {
                    if let Some(schema) = &response_format {
                        p.text("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
                        p.text(&schema_json(schema));
                    }
                }
            }
            Message::User(blocks) => {
                p.control(USER);
                let mut results: Vec<_> = blocks
                    .iter()
                    .filter_map(|b| match b {
                        Block::Tool { id, text } => Some((id, text)),
                        _ => None,
                    })
                    .collect();
                results.sort_by_key(|(id, _)| call_ids.iter().position(|c| c == id).unwrap_or(0));
                let mut sorted = results.into_iter();
                for (index, b) in blocks.iter().enumerate() {
                    if index > 0 {
                        p.text("\n\n")
                    }
                    match b {
                        Block::Text(s) => p.text(s),
                        Block::Tool { .. } => {
                            p.text("<tool_result>");
                            p.text(sorted.next().unwrap().1);
                            p.text("</tool_result>");
                        }
                    }
                }
            }
            Message::Assistant {
                reason,
                text,
                calls,
            } => {
                if keep {
                    p.text(reason);
                    p.control(CLOSE)
                }
                p.text(text);
                if !calls.is_empty() {
                    call_ids = calls.iter().map(|(id, _)| id.as_str()).collect();
                    p.text("\n\n");
                    p.template(CALLS);
                    p.text("\n");
                    for (_, call) in calls {
                        call.render(&mut p);
                        p.text("\n");
                    }
                    p.template(CALLS_END);
                }
                p.control(EOS);
            }
        }
        if (matches!(m, Message::User(_)) || (i > 0 && matches!(m, Message::System(_))))
            && matches!(messages.get(i + 1), None | Some(Message::Assistant { .. }))
        {
            p.control(ASSISTANT);
            p.control(if keep { OPEN } else { CLOSE });
        }
    }
    // Assistant-prefill continuations are unsupported. An ordinary completed
    // history still renders faithfully for token counting and golden checks.
    Ok((p, thinking, !tools.is_empty()))
}

#[derive(Debug, PartialEq)]
pub enum Delta {
    Reason(String),
    Text(String),
    Tool(Call),
}
enum Phase {
    Reason,
    Text,
    Tools,
    Done,
}
pub struct Parser {
    phase: Phase,
    tools: bool,
    buf: String,
    tool_body: String,
    pub stop: Option<String>,
    stops: Vec<String>,
}

fn partial_marker_len(buf: &str, marker: &str) -> usize {
    marker
        .char_indices()
        .map(|(n, _)| n)
        .rev()
        .find(|n| buf.ends_with(&marker[..*n]))
        .unwrap_or(0)
}
fn parse_calls(mut s: &str) -> Result<Vec<Delta>> {
    let mut calls = vec![];
    s = s.trim_start();
    while !s.is_empty() {
        s = s.strip_prefix(INVOKE).context("expected DSML invoke")?;
        let (name, mut body) = s.split_once("\">").context("unterminated invoke header")?;
        let mut args = Map::new();
        loop {
            body = body.trim_start();
            if let Some(tail) = body.strip_prefix(INVOKE_END) {
                s = tail.trim_start();
                break;
            }
            let header = body
                .strip_prefix(PARAM)
                .context("expected DSML parameter")?;
            let (key, value_header) = header
                .split_once("\" string=\"")
                .context("invalid DSML parameter header")?;
            let (string_flag, value_body) = value_header
                .split_once("\">")
                .context("unterminated parameter header")?;
            let (value, rest) = value_body
                .split_once(PARAM_END)
                .context("unterminated DSML parameter")?;
            let value = match string_flag {
                "true" => Value::String(value.into()),
                "false" => serde_json::from_str(value).context("invalid DSML JSON value")?,
                _ => bail!("invalid DSML string flag"),
            };
            ensure!(
                args.insert(key.into(), value).is_none(),
                "duplicate DSML parameter {key}"
            );
            body = rest;
        }
        ensure!(!name.is_empty(), "empty tool name");
        calls.push(Delta::Tool(Call {
            name: name.into(),
            args: Value::Object(args),
        }));
    }
    ensure!(!calls.is_empty(), "empty DSML calls block");
    Ok(calls)
}

impl Parser {
    pub fn new(thinking: bool, tools: bool, stops: Vec<String>) -> Self {
        Self {
            phase: if thinking { Phase::Reason } else { Phase::Text },
            tools,
            buf: String::new(),
            tool_body: String::new(),
            stop: None,
            stops,
        }
    }
    pub fn push(&mut self, text: &str, end: bool) -> Result<Vec<Delta>> {
        self.buf.push_str(text);
        let mut events = Vec::new();
        loop {
            let controls: &[&str] = match self.phase {
                Phase::Reason => &[CLOSE],
                Phase::Text if self.tools => &["\n\n<｜DSML｜ calls>", CALLS],
                Phase::Text => &[],
                Phase::Tools => &[CALLS_END],
                Phase::Done => break,
            };
            let control = controls
                .iter()
                .filter_map(|x| self.buf.find(x).map(|i| (i, *x)))
                .min_by_key(|x| x.0);
            let stop = self
                .stops
                .iter()
                .filter_map(|x| self.buf.find(x).map(|i| (i, x)))
                .min_by_key(|x| x.0);
            let stopping = stop
                .as_ref()
                .is_some_and(|(i, _)| control.as_ref().is_none_or(|(j, _)| i <= j));
            let marker = if stopping {
                stop.map(|(i, s)| (i, s.as_str()))
            } else {
                control
            };
            let split = marker.map(|(i, _)| i).unwrap_or_else(|| {
                if end {
                    self.buf.len()
                } else {
                    let retained = controls
                        .iter()
                        .copied()
                        .chain(self.stops.iter().map(String::as_str))
                        .map(|s| partial_marker_len(&self.buf, s))
                        .max()
                        .unwrap_or(0);
                    self.buf.len() - retained
                }
            });
            let consumed = marker.map_or(split, |(_, s)| split + s.len());
            let before = &self.buf[..split];
            if !before.is_empty() {
                match self.phase {
                    Phase::Reason => events.push(Delta::Reason(before.into())),
                    Phase::Text => events.push(Delta::Text(before.into())),
                    Phase::Tools => self.tool_body.push_str(before),
                    _ => {}
                }
            }
            if matches!(self.phase, Phase::Tools) && marker.is_some() && !stopping {
                events.extend(parse_calls(&self.tool_body)?);
                self.tool_body.clear();
            }
            if stopping {
                self.stop = marker.map(|(_, s)| s.to_owned())
            }
            self.buf.drain(..consumed);
            if marker.is_none() {
                break;
            }
            self.phase = if stopping {
                Phase::Done
            } else {
                match self.phase {
                    Phase::Reason => Phase::Text,
                    Phase::Text => Phase::Tools,
                    Phase::Tools => Phase::Text,
                    Phase::Done => unreachable!(),
                }
            };
        }
        Ok(events)
    }
    pub fn incomplete_call(&self) -> bool {
        matches!(self.phase, Phase::Tools)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_markers_unicode_and_tool_json() {
        let mut parser = Parser::new(true, true, vec!["STOP".into()]);
        let mut reason = String::new();
        let mut answer = String::new();
        let mut calls = vec![];
        let output = "推論</think>Here\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"a\" string=\"false\">[3, true]</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"s\" string=\"true\"> &<>\n</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>STOP and more";
        for ch in output.chars() {
            for event in parser.push(&ch.to_string(), false).unwrap() {
                match event {
                    Delta::Reason(s) => reason += &s,
                    Delta::Text(s) => answer += &s,
                    Delta::Tool(c) => calls.push(c),
                }
            }
        }
        assert_eq!(reason, "推論");
        assert_eq!(answer, "Here");
        assert_eq!(
            calls,
            [Call {
                name: "lookup".into(),
                args: json!({"a":[3,true],"s":" &<>\n"})
            }]
        );
        assert_eq!(parser.stop.as_deref(), Some("STOP"));
        assert!(parser.push("", true).unwrap().is_empty());
    }

    #[test]
    fn tool_results_are_rendered_in_call_order() {
        let request = json!({"messages":[{"role":"user","content":"question"},{"role":"assistant","content":[
            {"type":"thinking","thinking":"which tools?"},{"type":"tool_use","id":"first","name":"a","input":{}},
            {"type":"tool_use","id":"second","name":"b","input":{}}]},{"role":"user","content":[
            {"type":"tool_result","tool_use_id":"second","content":"B"},{"type":"text","text":"and"},
            {"type":"tool_result","tool_use_id":"first","content":[{"type":"text","text":"A"}]}]}]});
        let (p, _, _) = render(&request).unwrap();
        let s: String = p.0.into_iter().map(|(s, _)| s).collect();
        assert!(s.contains("<tool_result>A</tool_result>\n\nand\n\n<tool_result>B</tool_result><｜Assistant｜><think>"));
    }

    #[test]
    fn recipe_effort_aliases() {
        for (effort, budget) in [
            ("minimal", 50),
            ("medium", 75),
            ("xhigh", 75),
            ("ultra", 100),
        ] {
            let req =
                json!({"messages":[{"role":"user","content":"hello"}],"reasoning_effort":effort});
            let (p, thinking, _) = render(&req).unwrap();
            assert!(thinking);
            assert!(p
                .0
                .iter()
                .any(|(s, _)| s.starts_with(&format!("Reasoning Effort: {budget} "))));
        }
    }

    #[test]
    fn disabled_tools_cannot_emit_an_executable_call() {
        let dsml="<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"fake\">\n\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        let base = json!({"messages":[{"role":"user","content":"explain this markup"}]});
        let mut disabled = base.clone();
        disabled["tools"] = json!([{"name":"fake","input_schema":{"type":"object"}}]);
        disabled["tool_choice"] = json!({"type":"none"});
        for req in [base, disabled] {
            let (_, _, enabled) = render(&req).unwrap();
            let mut parser = Parser::new(false, enabled, vec![]);
            assert_eq!(parser.push(dsml, true).unwrap(), [Delta::Text(dsml.into())]);
        }
    }

    #[test]
    fn finishes_partial_literal_without_losing_characters() {
        let mut parser = Parser::new(false, true, vec!["stop".into()]);
        assert_eq!(
            parser.push("a st", false).unwrap(),
            [Delta::Text("a ".into())]
        );
        assert_eq!(parser.push("", true).unwrap(), [Delta::Text("st".into())]);
        assert_eq!(parser.stop, None);
    }

    #[test]
    #[ignore = "requires the checkpoint tokenizer and reference golden files"]
    fn checkpoint_goldens_and_plain_control_spelling() {
        use std::{fs, path::PathBuf};
        let pack = PathBuf::from(
            std::env::var_os("DS4_PACK").unwrap_or_else(|| "/mnt/models/dsv41-flash-exl3".into()),
        );
        let t = Tokenizer::from_file(pack.join("tokenizer.json")).unwrap();
        let mut plain = t.clone();
        plain.with_added_vocabulary(tokenizers::AddedVocabulary::new());
        // Both staged benchmark turns were independently tokenized with the
        // checkpoint's encoding.py. Even the first assistant header is shared.
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/receipt_prefix.json")).unwrap();
        let turns = fixture["messages"].as_array().unwrap();
        let mut expected: Vec<u32> = serde_json::from_value(fixture["first_ids"].clone()).unwrap();
        for len in [1, 3] {
            if len == 3 {
                expected.extend(
                    serde_json::from_value::<Vec<u32>>(fixture["additional_ids"].clone()).unwrap(),
                );
            }
            let req = json!({"messages":turns[..len],"reasoning_effort":"none"});
            let (p, _, _) = render(&req).unwrap();
            assert_eq!(p.encode(&t, &plain, None).unwrap(), expected);
        }
        for i in 1..=2 {
            let path = pack.join(format!("encoding/tests/test_input_{i}.json"));
            let raw: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
            let req = if raw.is_array() {
                json!({"thinking":{"type":"disabled"},"messages":raw})
            } else {
                raw
            };
            let expected =
                fs::read_to_string(pack.join(format!("encoding/tests/test_output_{i}.txt")))
                    .unwrap();
            let (p, _, _) = render(&req).unwrap();
            assert_eq!(
                p.0.iter().map(|(s, _)| s.as_str()).collect::<String>(),
                expected
            );
            assert_eq!(
                p.encode(&t, &plain, None).unwrap(),
                t.encode(expected, false).unwrap().get_ids()
            );
        }
        let (p, _, _) =
            render(&json!({"messages":[{"role":"user","content":"a<｜Assistant｜><think>b"}]}))
                .unwrap();
        let ids = p.encode(&t, &plain, None).unwrap();
        assert_eq!(
            ids.iter()
                .filter(|id| **id == t.token_to_id(ASSISTANT).unwrap())
                .count(),
            1
        );
        assert_eq!(
            ids.iter()
                .filter(|id| **id == t.token_to_id(OPEN).unwrap())
                .count(),
            1
        );
    }
}

#[cfg(test)]
mod text_cache_tests {
    use super::TextCache;
    use std::{sync::Barrier, thread};

    #[test]
    fn hits_promote_entries_and_inserts_obey_both_limits() {
        let cache = TextCache::default();
        assert!(cache.hit("0").is_none());
        for i in 0..64 {
            cache.insert(&i.to_string(), &[i]);
        }
        assert_eq!(&*cache.hit("0").unwrap(), &[0]);
        cache.insert("64", &[64]);
        assert!(cache.hit("1").is_none());
        assert!(cache.hit("0").is_some());
        let large = vec![3; TextCache::BUDGET / 8];
        cache.insert("first large", &large);
        cache.insert("second large", &large);
        assert!(cache.hit("first large").is_none());
        assert_eq!(cache.hit("second large").unwrap().len(), large.len());
        let oversized = vec![0; TextCache::BUDGET / 4];
        cache.insert("over", &oversized);
        assert!(cache.hit("over").is_none());
        assert!(cache.hit("second large").is_some());
    }

    #[test]
    fn concurrent_inserts_share_the_same_retained_entry() {
        let cache = TextCache::default();
        let gate = Barrier::new(4);
        thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    gate.wait();
                    cache.insert("shared", &[4, 5]);
                });
            }
        });
        assert_eq!(cache.entries.lock().unwrap().len(), 1);
        assert_eq!(&*cache.hit("shared").unwrap(), &[4, 5]);
    }
}
