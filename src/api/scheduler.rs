use super::*;

#[derive(Clone, Copy)]
pub(super) struct Limits {
    pub resident: usize,
    pub live: usize,
    pub quantum: usize,
}
impl Limits {
    pub(super) fn configured() -> Result<Self> {
        fn setting(name: &str, default: usize) -> Result<usize> {
            match std::env::var(name) {
                Ok(s) => s
                    .parse()
                    .with_context(|| format!("{name} must be an integer")),
                Err(std::env::VarError::NotPresent) => Ok(default),
                Err(e) => Err(e.into()),
            }
        }
        let resident = setting("DS4_MAX_SEQS", MAX_SEQS)?;
        let live = setting("DS4_MAX_LIVE_SEQS", 32)?;
        let quantum = setting("DS4_SWAP_QUANTUM_STEPS", 32)?;
        ensure!(
            (1..=MAX_SEQS).contains(&resident),
            "DS4_MAX_SEQS must be between 1 and {MAX_SEQS}"
        );
        ensure!(
            (resident..=64).contains(&live),
            "DS4_MAX_LIVE_SEQS must be between DS4_MAX_SEQS and 64"
        );
        ensure!(quantum > 0, "DS4_SWAP_QUANTUM_STEPS must be positive");
        Ok(Self {
            resident,
            live,
            quantum,
        })
    }
}

pub(super) trait BatchEngine {
    type Session;
    fn start(&mut self, ids: Vec<u32>, limit: usize, temperature: f32) -> Result<Self::Session>;
    fn finished_tool(&mut self, _session: Self::Session) {}
    fn idle_tool_delay(&self) -> Option<std::time::Duration> {
        None
    }
    fn idle_tool_step(&mut self) -> Result<usize> {
        Ok(0)
    }
    fn prefill(session: &Self::Session) -> bool;
    fn processed(session: &Self::Session) -> usize;
    fn step(&mut self, sessions: &mut [&mut Self::Session]) -> Vec<Result<Step>>;
    fn suspend(&mut self, session: &mut Self::Session) -> Result<()>;
    fn resume(&mut self, session: &mut Self::Session) -> Result<()>;
}
pub(super) struct ServingModel {
    pub model: Model,
    pub swap: SwapStore,
}
impl BatchEngine for ServingModel {
    type Session = Session;
    fn start(&mut self, ids: Vec<u32>, limit: usize, temperature: f32) -> Result<Session> {
        self.model.interrupt_tool_prefill();
        self.model.start_request(ids, limit, temperature)
    }
    fn finished_tool(&mut self, session: Session) {
        self.model.finished_tool(session)
    }
    fn idle_tool_delay(&self) -> Option<std::time::Duration> {
        self.model.idle_tool_delay()
    }
    fn idle_tool_step(&mut self) -> Result<usize> {
        self.model.idle_tool_step()
    }
    fn prefill(session: &Session) -> bool {
        session.is_prefill()
    }
    fn processed(session: &Session) -> usize {
        session.processed_tokens()
    }
    fn step(&mut self, sessions: &mut [&mut Session]) -> Vec<Result<Step>> {
        self.model.step_batch(sessions)
    }
    fn suspend(&mut self, s: &mut Session) -> Result<()> {
        self.model.suspend(s, &self.swap)
    }
    fn resume(&mut self, s: &mut Session) -> Result<()> {
        self.model.resume(s)
    }
}
fn admit<'t, E: BatchEngine>(
    engine: &mut E,
    tokenizer: &'t Tokenizer,
    job: Job,
    runtime: &tokio::runtime::Handle,
) -> Option<Box<Active<'t, E::Session>>> {
    let Job { input, wire, reply } = job;
    if reply.closed() {
        return None;
    }
    let output = Output::new(wire, &input);
    let session = output
        .start(reply.stream())
        .and_then(|_| engine.start(input.ids, input.max_tokens, input.temperature));
    match session {
        Ok(session) => Some(Box::new(Active {
            session,
            decoder: tokenizer.decode_stream(false),
            parser: Parser::new(input.thinking, input.tools, input.stops),
            output,
            reply,
            count: 0,
            ready: true,
            slice_steps: 0,
            suspend_retry: None,
        })),
        Err(err) => {
            respond(reply, wire, Err(err), runtime);
            None
        }
    }
}
enum Waiting<'t, S> {
    New(Job),
    Suspended(Box<Active<'t, S>>),
}
impl<S> Waiting<'_, S> {
    fn runnable(&self) -> bool {
        match self {
            Self::New(_) => true,
            Self::Suspended(a) => a.ready,
        }
    }
}

// Expired/closed streams release their slot or swap ticket immediately. A
// backed-up stream can drain its buffered events even while it is off the GPU.
fn refresh<S>(a: &mut Active<'_, S>) -> bool {
    if a.reply.closed() {
        return false;
    }
    if let Some(tx) = a.reply.stream() {
        match tx.ready() {
            Ok(ready) => a.ready = ready,
            Err(_) => return false,
        }
    }
    true
}

struct Rotation {
    cursor: usize,
    cooldown: usize,
}
impl Rotation {
    fn rotate<'t, E: BatchEngine>(
        &mut self,
        engine: &mut E,
        slots: &mut [Option<Box<Active<'t, E::Session>>>],
        waiting: &mut VecDeque<Waiting<'t, E::Session>>,
        limits: Limits,
        metrics: &Mutex<Metrics>,
    ) {
        if slots.iter().any(Option::is_none) || !waiting.iter().any(Waiting::runnable) {
            return;
        }
        if self.cooldown > 0 && slots.iter().flatten().any(|a| a.ready) {
            return;
        }
        // Evict at most one row this turn. Prefer a stalled reader, then scan
        // expired quanta in round-robin order. A fresh GPU slot gets useful work.
        let eligible = |a: &Active<'_, E::Session>| {
            a.suspend_retry
                .is_none_or(|(retry, _)| Instant::now() >= retry)
        };
        let order = (0..slots.len()).map(|n| (self.cursor + n) % slots.len());
        let victim = order
            .clone()
            .find(|&i| {
                let a = slots[i].as_ref().unwrap();
                !a.ready && eligible(a)
            })
            .or_else(|| {
                order.into_iter().find(|&i| {
                    let a = slots[i].as_ref().unwrap();
                    a.slice_steps >= limits.quantum && eligible(a)
                })
            });
        let Some(i) = victim else { return };
        self.cursor = (i + 1) % slots.len();
        self.cooldown = limits.quantum.div_ceil(slots.len());
        let a = slots[i].as_mut().unwrap();
        match engine.suspend(&mut a.session) {
            Ok(()) => {
                a.suspend_retry = None;
                waiting.push_back(Waiting::Suspended(slots[i].take().unwrap()));
                metrics.lock().unwrap().rotations += 1;
            }
            Err(err) => {
                // Model::suspend preserves the live KV on failure. Let this
                // request run to completion if neither tier can hold it.
                // Small sessions can keep releasing tickets which are still
                // too small for this row. Rate-limit per row independently.
                let delay = a.suspend_retry.map_or(Duration::from_secs(1), |(_, last)| {
                    (last * 2).min(Duration::from_secs(60))
                });
                a.suspend_retry = Some((Instant::now() + delay, delay));
                metrics.lock().unwrap().suspend_failures += 1;
                eprintln!("could not suspend {}: {err:#}", a.output.id);
            }
        }
    }
}

fn record<E: BatchEngine>(
    metrics: &Mutex<Metrics>,
    slots: &[Option<Box<Active<'_, E::Session>>>],
    waiting: &VecDeque<Waiting<'_, E::Session>>,
    batch: bool,
) {
    let mut sample = metrics.lock().unwrap();
    let mut runnable = 0;
    let mut decode_rows = 0;
    sample.active = 0;
    sample.prefilling = 0;
    sample.processed_tokens = 0;
    for a in slots.iter().flatten() {
        sample.active += 1;
        let prefill = E::prefill(&a.session);
        sample.prefilling += usize::from(prefill);
        sample.processed_tokens += E::processed(&a.session);
        if a.ready {
            runnable += 1;
            decode_rows += usize::from(!prefill);
        }
    }
    sample.resident = sample.active;
    sample.swapped = waiting
        .iter()
        .filter(|v| matches!(v, Waiting::Suspended(_)))
        .count();
    sample.waiting = waiting.len() - sample.swapped;
    sample.logical_active = sample.resident + waiting.len();
    sample.peak_logical_active = sample.peak_logical_active.max(sample.logical_active);
    sample.peak_swapped = sample.peak_swapped.max(sample.swapped);
    sample.decoding = sample.active - sample.prefilling;
    sample.peak_active = sample.peak_active.max(sample.active);
    sample.peak_processed_tokens = sample.peak_processed_tokens.max(sample.processed_tokens);
    if batch && runnable > 0 {
        sample.batch_steps += 1;
        sample.multi_sequence_steps += u64::from(runnable > 1);
        sample.multi_decode_steps += u64::from(decode_rows > 1);
        sample.peak_batch_size = sample.peak_batch_size.max(runnable);
        sample.peak_decode_batch = sample.peak_decode_batch.max(decode_rows);
    }
}
pub(super) fn schedule<E: BatchEngine>(
    engine: &mut E,
    tokenizer: &Tokenizer,
    mut jobs: mpsc::Receiver<Job>,
    limits: Limits,
    metrics: &Mutex<Metrics>,
    runtime: &tokio::runtime::Handle,
) {
    let mut slots: Vec<Option<Box<Active<'_, E::Session>>>> =
        (0..limits.resident).map(|_| None).collect();
    let mut waiting = VecDeque::new();
    let mut rotation = Rotation {
        cursor: 0,
        cooldown: 0,
    };
    loop {
        for slot in &mut slots {
            if slot.as_mut().is_some_and(|a| !refresh(a)) {
                *slot = None;
            }
        }
        waiting.retain_mut(|item| match item {
            Waiting::New(job) => !job.reply.closed(),
            Waiting::Suspended(a) => refresh(a),
        });
        let active = slots.iter().flatten().count();
        if active == 0 && waiting.is_empty() {
            record::<E>(metrics, &slots, &waiting, false);
            let job = match jobs.try_recv() {
                Ok(job) => job,
                Err(mpsc::error::TryRecvError::Disconnected) => break,
                Err(mpsc::error::TryRecvError::Empty) => {
                    if let Some(delay) = engine.idle_tool_delay() {
                        match runtime
                            .block_on(async { tokio::time::timeout(delay, jobs.recv()).await })
                        {
                            Ok(Some(job)) => {
                                waiting.push_back(Waiting::New(job));
                                continue;
                            }
                            Ok(None) => break,
                            Err(_) => (),
                        }
                    }
                    match engine.idle_tool_step() {
                        Ok(0) => (),
                        Ok(rows) => {
                            metrics.lock().unwrap().idle_tool_prefill_tokens += rows as u64;
                            continue;
                        }
                        Err(e) => eprintln!("idle tool prefill skipped: {e:#}"),
                    }
                    let Some(job) = jobs.blocking_recv() else {
                        break;
                    };
                    job
                }
            };
            waiting.push_back(Waiting::New(job));
        }
        // A fixed intake budget prevents a stream of disconnected arrivals
        // from consuming the model thread. FIFO includes both new and saved work.
        for _ in active + waiting.len()..limits.live {
            let Ok(job) = jobs.try_recv() else { break };
            if !job.reply.closed() {
                waiting.push_back(Waiting::New(job));
            }
        }
        rotation.rotate(engine, &mut slots, &mut waiting, limits, metrics);
        for slot in slots.iter_mut().filter(|v| v.is_none()) {
            let Some(i) = waiting.iter().position(Waiting::runnable) else {
                break;
            };
            *slot = match waiting.remove(i).unwrap() {
                Waiting::New(job) => {
                    let admitted = admit(engine, tokenizer, job, runtime);
                    if let Some(a) = &admitted {
                        // At admission the committed position is exactly the prefix
                        // restored for this prompt. Live swap restores are separate.
                        let mut m = metrics.lock().unwrap();
                        m.requests_started += 1;
                        m.prefix_reused_tokens += E::processed(&a.session) as u64;
                    }
                    admitted
                }
                Waiting::Suspended(mut a) => match engine.resume(&mut a.session) {
                    Ok(()) => {
                        a.slice_steps = 0;
                        metrics.lock().unwrap().resumes += 1;
                        Some(a)
                    }
                    Err(err) => {
                        metrics.lock().unwrap().resume_failures += 1;
                        respond(
                            a.reply,
                            a.output.wire,
                            Err(err.context("restoring live session")),
                            runtime,
                        );
                        None
                    }
                },
            };
        }
        record::<E>(metrics, &slots, &waiting, true);
        let mut states: Vec<_> = slots
            .iter_mut()
            .flatten()
            .filter(|a| a.ready)
            .map(|a| &mut a.session)
            .collect();
        if states.is_empty() {
            if slots.iter().any(Option::is_some) || !waiting.is_empty() {
                std::thread::sleep(Duration::from_millis(2));
            }
            continue;
        }
        let outcomes = engine.step(&mut states);
        let expected = states.len();
        // Include the newly committed target KV before removing any finished row.
        record::<E>(metrics, &slots, &waiting, false);
        rotation.cooldown = rotation.cooldown.saturating_sub(1);
        let cardinality_ok = outcomes.len() == expected;
        let mut outcomes = outcomes.into_iter();
        for slot in &mut slots {
            let Some(a) = slot else { continue };
            if !a.ready {
                continue;
            }
            a.slice_steps = a.slice_steps.saturating_add(1);
            let result = if cardinality_ok {
                outcomes.next().unwrap().and_then(|step| a.consume(step))
            } else {
                Err(anyhow!(
                    "inference returned a different number of results than sessions"
                ))
            };
            let terminal = match result {
                Ok(None) => continue,
                Ok(Some(v)) => Ok(v),
                Err(e) => Err(e),
            };
            let a = slot.take().unwrap();
            let tool = terminal.is_ok() && a.output.n_calls > 0 && a.parser.stop.is_none();
            respond(a.reply, a.output.wire, terminal, runtime);
            if tool {
                engine.finished_tool(a.session);
            }
        }
    }
}
