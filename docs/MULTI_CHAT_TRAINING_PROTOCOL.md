# Williams AI — multi-chat single-model training protocol v1.0

Status: COORDINATION DESIGN; genuine historical Qwen3-4B LoRA GPU training already exists (merged PR #53 and draft research PR #55), but **no new challenger in #55 has passed admission**. This protocol does not itself train weights. Does not change Williams Bot.

## Goal
Ten scoped research roles (manually separate chats or five scheduled groups under the observed ChatGPT limit) produce **reviewed training examples, unit tests, and research artifacts** for **one** Williams AI model line. All training updates are serialized through a single trainer and gate. Chat messages are not gradients, and research memory / RAG is not weight training. Never call an untrained GGUF "trained".

## Current baseline / unknowns
The main repository includes an actual verified Qwen3-4B rank8 / 32-step LoRA experiment from merged PR #53 (public-rule loss only), **not merely retrieval**. Existing DRAFT PR #55 contains verified rank16 dual-T4 GPU work, archived checkpoint hashes and independent audits; the latest reported 7-Nov research exam is 72/280 versus 14/280 parent and 4/280 base, but **all four admission gates fail**, including missing 15 required protective stops, exceeding risk in 12/32 sizing cases and NTD loss regression +10.82%. Do not start a new training implementation from zero; integrate and independently audit these existing artifacts before designing a next corrective curriculum. No challenger is qualified for automatic trading. Check archived artifacts/hashes, compatible tokenizer, licensed source custody and independent score reproduction before any further training. Keep private copyrighted source files and account credentials out of this public GitHub repo.

## Standard agent instructions (copy into each chat)
- Repository: 19wolfalone92/Williams-AI. Never modify 19wolfalone92/Williams.
- At start, read README.md, docs/AI_SELF_STUDY_RU.md, and this protocol. State exact initial commit SHA.
- Work only in your assigned scope and isolated branch `research/chat-NN-<topic>`; submit a PR, do not merge into main. Never force-push or delete checkpoints.
- Output **one verifiable small batch per iteration** (default <= 25 items); do not promise unbounded work. No need for permission for safe, reversible research.
- Public GitHub may hold only original paraphrases, short allowable extracts where needed, and citations to source edition + page/figure. Never commit full book/PDF/OCR text, API keys, private user data, or trained model files.
- Example schema: `sample_id`, `task`, `question`, `reference_answer`, `source_id`, `edition`, `page`, `figure`, `evidence_status`, `available_at_utc`, `split_hint`, `review_status`, `notes`. Machine results must distinguish `VERIFIED`, `DRAFT`, and `UNKNOWN`; only verified examples may enter training.
- Test bars must be closed when observable; forbid future candles, revised history and hypothetical hindsight. Separate TC2 **operational rules** from TC1/NTD/Course analytical proposals. No production trading, demo order submission, or risk-limit changes from these tasks.
- After two failed approaches or a missing prerequisite, write a BLOCKED issue comment with logs, exact blocker, reproducible steps, workaround and next independent task. Move on instead of looping.
- End each iteration with: PR or issue link, exact counts added/verified/rejected, tests run and results, remaining blocker, next atomic action. Never claim training happened without training job URL, dataset hash, base model hash, checkpoint hash and evaluator report.

## Work allocation
| Chat | Scope / first verifiable deliverable |
|---|---|
| 01 | TC1 book-rule mapping: 20 original source-grounded cases with edition/page refs |
| 02 | New Trading Dimensions: 20 nonduplicated cases and contradiction register |
| 03 | Trading Chaos 2: 30 detailed TC2 rule cases, long/short symmetric |
| 04 | Profitunity Course: 20 cases and diagrams requiring human figure review |
| 05 | Waves: 30 labelled synthetic + 20 historical candidate examples, uncertainty encoded |
| 06 | Indicator mathematics: 40 property tests (Alligator/AO/AC/fractal/MFI) |
| 07 | TC2 setup-to-exit deterministic scenarios: 30 test cases, including stops and restart |
| 08 | Multi-timeframe H1/H4/D1: 30 timestamp-grounded scenarios |
| 09 | Evaluation: deterministic golden set, temporal splits, leak and regression checks |
| 10 | Coordinator: trainer and manifests, single writer, checkpoints, gates, status dashboard |

## Central integration contract
Per repository AGENTS.md only one author of code may be active at a time. Research workers may work independently on proposals in their respective issue threads; code writing and PR integration are sequential and exclusively coordinated. Proposals go under `research/chat-NN/` once the coordinator has a writer slot; tests have isolated names. Agent 09 verifies reference integrity, deduplication, split boundaries, and no-lookahead; Agent 10 only then creates an immutable `dataset-vN` manifest. The sole trainer starts from one approved base + adapter lineage and records `base_model_sha`, `tokenizer_sha`, `dataset_sha`, `code_commit`, `seed`, `hyperparameters`, `resource`, `checkpoint_sha` and metrics. Do NOT train on holdout or merge adapters blindly.

## Milestones with stop/go gates
1. **Infrastructure**: accessible model and adapter manifest, dataset schema, training/evaluation CLI, stable artifact storage, smoke-test and synthetic training resume test.
2. **Golden set**: >=100 human-reviewed, nonduplicated cases spanning all four books and market scenarios with *time-separated* holdout never used to tune parameters.
3. **Pilot**: one small QLoRA run, checkpoint upload and restart/resume proven, inference can load resulting adapter; compare to exact unchanged baseline.
4. **Promotion**: only promote if pre-registered critical TC2 tests do not regress, hallucinated citations / forward leakage are zero in their critical test suite, and broader held-out quality improves by a predeclared threshold; otherwise keep previous champion.
5. **Scale**: expand verified data, targeted error analysis, chronological/walk-forward tests, model-card release; do not infer profitability from question-answer accuracy.

## Anti-stall / anti-budget-loss rules
- Time-box research iteration and record *artifact-first* progress every run. After 2 errors mark blocked, preserve logs and switch to independent backlog.
- No job may overwrite champion. Trainer serializes by lock / concurrency, uses immutable versioned outputs and checkpointing; GPU absence triggers CPU data preparation/evaluation, never pretend fine-tuning occurred.
- GitHub Actions cron isn't guaranteed to execute (may delay/skip; 60-day inactivity in public repo may disable); add separate heartbeat/status check. Scheduled actions are **not** ten autonomous ChatGPT chat sessions.
- CPU workflows perform schema lint, synthetic no-lookahead tests and dataset assembly only; external GPU jobs need explicit approved quota and known accessible storage. No paid resources without user approval.
- First deliverable is an independent targeted evaluator for failed risk/STOP/NTD admission in existing PR #55, not a claim of 100% understanding. Maintain single champion and rollback.
- Track statuses: TODO, IN_PROGRESS, BLOCKED, REVIEW, VERIFIED, TRAINING_READY, TRAINED, REGRESSION, PROMOTED. Status alone does not imply model-weight update.

## Definition of done
An issue closes only with artifacts and reproducible checks. Overall initiative completes when the entire reproducible end-to-end pipeline (source-grounded examples -> validated dataset -> LoRA training -> checkpoint -> evaluation -> promotion/rollback -> Android inference conversion) is demonstrated. First-pass target is **one independently verified improvement that passes the precommitted critical admission gates** relative to existing real GPU pilots, not ten indefinite training loops.

## Schedule / automation
ChatGPT automations may check GitHub status and report failures, but do not replace GPU jobs or automatically synchronize state between ten chats. Actual unattended execution belongs in approved workflows/runners and must honor costs, licenses and platform quotas.

## Operational schedule and single writer

Five active ChatGPT scheduled tasks cover ten GitHub issue roles: research pairs (#58/#59, #60/#61, #62/#63, #64/#65), then a central QA/trainer coordinator (#66/#67). Research scheduled tasks are **issue-comment only**, with no code writes. This adheres to existing AGENTS.md, which forbids simultaneous code writers. See [autonomous runbook](WILLIAMS_AI_AUTONOMOUS_RUNBOOK_RU.md) for observed task limits, times, actual GPU run evidence and critical blockers.
