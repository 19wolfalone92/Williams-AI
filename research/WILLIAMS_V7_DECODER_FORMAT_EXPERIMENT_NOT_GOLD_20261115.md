# V7 structured output research, not a book exam

The previous v6 GPU Qwen3-4B Base and genuine v5 LoRA correctly formatted respectively 43/100 and 66/100 outputs. CPU GGUF both produced 95/100. The backends and prompt lengths differ, so these numbers do **not** establish learning accuracy, trading returns or fair cross-backend speed.

This is a **new, distinct protocol** from frozen v6 (immutable results). Pinned MIT-licensed lm-format-enforcer 0.11.3 uses JSON Schema token filtering on HF generation, per-family enum, bounded reason, deterministic inference, no retries. A technical JSON error returns decision=null, not misleading ABSTAIN. GPU OOM is rethrown instead of relabeled as an AI refusal.

This CI tests only schema, parser availability and invalid-output safety. Real GPU pilot must be run and compared on identical frozen charts before claiming any improvement. Even if JSON becomes 100% syntactically valid, correctness of Bill Williams signals remains unknown without blinded expert labels. No trading or training in this change.
