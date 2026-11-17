# Williams AI v7.1: настоящий GPU-эксперимент завершён; почему JSON всё ещё ломается

**Дата доказательства:** 17.11.2026. **Только независимая техническая проверка**, НЕ книжный gold и НЕ торговая приемка.

## Подтверждённое завершение старого, уже созданного Kaggle kernel

- Kernel: `wolf19loner92/williams-v71-lmfe8-deps-20261116`
- Авторизованный GitHub Actions, который ТОЛЬКО опросил и скачал старый приватный kernel: run **44095124520**, completed/success, 17 November 2026.
- GitHub Actions artifact **14332089771**, ZIP SHA-256 `05f07224da4879e179631a429e26653212289f78e6c7de3e304ad9120aa63cd9`. Внутри получены **реальные** JSON: `v71_patched_lmfe_8_actual_answers_NO_GOLD.json`, `v71_same_8_inputs_vs_v6_NO_GOLD.json`, `v71_existing_kernel_status.json`. Его сравнение с исходным GPU v6 выполнено и `comparison_verified=true`.
- Original immutable GPU v6 artifact **14138759783** from run **43658377350**; baseline archive SHA-256 `47067bae5afb41b6fceae44ca1bace378716985d6771af80abf274d4f393bec7`.
- Matched SHA for all 8 historical H1 snapshots, exact GPU prompts, original CPU prompts, cutoff, frozen v5 adapter SHA `8bd26590df627c7452d56e81df6a4572885bfc82515259a7c9de4a3970c850b8`, Qwen3-4B revision `3fb356690bbdae7353c94efb3eecdce276e6ed1c`, source selection. **No new training/weights/trading.**

## Exactly what the 8×2 results establish

| Read-only paired H1 sample | Original v6 format stop | v7.1 format stop | v7.1 technical errors |
| --- | ---: | ---: | ---: |
| Qwen3-4B Base (8 charts) | 3/8 | **2/8** | **6/8 INVALID_JSON** |
| Qwen3-4B + authentic v5 LoRA (8 charts) | 5/8 (+1 length) | **4/8** | **4/8 INVALID_JSON** |

- 6 valid JSON results in total (2 Base, 4 LoRA); **10 failures in total**.
- Examining the actual private output tokens and decoded JSON, **all 10 failures emit Qwen `<|im_end|>` EOS inside an unterminated JSON `reason` string**, after only 12–15 generated tokens. Some repeated identical fragments appear for two model roles. They are decoder/termination errors, **not model-authorized ABSTAIN**.
- With prior LMFE dependency bug fixed, the **early-EOS cause now becomes apparent**: the constrained decoding path did not prevent an end token while a valid JSON object was still incomplete. This diagnosis is supported by actual `eos_seen`, incomplete string parsing and last token ID 151645; the exact implementation fault in LMFE token allowlisting still requires an instrumented v7.2 rerun.
- Six parseable replies contain nonempty `reason` text, but **five reach the strict 220-character limit**. No expert has verified correctness of these rationales or whether they faithfully justify signals. No conclusion about model book accuracy.
- Changed decisions on original GPU tests may reflect decoding protocol. v7.1 valid responses were **WAVE_HYPOTHESIS** twice for Base and **ABSTAIN** twice + **WAVE_HYPOTHESIS** twice for v5. `ABSTAIN` in a valid reply is different from an `INVALID_JSON` technical error.

## Root-cause preventive work after actual output inspection

- `williams_ai/v72_eos_guard.py` introduces a strict, testable **prefix-termination filter**: when LMFE returns the EOS token, it is blocked unless the *actually decoded continuation text* passes strict JSON object, allowed `decision` class, `reason` length and no extra/duplicate keys. If only premature EOS is allowed, fail closed.
- Guard applies in prefix-token callback against the generated suffix **after the prompt tokens**; no source PDF or private chart/gold is exported.
- Pure offline unit tests ensure early-EOS rejection, valid-JSON EOS allowance, schema-invalid closure rejection, multiple EOS IDs, malformed inputs, and no fabricated valid abstention.
- This **has not yet been shown to improve real T4 v7.2 generation**. Do not promote a unit-tested decoding guard to a successful GPU result without a separate genuine paired model inference.

## Research restrictions

Gold-reviewed H1 labels = **0**; book accuracy = UNKNOWN; 30 source-only reasoning probes ≠ market gold. **No LoRA training and no Williams Bot integration.** This status artifact is a technical paired generation diagnostic and cannot establish trading profitability or real knowledge gains.
