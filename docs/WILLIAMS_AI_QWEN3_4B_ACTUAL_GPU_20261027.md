# Williams AI: Qwen3-4B actual GPU LoRA, 2026-10-27

## Actual training (NOT a trading benchmark)

- Pretrained base: `Qwen/Qwen3-4B`, pinned revision `3fb356690bbdae7353c94efb3eecdce276e6ed1c`.
- Actual hardware: NVIDIA Tesla T4 via user's private Kaggle notebook.
- Algorithm: 4-bit NF4 QLoRA, PEFT rank 8, **32 real optimizer steps**, **5,898,240 trainable adapter parameters**.
- Data: 36 train and 10 validation independently authored, public rule-practice prompts from TC1, NTD, TC2 and COURSE (no overlapping rule IDs).
- **Actual mean text token validation loss: 3.102492571 (base) to 1.564500213 (adapted), relative improvement 49.572797%.** Per edition: TC1 2.874593 -> 1.526677; NTD 3.377219 -> 1.617993; TC2 3.437775 -> 1.825078; COURSE 2.666692 -> 1.176875. Source-language validation accepted, no edition regressed.
- Kaggle GPU notebook: https://www.kaggle.com/code/wolf19loner92/williams-ai-qwen4b-rank8-32steps-20261027
- The original upload/download GitHub Actions run was **red after successful GPU training** due to incorrect ZIP filename filtering. Independent GPU-artifact recovery **succeeded** in GitHub Actions https://github.com/19wolfalone92/Williams-AI/actions/runs/40457818054.
- GitHub Actions artifact ID **12713317419**, contains real saved weights, tokenizer and receipt.
- Authenticated trained adapter `adapter/adapter_model.safetensors`: **23,631,240 bytes**, SHA-256 `b953463217ee1adc7ecc081e5d4f6d0521033ed6749c6a690617ae6d2533caeb`. Source training commit: `b85af751aff558cc24f84fbd49d4602514430cf5`. SHA independently verified after recovery.

## Safety and limitations (mandatory)

**49.57% reduction is validation language-model token loss, NOT trading accuracy or trading return**. Full private PDF original four books were **not** used in this training. Book illustrations and scanned Course are not understood by default. Independently expert-annotated waves and WM1/WM2/WM3: **none**. Market holdout base-vs-adapter predictions: **not yet executed**; attempts to attach another private Kaggle notebook output to GPU were unsuccessful. No autonomous trading qualification is established. No Binance orders were created; live/demo execution must remain disabled. This training belongs ONLY to Williams AI; Williams Bot 2.1 is untouched.

Next verification: safely connect user-owned private original-book corpus (without public GitHub upload), run independently reviewed multi-timeframe chart examination and then causally sealed market walk-forward with fees/slippage. Unfavorable results must block deployment.
