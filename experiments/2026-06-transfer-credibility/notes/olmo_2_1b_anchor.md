# OLMo-2-1B reference recipe — freeze for B6 transfer-credibility test

**Purpose:** Lock the EXACT recipe specification we use as the 1B-scale ground-truth anchor in the Cross-Scale Downstream Pareto transfer-credibility test (the B6 phase of the Karpa pivot — see [docs/build_scope/02_scope_B6.md](../../../docs/build_scope/02_scope_B6.md)). This file is what the B6 pre-registration will hash. Any change to this file invalidates a downstream B6 run.

**Source of truth:** AllenAI [OLMo](https://github.com/allenai/OLMo) repository, file `configs/official-0425/OLMo2-1B-stage1.yaml` — the stage-1 (pretraining) phase of the 2025-04 official release.

**Decision recorded 2026-06-10**, before any B6 code or H100 spend.

---

## 1. Exact published parameters (verbatim from the OLMo config)

### Model architecture

| Field | Value |
|---|---|
| d_model | 2048 |
| n_layers | 16 |
| n_heads | 16 |
| mlp_ratio | 8 |
| max_sequence_length | 4096 |
| activation | SwiGLU |
| layer_norm | RMSNorm with affine, ε=1e-6, `norm_after: true` |
| attention | Flash-Attention 2; per-block `attention_layer_norm: true` |
| positional | RoPE, `rope_theta: 500000` |
| weight_tying | false |
| bias | none (`include_bias: false`) |
| init | normal(0, 0.02) with `init_cutoff_factor: 3` |
| vocab_size | 100278 |
| embedding_size | 100352 (padded for kernel alignment) |
| eos / pad | 100257 / 100277 |

Computed: **≈ 1.18B parameters total** (architecture-derived). This matches AllenAI's "OLMo-2-1B" naming.

### Optimizer

| Field | Value |
|---|---|
| name | AdamW |
| learning_rate (peak) | 4.0e-4 |
| weight_decay | 0.1 |
| eps | 1e-8 |
| betas | (0.9, 0.95) |
| decay_norm_and_bias | true |
| decay_embeddings | false |
| max_grad_norm | 1.0 |

### Schedule

| Field | Value | Notes |
|---|---|---|
| name | cosine_with_warmup | units: tokens |
| t_warmup | 8,388,608,000 tokens (~8.4B) | ~0.21% of full t_max |
| t_max | 5,000,000,000,000 tokens (5T) | |
| alpha_f | 0.1 | final LR = 0.1 × peak |
| warmup_min_lr | 0.0 | |

### Training duration + batch + precision

| Field | Value |
|---|---|
| max_duration | 4,000,000,000,000 tokens (4T) — stage 1 |
| global_train_batch_size | 512 |
| device_train_microbatch_size | 4 |
| precision | amp_bf16 (FSDP mixed) |
| FSDP sharding | SHARD_GRAD_OP |

### Auxiliary losses

| Field | Value |
|---|---|
| softmax_auxiliary_loss (z-loss) | true |
| auxiliary_loss_multiplier | 1e-5 |
| fused_loss | true |

### Tokenizer

`tokenizers/allenai_dolma2.json` — the AllenAI dolma2-tokenizer, vocab 100278. **This is NOT GPT-2 BPE.** The B6 GT side must use this tokenizer; per Q2 in the master plan, we committed to retokenizing FineWeb-Edu shards under this tokenizer for the GT runs.

### Data mix (stage 1)

Five top-level sources, blended:

- `preprocessed/dclm` — DataComp-LM (the largest source)
- `preprocessed/olmo-mix` — AllenAI's Dolma-derived internal mix
- `preprocessed/pes2o` — academic papers
- `preprocessed/proof-pile-2` — math (algebraic-stack + arxiv + open-web-math)
- `preprocessed/starcoder` — code

Total 1,122 file references in the published config; full mix is ~4T tokens.

---

## 2. Required adaptations for Karpa B6 (30B tokens, not 4T)

The published recipe is for 4T tokens. We run 30B. The following decisions explicitly adapt the recipe and MUST land in the B6 pre-registration:

### 2.1 Warmup scaling

**Decision:** Linear-scale warmup proportionally to total tokens: `t_warmup_b6 = 8.4B × (30B / 4T) = 63M tokens`. Round up to **64M tokens** (`67,108,864` = 2^26) for clean batch alignment.

**Why this and not "keep the 8.4B published":** keeping the published warmup would be 28% of training, far outside the published recipe's intent (~0.21%). The recipe specifies warmup as a *fraction of the schedule*, not an absolute count.

**Why this and not "1% of training":** OLMo's published 0.21% fraction is below the OLMo-team-recommended "1-2% of tokens" general guidance, but is what *the published recipe* specifies. Honoring the published fraction is the most defensible reading.

### 2.2 Cosine `t_max`

**Decision:** `t_max_b6 = 30,000,000,000` tokens. The cosine completes within the run; final LR = `alpha_f × peak = 4.0e-4 × 0.1 = 4.0e-5`.

**Why not `t_max=5T` (published) + early stop at 30B:** would leave the LR at near-peak for the whole run (30B/5T = 0.6% of cosine traversed → cosine value ≈ 0.998 × peak throughout). This is *not* the published recipe's intent.

### 2.3 Data mix at 30B

**Decision:** Sample from the published mix's five sources in proportion to the published config's per-source token weights. Concretely, for the first run we sample 30B from `dclm` only (~22T of the original mix; we just take the first 30B contiguous slice after the published shuffle seed). **Document this as an approximation in the B6 pre-registration.**

**Why DCLM-only first:** DCLM dominates the published mix and is the cleanest single-source comparison; mixing 5 sources at 30B introduces too much per-source noise at this token budget. If B6 fails on DCLM-only, the fallback (per the master plan §7) re-runs with the proportional 5-source mix.

### 2.4 Batch size

**Decision:** Keep `global_train_batch_size: 512`, `device_train_microbatch_size: 4` exactly. No scaling.

### 2.5 Sequence length

**Decision:** Keep `max_sequence_length: 4096` exactly. Tokens-per-step = 512 × 4096 = ~2.1M. Steps for 30B = ~14,300.

### 2.6 Precision + grad clip

**Decision:** Keep `amp_bf16` + `max_grad_norm: 1.0` exactly.

---

## 3. What this freezes vs what B6 owns

This document freezes the **reference-recipe specification**. It does NOT decide:

- The 12 candidate **recipe variants** B6 trains alongside this anchor (those are B6's pre-registration to author).
- The eval pipeline / downstream tasks used to rank the 1B GT-side outputs (CORE-22 + the swap'd hardness subset per [docs/license/hardness_subset_decision.md](../../../docs/license/hardness_subset_decision.md)).
- The H100 provider / instance type / spend cap for the GT runs (separate B6 spend authorization).
- The kill-switches that auto-halt the experiment (B6 owns the explicit list).

## 4. Open implementation questions for B6 author

1. **Tokenization corpus**: re-tokenize FineWeb-Edu under dolma2 (~16h of CPU) **vs** download `preprocessed/dclm` files from olmo-data.org and use them directly (saves CPU, adds ~3TB download). Recommended: download — matches published recipe byte-for-byte on the data side.
2. **Sharded vs unsharded checkpoints**: published uses `sharded_checkpointer: olmo_core` with FSDP `SHARD_GRAD_OP`. B6 reproduction needs to match for byte-identical results. Some non-OLMo-trained baselines may need a conversion step.
3. **Evaluation cadence during the 30B run**: published uses `eval_interval: 1000` (steps). At 14,300 steps total, that's ~14 evals per run × 12 recipes = 168 eval points to manage. Confirm storage budget for intermediate checkpoints OR cut eval to final-only.
4. **Run-name suffix convention**: `OLMo2-1B-stage1-karpa-b6-anchor-r0` for the anchor, `OLMo2-1B-stage1-karpa-b6-Rxx` for each of the 12 candidate variants.

## 5. Source verification

The published config was fetched from this URL on 2026-06-10:

```
https://raw.githubusercontent.com/allenai/OLMo/main/configs/official-0425/OLMo2-1B-stage1.yaml
```

A copy is preserved at `experiments/2026-06-transfer-credibility/refs/OLMo2-1B-stage1.yaml` and pinned by hash:

| Field | Value |
|---|---|
| SHA256 | `bd75e78bf7a818168d7f6e57b561888b380938b6a3abab4f6222ee9b340cd7c7` |
| BLAKE2b-256 | `0x2f4cd8cb415b4327dd154c94011135b7cfaead961074a69edcafd60462ea2024` |
| size | 164,375 bytes |

The B6 pre-registration MUST emit a `B6PreRegistration` chain event whose payload includes this BLAKE2b hash, BEFORE any spend authorization. If the file diverges from this hash at B6 dispatch time, the experiment is invalidated.

## 6. Change log

| Date | Change | By |
|---|---|---|
| 2026-06-10 | Initial freeze | (this commit) |
