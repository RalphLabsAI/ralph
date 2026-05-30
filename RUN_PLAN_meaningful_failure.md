# Run Plan — Meaningful-Failure 10% Reward (testnet 16)

**Goal:** demonstrate the protocol's three-class submission verdict in a single validator epoch on testnet 16. Two autonomous research agents take separate research directions; validator returns one of three verdicts per submission.

**Status:** code complete + tested locally (24/24 tests green). Awaiting H100 provisioning + launch.

---

## 1. Three-class verdict (new this run)

A submission that **passes all four cheap ops** (diff scan, attestation verify, log plausibility, hidden eval) is then classified:

| Class | Bar | Weight | What happens |
|---|---|---|---|
| **king_change** | Decisively beats king on val_bpb past the 0.013 noise floor | `1.0` | PR merged, recipe tag released, on-chain set_weights |
| **meaningful_failure** | Attested + non-trivial diff (>5 lines touching training config) + coherent rationale (≥200 non-WS chars, ≥2 paragraphs, ≥4 distinct sentences) + val_bpb landed within 2× the noise band of king | `0.1` | PR rejected, bundle archived to `queue/meaningful_failure/`, rationale published as a negative result |
| **plain_failure** | Anything else (val_bpb >2× noise worse than king, OR trivial diff, OR incoherent/missing rationale, OR no king to compare against) | `0.0` | PR rejected, bundle archived to `queue/scored/` |

A submission that fails **any of the four cheap ops** is **rejected** before classification (existing path).

---

## 2. Agent A — aggressive (likely outcome: meaningful_failure or king_change)

**Research target:** one *structural* change to the canonical recipe.

Pick **one** of these directions, run to completion, submit:

- **Optimizer swap.** Replace AdamW with **Lion** (`from torch.optim import Optimizer; class Lion(Optimizer): ...`). Adjust peak LR to ~30% of the AdamW value (Lion's sign-based updates need smaller LR). Keep schedule shape, batch size, and all other hyperparams identical to the current king.
- **LR schedule shape swap.** Replace cosine decay with **trapezoidal** (linear warmup → flat at peak → linear cooldown over last 20% of steps). Keep peak LR, weight decay, and optimizer identical to the current king.
- **Skip-connection scaling change.** Multiply residual stream output by `1/sqrt(2*n_layers)` per residual block (one of the DeepNet variants). Re-init nothing else. Keep all training hyperparams identical.

**Rationale.md requirements (must satisfy the meaningful_failure coherence bar):**
- ≥ 200 non-whitespace characters.
- ≥ 2 paragraphs.
- ≥ 4 distinct sentences.
- Should describe: **hypothesis** (what you expected), **observation** (what val_bpb landed at), **interpretation** (why it failed/succeeded), **next-step suggestion** (what to try if this approach is to be revisited).

**Why this agent is positioned this way:** structural changes are high-variance — likely either a clean win that beats the king past the noise floor, or a "close but not quite" result inside the 2× noise band. Either outcome exercises a protocol branch we want to demo. The aggressive choice is what makes meaningful_failure interesting — modest changes that come close are exactly the kind of negative result the corpus is supposed to capture.

---

## 3. Agent B — conservative (likely outcome: king_change)

**Research target:** micro-tune one or two existing hyperparameters in the king's training config.

Pick **one or two** of these:

- Peak LR: try ±15-20% of current king's peak LR (so 5e-3 → 5.8e-3 or 4.2e-3).
- Weight decay: try ±50% of current king's WD (so 0.10 → 0.05 or 0.15).
- Warmup steps: try ±50% (so 80 → 120 or 40).
- Final-LR ratio: try changing cosine_min_ratio from 0.1 to 0.05 or 0.2.

Keep the optimizer, schedule shape, batch size, and model architecture **identical** to the current king.

**Rationale.md:** same 4-element structure as Agent A. Doesn't have to be long — but it has to be coherent (the protocol won't tell the difference between A's and B's submissions; it judges both with the same bar).

**Why this agent is positioned this way:** small parameter perturbations within sensible ranges tend to *either* beat the king by a small margin past noise floor *or* land inside the noise band. Likely outcome is king_change; possible outcome is meaningful_failure if the tweak happens to be slightly worse than the king. Either way, exercises the protocol cleanly.

---

## 4. Anti-gaming notes — known holes (testnet only; address before mainnet)

- **No minimum training-step check.** A miner could train for only a few hundred steps, land val_bpb inside the 2× noise band by luck, and farm meaningful_failure credit. For testnet today (2 miners, no economic motive) this is fine, but pre-mainnet `op3_log_plausibility` needs a `min_steps ≥ X% of canonical baseline` check.
- **Rationale-coherence is structural heuristic only.** Any rationale that has 4+ distinct sentences and 200+ chars passes — no semantic check. A determined attacker could generate plausible-looking but content-free rationales with an LLM. The LLM-judge pass is the obvious followup (probably 1 call per meaningful_failure candidate, cheap enough since meaningful_failure should be rare in steady state).
- **No per-miner rate-limiting on meaningful_failure.** A miner could spam meaningful_failure submissions to farm 0.1× weight repeatedly within an epoch. Add: `max_meaningful_failures_per_miner_per_epoch = 1` before mainnet.
- **No HF-corpus negative-result push yet.** Today bundles archived to `queue/meaningful_failure/<bundle_id>/` locally only; HF dataset push is followup so the corpus gets the rationale published as a negative result.

---

## 5. Operational checklist

### Infrastructure (user runs)

- [ ] **Validator host.** Same Scaleway box as last run. Validator code is on `main` of `karpaai/karpa` after this patch lands.
- [ ] **Two miner hosts.** Two H100 PCIe boxes, separate hosts. Both should have `karpaai/karpa` and `karpaai/recipe` checkouts + the canonical Karpa proof-test Docker image pulled.
- [ ] **Bittensor registration.** Both miners registered on testnet 16 (`btcli subnet register --netuid 16`). Validator already registered from last run.
- [ ] **Env vars on validator host:**
  - `KARPA_BOT_GH_TOKEN` — karpa-bot's GH PAT (for PR merge + tag)
  - `KARPA_BOT_HF_TOKEN` — karpa-bot's HF token (for bundle PR merge)
  - `KARPA_HF_REPO=karpaai/proof-bundles`
  - `KARPA_SKIP_HANDSHAKE=1` (still needed until on-chain handshake commits land — followup #11)
- [ ] **Env vars on each miner host:**
  - `BT_WALLET_COLDKEY` + `BT_WALLET_HOTKEY` for the registered miner identity
  - Each miner's own GH PAT for opening the recipe PR
  - `HF_TOKEN` for opening the bundle PR

### Launch sequence

1. Confirm validator is up: `python -m validator.service --once` to do a no-bundle epoch and verify it doesn't crash with the new code.
2. Start validator in long-run mode: `python -m validator.service --epoch-seconds 60 --noise-floor 0.013`. (60s polling so we see verdicts fast; bump back to 120 after the demo.)
3. Launch Agent A on miner 1. It searches, proves, commits.
4. Launch Agent B on miner 2. It searches, proves, commits.
5. Watch validator log — should see one of:
   - `NEW KING: <miner>` → king_change verdict, PR merged, tag released
   - `MEANINGFUL FAILURE: <miner>` → meaningful_failure verdict, bundle archived
   - `below threshold: <miner>` → plain_failure verdict

### Receipts to collect

- [ ] `taostats` or `btcli` screenshot showing set_weights extrinsic on netuid 16 with the new weights `(king_miner: 1.0, meaningful_failure_miner: 0.1)` (or however the run lands).
- [ ] GitHub PR list on `karpaai/recipe` — accepted PR(s) merged with tag, rejected PR(s) closed-without-merge.
- [ ] HF dataset PR list on `karpaai/proof-bundles` — bundle for king_change merged, bundle for meaningful_failure archived locally (or pushed to a `negative-results/` prefix once that's implemented).
- [ ] Validator log lines showing all three verdicts (or however many actually landed).
- [ ] wandb runs for both miners + validator hidden_eval.

---

## 6. Code changes shipped in this run

- `validator/service.py`: new constants (`KING_CHANGE_WEIGHT`, `MEANINGFUL_FAILURE_WEIGHT`, `PLAIN_FAILURE_WEIGHT`, etc.); new helpers `_diff_is_nontrivial`, `_rationale_is_coherent`, `_classify_outcome`; `score_and_decide` now returns `classification` + `weight_credit`; `run_epoch` uses `weight_credit` for `round_scores` and has a `meaningful_failure` branch that archives to `queue/meaningful_failure/`.
- `tests/test_validator_meaningful_failure.py`: 20 new tests covering all three outcome classes + helper edge cases + the 10% ratio invariant. Full suite 24/24 green.
