# Karpa deep review — 2026-05-31

Multi-agent review (74 agents, 2.16M subagent tokens, 18 min). 57 confirmed findings, 5 refuted. 3 UI proposals → 1 recommendation.

## Executive summary

Karpa is at the end of Phase 0 with a working end-to-end loop on testnet 16, but the 57 confirmed findings reveal that the security and whitepaper-alignment surface is materially thinner than the v1.2 narrative claims. Five severities cluster into one tight story: every artifact a miner controls (checkpoint, patch.diff, attestation, training subprocess) is trusted with too few cross-checks, and several whitepaper §5.4/§5.6/§5.7 mechanisms exist only as docstrings (single attested-execution tier, restricted-path scan on the validator side, audit dispatcher, semantic novelty index, weekly eval rotation). Independent of those, there is one concrete RCE (torch.load weights_only=False on miner checkpoint) and one concrete secret-exfiltration vector (training subprocess inherits the full validator environ including KARPA_BOT_HF_TOKEN/BT_WALLET_PASSWORD) — either is sufficient to refuse mainnet register on its own. The chain layer has two latent bugs that will fail loudly at first contact (subtensor.commit doesn't exist, commit_weights missing salt) and one quiet correctness bug that already affects testnet (set_king implicitly double-calls set_weights and trips the rate limit, silently dropping every meaningful_failure 0.1 credit). The good news: ~80% of critical items are local, low-LOC fixes (allowlisted env, weights_only=True, hash patch.diff, remove redundant cr_enabled branch, hard-reject unverified tier, run the scanner validator-side). The bigger architectural gaps — audit dispatcher wiring, eval rotation, novelty index — are larger but well-scoped. The UI work is genuinely orthogonal to mainnet readiness and should not be allowed to delay register.

## Critical (blocks mainnet) — 13 findings

### 1. torch.load(weights_only=False) on attacker-controlled checkpoint

**Dimension:** validator_3class  
**File:** `/workspace/unicorn/karpathian/karpa/validator/validator.py:226-227`  

Any miner submitting a checkpoint with a pickle reducer (e.g. os.system) gets RCE as the validator UID. Reachable through op2's unverified-tier early-out — does not require forging attestation. Same pattern at audit.py:104,122. Owns the validator, steals the bot HF/GH tokens, exfils the private eval set, forges verdicts. This single bug is enough on its own to block mainnet register.

**Fix:** Switch to weights_only=True, store config in a sibling checkpoint_config.json, bound file size (<5GB), and reject pathological vocab_size/n_layers/dim/max_seq_len from config before instantiating KarpaBase. Fix all three call sites.

### 2. Patched training subprocess inherits full validator environ

**Dimension:** proof_runner  
**File:** `/workspace/unicorn/karpathian/karpa/proof/runner.py:236-244`  

subprocess.run is called with env={**os.environ, 'PYTHONPATH': ...}. The miner controls recipe/train.py via patch.diff — that's the product surface. Every secret in scope (BT_WALLET_PASSWORD, HF_TOKEN, KARPA_BOT_HF_TOKEN with merge rights on karpaai/proof-bundles, KARPA_BOT_GH_TOKEN, WANDB_API_KEY) is readable by miner code and trivially exfiltrable via the training log (which gets published) or socket. On a validator host running audits, the bot tokens leak → adversary can self-merge bundle PRs and rewrite the canonical recipe.

**Fix:** Allowlisted env: PATH/LANG/LC_ALL/HOME/CUDA_VISIBLE_DEVICES/PYTHONPATH plus a small vetted KARPA_* subset. Pop BT_WALLET_PASSWORD from os.environ immediately after BittensorChain.__init__ consumes it. Long-term: Phase 0.5 Docker --network none with cgroup CPU/RAM/wall-time limits and read-only mounts.

### 3. Verified-tier real attestation accepts garbage tokens and quotes

**Dimension:** proof_runner  
**File:** `/workspace/unicorn/karpathian/karpa/proof/real_attest.py:329-376`  

verify_gpu_token calls jwt.decode with options={'verify_signature': False} and returns True for any non-empty string if PyJWT is absent. verify_tdx_quote only checks len >= 256 — never verifies the Intel signature chain, RTMR/MRTD, or report_data binding. So any miner submitting attestation_type='real_nvcc_only' with a 256-byte blob and a 1-byte JWT passes op2 as 'verified'. Once mainnet pays emissions for the verified tier this lets anyone forge it with JSON.

**Fix:** Fail closed: if attestation_type startswith 'real_' but PyJWT or the TDX verifier library aren't linked, return ok=False. Wire verify_gpu_token to NRAS's JWKS, verify_tdx_quote to a real Intel TDX quote verifier (libtdx-attest / trustauthority-py) checking RTMRs against expected_container_measurement and report_data == sha256(nonce||user_data). Until those land, refuse real_* attestation types and only accept signed-by-team mock.

### 4. Mock attestation flagged as 'verified' tier

**Dimension:** whitepaper_alignment  
**File:** `/workspace/unicorn/karpathian/karpa/validator/validator.py:194-197`  

Line 195 reads `tier = 'verified' if is_real else 'verified'` — both branches collapse to verified. The mock HMAC key is sha256(public-constant + container_measurement), and the constant is hardcoded in the open-source repo (mock_attest.py:32). Any miner with a clone can forge a mock attestation that passes op2 as verified-tier. This is the exact denominator attack whitepaper §5.4 promises is architecturally impossible.

**Fix:** Require att.attestation_type.startswith('real_') for tier='verified'; anything else returns ('rejected','mock attestation rejected on mainnet'). Gate behind KARPA_ALLOW_MOCK_ATTESTATION env var so testnet stays loud-but-permissive, mainnet is hard-rejected.

### 5. Unverified-tier submissions still scored (whitepaper v1.2 retired this)

**Dimension:** whitepaper_alignment  
**File:** `/workspace/unicorn/karpathian/karpa/validator/validator.py:147-197`  

When attestation.json is missing, op2 returns (True, ..., 'unverified') with ok=True, bypassing the rejection guard at validator.py:276. The unverified tier is fully plumbed through scoring.py/service.py/router.py/runner.py. Whitepaper v1.2 §5.4 explicitly retired this — single attested-execution tier, no recourse. Shipping mainnet with this branch alive contradicts the published spec and removes the architectural anti-spoofing defense.

**Fix:** When attestation.json is missing return (False, 'attestation chain required', 'rejected'). Delete the tier='unverified' plumbing through scoring.py / service.py / router.py / proof/runner.py. Combine with the mock-as-verified fix and the unified-tier story becomes coherent end-to-end.

### 6. Patch handshake commits are silently dropped (wrong SDK method)

**Dimension:** chain_layer  
**File:** `/workspace/unicorn/karpathian/karpa/chain_layer/bittensor_chain.py:84-118`  

request_handshake_nonce calls self.subtensor.commit(...) which does not exist on bittensor 10.4.0 (the method is set_commitment). The AttributeError is swallowed by `except Exception as e: print(...)` and the function returns a nonce as if successful. Nothing has ever been written on-chain. This is the actual root cause of KARPA_SKIP_HANDSHAKE=1 being required, and it means there is no cryptographic on-chain binding of (miner_hotkey, patch_hash, nonce) — nonces can be replayed/spoofed across hosts.

**Fix:** Replace with self.subtensor.set_commitment(wallet=..., netuid=..., data=commit_hash). Tighten the except to raise (or at minimum log the exception type, not just str(e)) so future SDK breakages fail loudly. Until the commit lands successfully, refuse to return a nonce.

### 7. commit_weights branch is broken (missing salt) and redundant

**Dimension:** chain_layer  
**File:** `/workspace/unicorn/karpathian/karpa/chain_layer/bittensor_chain.py:194-244`  

When commit_reveal_enabled returns True, the code calls commit_weights without the mandatory `salt` argument — the SDK signature is (wallet, netuid, salt, uids, weights, ...). It will raise TypeError on the first CR-enabled subnet (mainnet may flip this at any time) and the bare except swallows it, returning False — validator weights never land. reveal_weights is also missing uids/weights/salt and is never called. Worse: the entire branch is redundant; bittensor's set_weights already handles CR internally via commit_timelocked_weights_extrinsic.

**Fix:** Delete the entire `if cr_enabled:` branch and the reveal_weights method. Always call self.subtensor.set_weights(...) — the SDK does the right thing. If manual commit/reveal is wanted later, persist (salt, uids, weights) and schedule reveal — but that's a separate workstream.

### 8. patch.diff is never integrity-checked vs manifest patch_sha256

**Dimension:** validator_3class  
**File:** `/workspace/unicorn/karpathian/karpa/validator/validator.py:128-144`  

op1's integrity pairs cover checkpoint/training_log/calibration/attestation but NOT patch.diff, even though the manifest has patch_sha256 (runner.py:286). bundle_hash is read from submission.json and never recomputed from disk. A miner produces a valid attested bundle, swaps patch.diff for anything (trivial diff, or a more impressive-looking diff), and the validator accepts the swap. _diff_is_nontrivial then reads the swapped patch. The PR-match fallback only runs with KARPA_BOT_GH_TOKEN set — which testnet 16 did not. This breaks the 'proof of research' claim at its core.

**Fix:** Append ('patch', proof_dir/'patch.diff', manifest['patch_sha256']) to op1's pairs. Recompute bundle_hash from disk and require equality with submission['bundle_hash'] AND manifest['bundle_hash']. Reject if patch_sha256 missing.

### 9. On-chain patch_hash never cross-checked against bundle patch_sha256

**Dimension:** miner_flow  
**File:** `/workspace/unicorn/karpathian/karpa/validator/validator.py:120-126`  

lookup_handshake returns a chain_entry with patch_hash, but op1 only checks chain_entry['miner_hotkey']. grep shows zero validator-side references to patch_hash. A miner can commit one hash on-chain and ship a different patch in the bundle. The cryptographic commitment the whitepaper §5.4 handshake hinges on is observed but unenforced.

**Fix:** After lookup_handshake in op1, assert chain_entry['patch_hash'] == manifest['patch_sha256']. Also include patch_hash in the signed payload of sign_submission so the binding is cryptographic even in KARPA_SKIP_HANDSHAKE mode.

### 10. Runner and validator compute container_measurement differently

**Dimension:** proof_runner  
**File:** `/workspace/unicorn/karpathian/karpa/validator/validator.py:80-97`  

Runner walks RECIPE_DIR for model/recipe/data/configs + karpa_root for eval/calibration/proof. Validator walks only karpa_root for model/recipe/data/eval/calibration/proof and omits configs/ entirely. Layered with the absolute-path bug below, no honest verified-tier submission can ever produce a matching measurement across hosts — every real-tier bundle is rejected at op2:193. Testnet 16 only hid this because no attestation was ever submitted (unverified early-out). The moment we hard-reject unverified per the v1.2 fix, EVERY honest submission fails until this is fixed.

**Fix:** Extract _list_proof_sources into a single shared proof/sources.py imported by both runner.py and validator.py. Accept both karpa_root and recipe_dir. Walk the union including 'configs'. Add a test that calls compute_container_measurement on the same checkout from both code paths and asserts equality.

### 11. container_measurement hashes absolute filesystem paths

**Dimension:** proof_runner  
**File:** `/workspace/unicorn/karpathian/karpa/proof/mock_attest.py:91-100`  

compute_container_measurement does h.update(str(path).encode()) where path is an absolute Path. A miner at /home/ubuntu/karpa/... and a validator at /workspace/.../karpa/... produce different digests over byte-identical content. Stacked on top of the runner-vs-validator source-set mismatch above, this guarantees verified-tier rejection across machines. Together they explain why KARPA_SKIP_HANDSHAKE=1 has been mandatory.

**Fix:** Hash repo-relative POSIX paths only: h.update(path.relative_to(base).as_posix().encode()). Sort relative paths lexicographically for canonical ordering. Until Docker is actually wired (Phase 0.5+), rename the field to source_tree_measurement so the on-chain semantics aren't misrepresented.

### 12. Validator never runs the restricted-file diff scanner

**Dimension:** whitepaper_alignment  
**File:** `/workspace/unicorn/karpathian/karpa/validator/validator.py:100-144`  

Whitepaper §5.4 Operation 1 explicitly assigns the restricted-path scan to the validator. The validator's op1 only does sig/handshake/hash checks — grep for scan_diff_for_restricted in validator/ returns zero hits. The scanner only fires miner-side at runner.py:184 — i.e. the entity being verified. Concrete exploit: a miner whose proof/runner skips the scan ships a patch.diff modifying calibration/ to fake low matmul_ms, inflating the cost-normalized score; service.py:276 feeds calibration straight from the miner-submitted file.

**Fix:** In op1_diff_and_integrity, after the signature check, load restricted_files.yaml on the validator side, read bundle_dir/'patch.diff', call scan_diff_for_restricted, and return rejected on any hit. Must run on the validator, not rely on miner's runner output.

### 13. Audit pipeline (§5.7) is completely unwired

**Dimension:** whitepaper_alignment  
**File:** `/workspace/unicorn/karpathian/karpa/validator/service.py:320-587`  

run_audit exists in validator/audit.py but is never called from service.py — grep confirms zero callers outside its own __main__. No 10% random sampling, no king-margin auto-audit, no probation counter, no blacklist/zero-weight path in chain_layer. The §5.7 deterrence math `EV(fraud) = 0.9·R − 0.1·(S+F+E)` collapses to ~0.9·R in the running code. Without this, every other gaming defense is moot because there's no consequence to being caught.

**Fix:** Add validator/audit_scheduler.py that, after score_and_decide returns accepted=True or king-margin distance < noise_floor, enqueues an audit job. Persist per-hotkey audit history in chain_layer to drive probation and reputation discount. On audit fail, append submission_fraud event and call new chain.blacklist(hotkey) that zeros that hotkey's weight in subsequent set_weights calls.

## High — 15 findings

### 1. scan_diff_for_restricted bypasses (rename headers, tab-suffix on exact-match patterns)

**Dimension:** proof_runner  

Empirically reproduced: a pure git rename (`rename from eval/score.py / rename to ...`) leaves the scanner returning [], but `patch -p1` (the actual applier at runner.py:126) understands rename headers and applies it. Separately, `--- a/restricted_files.yaml\t2024-...` survives the strip and bypasses the exact-match branch of _path_matches — so restricted_files.yaml itself can be modified. eval/** prefix patterns are NOT bypassed by tab-suffix (the reviewer's third claim is wrong) but the two confirmed paths are enough to defeat the integrity gate. _path_matches also lacks fnmatch support so any future pattern containing `*` silently fails closed-open.

**Fix:** Parse with unidiff.PatchSet or pivot to `git apply --numstat --check` for the authoritative file list. Scan rename from/to, copy from/to, diff --git, Index: lines. Strip tabs from paths (split on '\t', take head). Normalize via Path(p).as_posix() and os.path.normpath. Add tests/test_restricted_scanner.py covering rename, tab-suffix, --no-prefix, c/ prefix, quoted filenames, /dev/null.

### 2. Patched training subprocess has no filesystem/network sandbox

**Dimension:** proof_runner  

Beyond env leakage, the subprocess runs as the validator UID with unrestricted FS + network reach. It can rewrite the validator's own karpa_root (restricted_files.yaml is scanned only in the diff, not protected from runtime writes), drop a cron entry, or fetch a pre-trained checkpoint of the right shape over the network — defeating the supposed compute attestation in the unverified tier. The runner.py:18-22 comment acknowledges Docker is Phase 0.5+ future work; until it's there, mainnet should not register.

**Fix:** Bubblewrap/firejail/Docker with --network none, read-only mounts for the canonical recipe (workdir bind-mounted ro except train_out + a scratch tmpfs), cgroup CPU/RAM/wall-time. Reject any training log claiming reads outside workdir/train_out.

### 3. NaN/Inf metrics can crown a permanent null king

**Dimension:** validator_3class  

scoring.py never checks math.isfinite(val_bpb)/math.isfinite(benchmark_accuracy). The first-submission branch at service.py:281-282 forces decisively=True regardless, so a NaN bundle becomes the first king and from then on every comparison is NaN-vs-NaN — chain state permanently corrupted. eval/val_bpb.py:79 explicitly produces float('inf') when total_bytes==0, so non-finite values are reachable. Note: training-loss has a finite guard at validator.py:209 but eval val_bpb does not.

**Fix:** Top of score_bundle: return non-decisive ScoreReport with quality_gain=0 if not finite. In service.py score_and_decide: hard-reject with reason='nonfinite_metrics' if hidden_eval returns non-finite. Never crown a NaN king.

### 4. set_king implicitly double-calls set_weights, tripping the rate limit

**Dimension:** chain_layer  

BittensorChain.set_king (bittensor_chain.py:277) calls self.set_weights({king: 1.0}) after writing king.json. service.py:585 then calls chain.set_weights(round_scores) with the full per-miner distribution at end of epoch. With weights_rate_limit >= 100 blocks, the second call returns False silently — only {king: 1.0} lands on-chain. Net result: meaningful_failure miners never get their 0.1 credit on any king-change epoch. Already affecting testnet behavior whether the user has noticed it or not.

**Fix:** Drop the implicit set_weights from BittensorChain.set_king. service.py end-of-epoch call becomes the single authoritative writer. LocalChain.set_king has no such call — fix matches parity. Update interface.py docstring.

### 5. Hidden-eval set is static — no rotation, no commit-reveal, no pool refresh, no canary re-score

**Dimension:** whitepaper_alignment  

Whitepaper §5.7 mandates: weekly active-subset rotation via on-chain commit-reveal beacon, monthly pool refresh replacing the oldest 10%, canary re-score of merged patches against next rotation with demotion path. eval/private/ contains exactly one active_tokens.bin. grep across all .py for rotation|active_subset|next_active|monthly|weekly returns ONLY the docstring at hidden_eval.py:8. The hits for commit_reveal in bittensor_chain.py are for Bittensor weight commit-reveal (different system), not §5.7. Single biggest anti-overfit defense, completely unimplemented.

**Fix:** (1) eval/private/pool/ with 10x active subset + pool_manifest.json; (2) validator/rotation.py running at epoch boundary, draws next_active under commit-reveal via chain.commit(hash(next_active)) / chain.reveal(next_active); (3) monthly pool-refresh worker; (4) validator/canary.py re-scores current king vs new active subset, demotes via chain.set_king(previous_king) on regression > 2*noise_floor.

### 6. No semantic novelty / similarity check on meaningful_failures

**Dimension:** whitepaper_alignment  

Whitepaper §5.6 promises 'Validators query the index when scoring a non-winning submission: if it is too close to an existing entry, the submission earns no share of the 10%.' _classify_outcome uses only structural heuristics (char count, paragraph count, sentence-uniqueness within one rationale). grep -E 'novelty|near.dup|cosine|MiniLM|sentence-transform' across validator/ returns zero hits. A miner can paraphrase the same failure rationale every round and farm 0.1 weight indefinitely. This is the documented anti-flood defense for the 10% pool.

**Fix:** validator/novelty.py with an embedding index (sentence-transformers/all-MiniLM-L6-v2) over all meaningful_failure rationales from chain.events. Expose novelty_score(submission) -> float. Add as Bar 4 in _classify_outcome with threshold cosine_sim > 0.9 -> reject.

### 7. scoring.py still implements v1.1 two-tier α=0.5/1.0 that v1.2 §5.4 retired

**Dimension:** whitepaper_alignment  

ALPHA_VERIFIED=1.0/ALPHA_UNVERIFIED=0.5 still defined at scoring.py:21-22 and used at scoring.py:99-100 (cost_effective = cost_h100h / alpha). Whitepaper v1.2 §5.4/§5.5 explicitly removes the credibility factor. docs/whitepaper_v1.2_updates.md §B.4 flags this exact passage. Mainnet ship with this code advertises a one-tier architecture while shipping the two-tier penalty.

**Fix:** Remove ALPHA_* constants, drop the `tier` parameter from score_bundle, set cost_effective = cost_h100h unconditionally. Reject unverified at op2 instead of discounting.

### 8. Validator never checks miner hotkey is registered on the subnet

**Dimension:** chain_layer  

score_and_decide / run_epoch / op1 never call chain.is_hotkey_registered (the helper exists at chain_layer/interface.py:50 and is only used in scripts/miner_run.py:122). Unregistered hotkeys with a valid Ed25519 signature run all the way through op3+op4+hidden_eval (each ~$1+ on H100). The chain.set_weights uid==None filter prevents emission leak, but compute-grief and corpus pollution (submission_scored, meaningful_failure_archived events flow to dashboard) are real.

**Fix:** if not chain.is_hotkey_registered(miner_hotkey): return rejected('hotkey_not_registered') at the top of score_and_decide. Cache metagraph for the duration of one epoch.

### 9. Ed25519 miner private keys written world-readable (mode 0644)

**Dimension:** security  

miner/submit.py:99-108 _ensure_keypair calls sk_path.write_bytes(sk_bytes) with no chmod and no umask change. On disk: all 9 existing *.sk files are mode 644, keys dir is 755. sign_submission signs (miner_hotkey|nonce|bundle_hash). Any local UID on shared Shadeform/Hyperstack VMs can forge signatures.

**Fix:** sk_path.chmod(0o600), keys_dir.chmod(0o700). Retrofit existing files: chmod 600 miner/keys/*.sk && chmod 700 miner/keys. Wrap write in os.umask(0o077) for defense in depth. Add startup audit that refuses to load keys not at 0600.

### 10. .env world-readable, secrets exported into every subprocess

**Dimension:** security  

docs/h100_miner_setup.md tells miners to `cat > .env <<EOF ... EOF` (no umask wrapper, no chmod follow-up). Verified: karpa/.env is mode 0644 with BT_WALLET_PASSWORD + HF_TOKEN. Combined with `set -a && source .env && set +a` (line 73), secrets propagate into wandb/torch/HF child processes. chain_layer/config.py:_load_dotenv emits no permissions warning.

**Fix:** Docs: `(umask 077 && cat > .env <<EOF ... EOF) && chmod 600 .env`. chain_layer/config.py:_load_dotenv: warn loudly if .env st_mode & 0o077. Add SECURITY.md / 'Token hygiene' section.

### 11. Subprocess stdout/stderr re-raised verbatim — leaks into validator.log AND on-chain events

**Dimension:** security  

runner.py:243-244 raises RuntimeError(f'training failed:\nstdout:\n{...}\nstderr:\n{...}'). service.py:352 logs it AND service.py:360 commits `error: str(e)` via chain.append_event — meaning unredacted subprocess output (potentially including any libdump of os.environ from wandb/torch/HF) can land in events.jsonl which feeds the public dashboard. A _redact() helper already exists at miner/github_pr.py:50 but is unused in proof/runner.py. Worse than the original 'local log leak' framing.

**Fix:** Reuse miner/github_pr.py::_redact, instantiate at startup with the tuple of known secrets from os.environ (BT_WALLET_PASSWORD, HF_TOKEN, KARPA_*_TOKEN, SHADEFORM_API_KEY). Run captured stdout/stderr through it before embedding in RuntimeError. Combined with the env-allowlist fix (so secrets aren't visible to subprocesses in the first place), this is belt-and-braces.

### 12. Round_scores discarded if set_weights fails (rate limit / RPC error)

**Dimension:** chain_layer  

round_scores is a local dict in run_epoch (service.py:343). archive_bundle moves bundles out of pending/ at line 575 BEFORE set_weights is attempted at line 585. If set_weights returns False (rate-limited, RPC down, exception swallowed by bittensor_chain.py:227-229), all per-submission deltas for that epoch are lost — poll_queue only scans pending/, never re-derives weights from scored/ or from chain.append_event submission_scored records. The king floor (1.0 re-added every epoch at lines 581-583) masks the bug; the 0.1 meaningful_failure credits are silently dropped. Two-validator deployments will diverge.

**Fix:** Persist round_scores to chain_dir/pending_weights.json after scoring each bundle. On epoch start, load and merge into the new round_scores; clear only after confirmed-successful set_weights. Treat archive_bundle('scored') as committing score metadata, not weight credit.

### 13. MEANINGFUL_FAILURE_WEIGHT=0.1 per submission instead of §5.6 rank-split of 10% pool

**Dimension:** whitepaper_alignment  

Whitepaper §5.6: '10% goes to meaningful contributions… ranked by validator-assessed informativeness. If multiple qualify, they split the 10% by rank.' Code emits a constant 0.1 per qualifier with no ranking. After chain.set_weights L1-normalizes, 1 qualifier ≈ 90/10, 5 qualifiers ≈ 67/33, 10 qualifiers ≈ 50/50. The 'don't pay out if none qualify' rule cannot be honored because there's no fixed pool. Also, the 3-class king_change/meaningful_failure/plain_failure labels do not appear anywhere in v1.2 — documentation gap on top of mechanism mismatch.

**Fix:** (a) Whitepaper text: either describe the 3-class classifier explicitly or rename code labels to match the binary 'meaningful' bar. (b) run_epoch: split king_weight=0.9 and pool=0.1, allocate pool across qualifying meaningful_failures by descending validator-assessed score (until a real ranker exists, divide equally). Add a test.

### 14. No CI workflow — tests never run automatically

**Dimension:** tests  

No .github/workflows in karpa/. The 25 pytest functions only run if a developer remembers. Sibling subnet repos (oro, trajrl, trajrl-bench) all ship CI YAMLs. The configs/*.json gap in _diff_is_nontrivial broke once already and was caught retrospectively — a single CI run with adversarial bundle fixtures would catch the next one before H100 spend.

**Fix:** Add .github/workflows/ci.yml: pip install -e .[dev] && pytest -v && ruff check (lint config already exists at pyproject.toml:51-56 but is ungated). Matrix Python 3.10/3.11/3.12 per requires-python.

### 15. Zero tests on sign_submission / verify_signature, scan_diff_for_restricted, score_bundle

**Dimension:** tests  

sign_submission and verify_signature (the protocol's identity binding) have zero coverage; verify_signature swallows every Exception as False, so a bug is indistinguishable from a forgery. scan_diff_for_restricted (the only gate protecting eval/, calibration/, validator/) has zero adversarial tests despite the rename/tab-suffix bypasses above. score_bundle is the formula whose constants are tunable per design v1; a typo would silently change subnet economics. Existing tests cover only _classify_outcome / _diff_is_nontrivial / _rationale_is_coherent in isolation; no integration test exercises run_epoch end-to-end.

**Fix:** tests/test_submit_signing.py (valid roundtrip; tampered bundle_hash/nonce/pubkey/malformed hex → False, error path is logged not silently swallowed). tests/test_restricted_scanner.py (rename, tab-suffix, no-prefix, c/ prefix, quoted, /dev/null, every restricted_files.yaml pattern). tests/test_scoring.py (table-driven over king_bpb/val_bpb/tier combos). tests/test_run_epoch_integration.py (LocalChain + 3-4 prebaked bundles, asserts events.jsonl + king.json + archive dir state after run_epoch). Mock judge_submission to avoid a real training run.

## Medium — 13 findings

### 1. Round_scores lost on mid-epoch crash

**Dimension:** validator_3class  

Even if set_weights succeeds, a validator crash between archive_bundle and set_weights leaves N already-archived submissions with no on-chain weight credit. On restart they're in scored/, not pending/, so they're never rescored. King self-heals via the next-epoch floor; meaningful_failures silently lose. Replay un-flushed scored bundles by cross-referencing submission_scored events without matching weights_set events on validator restart.

### 2. Intra-epoch king change is order-dependent (alphabetical bundle_id sort)

**Dimension:** validator_3class  

poll_queue sorts by bundle_id alphabetically; chain.get_king() is re-read per bundle inside the loop, and chain.set_king() mutates the king mid-epoch. Two bundles in the same epoch compete head-to-head depending on bundle_id ordering. Fix: freeze start_of_epoch_king once at the top of run_epoch and pass to score_and_decide; alternatively record king_val_bpb_at_compare on submission_scored events for reconstruction.

### 3. KingRecord.proof_dir points to queue/pending/ then bundle moves to queue/scored/

**Dimension:** validator_3class  

set_king is called at service.py:441 with proof_dir=str(bundle_dir) where bundle_dir is the pending path; archive_bundle at service.py:575 moves it to scored/ right after. king.json on disk now has a permanently-broken path, and previous_king inherits the brokenness across rotations. Fix: call set_king AFTER archive_bundle with the final path, or store bundle_hash + HF URL instead of a filesystem path.

### 4. Huge val_bpb improvement + tiny benchmark regression demoted to meaningful_failure

**Dimension:** validator_3class  

decisively_beats_king at scoring.py:93-97 uses a single scalar noise_floor_margin for both axes. With +0.5 bpb gain but -0.02 benchmark, both clauses fail; in _classify_outcome the bundle then satisfies the meaningful_failure bars and earns 0.1 instead of 1.0 (and the king isn't dethroned). The noise_floor_margin is calibrated against val_bpb seed noise (0.013), not benchmark noise — a hidden second issue. Fix: define a Pareto-improvement with axis-specific margins, or relax benchmark slack proportionally to bpb gain. Add a test pinning the new contract.

### 5. Patch apply: lenient `patch -p1` (fuzz allowed) vs strict `git apply` in github_pr

**Dimension:** miner_flow  

Same diff can succeed in the proof-test and fail when the PR is opened, accepting a verdict for a patch that GitHub will then reject. Yesterday's run almost hit this exact case. Switch proof/runner.py to `git apply --check` + `git apply --whitespace=nowarn` against a scratch git init of workdir; refuse fuzz>0 so verdict matches PR-acceptability.

### 6. events.jsonl / handshakes.jsonl appends not multi-writer safe

**Dimension:** chain_layer  

No fcntl.flock; Python text-mode 'a' isn't guaranteed atomic for writes > PIPE_BUF. miner_run.py and validator share karpa_root/chain. Interleaved bytes -> bad JSON -> lookup_handshake / get_events crashes mid-iteration. Fix: fcntl.LOCK_EX on writes, try/except json.loads in readers; long-term replace JSONL with SQLite.

### 7. wait_for_finalization=False + append-only event log diverges from chain on reorgs

**Dimension:** chain_layer  

set_weights and commit_weights log success=True after inclusion. On reorg the extrinsic vanishes but events.jsonl says it landed. set_king has the same issue. Add a status='pending' → 'finalized' / 'reverted' reconciliation pass, or switch to wait_for_finalization=True.

### 8. --token / --hf-token CLI flags expose secrets via argv / ps aux

**Dimension:** security  

Five callsites (validator/service.py:599, hf_poller.py:198, miner/hub.py:207+213, scripts/miner_run.py:325). github_pr.py:192 already acknowledges the argv-leak threat model for the PAT push; no equivalent guard on HF tokens. Either remove the flag and force env-only, or warn when value didn't come from env default. Document the rule in docs/h100_miner_setup.md.

### 9. No documented token-rotation procedure; KARPA_BOT_HF_TOKEN silently falls back to HF_TOKEN

**Dimension:** security  

validator/service.py:545 falls back to HF_TOKEN for PR merges, collapsing the write/merge privilege separation. No SECURITY.md, .env.example doesn't list KARPA_BOT_HF_TOKEN. Add a SECRETS section with per-token scope / blast radius / rotation steps. Refuse to merge in service.py if only HF_TOKEN is set — log_err and skip.

### 10. github_pr / github_bot raise GitHub error bodies without _redact()


_gh_request at github_pr.py:42-47 raises 'github {method} {path} → {code}: {detail}' with no _redact, unlike the sibling _run which does. Caller print()s it. GitHub doesn't echo Authorization headers in 4xx bodies today, but the asymmetry is one defensive layer thinner than necessary. Wrap with _redact(text, secrets=(token,)). Same fix in validator/github_bot.py:_gh.

### 11. Hero SVG aria-hidden with no textual data fallback

**Dimension:** karpa_ui_landing  

HeroPlot.tsx:327 has aria-hidden=true and no <title>/<desc>; figcaption gives no numeric values. Real WCAG 1.1.1 / 1.3.1 gap on the page's marketing centerpiece. Add <title>/<desc> with the actual king count and bpb deltas; drop aria-hidden. Add aria-labelledby on the figure pointing to the figcaption id.

### 12. layout.tsx metadata exports only title+description

**Dimension:** karpa_ui_landing  

No openGraph, no twitter card (which directly harms the karpaai twitter posts workflow — link unfurls show as plain text), no metadataBase, no themeColor, no viewport export. Add metadataBase, openGraph with /og.png 1200x630, twitter summary_large_image with site:@karpaai, viewport with themeColor. Generate /og.png and check in.

### 13. Adversarial input gaps: patch.diff size cap + unicode normalization on rationale

**Dimension:** tests  

_diff_is_nontrivial at service.py:139-162 calls patch_path.read_text() unconditionally with no size cap — rationale has a 200KB cap at service.py:492 for the same defense, asymmetric. _rationale_is_coherent doesn't normalize unicode so ZWJ padding can satisfy the length threshold. (Reviewer's path-traversal and zip-bomb sub-claims are not real — every path is constructed inside the validator, and no bundle component is decompressed.) Add 10MB cap on patch.diff; unicodedata.normalize('NFKC', text) and strip zero-width chars before length check. Property tests with hypothesis are cheap to add.

## Low — 8 findings

### 1. patch_hash encoding drift on non-UTF-8 / CRLF


miner_run.py:141-143 reads text and hashes the re-encoded bytes; runner.py:286 hashes the on-disk file. On UTF-8 Linux they agree; on a Windows / non-UTF-8 locale miner they could drift. Combined with the (recommended) cross-check from validator finding, an honest non-Linux miner could get rejected. Read patch as bytes throughout: `patch_bytes = patch_path.read_bytes(); target_patch.write_bytes(patch_bytes); patch_hash = sha256(patch_bytes)`.

### 2. HF repo override foot-gun: typo'd KARPA_HF_REPO uploads to wrong namespace


(Original finding's silent-namespace-rewrite claim was wrong — HF returns 403 when create_repo lacks write to karpaai/, and community PRs still work without write.) The real issue is the --hf-repo / KARPA_HF_REPO override: a typo'd org name silently uploads under the miner's namespace and the validator polls only karpaai/proof-bundles. Submission silently lost. Drop create_repo from upload path; warn loudly if resolved repo_id != canonical.

### 3. External links missing rel="noreferrer"


17 components use rel="noopener" only. noopener alone blocks tabnapping; missing noreferrer only leaks the Referer header to GitHub/HF. Minor polish, not a security flaw. Add noreferrer across all <a target="_blank"> or factor an <ExtLink> wrapper.

### 4. Dark-mode CSS variables are unreachable dead code


globals.css:40-53,84-89 define :root[data-theme='dark'] but data-theme is never set, no prefers-color-scheme block. ~30 lines of dead CSS. Either add @media (prefers-color-scheme: dark) { :root { /* alias to dark vars */ } } or delete the dark block until a toggle ships.

### 5. scroll-behavior: smooth ignores prefers-reduced-motion


globals.css:56 sets smooth scroll globally; the existing @media (prefers-reduced-motion: reduce) block only resets the reveal animation, not html scroll-behavior. One-line fix inside the existing block: `html { scroll-behavior: auto; }`.

### 6. K-mark SVG duplicated in 3 places, drifts in dark mode


icon.svg and public/assets/karpa-mark.svg hard-code navy hex codes; KarpaMark.tsx uses CSS vars. In dark mode the inline SVG recolors, the favicon and static asset don't. public/assets/karpa-mark.svg + .png exports are unused (no <Image>/<img> references). Delete the unused static assets; document that icon.svg edits must mirror KarpaMark.tsx.

### 7. Mobile hero-meta grid lacks row-2 border-top at <=600px


Cells 3+4 wrap to row 2 with only the central vertical rule continuing; no horizontal divider. Minor polish. Add `border-top: 1px solid var(--rule); padding-top: 18px` for nth-child(3),(4) inside the <=600px block. (Reviewer's parallel .cols3 claim is wrong — .col-art:nth-child(2) border-top already cascades from <=920px.)

### 8. /root/.shadeform_api_key + .env have no mode check at load time


scripts/gpu.py:51-55 reads the API key with no st_mode validation; chain_layer/config.py:_load_dotenv same. Currently 0600 on the box, but rsync/cp to another machine could leave them 0644. Defense-in-depth: `if KEY_FILE.stat().st_mode & 0o077: raise SystemExit('chmod 600 it')`.

## UI recommendation

**Recommended angle:** The Karpa Quarterly (journal angle), with one operational-dashboard tile (/live) hosted inside the journal shell

Karpa's whole differentiator is 'proof of research' — an attested scientific corpus where verified-negative results compound. The journal angle is the only one of the three that makes the corpus itself the product, which is what the whitepaper actually promises. The operational dashboard sells a Bittensor abstraction (epoch, weights, validator health) that taostats already serves, and the researcher-search angle (corpus #3) is the right north star but premature — it depends on an indexer + embedding pipeline + LLM answer service that's at least a quarter of work before anything ships. The journal hits Karpa's voice (the existing karpa-ui Spectral serif + .figure-frame + .ledger-row design language carries directly into product surfaces, no rebrand), gives miners a public 'look what your patch became' page within days, and creates the canonical /experiment/[bundle_hash] permalink that future research-search work will inevitably need anyway. Hybrid hedge: borrow ONE tile from the dashboard angle — a small /live status strip in the masthead (current king bpb, blocks-since-set-weights, validator health pill) — so operators get the heartbeat without the journal losing its editorial register.

**First milestone (v0):** Ship /king as a single static Next.js page in karpa-ui, reading a hand-maintained public/king.json (schema in karpa/corpus_schema.py — one CLI command emits it from queue/scored). The page renders editorial header, one .figure-frame with HeroPlot.tsx ported to KingPlot.tsx (val_bpb trajectory vs prior king), rationale.md verbatim as the abstract, .ledger-row metrics block, bundle_hash / HF link / miner SS58 in the sidebar. One PR. No API server, no DB, no live chain reads. If engagement is real (X traction on the karpaai posts referencing /king, miners linking to it from their own bios, citations in any external Bittensor / ML writeup within ~30 days), build out /issues + /recipe/[tag] next. If nobody clicks, you've spent 1-2 days, not 3 months.

**Alternatives considered:** Operational dashboard (#2) is the easiest to copy from existing subnets (taostats, dolphin.bittensor, etc.) and the hardest to differentiate. Researcher-corpus (#3) is the right end-state but the v0 (table view of bundles) is exactly what the journal's /corpus page becomes anyway — and the journal frames each row as an article instead of a row, which is the more durable framing. Pure hybrid was tempting but the visual languages clash (dark monospace operational vs ivory serif editorial); embedding a small live tile inside the journal masthead is the cleanest reconciliation.

**First 4 routes to build:**
- /king — editorial 'front matter' for the current canonical recipe (v0 milestone above)
- /recipe/[tag] — single-article view: the citeable atom of the whole product, where every miner patch lives forever with a permalink
- /issues — chronological table-of-contents grouping each king change with its contained meaningful_failures (this is the route that makes negative results visually first-class)
- /corpus — filterable table-of-articles fallback for users who didn't find what they wanted via /issues; this is the route where #3's researcher angle starts to germinate

## Phase 1 readiness assessment

Not ready for mainnet register. Phase 0 produced a working end-to-end loop and the 3-class classifier landed on testnet 16 as designed — that's a real milestone. But the security and whitepaper-alignment surface has 13 critical findings that fall into two camps: (1) miner-controlled artifacts that are trusted with too few cross-checks (RCE via torch.load, secrets via subprocess env, swappable patch.diff, unenforced on-chain handshake patch_hash), and (2) whitepaper §5.4/§5.6/§5.7 mechanisms that exist as docstrings rather than code (single attested-execution tier, validator-side restricted-path scan, audit dispatcher, eval rotation with commit-reveal, semantic novelty index). The mock-attestation-as-verified bug alone means that the moment Karpa starts paying emissions, anyone with the repo can forge a valid mock attestation and farm. The torch.load RCE alone means a single malicious miner can own every validator host and steal the bot HF/GH token (which has merge rights on the canonical recipe — total subnet capture). Two of the three highest-impact items are local fixes (allowlist env in runner.py, weights_only=True in validator.py + audit.py); the other is the unverified-tier rejection. Realistic timeline: ~2 weeks of focused work clears the critical list. The bigger architectural items (audit dispatcher, eval rotation, novelty index) need 4-6 weeks but are well-scoped. UI work is genuinely orthogonal — don't let it delay register.

### Mainnet-register blockers
- [ ] torch-load-rce-on-miner-checkpoint — must land before any other validator runs untrusted bundles
- [ ] training-subprocess-inherits-validator-environ — paired with the RCE above, env-allowlist is non-negotiable
- [ ] real-attestation-verification-no-op — verify_tdx_quote and verify_gpu_token must fail-closed or be removed until real verification is wired
- [ ] mock-attestation-counts-as-verified — gate mock behind KARPA_ALLOW_MOCK_ATTESTATION env var; default-reject on mainnet
- [ ] unverified-tier-still-accepted — hard-reject when attestation.json missing per v1.2 §5.4
- [ ] subtensor-commit-method-does-not-exist — replace with set_commitment, raise on failure; this unblocks dropping KARPA_SKIP_HANDSHAKE
- [ ] commit-weights-missing-salt — delete the broken redundant cr_enabled branch, let SDK's set_weights handle CR
- [ ] patch-diff-not-integrity-checked — append patch.diff to op1's integrity pairs, recompute bundle_hash on disk
- [ ] patch-hash-onchain-not-enforced — assert chain_entry['patch_hash'] == manifest['patch_sha256'] in op1, include patch_hash in sign_submission
- [ ] container-measurement-mismatch-runner-vs-validator + container-measurement-includes-abspath — shared _list_proof_sources module, hash repo-relative POSIX paths, add cross-host equality test (these two MUST land together with the unverified-tier rejection — otherwise every honest submission fails)
- [ ] validator-no-restricted-file-scan — invoke scan_diff_for_restricted in op1 on the validator side
- [ ] audit-not-wired-up — at minimum the random-sample dispatcher + zero-weight blacklist before mainnet pays emissions; the §5.7 deterrence math requires real consequences
- [ ] set-king-double-set-weights — drop the implicit set_weights from BittensorChain.set_king (silently breaks meaningful_failure economics on every king-change epoch)
- [ ] two-tier-alpha-still-in-scoring — remove ALPHA_* constants, drop tier parameter, set cost_effective=cost_h100h unconditionally
- [ ] no-nan-inf-guard-on-val-bpb-benchmark — reject non-finite metrics before they can crown a permanent null king
- [ ] miner-private-keys-world-readable — chmod 600 on key write + retrofit existing keys + reject mode!=0600 at load time
- [ ] round-scores-lost-on-rate-limit — persist pending_weights.json, clear only after confirmed set_weights success
- [ ] scan-diff-restricted-untested + restricted-scanner-bypass-* — replace the homegrown scanner with unidiff/git-apply --numstat, add adversarial test corpus (rename, tab-suffix, quoted, /dev/null)
- [ ] no-ci-workflow + signing-roundtrip-untested + integration-test — minimum CI before mainnet: pytest + ruff + a single run_epoch integration test with LocalChain

### Nice-to-haves (post-register / Hardening 1.5)
- [ ] no-eval-rotation — high impact, larger workstream (~3-4 weeks) but the single biggest anti-overfit defense; if Phase 1 ships without it, schedule it for the first hardening release post-register
- [ ] no-novelty-index — meaningful_failure economics are vulnerable to paraphrase-farming without it; can ship a v0 with bge-small + cosine_sim>0.9 threshold in ~1 week
- [ ] meaningful-failure-weight-not-90-10-split — whitepaper-alignment fix; either rewrite §5.6 to describe the 3-class classifier or refactor run_epoch to honor the 90/10 fixed-pool split
- [ ] training-subprocess-no-filesystem-sandbox — Docker --network none is the right Phase 0.5 hardening, but the env-allowlist fix alone covers the immediate threat
- [ ] huge-bpb-improvement-but-bench-regression — tune decisively_beats_king to allow asymmetric axis margins; can iterate on testnet
- [ ] intra-epoch-king-change-ordering — freeze start_of_epoch_king at top of run_epoch; mostly affects multi-king-change epochs which are rare today
- [ ] wait-for-finalization-false-vs-reorg — add status='pending' -> 'finalized'/'reverted' reconciliation pass
- [ ] events-jsonl-not-multi-writer-safe — fcntl.LOCK_EX on writes, try/except in readers; eventually SQLite
- [ ] documented token-rotation procedure + KARPA_BOT_HF_TOKEN required-not-fallback — SECURITY.md with per-token scope/blast/rotation table

## Open questions for user

1. Are you willing to slip mainnet register by ~2 weeks to land the 5 most-load-bearing critical fixes (torch.load weights_only=True, subprocess env allowlist, hard-reject unverified tier, restricted-path scan on validator side, patch_hash on-chain cross-check), or do you want a tighter '1 week of fixes, ship with known gaps' plan with a public Hardening Phase 1.5 commitment?

2. Does the whitepaper text in §5.6 actually mean a fixed 10% pool split by rank (per the literal reading), or does it mean 'each qualifying meaningful failure is worth roughly 10% of a king change' (closer to what the code does)? The two interpretations imply different fixes — clarify before refactoring run_epoch.

3. For Phase 1 launch, do you want to ship with mock attestation accepted on mainnet (gated behind KARPA_ALLOW_MOCK_ATTESTATION=1, loud warning) so miners without TDX/H100-CC can still participate, or hard-reject mock and require real_nvcc_only from day 1? This decision drives whether the real_attest.py verification work is mainnet-blocking or post-register.

4. Is the bot HF account (karpaai-bot or whatever owns KARPA_BOT_HF_TOKEN) using a write-restricted token scoped only to karpaai/proof-bundles, or is it an admin token? If admin, rotate to a write-only token before any further validator runs — independent of the subprocess-env fix.

5. For the UI work, can the v0 /king page be wired to a single hand-maintained public/king.json (1-2 days of work, no backend), or do you want the indexer service from day 1? My recommendation is the static v0 to validate engagement first, but if you already have indexer infrastructure half-built, the calculus changes.

6. Should the audit dispatcher (run_audit wiring + 10% random sampling + king-margin auto-audit + blacklist) be mainnet-blocking, or is it acceptable to register first with manual/honor-system audits and ship the dispatcher in Hardening 1.5? The §5.7 deterrence math collapses to ~0.9·R without it, but no actual fraud has happened on testnet yet so the urgency is theoretical.

## Refuted findings (credibility marker)

Five initial findings were caught and refuted during verification — worth flagging as a credibility marker for the rest of the report. The 'legacy king with king_benchmark=None' scenario doesn't exist because KingRecord.benchmark_accuracy is typed `float` (not Optional) and both call sites consistently guard against king-absence. The 'github_pr clones the fork without PAT' panic is wrong because karpaai/recipe is public and forks of public repos are public on GitHub. The 'validator handshake bypasses chain abstraction' finding is a restatement of the already-known KARPA_SKIP_HANDSHAKE / chain.commit-not-wired followup. The '_list_proof_sources extension allowlist misses .pth Python hooks and binary patches' claim was based on factually wrong premises — Python doesn't process .pth files on PYTHONPATH (only site-packages), and `patch -p1` doesn't support git binary diffs. And the 'next-env.d.ts breaks production build' claim was empirically refuted — TypeScript treats unresolved side-effect .d.ts imports as non-fatal, and `next build` regenerates the file before typecheck anyway. The 57 confirmed findings have all been verified against the actual code; the refuted five give confidence that nothing in the confirmed list is hallucinated.