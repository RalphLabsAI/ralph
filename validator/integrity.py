"""Checkpoint-trainedness / log-consistency guard.

Motivation (real incident): a submission whose checkpoint was random-INITIALISED
(every weight at init std) shipped a `training_log.jsonl` claiming a full run and
got crowned — because op4 was scoring against random tokens at the time and the
re-train audit that would have caught the log/checkpoint mismatch never ran. The
checkpoint measured ~ln(vocab) nats/token (uniform output) yet the log claimed a
final loss of ~3 nats.

This is a CHEAP guard (no extra GPU work — it consumes the val_bpb op4 already
computes plus the miner's own declared `final_loss`):

  (a) UNTRAINED: the held-out loss sits within `random_fraction` of the random
      baseline ln(vocab_size) -> the checkpoint carries ~no learned signal.
  (b) LOG/CHECKPOINT MISMATCH: a declared training `final_loss` exists but the
      held-out loss is implausibly worse than it -> the scored checkpoint did
      not come from the declared training run.

Thresholds are deliberately generous so an honest model (held-out a bit worse
than training, never near random) is NEVER rejected; only garbage / fabricated
checkpoints trip it. Returns (ok, reason); ok=False means reject as fraud/broken.
"""
from __future__ import annotations

import math
import re

# Reject if held-out loss >= this fraction of the random baseline ln(vocab).
# A real ~254M model sits at ~3-4.5 nats/token; random is ~10.8 for vocab 50257.
# 0.80 -> reject above ~8.6 nats, leaving a wide safety margin under legit models.
DEFAULT_RANDOM_FRACTION = 0.80

# Reject if held-out loss > claimed_final_loss * FACTOR + MARGIN. Generous: a
# normal train->held-out gap is well under 1.5x; this only fires on gross
# mismatch (e.g. claimed 3.0, measured 9.0).
DEFAULT_MISMATCH_FACTOR = 2.5
DEFAULT_MISMATCH_MARGIN = 1.0


def nats_per_token_from_bpb(val_bpb: float, bytes_per_token: float) -> float:
    """Invert val_bpb = nats / (ln2 * bytes_per_token)."""
    return float(val_bpb) * math.log(2) * float(bytes_per_token)


def check_checkpoint_trained(
    measured_nats_per_token: float,
    vocab_size: int,
    *,
    claimed_final_loss: float | None = None,
    random_fraction: float = DEFAULT_RANDOM_FRACTION,
    mismatch_factor: float = DEFAULT_MISMATCH_FACTOR,
    mismatch_margin: float = DEFAULT_MISMATCH_MARGIN,
) -> tuple[bool, str]:
    """Cheap guard against untrained / log-mismatched checkpoints.

    Args:
      measured_nats_per_token: held-out cross-entropy (nats/token) the validator
        actually measured for this checkpoint (e.g. from op4's val_bpb via
        `nats_per_token_from_bpb`).
      vocab_size: the checkpoint's vocab — sets the random baseline ln(vocab).
      claimed_final_loss: the miner's declared training `final_loss` (nats/token)
        from final_state.json, if present. Enables the log-mismatch check.

    Returns (ok, reason). ok=False -> reject.
    """
    if not (isinstance(measured_nats_per_token, (int, float)) and math.isfinite(measured_nats_per_token)):
        return False, f"non-finite measured loss: {measured_nats_per_token!r}"
    if not (isinstance(vocab_size, int) and vocab_size > 1):
        return False, f"invalid vocab_size: {vocab_size!r}"

    random_baseline = math.log(vocab_size)  # nats/token of a uniform predictor
    if measured_nats_per_token >= random_fraction * random_baseline:
        return False, (
            f"untrained checkpoint: held-out {measured_nats_per_token:.2f} nats/token "
            f">= {random_fraction:.0%} of random baseline {random_baseline:.2f} "
            f"(vocab {vocab_size}) — weights appear at initialization"
        )

    if claimed_final_loss is not None and isinstance(claimed_final_loss, (int, float)) and claimed_final_loss > 0:
        bound = claimed_final_loss * mismatch_factor + mismatch_margin
        if measured_nats_per_token > bound:
            return False, (
                f"log/checkpoint mismatch: held-out {measured_nats_per_token:.2f} nats/token "
                f">> declared training final_loss {claimed_final_loss:.2f} "
                f"(plausible bound {bound:.2f}) — scored checkpoint not from the declared run"
            )

    return True, "ok"


# --- Compute-plausibility (anti compute-gaming) -------------------------------
#
# `wall_clock_s` is MINER-DECLARED (and not in bundle_hash), so a miner can
# under-claim it to look efficient and win the compute-weighted crown — train a
# real model over ~30 H100h but report ~2h. The give-away is physics: the implied
# model-FLOP rate (~6*N*tok/s) cannot exceed the GPU's bf16 matmul peak, and real
# sustained TRAINING MFU is ~30-55%. An implied MFU above the ceiling means the
# wall_clock_s (hence the compute cost) is fabricated.
MAX_PLAUSIBLE_MFU = 0.7
# bf16 dense matmul peak (TFLOP/s) per GPU family — the hard physical ceiling.
_GPU_BF16_PEAK_TFLOPS = {
    "a100": 312.0, "a800": 312.0, "l4": 121.0, "l40": 362.0, "4090": 165.0,
    "h100": 989.0, "h200": 989.0, "h800": 989.0,
    "b100": 1800.0, "b200": 2250.0, "gb200": 2500.0,
}
# Unknown GPU -> assume the fastest known part, so we NEVER false-reject; the gate
# only fires when even the fastest plausible GPU cannot explain the throughput.
_DEFAULT_PEAK_TFLOPS = 2500.0


def _gpu_bf16_peak_flops(gpu_name: str | None) -> float:
    g = (gpu_name or "").lower()
    for key, tflops in _GPU_BF16_PEAK_TFLOPS.items():
        if key in g:
            return tflops * 1e12
    return _DEFAULT_PEAK_TFLOPS * 1e12


def check_compute_plausibility(
    final_state: dict,
    calibration: dict | None = None,
    *,
    max_mfu: float = MAX_PLAUSIBLE_MFU,
) -> tuple[bool, str]:
    """Reject a bundle whose declared training throughput is physically impossible.

    tokens_seen / wall_clock_s implies ~6*N FLOPs/token; over the declared GPU's
    bf16 peak that is the achieved MFU. An implied MFU > `max_mfu` means the
    wall_clock_s (and the efficiency-gate compute cost it drives) is fabricated.
    Best-effort: a missing/incomplete training_summary is skipped (deferred to the
    other gates), not rejected. Returns (ok, reason); ok=False -> reject.
    """
    fs = final_state or {}
    try:
        tokens = float(fs.get("tokens_seen", 0) or 0)
        wall = float(fs.get("wall_clock_s", 0) or 0)
        n = float(fs.get("n_params", 0) or 0)
    except (TypeError, ValueError):
        return True, "compute-plausibility: non-numeric training_summary (skipped)"
    if tokens <= 0 or wall <= 0 or n <= 0:
        return True, "compute-plausibility: incomplete training_summary (skipped)"
    gpu = (calibration or {}).get("gpu_name") or fs.get("gpu_name") or fs.get("device") or ""
    flops_per_s = 6.0 * n * tokens / wall  # 6N FLOPs/token (fwd+bwd)
    mfu = flops_per_s / _gpu_bf16_peak_flops(gpu)
    if mfu > max_mfu:
        return False, (
            f"fabricated compute: {tokens / wall:,.0f} tok/s for a {n / 1e6:.0f}M model on "
            f"'{gpu or 'unknown'}' => {mfu * 100:.0f}% MFU (> {max_mfu * 100:.0f}% physical max); "
            f"wall_clock_s={wall:.0f}s for {tokens:,.0f} tokens is not achievable"
        )
    return True, f"compute plausible: {tokens / wall:,.0f} tok/s, {mfu * 100:.0f}% MFU"


# --- Training-timing plausibility (anti off-protocol-training) -----------------
#
# op2 attestation proves the canonical recipe/runner CODE was present in the
# enclave (container_measurement) and that the bundle is BOUND to it (report_data),
# but NOT that the code EXECUTED to produce the checkpoint. A miner with real CC
# hardware can train a model OFF-PROTOCOL (own box, any data/compute, no data-lock,
# no step/compute gate), then spin up the canonical container and mint an
# attestation over the pre-trained checkpoint + a fabricated final_state.
#
# The physical tell: a checkpoint that attests to the canonical code cannot have
# been trained for longer than that code has EXISTED. If the declared wall_clock_s
# exceeds the wall-clock time elapsed since the canonical code was committed, the
# run necessarily started before this code existed -> it was produced off-protocol
# and the enclave only attested a pre-trained model. Pairs with
# check_compute_plausibility: too-LONG wall_clock trips THIS gate; too-SHORT trips
# the MFU gate. Together they box in the off-protocol class (a real model needs
# real FLOPs => a minimum wall_clock the window cannot contain).
def check_training_timing(
    final_state: dict,
    *,
    canonical_code_epoch: float | None,
    now_epoch: float,
    slack_s: float = 7200.0,
) -> tuple[bool, str]:
    """Reject a checkpoint whose declared training duration exceeds the lifetime of
    the canonical code it attests to. Best-effort: skipped (ok) when the canonical
    code epoch is unknown or no wall_clock_s is declared. Returns (ok, reason);
    ok=False -> reject as off-protocol."""
    fs = final_state or {}
    if canonical_code_epoch is None:
        return True, "timing: unknown canonical code epoch (skipped)"
    try:
        wall = float(fs.get("wall_clock_s", 0) or 0)
    except (TypeError, ValueError):
        return True, "timing: non-numeric wall_clock_s (skipped)"
    if wall <= 0:
        return True, "timing: no declared wall_clock_s (skipped)"
    code_age = float(now_epoch) - float(canonical_code_epoch)
    if wall > code_age + slack_s:
        return False, (
            f"off-protocol training: declared wall_clock_s={wall:,.0f}s ({wall / 3600:.1f}h) "
            f"exceeds canonical code age {code_age:,.0f}s ({code_age / 3600:.1f}h, "
            f"+{slack_s / 3600:.1f}h slack) — the attested canonical recipe is younger than "
            f"the claimed run, so the checkpoint was trained before this code existed"
        )
    return True, f"timing plausible: wall {wall / 3600:.1f}h <= code age {code_age / 3600:.1f}h"


def check_checkpoint_not_blocklisted(checkpoint_sha256: str | None, blocked: set) -> tuple[bool, str]:
    """Reject a checkpoint whose SHA-256 was previously dethroned as fraud/off-protocol.

    Stopgap against re-submitting the IDENTICAL off-protocol model under a fresh
    bundle hash + adjusted final_state metadata. The timing gate weakens as the
    canonical code ages (a 7h claim becomes "possible" 7h after the cutover), so an
    unchanged fraud checkpoint can otherwise be re-crowned by simply waiting. The
    caller passes manifest['checkpoint_sha256'], which is authenticated against the
    on-disk checkpoint by the artifact-integrity loop. Returns (ok, reason)."""
    if isinstance(checkpoint_sha256, str) and checkpoint_sha256 in blocked:
        return False, (
            f"blocklisted checkpoint {checkpoint_sha256[:16]}… — this exact model was "
            f"previously dethroned as off-protocol/fabricated; re-derive on the canonical "
            f"code to resubmit"
        )
    return True, "checkpoint not blocklisted"


# --- Pre-crown re-derivation (proof of EXECUTION, not just presence) -----------
#
# op2 attests that the canonical code was PRESENT; nothing proves it EXECUTED to
# produce the checkpoint. The timing gate + fraud blocklist raise the cost but a
# patient attacker retrains a fresh off-protocol checkpoint and waits out the
# timing window. The only check that proves the declared run actually happened is
# to RE-RUN a slice of it: apply the miner's patch to the canonical recipe, run the
# real train.py for the first N steps on CANONICAL data with the miner's config +
# seed, and compare the re-derived per-step loss trajectory against the declared
# training_log.jsonl.
#
# Why it works: the step-0 loss (init-seed weights forward on the first canonical
# batch) is a near-deterministic FINGERPRINT of (arch, seed, data) — an off-protocol
# run on different data/arch, or a fabricated log, misses it. The next few logged
# points can't be reproduced without actually running the canonical optimizer on
# canonical data, so faking them == honestly training (the attacker gains nothing).
# Coverage limit: partial re-derivation proves the run STARTED honestly; a
# "run N canonical steps then swap the final checkpoint" attack needs the attacker
# to actually run N canonical steps AND the swapped checkpoint still faces op4 — it
# raises cost sharply but only full re-derivation closes it completely.
def compare_loss_trajectory(
    declared,
    rederived,
    *,
    step0_tol: float = 0.10,
    abs_tol: float = 0.40,
    rel_tol: float = 0.10,
    min_points: int = 2,
) -> tuple[bool, str]:
    """Compare a declared vs a re-derived training-loss trajectory.

    Args:
      declared/rederived: iterables of (step, loss), matched by step number.
      step0_tol: tight band for the step-0 fingerprint (init forward, deterministic).
      abs_tol/rel_tol: looser band for later steps (benign GPU/compile nondeterminism);
        a step passes if |declared - rederived| <= max(abs_tol, rel_tol*|declared|).
      min_points: minimum matched steps required to render a verdict.

    Returns (ok, reason). ok=False => the declared run was not reproduced on canonical
    data (off-protocol / fabricated log)."""
    dd = {int(s): float(v) for s, v in declared if v == v}  # drop NaN
    rr = {int(s): float(v) for s, v in rederived if v == v}
    common = sorted(set(dd) & set(rr))
    if len(common) < min_points:
        return False, (
            f"re-derivation produced too few comparable points ({len(common)} < {min_points}) "
            f"— cannot confirm the declared training ran on canonical code/data"
        )
    # Step-0 fingerprint: init-seed weights forwarded on the first canonical batch.
    # A miss here means different arch/seed/data than the canonical recipe.
    if 0 in common and abs(dd[0] - rr[0]) > step0_tol:
        return False, (
            f"re-derivation mismatch at step 0: declared loss {dd[0]:.3f} vs re-derived "
            f"{rr[0]:.3f} (tol {step0_tol:.2f}) — different init/data/arch than the canonical "
            f"recipe: the checkpoint was trained off-protocol or the log is fabricated"
        )
    fails = []
    for s in common:
        tol = step0_tol if s == 0 else max(abs_tol, rel_tol * abs(dd[s]))
        if abs(dd[s] - rr[s]) > tol:
            fails.append((s, dd[s], rr[s], tol))
    # Tolerate a single noisy point; a majority outside band = systematic divergence.
    if len(fails) > max(0, (len(common) - 1) // 2):
        s, d, r, t = fails[0]
        return False, (
            f"re-derivation trajectory mismatch: {len(fails)}/{len(common)} steps outside band "
            f"(e.g. step {s}: declared {d:.3f} vs re-derived {r:.3f}, tol {t:.2f}) — the declared "
            f"training was not reproduced on canonical data (off-protocol)"
        )
    return True, f"re-derivation reproduced {len(common) - len(fails)}/{len(common)} trajectory points"


def _added_config_jsons(patch_text: str) -> list[dict]:
    """Parse every NEW/whole configs/*.json the patch adds (best-effort)."""
    import json

    out: list[dict] = []
    path: str | None = None
    buf: list[str] = []

    def _flush() -> None:
        if path and path.endswith(".json") and "config" in path and buf:
            try:
                out.append(json.loads("\n".join(buf)))
            except Exception:  # noqa: BLE001 — partial/edited config, skip
                pass

    for ln in (patch_text or "").splitlines():
        if ln.startswith("+++ b/"):
            _flush()
            path, buf = ln[6:], []
        elif ln.startswith("+") and not ln.startswith("+++"):
            buf.append(ln[1:])
    _flush()
    return out


def check_recipe_config_matches_proof(patch_text: str, final_state: dict) -> tuple[bool, str]:
    """A submitted training config (configs/*.json) must match what the proof ran.

    If the patch declares `total_steps` that differs from the steps the proof
    recorded, the crowned checkpoint was NOT produced by the declared recipe (the
    submitted config is a decoy). Best-effort: skipped when no config is added or no
    proof step count exists. Returns (ok, reason); ok=False -> reject.
    """
    fs = final_state or {}
    proof_steps = fs.get("steps")
    if proof_steps is None:
        proof_steps = (fs.get("config") or {}).get("total_steps")
    if proof_steps is None:
        return True, "config-match: no proof step count (skipped)"
    for cfg in _added_config_jsons(patch_text):
        declared = cfg.get("total_steps")
        if declared is None:
            continue
        try:
            if int(declared) != int(proof_steps):
                return False, (
                    f"declared recipe mismatch: submitted config total_steps={declared} but the "
                    f"proof ran {proof_steps} steps — crowned checkpoint not from the submitted recipe"
                )
        except (TypeError, ValueError):
            continue
    return True, "config matches proof"


# Miner-host data paths a canonical run must never reference. A config that
# points manifest_path/data_base_dir at /home, /mnt, … is a data-lock bypass
# (the run trained on the miner's own data, possibly contaminated with the
# held-out, then claimed the canonical recipe). The in-the-wild case:
# manifest_path="/home/root/diony/recipe/data/data_manifest.json".
# ALLOWLIST, not blocklist: canonical data is either the container-RELATIVE
# path (e.g. "data/data_manifest.json") OR the one absolute value
# proof.runner itself pins (RECIPE_DIR/data, realpath-resolved — see
# _canonical_absolute_data_paths; the runner forces this via CLI args
# regardless of what the miner's config requests, so a legitimate
# final_state.config always carries exactly that absolute string). ANY OTHER
# absolute path ("/..."), home ("~"), or parent-escape ("..") points outside
# that tree to a miner-controlled location = a data-lock bypass, regardless
# of the specific prefix. The old per-prefix blocklist (/home|/mnt|...)
# missed /dstack; this matches every other absolute/escaping path by
# construction, so there is no prefix left to enumerate around.
_NONCANONICAL_PATH_RE = re.compile(r"^\s*(?:~|\.\.|/)")


def _canonical_absolute_data_paths() -> dict[str, str]:
    """The absolute values proof.runner itself pins for manifest_path/data_base_dir.

    run_proof_test forces --manifest and --data-base-dir to RECIPE_DIR/data,
    realpath-resolved, regardless of what the miner's patch/config requests —
    so every legitimate final_state.config carries THESE exact absolute
    strings, never a relative one. These are the only absolute values this
    allowlist may accept; anything else absolute/escaping is a data-lock bypass.
    """
    import ralph_bootstrap

    recipe_data = (ralph_bootstrap.RECIPE_DIR / "data").resolve()
    manifest = str(recipe_data / "data_manifest.json")
    data_base = str(recipe_data)
    return {
        "manifest_path": manifest,
        "data_base_dir": data_base,
        "data_dir": data_base,
        "data_path": data_base,
    }


def _canonical_manifest_hash() -> str | None:
    """SHA-256 `manifest_hash` of the validator's OWN canonical data manifest
    (RECIPE_DIR/data/data_manifest.json).

    This is a content hash of the locked corpus's manifest, so it is
    machine-INDEPENDENT: every honest run of the canonical corpus records the
    exact same hash in final_state, regardless of the absolute path the run
    happened to load it from (a miner box's /tmp/.../data vs the validator's
    own path). Returns None when the manifest isn't materialized on this
    validator, in which case callers fall back to the path heuristic.
    """
    try:
        import ralph_bootstrap
        from data.manifest import DataManifest

        mpath = ralph_bootstrap.RECIPE_DIR / "data" / "data_manifest.json"
        if not mpath.exists():
            return None
        return DataManifest.from_path(mpath).manifest_hash()
    except Exception:
        return None


def check_canonical_data_source(
    final_state: dict, canonical_manifest_hash: str | None = None
) -> tuple[bool, str]:
    """Reject a bundle whose training used a non-canonical data source.

    Lives against `final_state` (not the patch diff), so the restricted/exploit
    patch scanners miss it — op1 must check it here. Best-effort: skipped when
    there is no config. Returns (ok, reason).

    PRIMARY check — the manifest CONTENT hash. proof.runner pins --manifest to
    the canonical data tree and train.py records its `manifest_hash` in
    final_state; that hash is identical for every honest run of the locked
    corpus on ANY box, regardless of the absolute path string. Comparing hashes
    (not path strings) is what lets an honest run whose recorded
    manifest_path is an absolute miner-box path (e.g.
    /tmp/ralph/recipe/data/data_manifest.json) pass, while genuinely off-corpus
    training (different hash) is rejected — and unlike the path string, the hash
    cannot be laundered by rewriting the recorded path to look relative (the
    reverted #831 data-lock-bypass vector leaves the hash unchanged).

    FALLBACK — when the validator hasn't materialized its canonical manifest
    (hash unavailable) or the bundle predates `manifest_hash`, fall back to the
    older path heuristic: accept the container-relative path and the exact
    runner-pinned absolute path, reject any other absolute/escaping path.
    """
    cfg = (final_state or {}).get("config") or {}
    if not cfg:
        return True, "canonical data source (no config)"

    if canonical_manifest_hash is None:
        canonical_manifest_hash = _canonical_manifest_hash()
    recorded_hash = (final_state or {}).get("manifest_hash")
    if canonical_manifest_hash and isinstance(recorded_hash, str) and recorded_hash:
        if recorded_hash == canonical_manifest_hash:
            return True, f"canonical data source (manifest_hash {recorded_hash[:8]} matches)"
        return False, (
            f"non-canonical data source: final_state.manifest_hash="
            f"{recorded_hash[:8]} != canonical {canonical_manifest_hash[:8]} — "
            f"training used a different data manifest than the locked corpus"
        )

    canonical = _canonical_absolute_data_paths()
    for key in ("manifest_path", "data_base_dir", "data_dir", "data_path"):
        v = cfg.get(key)
        if not (isinstance(v, str) and v.strip()):
            continue
        if v == canonical[key]:
            continue
        if _NONCANONICAL_PATH_RE.match(v):
            return False, (
                f"non-canonical data source: config.{key}={v!r} is not a "
                f"container-relative path — canonical data is the relative data/ "
                f"tree; any absolute/escaping path bypasses the locked data_manifest"
            )
    return True, "canonical data source"
