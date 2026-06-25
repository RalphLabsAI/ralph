"""
val_bpb — validation bits-per-byte computation.

bpb is the per-byte negative log-likelihood under the model's predicted
distribution: bpb = (cross_entropy_in_nats * tokens_count) / (log(2) * byte_count).

It is vocabulary-independent (unlike perplexity), so architectural changes
that change tokenization are scored fairly. This is what autoresearch
optimizes by default; Ralph inherits the metric for the LLM
pretraining launch track.

Per-stream bytes_per_token (B2):
  The token-to-byte ratio varies across the sealed pool: English prose
  hits ~4.0 under GPT-2 BPE, but code/math/multilingual streams have
  meaningfully different ratios (Python ~3.2, OpenWebMath ~3.5, FineWeb-2
  non-European ~2.0). `compute_val_bpb` now accepts the ratio as a
  parameter; `compute_val_bpb_on_stream` is a convenience wrapper for
  callers holding a `SealedStreamBatch` that reads the per-stream value
  from the manifest spec. Backward-compat: when `bytes_per_token=None`,
  the 4.0 default fires — preserves the pre-B2 behaviour for
  eval/hidden_eval.py and existing single-stream callers.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .sealed_streams import SealedStreamBatch

# Default token-to-byte ratio when caller passes bytes_per_token=None.
# Matches the pre-B2 hardcoded value; preserved for single-stream callers
# and for Phase 0 smoke tests that don't construct a sealed pool.
DEFAULT_BYTES_PER_TOKEN = 4.0

# Validator-pinned eval window. The hidden eval must NOT use a sequence length
# derived from the miner's checkpoint config (cfg.max_seq_len // 2): a miner can
# enlarge it to score against an easier, longer-context eval than the king used.
# Callers cap with min(EVAL_SEQ_LEN, cfg.max_seq_len) so small-context models
# still load, but no miner can choose a window larger than the validator's.
EVAL_SEQ_LEN = 512


class NonCausalModelError(RuntimeError):
    """A model's logits at position t depend on tokens AFTER t.

    The validator runs the miner's own forward() to compute val_bpb. compute_val_bpb
    feeds the model the whole window in one call, and the target for position t is
    just input[t+1] — so a NON-causal forward can read the answer and emit a perfect
    prediction, collapsing val_bpb to ~0 (an unbeatable, fraudulent king). Honest
    causal LMs are invariant to future tokens; this error rejects ones that aren't.
    """


def assert_causal(
    model: torch.nn.Module,
    eval_tokens: np.ndarray,
    seq_len: int,
    device: torch.device | None = None,
    n_probes: int = 4,
    atol: float = 1e-3,
) -> None:
    """Reject a model whose logits at position t depend on tokens after t.

    Defeats the look-ahead exploit (the validator scores the miner's own
    forward(), and the answer for position t — input[t+1] — sits inside the
    model's input). For several interior split points k, we copy a real eval
    window and overwrite the FUTURE positions [k+1:] with a DIFFERENT real
    held-out slice, then check the logits at positions [:k+1] are unchanged.

    Using a *real, different* future (not uniform-random noise) is deliberate:
    a random future is trivially distinguishable from real text, which would let
    an adaptive cheat behave causally during the probe and look ahead only on the
    real eval. A realistic-but-different future closes that evasion — a causal
    model is still invariant, a look-ahead one is not.

    No-op when the stream is too short to build a base + decoy window (the caller
    deploys the real held-out shard in production). Raises NonCausalModelError on
    failure; the caller treats that as a rejected submission.

    `atol` is generous on purpose: an honest causal model's prefix logits are
    invariant to future tokens by construction (so the difference is ~0), while a
    look-ahead cheat must move logits by a large margin to drive val_bpb toward 0.
    The wide gap means a loose tolerance keeps detection certain yet leaves no room
    for a false positive from GPU/bf16 attention-kernel tiling noise.
    """
    eval_tokens = np.asarray(eval_tokens)
    if seq_len < 2 or len(eval_tokens) < 2 * (seq_len + 1):
        return
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    base = torch.from_numpy(eval_tokens[:seq_len].astype(np.int64))[None].to(device)
    decoy = torch.from_numpy(eval_tokens[seq_len : 2 * seq_len].astype(np.int64))[None].to(device)
    with torch.no_grad():
        base_logits, _ = model(base)
        for i in range(1, n_probes + 1):
            k = (seq_len * i) // (n_probes + 1)  # interior split points
            if k < 1 or k >= seq_len - 1:
                continue
            if torch.equal(base[:, k + 1 :], decoy[:, k + 1 :]):
                continue  # identical tail (degenerate stream) — this k proves nothing
            alt = base.clone()
            alt[:, k + 1 :] = decoy[:, k + 1 :]  # realistic, different future
            alt_logits, _ = model(alt)
            if not torch.allclose(base_logits[:, : k + 1], alt_logits[:, : k + 1], atol=atol):
                raise NonCausalModelError(
                    f"logits at positions <= {k} changed when future tokens were "
                    f"altered — non-causal forward(); val_bpb is untrustworthy, rejected"
                )


def compute_val_bpb(
    model: torch.nn.Module,
    eval_tokens: np.ndarray,
    seq_len: int,
    batch_size: int = 8,
    device: torch.device | None = None,
    *,
    bytes_per_token: float | None = None,
) -> dict:
    """
    Compute val_bpb over a held-out token stream.

    Args:
      bytes_per_token: empirical token-to-byte ratio for this stream. When
        None (default), uses `DEFAULT_BYTES_PER_TOKEN = 4.0` matching the
        pre-B2 behaviour. When set (typically by `compute_val_bpb_on_stream`
        reading from the SealedStreamManifest), uses that value. Must be
        > 0 or ValueError.

    The token-to-byte ratio is recovered from the tokenizer: GPT-2 BPE
    averages roughly 4.0 bytes per token on English text. The sealed
    pool's manifest carries per-stream ratios computed at
    construction time from each stream's decoded byte length.
    """
    if bytes_per_token is None:
        bytes_per_token = DEFAULT_BYTES_PER_TOKEN
    if bytes_per_token <= 0:
        raise ValueError(
            f"bytes_per_token must be > 0; got {bytes_per_token}"
        )
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    # Long-context tail probe (mirrors recipe/train.py's tail_val_bpb): the SAME
    # cross-entropy, normalized by the SAME bytes_per_token, but accumulated only
    # over the tail positions [seq_len//2 :] of each window. Penalizes recipes
    # that shorten effective training context (the 250M-transfer blind spot).
    # Recorded only — not yet consumed by the scorer.
    tail_start = seq_len // 2
    tail_nats = 0.0
    tail_tokens = 0

    # Pack into non-overlapping windows of (seq_len + 1).
    n = len(eval_tokens)
    n_windows = max(1, (n - 1) // seq_len)
    with torch.no_grad():
        batch_inp = []
        batch_tgt = []
        for w in range(n_windows):
            start = w * seq_len
            ids = eval_tokens[start : start + seq_len + 1]
            if len(ids) < seq_len + 1:
                break
            batch_inp.append(torch.from_numpy(ids[:-1].astype(np.int64)))
            batch_tgt.append(torch.from_numpy(ids[1:].astype(np.int64)))
            if len(batch_inp) == batch_size or w == n_windows - 1:
                inp = torch.stack(batch_inp).to(device)
                tgt = torch.stack(batch_tgt).to(device)
                logits, _ = model(inp)
                # cross-entropy in nats, summed (not mean) so we accumulate correctly
                loss_sum = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    tgt.reshape(-1),
                    reduction="sum",
                )
                total_nats += loss_sum.item()
                total_tokens += tgt.numel()
                # Tail slice: positions [tail_start:] along the sequence axis.
                # logits/tgt are (batch, seq_len, vocab) / (batch, seq_len).
                if tail_start < tgt.size(1):
                    tail_logits = logits[:, tail_start:, :]
                    tail_tgt = tgt[:, tail_start:]
                    tail_loss_sum = F.cross_entropy(
                        tail_logits.reshape(-1, tail_logits.size(-1)),
                        tail_tgt.reshape(-1),
                        reduction="sum",
                    )
                    tail_nats += tail_loss_sum.item()
                    tail_tokens += tail_tgt.numel()
                batch_inp.clear()
                batch_tgt.clear()

    total_bytes = total_tokens * bytes_per_token
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float("inf")
    nll_per_token = total_nats / max(total_tokens, 1)
    tail_bytes = tail_tokens * bytes_per_token
    tail_bpb = (
        tail_nats / (math.log(2) * tail_bytes) if tail_bytes > 0 else None
    )
    return {
        "val_bpb": bpb,
        "tail_val_bpb": tail_bpb,
        "nll_per_token": nll_per_token,
        "tokens_evaluated": total_tokens,
        "bytes_per_token": bytes_per_token,
    }


def compute_val_bpb_on_stream(
    model: torch.nn.Module,
    batch: SealedStreamBatch,
    seq_len: int,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> dict:
    """Convenience wrapper: compute val_bpb on a `SealedStreamBatch`.

    Reads `batch.spec.bytes_per_token` and passes it through to
    `compute_val_bpb`. The result dict includes a `stream_id` field so
    the validator's ladder code can route per-stream results into the
    right Pareto cell.

    Behaviour-equivalent to:
        compute_val_bpb(model, np.asarray(batch.tokens), seq_len,
                        batch_size, device,
                        bytes_per_token=batch.spec.bytes_per_token)
    plus the `stream_id` annotation.
    """
    result = compute_val_bpb(
        model,
        np.asarray(batch.tokens),
        seq_len,
        batch_size,
        device,
        bytes_per_token=batch.spec.bytes_per_token,
    )
    result["stream_id"] = batch.spec.id
    return result


def load_eval_tokens(path: Path | str) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")
