"""Eval-integrity tests: the validator must not trust a val_bpb produced by a
non-causal forward().

The validator scores the miner's OWN forward() to compute val_bpb, feeding the
whole window in one call. The target for position t is input[t+1] — so a
non-causal forward can read the answer and emit a perfect prediction, collapsing
val_bpb to ~0 and crowning an unbeatable, fraudulent king. `assert_causal`
rejects such models. These tests pin:

  1. an honest causal model passes the probe (no false positive);
  2. a look-ahead forward (peeks at input[t+1]) is rejected;
  3. run_hidden_eval surfaces the rejection end-to-end;
  4. the probe is a no-op on a stream too short to build a base+decoy window;
  5. the eval window EVAL_SEQ_LEN is a fixed validator constant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from eval import EVAL_SEQ_LEN, run_hidden_eval
from eval.val_bpb import NonCausalModelError, assert_causal

VOCAB = 50257  # matches the GPT-2 BPE vocab the eval stream uses


class CausalModel(nn.Module):
    """Genuinely causal: logits[t] depend only on input[t] (a per-position map),
    a strict subset of [0..t]. Stands in for any honest LM for the probe."""

    def __init__(self, vocab: int = VOCAB, dim: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, idx, targets=None):
        return self.head(self.emb(idx)), None


class LookAheadModel(nn.Module):
    """Malicious forward(): one-hots input[t+1] (== target[t]). The dummy
    parameter stands in for the trivial structural patch a miner adds to route
    op4 into the patched-eval path where their forward() runs."""

    def __init__(self, vocab: int = VOCAB) -> None:
        super().__init__()
        self.vocab = vocab
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        logits = torch.full((B, T, self.vocab), -30.0)
        nxt = idx[:, 1:]
        ar = torch.arange(T - 1)
        for b in range(B):
            logits[b, ar, nxt[b]] = 30.0
        return logits + self.p * 0, None


def _stream(n: int = 4096, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, VOCAB, size=n, dtype=np.uint16)


def test_causal_model_passes_probe():
    # No raise: an honest causal model is invariant to future tokens.
    assert_causal(CausalModel(), _stream(), seq_len=64)


def test_lookahead_model_is_rejected():
    with pytest.raises(NonCausalModelError):
        assert_causal(LookAheadModel(), _stream(), seq_len=64)


def test_run_hidden_eval_rejects_lookahead(tmp_path: Path):
    # Empty eval dir -> run_hidden_eval uses its synthetic stream; the probe runs
    # before compute_val_bpb and rejects the cheat end-to-end.
    with pytest.raises(NonCausalModelError):
        run_hidden_eval(LookAheadModel(), tmp_path, seq_len=64)


def test_short_stream_is_noop():
    # Too little data to build base+decoy -> probe skips rather than false-reject.
    assert_causal(LookAheadModel(), _stream(n=50), seq_len=64)


def test_eval_seq_len_is_a_pinned_constant():
    assert isinstance(EVAL_SEQ_LEN, int) and EVAL_SEQ_LEN > 0
