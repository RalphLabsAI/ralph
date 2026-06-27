"""End-to-end (CPU, no container): the sandbox entrypoint emits per-position
NLLs whose HOST-side reduction reproduces the in-process val_bpb exactly. This is
the correctness contract the op4 sandbox wiring depends on."""
from __future__ import annotations

import dataclasses
import json
import sys

import numpy as np
import pytest
import torch

import ralph_bootstrap

# The canonical model package lives under RECIPE_DIR.
RECIPE_DIR = str(ralph_bootstrap.RECIPE_DIR)
if RECIPE_DIR not in sys.path:
    sys.path.insert(0, RECIPE_DIR)

try:
    from model import RalphBase, RalphConfig  # noqa: E402
    _HAVE_MODEL = True
except Exception:  # noqa: BLE001
    _HAVE_MODEL = False

pytestmark = pytest.mark.skipif(not _HAVE_MODEL, reason="canonical model package not importable")


def _tiny_model_and_ckpt(tmp_path):
    torch.manual_seed(0)
    cfg = RalphConfig(
        vocab_size=64, dim=32, n_layers=2, n_heads=2, head_dim=16,
        ffn_mult=2.0, max_seq_len=16,
    )
    model = RalphBase(cfg)
    ckpt = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg)}, ckpt)
    return cfg, model, ckpt


def test_sandbox_eval_reduction_matches_in_process_val_bpb(tmp_path):
    from eval.host_reduce import expected_token_count, reduce_token_nlls
    from eval.val_bpb import compute_val_bpb
    from validator.sandbox_eval import run_sandbox_eval

    cfg, model, ckpt = _tiny_model_and_ckpt(tmp_path)
    rng = np.random.default_rng(7)
    tokens = rng.integers(0, cfg.vocab_size, size=200, dtype=np.uint16)
    eval_path = tmp_path / "active_tokens.bin"
    tokens.tofile(eval_path)

    out_dir = tmp_path / "out"
    # workdir = RECIPE_DIR so the canonical `model` package resolves (no patch).
    nlls = run_sandbox_eval(RECIPE_DIR, ckpt, eval_path, out_dir)

    # Container artifacts exist and are well-formed.
    saved = np.load(out_dir / "nlls.npy")
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["status"] == "ok"
    seq_len = cfg.max_seq_len // 2
    assert saved.shape[0] == expected_token_count(len(tokens), seq_len)
    assert manifest["tokens_emitted"] == saved.shape[0]

    # HOST reduction of the emitted NLLs == the in-process computation.
    ref = compute_val_bpb(model, tokens, seq_len, bytes_per_token=4.0)
    host = reduce_token_nlls(
        nlls, seq_len=seq_len, bytes_per_token=4.0,
        expected_tokens=expected_token_count(len(tokens), seq_len),
        eval_set_hash="x",
    )
    assert host.val_bpb == pytest.approx(ref["val_bpb"], rel=1e-5)
    assert host.tail_val_bpb == pytest.approx(ref["tail_val_bpb"], rel=1e-5)
    assert host.tokens_evaluated == ref["tokens_evaluated"]


def test_prepare_workdir_copies_canon_and_applies_empty_patch(tmp_path):
    from validator.sandbox_eval import prepare_workdir

    canon = tmp_path / "canon"
    (canon / "model").mkdir(parents=True)
    (canon / "model" / "x.py").write_text("# canonical\n")
    patch = tmp_path / "patch.diff"
    patch.write_text("")  # empty patch → no-op

    wd = prepare_workdir(canon, patch, tmp_path / "workdir")
    assert (wd / "model" / "x.py").read_text() == "# canonical\n"
