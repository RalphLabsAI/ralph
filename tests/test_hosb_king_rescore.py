"""HOSB enforce: the incumbent king is re-scored under the IDENTICAL HOSB path
before the crown comparison, so the lower-bound-Z bias cancels (common random
numbers via the epoch seed). Fail-closed if the king can't be re-scored or the
eval windows differ. Off/shadow mode is unchanged (no re-score).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
import validator.service as svc
from chain_layer.interface import KingRecord


def _challenger(val_bpb, bench, L=512):
    return types.SimpleNamespace(
        rejected=None, miner_hotkey="5F_chal", miner_github="", pr_url="", bundle_hash="bh",
        hidden_eval=types.SimpleNamespace(
            val_bpb=val_bpb, benchmark_accuracy=bench, val_seq_len=L,
            sealed_stream_manifest_hash=None, tail_val_bpb=None),
        operations={"op2_attestation": {"tier": "verified"}},
        calibration={"matmul_ms": 10.0},
        training_summary={"wall_clock_s": 1.0},
    )


class _Chain:
    def __init__(self, king):
        self._king = king

    def get_king(self):
        return self._king

    def get_high_water_mark(self):
        return None


def _king(tmp_path, *, with_ckpt=True, val_bpb=1.5, bench=0.9):
    pd = tmp_path / "king_proof"
    (pd / "training").mkdir(parents=True)
    if with_ckpt:
        (pd / "training" / "checkpoint.pt").write_bytes(b"x")
    return KingRecord(
        miner_hotkey="5F_king", bundle_hash="kbh", val_bpb=val_bpb,
        benchmark_accuracy=bench, compute_cost=1.0, crowned_at=0.0, proof_dir=str(pd),
    )


def _common(monkeypatch, mode, challenger):
    monkeypatch.setattr(svc, "_hosb_mode", lambda: mode)
    monkeypatch.setattr(svc, "judge_submission", lambda *a, **k: challenger)
    monkeypatch.setattr(svc, "_verify_pr_if_required", lambda r, b: (True, ""))


def test_enforce_rescores_king_and_crowns_on_hosb_bar(tmp_path, monkeypatch):
    _common(monkeypatch, "enforce", _challenger(1.0, 0.95))
    king_hosb = types.SimpleNamespace(val_bpb=1.5, benchmark_accuracy=0.9, val_seq_len=512)
    calls = {"n": 0}

    def fake_op4(root, pd, chain=None):
        calls["n"] += 1
        return True, "king", king_hosb

    monkeypatch.setattr(svc, "op4_hidden_eval", fake_op4)
    out = svc.score_and_decide(_Chain(_king(tmp_path)), tmp_path / "bundle", noise_floor_margin=0.02)
    assert calls["n"] == 1               # the king WAS re-scored under HOSB
    assert out["accepted"] is True       # challenger 1.0 decisively beats the HOSB king bar 1.5


def test_enforce_fail_closed_when_king_unscoreable(tmp_path, monkeypatch):
    _common(monkeypatch, "enforce", _challenger(1.0, 0.95))
    monkeypatch.setattr(svc, "op4_hidden_eval", lambda *a, **k: (False, "rescore failed", None))
    out = svc.score_and_decide(_Chain(_king(tmp_path)), tmp_path / "bundle", noise_floor_margin=0.02)
    assert out["accepted"] is False      # NOT crowned despite beating the stale recorded bar


def test_enforce_fail_closed_when_king_checkpoint_missing(tmp_path, monkeypatch):
    _common(monkeypatch, "enforce", _challenger(1.0, 0.95))
    monkeypatch.setattr(svc, "op4_hidden_eval", lambda *a, **k: pytest.fail("re-score attempted w/o ckpt"))
    out = svc.score_and_decide(_Chain(_king(tmp_path, with_ckpt=False)), tmp_path / "bundle", 0.02)
    assert out["accepted"] is False


def test_enforce_fail_closed_on_seq_len_mismatch(tmp_path, monkeypatch):
    _common(monkeypatch, "enforce", _challenger(1.0, 0.95, L=512))
    king_hosb = types.SimpleNamespace(val_bpb=1.5, benchmark_accuracy=0.9, val_seq_len=256)
    monkeypatch.setattr(svc, "op4_hidden_eval", lambda *a, **k: (True, "king", king_hosb))
    out = svc.score_and_decide(_Chain(_king(tmp_path)), tmp_path / "bundle", noise_floor_margin=0.02)
    assert out["accepted"] is False      # incomparable eval windows → no crown


def test_off_mode_is_unchanged_no_rescore(tmp_path, monkeypatch):
    _common(monkeypatch, "off", _challenger(1.0, 0.95))
    monkeypatch.setattr(svc, "op4_hidden_eval", lambda *a, **k: pytest.fail("re-score ran in off mode"))
    # off mode uses the stale recorded king.val_bpb=1.5 bar; challenger 1.0 beats it.
    out = svc.score_and_decide(_Chain(_king(tmp_path, val_bpb=1.5)), tmp_path / "bundle", 0.02)
    assert out["accepted"] is True
