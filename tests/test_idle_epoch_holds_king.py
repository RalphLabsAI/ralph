"""Idle-epoch weight behavior: a king-based subnet must KEEP weighting the
sitting king when no new submission beats it (king earns until dethroned).
Burning is correct ONLY when there is no king at all (genesis / post-reset).

Regression for the bug where empty epochs burned to uid 0 even with a sitting
king, starving the reigning king.
"""

import json

from chain_layer.interface import KingRecord
from chain_layer.local import LocalChain
from validator.service import run_epoch


def _last_weights(chain):
    events = [
        json.loads(ln)
        for ln in (chain.chain_dir / "events.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    ws = [e for e in events if e["type"] == "weights_set"]
    return ws[-1] if ws else None


def _run_idle(chain, tmp_path):
    qd = tmp_path / "queue"
    (qd / "pending").mkdir(parents=True, exist_ok=True)
    return run_epoch(
        chain, qd, noise_floor_margin=0.013,
        hf_repo=None, audit_reports_enabled=False,
    )


def test_idle_epoch_holds_sitting_king(tmp_path):
    chain = LocalChain(tmp_path / "chain")
    chain.set_king(KingRecord(
        miner_hotkey="5KINGhotkeyAAA", bundle_hash="bh", val_bpb=1.5,
        benchmark_accuracy=0.2, compute_cost=0.0, crowned_at=0.0,
    ))

    res = _run_idle(chain, tmp_path)

    assert res["submissions"] == 0
    w = _last_weights(chain)
    assert w is not None
    assert w.get("burn") is not True, "must NOT burn while a king sits"
    assert w["weights"] == {"5KINGhotkeyAAA": 1.0}, "sitting king keeps the weight"


def test_idle_epoch_with_no_king_burns(tmp_path):
    chain = LocalChain(tmp_path / "chain")  # no king set

    _run_idle(chain, tmp_path)

    w = _last_weights(chain)
    assert w is not None
    assert w.get("burn") is True, "no king (genesis) → burn is correct"
    assert w["weights"] == {"uid:0": 1.0}
