"""A burn lives on the trail, so every validator that follows the trail burns with it.

The record is signed and anchored like a round, scores nothing, pays one uid, leaves the kings
and contenders exactly where the lineage had them, and ends when the next scored record arrives.
Both halves run the real code: the trail comes from the auditor fixture (two scored rounds on a
real publisher + anchoring chain), the verdict from eval.rerun / eval.auditor."""
from __future__ import annotations

import io
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest

from eval.burn_round import build_burn_record, emission_of, is_burn
from eval.publish import LocalSink, RecordPublisher, publish_and_gate
from eval.rerun import audit, load_record, record_from_blob
from eval.signing import Ed25519Signer
from tests.test_crown_path import _auditor_fixture

OWNER = "5HijSRHd9wUmk51UE8Kia7vmx6kD2jwqJLn9bY1frQ4aiTUs"


def _records(root):
    idx = json.loads(LocalSink(root).get("index.json").decode())
    return [record_from_blob(LocalSink(root).get(r["name"])) for r in idx["rounds"]]


def _burn_on(dd, root, chain, sig, reason="paused while the next mechanism is built"):
    pub = RecordPublisher(LocalSink(root), window=8, state_path=str(Path(dd) / "hwm-burn.json"))
    n = len(pub.load_index()["rounds"]) + 1
    rec = build_burn_record(round_no=n, prev_anchor=pub.head_anchor(), to_hotkey=OWNER, to_uid=0,
                            reason=reason, parent_key="qwen3-8b", round_nonce="0xburn",
                            declared_at=1_800_000_000.0)
    rec.sign(sig)
    rep = publish_and_gate(pub, rec, anchor_fn=chain.publish_record,
                           head_anchor_fn=chain.head_anchor)
    return rec, rep, pub


def test_a_burn_record_is_published_anchored_and_reproduces():
    with tempfile.TemporaryDirectory() as dd:
        root, chain, sig, idx = _auditor_fixture(dd)
        before, head_before = _records(root), chain.head_anchor()
        rec, rep, pub = _burn_on(dd, root, chain, sig)
        assert rep.ok and rep.anchor_verified, rep.reasons
        assert rec.round == len(before) + 1 and rec.prev_anchor == head_before
        assert chain.head_anchor() != head_before, "anchoring the burn must advance the chain head"
        assert is_burn(rec) and emission_of(rec)["to_uid"] == 0
        assert rec.weights == {OWNER: 1.0} and not rec.events and not rec.submissions
        assert chain.head_anchor() == pub.head_anchor(), "the burn is the new head of the chain"

        # the plain audit: L0 reproduces, L1-L3 are optional skips, exit 0 — not INCOMPLETE
        a = audit(str(Path(root) / pub.load_index()["rounds"][-1]["name"]))
        assert not a.failed, [(c.name, c.detail) for c in a.failed]
        assert a.exit_code == 0, [(c.level, c.name, c.status, c.required) for c in a.checks]
        assert any(c.name == "weights are the declared burn, exactly" and c.status == "PASS"
                   for c in a.checks)


def test_a_burn_record_leaves_the_crowns_and_contenders_where_they_were():
    from eval.lineage import replay, replay_contenders
    with tempfile.TemporaryDirectory() as dd:
        root, chain, sig, idx = _auditor_fixture(dd)
        before = _records(root)
        kings, contenders = replay(before), replay_contenders(before)
        assert kings, "fixture crowned nobody?"
        _burn_on(dd, root, chain, sig)
        after = _records(root)
        assert len(after) == len(before) + 1
        assert replay(after) == kings
        assert replay_contenders(after) == contenders


def test_a_validator_following_the_trail_sets_the_burn_vector():
    from eval.auditor import ACCEPT, Auditor, AuditorConfig
    with tempfile.TemporaryDirectory() as dd:
        root, chain, sig, idx = _auditor_fixture(dd)
        _burn_on(dd, root, chain, sig)
        wrote, burned = [], []
        cfg = AuditorConfig(expected_signer=sig.public_id(), require=("L0", "L1"),
                            pool_from_trail=False, set_weights=True,
                            work_dir=str(Path(dd) / "follower"))
        a = Auditor(cfg, sink=LocalSink(root), signer=Ed25519Signer(seed=b"a" * 32),
                    head_anchor_fn=lambda: chain.head_anchor(),
                    set_weights_fn=lambda w: (wrote.append(dict(w)), True)[1],
                    set_burn_fn=lambda: (burned.append(1), True)[1], out=io.StringIO())
        vs = a.once()
        burn_v = vs[-1]
        assert burn_v.verdict == ACCEPT, (burn_v.verdict, burn_v.failed, burn_v.skipped, burn_v.note)
        assert burn_v.followed_round == burn_v.round
        # the follower PAYS the record's vector through set_weights — not the cold-start burn path
        assert wrote[-1] == {OWNER: 1.0} and not burned
        # and keeps paying it on a quiet pass
        wrote.clear(); a.once()
        assert wrote == [{OWNER: 1.0}]


def test_a_burn_that_measures_or_pays_anyone_else_is_refused():
    with tempfile.TemporaryDirectory() as dd:
        root, chain, sig, idx = _auditor_fixture(dd)
        rec, rep, pub = _burn_on(dd, root, chain, sig)
        name = pub.load_index()["rounds"][-1]["name"]

        def rigged(mutate):
            raw = json.loads(LocalSink(root).get(name).decode())
            mutate(raw)
            r2 = record_from_blob(json.dumps(raw).encode())
            r2.signature = r2.signer = r2.sig_scheme = ""
            r2.sign(sig)
            p = Path(dd) / "rigged.json"
            p.write_text(json.dumps(asdict(r2)))
            return audit(str(p))

        # a payee smuggled in beside the burn
        a = rigged(lambda r: r.__setitem__("weights", {OWNER: 0.5, "5Stranger": 0.5}))
        assert any(c.name == "weights are the declared burn, exactly" for c in a.failed)
        # the burn vector pointed somewhere other than the declared target
        a = rigged(lambda r: r.__setitem__("weights", {"5Stranger": 1.0}))
        assert any(c.name == "weights are the declared burn, exactly" for c in a.failed)
        # a "burn" that also crowns someone
        a = rigged(lambda r: r.__setitem__("events", [{"tier": "t", "action": "crown", "king": "x"}]))
        assert any(c.name == "burn record scores nothing" for c in a.failed)
        # a reason is not optional
        with pytest.raises(ValueError):
            build_burn_record(round_no=9, prev_anchor="", to_hotkey=OWNER, to_uid=0, reason="  ",
                              parent_key="qwen3-8b", round_nonce="0x0")
        # and the scored records before it are not burns
        assert not any(is_burn(r) for r in _records(root)[:-1])
