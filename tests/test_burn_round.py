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


# ---------------------------------------------------------------- owner allocation

ALLOC = "5G8iypbYKr8GL1CfWaoZe6Sk5tHXi8kh4WaacBpGeqN6NoFi"
ALLOC_CK = "5GC9V7pLdpnurDCTLXiYwcTQVYrjQaFUUAfLNc3QfmqNa5NU"
OWNER_CK = "5HeKCDfdTyk5gttniUXWVGvf6xSpbJJG86thJcZnqNZjThoU"


def _allocate_on(dd, root, chain, sig, reason="treasury for the next mechanism"):
    from eval.burn_round import build_allocation_record
    pub = RecordPublisher(LocalSink(root), window=8, state_path=str(Path(dd) / "hwm-alloc.json"))
    n = len(pub.load_index()["rounds"]) + 1
    rec = build_allocation_record(round_no=n, prev_anchor=pub.head_anchor(), to_hotkey=ALLOC,
                                  to_uid=13, to_coldkey=ALLOC_CK, owner_coldkey=OWNER_CK,
                                  reason=reason, parent_key="qwen3-8b", round_nonce="0xalloc",
                                  declared_at=1_800_000_000.0)
    rec.sign(sig)
    rep = publish_and_gate(pub, rec, anchor_fn=chain.publish_record,
                           head_anchor_fn=chain.head_anchor)
    return rec, rep, pub


def test_an_owner_allocation_is_declared_as_one_and_reproduces():
    from eval.burn_round import is_allocation, is_emission_record
    with tempfile.TemporaryDirectory() as dd:
        root, chain, sig, idx = _auditor_fixture(dd)
        rec, rep, pub = _allocate_on(dd, root, chain, sig)
        assert rep.ok and rep.anchor_verified, rep.reasons
        em = emission_of(rec)
        assert is_allocation(rec) and is_emission_record(rec) and not is_burn(rec)
        assert em["held_by"] == "subnet owner" and em["to_coldkey"] == ALLOC_CK
        assert em["owner_coldkey"] == OWNER_CK
        assert "not a competition result and not a burn" in em["statement"]
        assert rec.weights == {ALLOC: 1.0} and not rec.events and not rec.submissions

        a = audit(str(Path(root) / pub.load_index()["rounds"][-1]["name"]))
        assert not a.failed, [(c.name, c.detail) for c in a.failed]
        assert a.exit_code == 0
        assert any(c.name == "weights are the declared allocation, exactly" and c.status == "PASS"
                   for c in a.checks)


def test_a_follower_pays_the_allocation_and_says_what_it_is():
    from eval.auditor import ACCEPT, Auditor, AuditorConfig
    from eval.lineage import replay, replay_contenders
    with tempfile.TemporaryDirectory() as dd:
        root, chain, sig, idx = _auditor_fixture(dd)
        before = _records(root)
        _allocate_on(dd, root, chain, sig)
        after = _records(root)
        assert replay(after) == replay(before) and replay_contenders(after) == replay_contenders(before)

        wrote, out = [], io.StringIO()
        cfg = AuditorConfig(expected_signer=sig.public_id(), require=("L0", "L1"),
                            pool_from_trail=False, set_weights=True,
                            work_dir=str(Path(dd) / "follower"))
        a = Auditor(cfg, sink=LocalSink(root), signer=Ed25519Signer(seed=b"a" * 32),
                    head_anchor_fn=lambda: chain.head_anchor(),
                    set_weights_fn=lambda w: (wrote.append(dict(w)), True)[1],
                    set_burn_fn=lambda: True, out=out)
        v = a.once()[-1]
        assert v.verdict == ACCEPT, (v.verdict, v.failed, v.note)
        assert wrote[-1] == {ALLOC: 1.0}
        assert "OWNER ALLOCATION" in v.note and "uid 13" in v.note and "not a burn" in v.note
        assert "OWNER ALLOCATION" in out.getvalue(), "the operator's log must say what it is paying"


def test_an_allocation_that_hides_whose_it_is_is_refused():
    from eval.burn_round import build_allocation_record
    with tempfile.TemporaryDirectory() as dd:
        root, chain, sig, idx = _auditor_fixture(dd)
        rec, rep, pub = _allocate_on(dd, root, chain, sig)
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

        # the disclosure stripped out: no "held by the subnet owner"
        a = rigged(lambda r: r["manifest"]["emission"].pop("held_by"))
        assert any(c.name == "allocation target declared" for c in a.failed)
        # the payee's coldkey stripped out
        a = rigged(lambda r: r["manifest"]["emission"].pop("to_coldkey"))
        assert any(c.name == "allocation target declared" for c in a.failed)
        # a second payee beside the declared one
        a = rigged(lambda r: r.__setitem__("weights", {ALLOC: 0.5, OWNER: 0.5}))
        assert any(c.name == "weights are the declared allocation, exactly" for c in a.failed)
        # relabelled as a burn: the record alone reproduces (the label is the claim) — the
        # follower test below is where a "burn" paying someone else is caught
        a = rigged(lambda r: r["manifest"]["emission"].__setitem__("mode", "burn"))
        assert not a.failed
        with pytest.raises(ValueError):
            build_allocation_record(round_no=9, prev_anchor="", to_hotkey=ALLOC, to_uid=13,
                                    to_coldkey="", owner_coldkey=OWNER_CK, reason="x",
                                    parent_key="qwen3-8b", round_nonce="0x0")


def test_a_burn_that_pays_someone_else_is_refused_by_the_follower():
    """A burn record naming a hotkey other than the scoring validator's reproduces on its own —
    the label is only a claim. The follower knows the anchor hotkey from outside the trail and
    refuses: that is a payment labelled as a burn, and it is not paid."""
    from eval.auditor import ACCEPT, REJECT, Auditor, AuditorConfig
    with tempfile.TemporaryDirectory() as dd:
        root, chain, sig, idx = _auditor_fixture(dd)
        pub = RecordPublisher(LocalSink(root), window=8, state_path=str(Path(dd) / "hwm-x.json"))
        rec = build_burn_record(round_no=len(pub.load_index()["rounds"]) + 1,
                                prev_anchor=pub.head_anchor(), to_hotkey=ALLOC, to_uid=13,
                                reason="mislabelled", parent_key="qwen3-8b", round_nonce="0x",
                                declared_at=1_800_000_000.0)
        rec.sign(sig)
        assert publish_and_gate(pub, rec, anchor_fn=chain.publish_record,
                                head_anchor_fn=chain.head_anchor).ok

        def follower(work, anchor_hotkey):
            wrote = []
            cfg = AuditorConfig(expected_signer=sig.public_id(), validator_hotkey=anchor_hotkey,
                                require=("L0", "L1"), pool_from_trail=False, set_weights=True,
                                work_dir=str(Path(dd) / work))
            a = Auditor(cfg, sink=LocalSink(root), signer=Ed25519Signer(seed=b"a" * 32),
                        head_anchor_fn=lambda: chain.head_anchor(),
                        set_weights_fn=lambda w: (wrote.append(dict(w)), True)[1],
                        set_burn_fn=lambda: True, out=io.StringIO())
            return a.once()[-1], wrote

        v, wrote = follower("pinned", OWNER)
        assert v.verdict == REJECT
        assert any("labelled as a burn" in str(f) for f in v.failed), v.failed
        assert {ALLOC: 1.0} not in wrote, "paid a mislabelled burn"

        # and a real burn to the anchor hotkey is still followed under the same pin
        with tempfile.TemporaryDirectory() as d2:
            root2, chain2, sig2, _ = _auditor_fixture(d2)
            _burn_on(d2, root2, chain2, sig2)
            wrote2 = []
            cfg = AuditorConfig(expected_signer=sig2.public_id(), validator_hotkey=OWNER,
                                require=("L0", "L1"), pool_from_trail=False, set_weights=True,
                                work_dir=str(Path(d2) / "ok"))
            a = Auditor(cfg, sink=LocalSink(root2), signer=Ed25519Signer(seed=b"a" * 32),
                        head_anchor_fn=lambda: chain2.head_anchor(),
                        set_weights_fn=lambda w: (wrote2.append(dict(w)), True)[1],
                        set_burn_fn=lambda: True, out=io.StringIO())
            assert a.once()[-1].verdict == ACCEPT and wrote2[-1] == {OWNER: 1.0}
