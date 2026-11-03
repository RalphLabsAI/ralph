"""Declare a burn ON THE TRAIL: one signed, anchored record that pays the whole vector to one uid
and scores nothing.

WHY A RECORD AND NOT A FLAG. Validators that pay this subnet follow the published trail: they
verify each record and set the weights it carries, pinned to the on-chain anchor. A burn that
lived only in some operator's command line would have to be re-communicated to every one of them
and trusted on hearsay; a burn that is a record is re-derived by every auditor exactly like a
round, anchored exactly like a round, and ends exactly like a round — the next scored record
carries real weights again and every follower moves with it.

WHAT A BURN RECORD IS NOT. It carries no points, no submissions, no events and no noise floor,
and the audit refuses one that does: a record that both measures and burns is neither. Crowns
are untouched — the lineage replays events, and this record has none — so the kings keep their
titles and simply go unpaid while the burn stands. Contenders keep their streaks for the same
reason.

    python -m eval.burn_round --reason "paused while the next mechanism is built" --dry-run
    python -m eval.burn_round --reason "..."                 # sign, publish, anchor
    python -m eval.burn_round --reason "..." --set-weights   # ...and write the vector now

The target defaults to the signing validator's own hotkey (uid 0 on netuid 40 — the subnet
owner's uid, which is where Bittensor convention sends a burn). `--to <ss58>` overrides it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

BURN_MODE = "burn"


def emission_of(rec) -> dict:
    """The record's declared emission mode, from a RoundRecord or its dict form."""
    man = rec.get("manifest") if isinstance(rec, dict) else getattr(rec, "manifest", None)
    return dict((man or {}).get("emission") or {})


def is_burn(rec) -> bool:
    return emission_of(rec).get("mode") == BURN_MODE


def build_burn_record(*, round_no: int, prev_anchor: str, to_hotkey: str, to_uid: int,
                      reason: str, parent_key: str, round_nonce: str,
                      declared_at: float | None = None):
    """The record itself. Deterministic given its arguments, so a test can rebuild one."""
    from .round_record import RoundRecord
    if not reason.strip():
        raise ValueError("a burn needs a stated reason — it is read by every validator that follows it")
    now = time.time() if declared_at is None else float(declared_at)
    return RoundRecord(
        round=int(round_no), commit_root="", round_nonce=str(round_nonce),
        teacher=parent_key, judge="", base="", pile_id="",
        points=[], submissions=[], events=[], weights={to_hotkey: 1.0},
        prev_anchor=prev_anchor,
        manifest={
            "emission": {"mode": BURN_MODE, "to": to_hotkey, "to_uid": int(to_uid),
                         "reason": reason.strip(),
                         "declared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))},
            "parent": parent_key,
        },
        started_at=now, published_at=now)


def run(reason: str, to: str = "", dry_run: bool = False, set_weights: bool = False,
        out=sys.stdout) -> int:
    from .chain_bittensor import BittensorChainIO
    from .publish import HFSink, LocalSink, RecordPublisher, publish_and_gate
    from .run_orchestrated import Config
    from .signing import Ed25519Signer

    cfg = Config.from_env()
    w = out.write
    chain = BittensorChainIO(netuid=cfg.netuid, network=cfg.network, wallet_name=cfg.wallet,
                             hotkey_name=cfg.hotkey, read_only=dry_run)
    # the same trail and the same never-shrink mark the round runner uses, so a burn record is
    # round N+1 on the trail and nothing else can later publish an N+1 underneath it
    _local = os.environ.get("RALPH_RECORDS_DIR", "").strip()
    hwm = os.path.join(cfg.work_dir, "publish-hwm-" +
                       (_local if _local else cfg.records_repo).replace("/", "_").strip("_") +
                       ".json")
    publisher = RecordPublisher(LocalSink(_local) if _local else HFSink(cfg.records_repo),
                                window=8, state_path=hwm)
    idx = publisher.load_index()
    round_no = len(idx.get("rounds", [])) + 1
    prev = publisher.head_anchor()
    to_hotkey = (to or chain.me).strip()
    to_uid = chain.uid_of(to_hotkey)
    if to_uid is None:
        w(f"  {to_hotkey} is not registered on netuid {cfg.netuid}; a burn to it pays nobody\n")
        return 2
    rec = build_burn_record(round_no=round_no, prev_anchor=prev, to_hotkey=to_hotkey,
                            to_uid=to_uid, reason=reason, parent_key=cfg.parent_key,
                            round_nonce=chain.block_hash(chain.current_block()))
    w(f"  burn record: round {round_no}, 100% -> uid {to_uid} {to_hotkey[:12]}…, "
      f"prev anchor {(prev or '(none)')[:16]}\n  reason: {reason.strip()}\n")
    if dry_run:
        w(f"  --dry-run: digest {rec.sha256()[:16]}…; nothing signed, published or anchored\n")
        return 0
    seed = os.environ.get("RALPH_RECORD_SEED", "")
    if not seed:
        w("  RALPH_RECORD_SEED unset — the record would be unsigned; refusing\n")
        return 2
    rec.sign(Ed25519Signer(seed=bytes.fromhex(seed)[:32]))
    rep = publish_and_gate(publisher, rec, anchor_fn=chain.publish_record,
                           head_anchor_fn=chain.head_anchor, allow_unanchored=False)
    w(f"  published : {rep.published.uri if rep.published else None}\n"
      f"  anchored  : {rep.anchor_verified}   reasons={rep.reasons or '-'}\n")
    if not rep.ok:
        w("  WITHHELD — the burn did not verify on chain, so the previous vector stands\n")
        return 1
    if set_weights:
        ok = chain.set_weights(rec.weights)
        w(f"  weights   : {{uid {to_uid}: 1.0}} -> set={ok}"
          f"{'' if ok else '  (rate-limited or refused — a keeper following the trail will land it)'}\n")
    else:
        w("  weights   : not written here — a keeper following the trail sets them "
          "(eval.auditor --follow --set-weights)\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reason", required=True, help="why emission is burned; published in the record")
    ap.add_argument("--to", default="", help="target hotkey ss58 (default: this validator's own)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set-weights", action="store_true",
                    help="also write the burn vector from this validator now")
    a = ap.parse_args(argv)
    return run(a.reason, to=a.to, dry_run=a.dry_run, set_weights=a.set_weights)


if __name__ == "__main__":
    raise SystemExit(main())
