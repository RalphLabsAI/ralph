"""Optional Weights & Biases telemetry for the validator.

Opt-in via ``RALPH_WANDB=1`` (needs ``WANDB_API_KEY`` in the environment). Logs
one wandb point per epoch: submission counts, the current king's val_bpb, and
this validator's on-chain standing (uid / stake / vTrust / permit / block).

Entirely best-effort — if wandb is missing, unauthenticated, or errors, the
telemetry silently disables itself and NEVER disrupts validation. A single
resumed run (fixed id) is used so restarts continue the same history; steps are
auto-incremented (block is logged as a metric for use as the dashboard x-axis)
so an epoch counter that resets on restart can't break wandb's monotonic step.
"""

from __future__ import annotations

import os

_run = None
_enabled = False


def _truthy(v: str | None) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def init(netuid: int, epoch_seconds: int) -> bool:
    """Start (or resume) the wandb run. Returns True if telemetry is active."""
    global _run, _enabled
    if not _truthy(os.environ.get("RALPH_WANDB", "0")):
        return False
    try:
        import wandb

        os.environ.setdefault("WANDB_SILENT", "true")
        run_id = os.environ.get("RALPH_WANDB_RUN") or f"validator-sn{netuid}"
        _run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "ralph-validator"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=run_id,
            id=run_id,
            resume="allow",
            config={"netuid": netuid, "epoch_seconds": epoch_seconds},
        )
        _enabled = True
        print(f"[telemetry] wandb enabled: {getattr(_run, 'url', '?')}", flush=True)
    except Exception as e:  # noqa: BLE001 — telemetry must never break the validator
        _enabled = False
        print(f"[telemetry] wandb disabled ({type(e).__name__}: {e})", flush=True)
    return _enabled


def _standing(chain) -> dict:
    """This validator's on-chain standing, best-effort (empty dict on any miss)."""
    out: dict = {}
    try:
        mg = getattr(chain, "metagraph", None)
        wallet = getattr(chain, "wallet", None)
        if mg is None or wallet is None:
            return out
        ss58 = wallet.hotkey.ss58_address
        hotkeys = list(getattr(mg, "hotkeys", []))
        if ss58 not in hotkeys:
            return out
        uid = hotkeys.index(ss58)
        out["standing/uid"] = uid
        try:
            out["standing/stake"] = float(mg.S[uid])
        except Exception:
            pass
        for attr in ("Tv", "validator_trust"):
            try:
                out["standing/vtrust"] = float(getattr(mg, attr)[uid])
                break
            except Exception:
                continue
        try:
            out["standing/vpermit"] = int(bool(mg.validator_permit[uid]))
        except Exception:
            pass
    except Exception:
        pass
    return out


def log_epoch(chain, epoch: int, result: dict) -> None:
    """Log one epoch's metrics. No-op unless telemetry is active."""
    if not _enabled or _run is None:
        return
    try:
        import wandb

        m = {
            "epoch": epoch,
            "epoch/submissions": result.get("submissions", 0),
            "epoch/accepted": result.get("accepted", 0),
            "epoch/rejected": result.get("rejected", 0),
            "epoch/meaningful_failures": result.get("meaningful_failures", 0),
        }
        try:
            m["chain/block"] = chain.get_current_block()
        except Exception:
            pass
        try:
            king = chain.get_king()
            if king is not None:
                m["king/val_bpb"] = float(king.val_bpb)
                m["king/benchmark_accuracy"] = float(getattr(king, "benchmark_accuracy", 0.0) or 0.0)
        except Exception:
            pass
        m.update(_standing(chain))
        wandb.log(m)
    except Exception as e:  # noqa: BLE001
        print(f"[telemetry] log_epoch failed (non-fatal): {e}", flush=True)


def finish() -> None:
    global _run
    if _run is not None:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass
        _run = None
