"""Tests for the opt-in wandb validator telemetry — focus on the defensive
contract: disabled by default, and never raises regardless of chain shape."""

import validator.telemetry as telemetry


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RALPH_WANDB", raising=False)
    assert telemetry.init(netuid=40, epoch_seconds=120) is False


def test_log_epoch_noop_when_disabled():
    # No run initialised → must be a silent no-op even with a None chain.
    telemetry._enabled = False
    telemetry._run = None
    telemetry.log_epoch(None, 1, {"submissions": 7, "accepted": 0, "rejected": 7})  # no raise


def test_standing_best_effort_on_bad_chain():
    class Bad:
        pass

    assert telemetry._standing(Bad()) == {}


def test_standing_reads_metagraph():
    class MG:
        hotkeys = ["aa", "bb", "cc"]
        S = {1: 296.0}
        Tv = {1: 0.5}
        validator_permit = {1: True}

    class Chain:
        metagraph = MG()

        class wallet:  # noqa: N801
            class hotkey:  # noqa: N801
                ss58_address = "bb"

    out = telemetry._standing(Chain())
    assert out["standing/uid"] == 1
    assert out["standing/stake"] == 296.0
    assert out["standing/vtrust"] == 0.5
    assert out["standing/vpermit"] == 1
