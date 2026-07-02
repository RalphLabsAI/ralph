"""Tests for decrypt-failure handling in the HF poller.

A bundle sealed to an outdated validator pubkey can never be decrypted. Before,
poll_hub re-downloaded + re-failed it every epoch forever (never stamped done).
Now:
  - download_one returns "decrypt_failed" (permanent) vs "unavailable" (transient),
  - poll_hub stamps "decrypt_failed" done (stops the churn) but leaves "unavailable"
    for retry,
  - and the miner's PR is closed with the current pubkey so they can re-seal.
"""

import huggingface_hub

from validator import hf_poller as hp
from validator.version import VALIDATOR_VERSION


def _pr(prefix, num):
    return {
        "bundle_id": prefix + "0" * (64 - len(prefix)),
        "pr_num": num,
        "git_ref": f"refs/pr/{num}",
        "created_at": f"2026-06-2{num}T00:00:00+00:00",
    }


def test_decrypt_failed_marked_done_transient_retried(tmp_path, monkeypatch):
    prs = [_pr("ok", 1), _pr("de", 2), _pr("un", 3)]
    monkeypatch.setattr(hp, "list_remote_submissions", lambda repo_id, token=None: prs)

    status_by_prefix = {"ok": "ok", "de": "decrypt_failed", "un": "unavailable"}
    monkeypatch.setattr(
        hp, "download_one",
        lambda bid, repo_id, dest, **kw: status_by_prefix[bid[:2]],
    )

    downloaded = hp.poll_hub(tmp_path)
    assert downloaded == [prs[0]["bundle_id"]]  # only the 'ok' bundle materialised

    done = hp._load_state(tmp_path)["processed"]
    assert done.get(prs[0]["bundle_id"]) == VALIDATOR_VERSION   # ok → done
    assert done.get(prs[1]["bundle_id"]) == VALIDATOR_VERSION   # decrypt_failed → done (no re-churn)
    assert prs[2]["bundle_id"] not in done                      # unavailable → NOT done (retries)


def test_close_pr_posts_current_pubkey_and_closes(tmp_path, monkeypatch):
    from proof import bundle_crypto

    calls = {}

    class _Api:
        def __init__(self, token=None):
            pass

        def change_discussion_status(self, repo_id, discussion_num, new_status,
                                     repo_type=None, comment=None):
            calls.update(repo_id=repo_id, num=discussion_num, status=new_status,
                         repo_type=repo_type, comment=comment)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)

    hp._close_pr_wrong_key("RalphLabsAI/proof-bundles", 42, None, "a" * 64, "MAC check failed")

    assert calls["num"] == 42
    assert calls["status"] == "closed"
    assert calls["repo_type"] == "dataset"
    assert bundle_crypto.DEFAULT_VALIDATOR_PUBKEY in calls["comment"]
    assert "re-seal" in calls["comment"].lower()


def test_close_pr_swallows_api_errors(monkeypatch):
    """A failed close (perms/network) must not raise — dedup still proceeds."""
    class _Api:
        def __init__(self, token=None):
            pass

        def change_discussion_status(self, **kw):
            raise RuntimeError("403 forbidden")

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    hp._close_pr_wrong_key("RalphLabsAI/proof-bundles", 7, None, "b" * 64, "boom")  # no raise


def test_close_pr_noop_without_pr_num(monkeypatch):
    def _boom(token=None):
        raise AssertionError("HfApi should not be constructed when pr_num is None")

    monkeypatch.setattr(huggingface_hub, "HfApi", _boom)
    hp._close_pr_wrong_key("RalphLabsAI/proof-bundles", None, None, "c" * 64, "x")  # no raise
