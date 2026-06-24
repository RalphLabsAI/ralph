"""Policy: a submission must carry a valid GitHub recipe PR, else reject
(RALPH_REQUIRE_GH_PR, default on)."""

import validator.service as service


class _R:
    def __init__(self, pr_url):
        self.pr_url = pr_url


def test_empty_pr_url_rejected_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("RALPH_REQUIRE_GH_PR", raising=False)
    ok, detail = service._verify_pr_if_required(_R(""), tmp_path)
    assert ok is False
    assert "recipe PR" in detail


def test_empty_pr_url_allowed_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("RALPH_REQUIRE_GH_PR", "0")
    ok, _ = service._verify_pr_if_required(_R(""), tmp_path)
    assert ok is True


def test_pr_url_present_without_token_is_allowed_unverified(tmp_path, monkeypatch):
    monkeypatch.setenv("RALPH_REQUIRE_GH_PR", "1")
    monkeypatch.delenv("RALPH_BOT_GH_TOKEN", raising=False)
    ok, detail = service._verify_pr_if_required(
        _R("https://github.com/RalphLabsAI/recipe/pull/9"), tmp_path
    )
    assert ok is True
    assert "no bot token" in detail
