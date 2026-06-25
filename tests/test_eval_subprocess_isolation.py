"""op4 / patched hidden-eval run the checkpoint forward in a SUBPROCESS so a fatal
CUDA fault (illegal memory access / device-side assert) or a hang kills only the
child. `_run_eval_subprocess` converts a non-zero child exit or a timeout into a
clean (False, reason, None) that the caller rejects — instead of the validator
process aborting (a C++ CUDA abort cannot be caught in-process)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
import validator.validator as V


class _Res:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_crash_exit_is_caught_as_rejection(tmp_path, monkeypatch):
    # A CUDA abort surfaces as a non-zero child exit (SIGABRT core dump = 134).
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Res(134, stderr="Aborted (core dumped)"))
    ok, detail, res = V._run_eval_subprocess(tmp_path, tmp_path / "ckpt.pt", tmp_path, "canonical-eval")
    assert ok is False and res is None
    assert "canonical-eval subprocess exit=134" in detail


def test_timeout_is_caught_as_rejection(tmp_path, monkeypatch):
    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="eval_in_workdir.py", timeout=240)
    monkeypatch.setattr(subprocess, "run", _raise)
    ok, detail, res = V._run_eval_subprocess(tmp_path, tmp_path / "ckpt.pt", tmp_path, "canonical-eval")
    assert ok is False and res is None
    assert "timed out" in detail


def test_success_parses_result(tmp_path, monkeypatch):
    out = ("RALPH_EVAL_RESULT val_bpb=2.500000 benchmark_acc=0.600000 "
           "tokens_evaluated=1000 benchmark_examples=50 eval_set_hash=abc123 "
           "val_seq_len=128 sealed_stream_manifest_hash=none tail_val_bpb=none")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Res(0, stdout=out))
    ok, detail, res = V._run_eval_subprocess(tmp_path, tmp_path / "ckpt.pt", tmp_path, "canonical-eval")
    assert ok is True
    assert res is not None and abs(res.val_bpb - 2.5) < 1e-9
    assert res.val_seq_len == 128 and res.sealed_stream_manifest_hash is None
