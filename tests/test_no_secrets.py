"""Fail if a tracked file contains a credential shape or a secret filename.

Stdlib and git only. Scans `git ls-files`, so ignored trees (models, third_party,
build binaries) are outside the check. The patterns are shapes, not real tokens.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 2_000_000
SKIP_SUFFIX = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".gz", ".zip", ".7z",
    ".gguf", ".bin", ".moetrace", ".npy", ".npz", ".logits", ".exe", ".dll",
}
# Real Hugging Face tokens are `hf_` plus a long alphabet. The unit-test
# stand-in in tests/ is shorter than 30 characters and does not match.
PATTERNS = (
    r"hf_[A-Za-z0-9]{30,}",
    r"ghp_[A-Za-z0-9]{20,}",
    r"gho_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"glpat-[A-Za-z0-9_-]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
)
SECRET_NAMES = {".env", "credentials.json", "id_rsa", "id_ed25519"}


def _tracked() -> list[str]:
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [raw.decode("utf-8") for raw in out.split(b"\0") if raw]


def test_no_secret_filenames() -> None:
    bad = []
    for rel in _tracked():
        name = Path(rel).name
        if name in SECRET_NAMES or name.startswith(".env."):
            bad.append(rel)
    assert not bad, "tracked secret filenames:\n" + "\n".join(bad)


def test_no_secret_shapes() -> None:
    import re

    compiled = [re.compile(p) for p in PATTERNS]
    bad: list[str] = []
    for rel in _tracked():
        path = ROOT / rel
        if not path.is_file() or path.suffix.lower() in SKIP_SUFFIX:
            continue
        if path.stat().st_size > MAX_BYTES:
            continue
        data = path.read_bytes()
        if b"\0" in data[:8000]:
            continue
        text = data.decode("utf-8", errors="replace")
        for pat in compiled:
            match = pat.search(text)
            if match:
                bad.append(f"{rel}: {pat.pattern}")
                break
    assert not bad, "credential shapes in tracked files:\n" + "\n".join(bad)


def test_patterns_ignore_the_short_fixture() -> None:
    import re

    assert re.search(PATTERNS[0], "hf_TestToken123secret") is None
    assert re.search(PATTERNS[0], "hf_" + ("a" * 30)) is not None
