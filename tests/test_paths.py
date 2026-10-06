"""Unit tests for pigemma.paths (no models, no Docker; run on Windows and Linux).

Covers the manifest test field: registry sizes present; bin() per platform via monkeypatched
sys.platform/env; sha256_prefix equals hashlib of the first bytes; bench_lock refuses while a lock
with the current pid exists and replaces a dead-pid lock; `lock --hold` runs a trivial command and
forwards its exit code. Every lock test uses a private lock file under tmp_path, never the real
results/.bench.lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pigemma import paths

REPO = Path(__file__).resolve().parent.parent

EXPECTED_SIZES = {
    "26b-q4": 14439363584,
    "31b-q4": 17651001568,
    "26b-mtp": 251939520,
    "26b-dflash": 472433824,
    "31b-mtp": 279955968,
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "PIGEMMA_MIN_FREE_GB", "PIGEMMA_LOCK_FILE", "PIGEMMA_LOCK_HELD", "PIGEMMA_LOCK_STALE_S",
        "PIGEMMA_TOOLS_BIN", "PIGEMMA_LLAMA_BIN", "PIGEMMA_IN_CONTAINER", "PIGEMMA_MODELS_DIR",
        "MODEL_DIR", "LLAMA_DIR", "PIGEMMA_EMULATED",
    ):
        monkeypatch.delenv(var, raising=False)


def _cli_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PIGEMMA_")}
    env.update(extra)
    return env


# ------------------------------------------------------------------------------ registry / models
def test_registry_sizes_and_shape() -> None:
    assert set(paths.MODELS) == set(EXPECTED_SIZES)
    for mid, size in EXPECTED_SIZES.items():
        e = paths.MODELS[mid]
        assert e["bytes"] == size
        for key in ("path", "bytes", "kind", "quant", "draft_of", "repo", "file"):
            assert key in e, (mid, key)
        assert e["path"].startswith("models/") and e["path"].endswith(".gguf")
    assert paths.MODELS["26b-q4"]["kind"] == "moe"
    assert paths.MODELS["31b-q4"]["kind"] == "dense"
    assert paths.MODELS["26b-dflash"]["quant"] == "Q8_0"
    assert sorted(paths.drafts_for("26b-q4")) == ["26b-dflash", "26b-mtp"]
    assert paths.drafts_for("31b-q4") == ["31b-mtp"]
    assert paths.MODELS["26b-mtp"]["spec_type"] == "draft-mtp"
    assert paths.MODELS["26b-dflash"]["spec_type"] == "draft-dflash"
    assert paths.LLAMA_COMMIT == "a7b94df" and paths.LLAMA_BUILD == 11398


def test_repo_root_is_package_parent() -> None:
    assert paths.REPO_ROOT == REPO
    assert paths.LOCK_FILE == REPO / "results" / ".bench.lock"


def test_model_path_id_and_path(tmp_path: Path) -> None:
    p = paths.model_path("26b-q4")
    assert p == REPO / paths.MODELS["26b-q4"]["path"]
    f = tmp_path / "x-Q3_K_M.gguf"
    f.write_bytes(b"GGUF")
    assert paths.model_path(str(f)) == f
    assert paths.model_path(f) == f
    assert paths.guess_quant(f) == "Q3_K_M"
    assert paths.guess_quant("gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf") == "UD-IQ2_XXS"
    assert paths.guess_quant("gemma-4-26B_q4_0-it.gguf") == "Q4_0"


def test_model_dir_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    flat = tmp_path / "gemma-4-26B_q4_0-it.gguf"
    flat.write_bytes(b"x")
    monkeypatch.setenv("PIGEMMA_MODELS_DIR", str(tmp_path))
    assert paths.model_path("26b-q4") == flat


def test_require_model_missing_and_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = {
        "path": "models/acme__tiny-GGUF/tiny-Q4_0.gguf", "bytes": 100, "kind": "moe", "quant": "Q4_0",
        "draft_of": None, "family": "tiny", "repo": "acme/tiny-GGUF", "file": "tiny-Q4_0.gguf",
        "spec_type": None,
    }
    monkeypatch.setitem(paths.MODELS, "tiny", entry)
    monkeypatch.setenv("PIGEMMA_MODELS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError) as ei:
        paths.require_model("tiny")
    assert "acme/tiny-GGUF" in str(ei.value)
    d = tmp_path / "acme__tiny-GGUF"
    d.mkdir()
    f = d / "tiny-Q4_0.gguf"
    f.write_bytes(b"\0" * 99)
    with pytest.raises(FileNotFoundError) as ei:
        paths.require_model("tiny")
    assert "expected 100" in str(ei.value) and "acme/tiny-GGUF" in str(ei.value)
    f.write_bytes(b"\0" * 100)
    assert paths.require_model("tiny") == f
    ident = paths.model_identity("tiny")
    assert ident["id"] == "tiny" and ident["bytes"] == 100 and ident["quant"] == "Q4_0"
    assert ident["sha_prefix"] == hashlib.sha256(b"\0" * 100).hexdigest()
    with pytest.raises(FileNotFoundError):
        paths.require_model(str(tmp_path / "nope.gguf"))


# ------------------------------------------------------------------------------ binaries
def test_bin_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert paths.bin("llama-bench") == REPO / "third_party" / "llama.cpp-win" / "llama-bench.exe"
    assert paths.bin("llama-bench.exe") == REPO / "third_party" / "llama.cpp-win" / "llama-bench.exe"
    assert paths.bin("pigemma-probe") == REPO / "tools" / "build-linux-x86" / "bin" / "pigemma-probe"


def test_bin_linux_docker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    marker = tmp_path / ".dockerenv"
    marker.write_text("")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(paths.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(paths, "CONTAINER_MARKERS", (str(marker),))
    assert paths.in_container() is True
    assert paths.bin("llama-bench") == REPO / "third_party" / "llama.cpp" / "build-linux-x86" / "bin" / "llama-bench"
    assert paths.bin("pinner") == REPO / "tools" / "build-linux-x86" / "bin" / "pinner"
    # env override of the container detection
    monkeypatch.setenv("PIGEMMA_IN_CONTAINER", "0")
    assert paths.in_container() is False


def test_bin_linux_aarch64(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    marker = tmp_path / ".dockerenv"
    marker.write_text("")  # the emulated arm64 container has /.dockerenv too: aarch64 must win
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(paths.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(paths, "CONTAINER_MARKERS", (str(marker),))
    monkeypatch.setenv("LLAMA_DIR", str(tmp_path / "llama.cpp"))
    assert paths.bin("llama-bench") == tmp_path / "llama.cpp" / "build" / "bin" / "llama-bench"
    assert paths.bin("membw") == REPO / "tools" / "build" / "bin" / "membw"
    monkeypatch.delenv("LLAMA_DIR")
    assert paths.bin("llama-cli") == Path("~/llama.cpp").expanduser() / "build" / "bin" / "llama-cli"


def test_bin_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PIGEMMA_TOOLS_BIN", str(tmp_path / "tb"))
    monkeypatch.setenv("PIGEMMA_LLAMA_BIN", str(tmp_path / "lb"))
    monkeypatch.setattr(sys, "platform", "linux")
    assert paths.bin("pigemma-probe") == tmp_path / "tb" / "pigemma-probe"
    assert paths.bin("llama-server") == tmp_path / "lb" / "llama-server"
    with pytest.raises(FileNotFoundError):
        paths.bin("llama-server", check=True)


def test_llama_version_check_structure() -> None:
    res = paths.llama_version_check()
    assert res["expected_build"] == 11398 and res["expected_commit"] == "a7b94df"
    assert isinstance(res["sources"], list)
    if (REPO / "third_party" / "llama.cpp-win" / "VERSION").is_file():
        assert res["checked"] is True


# ------------------------------------------------------------------------------ host / files
def test_host_fingerprint_keys() -> None:
    fp = paths.host_fingerprint()
    for key in (
        "hostname", "os", "kernel", "cpu", "n_logical", "n_physical", "ram_total_bytes",
        "ram_avail_bytes", "page_size", "in_container", "cgroup_mem_max", "python",
    ):
        assert key in fp, key
    assert fp["page_size"] >= 4096 and fp["page_size"] % 4096 == 0
    assert fp["ram_total_bytes"] is None or fp["ram_total_bytes"] > 0
    assert isinstance(fp["in_container"], bool)
    json.dumps(fp)  # JSON-serialisable


def test_cgroup_mem_max(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    f = tmp_path / "memory.max"
    monkeypatch.setattr(paths, "CGROUP_V2_MEMORY_MAX", f)
    monkeypatch.setattr(paths, "CGROUP_V1_MEMORY_LIMIT", tmp_path / "absent")
    assert paths.cgroup_mem_max() is None
    f.write_text("max\n")
    assert paths.cgroup_mem_max() is None
    f.write_text("6442450944\n")
    assert paths.cgroup_mem_max() == 6442450944


def test_sha256_prefix(tmp_path: Path) -> None:
    data = os.urandom(300_000)
    f = tmp_path / "blob.bin"
    f.write_bytes(data)
    r = paths.sha256_prefix(f, n_bytes=100_000)
    assert r["sha256"] == hashlib.sha256(data[:100_000]).hexdigest()
    assert r["prefix_bytes"] == 100_000 and r["size"] == 300_000
    r2 = paths.sha256_prefix(f)  # default 16 MiB > file size -> whole file
    assert r2["sha256"] == hashlib.sha256(data).hexdigest() and r2["prefix_bytes"] == 300_000
    with pytest.raises(ValueError):
        paths.sha256_prefix(f, n_bytes=0)


def test_free_disk_and_container_path(tmp_path: Path) -> None:
    assert paths.free_disk_bytes(tmp_path / "does" / "not" / "exist") > 0
    assert paths.to_container(REPO / "models" / "a.gguf") == "/work/models/a.gguf"
    assert paths.to_container(REPO) == "/work"
    with pytest.raises(ValueError):
        paths.to_container(tmp_path)


# ------------------------------------------------------------------------------ bench lock
def _write_lock(path: Path, **fields: object) -> None:
    rec = {"pid": os.getpid(), "host": socket.gethostname(), "started": "2026-10-05T00:00:00Z",
           "cmd": "test", "token": "t0", "in_container": paths.in_container(), "boot_id": paths._boot_id()}
    rec.update(fields)
    path.write_text(json.dumps(rec), encoding="utf-8")


def _dead_pid() -> int:
    """Pid of a child that has exited and been reaped (its handle closed by Popen.__exit__)."""
    with subprocess.Popen([sys.executable, "-c", "pass"]) as proc:
        proc.wait()
        pid = proc.pid
    if paths._pid_alive(pid):
        pytest.skip("pid was reused immediately; cannot build a dead-pid lock")
    return pid


def test_bench_lock_acquire_release(tmp_path: Path) -> None:
    lock = tmp_path / "res" / ".bench.lock"
    with paths.bench_lock(0, lock_path=lock, cmd="unit") as info:
        assert lock.is_file()
        rec = json.loads(lock.read_text(encoding="utf-8"))
        assert rec["pid"] == os.getpid() and rec["cmd"] == "unit" and rec["token"] == info["token"]
        for key in ("pid", "host", "started", "cmd"):
            assert key in rec
        # same process again -> refused (not re-entrant)
        with pytest.raises(RuntimeError, match="another model-loading job is running"):
            with paths.bench_lock(0, lock_path=lock):
                pass
        assert paths.lock_status(lock)["state"] == "held"
    assert not lock.exists()
    assert paths.lock_status(lock)["state"] == "free"


def test_bench_lock_refuses_current_pid_lock(tmp_path: Path) -> None:
    lock = tmp_path / ".bench.lock"
    _write_lock(lock)
    with pytest.raises(paths.LockBusyError, match="another model-loading job is running"):
        with paths.bench_lock(0, lock_path=lock):
            pass
    assert json.loads(lock.read_text(encoding="utf-8"))["token"] == "t0"  # untouched


def test_bench_lock_replaces_dead_pid_lock(tmp_path: Path) -> None:
    lock = tmp_path / ".bench.lock"
    _write_lock(lock, pid=_dead_pid(), token="dead")
    assert paths.lock_status(lock)["state"] == "stale"
    with paths.bench_lock(0, lock_path=lock) as info:
        rec = json.loads(lock.read_text(encoding="utf-8"))
        assert rec["pid"] == os.getpid() and rec["token"] == info["token"] != "dead"
    assert not lock.exists()


def test_bench_lock_foreign_host_heartbeat(tmp_path: Path) -> None:
    lock = tmp_path / ".bench.lock"
    _write_lock(lock, host="some-container-1234", pid=1, token="foreign")
    with pytest.raises(paths.LockBusyError):
        with paths.bench_lock(0, lock_path=lock, stale_s=60):
            pass
    old = time.time() - 600
    os.utime(lock, (old, old))
    assert paths.lock_status(lock, stale_s=60)["state"] == "stale"
    with paths.bench_lock(0, lock_path=lock, stale_s=60):
        assert json.loads(lock.read_text(encoding="utf-8"))["host"] == socket.gethostname()


def test_bench_lock_ram_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tmp_path / ".bench.lock"
    with pytest.raises(paths.InsufficientMemoryError):
        with paths.bench_lock(1e9, lock_path=lock):  # 1e18 bytes: never available
            pass
    assert not lock.exists()
    monkeypatch.setenv("PIGEMMA_MIN_FREE_GB", "0")  # env overrides the argument
    with paths.bench_lock(1e9, lock_path=lock):
        assert lock.exists()


def test_bench_lock_inherited_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tmp_path / ".bench.lock"
    _write_lock(lock, token="parent-token")
    monkeypatch.setenv("PIGEMMA_LOCK_HELD", "parent-token")
    with paths.bench_lock(0, lock_path=lock) as info:
        assert info["inherited"] is True
    assert lock.exists()  # the parent's lock is not removed by the child


def test_release_keeps_foreign_replacement(tmp_path: Path) -> None:
    lock = tmp_path / ".bench.lock"
    with paths.bench_lock(0, lock_path=lock):
        _write_lock(lock, token="someone-else")
    assert json.loads(lock.read_text(encoding="utf-8"))["token"] == "someone-else"


# ------------------------------------------------------------------------------ CLI
def _cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pigemma.paths", *args],
        cwd=REPO, env=env or _cli_env(PIGEMMA_MIN_FREE_GB="0"),
        capture_output=True, text=True, encoding="utf-8", timeout=120,
    )


def test_cli_hold_forwards_exit_code(tmp_path: Path) -> None:
    lock = tmp_path / ".bench.lock"
    r = _cli("lock", "--hold", "--lock-file", str(lock), "--",
             sys.executable, "-c", "import sys; sys.exit(7)")
    assert r.returncode == 7, r.stderr
    assert not lock.exists()
    r = _cli("lock", "--hold", "--lock-file", str(lock), "--", sys.executable, "-c", "pass")
    assert r.returncode == 0, r.stderr


def test_cli_hold_child_sees_lock_and_nests(tmp_path: Path) -> None:
    lock = tmp_path / ".bench.lock"
    inner = (
        "import json, os, subprocess, sys; "
        "rec = json.load(open(os.environ['PIGEMMA_LOCK_FILE'], encoding='utf-8')); "
        "assert rec['token'] == os.environ['PIGEMMA_LOCK_HELD']; "
        "sys.exit(subprocess.call([sys.executable, '-m', 'pigemma.paths', 'lock', '--hold', '--', "
        "sys.executable, '-c', 'import sys; sys.exit(5)']))"
    )
    r = _cli("lock", "--hold", "--lock-file", str(lock), "--", sys.executable, "-c", inner)
    assert r.returncode == 5, r.stderr
    assert not lock.exists()


def test_cli_hold_refused_when_busy(tmp_path: Path) -> None:
    lock = tmp_path / ".bench.lock"
    _write_lock(lock)  # held by this (live) test process
    r = _cli("lock", "--hold", "--lock-file", str(lock), "--", sys.executable, "-c", "pass")
    assert r.returncode == paths.EXIT_REFUSED
    assert "another model-loading job is running" in r.stderr
    r = _cli("lock", "--status", "--lock-file", str(lock))
    assert r.returncode == 1 and json.loads(r.stdout)["state"] == "held"


def test_cli_hold_missing_command(tmp_path: Path) -> None:
    lock = tmp_path / ".bench.lock"
    r = _cli("lock", "--hold", "--lock-file", str(lock), "--", "pigemma-no-such-command-xyz")
    assert r.returncode == 127
    r = _cli("lock", "--hold", "--lock-file", str(lock))
    assert r.returncode == paths.EXIT_USAGE


def test_resolve_bash_prefers_git_bash_over_wsl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    wsl = r"C:\Windows\System32\bash.exe"
    gitbash = Path(r"C:\Program Files\Git\bin\bash.exe")
    monkeypatch.setattr(paths.shutil, "which", lambda name: wsl if name.startswith("bash") else None)
    monkeypatch.setattr(paths, "git_bash", lambda: gitbash)
    assert paths._resolve_exe("bash") == str(gitbash)
    assert paths._resolve_exe("sh") == str(gitbash)  # not on PATH -> Git Bash
    assert paths._resolve_exe(str(Path("/explicit/bash"))) == str(Path("/explicit/bash"))  # explicit path kept


def test_cli_misc(tmp_path: Path) -> None:
    r = _cli("model", "26b-q4")
    assert r.returncode == 0 and r.stdout.strip().endswith("gemma-4-26B_q4_0-it.gguf")
    r = _cli("model", str(tmp_path / "absent.gguf"), "--require")
    assert r.returncode == paths.EXIT_MISSING
    r = _cli("bin", "llama-bench")
    assert r.returncode == 0 and "llama-bench" in r.stdout
    f = tmp_path / "b.bin"
    f.write_bytes(b"abc" * 1000)
    r = _cli("sha", str(f), "--n-bytes", "10")
    assert r.returncode == 0 and json.loads(r.stdout)["sha256"] == hashlib.sha256(b"abcabcabca").hexdigest()
    r = _cli("free-disk", str(tmp_path))
    assert r.returncode == 0 and int(r.stdout.strip()) > 0
    r = _cli("container-path", str(REPO / "data" / "x.txt"))
    assert r.stdout.strip() == "/work/data/x.txt"
    r = _cli("host")
    assert r.returncode == 0 and "hostname" in json.loads(r.stdout)
    r = _cli("--help")
    assert r.returncode == 0 and "lock" in r.stdout
