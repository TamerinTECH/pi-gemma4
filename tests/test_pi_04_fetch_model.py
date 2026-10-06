"""Tests for pi/04_fetch_model.sh (manifest files[] 'pi/04_fetch_model.sh').

Manifest test field: `bash -n`; --dry-run selection matrix through the feasibility fixtures (16 GB -> 26b Q4_0;
8 GB NVMe -> 26b Q4_0 streaming; 4 GB -> refuses); URL construction unit test. The manifest names
tests/test_pi_scripts.py for the URL test, but that module belongs to the pi/lib/common.sh agent, so the
fetch-specific tests live here (its `bash -n` sweep still covers this script too).

Beyond the manifest: the real-Pi disk fallback (16 GB microSD, 15.50 GB free -> 26b-ud-q3_k_m), explicit
catalog lookups, placeholder refusal, and an end-to-end download against a local Range-capable HTTP server
(HF_ENDPOINT override) that drops the first connection half-way: resume with separate `curl -C -` calls,
size + sha256 verification, .part -> final rename, the .sha256 stamp, fetch.json, re-run detection, sha and
upstream-size mismatches, token handling, the fetch lock and --bg. Nothing touches the network or models/.
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import re
import shlex
import shutil
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures"
SCRIPT = "pi/04_fetch_model.sh"
HERMETIC = ["--", "--io-mem", "none", "--calibration", "none"]

CONFIG_VARS = (
    "PIGEMMA_HOME", "MODEL_DIR", "LLAMA_DIR", "LLAMA_COMMIT", "PIGEMMA_OUT", "THREADS", "CTX", "QUANT_OVERRIDE",
    "OS_RESERVE_GIB", "MIN_FREE_DISK_FACTOR", "KEEP_FREE_GB", "NICE_LEVEL", "DENY_HOSTS",
    "PROTECTED_SERVICE_GLOBS", "ALLOW_MLOCK", "TUNE_APPLY", "TUNE_NVME_GEN3", "TUNE_GOVERNOR", "TUNE_SWAPPINESS",
    "TUNE_READAHEAD_KB", "TUNE_MEMLOCK", "BUILD_JOBS", "HF_LIMIT_RATE", "PIN_BUDGET_BYTES",
)


# ------------------------------------------------------------------------------------------- helpers
def _p(path: Path | str) -> str:
    return Path(path).as_posix()


def _env(tmp_path: Path, fake: str | None = None, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("PIGEMMA_") and k not in CONFIG_VARS and k not in (
               "PY", "HF_TOKEN", "HUGGING_FACE_TOKEN", "HF_ENDPOINT")}
    env["PIGEMMA_PYTHON"] = _p(sys.executable)
    env["PIGEMMA_OUT"] = _p(tmp_path / "out")
    env["LC_ALL"] = env.get("LC_ALL") or "C.UTF-8"
    env["PIGEMMA_FETCH_POLL_S"] = "0.2"
    if fake:
        env["PIGEMMA_FAKE_PROC"] = _p(FIX / fake)
    env.update(extra)
    return env


def run(bash: str, args: list[str], env: dict[str, str], *, timeout: float = 240) -> subprocess.CompletedProcess:
    return subprocess.run([bash, SCRIPT, *args], cwd=REPO, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def kv(stdout: str) -> dict[str, str]:
    """FETCH_*=value lines (printf %q quoted) -> dict, via shlex like a shell would."""
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        if line.startswith("FETCH_") and "=" in line:
            k, _, v = line.partition("=")
            words = shlex.split(v) if v else [""]
            out[k] = words[0] if words else ""
    return out


def facts_copy(tmp_path: Path, name: str, **changes) -> Path:
    data = json.loads((FIX / f"facts_{name}.json").read_text(encoding="utf-8"))
    data.update(changes)
    out = tmp_path / f"facts_{name}_mod.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def out_dir_files(tmp_path: Path) -> list[str]:
    d = tmp_path / "out"
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else []


# --------------------------------------------------------------------------------------- basics
def test_bash_syntax(bash: str) -> None:
    r = subprocess.run([bash, "-n", SCRIPT], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def test_lf_line_endings_and_shebang() -> None:
    data = (REPO / SCRIPT).read_bytes()
    assert b"\r" not in data
    assert data.startswith(b"#!/usr/bin/env bash\n")


def test_help_and_usage_errors(bash: str, tmp_path: Path) -> None:
    env = _env(tmp_path)
    r = run(bash, ["--help"], env)
    assert r.returncode == 0 and "--from-desktop" in r.stdout and "Exit codes" in r.stdout
    r = run(bash, ["--bogus"], env)
    assert r.returncode == 1 and "unknown option" in r.stderr
    r = run(bash, ["--draft-type", "eagle", "--dry-run"], env)
    assert r.returncode == 1
    r = run(bash, ["--quant"], env)
    assert r.returncode == 1 and "needs a value" in r.stderr
    r = run(bash, ["--version"], env)
    assert r.returncode == 0 and r.stdout.startswith("pi/04_fetch_model.sh ")


# --------------------------------------------------------------------- dry-run selection matrix
@pytest.mark.parametrize("facts,model_id,verdict", [
    ("pi5-16gb-nvme-gen3", "26b-q4", "FITS_MAPPED"),   # 16 GB -> 26b Q4_0 in RAM (-nr)
    ("pi5-8gb-nvme-gen3", "26b-q4", "STREAM_OK"),      # 8 GB NVMe -> 26b Q4_0 streaming experts
    ("pi5-8gb-usb3", "26b-q4", "STREAM_OK"),
])
def test_dry_run_selection_matrix(bash: str, tmp_path: Path, facts: str, model_id: str, verdict: str) -> None:
    r = run(bash, ["--dry-run", "--facts", _p(FIX / f"facts_{facts}.json"), *HERMETIC], _env(tmp_path))
    assert r.returncode == 0, r.stderr
    v = kv(r.stdout)
    assert v["FETCH_MODEL_ID"] == model_id
    assert v["FETCH_VERDICT"] == verdict
    assert v["FETCH_SELECTION"] == "auto"
    assert v["FETCH_DRY_RUN"] == "1"
    assert v["FETCH_BYTES"] == "14439363584"
    assert v["FETCH_URL"] == ("https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/resolve/main/"
                              "gemma-4-26B_q4_0-it.gguf")
    assert "-nr" in v["FETCH_LLAMA_FLAGS"]
    assert v["FETCH_DEST"].endswith("/pi-gemma4/models")  # facts.model_dir is the default destination
    assert int(v["FETCH_ETA_S"]) == round(14439363584 / 10e6)  # assumed 10 MB/s in a dry run
    assert "assumed" in v["FETCH_ETA_SOURCE"]
    assert "# " in r.stdout and "curl" in r.stdout and "-C -" in r.stdout and "--retry " not in r.stdout
    assert "DRY RUN" in r.stderr
    assert out_dir_files(tmp_path) == []  # a dry run writes nothing


@pytest.mark.parametrize("facts,changes,expected_rc", [
    ("pi5-4gb-microsd", {}, 4),                               # protected context-friend-1 profile
    ("pi5-4gb-microsd", {"override": "PIGEMMA_I_KNOW"}, 3),   # 4 GB microSD: nothing viable
    ("pi5-8gb-nvme-busy", {}, 4),                             # busy host
    ("generic-linux-nodisk", {}, 3),                          # 1.5 GB free: NO_DISK everywhere
])
def test_dry_run_refusals(bash: str, tmp_path: Path, facts: str, changes: dict, expected_rc: int) -> None:
    f = facts_copy(tmp_path, facts, **changes)
    r = run(bash, ["--dry-run", "--facts", _p(f), *HERMETIC], _env(tmp_path))
    assert r.returncode == expected_rc, r.stdout + r.stderr
    assert "FETCH_MODEL_ID" not in r.stdout
    assert out_dir_files(tmp_path) == []


def test_real_pi_profile_falls_back_to_a_quant_that_fits_the_disk(bash: str, tmp_path: Path) -> None:
    # 16 GB Pi, microSD root fs with 15.50 GB free: 26B Q4_0 fits in RAM but would leave < KEEP_FREE_GB.
    f = facts_copy(tmp_path, "pi5-16gb-microsd-real", override="PIGEMMA_I_KNOW", model_dir_on_root_fs=True)
    r = run(bash, ["--dry-run", "--facts", _p(f), *HERMETIC], _env(tmp_path))
    assert r.returncode == 0, r.stderr
    v = kv(r.stdout)
    assert v["FETCH_SELECTION"] == "auto-disk-fallback"
    assert v["FETCH_MODEL_ID"] == "26b-ud-q3_k_m"
    assert v["FETCH_VERDICT"] in ("FITS_REPACKED", "FITS_MAPPED")
    assert int(v["FETCH_NEED_BYTES"]) <= int(v["FETCH_FREE_BYTES"]) == 15501496320
    assert "DISK FALLBACK" in r.stderr
    r = run(bash, ["--dry-run", "--no-disk-fallback", "--facts", _p(f), *HERMETIC], _env(tmp_path))
    assert r.returncode == 3 and "not enough free disk for 26b-q4" in r.stderr
    # 31B: only slow operation (exit 2) -> fetched with a warning, refused with --strict
    r = run(bash, ["--dry-run", "--model", "31b", "--facts", _p(f), *HERMETIC], _env(tmp_path))
    assert r.returncode == 0, r.stderr
    assert kv(r.stdout)["FETCH_FAMILY"] == "31b"
    assert kv(r.stdout)["FETCH_VERDICT"] == "NEVER_GOOD_SPEED"
    r = run(bash, ["--dry-run", "--model", "31b", "--strict", "--facts", _p(f), *HERMETIC], _env(tmp_path))
    assert r.returncode == 2


def test_fresh_facts_on_fake_real_pi(bash: str, tmp_path: Path) -> None:
    """Without --facts the script collects facts (MODEL_DIR = --dest) through pi/lib/common.sh."""
    env = _env(tmp_path, "pi_proc_cf1_16gb")
    r = run(bash, ["--dry-run", "--dest", "/home/pi/pi-gemma4/models", *HERMETIC], env)
    assert r.returncode == 4  # protected host without PIGEMMA_I_KNOW
    env["PIGEMMA_I_KNOW"] = "1"
    r = run(bash, ["--dry-run", "--dest", "/home/pi/pi-gemma4/models", *HERMETIC], env)
    assert r.returncode == 0, r.stderr
    v = kv(r.stdout)
    assert v["FETCH_MODEL_ID"] == "26b-ud-q3_k_m" and v["FETCH_SELECTION"] == "auto-disk-fallback"
    assert out_dir_files(tmp_path) == []
    r = run(bash, ["--dry-run", "--quant", "26b-q4", "--dest", "/home/pi/pi-gemma4/models"], env)
    assert r.returncode == 3 and "keep-free 2.00 GB" in r.stderr  # explicit quants never fall back


# ---------------------------------------------------------------------------- URL construction
def test_url_construction_explicit(bash: str, tmp_path: Path) -> None:
    env = _env(tmp_path)
    dest = _p(tmp_path / "models")
    r = run(bash, ["--dry-run", "--quant", "Q4_0", "--model", "31b", "--draft", "--dest", dest], env)
    assert r.returncode == 0, r.stderr
    v = kv(r.stdout)
    assert v["FETCH_SELECTION"] == "explicit" and v["FETCH_VERDICT"] == ""
    assert v["FETCH_URL"] == ("https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf/resolve/main/"
                              "gemma-4-31B_q4_0-it.gguf")
    # the 31B MTP sidecar lives in a sub-directory of its repo: kept in the URL and under --dest
    assert v["FETCH_DRAFT_URL"] == ("https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF/resolve/main/"
                                    "MTP/mtp-gemma-4-31B-it-Q4_0.gguf")
    assert v["FETCH_DRAFT_PATH"] == dest + "/MTP/mtp-gemma-4-31B-it-Q4_0.gguf"
    assert v["FETCH_DRAFT_SPEC_TYPE"] == "draft-mtp"
    cat = json.loads((REPO / "pigemma/data/catalog.json").read_text(encoding="utf-8"))
    rev = {m["id"]: m["revision"] for m in cat["models"]}
    r = run(bash, ["--dry-run", "--quant", "Q4_0", "--model", "31b", "--draft", "--pin-revision",
                   "--dest", dest], env)
    v = kv(r.stdout)
    assert v["FETCH_URL"].endswith(f"/resolve/{rev['31b-q4']}/gemma-4-31B_q4_0-it.gguf")
    assert v["FETCH_DRAFT_URL"].endswith(f"/resolve/{rev['31b-mtp']}/MTP/mtp-gemma-4-31B-it-Q4_0.gguf")
    env["HF_ENDPOINT"] = "http://mirror.example:8080/"
    r = run(bash, ["--dry-run", "--quant", "ud-q3_k_m", "--draft", "--draft-type", "dflash", "--dest", dest], env)
    v = kv(r.stdout)
    assert v["FETCH_MODEL_ID"] == "26b-ud-q3_k_m"
    assert v["FETCH_URL"] == ("http://mirror.example:8080/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/"
                              "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf")
    assert v["FETCH_DRAFT_ID"] == "26b-dflash"


def test_url_quoting_and_catalog_errors(bash: str, tmp_path: Path) -> None:
    cat = {"models": [
        {"id": "odd-q4", "family": "26b", "kind": "moe", "quant": "Q4_0", "repo": "org/odd repo",
         "file": "sub dir/a+b.gguf", "bytes": 1000, "bytes_exact": True, "sha256": "ab" * 32, "drafts": []},
        {"id": "ph-q2", "family": "26b", "kind": "moe", "quant": "Q2", "repo": "org/r",
         "file": "<fill from HF listing>", "bytes": 9900000000, "bytes_exact": False, "drafts": []},
    ]}
    cp = tmp_path / "cat.json"
    cp.write_text(json.dumps(cat), encoding="utf-8")
    env = _env(tmp_path)
    base = ["--dry-run", "--catalog", _p(cp), "--dest", _p(tmp_path / "m")]
    r = run(bash, [*base, "--quant", "odd-q4"], env)
    assert r.returncode == 0, r.stderr
    assert kv(r.stdout)["FETCH_URL"] == "https://huggingface.co/org/odd%20repo/resolve/main/sub%20dir/a%2Bb.gguf"
    r = run(bash, [*base, "--quant", "Q2"], env)
    assert r.returncode == 3 and "placeholder" in r.stderr and "catalog-refresh" in r.stderr
    r = run(bash, [*base, "--quant", "Q9"], env)
    assert r.returncode == 3 and "available" in r.stderr
    r = run(bash, ["--dry-run", "--quant", "Q4_0", "--dest", _p(tmp_path / "m")], env)
    assert r.returncode == 1 and "several families" in r.stderr
    r = run(bash, [*base, "--quant", "odd-q4", "--draft"], env)
    assert r.returncode == 3 and "no mtp draft" in r.stderr


# ------------------------------------------------------------------------ end-to-end download
class _State:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.linked: dict[str, tuple[int, str]] = {}  # name -> (X-Linked-Size, X-Linked-ETag sha)
        self.drop_first: set[str] = set()
        self.no_etag: set[str] = set()  # names whose redirect carries no X-Linked-ETag
        self.need_token: str | None = None
        self.log: list[tuple[str, str, str, str]] = []  # method, path, range, auth
        self.lock = threading.Lock()


class _Handler(http.server.BaseHTTPRequestHandler):
    state: _State

    def log_message(self, *a) -> None:  # silence
        pass

    def _record(self) -> None:
        with self.state.lock:
            self.state.log.append((self.command, self.path, self.headers.get("Range") or "",
                                   self.headers.get("Authorization") or ""))

    def _resolve(self, head: bool) -> None:
        name = self.path.rsplit("/resolve/", 1)[1].split("/", 1)[1]
        st = self.state
        if st.need_token and self.headers.get("Authorization") != f"Bearer {st.need_token}":
            self.send_response(401)
            self.end_headers()
            return
        if name not in st.files:
            self.send_response(404)
            self.end_headers()
            return
        size, sha = st.linked.get(name, (len(st.files[name]), hashlib.sha256(st.files[name]).hexdigest()))
        self.send_response(302)
        self.send_header("Location", f"/cdn/{name}")
        self.send_header("X-Linked-Size", str(size))
        if name not in st.no_etag:
            self.send_header("X-Linked-ETag", f'"{sha}"')
        self.send_header("X-Repo-Commit", "0123456789abcdef0123456789abcdef01234567")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self) -> None:  # noqa: N802
        self._record()
        if "/resolve/" in self.path:
            return self._resolve(True)
        self.send_response(405)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        self._record()
        if "/resolve/" in self.path:
            return self._resolve(False)
        name = self.path[len("/cdn/"):]
        data = self.state.files.get(name)
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        start, end = 0, len(data) - 1
        rng = self.headers.get("Range")
        if rng:
            m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng.strip())
            assert m, rng
            start = int(m.group(1))
            if m.group(2):
                end = min(end, int(m.group(2)))
            if start >= len(data):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(data)}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:end + 1]
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with self.state.lock:
            drop = name in self.state.drop_first and not rng
            if drop:
                self.state.drop_first.discard(name)
        if drop:
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            return  # HTTP/1.0: the connection closes with half the promised bytes -> curl exit 18
        self.wfile.write(body)


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@pytest.fixture()
def hf_server():
    state = _State()
    handler = type("H", (_Handler,), {"state": state})
    srv = _Server(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", state
    finally:
        srv.shutdown()
        srv.server_close()


def _tiny_catalog(tmp_path: Path, model: bytes, draft: bytes, *, model_bytes: int | None = None,
                  model_sha: str | None = None) -> Path:
    cat = {"models": [
        {"id": "tiny-q4", "family": "26b", "kind": "moe", "quant": "TQ4", "repo": "test-org/tiny-GGUF",
         "file": "tiny-q4.gguf", "bytes": model_bytes if model_bytes is not None else len(model),
         "bytes_exact": True, "sha256": model_sha if model_sha is not None else hashlib.sha256(model).hexdigest(),
         "revision": "0123456789abcdef0123456789abcdef01234567", "drafts": ["tiny-mtp"]},
        {"id": "tiny-mtp", "family": "26b", "kind": "draft", "quant": "Q4_0", "repo": "test-org/tiny-GGUF",
         "file": "MTP/tiny-mtp.gguf", "bytes": len(draft), "bytes_exact": True,
         "sha256": hashlib.sha256(draft).hexdigest(), "draft_of": "tiny-q4", "draft_type": "mtp",
         "spec_type": "draft-mtp", "drafts": []},
    ]}
    p = tmp_path / "tiny_catalog.json"
    p.write_text(json.dumps(cat, indent=1), encoding="utf-8")
    return p


def _payload(n: int, seed: int) -> bytes:
    out = bytearray()
    h = hashlib.sha256(str(seed).encode()).digest()
    while len(out) < n:
        h = hashlib.sha256(h).digest()
        out += h
    return bytes(out[:n])


def _real_env(tmp_path: Path, endpoint: str, **extra: str) -> dict[str, str]:
    # fake 16 GB NVMe Pi root: not protected, df says 455 GB free on /mnt/nvme (not the root fs)
    return _env(tmp_path, "pi_proc_16gb", HF_ENDPOINT=endpoint, **extra)


def test_end_to_end_download_resume_verify_and_rerun(bash: str, tmp_path: Path, hf_server) -> None:
    endpoint, st = hf_server
    model, draft = _payload(3_000_000, 1), _payload(400_000, 2)
    st.files = {"tiny-q4.gguf": model, "MTP/tiny-mtp.gguf": draft}
    st.drop_first = {"tiny-q4.gguf"}
    cat = _tiny_catalog(tmp_path, model, draft)
    dest = tmp_path / "models"
    args = ["--quant", "tiny-q4", "--draft", "--catalog", _p(cat), "--dest", _p(dest), "--retry-delay", "0"]
    env = _real_env(tmp_path, endpoint, PIGEMMA_FETCH_PROBE_MIN="0")
    r = run(bash, args, env)
    assert r.returncode == 0, r.stdout + r.stderr
    final, dfinal = dest / "tiny-q4.gguf", dest / "MTP" / "tiny-mtp.gguf"
    assert final.read_bytes() == model and dfinal.read_bytes() == draft
    assert not (dest / "tiny-q4.gguf.part").exists() and not (dest / "MTP" / "tiny-mtp.gguf.part").exists()
    assert (dest / "tiny-q4.gguf.sha256").read_text() == f"{hashlib.sha256(model).hexdigest()}  tiny-q4.gguf\n"
    # the first GET was cut half-way; the second attempt resumed from the .part size (no curl --retry)
    gets = [(p, rng) for (m, p, rng, _) in st.log if m == "GET" and p == "/cdn/tiny-q4.gguf"]
    assert gets[0][1] == "" or gets[0][1].startswith("bytes=0-")  # the speed probe may come first
    resumed = [rng for _, rng in gets if rng.startswith("bytes=") and not rng.startswith("bytes=0-")]
    assert resumed == [f"bytes={len(model) // 2}-"], gets
    assert any(m == "HEAD" for (m, *_rest) in st.log)
    v = kv(r.stdout)
    assert v["FETCH_STATUS"] == "downloaded" and v["FETCH_SHA256_VERIFIED"] == "true"
    assert v["FETCH_DRAFT_SHA256_VERIFIED"] == "true"
    fj = json.loads((tmp_path / "out" / "fetch.json").read_text(encoding="utf-8"))
    for key in ("model_id", "file", "bytes", "sha256_verified", "seconds"):
        assert key in fj
    assert fj["model_id"] == "tiny-q4" and fj["file"] == "tiny-q4.gguf" and fj["bytes"] == len(model)
    assert fj["sha256_verified"] is True and fj["sha256"] == hashlib.sha256(model).hexdigest()
    assert fj["sha256_source"] == "catalog" and fj["status"] == "downloaded" and fj["attempts"] == 2
    assert fj["downloaded_bytes"] == len(model) and isinstance(fj["seconds"], int)
    assert fj["selection"] == "explicit" and fj["verdict"] is None
    assert fj["draft"]["model_id"] == "tiny-mtp" and fj["draft"]["sha256_verified"] is True
    assert fj["draft"]["path"].endswith("MTP/tiny-mtp.gguf")
    assert fj["probe_mbps"] is not None  # the probe ran (PIGEMMA_FETCH_PROBE_MIN=0)
    assert not (tmp_path / "out" / ".fetch.lock").exists()
    assert [n for n in out_dir_files(tmp_path) if n.startswith(".fetch.")] == []

    # second run: present and verified through the stamp - no HEAD, no GET
    st.log.clear()
    r = run(bash, args, env)
    assert r.returncode == 0, r.stderr
    assert st.log == []
    assert kv(r.stdout)["FETCH_STATUS"] == "present"
    fj = json.loads((tmp_path / "out" / "fetch.json").read_text(encoding="utf-8"))
    assert fj["status"] == "present" and fj["sha256_verified"] is True and fj["downloaded_bytes"] == 0
    # --reverify re-hashes
    r = run(bash, [*args, "--reverify"], env)
    assert r.returncode == 0 and "computing sha256" in r.stderr


def test_sha_mismatch_keeps_part_and_fails(bash: str, tmp_path: Path, hf_server) -> None:
    endpoint, st = hf_server
    model, draft = _payload(500_000, 3), _payload(1000, 4)
    st.files = {"tiny-q4.gguf": model, "MTP/tiny-mtp.gguf": draft}
    wrong = hashlib.sha256(b"something else").hexdigest()
    st.linked = {"tiny-q4.gguf": (len(model), wrong)}  # server agrees with the (wrong) catalog
    cat = _tiny_catalog(tmp_path, model, draft, model_sha=wrong)
    dest = tmp_path / "models"
    r = run(bash, ["--quant", "tiny-q4", "--catalog", _p(cat), "--dest", _p(dest), "--retry-delay", "0"],
            _real_env(tmp_path, endpoint))
    assert r.returncode == 1
    assert "sha256 MISMATCH" in r.stderr
    assert (dest / "tiny-q4.gguf.part").stat().st_size == len(model)
    assert not (dest / "tiny-q4.gguf").exists()
    assert not (tmp_path / "out" / "fetch.json").exists()


def test_upstream_size_change_refused_before_download(bash: str, tmp_path: Path, hf_server) -> None:
    endpoint, st = hf_server
    model, draft = _payload(200_000, 5), _payload(1000, 6)
    st.files = {"tiny-q4.gguf": model, "MTP/tiny-mtp.gguf": draft}
    cat = _tiny_catalog(tmp_path, model, draft, model_bytes=len(model) + 7)
    r = run(bash, ["--quant", "tiny-q4", "--catalog", _p(cat), "--dest", _p(tmp_path / "models")],
            _real_env(tmp_path, endpoint))
    assert r.returncode == 1 and "changed upstream" in r.stderr
    assert not any(m == "GET" for (m, *_rest) in st.log)


def test_existing_wrong_size_file_is_never_touched(bash: str, tmp_path: Path, hf_server) -> None:
    endpoint, st = hf_server
    model, draft = _payload(100_000, 7), _payload(1000, 8)
    st.files = {"tiny-q4.gguf": model, "MTP/tiny-mtp.gguf": draft}
    cat = _tiny_catalog(tmp_path, model, draft)
    dest = tmp_path / "models"
    dest.mkdir()
    (dest / "tiny-q4.gguf").write_bytes(b"old revision")
    r = run(bash, ["--quant", "tiny-q4", "--catalog", _p(cat), "--dest", _p(dest)], _real_env(tmp_path, endpoint))
    assert r.returncode == 1 and "nothing was changed" in r.stderr
    assert (dest / "tiny-q4.gguf").read_bytes() == b"old revision"
    assert st.log == []


def test_token_is_sent_but_never_printed(bash: str, tmp_path: Path, hf_server) -> None:
    endpoint, st = hf_server
    model, draft = _payload(50_000, 9), _payload(1000, 10)
    st.files = {"tiny-q4.gguf": model, "MTP/tiny-mtp.gguf": draft}
    token = "hf_TestToken123secret"
    st.need_token = token
    cat = _tiny_catalog(tmp_path, model, draft)
    tf = tmp_path / "token.env"
    tf.write_text(f"# test\nHUGGING_FACE_TOKEN={token}\n", encoding="utf-8")
    args = ["--quant", "tiny-q4", "--catalog", _p(cat), "--dest", _p(tmp_path / "models")]
    r = run(bash, args, _real_env(tmp_path, endpoint))
    assert r.returncode == 1 and "401" in r.stderr  # no token -> access denied, explained
    r = run(bash, [*args, "--token-file", _p(tf)], _real_env(tmp_path, endpoint))
    assert r.returncode == 0, r.stderr
    assert token not in r.stdout and token not in r.stderr
    assert token not in (tmp_path / "out" / "fetch.json").read_text(encoding="utf-8")
    assert any(a == f"Bearer {token}" for (m, p, _r, a) in st.log if "/resolve/" in p)
    assert [n for n in out_dir_files(tmp_path) if "auth" in n] == []  # header file removed at exit


def test_fetch_lock(bash: str, tmp_path: Path, hf_server) -> None:
    endpoint, st = hf_server
    model, draft = _payload(20_000, 11), _payload(1000, 12)
    st.files = {"tiny-q4.gguf": model, "MTP/tiny-mtp.gguf": draft}
    cat = _tiny_catalog(tmp_path, model, draft)
    args = ["--quant", "tiny-q4", "--catalog", _p(cat), "--dest", _p(tmp_path / "models")]
    lock = tmp_path / "out" / ".fetch.lock"
    holder = subprocess.Popen([bash, "-c", "echo $$; exec sleep 60"], stdout=subprocess.PIPE, text=True)
    try:
        pid = holder.stdout.readline().strip()
        lock.mkdir(parents=True)
        (lock / "pid").write_text(pid + "\n", encoding="utf-8")
        r = run(bash, args, _real_env(tmp_path, endpoint))
        assert r.returncode == 4 and "another pi/04_fetch_model.sh" in r.stderr
    finally:
        holder.kill()
        holder.wait(timeout=30)
    (lock / "pid").write_text("999999\n", encoding="utf-8")  # stale
    r = run(bash, args, _real_env(tmp_path, endpoint))
    assert r.returncode == 0, r.stderr
    assert "stale fetch lock" in r.stderr and not lock.exists()


def test_background_mode(bash: str, tmp_path: Path, hf_server) -> None:
    endpoint, st = hf_server
    model, draft = _payload(30_000, 13), _payload(1000, 14)
    st.files = {"tiny-q4.gguf": model, "MTP/tiny-mtp.gguf": draft}
    cat = _tiny_catalog(tmp_path, model, draft)
    r = run(bash, ["--bg", "--quant", "tiny-q4", "--catalog", _p(cat), "--dest", _p(tmp_path / "models")],
            _real_env(tmp_path, endpoint))
    assert r.returncode == 0, r.stderr
    v = kv(r.stdout)
    assert v["FETCH_BG_PID"].isdigit() and v["FETCH_LOG"].endswith("fetch.log")
    fj = tmp_path / "out" / "fetch.json"
    deadline = time.time() + 120
    while time.time() < deadline and not fj.exists():
        time.sleep(0.5)
    assert fj.exists(), (tmp_path / "out" / "fetch.log").read_text(encoding="utf-8", errors="replace")
    assert json.loads(fj.read_text(encoding="utf-8"))["sha256_verified"] is True
    assert (tmp_path / "models" / "tiny-q4.gguf").read_bytes() == model


def test_not_enough_disk_refuses_real_run(bash: str, tmp_path: Path, hf_server) -> None:
    endpoint, st = hf_server
    model, draft = _payload(10_000, 15), _payload(1000, 16)
    st.files = {"tiny-q4.gguf": model, "MTP/tiny-mtp.gguf": draft}
    cat = _tiny_catalog(tmp_path, model, draft)
    fake = tmp_path / "fake"
    shutil.copytree(FIX / "pi_proc_16gb", fake)
    (fake / "df.txt").write_text("Filesystem 1-blocks Used Available Capacity Mounted on\n"
                                 "/dev/nvme0n1p1 500000 495000 5000 99% /mnt/nvme\n", encoding="utf-8")
    env = _real_env(tmp_path, endpoint)
    env["PIGEMMA_FAKE_PROC"] = _p(fake)
    r = run(bash, ["--quant", "tiny-q4", "--catalog", _p(cat), "--dest", _p(tmp_path / "models")], env)
    assert r.returncode == 3 and "not enough free disk" in r.stderr
    assert st.log == [] and not (tmp_path / "models").exists()


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed (Git Bash has none)")
def test_from_desktop_local_rsync(bash: str, tmp_path: Path) -> None:
    model, draft = _payload(40_000, 17), _payload(1000, 18)
    src = tmp_path / "desk" / "models"
    (src / "test-org__tiny-GGUF" / "MTP").mkdir(parents=True)
    (src / "test-org__tiny-GGUF" / "tiny-q4.gguf").write_bytes(model)
    (src / "test-org__tiny-GGUF" / "MTP" / "tiny-mtp.gguf").write_bytes(draft)
    cat = _tiny_catalog(tmp_path, model, draft)
    r = run(bash, ["--quant", "tiny-q4", "--draft", "--catalog", _p(cat), "--dest", _p(tmp_path / "models"),
                   "--from-desktop", _p(src)], _env(tmp_path, "pi_proc_16gb"))
    assert r.returncode == 0, r.stderr
    fj = json.loads((tmp_path / "out" / "fetch.json").read_text(encoding="utf-8"))
    assert fj["status"] == "copied" and fj["sha256_verified"] is True and fj["source"].startswith("desktop:")
    assert (tmp_path / "models" / "MTP" / "tiny-mtp.gguf").read_bytes() == draft


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed (Git Bash has none)")
def test_from_desktop_flat_layout_and_partial_resume(bash: str, tmp_path: Path) -> None:
    """SRC/<file> (no repo sub-directory) is found after rsync exit 23, and an existing .part is appended to."""
    model, draft = _payload(60_000, 19), _payload(2000, 20)
    src = tmp_path / "desk"
    (src / "MTP").mkdir(parents=True)
    (src / "tiny-q4.gguf").write_bytes(model)
    (src / "MTP" / "tiny-mtp.gguf").write_bytes(draft)
    cat = _tiny_catalog(tmp_path, model, draft)
    dest = tmp_path / "models"
    dest.mkdir()
    (dest / "tiny-q4.gguf.part").write_bytes(model[:25_000])  # an earlier, interrupted copy
    r = run(bash, ["--quant", "tiny-q4", "--draft", "--catalog", _p(cat), "--dest", _p(dest),
                   "--from-desktop", _p(src) + "/", "--retry-delay", "0"], _env(tmp_path, "pi_proc_16gb"))
    assert r.returncode == 0, r.stderr
    assert "trying the next layout" in r.stderr
    assert (dest / "tiny-q4.gguf").read_bytes() == model and not (dest / "tiny-q4.gguf.part").exists()
    assert (dest / "MTP" / "tiny-mtp.gguf").read_bytes() == draft
    fj = json.loads((tmp_path / "out" / "fetch.json").read_text(encoding="utf-8"))
    assert fj["downloaded_bytes"] == len(model) - 25_000 and fj["sha256_verified"] is True
    assert fj["draft"]["status"] == "copied"


def test_from_desktop_dry_run_plan(bash: str, tmp_path: Path) -> None:
    """--from-desktop plans an rsync (no HTTP HEAD/curl, no ETA) and needs neither rsync nor ssh for a dry run."""
    dest = _p(tmp_path / "models")
    src = "user@desktop:REPO_ROOT/models"
    r = run(bash, ["--dry-run", "--quant", "Q4_0", "--model", "26b", "--draft", "--dest", dest,
                   "--from-desktop", src], _env(tmp_path))
    assert r.returncode == 0, r.stderr
    v = kv(r.stdout)
    assert v["FETCH_SOURCE"] == "desktop:" + src and v["FETCH_MODEL_ID"] == "26b-q4"
    assert v["FETCH_ETA_S"] == "" and v["FETCH_ETA_SOURCE"].startswith("n/a")
    plan = [line for line in r.stdout.splitlines() if line.startswith("# ")]
    rs = [line for line in plan if "rsync -aP --partial" in line]
    # the desktop layout models/<org>__<repo>/<file> comes from the catalog local_path
    assert any(src + "/google__gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf" in line for line in rs)
    assert any("mtp-gemma-4-26B-A4B-it-Q4_0.gguf" in line for line in rs)
    assert not any(" -I " in line or "curl " in line for line in plan)
    assert out_dir_files(tmp_path) == []


def test_sha_from_x_linked_etag_or_unverified(bash: str, tmp_path: Path, hf_server) -> None:
    """No catalog sha256: the server's X-Linked-ETag is the reference; without either, recorded unverified."""
    endpoint, st = hf_server
    model, draft = _payload(70_000, 21), _payload(1000, 22)
    st.files = {"tiny-q4.gguf": model, "MTP/tiny-mtp.gguf": draft}
    cat = _tiny_catalog(tmp_path, model, draft, model_sha="")
    args = ["--quant", "tiny-q4", "--catalog", _p(cat), "--retry-delay", "0"]
    r = run(bash, [*args, "--dest", _p(tmp_path / "m1")], _real_env(tmp_path, endpoint))
    assert r.returncode == 0, r.stderr
    assert "from X-Linked-ETag; the catalog has none" in r.stderr
    fj = json.loads((tmp_path / "out" / "fetch.json").read_text(encoding="utf-8"))
    assert fj["sha256_verified"] is True and fj["sha256_source"] == "x-linked-etag"
    assert fj["sha256"] == hashlib.sha256(model).hexdigest()
    st.no_etag = {"tiny-q4.gguf"}
    r = run(bash, [*args, "--dest", _p(tmp_path / "m2")], _real_env(tmp_path, endpoint))
    assert r.returncode == 0, r.stderr
    assert "UNVERIFIED" in r.stderr
    fj = json.loads((tmp_path / "out" / "fetch.json").read_text(encoding="utf-8"))
    assert fj["sha256_verified"] is False and fj["sha256_source"] is None
    assert fj["sha256"] == hashlib.sha256(model).hexdigest()  # recorded for later comparison
    assert (tmp_path / "m2" / "tiny-q4.gguf").read_bytes() == model
    assert not (tmp_path / "m2" / "tiny-q4.gguf.sha256").exists()  # no stamp without a verified hash
