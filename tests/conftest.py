"""Shared pytest configuration and fixtures for the pigemma test suite.

Owner: docs/design/manifest.json files[] entry "tests/conftest.py (+ tests/synth.py, tests/test_*.py
per module)". Goal: every unit test runs on Windows in well under 60 s with no model files, no
Docker and no llama.cpp binaries; ``pytest -m integration`` (models / Docker / binaries) is run by
the orchestrator only.

What this file provides
-----------------------
Hooks
  * ``linux``-marked tests are skipped unless ``sys.platform`` is Linux (pyproject registers the
    marker; this hook does the skipping).
  * Markers ``integration`` and ``linux`` are (re-)registered so the suite also works when pytest is
    started with another config file (``-c``).
  * The report header shows platform, numpy / gguf availability and the Git Bash path.

Fixtures (session-scoped unless noted; built in a session tmp dir; the synthetic ones import
numpy / gguf lazily, so stdlib-only test modules such as test_feasibility.py still run on a bare
Pi Python 3.13 with only pytest installed)
  * ``repo_root`` / ``fixtures_dir``: Path of the repo and of tests/fixtures.
  * ``tiny_moe_gguf``: ``synth.TinyGguf`` of the manifest's gemma4 MoE (2 layers, 4 experts, top-2,
    n_embd 64, n_ff 48, expert ff 32, ...). Acts as a path (``Path(x)``, ``os.fspath(x)``,
    ``str(x)``) and unpacks to ``(path, offsets)`` where offsets = {tensor name: absolute offset}
    as laid out by gguf.GGUFWriter. ``.tensors[name]`` has type/shape/n_bytes/offset/kind/layer and
    the source arrays; ``.expert_layers()``, ``.expert_ranges(il, e)``, ``.summary()``.
  * ``tiny_dense_gguf``: same, 2 layers, no experts, ffn gate/up/down Q4_0 [64,256]/[256,64].
  * ``tiny_llama_gguf`` / ``tiny_llama_gguf_f32``: runnable llama-architecture GGUFs (Q4_0 / F32,
    same underlying weights) for integration tests that need llama.cpp to load a model in ms.
  * ``synth_trace_file``: the default synthetic trace (T=2000, L=4, K=8, E=16, alpha=1.2,
    layer_corr=0.0, seed=0) written with pigemma.trace.TraceWriter, with gen.txt + meta.json.
    Returns ``synth.SynthTrace`` (``.path`` is the .moetrace).
  * ``synth_trace_corr_file``: T=2000, L=4, K=8, **E=64**, layer_corr=0.9: a cross-layer predictor
    clearly beats the static top-16 baseline (measured: precision@16 0.895 vs 0.621; with E=16 the
    static top-16 baseline is trivially 1.0, so E=64 is used here).
  * ``make_synth_trace`` (function scope): factory ``make_synth_trace(name="trace.moetrace",
    **synth_trace_kwargs_or_trace=...) -> SynthTrace`` writing into the test's tmp_path.
  * ``synth_acts_dir``: ``synth.SynthActs``, an act-dump directory over ``tiny_dense_gguf`` layers
    0 and 1 (T=32) obeying ffn_out = W_down @ ffn_geglu with the file's dequantised weights.
  * ``fake_facts``: callable ``fake_facts(name, **overrides) -> dict`` loading
    tests/fixtures/facts_<name>.json (fresh deep copy each call). Names: ``synth.FACTS_NAMES``.
  * ``fake_facts_path``: callable ``fake_facts_path(name) -> Path`` of the checked-in file.
  * ``fixture_text`` / ``fixture_json``: callables reading tests/fixtures/<name>.
  * ``bash``: a POSIX bash that can run the repo's .sh files (Git Bash on Windows, never WSL's
    System32 bash; /bin/bash on Linux); skips the test when none exists.
  * ``isolated_lock`` (function scope): points PIGEMMA_LOCK_FILE at a tmp file and clears the
    other lock variables, so code under test that takes paths.bench_lock() never touches
    results/.bench.lock. Returns the lock path.
  * Integration helpers (they skip, never fail, when the resource is missing):
    ``docker`` (docker CLI path, daemon reachable), ``require_model`` (callable: registry id or
    path -> Path, skip if missing / wrong size), ``llama_bin`` (callable: tool -> Path, skip if
    not built).

Import the builders directly with ``import synth`` (tests/ is on sys.path); do not use
``import tests.synth`` (a second module object breaks isinstance checks).
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
FIXTURES_DIR = TESTS_DIR / "fixtures"

# tests/ for ``import synth``; the repo root for ``import pigemma`` even when pytest runs without
# pyproject's pythonpath setting (e.g. ``pytest -c other.ini``).
for _p in (str(TESTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

IS_LINUX = sys.platform.startswith("linux")


# ------------------------------------------------------------------------------------------- hooks
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: needs the real Gemma 4 GGUFs, Docker or native llama.cpp binaries; "
        "deselected by default, run with `pytest -m integration`",
    )
    config.addinivalue_line(
        "markers",
        "linux: needs Linux (/proc, mmap/mlock/mincore semantics, bash); skipped on other platforms "
        "by tests/conftest.py",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if IS_LINUX:
        return
    skip_linux = pytest.mark.skip(reason=f"marked 'linux': needs Linux, running on {sys.platform}")
    for item in items:
        if item.get_closest_marker("linux") is not None:
            item.add_marker(skip_linux)


def _module_version(name: str) -> str:
    if importlib.util.find_spec(name) is None:
        return "missing"
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", "present"))
    except Exception as exc:  # noqa: BLE001 - only for the report header
        return f"import error: {type(exc).__name__}"


def pytest_report_header(config: pytest.Config) -> list[str]:
    try:
        import pigemma  # noqa: F401  (sets up the gguf-py path)

        gguf_state = "present" if importlib.util.find_spec("gguf") else "missing"
    except Exception as exc:  # noqa: BLE001
        gguf_state = f"pigemma import error: {type(exc).__name__}"
    bash = _find_bash()
    return [
        f"pigemma: repo={REPO_ROOT} platform={sys.platform} python={sys.version.split()[0]}",
        f"pigemma: numpy={_module_version('numpy')} gguf={gguf_state} bash={bash or 'missing'}",
    ]


# ------------------------------------------------------------------------------------------- basics
@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


def _need_numpy() -> None:
    pytest.importorskip("numpy", reason="numpy is needed for synthetic traces / GGUFs")


def _need_gguf() -> None:
    _need_numpy()
    try:
        import pigemma  # noqa: F401  (puts third_party/llama.cpp/gguf-py on sys.path when needed)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"pigemma import failed: {exc}")
    pytest.importorskip("gguf", reason="gguf-py (third_party/llama.cpp/gguf-py, needs pyyaml) is not importable")


def _synth() -> Any:
    _need_numpy()
    import synth  # noqa: PLC0415 - lazy on purpose (numpy)

    return synth


# ------------------------------------------------------------------------------------------- GGUFs
@pytest.fixture(scope="session")
def tiny_moe_gguf(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """synth.TinyGguf: manifest tiny gemma4 MoE (also usable as a path; unpacks to (path, offsets))."""
    _need_gguf()
    s = _synth()
    return s.write_tiny_gguf(tmp_path_factory.mktemp("gguf") / "tiny-moe.gguf")  # defaults == TINY_MOE_SPEC


@pytest.fixture(scope="session")
def tiny_dense_gguf(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """synth.TinyGguf: tiny gemma4 dense (2 layers, ffn Q4_0 [64,256]/[256,64], no experts)."""
    _need_gguf()
    s = _synth()
    return s.write_tiny_dense_gguf(tmp_path_factory.mktemp("gguf") / "tiny-dense.gguf")


@pytest.fixture(scope="session")
def tiny_llama_gguf(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """synth.TinyGguf: runnable llama-arch Q4_0 GGUF (byte-level SPM vocab, random weights)."""
    _need_gguf()
    s = _synth()
    return s.write_tiny_llama_gguf(tmp_path_factory.mktemp("gguf") / "tiny-llama-q4_0.gguf", qtype="Q4_0")


@pytest.fixture(scope="session")
def tiny_llama_gguf_f32(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """synth.TinyGguf: the F32 original of ``tiny_llama_gguf`` (same seed, same weights)."""
    _need_gguf()
    s = _synth()
    return s.write_tiny_llama_gguf(tmp_path_factory.mktemp("gguf") / "tiny-llama-f32.gguf", qtype="F32")


# ------------------------------------------------------------------------------------------- traces
@pytest.fixture(scope="session")
def synth_trace_file(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """synth.SynthTrace with the manifest defaults, written (+ gen.txt, meta.json); ``.path`` is set."""
    s = _synth()
    return s.write_synth_trace(tmp_path_factory.mktemp("trace") / "synth.moetrace",
                               write_gen=True, write_meta=True)


@pytest.fixture(scope="session")
def synth_trace_corr_file(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """T=2000, L=4, K=8, E=64, layer_corr=0.9: cross-layer predictor >> static top-16 baseline."""
    s = _synth()
    return s.write_synth_trace(tmp_path_factory.mktemp("trace") / "synth_corr.moetrace", E=64, layer_corr=0.9)


@pytest.fixture()
def make_synth_trace(tmp_path: Path) -> Callable[..., Any]:
    """Factory: ``make_synth_trace(name="trace.moetrace", *, trace=None, header=None, write_gen=False,
    write_meta=False, strict=True, **synth_trace_kwargs) -> SynthTrace`` written under tmp_path."""
    s = _synth()

    def _make(name: str = "trace.moetrace", **kwargs: Any) -> Any:
        return s.write_synth_trace(tmp_path / name, **kwargs)

    return _make


@pytest.fixture(scope="session")
def synth_acts_dir(tmp_path_factory: pytest.TempPathFactory, tiny_dense_gguf: Any) -> Any:
    """synth.SynthActs: acts dir for tiny_dense_gguf layers 0,1 (T=32); ffn_out = W_down @ geglu."""
    s = _synth()
    return s.write_synth_acts_dir(tmp_path_factory.mktemp("acts") / "tiny-dense", tiny_dense_gguf)


# ------------------------------------------------------------------------------------------- facts / files
# Stdlib-only on purpose (tests/synth.py imports numpy); synth.load_facts / synth.facts_path are the
# same functions for modules that already import synth.
def facts_path(name: str) -> Path:
    """tests/fixtures/facts_<name>.json; FileNotFoundError listing the available names if absent."""
    p = FIXTURES_DIR / f"facts_{name}.json"
    if not p.is_file():
        known = sorted(x.name[len("facts_"):-len(".json")] for x in FIXTURES_DIR.glob("facts_*.json"))
        raise FileNotFoundError(f"no facts fixture {p.name}; available: {', '.join(known)}")
    return p


def load_facts(name: str, **overrides: Any) -> dict:
    """Fresh dict from facts_<name>.json; overrides replace top-level keys, a dict value is merged
    into a dict key (``storage={...}``, ``tools={...}``)."""
    import copy
    import json

    facts = json.loads(facts_path(name).read_text(encoding="utf-8"))
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(facts.get(k), dict):
            facts[k] = {**facts[k], **copy.deepcopy(v)}
        else:
            facts[k] = copy.deepcopy(v)
    return facts


@pytest.fixture(scope="session")
def fake_facts() -> Callable[..., dict]:
    """``fake_facts(name, **overrides) -> dict`` from tests/fixtures/facts_<name>.json (stdlib only)."""
    return load_facts


@pytest.fixture(scope="session")
def fake_facts_path() -> Callable[[str], Path]:
    """``fake_facts_path(name) -> Path`` of tests/fixtures/facts_<name>.json (stdlib only)."""
    return facts_path


@pytest.fixture(scope="session")
def fixture_text() -> Callable[[str], str]:
    """``fixture_text(name) -> str`` (UTF-8) of tests/fixtures/<name>."""

    def _read(name: str) -> str:
        return (FIXTURES_DIR / name).read_text(encoding="utf-8")

    return _read


@pytest.fixture(scope="session")
def fixture_json() -> Callable[[str], Any]:
    """``fixture_json(name) -> object`` parsed from tests/fixtures/<name>."""
    import json

    def _read(name: str) -> Any:
        return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))

    return _read


# ------------------------------------------------------------------------------------------- environment
def _find_bash() -> str | None:
    if sys.platform.startswith("win"):
        try:
            from pigemma import paths  # noqa: PLC0415

            gb = paths.git_bash()
        except Exception:  # noqa: BLE001
            gb = None
        if gb is not None and Path(gb).is_file():
            return str(gb)
        for cand in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
            if Path(cand).is_file():
                return cand
        return None  # never fall back to System32\bash.exe (WSL): it cannot run the repo scripts
    return shutil.which("bash")


@pytest.fixture(scope="session")
def bash() -> str:
    """Path of a bash that runs the repo's scripts (Git Bash on Windows); skip if there is none."""
    b = _find_bash()
    if not b:
        pytest.skip("no usable bash (Git Bash on Windows, bash on Linux)")
    return b


@pytest.fixture()
def isolated_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """PIGEMMA_LOCK_FILE -> tmp_path/bench.lock; inherited-lock and threshold variables cleared."""
    lock = tmp_path / "bench.lock"
    monkeypatch.setenv("PIGEMMA_LOCK_FILE", str(lock))
    for var in ("PIGEMMA_LOCK_HELD", "PIGEMMA_LOCK_STALE_S", "PIGEMMA_MIN_FREE_GB"):
        monkeypatch.delenv(var, raising=False)
    return lock


@pytest.fixture(scope="session")
def docker() -> str:
    """docker CLI path when the daemon answers within 20 s; otherwise skip (integration only)."""
    exe = shutil.which("docker")
    if not exe:
        pytest.skip("docker CLI not found")
    try:
        r = subprocess.run([exe, "version", "--format", "{{.Server.Version}}"], capture_output=True,
                           text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"docker not usable: {exc}")
    if r.returncode != 0 or not r.stdout.strip():
        pytest.skip(f"docker daemon not reachable: {(r.stderr or r.stdout).strip()[:200]}")
    return exe


@pytest.fixture(scope="session")
def require_model() -> Callable[[str], Path]:
    """``require_model(id_or_path) -> Path``; skips the test when the GGUF is missing or the wrong size."""
    from pigemma import paths  # noqa: PLC0415

    def _req(id_or_path: str) -> Path:
        try:
            return paths.require_model(id_or_path)
        except FileNotFoundError as exc:
            pytest.skip(str(exc))
        raise AssertionError("unreachable")  # pragma: no cover

    return _req


@pytest.fixture(scope="session")
def llama_bin() -> Callable[[str], Path]:
    """``llama_bin(tool) -> Path`` via pigemma.paths.bin; skips the test when the binary is not built."""
    from pigemma import paths  # noqa: PLC0415

    def _bin(tool: str) -> Path:
        p = paths.bin(tool)
        if not Path(p).is_file():
            pytest.skip(f"{tool} not built ({p})")
        return Path(p)

    return _bin

