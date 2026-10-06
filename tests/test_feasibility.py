"""Unit tests for pigemma/feasibility.py (manifest files[] 'pigemma/feasibility.py', experiment E8).

Covers the manifest test field:
  * five facts fixtures: 4GB+microSD -> NOT_VIABLE for all 26b rows, exit 3 (exit 4 with contextfriend-*
    services or the context-friend-1 hostname); 8GB+USB3 -> 26b streaming verdict asserted against hand
    arithmetic; 8GB+NVMe -> STREAM_OK, exit 0, flags contain '-nr'; 16GB+NVMe -> 26b-q4 FITS_MAPPED with
    FITS_REPACKED false and flags '-nr -lm mmap+mlock'; 31b always NEVER_GOOD_SPEED with the s/token string;
    busy facts -> exit 4;
  * --emit env output is shell-safe (every line one shlex token, no unquoted spaces);
  * --sim overrides the assumed hit-rate curve (source 'simulated').
Plus: stdlib-only imports, io_mem/calibration merges, NO_DISK, cgroup caps, CLI exit codes.

The facts fixtures are inline (tests/fixtures/facts_*.json belong to tests/conftest.py's author); when those
files exist they are additionally checked against loose invariants. Runs on Windows and on the Pi without
models: stdlib + pytest only.
"""
from __future__ import annotations

import ast
import copy
import io
import json
import re
import shlex
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pytest

from pigemma import feasibility as F

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "pigemma" / "data" / "catalog.json"
FIXTURES = REPO_ROOT / "tests" / "fixtures"

# Spec literals (manifest files[] 'pigemma/feasibility.py' + catalog device profiles), used for hand arithmetic.
KV_26B = 160e6
KV_31B = 420e6
EFF = 0.65
MEM_BW = 10.0e9
CURVE = [(0.15, 0.60), (0.25, 0.74), (0.35, 0.84), (0.45, 0.88)]

_BASE: dict[str, Any] = {
    "hostname": "pi-lab", "kernel": "6.18.0-v8-16k+", "page_size": 16384, "cpu_model": "Cortex-A76",
    "cpu_flags": ["fp", "asimd", "evtstrm", "aes", "pmull", "sha1", "sha2", "crc32", "atomics", "fphp",
                  "asimdhp", "cpuid", "asimdrdm", "lrcpc", "dcpop", "asimddp"],
    "nproc": 4, "swap_total_kb": 0, "loadavg1": 0.25, "model_dir": "/nonexistent/pigemma-models",
    "free_disk_bytes": 400_000_000_000, "running_services": ["ssh.service", "cron.service"],
    "hat_product": None, "pciex1_gen_cfg": None, "governor": "performance", "memlock_limit": "unlimited",
    "tools": {"cmake": "3.31.6", "gcc": "14.2.0", "git": "2.47.3", "python3": "3.13.5"},
    "override": None, "collected_at": "2026-10-05T12:00:00Z",
}


def _storage(stype: str, speed: str | None = None, dev: str = "/dev/sda1") -> dict:
    return {"dev": dev, "type": stype, "rotational": False, "pcie_link_speed": speed,
            "pcie_link_width": 1 if speed else None, "read_ahead_kb": 128}


def make_facts(name: str, **over: Any) -> dict:
    """The five profiles of the manifest test field (RAM in decimal-nominal kB except the probed 4 GB)."""
    f = copy.deepcopy(_BASE)
    if name == "4gb-microsd":  # the probed context-friend-1 shape, but under a neutral hostname
        f.update(ram_total_kb=4_039_000, ram_available_kb=3_100_000, free_disk_bytes=15_000_000_000,
                 storage=_storage("microsd", dev="/dev/mmcblk0p2"), memlock_limit=8192, governor="ondemand")
    elif name == "8gb-usb3":
        f.update(ram_total_kb=7_812_500, ram_available_kb=7_000_000, storage=_storage("usb-ssd"))
    elif name == "8gb-nvme":
        f.update(ram_total_kb=7_812_500, ram_available_kb=7_000_000,
                 storage=_storage("nvme", "8.0 GT/s PCIe", "/dev/nvme0n1p1"), pciex1_gen_cfg=3)
    elif name == "16gb-nvme":
        f.update(ram_total_kb=15_625_000, ram_available_kb=15_000_000,
                 storage=_storage("nvme", "8.0 GT/s PCIe", "/dev/nvme0n1p1"), pciex1_gen_cfg=3)
    elif name == "busy":
        f.update(ram_total_kb=7_812_500, ram_available_kb=7_000_000,
                 storage=_storage("nvme", "8.0 GT/s PCIe", "/dev/nvme0n1p1"), loadavg1=3.6)
    else:
        raise KeyError(name)
    f.update(over)
    return f


PROFILES = ("4gb-microsd", "8gb-usb3", "8gb-nvme", "16gb-nvme")


@pytest.fixture(scope="module")
def catalog() -> dict:
    return F.load_catalog(CATALOG_PATH)


@pytest.fixture(scope="module")
def models(catalog: dict) -> dict[str, dict]:
    return {m["id"]: m for m in catalog["models"]}


def run(facts: dict, catalog: dict, **kw: Any) -> F.Verdict:
    kw.setdefault("check_present", False)
    return F.verdict(facts, copy.deepcopy(catalog), **kw)


def rows_by_id(v: F.Verdict) -> dict[str, dict]:
    return {r["model_id"]: r for r in v.rows}


def interp(x: float) -> float:
    """Independent re-implementation of the catalog curve rule (linear to (0,0), hold above)."""
    if x <= CURVE[0][0]:
        return CURVE[0][1] * x / CURVE[0][0]
    for (xa, ya), (xb, yb) in zip(CURVE, CURVE[1:]):
        if x <= xb:
            return ya + (yb - ya) * (x - xa) / (xb - xa)
    return CURVE[-1][1]


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = F.main(argv, environ={})
    return code, out.getvalue(), err.getvalue()


def write_facts(tmp_path: Path, facts: dict, name: str = "facts.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(facts), encoding="utf-8")
    return p


# ------------------------------------------------------------------------------------------- module shape
def test_stdlib_only_imports() -> None:
    src = Path(F.__file__).read_text(encoding="utf-8")
    allowed = set(sys.stdlib_module_names) | {"pigemma", "__future__"}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for n in names:
            assert n.split(".")[0] in allowed, f"non-stdlib import {n!r} in feasibility.py"


def test_interpolate_curve_rule() -> None:
    assert F.interpolate_curve(CURVE, 0.0) == 0.0
    assert F.interpolate_curve(CURVE, 0.075) == pytest.approx(0.30)
    assert F.interpolate_curve(CURVE, 0.15) == pytest.approx(0.60)
    assert F.interpolate_curve(CURVE, 0.30) == pytest.approx(0.79)
    assert F.interpolate_curve(CURVE, 0.45) == pytest.approx(0.88)
    assert F.interpolate_curve(CURVE, 0.90) == pytest.approx(0.88)


def test_kv_scaling_and_tiers() -> None:
    assert F.kv_bytes({"family": "26b"}, 4096) == 160_000_000
    assert F.kv_bytes({"family": "31b"}, 4096) == 420_000_000
    assert F.kv_bytes({"family": "26b"}, 2048) == 160_000_000  # never below the 4096 figure
    assert F.kv_bytes({"family": "26b"}, 8192) == 320_000_000
    assert F.kv_bytes({"family": "??"}, 4096) == 420_000_000
    assert F.nominal_ram_gb(4_039_000 * 1024) == 4
    assert F.nominal_ram_gb(8e9) == 8 and F.nominal_ram_gb(16e9) == 16
    assert F.tier_os_reserve(4_039_000 * 1024) == 700_000_000
    assert F.tier_os_reserve(8e9) == 800_000_000
    assert F.tier_os_reserve(16e9) == 900_000_000


# ------------------------------------------------------------------------------------------- 4 GB microSD
def test_4gb_microsd_not_viable_exit3(catalog: dict) -> None:
    v = run(make_facts("4gb-microsd"), catalog)
    assert v.exit_code == 3 and not v.ok and not v.host_refused and v.recommended is None
    for r in v.rows:
        if r["family"] == "26b":
            assert r["verdict"] == F.NOT_VIABLE, (r["model_id"], r["reason"])
        else:
            assert r["verdict"] == F.NEVER_GOOD_SPEED, r["model_id"]
        assert not r["runnable"] and r["flags"] is None
    assert v.device["matched_catalog_device"] == "pi5-4gb-microsd"


@pytest.mark.parametrize("over", [
    {"running_services": ["ssh.service", "contextfriend-api.service"]},
    {"running_services": ["context-friend-worker.service"]},
    {"hostname": "context-friend-1"},
])
def test_4gb_protected_host_exit4(catalog: dict, over: dict) -> None:
    v = run(make_facts("4gb-microsd", **over), catalog)
    assert v.exit_code == 4 and v.host_refused and not v.ok
    assert v.recommended is None
    assert v.rows, "rows are still computed for the operator (informational)"
    assert all(r["verdict"] in (F.NOT_VIABLE, F.NEVER_GOOD_SPEED) for r in v.rows)
    assert any("protected" in r for r in v.refusal_reasons)
    assert v.to_dict()["rows_informational_only"] is True


def test_override_lifts_protected_refusal_only(catalog: dict) -> None:
    f = make_facts("4gb-microsd", running_services=["contextfriend-api.service"], override="PIGEMMA_I_KNOW")
    v = run(f, catalog)
    assert v.exit_code == 3 and not v.host_refused
    assert any("overridden" in r for r in v.reasons)
    busy = make_facts("busy", hostname="context-friend-1", override="PIGEMMA_I_KNOW")
    assert run(busy, catalog).exit_code == 4  # the override never lifts the busy refusal


# --------------------------------------------------------------------------------------------- 8 GB USB3
def test_8gb_usb3_stream_matches_hand_arithmetic(catalog: dict, models: dict) -> None:
    v = run(make_facts("8gb-usb3"), catalog)
    r = rows_by_id(v)["26b-q4"]
    m = models["26b-q4"]
    # profile pi5-8gb-usb3: RAM 8e9 (facts), OS 0.8e9, KV 160e6 (26B @4096 q8_0), compute 0.3e9
    usable = 8.0e9 - 0.8e9 - KV_26B - 0.3e9
    cache = usable - 1_577_172_088
    frac = cache / 12_846_366_720
    hit = interp(frac)
    miss = (1 - hit) * 802_897_920
    n_miss = (1 - hit) * 240
    t_io = miss / 350e6 + n_miss * 0.5e-3
    t_ram = (1_577_172_088 + hit * 802_897_920) / (MEM_BW * EFF)
    tok = 1 / (t_ram + t_io)
    assert m["expert_bytes_per_token_worst"] == 802_897_920 and m["n_moe_layers"] * m["n_expert_used"] == 240
    assert r["usable_bytes"] == pytest.approx(usable)
    assert r["expert_cache_bytes"] == pytest.approx(cache)
    assert r["resident_fraction"] == pytest.approx(frac)
    assert r["hit_rate"] == pytest.approx(hit) and r["hit_kind"] == "assumed"
    assert r["t_io_s"] == pytest.approx(t_io) and r["t_ram_s"] == pytest.approx(t_ram)
    assert r["tok_s"] == pytest.approx(tok, rel=1e-9)
    assert tok == pytest.approx(1.4596, abs=5e-4)  # pinned: catalog values as of 2026-10-05
    expected = F.STREAM_OK if (tok >= 1.0 and cache >= 2e9) else (F.STREAM_SLOW if tok >= 0.3 else F.NOT_VIABLE)
    assert r["verdict"] == expected
    assert v.exit_code == (0 if expected == F.STREAM_OK else 2)


# --------------------------------------------------------------------------------------------- 8 GB NVMe
def test_8gb_nvme_stream_ok_with_nr(catalog: dict) -> None:
    v = run(make_facts("8gb-nvme"), catalog)
    assert v.exit_code == 0 and v.ok
    r = rows_by_id(v)["26b-q4"]
    assert r["verdict"] == F.STREAM_OK
    assert not r["fits_mapped"] and not r["fits_repacked"]
    assert v.device["storage"]["class"] == "nvme-gen3"
    # hand check: rand 700 MB/s, access 0.2 ms
    usable = 8.0e9 - 0.8e9 - KV_26B - 0.3e9
    hit = interp((usable - 1_577_172_088) / 12_846_366_720)
    t_io = (1 - hit) * 802_897_920 / 700e6 + (1 - hit) * 240 * 0.2e-3
    t_ram = (1_577_172_088 + hit * 802_897_920) / (MEM_BW * EFF)
    assert r["tok_s"] == pytest.approx(1 / (t_io + t_ram), rel=1e-9)
    rec = v.recommended
    assert rec is not None and rec["family"] == "26b"
    assert rec["flags"].startswith("-nr ") and "-lm mmap" in rec["flags"] and "-b 256 -ub 64" in rec["flags"]
    assert rec["model_id"] == "26b-q4"  # all 26b quants STREAM_OK -> higher-bit quant wins
    assert rec["pin_budget_bytes"] > 0


# -------------------------------------------------------------------------------------------- 16 GB NVMe
def test_16gb_nvme_fits_mapped_not_repacked(catalog: dict) -> None:
    v = run(make_facts("16gb-nvme"), catalog)
    assert v.exit_code == 0 and v.ok
    r = rows_by_id(v)["26b-q4"]
    usable = 16.0e9 - 0.9e9 - KV_26B - 0.4e9  # pi5-16gb-nvme-gen3: compute buffers 0.4e9
    assert r["usable_bytes"] == pytest.approx(usable)
    assert r["verdict"] == F.FITS_MAPPED
    assert r["fits_mapped"] is True and r["fits_repacked"] is False
    assert "-nr -lm mmap+mlock" in r["flags"]
    rec = v.recommended
    assert rec["model_id"] == "26b-q4" and rec["fits_repacked"] is False
    assert rec["flags"] == "-nr -lm mmap+mlock -t 4 -c 4096 -fa on -ctk q8_0 -ctv q8_0"
    assert v.to_dict()["llama_flags"] == rec["flags"]
    # in-RAM bound: (non-expert + worst expert bytes) / (mem_bw x eff)
    assert r["tok_s"] == pytest.approx(MEM_BW * EFF / (1_577_172_088 + 802_897_920), rel=1e-9)


def test_repacked_flags_drop_nr(catalog: dict) -> None:
    v = run(make_facts("16gb-nvme"), catalog)
    r = rows_by_id(v)["26b-ud-iq2_xxs"]  # 9.92 GB + 0.3 GB <= usable - 1 GB
    assert r["verdict"] == F.FITS_REPACKED and r["fits_repacked"] is True
    assert r["flags"] == "-lm mmap -t 4 -c 4096 -fa on -ctk q8_0 -ctv q8_0"


# --------------------------------------------------------------------------------------------- 31b dense
@pytest.mark.parametrize("profile", PROFILES)
def test_31b_always_never_good_speed(catalog: dict, profile: str) -> None:
    v = run(make_facts(profile), catalog)
    dense = [r for r in v.rows if r["family"] == "31b"]
    assert len(dense) == 6
    for r in dense:
        assert r["verdict"] == F.NEVER_GOOD_SPEED, (profile, r["model_id"])
        assert "s/token" in r["reason"]
        assert r["tok_s"] < 1.0


def test_31b_stream_string_is_file_over_storage_bw(catalog: dict) -> None:
    v = run(make_facts("16gb-nvme"), catalog)
    r = rows_by_id(v)["31b-q4"]
    s_tok = 17_651_001_568 / 850e6  # file bytes / NVMe Gen3 seq read
    assert r["s_per_token"] == pytest.approx(s_tok, rel=1e-9)
    assert f"{s_tok:.1f} s/token" in r["reason"] and "850 MB/s" in r["reason"]
    assert not r["runnable"]


def test_want_31b_on_16gb_is_slow_only(catalog: dict) -> None:
    v = run(make_facts("16gb-nvme"), catalog, want="31b")
    assert {r["family"] for r in v.rows} == {"31b"}
    assert v.exit_code == 2 and not v.ok and v.runnable
    rec = v.recommended
    assert rec["verdict"] == F.NEVER_GOOD_SPEED and rec["runnable"]
    assert 0.3 <= rec["tok_s"] < 1.0
    assert rec["flags"] is not None
    with pytest.raises(ValueError):
        run(make_facts("16gb-nvme"), catalog, want="70b")


def test_want_model_id(catalog: dict) -> None:
    v = run(make_facts("8gb-nvme"), catalog, want="26b-ud-iq2_xxs")
    assert [r["model_id"] for r in v.rows] == ["26b-ud-iq2_xxs"]
    assert v.recommended["model_id"] == "26b-ud-iq2_xxs"


# --------------------------------------------------------------------------------------------------- busy
def test_busy_loadavg_exit4(catalog: dict) -> None:
    v = run(make_facts("busy"), catalog)
    assert v.exit_code == 4 and v.busy and not v.host_refused and v.recommended is None
    assert any("loadavg1" in r for r in v.refusal_reasons)


def test_busy_low_memavailable_exit4(catalog: dict) -> None:
    v = run(make_facts("8gb-nvme", ram_available_kb=500_000), catalog)
    assert v.exit_code == 4 and v.busy
    assert any("MemAvailable" in r for r in v.refusal_reasons)


def test_busy_threshold_scales_with_nproc(catalog: dict) -> None:
    # a 12-vCPU emulated container at loadavg 3.6 is not busy (limit 2.0 x 12/4 = 6.0) ...
    assert run(make_facts("busy", nproc=12), catalog).exit_code == 0
    # ... unless an explicit limit is given
    assert run(make_facts("busy", nproc=12), catalog, max_loadavg1=2.0).exit_code == 4


# ------------------------------------------------------------------------------------------------- disk
def test_no_disk(catalog: dict) -> None:
    v = run(make_facts("16gb-nvme", free_disk_bytes=5_000_000_000), catalog)
    assert v.exit_code == 3 and v.recommended is None
    for r in v.rows:
        if r["family"] == "26b":
            assert r["verdict"] == F.NO_DISK and r["verdict_without_disk"] in (F.FITS_MAPPED, F.FITS_REPACKED)
        else:
            assert r["verdict"] == F.NEVER_GOOD_SPEED and not r["runnable"]


def test_present_file_skips_disk_check(tmp_path: Path, catalog: dict) -> None:
    cat = copy.deepcopy(catalog)
    tiny = next(m for m in cat["models"] if m["id"] == "26b-q4")
    tiny.update(id="tiny-moe", file="tiny-moe.gguf", repo="test/tiny", bytes=4096, nonexpert_bytes=1024,
                expert_bytes=3072, expert_bytes_per_token_worst=480, nonexpert_bytes_per_token=1024,
                expert_unit_bytes=2)
    cat["models"] = [tiny]
    f = make_facts("16gb-nvme", free_disk_bytes=0, model_dir=str(tmp_path))
    assert F.verdict(f, copy.deepcopy(cat)).rows[0]["verdict"] == F.NO_DISK
    (tmp_path / "tiny-moe.gguf").write_bytes(b"\0" * 4096)
    r = F.verdict(f, copy.deepcopy(cat)).rows[0]
    assert r["present_bytes"] == 4096 and r["disk_ok"] is True
    assert r["verdict"] in (F.FITS_MAPPED, F.FITS_REPACKED)


# -------------------------------------------------------------------------------------------- emit env
_ENV_LINE = re.compile(r"^[A-Z][A-Z0-9_]*=(?:'[^']*'|[^\s'\"]*)$")


def _parse_env(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        assert _ENV_LINE.match(line), f"not shell-safe: {line!r}"
        toks = shlex.split(line)
        assert len(toks) == 1, f"line splits into {len(toks)} shell words: {line!r}"
        k, _, val = toks[0].partition("=")
        env[k] = val
    return env


def test_emit_env_shell_safe(tmp_path: Path) -> None:
    fp = write_facts(tmp_path, make_facts("16gb-nvme"))
    code, out, err = run_cli(["--facts", str(fp), "--io-mem", "none", "--calibration", "none", "--emit", "env"])
    assert code == 0
    env = _parse_env(out)
    for key in ("MODEL_ID", "MODEL_REPO", "MODEL_FILE", "MODEL_BYTES", "LLAMA_FLAGS", "DRAFT_REPO",
                "DRAFT_FILE", "PIN_BUDGET_BYTES", "VERDICT_EXIT"):
        assert key in env, key
    assert env["MODEL_ID"] == "26b-q4" and env["MODEL_BYTES"] == "14439363584"
    assert env["LLAMA_FLAGS"] == "-nr -lm mmap+mlock -t 4 -c 4096 -fa on -ctk q8_0 -ctv q8_0"
    assert "LLAMA_FLAGS='-nr -lm mmap+mlock" in out  # spaces only inside quotes
    assert env["DRAFT_FILE"] == "mtp-gemma-4-26B-A4B-it-Q4_0.gguf" and env["DRAFT_SPEC_TYPE"] == "draft-mtp"
    assert env["FITS_REPACKED"] == "0" and env["VERDICT_EXIT"] == "0"
    assert "recommended:" in err  # the human report goes to stderr in --emit mode


def test_emit_env_when_refused(tmp_path: Path) -> None:
    fp = write_facts(tmp_path, make_facts("4gb-microsd", hostname="context-friend-1"))
    code, out, err = run_cli(["--facts", str(fp), "--io-mem", "none", "--calibration", "none", "--emit", "env",
                              "--quiet"])
    assert code == 4
    env = _parse_env(out)
    assert env["MODEL_ID"] == "" and env["LLAMA_FLAGS"] == "" and env["VERDICT_EXIT"] == "4"
    assert "REFUSED" in err


# ------------------------------------------------------------------------------------------------- sim
def _sim(points: list[tuple[int, float]]) -> dict:
    return {"trace": "traces/26b/all.moetrace", "map": "results/map26.json", "runs": [
        {"policy": "lru", "budget_bytes": cap * 3_345_408, "capacity_items": cap, "profile": None,
         "steady": {"hit_rate": hit}, "per_layer_hit_rate": [hit] * 30} for cap, hit in points
    ] + [{"policy": "belady", "budget_bytes": 3_345_408 * 100, "capacity_items": 100,
          "steady": {"hit_rate": 0.999}, "per_layer_hit_rate": [0.999] * 30}]}


def test_sim_overrides_assumed_curve(tmp_path: Path, catalog: dict) -> None:
    sim = _sim([(0, 0.0), (1000, 0.90), (2000, 0.97), (3840, 1.0)])
    facts = make_facts("8gb-usb3")
    base = rows_by_id(run(facts, catalog))["26b-q4"]
    v = run(facts, catalog, sim=sim)
    r = rows_by_id(v)["26b-q4"]
    assert v.hit_source == "simulated" and r["hit_kind"] == "simulated"
    assert r["hit_source"].startswith("simulated")
    items = r["expert_cache_bytes"] / 3_345_408
    assert 1000 < items < 2000
    assert r["hit_rate"] == pytest.approx(0.90 + (items - 1000) / 1000 * 0.07)
    assert r["hit_rate"] != pytest.approx(base["hit_rate"]) and r["tok_s"] > base["tok_s"]
    assert "simulated" in r["tok_s_source"]
    # dense rows are untouched by the sim
    assert rows_by_id(v)["31b-q4"]["hit_kind"] is None
    # the same through the CLI and a sim.json path
    sp = tmp_path / "sim.json"
    sp.write_text(json.dumps(sim), encoding="utf-8")
    fp = write_facts(tmp_path, facts)
    jp = tmp_path / "out.json"
    code, _, _ = run_cli(["--facts", str(fp), "--sim", str(sp), "--json", str(jp), "--io-mem", "none",
                          "--calibration", "none", "--quiet"])
    data = json.loads(jp.read_text(encoding="utf-8"))
    assert code == data["exit_code"] and data["hit_source"] == "simulated"
    assert {x["model_id"]: x for x in data["rows"]}["26b-q4"]["hit_rate"] == pytest.approx(r["hit_rate"])


def test_sim_without_lru_falls_back(catalog: dict) -> None:
    sim = {"runs": [{"policy": "belady", "capacity_items": 10, "steady": {"hit_rate": 0.5}}]}
    v = run(make_facts("8gb-usb3"), catalog, sim=sim)
    assert v.hit_source == "assumed"
    assert any("no usable 'lru' runs" in r for r in v.reasons)


# --------------------------------------------------------------------------- io_mem / calibration / device
IO_MEM = {
    "host": "pi-lab", "device": "pi5-8gb-nvme-gen3", "collected_at": "2026-10-05T12:00:00Z",
    "storage": {"seq_read_mbps_qd1": {"value": 790.0, "source": "measured", "cmd": "fio seq qd1"},
                "seq_read_mbps_qd8": {"value": 805.0, "source": "measured", "cmd": "fio seq qd8"},
                "rand_read_3mb_mbps_qd1": {"value": 600.0, "source": "measured", "cmd": "fio r3m qd1"},
                "rand_read_3mb_mbps_qd4": {"value": 640.0, "source": "measured", "cmd": "fio r3m qd4"},
                "access_ms": {"value": 0.15, "source": "measured", "cmd": "16k qd1 latency"}},
    "memory": {"read_gbps_by_threads": {"1": 5.0, "2": 7.0, "4": 9.0}, "copy_gbps": 4.0, "triad_gbps": 6.0,
               "source": "measured"},
}


def test_io_mem_merge_and_host_match(tmp_path: Path) -> None:
    before = CATALOG_PATH.read_bytes()
    p = tmp_path / "io_mem.json"
    p.write_text(json.dumps(IO_MEM), encoding="utf-8")
    cat = F.load_catalog(CATALOG_PATH, io_mem=p)
    dev = next(d for d in cat["devices"] if d["name"] == "pi5-8gb-nvme-gen3")
    assert dev["storage"]["seq_read_mbps"]["value"] == 805.0  # max(qd1, qd8)
    assert dev["storage"]["rand_read_3mb_mbps"]["value"] == 600.0  # qd1 (conservative)
    assert dev["mem_bw_gbps"]["value"] == 9.0  # best thread count
    for fld in (dev["storage"]["seq_read_mbps"], dev["storage"]["access_ms"], dev["mem_bw_gbps"]):
        assert fld["source"].startswith("measured") and "assumed" not in fld["source"]
    assert CATALOG_PATH.read_bytes() == before
    # the measured profile is used for facts from the same host even though its link reads Gen2
    v = F.verdict(make_facts("8gb-nvme", storage=_storage("nvme", "5.0 GT/s PCIe")), cat, check_present=False)
    assert v.device["matched_catalog_device"] == "pi5-8gb-nvme-gen3"
    assert v.assumptions["rand_read_3mb_mbps"]["value"] == 600.0
    assert v.assumptions["mem_bw_gbps"]["source"].startswith("measured")
    r = rows_by_id(v)["26b-q4"]
    assert "storage measured" in r["tok_s_source"] and "mem_bw measured" in r["tok_s_source"]
    # another host keeps the class-based assumed numbers (nvme-gen2: 400 MB/s)
    v2 = F.verdict(make_facts("8gb-nvme", hostname="other", storage=_storage("nvme", "5.0 GT/s PCIe")), cat,
                   check_present=False)
    assert v2.device["matched_catalog_device"] is None
    assert v2.device["storage"]["class"] == "nvme-gen2"
    assert v2.assumptions["rand_read_3mb_mbps"]["value"] == 400


def test_io_mem_unknown_device_is_synthesized() -> None:
    io = dict(copy.deepcopy(IO_MEM), device="pi5-8gb-nvme-gen2")
    cat = F.load_catalog(CATALOG_PATH, io_mem=io)
    dev = next(d for d in cat["devices"] if d["name"] == "pi5-8gb-nvme-gen2")
    assert dev["synthesized"] and dev["ram_bytes"] == 8_000_000_000
    assert dev["storage"]["class"] == "nvme-gen2"
    assert dev["storage"]["rand_read_3mb_mbps"]["source"].startswith("measured")


def test_calibration_overrides_efficiency(catalog: dict) -> None:
    facts = make_facts("16gb-nvme")
    base = rows_by_id(run(facts, catalog))["26b-q4"]
    v = run(facts, catalog, calibration={"llama_efficiency": {"value": 0.5, "source": "measured: pi/05"}})
    r = rows_by_id(v)["26b-q4"]
    assert r["tok_s"] == pytest.approx(base["tok_s"] * 0.5 / 0.65, rel=1e-9)
    assert v.assumptions["llama_efficiency"]["value"] == 0.5
    assert "eff measured" in r["tok_s_source"]
    with pytest.raises(ValueError):
        run(facts, catalog, calibration={"llama_efficiency": {"value": 7}})


def test_storage_class_detection(catalog: dict) -> None:
    assert F.storage_class_for(_storage("nvme", "8.0 GT/s PCIe"), catalog)[1] == "nvme-gen3"
    assert F.storage_class_for(_storage("nvme", "5.0 GT/s PCIe"), catalog)[1] == "nvme-gen2"
    t, c, notes = F.storage_class_for(_storage("nvme", None), catalog)
    assert c == "nvme-gen2" and notes
    assert F.storage_class_for(_storage("usb-ssd"), catalog)[1] == "usb-ssd"
    assert F.storage_class_for(_storage("microsd"), catalog)[1] == "microsd"
    assert F.storage_class_for({"type": "overlay"}, catalog)[:2] == ("unknown", "unknown")
    assert F.storage_class_for(None, catalog)[:2] == ("unknown", "unknown")


def test_synthesized_profile_uses_tier_rule(catalog: dict) -> None:
    # 8 GB + NVMe Gen2: no catalog device has that class -> tier OS reserve, default compute, class metrics
    v = run(make_facts("8gb-nvme", storage=_storage("nvme", "5.0 GT/s PCIe")), catalog)
    assert v.device["matched_catalog_device"] is None
    assert v.device["os_reserve_bytes"] == 800_000_000 and v.device["compute_buffers_bytes"] == 300_000_000
    assert v.assumptions["rand_read_3mb_mbps"]["value"] == 400
    assert v.assumptions["os_reserve_bytes"]["source"].startswith("assumed")


def test_cgroup_cap_and_emulated_label(catalog: dict) -> None:
    f = make_facts("8gb-nvme", ram_total_kb=30_000_000, ram_available_kb=29_000_000, cgroup_mem_max=8_000_000_000,
                   nproc=12, loadavg1=3.0)
    v = run(f, catalog, emulated=True)
    assert v.device["ram_total_bytes"] == 8_000_000_000
    assert v.exit_code == 0
    assert any("cgroup" in r for r in v.reasons)
    assert v.reasons[0].startswith("EMULATED")


def test_explicit_device_and_unknown_device(catalog: dict) -> None:
    v = run(make_facts("8gb-usb3"), catalog, device="pi5-8gb-nvme-gen3")
    assert v.device["matched_catalog_device"] == "pi5-8gb-nvme-gen3"
    assert v.assumptions["rand_read_3mb_mbps"]["value"] == 700
    with pytest.raises(ValueError):
        run(make_facts("8gb-usb3"), catalog, device="pi9-1tb")


# ---------------------------------------------------------------------------------------- common output
@pytest.mark.parametrize("name", PROFILES + ("busy",))
def test_recommendation_line_and_projection_labels(catalog: dict, name: str) -> None:
    v = run(make_facts(name), catalog)
    assert v.reasons[-1] == F.RECOMMENDATION_LINE
    assert v.to_dict()["kind"] == "projected"
    for r in v.rows:
        assert r["tok_s_kind"] == "projected"
        if r["tok_s"] is not None and r["tok_s"] > 0:
            assert r["tok_s_source"]
    json.dumps(v.to_dict(), allow_nan=False)  # strict JSON


def test_memlock_warning(catalog: dict) -> None:
    v = run(make_facts("16gb-nvme", memlock_limit=8192), catalog)
    assert v.recommended["flags"].startswith("-nr -lm mmap+mlock")  # flags stay as specified
    assert any("memlock limit 8192 kB" in r for r in v.reasons)


def test_ai_hat_note(catalog: dict) -> None:
    v = run(make_facts("8gb-nvme", hat_product="Raspberry Pi AI HAT+ 2 (Hailo-10H)"), catalog)
    assert any("AI HAT detected" in r for r in v.reasons)


# ---------------------------------------------------------------------------------------------------- CLI
def test_cli_table_json_and_exit_codes(tmp_path: Path) -> None:
    fp = write_facts(tmp_path, make_facts("4gb-microsd"))
    jp = tmp_path / "sub" / "feasibility.json"
    code, out, _ = run_cli(["--facts", str(fp), "--json", str(jp), "--io-mem", "none", "--calibration", "none"])
    assert code == 3
    assert "NOT_VIABLE" in out and "verdict: exit 3" in out and F.RECOMMENDATION_LINE in out
    data = json.loads(jp.read_text(encoding="utf-8"))
    assert data["exit_code"] == 3 and data["recommended"] is None and len(data["rows"]) == 12
    assert data["inputs"]["io_mem"] is None
    fp2 = write_facts(tmp_path, make_facts("8gb-nvme"), "f2.json")
    code, out, _ = run_cli(["--facts", str(fp2), "--io-mem", "none", "--calibration", "none", "--want", "26b"])
    assert code == 0 and "31b-q4" not in out


def test_cli_errors_exit_1_not_2(tmp_path: Path) -> None:
    code, _, err = run_cli(["--facts", str(tmp_path / "missing.json"), "--io-mem", "none"])
    assert code == 1 and "ERROR" in err
    with pytest.raises(SystemExit) as ei, redirect_stderr(io.StringIO()):
        F.main(["--bogus-flag"], environ={})
    assert ei.value.code == 1  # exit 2 is reserved for STREAM_SLOW
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"hostname": "x"}), encoding="utf-8")
    assert run_cli(["--facts", str(bad), "--io-mem", "none"])[0] == 1


def test_cli_env_overrides(tmp_path: Path) -> None:
    fp = write_facts(tmp_path, make_facts("16gb-nvme"))
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = F.main(["--facts", str(fp), "--io-mem", "none", "--calibration", "none", "--emit", "env", "--quiet"],
                      environ={"THREADS": "3", "CTX": "8192", "DENY_HOSTS": "other-host pi-lab"})
    assert code == 4  # DENY_HOSTS from pi_config.env joins catalog.protected_hosts
    with redirect_stdout(out := io.StringIO()), redirect_stderr(io.StringIO()):
        code = F.main(["--facts", str(fp), "--io-mem", "none", "--calibration", "none", "--emit", "env", "--quiet"],
                      environ={"THREADS": "3", "CTX": "8192"})
    env = _parse_env(out.getvalue())
    assert code == 0 and "-t 3 -c 8192" in env["LLAMA_FLAGS"]


# --------------------------------------------------- optional: the conftest author's facts fixtures (if present)
@pytest.mark.parametrize("path", sorted(FIXTURES.glob("facts_*.json")) or [None])
def test_conftest_facts_fixtures_invariants(path: Path | None, catalog: dict) -> None:
    if path is None:
        pytest.skip("tests/fixtures/facts_*.json not written yet (tests/conftest.py owner)")
    facts = json.loads(path.read_text(encoding="utf-8"))
    v = F.verdict(facts, copy.deepcopy(catalog), check_present=False)
    assert v.exit_code in (0, 2, 3, 4)
    assert v.reasons[-1] == F.RECOMMENDATION_LINE
    if v.device["platform"] == "pi5" or "pi5" in path.name:
        assert all(r["verdict"] == F.NEVER_GOOD_SPEED for r in v.rows if r["family"] == "31b")
    if "4gb" in path.name:
        assert v.exit_code in (3, 4)
