"""Tests for the Pi runtime: pi/lib/common.sh + pi/pi_config.env, plus `bash -n` on every pi/pc script.

Manifest test field (files[] 'pi/lib/common.sh (+ pi/pi_config.env)'):
  * `bash -n` on every pi/*.sh and pc/*.sh (also pi/lib/*.sh and pi/pi_config.env); shellcheck when on PATH;
  * with PIGEMMA_FAKE_PROC=tests/fixtures/pi_proc_4gb collect_facts produces JSON with MemTotal 4039000,
    storage type microsd, page_size 16384, and assert_safe_host exits 4 (hostname context-friend-1);
  * the 16 GB fixture passes.
Plus: the facts.json key contract (manifest data_formats 'facts.json'), the PIGEMMA_I_KNOW override, the
container fixture (no systemd, overlay root, cgroup cap), require_verdict_ok / feasibility_env against the
real pigemma.feasibility, pick_jobs, fmt_gb, require_free_disk (KEEP_FREE_GB on the root fs), die codes,
config precedence/validation, JSON escaping, mlock safety and the direct-execution CLI.

The fake system roots live in tests/fixtures/pi_proc_<name>/ (layout documented in pi/lib/common.sh).
bash is Git Bash on Windows (pigemma.paths.git_bash(), never WSL's System32 bash) and /bin/bash elsewhere;
everything is skipped when no bash is found. Later pi/*.sh scripts are covered by the `bash -n` sweep
automatically (collected by glob); their own behaviour tests can be added below or in sibling modules.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures"
COMMON = "pi/lib/common.sh"

#: Keys required by manifest data_formats 'facts.json'.
FACTS_KEYS = {
    "hostname", "kernel", "page_size", "cpu_model", "cpu_flags", "nproc", "ram_total_kb", "ram_available_kb",
    "swap_total_kb", "loadavg1", "model_dir", "storage", "free_disk_bytes", "running_services", "hat_product",
    "pciex1_gen_cfg", "governor", "memlock_limit", "tools", "override", "collected_at",
}
STORAGE_KEYS = {"dev", "type", "rotational", "pcie_link_speed", "pcie_link_width", "read_ahead_kb"}
TOOLS_KEYS = {"cmake", "gcc", "git", "python3"}

#: Config variables of pi/pi_config.env (stripped from the inherited environment so tests are hermetic).
CONFIG_VARS = (
    "PIGEMMA_HOME", "MODEL_DIR", "LLAMA_DIR", "LLAMA_COMMIT", "PIGEMMA_OUT", "THREADS", "CTX", "QUANT_OVERRIDE",
    "OS_RESERVE_GIB", "MIN_FREE_DISK_FACTOR", "KEEP_FREE_GB", "NICE_LEVEL", "DENY_HOSTS",
    "PROTECTED_SERVICE_GLOBS", "ALLOW_MLOCK", "TUNE_APPLY", "TUNE_NVME_GEN3", "TUNE_GOVERNOR", "TUNE_SWAPPINESS",
    "TUNE_READAHEAD_KB", "TUNE_MEMLOCK", "BUILD_JOBS", "HF_LIMIT_RATE", "PIN_BUDGET_BYTES",
)


# ------------------------------------------------------------------------------------------- helpers
def _find_bash() -> str | None:
    if sys.platform == "win32":
        try:
            from pigemma import paths
        except Exception:  # pragma: no cover - pigemma not importable
            return None
        gb = paths.git_bash()
        return str(gb) if gb else None
    return shutil.which("bash")


BASH = _find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no bash found (Git Bash on Windows)")


def _p(path: Path | str) -> str:
    """A path as bash (Git Bash or Linux) accepts it: forward slashes, drive letter kept."""
    return Path(path).as_posix()


def bash_env(fake: str | None = None, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("PIGEMMA_") and k not in CONFIG_VARS and k != "PY"}
    env["PIGEMMA_PYTHON"] = _p(sys.executable)
    env["LC_ALL"] = env.get("LC_ALL") or "C.UTF-8"
    if fake:
        env["PIGEMMA_FAKE_PROC"] = _p(FIX / fake)
    env.update(extra)
    return env


def run_bash(script: str, env: dict[str, str], *, timeout: float = 180) -> subprocess.CompletedProcess:
    assert BASH is not None
    return subprocess.run([BASH, "-c", script], cwd=REPO, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def lib(body: str) -> str:
    """A bash script that sources the library the way every pi script does."""
    return f"set -euo pipefail\n. {COMMON}\n{body}\n"


def facts_for(fake: str, tmp_path: Path, **extra: str) -> dict:
    env = bash_env(fake, MODEL_DIR=extra.pop("MODEL_DIR", "/srv/pigemma/models"),
                   PIGEMMA_OUT=_p(tmp_path), **extra)
    r = run_bash(lib("collect_facts"), env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def clone_fixture(src: str, dst: Path, **files: str) -> Path:
    """Copy a fixture root and overwrite / add files (relative path -> content; '' content deletes)."""
    shutil.copytree(FIX / src, dst)
    for rel, content in files.items():
        f = dst / rel.replace("__", "/")
        if content == "":
            f.unlink(missing_ok=True)
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8", newline="\n")
    return dst


# --------------------------------------------------------------------------------------- bash -n sweep
def _scripts() -> list[Path]:
    files = sorted({*REPO.glob("pi/*.sh"), *REPO.glob("pi/lib/*.sh"), *REPO.glob("pc/*.sh")})
    cfg = REPO / "pi" / "pi_config.env"
    if cfg.is_file():
        files.append(cfg)
    return files


@needs_bash
@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_bash_syntax(script: Path) -> None:
    r = subprocess.run([BASH, "-n", _p(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_shellcheck(script: Path) -> None:
    r = subprocess.run(["shellcheck", "-s", "bash", "-S", "warning", "-x", str(script)], cwd=REPO,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_scripts_have_lf_line_endings() -> None:
    for f in (REPO / "pi" / "lib" / "common.sh", REPO / "pi" / "pi_config.env"):
        assert b"\r" not in f.read_bytes(), f"{f} has CR characters (bash on the Pi would break)"


# ------------------------------------------------------------------------------------ manifest checks
@needs_bash
def test_collect_facts_4gb_fixture(tmp_path: Path) -> None:
    d = facts_for("pi_proc_4gb", tmp_path)
    assert FACTS_KEYS <= d.keys()
    assert STORAGE_KEYS <= d["storage"].keys()
    assert TOOLS_KEYS <= d["tools"].keys()
    assert d["ram_total_kb"] == 4039000
    assert d["ram_available_kb"] == 3100000
    assert d["storage"]["type"] == "microsd"
    assert d["page_size"] == 16384
    assert d["hostname"] == "context-friend-1"
    st = d["storage"]
    assert st["dev"] == "/dev/mmcblk0p2" and st["disk"] == "mmcblk0" and st["mount"] == "/"
    assert st["rotational"] is False and st["read_ahead_kb"] == 128
    assert st["pcie_link_speed"] is None and st["pcie_link_width"] is None
    assert d["model_dir_on_root_fs"] is True
    assert d["free_disk_bytes"] == 15_000_000_000
    assert d["cpu_model"] == "Cortex-A76"
    assert d["cpu_has"]["asimddp"] is True and d["cpu_has"]["i8mm"] is False and d["cpu_has"]["sve"] is False
    assert "asimddp" in d["cpu_flags"] and "fphp" in d["cpu_flags"]
    assert d["nproc"] == 4 and d["swap_total_kb"] == 2097136 and d["loadavg1"] == pytest.approx(0.42)
    assert "contextfriend-box.service" in d["running_services"]
    assert d["services_source"].startswith("systemctl")
    assert d["hat_product"] == "AI HAT+"  # trailing NUL of the device-tree string stripped
    assert d["board_model"] == "Raspberry Pi 5 Model B Rev 1.0"
    assert d["pciex1_gen_cfg"] == 3  # the commented '#dtparam=pciex1_gen=2' is ignored
    assert d["boot_config"] == "/boot/firmware/config.txt"
    assert d["governor"] == "ondemand"
    assert d["memlock_limit"] == 8192
    assert d["tools"] == {"cmake": "3.31.6", "gcc": "14.2.0", "git": "2.47.3", "python3": "3.13.5",
                          "fio": None, "curl": "8.14.1"}
    assert d["override"] is None
    assert d["cpu_temp_c"] == pytest.approx(51.2) and d["throttled"] == "0x0"
    assert d["cgroup_mem_max"] is None and d["in_container"] is False and d["emulated"] is False
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", d["collected_at"])
    assert d["model_dir"] == "/srv/pigemma/models"


@needs_bash
def test_assert_safe_host_refuses_4gb_fixture(tmp_path: Path) -> None:
    env = bash_env("pi_proc_4gb", PIGEMMA_OUT=_p(tmp_path))
    r = run_bash(lib("assert_safe_host; echo SHOULD_NOT_PRINT"), env)
    assert r.returncode == 4, r.stderr
    assert "SHOULD_NOT_PRINT" not in r.stdout
    assert "REFUSED" in r.stderr and "context-friend-1" in r.stderr
    assert "contextfriend-box.service" in r.stderr
    assert "PIGEMMA_I_KNOW=1" in r.stderr


@needs_bash
def test_16gb_fixture_passes(tmp_path: Path) -> None:
    d = facts_for("pi_proc_16gb", tmp_path, MODEL_DIR="/mnt/nvme/models")
    st = d["storage"]
    assert st["type"] == "nvme" and st["dev"] == "/dev/nvme0n1p1" and st["disk"] == "nvme0n1"
    assert st["pcie_link_speed"] == "8.0 GT/s PCIe" and st["pcie_link_width"] == 1
    assert st["read_ahead_kb"] == 2048 and st["mount"] == "/mnt/nvme"
    assert d["model_dir_on_root_fs"] is False
    assert d["pciex1_gen_cfg"] == 3  # comma form 'dtparam=pciex1,pciex1_gen=3'
    assert d["memlock_limit"] == "unlimited"
    assert d["hat_product"] is None
    assert d["ram_total_kb"] == 16607552
    env = bash_env("pi_proc_16gb", PIGEMMA_OUT=_p(tmp_path))
    r = run_bash(lib("assert_safe_host; echo SAFE_OK"), env)
    assert r.returncode == 0, r.stderr
    assert "SAFE_OK" in r.stdout  # friendly-context.service must not match 'contextfriend*'


# ------------------------------------------------------------------------------- host safety details
@needs_bash
@pytest.mark.parametrize("fixture", ["pi_proc_4gb", "pi_proc_cf1_16gb"])
def test_override_with_pigemma_i_know(fixture: str, tmp_path: Path) -> None:
    env = bash_env(fixture, PIGEMMA_OUT=_p(tmp_path), PIGEMMA_I_KNOW="1")
    r = run_bash(lib('assert_safe_host; echo "SAFE_OK override=$PIGEMMA_PROTECTED_OVERRIDE"'), env)
    assert r.returncode == 0, r.stderr
    assert "SAFE_OK override=1" in r.stdout
    assert "PROTECTED HOST OVERRIDE" in r.stderr and "no mlock" in r.stderr
    d = facts_for(fixture, tmp_path, PIGEMMA_I_KNOW="1")
    assert d["override"] == "PIGEMMA_I_KNOW"


@needs_bash
def test_cf1_16gb_refused_without_override(tmp_path: Path) -> None:
    r = run_bash(lib("assert_safe_host"), bash_env("pi_proc_cf1_16gb", PIGEMMA_OUT=_p(tmp_path)))
    assert r.returncode == 4
    d = facts_for("pi_proc_cf1_16gb", tmp_path)
    assert d["ram_total_kb"] == 16607552 and d["free_disk_bytes"] == 15501496320
    assert d["hat_product"] is None and d["pciex1_gen_cfg"] is None and d["tools"]["fio"] is None


@needs_bash
def test_deny_hosts_from_config_env(tmp_path: Path) -> None:
    env = bash_env("pi_proc_16gb", PIGEMMA_OUT=_p(tmp_path), DENY_HOSTS="other-host, PI5-LAB-16G")
    r = run_bash(lib("assert_safe_host"), env)
    assert r.returncode == 4
    assert "deny-listed" in r.stderr


@needs_bash
def test_catalog_protected_hosts_multiline(tmp_path: Path) -> None:
    cat = json.loads((REPO / "pigemma" / "data" / "catalog.json").read_text(encoding="utf-8"))
    cat["protected_hosts"] = ["pi5-lab-16g"]
    cat["protected_service_globs"] = []
    alt = tmp_path / "catalog.json"
    alt.write_text(json.dumps(cat, indent=2), encoding="utf-8")  # arrays span several lines here
    env = bash_env("pi_proc_16gb", PIGEMMA_OUT=_p(tmp_path), PIGEMMA_CATALOG=_p(alt))
    r = run_bash(lib("assert_safe_host"), env)
    assert r.returncode == 4, r.stderr
    assert "pi5-lab-16g" in r.stderr


@needs_bash
def test_service_glob_override_and_suffix(tmp_path: Path) -> None:
    # a config glob without the .service suffix still matches the full unit name
    env = bash_env("pi_proc_16gb", PIGEMMA_OUT=_p(tmp_path), PROTECTED_SERVICE_GLOBS="ssh")
    r = run_bash(lib("assert_safe_host"), env)
    assert r.returncode == 4
    assert "ssh.service" in r.stderr


@needs_bash
def test_container_fixture(tmp_path: Path) -> None:
    d = facts_for("pi_proc_container", tmp_path, PIGEMMA_EMULATED="1", MODEL_DIR="/work/models")
    assert d["storage"]["type"] == "unknown" and d["storage"]["dev"] == "overlay"
    assert d["storage"]["notes"] and "unknown" in d["storage"]["notes"][0]
    assert d["running_services"] == []
    assert d["services_source"].startswith("unavailable")
    assert d["cgroup_mem_max"] == 8589934592
    assert d["ram_available_kb"] == (8589934592 - (300000000 - 100000000)) // 1024
    assert d["meminfo_available_kb"] == 24186928
    assert d["page_size"] == 4096 and d["nproc"] == 12
    assert d["in_container"] is True and d["emulated"] is True
    assert d["hat_product"] is None and d["pciex1_gen_cfg"] is None and d["boot_config"] is None
    assert d["governor"] is None and d["board_model"] is None
    assert d["cpu_model"] == "ARMv8 Processor rev 0 (v8l)"
    assert any("boot config" in w for w in d["warnings"])
    r = run_bash(lib("assert_safe_host; echo SAFE_OK"), bash_env("pi_proc_container", PIGEMMA_OUT=_p(tmp_path)))
    assert r.returncode == 0 and "SAFE_OK" in r.stdout
    assert "could not be listed" in r.stderr


@needs_bash
def test_usb_and_sata_storage_variants(tmp_path: Path) -> None:
    usb = clone_fixture("pi_proc_16gb", tmp_path / "usb", **{
        "findmnt.txt": 'SOURCE="/dev/sda2" FSTYPE="ext4" TARGET="/mnt/ssd" MAJ:MIN="8:2"\n',
        "lsblk.txt": "mmcblk0\nsda usb\n",
        "sys/block/sda/queue/rotational": "1\n",
        "sys/block/sda/queue/read_ahead_kb": "128\n",
    })
    env = bash_env(None, PIGEMMA_FAKE_PROC=_p(usb), PIGEMMA_OUT=_p(tmp_path), MODEL_DIR="/mnt/ssd/models")
    d = json.loads(run_bash(lib("collect_facts"), env).stdout)
    assert d["storage"]["type"] == "usb-ssd" and d["storage"]["disk"] == "sda"
    assert d["storage"]["transport"] == "usb" and d["storage"]["rotational"] is True
    assert any("rotational=1" in n for n in d["storage"]["notes"])
    sata = clone_fixture("pi_proc_16gb", tmp_path / "sata", **{
        "findmnt.txt": "/dev/sda1\n",  # plain SOURCE form is accepted too
        "lsblk.txt": "sda sata\n",
    })
    env = bash_env(None, PIGEMMA_FAKE_PROC=_p(sata), PIGEMMA_OUT=_p(tmp_path))
    d = json.loads(run_bash(lib("collect_facts"), env).stdout)
    assert d["storage"]["type"] == "unknown" and d["storage"]["disk"] == "sda"
    assert d["storage"]["transport"] == "sata"


@needs_bash
def test_json_escaping_and_strict_parse(tmp_path: Path) -> None:
    weird = 'C:/odd dir/with "quotes" and \\backslash\tand tab'
    d = facts_for("pi_proc_16gb", tmp_path, MODEL_DIR=weird)
    assert d["model_dir"] == weird


# ---------------------------------------------------------------------------- verdict (feasibility)
@needs_bash
@pytest.mark.parametrize("fixture,i_know,expected_rc", [
    ("pi_proc_16gb", "0", 0),
    ("pi_proc_4gb", "0", 4),
    ("pi_proc_4gb", "1", 3),  # 4 GB + microSD: nothing viable
])
def test_require_verdict_ok(fixture: str, i_know: str, expected_rc: int, tmp_path: Path) -> None:
    env = bash_env(fixture, PIGEMMA_OUT=_p(tmp_path), PIGEMMA_I_KNOW=i_know)
    script = lib('require_verdict_ok --quiet -- --io-mem none --calibration none\n'
                 'echo "RC=$PIGEMMA_VERDICT_RC"')
    r = run_bash(script, env)
    assert r.returncode == expected_rc, r.stdout + r.stderr
    facts = json.loads((tmp_path / "facts.json").read_text(encoding="utf-8"))
    assert facts["ram_total_kb"] > 0
    fz = json.loads((tmp_path / "feasibility.json").read_text(encoding="utf-8"))
    assert fz["exit_code"] == expected_rc
    if expected_rc == 0:
        assert "RC=0" in r.stdout
        assert fz["recommended"]["model_id"] == "26b-q4"
        assert "-nr" in fz["recommended"]["flags"]


@needs_bash
def test_require_verdict_allow_slow(tmp_path: Path) -> None:
    # the container profile (unknown storage, 8 GiB cgroup) only streams slowly -> exit 2
    env = bash_env("pi_proc_container", PIGEMMA_OUT=_p(tmp_path))
    gate = 'require_verdict_ok --quiet {opt} -- --io-mem none --calibration none\necho "RC=$PIGEMMA_VERDICT_RC"'
    r = run_bash(lib(gate.format(opt="")), env)
    assert r.returncode == 2, r.stderr
    r = run_bash(lib(gate.format(opt="--allow-slow")), env)
    assert r.returncode == 0, r.stderr
    assert "RC=2" in r.stdout


@needs_bash
def test_feasibility_env_sources_recommendation(tmp_path: Path) -> None:
    env = bash_env("pi_proc_16gb", PIGEMMA_OUT=_p(tmp_path))
    script = lib('feasibility_env --quiet -- --io-mem none --calibration none\n'
                 'echo "MODEL_ID=$MODEL_ID VERDICT=$VERDICT EXIT=$VERDICT_EXIT"\n'
                 'echo "FLAGS=$LLAMA_FLAGS"\n'
                 'echo "SAFE=$(safe_llama_flags "$LLAMA_FLAGS")"')
    r = run_bash(script, env)
    assert r.returncode == 0, r.stderr
    assert "MODEL_ID=26b-q4 VERDICT=FITS_MAPPED EXIT=0" in r.stdout
    assert "FLAGS=-nr -lm mmap+mlock" in r.stdout
    assert "SAFE=-nr -lm mmap+mlock" in r.stdout  # unprotected host: mlock allowed


@needs_bash
def test_mlock_safety_on_protected_host(tmp_path: Path) -> None:
    env = bash_env("pi_proc_cf1_16gb", PIGEMMA_OUT=_p(tmp_path), PIGEMMA_I_KNOW="1")
    script = lib('if mlock_allowed; then echo MLOCK=yes; else echo MLOCK=no; fi\n'
                 'safe_llama_flags "-nr -lm mmap+mlock -t 4 -c 4096 --mlock -fa on"\n'
                 'ALLOW_MLOCK=1 safe_llama_flags "-lm mmap+mlock"')
    r = run_bash(script, env)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == "MLOCK=no"
    assert lines[1] == "-nr -lm mmap -t 4 -c 4096 -fa on"
    assert lines[2] == "-lm mmap+mlock"


# ------------------------------------------------------------------------------ resource helpers
@needs_bash
@pytest.mark.parametrize("fixture,expected", [("pi_proc_4gb", 2), ("pi_proc_16gb", 4), ("pi_proc_container", 4)])
def test_pick_jobs(fixture: str, expected: int, tmp_path: Path) -> None:
    r = run_bash(lib("pick_jobs"), bash_env(fixture, PIGEMMA_OUT=_p(tmp_path)))
    assert r.returncode == 0 and r.stdout.strip() == str(expected), r.stderr


@needs_bash
def test_pick_jobs_build_jobs_override(tmp_path: Path) -> None:
    env = bash_env("pi_proc_4gb", PIGEMMA_OUT=_p(tmp_path), BUILD_JOBS="3")
    r = run_bash(lib('pick_jobs; pick_jobs --auto'), env)
    assert r.stdout.split() == ["3", "2"]
    assert "exceeds the RAM-safe value" in r.stderr


@needs_bash
def test_fmt_gb_and_gib(tmp_path: Path) -> None:
    r = run_bash(lib("fmt_gb 14439363584; fmt_gb 802897920 3; fmt_gb abc; fmt_gib 1073741824"),
                 bash_env(None, PIGEMMA_OUT=_p(tmp_path)))
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["14.44 GB", "0.803 GB", "n/a", "1.00 GiB"]


@needs_bash
def test_require_free_disk_keeps_root_reserve(tmp_path: Path) -> None:
    env = bash_env("pi_proc_cf1_16gb", PIGEMMA_OUT=_p(tmp_path))
    # 26B UD-Q3_K_M (12.73 GB) leaves 2.77 GB on the 15.50 GB-free root fs -> allowed
    ok = run_bash(lib("require_free_disk /home/pi/models 12730000000 UD-Q3_K_M 1.1; echo FITS"), env)
    assert ok.returncode == 0 and "FITS" in ok.stdout, ok.stderr
    # 26B QAT Q4_0 (14.44 GB) would leave < 2 GB on / -> refused with exit 3
    bad = run_bash(lib("require_free_disk /home/pi/models 14439363584 26b-q4 1.1"), env)
    assert bad.returncode == 3
    assert "not enough free disk" in bad.stderr
    # not on the root fs (NVMe mount): only the factor applies
    env16 = bash_env("pi_proc_16gb", PIGEMMA_OUT=_p(tmp_path))
    nv = run_bash(lib("require_free_disk /mnt/nvme/models 454000000000 big 1.0; echo FITS"), env16)
    assert nv.returncode == 0 and "FITS" in nv.stdout, nv.stderr


@needs_bash
def test_require_mem_available_and_not_busy(tmp_path: Path) -> None:
    env = bash_env("pi_proc_4gb", PIGEMMA_OUT=_p(tmp_path))
    r = run_bash(lib("require_mem_available 4000000000 build"), env)
    assert r.returncode == 3 and "not enough available RAM" in r.stderr
    r = run_bash(lib("require_mem_available 1500000000 build; assert_not_busy; echo OK"), env)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr
    busy = bash_env("pi_proc_4gb", PIGEMMA_OUT=_p(tmp_path), PIGEMMA_MAX_LOADAVG="0.1")
    r = run_bash(lib("assert_not_busy"), busy)
    assert r.returncode == 4 and "busy host" in r.stderr


@needs_bash
def test_die_and_need_cmd(tmp_path: Path) -> None:
    env = bash_env(None, PIGEMMA_OUT=_p(tmp_path))
    r = run_bash(lib('die 3 "no disk"'), env)
    assert r.returncode == 3 and "STOP (exit 3" in r.stderr and "no disk" in r.stderr
    r = run_bash(lib('die "plain message"'), env)
    assert r.returncode == 1 and "plain message" in r.stderr
    r = run_bash(lib('die 4 "nope"'), env)
    assert r.returncode == 4 and "REFUSED (exit 4)" in r.stderr
    r = run_bash(lib("need_cmd pigemma-definitely-missing-tool fio 'disk benchmark'"), env)
    assert r.returncode == 3
    assert "apt-get install -y fio" in r.stderr and "disk benchmark" in r.stderr
    r = run_bash(lib("need_cmd bash; echo HAVE"), env)
    assert r.returncode == 0 and "HAVE" in r.stdout


# --------------------------------------------------------------------------------------- config
def test_pi_config_defaults_match_spec() -> None:
    text = (REPO / "pi" / "pi_config.env").read_text(encoding="utf-8")
    from pigemma import paths

    def default(name: str) -> str:
        m = re.search(rf'^{name}="\$\{{{name}:?-(.*)\}}"$', text, re.M)
        assert m, f"{name} not assigned with an environment-overridable default"
        return m.group(1)

    assert default("LLAMA_COMMIT") == paths.LLAMA_COMMIT == "a7b94df"
    assert default("THREADS") == "4" and default("CTX") == "4096"
    assert default("MIN_FREE_DISK_FACTOR") == "1.1"
    assert default("QUANT_OVERRIDE") == "" and default("OS_RESERVE_GIB") == "" and default("HF_LIMIT_RATE") == ""
    assert default("PROTECTED_SERVICE_GLOBS") == "contextfriend* context-friend*"
    assert default("DENY_HOSTS") == ""  # FACTS.md 2026-10-05: context-friend-1 is the real target now
    assert default("TUNE_APPLY") == "0" and default("TUNE_NVME_GEN3") == "0"
    assert default("TUNE_GOVERNOR") == "performance" and default("TUNE_SWAPPINESS") == "10"
    assert default("TUNE_READAHEAD_KB") == "2048" and default("TUNE_MEMLOCK") == "1"
    assert default("BUILD_JOBS") == "auto" and default("PIN_BUDGET_BYTES") == "auto"
    assert default("PIGEMMA_I_KNOW") == "0"
    assert "$PIGEMMA_HOME/models" in default("MODEL_DIR")
    assert "llama.cpp" in default("LLAMA_DIR")


@needs_bash
def test_config_precedence_and_print(tmp_path: Path) -> None:
    env = bash_env(None, PIGEMMA_OUT=_p(tmp_path / "out"), THREADS="3", MODEL_DIR="/mnt/nvme/models")
    r = run_bash(lib('pigemma_init\necho "T=$THREADS C=$CTX M=$MODEL_DIR D=[$DENY_HOSTS]"\n'
                     'bash -c \'echo "child THREADS=$THREADS CTX=$CTX"\''), env)
    assert r.returncode == 0, r.stderr
    assert "T=3 C=4096 M=/mnt/nvme/models D=[]" in r.stdout
    assert "child THREADS=3 CTX=4096" in r.stdout  # exported for python3 -m pigemma.feasibility
    assert re.search(r"THREADS\s+= '3'", r.stderr)
    assert "PIGEMMA_CATALOG" in r.stderr
    assert (tmp_path / "out").is_dir()


@needs_bash
@pytest.mark.parametrize("var,value", [("THREADS", "abc"), ("BUILD_JOBS", "0"), ("ALLOW_MLOCK", "maybe"),
                                       ("MIN_FREE_DISK_FACTOR", "0.5"), ("LLAMA_COMMIT", "xyz")])
def test_config_validation(var: str, value: str, tmp_path: Path) -> None:
    env = bash_env(None, PIGEMMA_OUT=_p(tmp_path), **{var: value})
    r = run_bash(lib("pigemma_init; echo NOT_REACHED"), env)
    assert r.returncode == 1
    assert var in r.stderr and "NOT_REACHED" not in r.stdout


@needs_bash
def test_alternate_config_file(tmp_path: Path) -> None:
    cfg = tmp_path / "alt.env"
    cfg.write_text('THREADS="${THREADS:-2}"\nCTX="${CTX:-1024}"\n', encoding="utf-8", newline="\n")
    env = bash_env(None, PIGEMMA_OUT=_p(tmp_path), PIGEMMA_CONFIG=_p(cfg))
    r = run_bash(lib('echo "T=$THREADS C=$CTX G=$TUNE_GOVERNOR"'), env)
    assert r.returncode == 0, r.stderr
    assert "T=2 C=1024 G=performance" in r.stdout  # missing keys fall back to built-in defaults


# ------------------------------------------------------------------------------------------ misc
@needs_bash
def test_write_facts_atomic_and_cli(tmp_path: Path) -> None:
    env = bash_env("pi_proc_16gb", PIGEMMA_OUT=_p(tmp_path))
    r = run_bash(f"bash {COMMON} write-facts", env)
    assert r.returncode == 0, r.stderr
    d = json.loads((tmp_path / "facts.json").read_text(encoding="utf-8"))
    assert d["hostname"] == "pi5-lab-16g"
    assert not list(tmp_path.glob("facts.json.tmp.*"))
    r = run_bash(f"bash {COMMON} storage /mnt/nvme/models", env)
    assert r.returncode == 0 and "STORAGE_TYPE=nvme" in r.stdout and "STORAGE_DISK=nvme0n1" in r.stdout
    r = run_bash(f"bash {COMMON} safe-host", env)
    assert r.returncode == 0
    r = run_bash(f"bash {COMMON} --help", env)
    assert r.returncode == 0 and "Exit codes" in r.stdout
    r = run_bash(f"bash {COMMON} no-such-command", env)
    assert r.returncode == 1
    r = run_bash(f"bash {COMMON} safe-host", bash_env("pi_proc_4gb", PIGEMMA_OUT=_p(tmp_path)))
    assert r.returncode == 4


@needs_bash
def test_fake_root_is_never_written(tmp_path: Path) -> None:
    root = FIX / "pi_proc_4gb"
    before = sorted((p.relative_to(root).as_posix(), p.stat().st_size) for p in root.rglob("*") if p.is_file())
    env = bash_env("pi_proc_4gb", PIGEMMA_OUT=_p(tmp_path), PIGEMMA_I_KNOW="1")
    run_bash(lib("collect_facts >/dev/null; assert_safe_host; pick_jobs >/dev/null"), env)
    after = sorted((p.relative_to(root).as_posix(), p.stat().st_size) for p in root.rglob("*") if p.is_file())
    assert before == after


@needs_bash
def test_missing_fake_root_is_an_error(tmp_path: Path) -> None:
    env = bash_env(None, PIGEMMA_OUT=_p(tmp_path), PIGEMMA_FAKE_PROC=_p(tmp_path / "nope"))
    r = run_bash(lib("echo NOT_REACHED"), env)
    assert r.returncode == 1 and "not a directory" in r.stderr
