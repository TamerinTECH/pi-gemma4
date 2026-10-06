"""Tests for pi/01_tune_os.sh (print-first OS tuning of the Pi: PCIe Gen3, governor, sysctl, read-ahead, memlock).

Manifest test field (files[] 'pi/01_tune_os.sh'):
  * `bash -n`;
  * the dry run inside Docker prints the plan without touching files -> integration test at the bottom (plus
    hermetic dry runs on every fake root here, checked with a digest of the fixture and the output tree);
  * a fixture test that the config.txt line is not duplicated on a second --apply (grep guard), on a temp copy
    of config.txt passed through PIGEMMA_BOOT_CONFIG.
Plus: the protected-host refusal (exit 4) and the PIGEMMA_I_KNOW rules (runtime-only, no boot edit without
--allow-boot-edit, no memlock change), the AI HAT lane rule (HAT EEPROM and lspci), config.txt section
awareness, the apply -> revert round trip with recorded live values, foreign files left alone, --apply refusals
(fake root without a test sysroot, non-Linux / container), item selection and usage errors.

Every run is hermetic: PIGEMMA_* and pi_config variables are stripped from the inherited environment,
PIGEMMA_OUT points into tmp_path, and --apply only ever runs with PIGEMMA_TUNE_ROOT=<tmp> (the script's test
sysroot: every write lands below it and systemctl / udevadm are never executed). bash is Git Bash on Windows
(conftest `bash` fixture, never WSL's).
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures"
SCRIPT = "pi/01_tune_os.sh"

AI_HAT_MSG = "PCIe lane used by the AI HAT: NVMe impossible without a dual-slot HAT"
BEGIN = "# >>> pi-gemma4 tune_os"
END = "# <<< pi-gemma4 tune_os"
MARK = "Managed by pi-gemma4 pi/01_tune_os.sh"
F_SYSCTL = "etc/sysctl.d/90-pigemma.conf"
F_UNIT = "etc/systemd/system/pigemma-governor.service"
F_UDEV = "etc/udev/rules.d/90-pigemma-readahead.rules"
F_LIMITS = "etc/security/limits.d/90-pigemma.conf"
F_SDCONF = "etc/systemd/system.conf.d/pigemma.conf"
OUR_FILES = (F_SYSCTL, F_UNIT, F_UDEV, F_LIMITS, F_SDCONF)

#: pi/pi_config.env variables (stripped from the inherited environment so the tests are hermetic).
CONFIG_VARS = (
    "PIGEMMA_HOME", "MODEL_DIR", "LLAMA_DIR", "LLAMA_COMMIT", "PIGEMMA_OUT", "THREADS", "CTX", "QUANT_OVERRIDE",
    "OS_RESERVE_GIB", "MIN_FREE_DISK_FACTOR", "KEEP_FREE_GB", "NICE_LEVEL", "DENY_HOSTS",
    "PROTECTED_SERVICE_GLOBS", "ALLOW_MLOCK", "TUNE_APPLY", "TUNE_NVME_GEN3", "TUNE_GOVERNOR", "TUNE_SWAPPINESS",
    "TUNE_READAHEAD_KB", "TUNE_MEMLOCK", "BUILD_JOBS", "HF_LIMIT_RATE", "PIN_BUDGET_BYTES",
    "TUNE_VFS_CACHE_PRESSURE",
)

#: An active PCIe Gen3 setting (comments excluded), as the Pi firmware would read it.
GEN3_RE = re.compile(r"^[ \t]*dtparam[ \t]*=[^#\n]*\bpciex1_gen[ \t]*=[ \t]*3\b", re.M)


# ------------------------------------------------------------------------------------------- helpers
def _p(path: Path | str) -> str:
    """A path as bash (Git Bash or Linux) accepts it."""
    return Path(path).as_posix()


def _env(tmp_path: Path, fake: str | Path | None = "pi_proc_16gb", **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("PIGEMMA_") and k not in CONFIG_VARS and k != "PY"}
    env["PIGEMMA_PYTHON"] = _p(sys.executable)
    env["PIGEMMA_OUT"] = _p(tmp_path / "out")
    env["MODEL_DIR"] = "/mnt/nvme/models"  # the 16 GB fixture's NVMe mount (findmnt.txt / df.txt)
    env["LC_ALL"] = env.get("LC_ALL") or "C.UTF-8"
    if fake is not None:
        env["PIGEMMA_FAKE_PROC"] = _p(fake if isinstance(fake, Path) else FIX / fake)
    env.update(extra)
    return env


def run01(bash: str, env: dict[str, str], *args: str, timeout: float = 180) -> subprocess.CompletedProcess:
    return subprocess.run([bash, SCRIPT, *args], cwd=REPO, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def _tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    if root.exists():
        for f in sorted(p for p in root.rglob("*") if p.is_file()):
            h.update(f.relative_to(root).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def _files(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def _item(stdout: str, letter: str) -> str:
    """The block of one item: from '(x) ' up to the next item, the summary or the apply section."""
    m = re.search(rf"^\({letter}\) .*?(?=^\([a-f]\) |^summary |^== applying|\Z)", stdout, re.M | re.S)
    assert m, f"item ({letter}) not in output:\n{stdout}"
    return m.group(0)


def _status(stdout: str, letter: str) -> str:
    blk = _item(stdout, letter)
    m = re.search(r"^    status  : (.*)$", blk, re.M)
    assert m, blk
    return m.group(1)


def _applied(stdout: str) -> str:
    assert "== applying" in stdout, stdout
    return stdout.split("== applying", 1)[1]


def _clone(src: str, dst: Path, **files: str) -> Path:
    """Copy a fake root and overwrite / add files ('a__b' = 'a/b'; '' content deletes)."""
    shutil.copytree(FIX / src, dst)
    for rel, content in files.items():
        f = dst / rel.replace("__", "/")
        if content == "":
            f.unlink(missing_ok=True)
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8", newline="\n")
    return dst


def _boot_copy(tmp_path: Path, src: str = "pi_proc_cf1_16gb", text: str | None = None) -> Path:
    """A temp config.txt (default: the stock cf1 file, which has no pciex1_gen line)."""
    cfg = tmp_path / "config.txt"
    if text is None:
        shutil.copyfile(FIX / src / "boot" / "firmware" / "config.txt", cfg)
    else:
        cfg.write_text(text, encoding="utf-8", newline="\n")
    return cfg


def _tunable_fake(tmp_path: Path) -> Path:
    """The 16 GB NVMe fixture with untuned live values (ondemand x2, swappiness 60, vfs 100, read-ahead 128)."""
    return _clone("pi_proc_16gb", tmp_path / "fake", **{
        "sys__devices__system__cpu__cpufreq__policy0__scaling_governor": "ondemand\n",
        "sys__devices__system__cpu__cpufreq__policy0__scaling_available_governors": "performance ondemand powersave\n",
        "sys__devices__system__cpu__cpufreq__policy1__scaling_governor": "ondemand\n",
        "sys__vm__swappiness": "60\n",  # = /proc/sys/vm/swappiness in the fake root
        "sys__vm__vfs_cache_pressure": "100\n",
        "sys__block__nvme0n1__queue__read_ahead_kb": "128\n",
    })


# ------------------------------------------------------------------------------------------ lint
def test_bash_syntax(bash: str) -> None:
    r = subprocess.run([bash, "-n", SCRIPT], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def test_script_shape() -> None:
    raw = (REPO / SCRIPT).read_bytes()
    assert b"\r\n" not in raw, "pi scripts must use LF line endings"
    text = raw.decode("utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    assert '. "$PI_DIR/lib/common.sh"' in text
    # code only (comments dropped, quoted strings blanked: messages may mention these words)
    code = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        s = re.sub(r'"[^"]*"', '""', s)
        s = re.sub(r"'[^']*'", "''", s)
        code.append(s)
    joined = "\n".join(code)
    assert re.search(r"^assert_safe_host$", joined, re.M), "assert_safe_host must run at the top level"
    assert joined.index("\nassert_safe_host\n") < joined.index("\nplan_a\n"), "host gate before any plan"
    # forbidden commands in command position (values such as I_FLAG[a]=reboot and messages are fine)
    cmd_pos = r"(?:^|[;&|(]|\bthen\b|\bdo\b|\bsudo(?:\s+-n)?)\s*"
    for pat in (r"(reboot|shutdown|poweroff|halt)\b", r"(pkill|killall)\b", r"kill\s+-",
                r"apt(-get)?\s+(install|remove|purge)\b", r"sysctl\s+--system\b",
                r"systemctl\s+(stop|restart|start|kill|daemon-reexec)\b", r"udevadm\s+trigger\b"):
        assert not re.search(cmd_pos + pat, joined, re.M), pat
    # the only systemctl verbs that appear anywhere in the script (plan commands built from strings, the
    # is-enabled query, and messages that tell the user what pi/01 does NOT run)
    allowed = {"daemon-reload", "enable", "disable", "is-enabled", "daemon-reexec", "may"}
    for verb in re.findall(r"systemctl ([a-z-]+)", text):
        assert verb in allowed, verb
    plan_cmds = set(re.findall(r'_t_act [a-f] cmd "" "([a-z]+ [a-z-]+)', text))
    assert plan_cmds == {"systemctl disable", "systemctl daemon-reload", "systemctl enable",
                         "udevadm control"}, plan_cmds


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck() -> None:  # pragma: no cover - shellcheck is not installed on the desktop
    r = subprocess.run(["shellcheck", "-x", SCRIPT], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


# ------------------------------------------------------------------------------------------ usage
def test_help_and_version(bash: str, tmp_path: Path) -> None:
    env = _env(tmp_path)
    r = run01(bash, env, "--help")
    assert r.returncode == 0, r.stderr
    for word in ("--apply", "--no-pcie", "--revert", "--only", "REBOOT REQUIRED", "dtparam=pciex1_gen=3"):
        assert word in r.stdout, word
    r = run01(bash, env, "--version")
    assert r.returncode == 0
    assert re.match(r"^pi/01_tune_os\.sh \d+\.\d+\.\d+$", r.stdout.strip())
    assert not (tmp_path / "out").exists(), "--help / --version must not create PIGEMMA_OUT"


@pytest.mark.parametrize("args", [("--bogus",), ("--only",), ("--only", "x"), ("--skip", "a,z"), ("--only", ",")])
def test_usage_errors_exit_1(bash: str, tmp_path: Path, args: tuple[str, ...]) -> None:
    r = run01(bash, _env(tmp_path), *args)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "ERROR (exit 1)" in r.stderr
    assert "status  :" not in r.stdout, "no plan after a usage error"


def test_bad_vfs_cache_pressure_exits_1(bash: str, tmp_path: Path) -> None:
    r = run01(bash, _env(tmp_path, TUNE_VFS_CACHE_PRESSURE="fifty"))
    assert r.returncode == 1
    assert "TUNE_VFS_CACHE_PRESSURE" in r.stderr


# --------------------------------------------------------------------------------------- dry runs
def test_dry_run_16gb_prints_plan_and_touches_nothing(bash: str, tmp_path: Path) -> None:
    fix = FIX / "pi_proc_16gb"
    before = _tree_digest(fix)
    r = run01(bash, _env(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert out.startswith("pi-gemma4 OS tuning (pi/01_tune_os.sh ")
    assert "DRY RUN" in out.splitlines()[0]
    for x in "abcdef":
        blk = _item(out, x)
        assert "    current : " in blk and "    proposed: " in blk and "    status  : " in blk, x
    # (a) the fixture's config.txt already has pciex1_gen=3 under [all]
    assert _status(out, "a").startswith("OK - already Gen3")
    # the exact commands are shown and none is executed
    assert "commands (dry run - not executed):" in out
    assert "tee /etc/sysctl.d/90-pigemma.conf" in out
    assert "vm.swappiness = 10" in out and "vm.vfs_cache_pressure = 50" in out
    assert 'KERNEL=="nvme0n1", ATTR{queue/read_ahead_kb}="2048"' in out
    assert "*    soft    memlock    unlimited" in out and "*    hard    memlock    unlimited" in out
    assert "DefaultLimitMEMLOCK=infinity" in out
    assert "systemctl enable pigemma-governor.service" in out
    assert "Active Cooler" in out and "80 C" in out
    assert _status(out, "f") == "ADVICE"
    assert re.search(r"^summary : \d+ to change, \d+ already OK", out, re.M)
    assert "DRY RUN - nothing was changed" in out
    assert _tree_digest(fix) == before
    assert _files(tmp_path / "out") == [], "a dry run writes no state file"


def test_dry_run_with_gen3_requested_leaves_config_alone(bash: str, tmp_path: Path) -> None:
    cfg = _boot_copy(tmp_path)
    original = cfg.read_bytes()
    r = run01(bash, _env(tmp_path, PIGEMMA_BOOT_CONFIG=_p(cfg), TUNE_NVME_GEN3="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    blk = _item(r.stdout, "a")
    assert _status(r.stdout, "a") == "CHANGE"
    assert "tee -a" in blk and "dtparam=pciex1_gen=3" in blk and "[all]" in blk
    assert "REBOOT REQUIRED" in blk
    assert "REBOOT REQUIRED after --apply for item (a)" in r.stdout
    assert cfg.read_bytes() == original
    assert not list(tmp_path.glob("*.bak"))


def test_gen3_needs_the_tune_flag(bash: str, tmp_path: Path) -> None:
    cfg = _boot_copy(tmp_path)
    original = cfg.read_bytes()
    env = _env(tmp_path, PIGEMMA_BOOT_CONFIG=_p(cfg), PIGEMMA_TUNE_ROOT=_p(tmp_path / "root"), TUNE_NVME_GEN3="0")
    r = run01(bash, env, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _status(r.stdout, "a").startswith("NOT SELECTED - TUNE_NVME_GEN3=0")
    assert "commands (NOT executed: not selected)" in _item(r.stdout, "a")
    assert cfg.read_bytes() == original


def test_no_pcie_flag_skips_item_a(bash: str, tmp_path: Path) -> None:
    cfg = _boot_copy(tmp_path)
    r = run01(bash, _env(tmp_path, PIGEMMA_BOOT_CONFIG=_p(cfg), TUNE_NVME_GEN3="1"), "--no-pcie")
    assert r.returncode == 0, r.stderr
    assert _status(r.stdout, "a") == "SKIP - skipped (--no-pcie)"
    assert "items selected: b c d e f" in r.stdout


# ------------------------------------------------------------------- (a) grep guard (manifest test)
def test_second_apply_does_not_duplicate_the_config_line(bash: str, tmp_path: Path) -> None:
    cfg = _boot_copy(tmp_path)
    original = cfg.read_text(encoding="utf-8")
    assert not GEN3_RE.search(original)
    fix_before = _tree_digest(FIX / "pi_proc_16gb")
    env = _env(tmp_path, PIGEMMA_BOOT_CONFIG=_p(cfg), PIGEMMA_TUNE_ROOT=_p(tmp_path / "root"), TUNE_NVME_GEN3="1")

    r1 = run01(bash, env, "--apply")
    assert r1.returncode == 0, r1.stdout + r1.stderr
    text1 = cfg.read_text(encoding="utf-8")
    assert len(GEN3_RE.findall(text1)) == 1
    assert text1.count(BEGIN) == 1 and text1.count(END) == 1
    assert text1.startswith(original), "the line is APPENDED; nothing above it changes"
    block = text1[len(original):]
    assert re.search(r"^\[all\]\ndtparam=pciex1_gen=3\n", block, re.M), block
    assert "REBOOT REQUIRED" in r1.stdout and "never reboots" in r1.stdout
    baks = sorted(tmp_path.glob("config.txt.pigemma-*.bak"))
    assert len(baks) == 1 and baks[0].read_text(encoding="utf-8") == original

    r2 = run01(bash, env, "--apply")
    assert r2.returncode == 0, r2.stdout + r2.stderr
    text2 = cfg.read_text(encoding="utf-8")
    assert text2 == text1, "the second --apply must not touch config.txt"
    assert len(GEN3_RE.findall(text2)) == 1
    st = _status(r2.stdout, "a")
    assert st.startswith("OK - already added by pi/01_tune_os.sh") and "grep guard" in st
    assert "nothing to do" in r2.stdout
    assert "REBOOT REQUIRED" not in _applied(r2.stdout)
    assert len(sorted(tmp_path.glob("config.txt.pigemma-*.bak"))) == 1
    assert _tree_digest(FIX / "pi_proc_16gb") == fix_before


@pytest.mark.parametrize("text, status", [
    # a Gen3 line under a [cm4] filter does not apply to a Pi 5 -> still proposed
    ("arm_64bit=1\n[cm4]\ndtparam=pciex1_gen=3\n[all]\n", "CHANGE"),
    # commented out -> still proposed
    ("arm_64bit=1\n#dtparam=pciex1_gen=3\n", "CHANGE"),
    # already set by hand under [all] (combined dtparam list) -> OK, nothing appended
    ("arm_64bit=1\n[all]\ndtparam=pciex1,pciex1_gen=3\n", "OK - already Gen3"),
    # set under [pi5] -> OK
    ("[pi5]\ndtparam=pciex1_gen=3\n", "OK - already Gen3"),
])
def test_boot_config_sections(bash: str, tmp_path: Path, text: str, status: str) -> None:
    cfg = _boot_copy(tmp_path, text=text)
    r = run01(bash, _env(tmp_path, PIGEMMA_BOOT_CONFIG=_p(cfg), TUNE_NVME_GEN3="1"), "--only", "a")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _status(r.stdout, "a").startswith(status), _item(r.stdout, "a")


def test_file_without_trailing_newline_gets_the_block_on_its_own_lines(bash: str, tmp_path: Path) -> None:
    cfg = _boot_copy(tmp_path, text="arm_64bit=1\n[all]\ndtparam=audio=on")
    env = _env(tmp_path, PIGEMMA_BOOT_CONFIG=_p(cfg), PIGEMMA_TUNE_ROOT=_p(tmp_path / "root"), TUNE_NVME_GEN3="1")
    r = run01(bash, env, "--apply", "--only", "a")
    assert r.returncode == 0, r.stdout + r.stderr
    lines = cfg.read_text(encoding="utf-8").splitlines()
    assert "dtparam=audio=on" in lines
    assert any(ln.startswith(BEGIN) for ln in lines)
    assert "dtparam=pciex1_gen=3" in lines


# ---------------------------------------------------------------------------------- (a) AI HAT rule
def test_ai_hat_eeprom_blocks_gen3(bash: str, tmp_path: Path) -> None:
    # context-friend-1 as first probed (4 GB fixture, 'AI HAT+' EEPROM), with the authorised override
    r = run01(bash, _env(tmp_path, "pi_proc_4gb", PIGEMMA_I_KNOW="1", TUNE_NVME_GEN3="1",
                         MODEL_DIR="/srv/pigemma/models"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert _status(r.stdout, "a") == f"SKIP - {AI_HAT_MSG}"
    assert "HAT EEPROM product 'AI HAT+'" in _item(r.stdout, "a")


def test_hailo_on_lspci_blocks_gen3_even_with_nvme(bash: str, tmp_path: Path) -> None:
    fake = _clone("pi_proc_16gb", tmp_path / "fake", **{
        "lspci.txt": "0001:01:00.0 Co-processor [0b40]: Hailo Technologies Ltd. Hailo-8 AI Processor "
                     "[1e60:2864] (rev 01)\n",
    })
    cfg = _boot_copy(tmp_path)
    original = cfg.read_bytes()
    env = _env(tmp_path, fake, PIGEMMA_BOOT_CONFIG=_p(cfg), PIGEMMA_TUNE_ROOT=_p(tmp_path / "root"),
               TUNE_NVME_GEN3="1")
    r = run01(bash, env, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    st = _status(r.stdout, "a")
    assert st.startswith(f"SKIP - {AI_HAT_MSG}"), st
    assert "lspci: " in _item(r.stdout, "a")
    assert cfg.read_bytes() == original


def test_no_nvme_and_not_a_pi5(bash: str, tmp_path: Path) -> None:
    cfg = _boot_copy(tmp_path)
    fake = _clone("pi_proc_16gb", tmp_path / "fake")
    shutil.rmtree(fake / "sys" / "class" / "nvme")
    r = run01(bash, _env(tmp_path, fake, PIGEMMA_BOOT_CONFIG=_p(cfg), TUNE_NVME_GEN3="1"), "--only", "a")
    assert r.returncode == 0, r.stderr
    assert _status(r.stdout, "a").startswith("SKIP - no NVMe present")
    fake4 = _clone("pi_proc_16gb", tmp_path / "fake4", **{"device-tree__model": "Raspberry Pi 4 Model B Rev 1.5\0"})
    r = run01(bash, _env(tmp_path, fake4, PIGEMMA_BOOT_CONFIG=_p(cfg), TUNE_NVME_GEN3="1"), "--only", "a")
    assert r.returncode == 0, r.stderr
    assert "is not a Raspberry Pi 5" in _status(r.stdout, "a")


# ---------------------------------------------------------------------------------- protected host
@pytest.mark.parametrize("fixture", ["pi_proc_4gb", "pi_proc_cf1_16gb"])
def test_protected_host_refused_before_any_plan(bash: str, tmp_path: Path, fixture: str) -> None:
    r = run01(bash, _env(tmp_path, fixture, MODEL_DIR="/srv/pigemma/models"), "--apply")
    assert r.returncode == 4, r.stdout + r.stderr
    assert "REFUSED (exit 4)" in r.stderr
    assert r.stdout.strip() == "", "no plan on a refused host"
    assert _files(tmp_path / "out") == []


def test_protected_override_is_runtime_only(bash: str, tmp_path: Path) -> None:
    root = tmp_path / "root"
    env = _env(tmp_path, "pi_proc_cf1_16gb", PIGEMMA_I_KNOW="1", MODEL_DIR="/srv/pigemma/models",
               PIGEMMA_TUNE_ROOT=_p(root))
    r = run01(bash, env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "persistence files no (runtime-only)" in r.stdout
    assert "PROTECTED HOST OVERRIDE" in r.stdout
    assert "tee /etc/" not in r.stdout, "no persistence file on a protected host"
    assert _status(r.stdout, "a").startswith("SKIP - no NVMe present")
    assert _status(r.stdout, "e").startswith("SKIP - mlock of model weights is not allowed")
    assert "scaling_governor" in _item(r.stdout, "b")
    assert "/sys/block/mmcblk0/queue/read_ahead_kb" in _item(r.stdout, "d")
    # --apply in the test sysroot: live values only, no /etc file
    r = run01(bash, env, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not any(f.startswith("etc/") for f in _files(root)), _files(root)
    assert (root / "sys/devices/system/cpu/cpufreq/policy0/scaling_governor").read_text().strip() == "performance"
    assert (root / "sys/block/mmcblk0/queue/read_ahead_kb").read_text().strip() == "2048"
    state = (tmp_path / "out" / "tune_os.state").read_text(encoding="utf-8")
    assert "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor\tondemand" in state
    assert "/sys/block/mmcblk0/queue/read_ahead_kb\t128" in state


def test_protected_override_boot_edit_needs_explicit_approval(bash: str, tmp_path: Path) -> None:
    # a 16 GB NVMe board that runs a protected service
    svc = (FIX / "pi_proc_16gb" / "systemctl.txt").read_text(encoding="utf-8")
    svc += "contextfriend-box.service         loaded active running ContextFriend Box\n"
    fake = _clone("pi_proc_16gb", tmp_path / "fake", **{"systemctl.txt": svc})
    cfg = _boot_copy(tmp_path)
    original = cfg.read_bytes()
    base = {"PIGEMMA_BOOT_CONFIG": _p(cfg), "PIGEMMA_TUNE_ROOT": _p(tmp_path / "root"), "TUNE_NVME_GEN3": "1"}
    r = run01(bash, _env(tmp_path, fake, **base), "--apply")
    assert r.returncode == 4, r.stdout + r.stderr
    r = run01(bash, _env(tmp_path, fake, PIGEMMA_I_KNOW="1", **base), "--apply", "--only", "a")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--allow-boot-edit" in _status(r.stdout, "a")
    assert cfg.read_bytes() == original
    r = run01(bash, _env(tmp_path, fake, PIGEMMA_I_KNOW="1", **base), "--apply", "--only", "a", "--allow-boot-edit")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(GEN3_RE.findall(cfg.read_text(encoding="utf-8"))) == 1


# ---------------------------------------------------------------------------- apply / revert round trip
def test_apply_then_revert_round_trip(bash: str, tmp_path: Path) -> None:
    fake = _tunable_fake(tmp_path)
    fake_before = _tree_digest(fake)
    root = tmp_path / "root"
    cfg = _boot_copy(tmp_path)
    original = cfg.read_bytes()
    env = _env(tmp_path, fake, PIGEMMA_BOOT_CONFIG=_p(cfg), PIGEMMA_TUNE_ROOT=_p(root), TUNE_NVME_GEN3="1")

    r = run01(bash, env, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "summary : 5 applied, 0 already OK" in r.stdout
    assert "REBOOT REQUIRED" in r.stdout and "RE-LOGIN REQUIRED" in r.stdout
    applied = _applied(r.stdout)
    for x in "abcde":
        blk = _item(applied, x)
        assert "    before  : " in blk and "    after   : " in blk and "status  : APPLIED" in blk, x
    assert "(test sysroot: not executed)" in r.stdout  # systemctl / udevadm
    for f in OUR_FILES:
        assert MARK in (root / f).read_text(encoding="utf-8"), f
    assert "vm.swappiness = 10" in (root / F_SYSCTL).read_text(encoding="utf-8")
    assert 'KERNEL=="nvme0n1"' in (root / F_UDEV).read_text(encoding="utf-8")
    assert "ExecStart=/bin/sh -c 'for f in" in (root / F_UNIT).read_text(encoding="utf-8")
    assert (root / "proc/sys/vm/swappiness").read_text().strip() == "10"
    assert (root / "proc/sys/vm/vfs_cache_pressure").read_text().strip() == "50"
    for pol in ("policy0", "policy1"):
        assert (root / f"sys/devices/system/cpu/cpufreq/{pol}/scaling_governor").read_text().strip() == "performance"
    assert (root / "sys/block/nvme0n1/queue/read_ahead_kb").read_text().strip() == "2048"
    state = (tmp_path / "out" / "tune_os.state").read_text(encoding="utf-8")
    for line in ("/proc/sys/vm/swappiness\t60", "/proc/sys/vm/vfs_cache_pressure\t100",
                 "/sys/block/nvme0n1/queue/read_ahead_kb\t128",
                 "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor\tondemand",
                 "/sys/devices/system/cpu/cpufreq/policy1/scaling_governor\tondemand"):
        assert line in state, line
    assert _tree_digest(fake) == fake_before, "the fake root is read-only input"

    # idempotent: nothing left to do
    r = run01(bash, env, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "summary : 0 applied, 5 already OK" in r.stdout

    # the revert plan (dry run) changes nothing
    snap = (_tree_digest(root), cfg.read_bytes())
    r = run01(bash, env, "--revert")
    assert r.returncode == 0, r.stdout + r.stderr
    assert " - revert - DRY RUN" in r.stdout.splitlines()[0]
    assert "sed -i '/^# >>> pi-gemma4 tune_os/,/^# <<< pi-gemma4 tune_os/d'" in r.stdout
    assert (_tree_digest(root), cfg.read_bytes()) == snap

    # revert --apply: our files gone, config.txt byte-identical, live values restored, state file removed
    r = run01(bash, env, "--revert", "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    assert cfg.read_bytes() == original
    for f in OUR_FILES:
        assert not (root / f).exists(), f
    assert (root / "proc/sys/vm/swappiness").read_text().strip() == "60"
    assert (root / "proc/sys/vm/vfs_cache_pressure").read_text().strip() == "100"
    assert (root / "sys/devices/system/cpu/cpufreq/policy1/scaling_governor").read_text().strip() == "ondemand"
    assert (root / "sys/block/nvme0n1/queue/read_ahead_kb").read_text().strip() == "128"
    assert not (tmp_path / "out" / "tune_os.state").exists()


def test_foreign_files_are_left_alone(bash: str, tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "etc/sysctl.d").mkdir(parents=True)
    foreign = "# admin's own tuning\nvm.swappiness = 1\n"
    (root / F_SYSCTL).write_text(foreign, encoding="utf-8", newline="\n")
    env = _env(tmp_path, _tunable_fake(tmp_path), PIGEMMA_TUNE_ROOT=_p(root))
    r = run01(bash, env, "--apply", "--only", "c")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "NOT written by pi-gemma4" in _item(r.stdout, "c")
    assert (root / F_SYSCTL).read_text(encoding="utf-8") == foreign
    assert (root / "proc/sys/vm/swappiness").read_text().strip() == "10"  # the live value is still applied
    r = run01(bash, env, "--revert", "--apply", "--only", "c")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (root / F_SYSCTL).read_text(encoding="utf-8") == foreign, "--revert never deletes a foreign file"


def test_governor_choices(bash: str, tmp_path: Path) -> None:
    fake = _tunable_fake(tmp_path)
    r = run01(bash, _env(tmp_path, fake, TUNE_GOVERNOR="schedutil"), "--only", "b")
    assert r.returncode == 0, r.stderr
    assert "is not available here" in _status(r.stdout, "b")
    r = run01(bash, _env(tmp_path, fake, TUNE_GOVERNOR="keep"), "--only", "b")
    assert r.returncode == 0, r.stderr
    assert _status(r.stdout, "b").startswith("SKIP - TUNE_GOVERNOR=keep")


def test_memlock_flag_off(bash: str, tmp_path: Path) -> None:
    r = run01(bash, _env(tmp_path, TUNE_MEMLOCK="0"), "--only", "e")
    assert r.returncode == 0, r.stderr
    assert _status(r.stdout, "e").startswith("SKIP - TUNE_MEMLOCK=0")


def test_selection_and_tune_apply_env(bash: str, tmp_path: Path) -> None:
    root = tmp_path / "root"
    r = run01(bash, _env(tmp_path, PIGEMMA_TUNE_ROOT=_p(root)), "--only", "c,e", "--skip", "e")
    assert r.returncode == 0, r.stderr
    assert "items selected: c\n" in r.stdout
    for x in "abdef":
        st = _status(r.stdout, x)
        assert "not selected" in st or "skipped" in st, (x, st)
    # TUNE_APPLY=1 applies; --dry-run wins over it
    r = run01(bash, _env(tmp_path, PIGEMMA_TUNE_ROOT=_p(root), TUNE_APPLY="1"), "--only", "c", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout.splitlines()[0]
    assert not root.exists()
    r = run01(bash, _env(tmp_path, PIGEMMA_TUNE_ROOT=_p(root), TUNE_APPLY="1"), "--only", "c")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (root / F_SYSCTL).is_file()


# --------------------------------------------------------------------------------------- refusals
def test_apply_with_fake_root_needs_a_test_sysroot(bash: str, tmp_path: Path) -> None:
    r = run01(bash, _env(tmp_path), "--apply")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "PIGEMMA_TUNE_ROOT" in r.stderr
    assert r.stdout.strip() == ""


def test_container_fixture_dry_run_and_apply_refusal(bash: str, tmp_path: Path) -> None:
    r = run01(bash, _env(tmp_path, "pi_proc_container", MODEL_DIR="/work/models"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CONTAINER: --apply is refused here" in r.stdout
    assert _status(r.stdout, "a").startswith("SKIP - no boot config")
    assert _status(r.stdout, "b").startswith("SKIP - cpufreq is not exposed")
    assert _status(r.stdout, "d").startswith("SKIP - MODEL_DIR is not on a block device")
    assert "running services could not be listed" in r.stderr  # degrades with a logged warning
    # a real (non-fake) --apply in a container / on a non-Linux host is refused before anything changes
    env = _env(tmp_path, None, PIGEMMA_IN_CONTAINER="1", MODEL_DIR=_p(tmp_path / "models"))
    r = run01(bash, env, "--apply")
    if r.returncode == 4:  # pragma: no cover - only on a protected development host
        pytest.skip("this host is protected: " + r.stderr[-200:])
    assert r.returncode == 3, r.stdout + r.stderr
    assert ("inside a container" in r.stderr) or ("needs Linux" in r.stderr)
    assert r.stdout.strip() == ""


# ------------------------------------------------------------------------------------ integration
@pytest.mark.integration
def test_dry_run_inside_the_docker_build_image_touches_nothing(bash: str, docker: str, tmp_path: Path) -> None:
    probe = tmp_path / "probe.sh"
    probe.write_text(
        "set -u\n"
        "snap() { find /etc /proc/sys/vm -xdev -maxdepth 3 -type f 2>/dev/null | sort | md5sum; }\n"
        "before=$(snap)\n"
        "bash pi/01_tune_os.sh >/tmp/o.txt 2>/tmp/e.txt; rc=$?\n"
        "bash pi/01_tune_os.sh --apply >/dev/null 2>/tmp/e2.txt; rc2=$?\n"
        "after=$(snap)\n"
        "cat /tmp/o.txt\n"
        "echo \"RC=$rc RC2=$rc2\"\n"
        "tail -n 1 /tmp/e2.txt\n"
        "if [ \"$before\" = \"$after\" ]; then echo ETC_UNCHANGED; fi\n"
        "n=0; for f in /etc/sysctl.d/90-pigemma.conf /etc/security/limits.d/90-pigemma.conf; do\n"
        "  if [ -e \"$f\" ]; then n=$((n + 1)); fi; done; echo \"PIGEMMA_FILES=$n\"\n",
        encoding="utf-8", newline="\n")
    probe_in_container = "/tmp/pg01-probe.sh"
    pi_before = _tree_digest(REPO / "pi")
    r = subprocess.run(
        [bash, "pc/docker_run.sh", "--no-tty", "--no-stdin", "--env", "PIGEMMA_OUT=/tmp/pg01",
         "--volume", f"{_p(probe)}:{probe_in_container}:ro", "--", "bash", probe_in_container],
        cwd=REPO, env=_env(tmp_path, None), capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert "RC=0 RC2=3" in out, out
    assert "DRY RUN" in out and "CONTAINER: --apply is refused here" in out
    for x in "abcdef":
        assert f"\n({x}) " in out, x
    assert "inside a container" in out
    assert "ETC_UNCHANGED" in out
    assert "PIGEMMA_FILES=0" in out
    assert _tree_digest(REPO / "pi") == pi_before
