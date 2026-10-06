"""Serialize and monitor our own Pi job; never change services or host settings."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import resource
import subprocess
import time

SERVICES = ["contextfriend-box", "contextfriend-gpu-tunnel", "hailort", "whisper-bench@base"]


def sample(child_pid=None):
    mem = {line.split(":")[0]: int(line.split()[1])*1024
           for line in Path("/proc/meminfo").read_text().splitlines()}
    statuses = subprocess.run(["systemctl", "is-active", *SERVICES], capture_output=True, text=True).stdout.splitlines()
    throttle = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
    result = dict(time=time.time(), mem_available=mem["MemAvailable"],
                root_free=os.statvfs("/").f_bavail*os.statvfs("/").f_frsize,
                temperature_c=int(Path("/sys/class/thermal/thermal_zone0/temp").read_text())/1000,
                services=dict(zip(SERVICES, statuses)), throttled=throttle)
    if child_pid:
        # Descendants belong to the new process group; inspect only our job.
        owned = []
        for proc in Path("/proc").glob("[0-9]*"):
            try:
                fields = (proc/"stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == child_pid:
                    owned.append({"pid": int(proc.name), "status": (proc/"status").read_text(),
                                  "io": (proc/"io").read_text()})
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        result["owned_processes"] = owned
    return result


def reason(s, initial=False):
    if len(s["services"]) != 4 or any(v != "active" for v in s["services"].values()):
        return "protected service unhealthy"
    if s["root_free"] < 2_000_000_000:
        return "root disk below 2 GB"
    if s["mem_available"] < (3_000_000_000 if initial else 1_000_000_000):
        return "insufficient safe memory headroom"
    if s["temperature_c"] >= 78:
        return "temperature at abort threshold"
    if s["throttled"] != "throttled=0x0":
        return "throttling or undervoltage indication"
    return None


def main():
    # Applies only to this experiment runner and its descendants, not services.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        p.error("command required")
    args.out.mkdir(parents=True, exist_ok=False)
    with (Path.home()/"pigemma/.bench.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = sample()
        error = reason(initial, True)
        for proc in Path("/proc").glob("[0-9]*/comm"):
            try:
                name = proc.read_text().strip()
                if name.startswith("llama-") or name == "cf-run":
                    error = f"existing model process {proc.parent.name}: {name}"
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        info = dict(command=command, initial=initial, error=error, pid=None)
        (args.out/"job.json").write_text(json.dumps(info, indent=2))
        if error:
            raise SystemExit(error)
        start = time.monotonic()
        with (args.out/"stdout.txt").open("w") as out, (args.out/"stderr.txt").open("w") as err, (args.out/"health.jsonl").open("w") as health:
            child = subprocess.Popen(["nice", "-n", "10", *command], stdout=out, stderr=err, start_new_session=True)
            info["pid"] = child.pid
            while child.poll() is None:
                current = sample(child.pid)
                health.write(json.dumps(current)+"\n");health.flush()
                error = reason(current)
                if time.monotonic()-start > args.timeout:
                    error = "job timeout"
                if error:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(5)
            info.update(exit_code=child.wait(), error=error, elapsed_s=time.monotonic()-start, final=sample())
        (args.out/"job.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info))
        return info["exit_code"] if not error else 4


if __name__ == "__main__":
    raise SystemExit(main())
