#!/usr/bin/env bash
# pi/04_fetch_model.sh - fetch the Gemma 4 GGUF that the feasibility verdict recommends (or an explicit
# quant) onto the Raspberry Pi, plus optionally its speculative-decoding sidecar, with resume, byte-size and
# sha256 verification, inside the disk budget. Alternatively copies it from the desktop over the LAN (rsync).
#
# Usage (from anywhere; run long downloads detached on the Wi-Fi Pi: --bg, or nohup ... &):
#     pi/04_fetch_model.sh [--model 26b|31b|ID] [--quant auto|Q4_0|UD-Q3_K_M|...|ID] [--draft]
#                          [--dest $MODEL_DIR] [--from-desktop user@host:REPO_ROOT/models]
#                          [--dry-run] [more options: --help] [-- extra pigemma.feasibility args]
#
# SELECTION
#   * --quant auto (default, unless QUANT_OVERRIDE is set in pi/pi_config.env): fresh facts.json (MODEL_DIR =
#     the destination, so free disk and storage are measured there), then
#     `python3 -m pigemma.feasibility --facts pi/out/facts.json --json pi/out/feasibility.json --emit env
#     [--want <--model>]`; its MODEL_ID is the model to fetch. Verdict exit 3/4 -> this script exits 3/4;
#     exit 2 (only slow operation, e.g. --model 31b) -> warning and continue (--strict: exit 2).
#   * --quant QUANT|ID: plain catalog lookup (pigemma/data/catalog.json), no verdict. QUANT is matched
#     case-insensitively within --model's family; a quant that exists in both families needs --model.
#   * Placeholder catalog entries (file name with '<...>', bytes_exact false, no byte count) are refused
#     (exit 3): run `python -m pigemma.gguf_map catalog-refresh` on a machine with network first.
#   * --draft: the family's sidecar from the model's catalog `drafts` list (spec_type draft-mtp by default,
#     --draft-type dflash for the 26B DFlash draft).
#
# DISK RULE (same formula as pi/lib/common.sh require_free_disk, applied to the bytes still missing)
#   need = max(MIN_FREE_DISK_FACTOR x remaining, remaining + KEEP_FREE_GB x 1e9 if the destination is on the
#   root filesystem); remaining = model + draft bytes not yet on disk (a .part counts as already fetched).
#   In auto mode, when the recommendation does not fit (e.g. the real 16 GB Pi: 26B QAT Q4_0 14.44 GB on a
#   microSD with 15.50 GB free), pigemma.feasibility is re-run on a copy of the facts whose free_disk_bytes is
#   the fetch budget translated into its own 1.05 x rule, so its ranking picks the best model that fits
#   (there: 26b-ud-q3_k_m). --no-disk-fallback refuses instead (exit 3). Explicit quants never fall back.
#   Real runs re-check the live free space (require_free_disk) right before downloading.
#
# DOWNLOAD (Hugging Face; HF_ENDPOINT overrides https://huggingface.co, e.g. a mirror or a test server)
#   URL  = $HF_ENDPOINT/<repo>/resolve/main/<file> (--pin-revision: resolve/<catalog revision>).
#   HEAD = first hop only: X-Linked-Size must equal the catalog bytes and X-Linked-ETag (sha256) the catalog
#          sha256, otherwise the upstream file changed -> exit 1 before any byte is downloaded.
#   ETA  = a 20 MB ranged probe (`curl -r 0-19999999 -o /dev/null`) timed; --no-probe or failure -> the
#          assumed ~10 MB/s of FACTS.md (Pi Wi-Fi to Hugging Face).
#   Each attempt: nice -n $NICE_LEVEL [ionice -c3] curl -fL -C - -sS --connect-timeout 30 --speed-limit 100000
#          --speed-time 60 ${HF_LIMIT_RATE:+--limit-rate $HF_LIMIT_RATE} -o <dest>/<file>.part <URL>
#   DEVIATION from the manifest's `curl --retry 10 --retry-delay 5`: curl's own --retry restarts from the
#   offset of the FIRST attempt and truncated a 4.3 GB partial on the real Pi (FACTS.md 2026-10-05), so the
#   retries are separate `curl -C -` calls that each resume from the CURRENT .part size; the loop gives up
#   after --attempts (default 30) consecutive attempts without progress, waiting --retry-delay (10 s,
#   growing to 60 s) between attempts. Progress is logged every 30 s (for nohup logs).
#   DEVIATION: data goes to <file>.part and is renamed to <file> only after the size equals the catalog bytes
#   and the sha256 matches (catalog sha256, else the server's X-Linked-ETag), so no other script can ever
#   load a truncated GGUF. A verified file gets a `<file>.sha256` stamp (sha256sum format); a later run
#   finds the file complete and skips the hash while the stamp is newer than the file (--reverify forces).
#   Token (optional, the Gemma 4 GGUFs are ungated): env HF_TOKEN / HUGGING_FACE_TOKEN or --token-file; it is
#   written to a 0600 header file used as `curl -H @file` (never on a command line, never printed, removed at
#   exit; curl drops it on the cross-host redirect to the CDN).
#
# DESKTOP COPY (--from-desktop SRC; SRC = user@host:/path/to/models or a local/mounted directory)
#   rsync -aP --partial --append-verify -e "ssh $SSH_OPTS" SRC/<repo with / -> __>/<file> <dest>/<file>.part
#   (then SRC/<file>); same verification and rename. --append-verify (beyond the manifest's -aP --partial)
#   resumes a 14 GB file by appending instead of re-checksumming it. The desktop needs sshd + rsync.
#
# OUTPUTS
#   <dest>/<file> (+ .sha256 stamp); pi/out/fetch.json {model_id, file, bytes, sha256_verified, seconds} plus
#   path, url, source, status (downloaded|copied|present), sha256, sha256_expected, sha256_source,
#   downloaded_bytes, avg_mbps, attempts, verify_seconds, selection, verdict, verdict_exit, llama_flags,
#   fits_repacked, projected_tok_s, quant, family, repo, revision, dest, probe_mbps, eta_s, host, emulated,
#   override, created_at, notes[], draft {same item keys} | null.
#   stdout: shell-safe FETCH_*=value lines (eval-able; --dry-run also prints the planned commands as '#'
#   comment lines first); everything human-readable goes to stderr.
#
# EXIT CODES (shared pi/* convention)
#   0 ok (fetched, already present and verified, or a dry-run plan) | 1 usage / network / verification error
#   2 only slow operation possible and --strict | 3 not viable, placeholder entry, not enough disk, missing
#   tool | 4 refused (protected or busy host, another fetch running)
#
# SAFETY: assert_safe_host first (exit 4 on a protected host unless PIGEMMA_I_KNOW=1); downloads and hashing
# run under nice -n $NICE_LEVEL + ionice -c3; one fetch per PIGEMMA_OUT (lock); never deletes or overwrites a
# file it did not create (a wrong-size <file> or an oversized/bad .part -> exit 1 with the rm command to run);
# only ever signals its own curl/rsync child. --dry-run touches no network and writes nothing.
#
# Environment knobs: HF_ENDPOINT, HF_TOKEN, HUGGING_FACE_TOKEN, PIGEMMA_FETCH_POLL_S (2 s between checks of
# the background transfer), PIGEMMA_FETCH_PROGRESS_S (30 s between progress lines), PIGEMMA_FETCH_PROBE_MIN
# (200000000: no speed probe for smaller downloads), plus everything of pi/pi_config.env / pi/lib/common.sh.
set -euo pipefail

PI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
. "$PI_DIR/lib/common.sh"

FM_VERSION="1.0.0"
FM_SELF="$PI_DIR/04_fetch_model.sh"
FM_PROBE_BYTES=20000000
FM_ASSUMED_MBPS=10
FM_ASSUMED_SOURCE="assumed: ~10 MB/s from Hugging Face on the Pi's Wi-Fi (docs/design/FACTS.md 2026-10-05)"
FM_SSH_OPTS_DEFAULT="-o BatchMode=yes -o ConnectTimeout=20"
FM_POLL_S=${PIGEMMA_FETCH_POLL_S:-2}
FM_PROGRESS_S=${PIGEMMA_FETCH_PROGRESS_S:-30}
FM_PROBE_MIN=${PIGEMMA_FETCH_PROBE_MIN:-200000000}
FM_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
FM_ENDPOINT=${FM_ENDPOINT%/}

# ============================================================================= python helper (stdlib)
# Catalog lookup, URL construction, facts handling, JSON writing and a sha256 fallback. Commands:
#   entry CATALOG ARR SPEC FAMILY ENDPOINT   ARR[key]=value lines for one runnable model (id or quant)
#   draft CATALOG ARR MODEL_ID TYPE ENDPOINT ARR[key]=value lines for the model's mtp|dflash sidecar
#   facts-disk FACTS                         FD[free|on_root|model_dir|override|hostname]=value lines
#   adjust-facts IN OUT FREE                 copy of IN with free_disk_bytes = FREE
#   disk-factor HOME                         pigemma.feasibility.DISK_FACTOR (1.05 fallback)
#   write-json OUT key[:type]=value...       atomic JSON (types s i f b n sn a; dotted keys nest)
#   sha256 FILE                              hex digest
IFS= read -r -d '' _FM_PY <<'PY' || true
import hashlib, json, os, re, shlex, sys
from urllib.parse import quote

try:
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
except (AttributeError, ValueError):
    pass


def fail(code, msg):
    sys.stderr.write("pi/04_fetch_model.sh: " + msg + "\n")
    sys.exit(code)


def load(path):
    try:
        with open(path, encoding="utf-8-sig") as fh:
            return json.load(fh)
    except OSError as exc:
        fail(3, "cannot read %s: %s" % (path, exc))
    except ValueError as exc:
        fail(1, "invalid JSON in %s: %s" % (path, exc))


def int_of(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v == v and v >= 0:
        return int(v)
    return None


def sha_of(v):
    s = str(v or "").strip().strip('"').lower()
    return s if re.fullmatch(r"[0-9a-f]{64}", s) else ""


def placeholder(m):
    f = str(m.get("file") or "")
    b = int_of(m.get("bytes"))
    return (not f) or "<" in f or ">" in f or m.get("bytes_exact") is False or b is None or b <= 0


def hf_url(endpoint, repo, rev, file):
    return "%s/%s/resolve/%s/%s" % (endpoint.rstrip("/"), quote(repo, safe="/"), quote(rev, safe=""),
                                    quote(file, safe="/"))


def models_of(cat):
    ms = cat.get("models") if isinstance(cat, dict) else None
    if not isinstance(ms, list):
        fail(1, "catalog has no models[] list")
    return [m for m in ms if isinstance(m, dict)]


def lines(arr, vals):
    return "".join("%s[%s]=%s\n" % (arr, k, shlex.quote("" if v is None else str(v))) for k, v in vals)


def assign(arr, m, endpoint):
    repo = str(m.get("repo") or "")
    file = str(m.get("file") or "")
    rev = str(m.get("revision") or "")
    lp = str(m.get("local_path") or "")
    b = int_of(m.get("bytes"))
    ok = bool(repo and file and "<" not in file)
    return lines(arr, [
        ("id", m.get("id")), ("kind", m.get("kind")), ("family", m.get("family")), ("quant", m.get("quant")),
        ("repo", repo), ("file", file), ("bytes", "" if b is None else b),
        ("bytes_exact", "1" if m.get("bytes_exact") is True else "0"),
        ("sha256", sha_of(m.get("sha256"))), ("revision", rev),
        ("placeholder", "1" if placeholder(m) else "0"),
        ("repo_dir", repo.replace("/", "__")),
        ("local_rel", lp[len("models/"):] if lp.startswith("models/") else ""),
        ("url_main", hf_url(endpoint, repo, "main", file) if ok else ""),
        ("url_pinned", hf_url(endpoint, repo, rev, file) if ok and rev else ""),
        ("spec_type", m.get("spec_type") or ""), ("draft_type", m.get("draft_type") or ""),
        ("estimated", "1" if m.get("estimated") else "0"),
    ])


def cmd_entry(catalog, arr, spec, family, endpoint):
    runnable = [m for m in models_of(load(catalog)) if m.get("kind") != "draft"]
    s = spec.strip().lower()
    fam = family.strip().lower()
    fams = sorted({str(m.get("family", "")).lower() for m in runnable})
    if fam and fam not in fams:
        by_id = [m for m in runnable if str(m.get("id", "")).lower() == fam]
        if not by_id:
            fail(3, "--model %r matches no family (%s) and no model id" % (family, ", ".join(fams)))
        fam = str(by_id[0].get("family", "")).lower()
    hit = [m for m in runnable if str(m.get("id", "")).lower() == s]
    if hit:
        m = hit[0]
        if fam and str(m.get("family", "")).lower() != fam:
            fail(1, "model id %r belongs to family %r, not %r" % (spec, m.get("family"), family))
    else:
        cands = [m for m in runnable if str(m.get("quant", "")).lower() == s
                 and (not fam or str(m.get("family", "")).lower() == fam)]
        if not cands:
            avail = "; ".join("%s: %s" % (f, ", ".join("%s (%s)" % (m.get("quant"), m.get("id"))
                                                     for m in runnable if str(m.get("family", "")).lower() == f))
                              for f in fams if not fam or f == fam)
            fail(3, "no catalog model with quant or id %r%s; available: %s"
                 % (spec, " in family %r" % fam if fam else "", avail))
        if len(cands) > 1:
            fail(5, "quant %r exists in several families (%s): add --model 26b|31b"
                 % (spec, ", ".join(str(m.get("id")) for m in cands)))
        m = cands[0]
    sys.stdout.write(lines(arr, [("family_filter", fam)]) + assign(arr, m, endpoint))


def cmd_draft(catalog, arr, model_id, typ, endpoint):
    ms = models_of(load(catalog))
    by_id = {str(m.get("id")): m for m in ms}
    m = by_id.get(model_id)
    if m is None:
        fail(3, "model id %r not in the catalog" % model_id)
    ids = [str(i) for i in (m.get("drafts") or [])]
    cands = [by_id[i] for i in ids if i in by_id and by_id[i].get("kind") == "draft"]
    pick = [d for d in cands if d.get("spec_type") == "draft-" + typ or d.get("draft_type") == typ]
    if not pick:
        fail(3, "model %s has no %s draft in the catalog (drafts: %s)" % (model_id, typ, ", ".join(ids) or "none"))
    sys.stdout.write(assign(arr, pick[0], endpoint))


def cmd_facts_disk(path):
    f = load(path)
    if not isinstance(f, dict):
        fail(1, "%s is not a facts object" % path)
    on_root = f.get("model_dir_on_root_fs")
    free = int_of(f.get("free_disk_bytes"))
    sys.stdout.write(lines("FD", [
        ("free", "" if free is None else free),
        ("on_root", "1" if on_root is True else ("0" if on_root is False else "")),
        ("model_dir", f.get("model_dir") or ""), ("override", f.get("override") or ""),
        ("hostname", f.get("hostname") or ""),
    ]))


def cmd_adjust_facts(src, out, free):
    f = load(src)
    f["free_disk_bytes"] = max(0, int(free))
    f["fetch_disk_budget_note"] = ("pi/04_fetch_model.sh: free_disk_bytes replaced by the fetch disk budget "
                                   "(MIN_FREE_DISK_FACTOR / KEEP_FREE_GB) for the disk-fallback verdict")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(f, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, out)


def cmd_disk_factor(home):
    sys.path.insert(0, home)
    try:
        from pigemma.feasibility import DISK_FACTOR
        print(float(DISK_FACTOR))
    except Exception:
        print("1.05")


def cmd_write_json(out, items):
    doc = {}
    for it in items:
        key, sep, val = it.partition("=")
        if not sep:
            fail(1, "write-json: bad item %r" % it)
        name, typ = (key.rsplit(":", 1) + ["s"])[:2] if ":" in key else (key, "s")
        parts = name.split(".")
        cur = doc
        for p in parts[:-1]:
            if not isinstance(cur.get(p), dict):
                cur[p] = {}
            cur = cur[p]
        last = parts[-1]
        if typ == "a":
            cur.setdefault(last, []).append(val)
            continue
        if typ == "s":
            v = val
        elif typ == "sn":
            v = val if val != "" else None
        elif typ == "i":
            v = int(val) if val != "" else None
        elif typ == "f":
            v = round(float(val), 3) if val != "" else None
        elif typ == "b":
            v = None if val == "" else val.lower() in ("1", "true", "yes")
        elif typ == "n":
            v = None
        else:
            fail(1, "write-json: unknown type %r" % typ)
        cur[last] = v
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, out)


def cmd_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(8 << 20)
            if not b:
                break
            h.update(b)
    print(h.hexdigest())


def main(a):
    cmds = {"entry": (cmd_entry, 5), "draft": (cmd_draft, 5), "facts-disk": (cmd_facts_disk, 1),
            "adjust-facts": (cmd_adjust_facts, 3), "disk-factor": (cmd_disk_factor, 1), "sha256": (cmd_sha256, 1)}
    if not a:
        fail(1, "helper: no command")
    if a[0] == "write-json":
        if len(a) < 2:
            fail(1, "write-json: OUT required")
        return cmd_write_json(a[1], a[2:])
    if a[0] not in cmds or len(a) - 1 != cmds[a[0]][1]:
        fail(1, "helper: bad command line %r" % (a,))
    return cmds[a[0]][0](*a[1:])


main(sys.argv[1:])
PY

# ======================================================================================= small helpers
usage() {
    cat <<EOF
pi/04_fetch_model.sh $FM_VERSION - download a Gemma 4 GGUF for this Pi (resume, size + sha256 verification)

Usage: pi/04_fetch_model.sh [options] [-- extra pigemma.feasibility args]
  --model 26b|31b|ID        family (or one catalog model id) to consider (default: every runnable model)
  --quant auto|QUANT|ID     auto (default; QUANT_OVERRIDE='${QUANT_OVERRIDE:-}' replaces it): fetch the
                            pigemma.feasibility recommendation; QUANT (Q4_0, UD-Q3_K_M, ...) or a catalog id:
                            plain catalog lookup without a verdict (a quant in both families needs --model)
  --draft                   also fetch the family's speculative-decoding sidecar (-md for llama.cpp)
  --draft-type mtp|dflash   which sidecar (default mtp; dflash exists for the 26B only)
  --dest DIR                target directory (default MODEL_DIR='$MODEL_DIR'; with --facts: its model_dir)
  --from-desktop SRC        rsync from SRC (user@host:REPO_ROOT/models or a mounted dir)
                            instead of Hugging Face; tries SRC/<repo with / -> __>/<file>, then SRC/<file>
  --ssh-opts "OPTS"         ssh options for --from-desktop (default: $FM_SSH_OPTS_DEFAULT)
  --facts FILE              use this facts.json instead of collecting fresh facts (planning, tests)
  --sim FILE                cache-simulation results for the verdict (pigemma.feasibility --sim)
  --catalog FILE            model catalog (default: \$PIGEMMA_CATALOG = pigemma/data/catalog.json)
  --json FILE               result file (default: \$PIGEMMA_OUT/fetch.json = pi/out/fetch.json)
  --strict                  exit 2 when only slow operation is possible (default: warn and fetch)
  --no-disk-fallback        auto mode: exit 3 instead of fetching the best model that fits the disk budget
  --pin-revision            download resolve/<catalog revision> instead of resolve/main
  --token-file FILE         HF token file (HF_TOKEN=... / HUGGING_FACE_TOKEN=... / bare token); default env
                            HF_TOKEN or HUGGING_FACE_TOKEN; optional (the Gemma 4 GGUFs are ungated)
  --no-probe                skip the 20 MB speed probe (the ETA then uses the assumed ${FM_ASSUMED_MBPS} MB/s)
  --no-verify               skip the sha256 check (the byte size is always checked)
  --reverify                re-hash a present file even when its .sha256 stamp is current
  --attempts N              give up after N consecutive attempts without progress (default 30)
  --retry-delay S           first wait between attempts in s (default 10, grows to 60)
  --bg                      run detached (nohup); log in \$PIGEMMA_OUT/fetch.log, result in fetch.json
  --dry-run                 select, check the disk budget, print the plan and the commands; no network,
                            no writes (exit code as a real run would have up to the download)
  -h, --help | --version

Disk rule: need = max(MIN_FREE_DISK_FACTOR (${MIN_FREE_DISK_FACTOR}) x remaining bytes, remaining + KEEP_FREE_GB
(${KEEP_FREE_GB}) GB when the destination is on the root filesystem).
Exit codes: 0 ok | 1 usage/network/verification error | 2 slow only with --strict | 3 not viable, placeholder,
not enough disk, missing tool | 4 refused (protected/busy host, another fetch running).
stdout: FETCH_*=value lines (shell-safe); the report goes to stderr.
Examples:
  pi/04_fetch_model.sh --dry-run                     # what would be fetched, how big, how long
  PIGEMMA_I_KNOW=1 pi/04_fetch_model.sh --model 26b --bg
  pi/04_fetch_model.sh --quant UD-Q3_K_M --draft     # explicit 3-bit 26B + its MTP sidecar
  pi/04_fetch_model.sh --model 26b --from-desktop user@desktop:REPO_ROOT/models
EOF
}

_fm_size() { # _fm_size FILE -> bytes (0 when absent)
    local s=""
    if [ -f "$1" ]; then
        s=$(stat -c %s -- "$1" 2>/dev/null) || s=$(wc -c <"$1" 2>/dev/null) || s=0
        s=${s//[[:space:]]/}
    fi
    printf '%s\n' "${s:-0}"
}

_fm_dur() { # _fm_dur SECONDS -> "1h 02m 03s"
    local s=${1:-}
    if ! _pg_is_uint "$s"; then
        printf 'n/a\n'
        return 0
    fi
    printf '%dh %02dm %02ds\n' $((s / 3600)) $((s % 3600 / 60)) $((s % 60))
}

_fm_mbps() { # _fm_mbps BYTES SECONDS -> MB/s (1e6) with one decimal
    awk -v b="$1" -v s="$2" 'BEGIN { if (s + 0 <= 0) s = 1; printf "%.1f\n", b / s / 1e6 }'
}

_fm_eta() { # _fm_eta BYTES MBPS -> seconds (integer), empty when the speed is unknown
    awk -v b="$1" -v m="$2" 'BEGIN { if (m + 0 <= 0) { print ""; exit } printf "%.0f\n", b / (m * 1e6) }'
}

_fm_rate_mbps() { # curl --limit-rate value (k/m/g = 1024 multiples) -> MB/s, empty when unset
    local r=${1:-} n u
    [ -n "$r" ] || return 0
    n=${r%[KkMmGg]}
    u=${r#"$n"}
    awk -v n="$n" -v u="$u" 'BEGIN { m = 1; if (u ~ /[Kk]/) m = 1024; if (u ~ /[Mm]/) m = 1048576;
        if (u ~ /[Gg]/) m = 1073741824; printf "%.3f\n", n * m / 1e6 }'
}

_fm_q() { # _fm_q WORDS... -> shell-quoted command line
    local out="" w
    for w in "$@"; do
        out+="$(printf '%q' "$w") "
    done
    printf '%s\n' "${out% }"
}

_fm_py() { # _fm_py CMD ARGS... (stdlib helper above)
    _pg_need_python
    "$PY" -c "$_FM_PY" "$@"
}

_fm_abs() { # _fm_abs PATH -> absolute path (not resolved)
    case $1 in
        /* | [A-Za-z]:/* | [A-Za-z]:\\*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "$PWD/$1" ;;
    esac
}

# ======================================================================================== arguments
FM_ORIG_ARGS=("$@")
FM_MODEL=""
FM_QUANT=${QUANT_OVERRIDE:-}
FM_QUANT_SRC="QUANT_OVERRIDE"
FM_DRAFT=0
FM_DRAFT_TYPE=mtp
FM_DEST=""
FM_FROM=""
FM_SSH_OPTS=$FM_SSH_OPTS_DEFAULT
FM_FACTS=""
FM_SIM=""
FM_CATALOG=${PIGEMMA_CATALOG}
FM_JSON=""
FM_STRICT=0
FM_FALLBACK=1
FM_PIN_REV=0
FM_TOKEN_FILE=""
FM_PROBE=1
FM_VERIFY=1
FM_REVERIFY=0
FM_STALLS=30
FM_RETRY_DELAY=10
FM_BG=0
FM_DRY=0
FM_EXTRA=()

# --opt=value -> --opt value (up to a bare --)
_fm_norm=()
_fm_seen_dd=0
for _fm_a in "$@"; do
    if [ "$_fm_seen_dd" = 1 ]; then
        _fm_norm+=("$_fm_a")
        continue
    fi
    case $_fm_a in
        --) _fm_seen_dd=1; _fm_norm+=("$_fm_a") ;;
        --*=*) _fm_norm+=("${_fm_a%%=*}" "${_fm_a#*=}") ;;
        *) _fm_norm+=("$_fm_a") ;;
    esac
done
set -- "${_fm_norm[@]+"${_fm_norm[@]}"}"

_fm_need_val() { # _fm_need_val OPT NARGS
    if [ "$2" -lt 2 ]; then
        usage >&2
        die 1 "option $1 needs a value"
    fi
}

while [ $# -gt 0 ]; do
    case $1 in
        --model) _fm_need_val "$1" $#; FM_MODEL=$2; shift 2 ;;
        --quant) _fm_need_val "$1" $#; FM_QUANT=$2; FM_QUANT_SRC="--quant"; shift 2 ;;
        --draft) FM_DRAFT=1; shift ;;
        --draft-type) _fm_need_val "$1" $#; FM_DRAFT_TYPE=$2; FM_DRAFT=1; shift 2 ;;
        --dest) _fm_need_val "$1" $#; FM_DEST=$2; shift 2 ;;
        --from-desktop) _fm_need_val "$1" $#; FM_FROM=$2; shift 2 ;;
        --ssh-opts) _fm_need_val "$1" $#; FM_SSH_OPTS=$2; shift 2 ;;
        --facts) _fm_need_val "$1" $#; FM_FACTS=$2; shift 2 ;;
        --sim) _fm_need_val "$1" $#; FM_SIM=$2; shift 2 ;;
        --catalog) _fm_need_val "$1" $#; FM_CATALOG=$2; shift 2 ;;
        --json) _fm_need_val "$1" $#; FM_JSON=$2; shift 2 ;;
        --strict) FM_STRICT=1; shift ;;
        --no-disk-fallback) FM_FALLBACK=0; shift ;;
        --pin-revision) FM_PIN_REV=1; shift ;;
        --token-file) _fm_need_val "$1" $#; FM_TOKEN_FILE=$2; shift 2 ;;
        --no-probe) FM_PROBE=0; shift ;;
        --no-verify) FM_VERIFY=0; shift ;;
        --reverify) FM_REVERIFY=1; shift ;;
        --attempts) _fm_need_val "$1" $#; FM_STALLS=$2; shift 2 ;;
        --retry-delay) _fm_need_val "$1" $#; FM_RETRY_DELAY=$2; shift 2 ;;
        --bg | --background) FM_BG=1; shift ;;
        --dry-run) FM_DRY=1; shift ;;
        -h | --help) usage; exit 0 ;;
        --version) printf 'pi/04_fetch_model.sh %s\n' "$FM_VERSION"; exit 0 ;;
        --) shift; FM_EXTRA=("$@"); break ;;
        *) usage >&2; die 1 "unknown option or argument '$1' (see --help)" ;;
    esac
done

case ${FM_QUANT,,} in "" | auto) FM_QUANT=auto ;; esac
if [ "$FM_QUANT" != auto ] && ! [[ $FM_QUANT =~ ^[A-Za-z0-9_.-]+$ ]]; then die 1 "--quant '$FM_QUANT' is not a quant name or model id"; fi
if [ -n "$FM_MODEL" ] && ! [[ $FM_MODEL =~ ^[A-Za-z0-9_.-]+$ ]]; then die 1 "--model '$FM_MODEL' is not a family or model id"; fi
case $FM_DRAFT_TYPE in mtp | dflash) ;; *) die 1 "--draft-type must be mtp or dflash, got '$FM_DRAFT_TYPE'" ;; esac
_pg_is_uint "$FM_STALLS" && [ "$FM_STALLS" -ge 1 ] || die 1 "--attempts must be a positive integer, got '$FM_STALLS'"
_pg_is_uint "$FM_RETRY_DELAY" || die 1 "--retry-delay must be a non-negative integer, got '$FM_RETRY_DELAY'"
_pg_is_num "$FM_POLL_S" || die 1 "PIGEMMA_FETCH_POLL_S must be a number, got '$FM_POLL_S'"
_pg_is_uint "$FM_PROGRESS_S" || die 1 "PIGEMMA_FETCH_PROGRESS_S must be an integer, got '$FM_PROGRESS_S'"
_pg_is_uint "$FM_PROBE_MIN" || die 1 "PIGEMMA_FETCH_PROBE_MIN must be an integer, got '$FM_PROBE_MIN'"
[[ $FM_ENDPOINT =~ ^https?://[^[:space:]]+$ ]] || die 1 "HF_ENDPOINT='$FM_ENDPOINT' is not an http(s) URL"
if [ -n "$FM_FACTS" ] && [ ! -f "$FM_FACTS" ]; then die 3 "--facts file not found: $FM_FACTS"; fi
if [ -n "$FM_SIM" ] && [ ! -f "$FM_SIM" ]; then die 3 "--sim file not found: $FM_SIM"; fi
if [ ! -f "$FM_CATALOG" ]; then die 3 "catalog not found: $FM_CATALOG"; fi
# absolute paths: pigemma.feasibility runs with cwd = PIGEMMA_HOME
FM_CATALOG=$(_fm_abs "$FM_CATALOG")
if [ -n "$FM_FACTS" ]; then FM_FACTS=$(_fm_abs "$FM_FACTS"); fi
if [ -n "$FM_SIM" ]; then FM_SIM=$(_fm_abs "$FM_SIM"); fi
if [ -n "$FM_JSON" ]; then FM_JSON=$(_fm_abs "$FM_JSON"); fi
if [ -n "$FM_TOKEN_FILE" ]; then FM_TOKEN_FILE=$(_fm_abs "$FM_TOKEN_FILE"); fi
if [ -n "$FM_TOKEN_FILE" ] && [ ! -r "$FM_TOKEN_FILE" ]; then die 3 "--token-file not readable: $FM_TOKEN_FILE"; fi

# ============================================================================================ start
pigemma_init
assert_safe_host
FM_JSON=${FM_JSON:-$PIGEMMA_OUT/fetch.json}
FM_NOTES=()
if [ "${PIGEMMA_PROTECTED_OVERRIDE:-0}" = 1 ]; then
    FM_NOTES+=("protected host override PIGEMMA_I_KNOW=1 (nice -n $NICE_LEVEL, ionice -c3, keep ${KEEP_FREE_GB} GB free on /)")
fi
if is_emulated; then FM_NOTES+=("emulated host (PIGEMMA_EMULATED=1)"); fi

if [ "$FM_BG" = 1 ] && [ "$FM_DRY" = 0 ]; then
    _fm_child_args=()
    for _fm_a in "${FM_ORIG_ARGS[@]+"${FM_ORIG_ARGS[@]}"}"; do
        case $_fm_a in --bg | --background) ;; *) _fm_child_args+=("$_fm_a") ;; esac
    done
    _fm_log="$PIGEMMA_OUT/fetch.log"
    nohup bash "$FM_SELF" "${_fm_child_args[@]+"${_fm_child_args[@]}"}" >"$_fm_log" 2>&1 </dev/null &
    _fm_bg_pid=$!
    log "fetch running detached (pid $_fm_bg_pid): tail -f $_fm_log ; result -> $FM_JSON"
    printf 'FETCH_BG_PID=%s\nFETCH_LOG=%q\nFETCH_JSON=%q\n' "$_fm_bg_pid" "$_fm_log" "$FM_JSON"
    exit 0
fi

# destination: --dest > facts.model_dir (with --facts) > MODEL_DIR
declare -A FD=()
if [ -n "$FM_FACTS" ]; then
    _fm_out=$(_fm_py facts-disk "$(native_path "$FM_FACTS")") || die 1 "cannot read --facts $FM_FACTS"
    eval "$_fm_out"
fi
if [ -z "$FM_DEST" ]; then
    if [ -n "$FM_FACTS" ] && [ -n "${FD[model_dir]:-}" ]; then
        FM_DEST=${FD[model_dir]}
    else
        FM_DEST=$MODEL_DIR
    fi
fi
FM_DEST=$(_fm_abs "$FM_DEST")
FM_DEST=${FM_DEST%/}
if [ -n "$FM_FACTS" ] && [ -n "${FD[model_dir]:-}" ] && [ "${FD[model_dir]%/}" != "$FM_DEST" ]; then
    warn "--facts describes model_dir ${FD[model_dir]} but the destination is $FM_DEST: the disk numbers below come from the facts file"
fi
MODEL_DIR=$FM_DEST
export MODEL_DIR

# temp files (dry-run writes nothing outside these, all removed at exit)
FM_TMP="$PIGEMMA_OUT/.fetch.$$"
_fm_cleanup_tmp() { rm -f "$FM_TMP".* 2>/dev/null || true; }
add_cleanup _fm_cleanup_tmp
FM_CHILD=""
_fm_kill_child() {
    if [ -n "${FM_CHILD:-}" ] && kill -0 "$FM_CHILD" 2>/dev/null; then
        kill -TERM "$FM_CHILD" 2>/dev/null || true
        warn "stopped our download process (pid $FM_CHILD); the .part file is kept and the next run resumes it"
    fi
}
add_cleanup _fm_kill_child
trap 'exit 130' INT
trap 'exit 143' TERM

# ========================================================================================= selection
declare -A M=() D=()
FM_SELECTION=explicit
FM_VERDICT=""
FM_VERDICT_EXIT=""
FM_LLAMA_FLAGS=""
FM_FITS_REPACKED=""
FM_TOK_S=""
FM_FACTS_USED=""

_fm_feasibility() { # _fm_feasibility FACTS JSON_OUT QUIET(0|1) -> sources the env lines; returns the exit code
    local facts=$1 json=$2 quiet=$3 rc=0 envf="$FM_TMP.env" keep_pin=${PIN_BUDGET_BYTES:-auto}
    local args=(--facts "$(native_path "$facts")" --json "$(native_path "$json")" --catalog "$(native_path "$FM_CATALOG")" --emit env)
    if [ -n "$FM_MODEL" ]; then args+=(--want "$FM_MODEL"); fi
    if [ -n "$FM_SIM" ]; then args+=(--sim "$(native_path "$FM_SIM")"); fi
    if [ "$quiet" = 1 ]; then args+=(--quiet); fi
    MODEL_ID=""
    VERDICT=""
    VERDICT_EXIT=""
    LLAMA_FLAGS=""
    FITS_REPACKED=""
    PROJECTED_TOK_S=""
    run_feasibility "${args[@]}" "${FM_EXTRA[@]+"${FM_EXTRA[@]}"}" >"$envf" || rc=$?
    if [ -s "$envf" ]; then
        # shellcheck disable=SC1090
        . "$envf" # KEY='value' lines, shlex-quoted by pigemma.feasibility.emit_env
    fi
    rm -f "$envf"
    PIN_BUDGET_BYTES=$keep_pin # the verdict's pin budget is pi/06's business; keep the config value
    return "$rc"
}

_fm_gate() { # _fm_gate RC CONTEXT -> continue or die with the verdict code
    case $1 in
        0) return 0 ;;
        2)
            if [ "$FM_STRICT" = 1 ]; then
                die 2 "verdict for $2: only slow operation is possible (exit 2) and --strict was given"
            fi
            warn "verdict for $2: only slow operation is possible (exit 2) - fetching anyway (use --strict to refuse)"
            FM_NOTES+=("verdict exit 2 (slow only) accepted")
            ;;
        3) die 3 "verdict for $2: nothing viable on this host (exit 3) - nothing to fetch; see the table above" ;;
        4) die 4 "verdict for $2: refused (protected or busy host, exit 4) - nothing was fetched" ;;
        *) die 1 "pigemma.feasibility failed (exit $1) for $2: check the facts and the catalog" ;;
    esac
}

_fm_entry() { # _fm_entry ARR SPEC FAMILY -> fills the associative array ARR from the catalog
    local out rc=0
    out=$(_fm_py entry "$(native_path "$FM_CATALOG")" "$1" "$2" "$3" "$FM_ENDPOINT") || rc=$?
    case $rc in
        0) eval "$out" ;;
        3) die 3 "catalog lookup failed for '$2' (see the message above)" ;;
        5) die 1 "ambiguous --quant '$2' (see the message above)" ;;
        *) die 1 "catalog lookup failed for '$2' (exit $rc)" ;;
    esac
}

_fm_draft_entry() { # fills D for model M[id]
    local out rc=0
    out=$(_fm_py draft "$(native_path "$FM_CATALOG")" D "${M[id]}" "$FM_DRAFT_TYPE" "$FM_ENDPOINT") || rc=$?
    case $rc in
        0) eval "$out" ;;
        *) die 3 "no $FM_DRAFT_TYPE draft for ${M[id]} in the catalog (see the message above)" ;;
    esac
}

_fm_check_entry() { # _fm_check_entry ARR LABEL -> die 3 on placeholder entries
    local -n _ce=$1
    if [ "${_ce[placeholder]}" = 1 ]; then
        die 3 "catalog entry ${_ce[id]} ($2) is a placeholder (file '${_ce[file]}', bytes '${_ce[bytes]}', bytes_exact ${_ce[bytes_exact]}): run 'python -m pigemma.gguf_map catalog-refresh --ids ${_ce[id]}' on a machine with network, then retry"
    fi
    if [ -z "${_ce[url_main]}" ]; then
        die 3 "catalog entry ${_ce[id]} has no repo/file to download"
    fi
}

_fm_select() { # sets M (+ D) from the verdict or the catalog
    if [ "$FM_QUANT" = auto ]; then
        FM_SELECTION=auto
        if [ -n "$FM_FACTS" ]; then
            FM_FACTS_USED=$FM_FACTS
        elif [ "$FM_DRY" = 1 ]; then
            FM_FACTS_USED="$FM_TMP.facts.json"
            write_facts "$FM_FACTS_USED"
        else
            FM_FACTS_USED="$PIGEMMA_OUT/facts.json"
            write_facts "$FM_FACTS_USED"
        fi
        local fz="$PIGEMMA_OUT/feasibility.json" rc=0
        if [ "$FM_DRY" = 1 ] || [ -n "$FM_FACTS" ]; then fz="$FM_TMP.feasibility.json"; fi
        _fm_feasibility "$FM_FACTS_USED" "$fz" 0 || rc=$?
        FM_VERDICT=$VERDICT
        FM_VERDICT_EXIT=$rc
        _fm_gate "$rc" "${FM_MODEL:-all models}"
        if [ -z "${MODEL_ID:-}" ]; then
            die 3 "pigemma.feasibility recommended nothing (verdict ${VERDICT:-?}) - nothing to fetch"
        fi
        FM_LLAMA_FLAGS=$LLAMA_FLAGS
        FM_FITS_REPACKED=$FITS_REPACKED
        FM_TOK_S=$PROJECTED_TOK_S
        _fm_entry M "$MODEL_ID" ""
        log "verdict: $MODEL_ID (${M[quant]}) $VERDICT, projected ${PROJECTED_TOK_S:-?} tok/s (PROJECTED, see the table above)"
    else
        FM_SELECTION=explicit
        _fm_entry M "$FM_QUANT" "$FM_MODEL"
        if [ -n "$FM_FACTS" ]; then FM_FACTS_USED=$FM_FACTS; fi
        log "explicit selection ($FM_QUANT_SRC=$FM_QUANT${FM_MODEL:+, --model $FM_MODEL}): ${M[id]} - no feasibility verdict consulted (pi/00_feasibility.sh --want ${M[id]} shows the projection)"
    fi
    _fm_check_entry M model
    if [ "$FM_DRAFT" = 1 ]; then
        D=()
        _fm_draft_entry
        _fm_check_entry D draft
    fi
}

# ============================================================================== presence + disk budget
_fm_inspect() { # _fm_inspect ARR -> final, part, stamp, state (complete|partial|absent), present_path, part_bytes, remaining
    local -n _in=$1
    local f=${_in[file]} c sz cands=()
    _in[final]="$FM_DEST/$f"
    _in[part]="${_in[final]}.part"
    _in[stamp]="${_in[final]}.sha256"
    _in[state]=absent
    _in[present_path]=""
    _in[part_bytes]=0
    cands=("${_in[final]}")
    if [ -n "${_in[repo_dir]}" ]; then cands+=("$FM_DEST/${_in[repo_dir]}/$f"); fi
    cands+=("$FM_DEST/${f##*/}")
    for c in "${cands[@]}"; do
        [ -f "$c" ] || continue
        sz=$(_fm_size "$c")
        if [ "$sz" = "${_in[bytes]}" ]; then
            _in[state]=complete
            _in[present_path]=$c
            break
        fi
        if [ "$c" = "${_in[final]}" ]; then
            die 1 "${_in[final]} exists with $sz bytes but the catalog says ${_in[bytes]} for ${_in[id]} (another revision or a truncated copy): move or delete it yourself (rm -- '${_in[final]}'), nothing was changed"
        fi
    done
    if [ "${_in[state]}" != complete ] && [ -f "${_in[part]}" ]; then
        _in[part_bytes]=$(_fm_size "${_in[part]}")
        if [ "${_in[part_bytes]}" -gt "${_in[bytes]}" ]; then
            die 1 "${_in[part]} has ${_in[part_bytes]} bytes, more than the catalog's ${_in[bytes]}: delete it yourself (rm -- '${_in[part]}') and retry"
        fi
        _in[state]=partial
    fi
    if [ "${_in[state]}" = complete ]; then
        _in[remaining]=0
    else
        _in[remaining]=$((${_in[bytes]} - ${_in[part_bytes]}))
    fi
}

FM_FREE=""
FM_ON_ROOT=""
FM_DISK_SRC=""
FM_KEEP=0
FM_NEED=0
_fm_disk_numbers() { # FM_FREE FM_ON_ROOT FM_KEEP FM_DISK_SRC from the facts used (or live df)
    local d
    if [ -n "$FM_FACTS_USED" ]; then
        local out
        out=$(_fm_py facts-disk "$(native_path "$FM_FACTS_USED")") || die 1 "cannot read $FM_FACTS_USED"
        eval "$out"
        FM_FREE=${FD[free]:-}
        FM_ON_ROOT=${FD[on_root]:-}
        FM_DISK_SRC="facts ($FM_FACTS_USED)"
        if [ -z "$FM_ON_ROOT" ]; then
            FM_ON_ROOT=1
            FM_DISK_SRC="$FM_DISK_SRC; root filesystem assumed (model_dir_on_root_fs unknown)"
        fi
    else
        d=$(_pg_existing_dir "$FM_DEST")
        _pg_df "$d"
        FM_FREE=$DF_AVAIL
        if [ "$DF_MOUNT" = / ]; then FM_ON_ROOT=1; else FM_ON_ROOT=0; fi
        FM_DISK_SRC="live df ($d on ${DF_MOUNT:-?})"
    fi
    if ! _pg_is_uint "$FM_FREE"; then
        die 3 "cannot determine the free disk space for $FM_DEST ($FM_DISK_SRC) - refusing to download"
    fi
    if [ "$FM_ON_ROOT" = 1 ]; then
        FM_KEEP=$(awk -v g="$KEEP_FREE_GB" 'BEGIN { printf "%.0f\n", g * 1e9 }')
    else
        FM_KEEP=0
    fi
}

_fm_disk_fits() { # _fm_disk_fits REMAINING -> 0 if it fits; sets FM_NEED
    FM_NEED=$(awk -v b="$1" -v f="$MIN_FREE_DISK_FACTOR" -v k="$FM_KEEP" \
        'BEGIN { if (b + 0 <= 0) { print 0; exit } a = b * f; c = b + k; n = (a > c) ? a : c;
                 if (n > int(n)) n = int(n) + 1; printf "%.0f\n", n }')
    [ "$1" -le 0 ] || [ "$FM_FREE" -ge "$FM_NEED" ]
}

_fm_remaining_total() {
    local t=${M[remaining]}
    if [ "$FM_DRAFT" = 1 ]; then t=$((t + ${D[remaining]})); fi
    printf '%s\n' "$t"
}

_fm_inspect_all() {
    _fm_inspect M
    if [ "$FM_DRAFT" = 1 ]; then _fm_inspect D; fi
}

_fm_disk_budget() { # checks the disk rule; auto mode may fall back to a smaller model
    _fm_disk_numbers
    _fm_inspect_all
    local rem
    rem=$(_fm_remaining_total)
    if _fm_disk_fits "$rem"; then
        return 0
    fi
    local first="${M[id]}" first_need=$FM_NEED
    local what=${M[id]}
    if [ "$FM_DRAFT" = 1 ]; then what="$what + ${D[id]}"; fi
    local msg="not enough free disk for $what: free $(fmt_gb "$FM_FREE") ($FM_DISK_SRC), need $(fmt_gb "$FM_NEED") = max(${MIN_FREE_DISK_FACTOR} x $(fmt_gb "$rem") remaining, $(fmt_gb "$rem") + keep-free $(fmt_gb "$FM_KEEP"))"
    if [ "$FM_SELECTION" != auto ] || [ "$FM_FALLBACK" = 0 ]; then
        die 3 "$msg"
    fi
    warn "$msg"
    # translate the fetch rule (remaining <= cap) into pigemma.feasibility's 1.05 x (bytes - present) <= free
    local factor cap draft_rem=0 free_adj adj="$FM_TMP.facts-fetch.json" fz rc=0
    factor=$(_fm_py disk-factor "$(native_path "$PIGEMMA_HOME")")
    if [ "$FM_DRAFT" = 1 ]; then draft_rem=${D[remaining]}; fi
    cap=$(awk -v f="$FM_FREE" -v m="$MIN_FREE_DISK_FACTOR" -v k="$FM_KEEP" -v d="$draft_rem" \
        'BEGIN { a = f / m; b = f - k; c = (a < b) ? a : b; c -= d; if (c < 0) c = 0; printf "%.0f\n", c }')
    free_adj=$(awk -v c="$cap" -v x="$factor" 'BEGIN { printf "%.0f\n", int(c * x) }')
    log "disk fallback: re-running pigemma.feasibility with free_disk_bytes = $(fmt_gb "$free_adj") (= ${factor} x fetch cap $(fmt_gb "$cap"))"
    _fm_py adjust-facts "$(native_path "$FM_FACTS_USED")" "$(native_path "$adj")" "$free_adj" ||
        die 1 "cannot write $adj"
    fz="$FM_TMP.feasibility-fallback.json"
    if [ "$FM_DRY" = 0 ] && [ -z "$FM_FACTS" ]; then fz="$PIGEMMA_OUT/feasibility.fetch.json"; fi
    _fm_feasibility "$adj" "$fz" 1 || rc=$?
    if [ "$rc" = 3 ] || [ -z "${MODEL_ID:-}" ]; then
        die 3 "no ${FM_MODEL:-} model fits the fetch disk budget ($(fmt_gb "$cap") for the model on ${FM_DEST}); free space first or point --dest / MODEL_DIR at a bigger disk (USB 3 SSD or NVMe)"
    fi
    _fm_gate "$rc" "${FM_MODEL:-all models} within the disk budget"
    if [ "$MODEL_ID" = "$first" ]; then
        die 3 "$msg (the disk-budget verdict still picks $first)"
    fi
    FM_SELECTION=auto-disk-fallback
    FM_VERDICT=$VERDICT
    FM_VERDICT_EXIT=$rc
    FM_LLAMA_FLAGS=$LLAMA_FLAGS
    FM_FITS_REPACKED=$FITS_REPACKED
    FM_TOK_S=$PROJECTED_TOK_S
    M=()
    _fm_entry M "$MODEL_ID" ""
    _fm_check_entry M model
    if [ "$FM_DRAFT" = 1 ]; then
        D=()
        _fm_draft_entry
        _fm_check_entry D draft
    fi
    FM_NOTES+=("disk fallback: $first needs $(fmt_gb "$first_need") free, have $(fmt_gb "$FM_FREE"); fetched the best model within the budget instead")
    warn "DISK FALLBACK: fetching ${M[id]} (${M[quant]}, $(fmt_gb "${M[bytes]}"), verdict $VERDICT, projected ${PROJECTED_TOK_S:-?} tok/s) instead of $first"
    _fm_inspect_all
    rem=$(_fm_remaining_total)
    if ! _fm_disk_fits "$rem"; then
        die 3 "not enough free disk even for ${M[id]}: free $(fmt_gb "$FM_FREE"), need $(fmt_gb "$FM_NEED")"
    fi
}

# =============================================================================== token / network
FM_AUTH_FILE=""
FM_HAVE_TOKEN=0
_fm_token_value() { # prints the token (never logged), empty when none
    local t="" line k v
    if [ -n "${HF_TOKEN:-}" ]; then
        t=$HF_TOKEN
    elif [ -n "${HUGGING_FACE_TOKEN:-}" ]; then
        t=$HUGGING_FACE_TOKEN
    elif [ -n "$FM_TOKEN_FILE" ]; then
        while IFS= read -r line || [ -n "$line" ]; do
            line=${line%$'\r'}
            case $line in '' | '#'*) continue ;; esac
            line=${line#export }
            if [[ $line == *=* ]]; then
                k=${line%%=*}
                v=${line#*=}
                case $k in HF_TOKEN | HUGGING_FACE_TOKEN | HUGGINGFACE_TOKEN | HUGGING_FACE_HUB_TOKEN) ;; *) continue ;; esac
            else
                v=$line
            fi
            v=${v#\"}; v=${v%\"}; v=${v#\'}; v=${v%\'}
            _pg_trim "$v"
            t=$_R
            [ -n "$t" ] && break
        done <"$FM_TOKEN_FILE"
    fi
    printf '%s' "$t"
}

_fm_setup_auth() {
    local tok
    tok=$(_fm_token_value)
    if [ -z "$tok" ]; then
        return 0
    fi
    if ! [[ $tok =~ ^[A-Za-z0-9_.-]+$ ]]; then
        die 1 "the Hugging Face token has unexpected characters (not printed); check HF_TOKEN / --token-file"
    fi
    FM_HAVE_TOKEN=1
    if [ "$FM_DRY" = 1 ]; then
        return 0
    fi
    FM_AUTH_FILE="$FM_TMP.auth"
    (umask 077 && printf 'Authorization: Bearer %s\n' "$tok" >"$FM_AUTH_FILE") || die 1 "cannot write the auth header file"
    log "Hugging Face token: present (sent as a header from a 0600 temp file, never printed)"
}

_fm_curl_common() { # -> FM_CURL array: curl + shared options
    FM_CURL=(curl -sS --connect-timeout 30)
    if [ -n "$FM_AUTH_FILE" ]; then FM_CURL+=(-H "@$(native_path "$FM_AUTH_FILE")"); fi
}

FM_NICE=()
_fm_nice_prefix() {
    FM_NICE=()
    if have_cmd nice; then FM_NICE=(nice -n "$NICE_LEVEL"); fi
    if have_cmd ionice && ionice -c3 true >/dev/null 2>&1; then
        FM_NICE+=(ionice -c3)
    else
        log "ionice -c3 not available here: downloads and hashing run with nice only"
    fi
}

_fm_head() { # _fm_head ARR -> _hd[http_code size etag commit]; never dies on network errors
    local -n _hd=$1
    local hdrs line code="" size="" etag="" commit="" proto rest
    _fm_curl_common
    hdrs=$("${FM_CURL[@]}" -I --max-time 30 "${_hd[url]}" 2>/dev/null) || hdrs=""
    while IFS= read -r line; do
        line=${line%$'\r'}
        case ${line,,} in
            http/*) read -r proto code rest <<<"$line" || true ;;
            x-linked-size:*) size=${line#*:} ;;
            x-linked-etag:*) etag=${line#*:} ;;
            x-repo-commit:*) commit=${line#*:} ;;
        esac
    done <<<"$hdrs"
    _pg_trim "$size"; size=$_R
    _pg_trim "$etag"; etag=${_R//\"/}; etag=${etag,,}
    _pg_trim "$commit"; commit=$_R
    _hd[http_code]=$code
    _hd[remote_size]=$size
    _hd[remote_sha]=""
    if [[ $etag =~ ^[0-9a-f]{64}$ ]]; then _hd[remote_sha]=$etag; fi
    _hd[remote_commit]=$commit
}

_fm_check_remote() { # _fm_check_remote ARR LABEL -> die before downloading when upstream changed
    local -n _rc=$1
    _fm_head "$1"
    local code=${_rc[http_code]}
    case $code in
        "") warn "HEAD $2 failed (network?): continuing; the download loop retries"; return 0 ;;
        2?? | 3??) ;;
        401 | 403) die 1 "HTTP $code for ${_rc[url]}: access denied - set HF_TOKEN (or --token-file) if the repo is gated" ;;
        404) die 1 "HTTP 404 for ${_rc[url]}: file not found upstream - run 'python -m pigemma.gguf_map catalog-refresh --ids ${_rc[id]}'" ;;
        *) warn "HEAD $2 returned HTTP $code: continuing; the download loop retries" ; return 0 ;;
    esac
    if [ -n "${_rc[remote_size]}" ] && [ "${_rc[remote_size]}" != "${_rc[bytes]}" ]; then
        die 1 "upstream size of ${_rc[file]} is ${_rc[remote_size]} B but the catalog says ${_rc[bytes]} B (the file changed upstream): run 'python -m pigemma.gguf_map catalog-refresh --ids ${_rc[id]}', or use --pin-revision for the catalog's revision"
    fi
    if [ -n "${_rc[remote_sha]}" ] && [ -n "${_rc[sha256]}" ] && [ "${_rc[remote_sha]}" != "${_rc[sha256]}" ]; then
        die 1 "upstream sha256 of ${_rc[file]} (X-Linked-ETag ${_rc[remote_sha]}) differs from the catalog's ${_rc[sha256]}: run 'python -m pigemma.gguf_map catalog-refresh --ids ${_rc[id]}' or use --pin-revision"
    fi
    if [ -n "${_rc[remote_commit]}" ] && [ -n "${_rc[revision]}" ] && [ "${_rc[remote_commit]}" != "${_rc[revision]}" ]; then
        log "$2: repo head ${_rc[remote_commit]:0:10} != catalog revision ${_rc[revision]:0:10} (the file itself is unchanged: size/sha match)"
    fi
    local shamsg="not sent (X-Linked-ETag missing)"
    if [ -n "${_rc[remote_sha]}" ] && [ -n "${_rc[sha256]}" ]; then
        shamsg="matches the catalog"
    elif [ -n "${_rc[remote_sha]}" ]; then
        shamsg="${_rc[remote_sha]} (from X-Linked-ETag; the catalog has none)"
    fi
    log "$2: upstream OK (HTTP $code, size ${_rc[remote_size]:-?}, sha256 $shamsg)"
}

FM_PROBE_MBPS=""
_fm_probe() { # _fm_probe URL -> FM_PROBE_MBPS (MB/s) or empty
    local out code size tt ts rc=0
    _fm_curl_common
    local cmd=("${FM_CURL[@]}" -L -r "0-$((FM_PROBE_BYTES - 1))" --max-time 60)
    if [ -n "$HF_LIMIT_RATE" ]; then cmd+=(--limit-rate "$HF_LIMIT_RATE"); fi
    log "speed probe: first $(fmt_gb "$FM_PROBE_BYTES") of the file (ranged request, discarded)"
    out=$("${cmd[@]}" -o /dev/null -w '%{http_code} %{size_download} %{time_total} %{time_starttransfer}' "$1" 2>/dev/null) || rc=$?
    read -r code size tt ts <<<"$out" || true
    if [ "$rc" != 0 ] || ! [[ ${code:-} =~ ^20[06]$ ]] || ! _pg_is_uint "${size:-}" || [ "${size:-0}" -lt 1000000 ]; then
        warn "speed probe failed (curl exit $rc, HTTP ${code:-?}, ${size:-0} B): ETA uses the assumed ${FM_ASSUMED_MBPS} MB/s"
        FM_PROBE_MBPS=""
        return 0
    fi
    FM_PROBE_MBPS=$(awk -v b="$size" -v t="$tt" -v s="$ts" 'BEGIN { d = t - s; if (d < 0.05) d = t; if (d <= 0) d = 0.001; printf "%.2f\n", b / d / 1e6 }')
    log "speed probe: $size B in ${tt}s (first byte after ${ts}s) = ${FM_PROBE_MBPS} MB/s (measured)"
}

# ============================================================================== transfer engines
FM_RC=0
_fm_watch() { # _fm_watch FILE TOTAL LABEL WOUT CMD... -> FM_RC; CMD runs in the background, progress logged
    local file=$1 total=$2 label=$3 wout=$4
    shift 4
    "$@" >"$wout" &
    FM_CHILD=$!
    local t_last now sz sz_last eta mbps
    t_last=$(date +%s)
    sz_last=$(_fm_size "$file")
    while kill -0 "$FM_CHILD" 2>/dev/null; do
        sleep "$FM_POLL_S"
        now=$(date +%s)
        if [ $((now - t_last)) -ge "$FM_PROGRESS_S" ]; then
            sz=$(_fm_size "$file")
            mbps=$(_fm_mbps $((sz - sz_last)) $((now - t_last)))
            eta=$(_fm_eta $((total - sz)) "$mbps")
            log "$label: $(fmt_gb "$sz") / $(fmt_gb "$total") ($(awk -v a="$sz" -v b="$total" 'BEGIN { printf "%.1f", (b > 0) ? 100 * a / b : 0 }')%), ${mbps} MB/s over the last $((now - t_last)) s, ETA $(_fm_dur "$eta")"
            t_last=$now
            sz_last=$sz
        fi
    done
    if wait "$FM_CHILD"; then FM_RC=0; else FM_RC=$?; fi
    FM_CHILD=""
}

_fm_delay() { # _fm_delay STALLS
    local d=$((FM_RETRY_DELAY * (${1} > 0 ? ${1} : 1)))
    if [ "$d" -gt 60 ]; then d=60; fi
    if [ "$d" -lt "$FM_RETRY_DELAY" ]; then d=$FM_RETRY_DELAY; fi
    if [ "$d" -gt 0 ]; then
        log "retrying in ${d} s"
        sleep "$d"
    fi
}

_fm_download_hf() { # _fm_download_hf ARR LABEL -> .part complete (size == bytes) or die
    local -n _dl=$1
    local label=$2 part=${_dl[part]} bytes=${_dl[bytes]} url=${_dl[url]}
    local cur prev code stalls=0 attempts=0 wout="$FM_TMP.w"
    mkdir -p "${part%/*}" || die 1 "cannot create ${part%/*}"
    _fm_curl_common
    local base=("${FM_CURL[@]}" -fL -C - --speed-limit 100000 --speed-time 60)
    if [ -n "$HF_LIMIT_RATE" ]; then base+=(--limit-rate "$HF_LIMIT_RATE"); fi
    while :; do
        cur=$(_fm_size "$part")
        if [ "$cur" -eq "$bytes" ]; then break; fi
        if [ "$cur" -gt "$bytes" ]; then
            die 1 "$part grew to $cur B, beyond the catalog's $bytes B (the server sent a different file): delete it yourself (rm -- '$part') and run catalog-refresh"
        fi
        attempts=$((attempts + 1))
        prev=$cur
        log "$label: attempt $attempts, resuming at $(fmt_gb "$cur") of $(fmt_gb "$bytes")"
        _fm_watch "$part" "$bytes" "$label" "$wout" \
            "${FM_NICE[@]+"${FM_NICE[@]}"}" "${base[@]}" -w '%{http_code}\n' -o "$(native_path "$part")" "$url"
        code=$(tr -d '\r\n ' <"$wout" 2>/dev/null || true)
        cur=$(_fm_size "$part")
        if [ "$FM_RC" = 0 ] && [ "$cur" -eq "$bytes" ]; then break; fi
        case $FM_RC in
            0) warn "$label: transfer ended at $cur B of $bytes B (HTTP ${code:-?}) - resuming" ;;
            22)
                case $code in
                    401 | 403) die 1 "$label: HTTP $code - access denied (set HF_TOKEN / --token-file if the repo is gated)" ;;
                    404) die 1 "$label: HTTP 404 - $url not found (catalog-refresh needed?)" ;;
                    416) die 1 "$label: HTTP 416 at offset $cur - the server's file is not the catalog's $bytes B file (catalog-refresh needed?); the .part is kept" ;;
                    *) warn "$label: HTTP ${code:-error} (curl exit 22) at $cur B" ;;
                esac
                ;;
            23) die 3 "$label: curl could not write $part (curl exit 23: disk full or I/O error)" ;;
            33 | 36) die 1 "$label: the server refused to resume at $cur B (curl exit $FM_RC): delete '$part' to restart from 0" ;;
            *) warn "$label: curl exit $FM_RC at $(fmt_gb "$cur") (HTTP ${code:-?})" ;;
        esac
        if [ "$cur" -gt "$prev" ]; then stalls=0; else stalls=$((stalls + 1)); fi
        if [ "$stalls" -ge "$FM_STALLS" ]; then
            die 1 "$label: giving up after $stalls consecutive attempts without progress ($(fmt_gb "$cur") of $(fmt_gb "$bytes") in $part); rerun to resume"
        fi
        _fm_delay "$stalls"
    done
    _dl[attempts]=$attempts
}

_fm_download_rsync() { # _fm_download_rsync ARR LABEL -> .part complete or die
    local -n _rs=$1
    local label=$2 part=${_rs[part]} bytes=${_rs[bytes]} f=${_rs[file]} src="${FM_FROM%/}" c cur prev
    local cands=() stalls=0 attempts=0 wout="$FM_TMP.w" idx=0
    mkdir -p "${part%/*}" || die 1 "cannot create ${part%/*}"
    if [ -n "${_rs[local_rel]}" ]; then cands+=("$src/${_rs[local_rel]}"); fi
    if [ -n "${_rs[repo_dir]}" ] && [ "${_rs[repo_dir]}/$f" != "${_rs[local_rel]}" ]; then cands+=("$src/${_rs[repo_dir]}/$f"); fi
    cands+=("$src/$f")
    if [ "${f##*/}" != "$f" ]; then cands+=("$src/${f##*/}"); fi
    while :; do
        cur=$(_fm_size "$part")
        if [ "$cur" -eq "$bytes" ]; then break; fi
        if [ "$cur" -gt "$bytes" ]; then die 1 "$part has $cur B, more than the catalog's $bytes B: delete it yourself and retry"; fi
        c=${cands[$idx]}
        attempts=$((attempts + 1))
        prev=$cur
        log "$label: rsync attempt $attempts from $c (resuming at $(fmt_gb "$cur"))"
        _fm_watch "$part" "$bytes" "$label" "$wout" \
            "${FM_NICE[@]+"${FM_NICE[@]}"}" rsync -aP --partial --append-verify -e "ssh $FM_SSH_OPTS" "$c" "$part"
        cur=$(_fm_size "$part")
        if [ "$FM_RC" = 0 ] && [ "$cur" -eq "$bytes" ]; then break; fi
        if [ "$FM_RC" = 0 ]; then
            die 1 "$label: rsync copied $c but it has $cur B, not the catalog's $bytes B (a different file on the desktop); the .part is kept"
        fi
        if [ "$FM_RC" = 23 ] && [ "$cur" -eq "$prev" ] && [ $((idx + 1)) -lt "${#cands[@]}" ]; then
            log "$label: $c not found on the source (rsync exit 23) - trying the next layout"
            idx=$((idx + 1))
            continue
        fi
        warn "$label: rsync exit $FM_RC at $(fmt_gb "$cur")"
        if [ "$cur" -gt "$prev" ]; then stalls=0; else stalls=$((stalls + 1)); fi
        if [ "$stalls" -ge "$FM_STALLS" ]; then
            die 1 "$label: giving up after $stalls consecutive rsync attempts without progress (last exit $FM_RC); check ssh access and that rsync exists on the desktop"
        fi
        _fm_delay "$stalls"
    done
    _rs[attempts]=$attempts
}

_fm_sha256() { # _fm_sha256 FILE -> hex
    local out
    if have_cmd sha256sum; then
        out=$("${FM_NICE[@]+"${FM_NICE[@]}"}" sha256sum -- "$1") || return 1
        out=${out#\\}
        printf '%s\n' "${out%% *}"
    else
        "${FM_NICE[@]+"${FM_NICE[@]}"}" "$PY" -c "$_FM_PY" sha256 "$(native_path "$1")"
    fi
}

_fm_stamp_ok() { # _fm_stamp_ok FILE STAMP EXPECTED -> 0 when the stamp proves FILE was verified
    local s=""
    [ -n "$3" ] && [ -f "$2" ] && [ "$2" -nt "$1" ] || return 1
    s=$(head -c 64 "$2" 2>/dev/null || true)
    [ "$s" = "$3" ]
}

_fm_verify() { # _fm_verify ARR FILE -> sha256, sha256_verified, verify_seconds; 1 on mismatch
    local -n _vf=$1
    local f=$2 exp=${_vf[sha_expected]} got t0
    _vf[verify_seconds]=0
    if [ "$FM_VERIFY" = 0 ]; then
        _vf[sha256_verified]=false
        warn "${_vf[id]}: sha256 NOT verified (--no-verify); the byte size matches the catalog"
        return 0
    fi
    t0=$(date +%s)
    log "${_vf[id]}: computing sha256 of $(fmt_gb "${_vf[bytes]}") (nice/ionice; ~$(_fm_dur "$(_fm_eta "${_vf[bytes]}" 90)") at microSD speed)"
    got=$(_fm_sha256 "$f") || die 1 "sha256 of $f failed"
    _vf[sha256]=$got
    _vf[verify_seconds]=$(($(date +%s) - t0))
    if [ -z "$exp" ]; then
        _vf[sha256_verified]=false
        warn "${_vf[id]}: no expected sha256 (catalog has none and the server sent no X-Linked-ETag): recorded $got UNVERIFIED"
        return 0
    fi
    if [ "$got" != "$exp" ]; then
        _vf[sha256_verified]=false
        return 1
    fi
    _vf[sha256_verified]=true
    log "${_vf[id]}: sha256 OK ($got, source ${_vf[sha_source]})"
}

_fm_write_stamp() { # _fm_write_stamp ARR
    local -n _st=$1
    if [ "${_st[sha256_verified]:-false}" = true ]; then
        printf '%s  %s\n' "${_st[sha256]}" "${_st[final]##*/}" >"${_st[stamp]}" ||
            warn "cannot write ${_st[stamp]} (the next run re-hashes)"
    fi
}

_fm_process() { # _fm_process ARR LABEL -> fetch / verify one item; fills the result keys
    local -n _it=$1
    local label=$2 t0 t1
    _it[sha256]=""
    _it[sha256_verified]=false
    _it[downloaded_bytes]=0
    _it[attempts]=0
    _it[seconds]=0
    _it[avg_mbps]=""
    _it[sha_expected]=${_it[sha256_cat]}
    _it[sha_source]=""
    if [ -n "${_it[sha256_cat]}" ]; then
        _it[sha_source]=catalog
    elif [ -n "${_it[remote_sha]:-}" ]; then
        _it[sha_expected]=${_it[remote_sha]}
        _it[sha_source]=x-linked-etag
    fi
    if [ "${_it[state]}" = complete ]; then
        _it[status]=present
        _it[path]=${_it[present_path]}
        _it[source]=present
        if [ "$FM_REVERIFY" = 0 ] && _fm_stamp_ok "${_it[path]}" "${_it[path]}.sha256" "${_it[sha_expected]}"; then
            _it[sha256]=${_it[sha_expected]}
            _it[sha256_verified]=true
            _it[verify_seconds]=0
            log "$label: already present and verified ($(fmt_gb "${_it[bytes]}"), ${_it[path]}; stamp ${_it[path]}.sha256)"
            return 0
        fi
        if ! _fm_verify "$1" "${_it[path]}"; then
            die 1 "$label: ${_it[path]} has the catalog size but sha256 ${_it[sha256]} != expected ${_it[sha_expected]} (${_it[sha_source]}): delete it yourself (rm -- '${_it[path]}') and rerun"
        fi
        if [ "${_it[path]}" = "${_it[final]}" ]; then _fm_write_stamp "$1"; else
            local save=${_it[stamp]}
            _it[stamp]="${_it[path]}.sha256"
            _fm_write_stamp "$1"
            _it[stamp]=$save
        fi
        log "$label: already present (${_it[path]})"
        return 0
    fi
    local start_bytes
    start_bytes=$(_fm_size "${_it[part]}")
    t0=$(date +%s)
    if [ -n "$FM_FROM" ]; then
        _it[source]="desktop:$FM_FROM"
        _it[status]=copied
        _fm_download_rsync "$1" "$label"
    else
        _it[source]=huggingface
        _it[status]=downloaded
        _fm_download_hf "$1" "$label"
    fi
    t1=$(date +%s)
    _it[seconds]=$((t1 - t0))
    _it[downloaded_bytes]=$((${_it[bytes]} - start_bytes))
    _it[avg_mbps]=$(_fm_mbps "${_it[downloaded_bytes]}" "${_it[seconds]}")
    log "$label: transfer complete: $(fmt_gb "${_it[downloaded_bytes]}") in $(_fm_dur "${_it[seconds]}") (${_it[avg_mbps]} MB/s average, ${_it[attempts]} attempt(s)); size == catalog ${_it[bytes]} B"
    if ! _fm_verify "$1" "${_it[part]}"; then
        die 1 "$label: sha256 MISMATCH for ${_it[part]}: got ${_it[sha256]}, expected ${_it[sha_expected]} (${_it[sha_source]}). The file is kept for inspection; delete it (rm -- '${_it[part]}') and rerun"
    fi
    mv -f -- "${_it[part]}" "${_it[final]}" || die 1 "cannot rename ${_it[part]} -> ${_it[final]}"
    _it[path]=${_it[final]}
    _fm_write_stamp "$1"
    log "$label: saved ${_it[final]}"
}

# ====================================================================================== reporting
_fm_url_select() { # _fm_url_select ARR
    local -n _us=$1
    if [ "$FM_PIN_REV" = 1 ]; then
        if [ -z "${_us[url_pinned]}" ]; then die 3 "--pin-revision: catalog entry ${_us[id]} has no revision"; fi
        _us[url]=${_us[url_pinned]}
    else
        _us[url]=${_us[url_main]}
    fi
    _us[sha256_cat]=${_us[sha256]}
}

_fm_summary_lines() { # FETCH_*=value lines on stdout
    local k v src="huggingface"
    if [ -n "$FM_FROM" ]; then src="desktop:$FM_FROM"; fi
    local -a kv=(
        FETCH_DRY_RUN "$FM_DRY" FETCH_SELECTION "$FM_SELECTION" FETCH_VERDICT "$FM_VERDICT"
        FETCH_VERDICT_EXIT "$FM_VERDICT_EXIT" FETCH_MODEL_ID "${M[id]}" FETCH_FAMILY "${M[family]}"
        FETCH_QUANT "${M[quant]}" FETCH_REPO "${M[repo]}" FETCH_FILE "${M[file]}" FETCH_BYTES "${M[bytes]}"
        FETCH_SHA256 "${M[sha256_cat]}" FETCH_URL "${M[url]}" FETCH_SOURCE "$src"
        FETCH_DEST "$FM_DEST" FETCH_PATH "${M[path]:-${M[final]}}" FETCH_STATE "${M[state]}"
        FETCH_STATUS "${M[status]:-planned}" FETCH_REMAINING_BYTES "$(_fm_remaining_total)"
        FETCH_FREE_BYTES "$FM_FREE" FETCH_NEED_BYTES "$FM_NEED" FETCH_ETA_S "$FM_ETA_S" FETCH_ETA_SOURCE "$FM_ETA_SRC"
        FETCH_LLAMA_FLAGS "$FM_LLAMA_FLAGS" FETCH_PROJECTED_TOK_S "$FM_TOK_S"
        FETCH_DRAFT_ID "${D[id]:-}" FETCH_DRAFT_FILE "${D[file]:-}" FETCH_DRAFT_BYTES "${D[bytes]:-}"
        FETCH_DRAFT_SPEC_TYPE "${D[spec_type]:-}" FETCH_DRAFT_URL "${D[url]:-}"
        FETCH_DRAFT_PATH "${D[path]:-${D[final]:-}}"
    )
    if [ "$FM_DRY" = 0 ]; then
        kv+=(FETCH_DOWNLOADED_BYTES "${M[downloaded_bytes]:-0}" FETCH_SHA256_VERIFIED "${M[sha256_verified]:-false}"
            FETCH_DRAFT_SHA256_VERIFIED "${D[sha256_verified]:-}" FETCH_JSON "$FM_JSON")
    fi
    local i
    for ((i = 0; i < ${#kv[@]}; i += 2)); do
        k=${kv[$i]}
        v=${kv[$((i + 1))]}
        printf '%s=%q\n' "$k" "$v"
    done
}

_fm_plan() { # human-readable plan on stderr
    local lines=() rem
    rem=$(_fm_remaining_total)
    lines+=("selection : $FM_SELECTION${FM_VERDICT:+ - verdict $FM_VERDICT (exit $FM_VERDICT_EXIT), projected ${FM_TOK_S:-?} tok/s}")
    lines+=("model     : ${M[id]} (${M[quant]}, family ${M[family]}) $(fmt_gb "${M[bytes]}") = ${M[bytes]} B")
    lines+=("source    : ${FM_FROM:+rsync $FM_FROM (LAN)}${FM_FROM:-${M[url]}}")
    lines+=("sha256    : ${M[sha256_cat]:-none in the catalog (X-Linked-ETag used when the server sends it)}")
    lines+=("dest      : ${M[final]} (state ${M[state]}${M[present_path]:+ at ${M[present_path]}}, partial $(fmt_gb "${M[part_bytes]}"))")
    if [ "$FM_DRAFT" = 1 ]; then
        lines+=("draft     : ${D[id]} (${D[spec_type]}) $(fmt_gb "${D[bytes]}") -> ${D[final]} (state ${D[state]})")
        lines+=("draft url : ${FM_FROM:-${D[url]}}")
    fi
    lines+=("disk      : remaining $(fmt_gb "$rem"), free $(fmt_gb "$FM_FREE"), need $(fmt_gb "$FM_NEED") (factor $MIN_FREE_DISK_FACTOR, keep-free $(fmt_gb "$FM_KEEP")) - $FM_DISK_SRC")
    lines+=("ETA       : $(_fm_dur "$FM_ETA_S") for $(fmt_gb "$rem") at ${FM_ETA_MBPS:-?} MB/s ($FM_ETA_SRC)")
    if [ -n "$FM_LLAMA_FLAGS" ]; then lines+=("flags     : $FM_LLAMA_FLAGS (pi/05_run.sh applies the mlock safety rule)"); fi
    local title="pi/04_fetch_model.sh $FM_VERSION plan"
    if [ "$FM_DRY" = 1 ]; then title="$title (DRY RUN: nothing is downloaded or written)"; fi
    banner "$title" "${lines[@]}"
}

FM_ETA_S=""
FM_ETA_SRC=""
FM_ETA_MBPS=""
_fm_eta_compute() { # FM_ETA_* from the probe, HF_LIMIT_RATE or the assumption
    local rem lim mbps src
    rem=$(_fm_remaining_total)
    if [ -n "$FM_FROM" ]; then
        FM_ETA_S=""
        FM_ETA_MBPS=""
        FM_ETA_SRC="n/a (LAN rsync from $FM_FROM)"
        return 0
    fi
    if [ -n "$FM_PROBE_MBPS" ]; then
        mbps=$FM_PROBE_MBPS
        src="measured: ${FM_PROBE_BYTES} B ranged probe"
    else
        mbps=$FM_ASSUMED_MBPS
        src=$FM_ASSUMED_SOURCE
    fi
    lim=$(_fm_rate_mbps "$HF_LIMIT_RATE")
    if [ -n "$lim" ] && awk -v l="$lim" -v m="$mbps" 'BEGIN { exit !(l < m) }'; then
        mbps=$lim
        src="$src, capped by HF_LIMIT_RATE=$HF_LIMIT_RATE"
    fi
    FM_ETA_MBPS=$mbps
    FM_ETA_SRC=$src
    FM_ETA_S=$(_fm_eta "$rem" "$mbps")
}

_fm_dry_commands() { # the commands a real run would execute, as '#' comment lines on stdout
    local a it_name
    printf '# pi/04_fetch_model.sh --dry-run: planned commands (token never shown; nothing was executed)\n'
    for it_name in M D; do
        if [ "$it_name" = D ] && [ "$FM_DRAFT" = 0 ]; then continue; fi
        local -n _dc=$it_name
        if [ "${_dc[state]}" = complete ]; then
            printf '# %s: %s is present (%s B): sha256 check only (skipped when %s.sha256 is current)\n' \
                "${_dc[id]}" "${_dc[present_path]}" "${_dc[bytes]}" "${_dc[present_path]}"
            unset -n _dc
            continue
        fi
        if [ -n "$FM_FROM" ]; then
            a=$(_fm_q nice -n "$NICE_LEVEL" rsync -aP --partial --append-verify -e "ssh $FM_SSH_OPTS" "${FM_FROM%/}/${_dc[local_rel]:-${_dc[repo_dir]}/${_dc[file]}}" "${_dc[part]}")
            printf '# %s   (then %s/%s if absent; repeated until the size is %s B)\n' "$a" "${FM_FROM%/}" "${_dc[file]}" "${_dc[bytes]}"
        else
            local auth=()
            if [ "$FM_HAVE_TOKEN" = 1 ]; then auth=(-H "@<0600 auth header file>"); fi
            a=$(_fm_q curl -sS --connect-timeout 30 "${auth[@]+"${auth[@]}"}" -I --max-time 30 "${_dc[url]}")
            printf '# %s   (X-Linked-Size must be %s, X-Linked-ETag %s)\n' "$a" "${_dc[bytes]}" "${_dc[sha256_cat]:-recorded}"
            local lim=()
            if [ -n "$HF_LIMIT_RATE" ]; then lim=(--limit-rate "$HF_LIMIT_RATE"); fi
            a=$(_fm_q nice -n "$NICE_LEVEL" ionice -c3 curl -sS --connect-timeout 30 "${auth[@]+"${auth[@]}"}" -fL -C - --speed-limit 100000 --speed-time 60 "${lim[@]+"${lim[@]}"}" -o "${_dc[part]}" "${_dc[url]}")
            printf '# %s   (repeated, each attempt resuming from the current size, until it is %s B; give up after %s attempts without progress)\n' "$a" "${_dc[bytes]}" "$FM_STALLS"
        fi
        if [ "$FM_VERIFY" = 1 ]; then
            printf '# %s   (expect %s)\n' "$(_fm_q nice -n "$NICE_LEVEL" sha256sum -- "${_dc[part]}")" "${_dc[sha256_cat]:-X-Linked-ETag}"
        fi
        printf '# %s\n' "$(_fm_q mv -f -- "${_dc[part]}" "${_dc[final]}")"
        unset -n _dc
    done
    printf '# write %s\n' "$FM_JSON"
}

_fm_item_json() { # _fm_item_json ARR PREFIX -> appends key:type=value items to FM_J
    local -n _ij=$1
    local p=$2
    FM_J+=("${p}model_id:s=${_ij[id]}" "${p}kind:s=${_ij[kind]}" "${p}quant:s=${_ij[quant]}"
        "${p}family:s=${_ij[family]}" "${p}repo:s=${_ij[repo]}" "${p}revision:sn=${_ij[revision]}"
        "${p}file:s=${_ij[file]}" "${p}path:s=${_ij[path]}" "${p}url:s=${_ij[url]}" "${p}source:s=${_ij[source]}"
        "${p}status:s=${_ij[status]}" "${p}bytes:i=${_ij[bytes]}" "${p}sha256_verified:b=${_ij[sha256_verified]}"
        "${p}sha256:sn=${_ij[sha256]}" "${p}sha256_expected:sn=${_ij[sha_expected]}"
        "${p}sha256_source:sn=${_ij[sha_source]}" "${p}seconds:i=${_ij[seconds]}"
        "${p}downloaded_bytes:i=${_ij[downloaded_bytes]}" "${p}avg_mbps:f=${_ij[avg_mbps]}"
        "${p}attempts:i=${_ij[attempts]}" "${p}verify_seconds:i=${_ij[verify_seconds]:-0}")
    if [ -n "${_ij[spec_type]}" ]; then FM_J+=("${p}spec_type:s=${_ij[spec_type]}"); fi
}

_fm_write_json() {
    local now host n emu=false ovr=""
    now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    host=$(host_name)
    # computed up front: a failing $(...) inside the array assignment below would trip set -e
    if is_emulated; then emu=true; fi
    if [ "${PIGEMMA_PROTECTED_OVERRIDE:-0}" = 1 ]; then ovr=PIGEMMA_I_KNOW; fi
    FM_J=("format_version:i=1" "tool:s=pi/04_fetch_model.sh $FM_VERSION" "created_at:s=$now" "host:s=$host"
        "dest:s=$FM_DEST" "selection:s=$FM_SELECTION" "verdict:sn=$FM_VERDICT" "verdict_exit:i=$FM_VERDICT_EXIT"
        "llama_flags:sn=$FM_LLAMA_FLAGS" "fits_repacked:b=$FM_FITS_REPACKED" "projected_tok_s:f=$FM_TOK_S"
        "projected_tok_s_kind:s=projected" "probe_mbps:f=$FM_PROBE_MBPS" "eta_s:i=$FM_ETA_S"
        "eta_source:sn=$FM_ETA_SRC" "total_seconds:i=$(($(date +%s) - FM_T_START))"
        "emulated:b=$emu" "override:sn=$ovr")
    _fm_item_json M ""
    if [ "$FM_DRAFT" = 1 ]; then
        _fm_item_json D "draft."
    else
        FM_J+=("draft:n=")
    fi
    for n in "${FM_NOTES[@]+"${FM_NOTES[@]}"}"; do FM_J+=("notes:a=$n"); done
    FM_J+=("notes:a=bytes, sha256 and timings are MEASURED on this host; verdict and projected_tok_s are PROJECTED by pigemma.feasibility (null for an explicit --quant)")
    _fm_py write-json "$(native_path "$FM_JSON")" "${FM_J[@]}" || die 1 "cannot write $FM_JSON"
    log "result: $FM_JSON"
}

# ============================================================================================== main
FM_T_START=$(date +%s)
_fm_select
_fm_url_select M
if [ "$FM_DRAFT" = 1 ]; then _fm_url_select D; fi
_fm_disk_budget
_fm_url_select M
if [ "$FM_DRAFT" = 1 ]; then _fm_url_select D; fi
_fm_setup_auth

if [ "$FM_DRY" = 1 ]; then
    _fm_eta_compute
    _fm_plan
    if [ -z "$FM_FROM" ] && ! have_cmd curl; then warn "curl is not installed: a real run would stop (sudo apt-get install -y curl)"; fi
    if [ -n "$FM_FROM" ] && ! have_cmd rsync; then warn "rsync is not installed: a real run would stop (sudo apt-get install -y rsync)"; fi
    _fm_dry_commands
    _fm_summary_lines
    log "dry run complete: nothing was downloaded or written"
    exit 0
fi

# ----------------------------------------------------------------------------------------- real run
if [ -n "$FM_FROM" ]; then
    need_cmd rsync rsync "--from-desktop copies with rsync over ssh"
    if [[ $FM_FROM == *:* ]]; then need_cmd ssh openssh-client "--from-desktop uses ssh"; fi
else
    need_cmd curl curl "model downloads"
fi
_fm_nice_prefix

# one fetch per PIGEMMA_OUT at a time (two curls on one .part would corrupt it)
FM_LOCK="$PIGEMMA_OUT/.fetch.lock"
_fm_unlock() {
    if [ -f "$FM_LOCK/pid" ] && [ "$(cat "$FM_LOCK/pid" 2>/dev/null || true)" = "$$" ]; then
        rm -rf -- "$FM_LOCK"
    fi
}
if ! mkdir "$FM_LOCK" 2>/dev/null; then
    _fm_lpid=$(cat "$FM_LOCK/pid" 2>/dev/null || true)
    if _pg_is_uint "$_fm_lpid" && kill -0 "$_fm_lpid" 2>/dev/null; then
        die 4 "another pi/04_fetch_model.sh (pid $_fm_lpid) is running (lock $FM_LOCK): wait for it - nothing was changed"
    fi
    warn "removing a stale fetch lock $FM_LOCK (pid ${_fm_lpid:-?} is not running)"
    rm -rf -- "$FM_LOCK"
    mkdir "$FM_LOCK" || die 1 "cannot create the lock $FM_LOCK"
fi
printf '%s\n' "$$" >"$FM_LOCK/pid"
add_cleanup _fm_unlock

if [ -z "$FM_FROM" ]; then
    for _fm_it in M D; do
        if [ "$_fm_it" = D ] && [ "$FM_DRAFT" = 0 ]; then continue; fi
        declare -n _fm_ref=$_fm_it
        if [ "${_fm_ref[state]}" != complete ]; then
            _fm_check_remote "$_fm_it" "${_fm_ref[id]}"
        fi
        unset -n _fm_ref
    done
    if [ "$FM_PROBE" = 1 ] && [ "$(_fm_remaining_total)" -gt "$FM_PROBE_MIN" ]; then
        if [ "${M[state]}" != complete ]; then _fm_probe "${M[url]}"; else _fm_probe "${D[url]}"; fi
    fi
fi
_fm_eta_compute
_fm_plan

# authoritative live disk check right before writing (the facts may be older than this moment)
_fm_rem_now=$(_fm_remaining_total)
if [ "$_fm_rem_now" -gt 0 ]; then
    require_free_disk "$FM_DEST" "$_fm_rem_now" "the download of ${M[id]}${D[id]:+ + ${D[id]}}" "$MIN_FREE_DISK_FACTOR"
fi
mkdir -p "$FM_DEST" || die 1 "cannot create $FM_DEST"

_fm_process M "model ${M[id]}"
if [ "$FM_DRAFT" = 1 ]; then
    _fm_process D "draft ${D[id]}"
fi
_fm_write_json
_fm_summary_lines
log "done: ${M[path]}${D[path]:+ + ${D[path]}}"
log "next: pi/05_run.sh bench --model ${M[path]}${D[path]:+ (speculative: --draft ${D[path]})}"
if [ "$FM_DEST" != "$PIGEMMA_HOME/models" ]; then
    log "note: later scripts find it with MODEL_DIR=$FM_DEST (set it in pi/pi_config.env or the environment)"
fi
exit 0
