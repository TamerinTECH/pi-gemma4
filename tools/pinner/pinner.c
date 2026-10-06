/*
 * pinner.c - hot-set pinning daemon for pi-gemma4 (tools/pinner).
 *
 * WHY
 *   With `llama.cpp -nr` (--no-repack) the whole GGUF stays a file-backed
 *   mapping (CPU_Mapped), so the kernel page cache decides which weight bytes
 *   are in RAM. On a Pi whose RAM is smaller than the model, the hot part of
 *   the model (all non-expert tensors + the most frequently routed experts, as
 *   chosen by `python -m pigemma.gguf_map hotset`) should never be evicted
 *   while cold experts stream from storage. The pinner maps the SAME file
 *   MAP_SHARED and mlock()s the hot byte ranges. mlocked page-cache pages are
 *   unevictable for every mapping of that file, llama.cpp's own included,
 *   because all mappings of a file share one page cache.
 *
 * USAGE
 *   pinner --gguf FILE --ranges hotset.ranges.txt [--verify] [--once]
 *          [--max-bytes N] [--pidfile P] [--json] [--hold-seconds S]
 *          [--no-mem-check] [--best-effort] [--no-warm] [--quiet]
 *          [--allow-protected-host] [--allow-empty]
 *   pinner --gguf FILE --ranges hotset.ranges.txt --verify-only [--json]
 *   pinner --help | --version
 *   Both "--opt value" and "--opt=value" are accepted.
 *
 * MODES
 *   daemon (default)  lock, print the summary, hold the locks until SIGTERM
 *                     (or SIGINT/SIGHUP when those are not ignored at start),
 *                     then munlockall, munmap, remove the pidfile and exit.
 *   --once            lock, print the summary, exit (the locks go away; the
 *                     pages stay in the page cache: a warm-up). With
 *                     --hold-seconds S it holds S seconds first (debug/test
 *                     aid: lets a test read VmLck of the running process).
 *   --verify-only     NOT in the manifest; added for `pi/06_pin_hotset.sh
 *                     status`. No locking and no guards: maps the file and
 *                     reports mincore() residency of the ranges, exit 0. Use
 *                     it to inspect a running daemon's hot set: a second
 *                     `--verify --once` would try to lock the whole set again
 *                     and the memory guard may refuse (the daemon's pages are
 *                     already unevictable and no longer count as available).
 *
 * METHOD
 *   1. Parse the ranges file (grammar below). Any malformed line -> exit 1
 *      naming the line number. Zero ranges -> exit 1, unless --allow-empty
 *      (added 1.1.0 for the P5 controlled-memory harness, method B: the same
 *      process, the same whole-file MAP_SHARED mapping and the same reporting
 *      as a real pin set, but nothing is locked; requested/locked bytes are 0;
 *      since 1.1.1 the daemon also HOLDS this empty set until SIGTERM, so the
 *      helper process and its mapping exist for the whole cell as in C/D).
 *   2. open(O_RDONLY), fstat. The header's size= must equal the file size
 *      (exit 1 otherwise: the ranges belong to another file); a different
 *      gguf= basename is only a warning (copies may be renamed). Every range
 *      must satisfy length > 0 and offset + length <= file size (exit 1).
 *   3. page = sysconf(_SC_PAGESIZE) (16384 on the Pi 5 kernel, 4096 in Docker
 *      and under qemu). Every range is aligned down/up to whole pages.
 *      requested_bytes = size of the UNION of the aligned ranges (a page
 *      shared by two ranges is counted once, which is also what
 *      RLIMIT_MEMLOCK and VmLck count).
 *   4. Guards, all before any page is touched (refuse = exit 3, the numbers on
 *      stderr, and a summary with "status":"refused" on stdout):
 *        0. Protected host (docs/design/FACTS.md, 2026-10-05: on the real
 *           target Pi context-friend-1 the rule is "no mlock of model
 *           weights; the page cache must stay reclaimable for the
 *           services"): the hostname is in DEFAULT_PROTECTED_HOSTS or
 *           $PIGEMMA_PROTECTED_HOSTS, or a process runs whose comm, or a
 *           '/'-separated component of one of its arguments, starts with a
 *           DEFAULT_PROTECTED_SERVICES / $PIGEMMA_PROTECTED_SERVICES pattern
 *           (defaults mirror catalog.json) -> refuse unless
 *           --allow-protected-host (only with the owner's explicit approval;
 *           deliberately a flag, not an environment variable).
 *           --verify-only locks nothing and is always allowed.
 *        a. --pidfile names a live process whose comm starts with "pinner"
 *           -> refuse (a dead or foreign pid is a stale pidfile: replaced).
 *        b. requested_bytes > --max-bytes -> refuse.
 *        c. RLIMIT_MEMLOCK soft limit < requested_bytes and no CAP_IPC_LOCK
 *           -> refuse with the fix (ulimit -l unlimited / limits.d /
 *           LimitMEMLOCK / CAP_IPC_LOCK). --best-effort turns this into a
 *           warning and locks in file order until the limit is hit.
 *        d. requested_bytes + 512 MiB > available memory -> refuse.
 *           available = min(MemAvailable from /proc/meminfo, finite cgroup
 *           headroom). cgroup headroom = memory.max - (memory.current -
 *           active_file - inactive_file) (v2), or the v1 equivalents; it is
 *           consulted because inside a `docker run --memory` container
 *           /proc/meminfo shows the HOST. --no-mem-check skips d.
 *      The guard compares the whole request (conservative): pages of the
 *      ranges that are already cached are counted in MemAvailable today and
 *      stop counting once locked, so they cost "available" memory too.
 *   5. mmap(NULL, file_size, PROT_READ, MAP_SHARED) of the whole file, and
 *      madvise(MADV_RANDOM) on THIS mapping only (keeps the pinner's own
 *      faults from reading around into cold neighbours; it has no effect on
 *      llama.cpp's mapping - readahead there is the device read_ahead_kb,
 *      set by pi/01_tune_os.sh).
 *   6. For every range in FILE ORDER (the producer writes non-expert tensors
 *      first, then experts by descending frequency, so a partial lock keeps
 *      the most valuable bytes), in 64 MiB chunks: madvise(MADV_WILLNEED),
 *      then a sequential pread() warm-up of the chunk through an 8 MiB
 *      scratch buffer (skipped when mincore() shows the chunk already
 *      resident; --no-warm disables it), then mlock(chunk). The warm-up exists
 *      because mlock() alone faults cold pages in one at a time: MADV_RANDOM
 *      on this mapping disables read-around and the kernel caps WILLNEED
 *      readahead at the device readahead window. MEASURED 2026-10-05 in
 *      Docker Desktop (WSL2 disk), cold 667.6 MiB hot set: 21.3-23.1 s
 *      (30-33 MB/s) with --no-warm vs 1.42-1.44 s (~490 MB/s) with the
 *      warm-up (two runs each).
 *      posix_fadvise(SEQUENTIAL) is applied to the pinner's own descriptor
 *      only (per open file, so llama.cpp's reads are unaffected). Errors:
 *        EPERM, or ENOMEM while VmLck + chunk > RLIMIT_MEMLOCK without
 *          CAP_IPC_LOCK  -> "rlimit": stop locking, print the fix;
 *        ENOMEM otherwise -> "memory" (out of memory, or vm.max_map_count
 *          reached by VMA splits): stop locking, print the numbers;
 *        EAGAIN -> touch every page of the chunk once and retry once; if it
 *          still fails the range counts as failed and the next one is tried;
 *        anything else -> the range counts as failed, the next one is tried.
 *      Pending SIGTERM/SIGINT/SIGHUP is checked between chunks; a stop request
 *      during locking aborts, cleans up and exits 1 ("interrupted").
 *   7. resident_bytes = mincore() residency of the union of the ranges
 *      (always measured). --verify adds resident_before_bytes (measured just
 *      before locking), per-kind totals when the file has '# kind=' comments,
 *      and a check that every locked page is resident.
 *   8. Summary on stdout (one line with --json; flushed before the daemon
 *      blocks so a redirected log shows it at once). In the holding modes the
 *      pidfile is written BEFORE locking starts, so "summary line present"
 *      implies "pidfile present", and a SIGTERM sent through the pidfile
 *      while still locking is honoured (exit 1, "interrupted").
 *
 * RANGES FILE (docs/TRACE_FORMAT.md section 9.3, normative)
 *   # pigemma-ranges v1 gguf=<basename> size=<bytes> page=<page> budget=<bytes>
 *   <offset> <length> [# kind=<kind> layer=<int|-> expert=<int|->]
 *   - ASCII, LF line endings; a trailing CR is stripped; a UTF-8 BOM is
 *     tolerated; blank lines and lines whose first non-blank char is '#' are
 *     comments.
 *   - The first non-blank line is the header when it starts with
 *     "# pigemma-ranges"; a version other than v1 is an error (exit 1). A file
 *     without the header is accepted with a warning (size not cross-checked):
 *     the header is the producer's MUST, the pinner stays lenient there.
 *     Header values are unsigned decimals; unknown keys are ignored.
 *   - Range lines: two unsigned decimal integers (no sign, no 0x, no
 *     separators) separated by spaces/tabs, then optionally a '#' comment.
 *     "kind=<word>" inside the comment labels the range (per-kind totals);
 *     the rest of the comment is ignored. Anything else is malformed.
 *   - Sorting/overlap are not relied upon; unsorted or overlapping input only
 *     produces a note/warning.
 *
 * OUTPUT
 *   stdout, text (default): one summary line of key=value pairs starting with
 *     "pinner: status=", plus one "pinner: kind=..." line per kind with
 *     --verify, plus "pinner: reason=..." when there is a reason.
 *   stdout, --json: exactly one line, a JSON object. Manifest keys (always
 *     present): requested_bytes, locked_bytes, resident_bytes, ranges, failed,
 *     page_size, memlock_limit, vm_lck_kb. Extra keys (stable): tool,
 *     version, status ("ok"|"partial"|"refused"|"interrupted"), exit_code,
 *     mode ("daemon"|"once"|"verify-only"), pid, gguf, ranges_file,
 *     file_size, header{gguf,size,page,budget}|null, segments,
 *     requested_raw_bytes (sum of the unaligned lengths),
 *     resident_before_bytes, not_attempted, errors{EPERM,ENOMEM,EAGAIN,other},
 *     first_error{line,errno,message}|null, memlock_hard, memlock_unlimited,
 *     cap_ipc_lock, mem_available_bytes, mem_available_source, max_bytes,
 *     mincore_reliable, verify ("ok"|"mismatch"|null),
 *     kinds{<kind>:{ranges,requested_bytes,locked_bytes,resident_bytes,
 *     resident_before_bytes}}|null (ranges without a kind: "(none)"),
 *     lock_s, hold_s (0 = exits now, -1 = holds until a signal, else the
 *     seconds it will hold), reason (string|null).
 *     memlock_limit / memlock_hard are bytes, or null when unlimited.
 *     resident_bytes / resident_before_bytes are null when not measured
 *     (refusal, no --verify) or when mincore is unreliable (see below).
 *     vm_lck_kb is /proc/self/status VmLck after locking (null if unreadable).
 *   locked_bytes = union of the page ranges this process locked (a range that
 *     failed half way contributes its locked prefix); vm_lck_kb is the
 *     kernel's own count and equals locked_bytes / 1024 in the normal case.
 *   stderr: notes, warnings, progress every 2 s while locking, hints.
 *
 * EXIT CODES
 *   0 every range locked (or --verify-only finished); 2 partial: some ranges
 *   failed or were not attempted (the count is printed; a daemon still holds
 *   what it locked and exits 2 after the stop signal); 3 refused (guards) or
 *   permission (RLIMIT_MEMLOCK / EPERM with nothing locked); 1 usage error,
 *   I/O error, malformed ranges file, unwritable pidfile/stdout, or a stop
 *   signal received while still locking.
 *
 * MINCORE CAVEAT
 *   Since Linux 5.0 mincore() on a file mapping reports page-cache residency
 *   only when the caller owns the file, has CAP_FOWNER, or could open it for
 *   writing; otherwise the kernel claims EVERY page is resident. The pinner
 *   checks those conditions and, when none holds, reports resident values as
 *   null with a warning ("mincore_reliable": false).
 *
 * DOCKER NOTES (measured 2026-10-05, Docker Desktop / WSL2 kernel 5.15, image
 *   pi-gemma4-build:latest): containers start with RLIMIT_MEMLOCK unlimited
 *   (soft and hard), root has CAP_FOWNER but NOT CAP_IPC_LOCK, page size 4096.
 *   `bash -c 'ulimit -l 64; pinner ...'` therefore reproduces the refusal.
 *   Pages locked inside a `--memory` capped container are charged to that
 *   container's cgroup.
 *
 * BUILD
 *   tools/CMakeLists.txt target "pinner" (Linux only):
 *     -std=c11 -D_GNU_SOURCE -Wall -Wextra -Werror -O2
 *   Stand-alone:
 *     gcc -std=c11 -D_GNU_SOURCE -Wall -Wextra -Werror -O2 -o pinner tools/pinner/pinner.c
 *   _GNU_SOURCE is defined here only when the command line did not define it
 *   (an unguarded redefinition would be a -Werror failure). libc only.
 *
 * NOT DONE ON PURPOSE
 *   - No cross-process madvise: MADV_RANDOM/MADV_SEQUENTIAL from here would
 *     not change llama.cpp's mapping.
 *   - No mlockall, no MCL_ONFAULT: the point is to fault the hot set in now
 *     and keep it, and to touch nothing else.
 */

#ifndef __linux__
#error "pinner is Linux-only: it needs mmap(MAP_SHARED), mlock, mincore and /proc"
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64 /* pread offsets beyond 2 GiB on 32-bit userlands */
#endif

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define PINNER_VERSION "1.1.1"

/* Exit codes (manifest contract). */
#define PX_OK 0
#define PX_ERROR 1
#define PX_PARTIAL 2
#define PX_REFUSED 3

#define HEADROOM_BYTES ((uint64_t)512 << 20)   /* memory guard headroom */
#define LOCK_CHUNK_BYTES ((uint64_t)64 << 20)  /* mlock granularity */
#define WARM_BUF_BYTES ((size_t)8 << 20)       /* pread warm-up buffer */
#define MINCORE_CHUNK_PAGES ((size_t)65536)    /* pages per mincore call */
#define MAX_RANGES ((size_t)50000000)
#define MAX_KINDS 64
#define KIND_LEN 48
#define GGUF_NAME_LEN 256
#define PROGRESS_EVERY_S 2.0
#define MAX_HOLD_S 31536000.0                  /* one year */
#define CGROUP_UNLIMITED ((uint64_t)1 << 60)
#define CAP_BIT_FOWNER 3
#define CAP_BIT_IPC_LOCK 14
#define HEADER_TAG "# pigemma-ranges"
/* Defaults mirror pigemma/data/catalog.json protected_hosts /
 * protected_service_globs ('contextfriend*', 'context-friend*'); the env
 * variables PIGEMMA_PROTECTED_HOSTS / PIGEMMA_PROTECTED_SERVICES (comma
 * separated) add to them. */
#define DEFAULT_PROTECTED_HOSTS "context-friend-1"
#define DEFAULT_PROTECTED_SERVICES "contextfriend,context-friend"
#define HEADER_TAG_LEN 16

/* ------------------------------------------------------------------------ */
/* Messages                                                                  */
/* ------------------------------------------------------------------------ */

static int g_quiet = 0;

static void vmsg(const char *tag, const char *fmt, va_list ap)
{
    fputs("pinner: ", stderr);
    if (tag != NULL) {
        fputs(tag, stderr);
    }
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
}

__attribute__((format(printf, 1, 2))) static void note(const char *fmt, ...)
{
    if (g_quiet) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vmsg(NULL, fmt, ap);
    va_end(ap);
}

__attribute__((format(printf, 1, 2))) static void warn_msg(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vmsg("warning: ", fmt, ap);
    va_end(ap);
}

__attribute__((format(printf, 1, 2))) static void err_msg(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vmsg("error: ", fmt, ap);
    va_end(ap);
}

__attribute__((format(printf, 1, 2))) static void hint_msg(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vmsg("hint: ", fmt, ap);
    va_end(ap);
}

__attribute__((format(printf, 1, 2))) static void refuse_msg(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vmsg("REFUSED: ", fmt, ap);
    va_end(ap);
}

static double mib(uint64_t b)
{
    return (double)b / 1048576.0;
}

static double pct(uint64_t part, uint64_t whole)
{
    return whole > 0 ? 100.0 * (double)part / (double)whole : 100.0;
}

static double now_s(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static const char *errno_name(int e)
{
    switch (e) {
    case EPERM: return "EPERM";
    case ENOMEM: return "ENOMEM";
    case EAGAIN: return "EAGAIN";
    case EINVAL: return "EINVAL";
    case EFAULT: return "EFAULT";
    case EIO: return "EIO";
    case EACCES: return "EACCES";
    case ENODEV: return "ENODEV";
    case EOVERFLOW: return "EOVERFLOW";
    default: return "other";
    }
}

/* ------------------------------------------------------------------------ */
/* Number parsing                                                            */
/* ------------------------------------------------------------------------ */

/* Unsigned decimal at *pp: digits only (no sign, no 0x, no separators).
 * Returns 1 and advances *pp; 0 when no digit is there; -1 on overflow. */
static int scan_u64(const char **pp, uint64_t *out)
{
    const char *p = *pp;
    uint64_t v = 0;
    int nd = 0;
    while (*p >= '0' && *p <= '9') {
        const uint64_t d = (uint64_t)(*p - '0');
        if (v > (UINT64_MAX - d) / 10u) {
            return -1;
        }
        v = v * 10u + d;
        p++;
        nd++;
    }
    if (nd == 0) {
        return 0;
    }
    *out = v;
    *pp = p;
    return 1;
}

static int parse_u64_full(const char *s, uint64_t *out)
{
    const char *p = s;
    return scan_u64(&p, out) == 1 && *p == '\0';
}

/* Same, for a span [s, s+n) that is not NUL-terminated. */
static int parse_u64_span(const char *s, size_t n, uint64_t *out)
{
    char buf[32];
    if (n == 0 || n >= sizeof(buf)) {
        return 0;
    }
    memcpy(buf, s, n);
    buf[n] = '\0';
    return parse_u64_full(buf, out);
}

static int parse_seconds(const char *s, double *out)
{
    if (*s == '\0' || *s == '-' || *s == '+' || *s == ' ' || *s == '\t') {
        return 0;
    }
    char *end = NULL;
    errno = 0;
    const double v = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0') {
        return 0;
    }
    if (!(v >= 0.0 && v <= MAX_HOLD_S)) { /* also rejects NaN and inf */
        return 0;
    }
    *out = v;
    return 1;
}

/* ------------------------------------------------------------------------ */
/* /proc and cgroup readers                                                  */
/* ------------------------------------------------------------------------ */

/* First number after "<key>:" or "<key> " at the start of a line of a
 * /proc-style file. *is_kb (optional) is set when "kB" follows it. */
static int read_keyed_u64(const char *path, const char *key, uint64_t *out, int *is_kb)
{
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return 0;
    }
    char line[512];
    const size_t klen = strlen(key);
    int found = 0;
    while (fgets(line, sizeof(line), f) != NULL) {
        if (strncmp(line, key, klen) != 0) {
            continue;
        }
        const char *p = line + klen;
        if (*p == ':') {
            p++;
        }
        if (*p != ' ' && *p != '\t') {
            continue; /* prefix of a longer key */
        }
        while (*p == ' ' || *p == '\t') {
            p++;
        }
        uint64_t v = 0;
        if (scan_u64(&p, &v) != 1) {
            continue;
        }
        while (*p == ' ' || *p == '\t') {
            p++;
        }
        if (is_kb != NULL) {
            *is_kb = strncmp(p, "kB", 2) == 0;
        }
        *out = v;
        found = 1;
        break;
    }
    fclose(f);
    return found;
}

/* A file holding one unsigned number ("max" and garbage -> 0). */
static int read_single_u64(const char *path, uint64_t *out)
{
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return 0;
    }
    char buf[64];
    const char *got = fgets(buf, sizeof(buf), f);
    fclose(f);
    if (got == NULL) {
        return 0;
    }
    const char *p = buf;
    return scan_u64(&p, out) == 1;
}

static int read_vmlck_kb(uint64_t *kb)
{
    return read_keyed_u64("/proc/self/status", "VmLck", kb, NULL);
}

static int read_cap_eff(uint64_t *caps)
{
    FILE *f = fopen("/proc/self/status", "r");
    if (f == NULL) {
        return 0;
    }
    char line[256];
    int found = 0;
    while (fgets(line, sizeof(line), f) != NULL) {
        if (strncmp(line, "CapEff:", 7) == 0) {
            char *end = NULL;
            errno = 0;
            const unsigned long long v = strtoull(line + 7, &end, 16);
            if (errno == 0 && end != line + 7) {
                *caps = (uint64_t)v;
                found = 1;
            }
            break;
        }
    }
    fclose(f);
    return found;
}

/* Headroom of one cgroup directory for page-cache pinning:
 *   limit - (usage - active_file - inactive_file)
 * i.e. file-backed LRU pages count as reclaimable (some of them may be the
 * very pages about to be pinned); anon, shmem, kernel and already-locked
 * (unevictable) memory count as used.
 *   v2: memory.max, memory.current, memory.stat active_file/inactive_file
 *   v1: memory.limit_in_bytes, memory.usage_in_bytes,
 *       memory.stat total_active_file/total_inactive_file
 * Returns 1 when a finite limit was found there. */
static int cgroup_dir_headroom(const char *dir, int v1, uint64_t *avail, char *label,
                               size_t label_len)
{
    const char *f_max = v1 ? "memory.limit_in_bytes" : "memory.max";
    const char *f_cur = v1 ? "memory.usage_in_bytes" : "memory.current";
    const char *k_act = v1 ? "total_active_file" : "active_file";
    const char *k_inact = v1 ? "total_inactive_file" : "inactive_file";
    char p_max[700], p_cur[700], p_stat[700];
    const int w1 = snprintf(p_max, sizeof(p_max), "%s/%s", dir, f_max);
    const int w2 = snprintf(p_cur, sizeof(p_cur), "%s/%s", dir, f_cur);
    const int w3 = snprintf(p_stat, sizeof(p_stat), "%s/memory.stat", dir);
    if (w1 < 0 || w2 < 0 || w3 < 0 || (size_t)w1 >= sizeof(p_max) ||
        (size_t)w2 >= sizeof(p_cur) || (size_t)w3 >= sizeof(p_stat)) {
        return 0;
    }
    uint64_t mx = 0, cur = 0, act = 0, inact = 0;
    if (!read_single_u64(p_max, &mx) || mx >= CGROUP_UNLIMITED ||
        !read_single_u64(p_cur, &cur)) {
        return 0;
    }
    if (!read_keyed_u64(p_stat, k_act, &act, NULL)) {
        act = 0;
    }
    if (!read_keyed_u64(p_stat, k_inact, &inact, NULL)) {
        inact = 0;
    }
    const uint64_t reclaimable = act + inact;
    const uint64_t used = cur > reclaimable ? cur - reclaimable : 0;
    *avail = mx > used ? mx - used : 0;
    if (snprintf(label, label_len, "cgroup %s", p_max) < 0 && label_len > 0) {
        label[0] = '\0';
    }
    return 1;
}

static int has_token(const char *list, const char *tok)
{
    const size_t tl = strlen(tok);
    const char *p = list;
    while (*p != '\0') {
        const size_t n = strcspn(p, ",");
        if (n == tl && strncmp(p, tok, tl) == 0) {
            return 1;
        }
        p += n;
        if (*p == ',') {
            p++;
        }
    }
    return 0;
}

/* Smallest finite cgroup headroom of this process: its own cgroup (from
 * /proc/self/cgroup) and the mount roots (which ARE the container's cgroup
 * under a private cgroup namespace); v2 and the v1 memory controller. */
static int cgroup_headroom(uint64_t *avail, char *where, size_t where_len)
{
    char v2path[512] = "";
    char v1path[512] = "";
    FILE *f = fopen("/proc/self/cgroup", "r");
    if (f != NULL) {
        char line[1024];
        while (fgets(line, sizeof(line), f) != NULL) {
            line[strcspn(line, "\r\n")] = '\0';
            char *c1 = strchr(line, ':');
            char *c2 = c1 != NULL ? strchr(c1 + 1, ':') : NULL;
            if (c2 == NULL) {
                continue;
            }
            *c2 = '\0';
            const char *ctrls = c1 + 1;
            const char *path = c2 + 1;
            char *dst = NULL;
            if (c1 == line + 1 && line[0] == '0' && ctrls[0] == '\0') {
                dst = v2path;
            } else if (has_token(ctrls, "memory")) {
                dst = v1path;
            }
            const size_t n = strlen(path);
            if (dst != NULL && n < sizeof(v2path)) {
                memcpy(dst, path, n + 1u);
            }
        }
        fclose(f);
    }
    char own2[600] = "";
    char own1[600] = "";
    if (v2path[0] == '/' && v2path[1] != '\0') {
        const int wr = snprintf(own2, sizeof(own2), "/sys/fs/cgroup%s", v2path);
        if (wr < 0 || (size_t)wr >= sizeof(own2)) {
            own2[0] = '\0';
        }
    }
    if (v1path[0] == '/' && v1path[1] != '\0') {
        const int wr = snprintf(own1, sizeof(own1), "/sys/fs/cgroup/memory%s", v1path);
        if (wr < 0 || (size_t)wr >= sizeof(own1)) {
            own1[0] = '\0';
        }
    }
    const char *dirs[4] = {own2, "/sys/fs/cgroup", own1, "/sys/fs/cgroup/memory"};
    const int is_v1[4] = {0, 0, 1, 1};
    int have = 0;
    uint64_t best = 0;
    for (unsigned i = 0; i < 4u; ++i) {
        if (dirs[i][0] == '\0') {
            continue;
        }
        uint64_t v = 0;
        char label[720] = "";
        if (cgroup_dir_headroom(dirs[i], is_v1[i], &v, label, sizeof(label))) {
            if (!have || v < best) {
                best = v;
                if (snprintf(where, where_len, "%s", label) < 0 && where_len > 0) {
                    where[0] = '\0';
                }
            }
            have = 1;
        }
    }
    *avail = best;
    return have;
}

typedef struct {
    int have;           /* any source known */
    uint64_t avail;     /* min of the known sources, bytes */
    int have_meminfo;
    uint64_t meminfo;   /* /proc/meminfo MemAvailable, bytes */
    int have_cgroup;
    uint64_t cgroup;    /* cgroup headroom, bytes */
    char where[760];    /* which source binds */
} memavail_t;

static void memory_available(memavail_t *m)
{
    memset(m, 0, sizeof(*m));
    uint64_t v = 0;
    int is_kb = 0;
    if (read_keyed_u64("/proc/meminfo", "MemAvailable", &v, &is_kb)) {
        m->have_meminfo = 1;
        m->meminfo = is_kb ? v * 1024u : v;
        m->have = 1;
        m->avail = m->meminfo;
        if (snprintf(m->where, sizeof(m->where), "/proc/meminfo MemAvailable") < 0) {
            m->where[0] = '\0';
        }
    }
    char cg_where[720] = "";
    if (cgroup_headroom(&v, cg_where, sizeof(cg_where))) {
        m->have_cgroup = 1;
        m->cgroup = v;
        if (!m->have || v < m->avail) {
            m->avail = v;
            if (snprintf(m->where, sizeof(m->where), "%s headroom", cg_where) < 0) {
                m->where[0] = '\0';
            }
        }
        m->have = 1;
    }
}

/* ------------------------------------------------------------------------ */
/* Limits and capabilities                                                   */
/* ------------------------------------------------------------------------ */

typedef struct {
    int have_rl;
    struct rlimit rl;
    int have_caps;
    int cap_ipc_lock;
    int cap_fowner;
} limits_t;

static void read_limits(limits_t *L)
{
    memset(L, 0, sizeof(*L));
    if (getrlimit(RLIMIT_MEMLOCK, &L->rl) == 0) {
        L->have_rl = 1;
    }
    uint64_t caps = 0;
    if (read_cap_eff(&caps)) {
        L->have_caps = 1;
        L->cap_ipc_lock = (int)((caps >> CAP_BIT_IPC_LOCK) & 1u);
        L->cap_fowner = (int)((caps >> CAP_BIT_FOWNER) & 1u);
    }
}

/* 1 when RLIMIT_MEMLOCK bounds this process (finite soft limit, no
 * CAP_IPC_LOCK). */
static int rlimit_binds(const limits_t *L)
{
    return L->have_rl && L->rl.rlim_cur != RLIM_INFINITY && !L->cap_ipc_lock;
}

static void fmt_rlim(rlim_t v, char *buf, size_t n)
{
    int w;
    if (v == RLIM_INFINITY) {
        w = snprintf(buf, n, "unlimited");
    } else {
        w = snprintf(buf, n, "%llu B (%llu KiB)", (unsigned long long)v,
                     (unsigned long long)(v >> 10));
    }
    if (w < 0 && n > 0) {
        buf[0] = '\0';
    }
}

static void print_rlimit_hint(const limits_t *L, uint64_t requested)
{
    static int printed = 0;
    if (printed) {
        return; /* once per run is enough */
    }
    printed = 1;
    char soft[64] = "unknown", hard[64] = "unknown";
    if (L->have_rl) {
        fmt_rlim(L->rl.rlim_cur, soft, sizeof(soft));
        fmt_rlim(L->rl.rlim_max, hard, sizeof(hard));
    }
    hint_msg("RLIMIT_MEMLOCK soft=%s hard=%s is below the requested %" PRIu64
             " B (%.1f MiB) and the process lacks CAP_IPC_LOCK",
             soft, hard, requested, mib(requested));
    hint_msg("fix one of: (1) 'ulimit -l unlimited' in the launching shell (only works "
             "when the hard limit allows it);");
    hint_msg("  (2) /etc/security/limits.d/90-pigemma.conf with '<user> - memlock "
             "unlimited', then log in again (pi/01_tune_os.sh --apply installs it);");
    hint_msg("  (3) systemd unit: LimitMEMLOCK=infinity;");
    hint_msg("  (4) CAP_IPC_LOCK: 'sudo setcap cap_ipc_lock+ep <pinner>', or in Docker "
             "'--cap-add IPC_LOCK' / '--ulimit memlock=-1:-1'.");
}

/* ------------------------------------------------------------------------ */
/* Protected hosts (docs/design/FACTS.md safety rules)                       */
/* ------------------------------------------------------------------------ */

/* Case-insensitive: does the comma-separated `list` contain `word` exactly? */
static int list_has_word(const char *list, const char *word)
{
    const size_t wl = strlen(word);
    const char *p = list;
    while (p != NULL && *p != '\0') {
        while (*p == ',' || *p == ' ') {
            p++;
        }
        const size_t n = strcspn(p, ",");
        size_t m = n;
        while (m > 0 && p[m - 1] == ' ') {
            m--;
        }
        if (m == wl && m > 0) {
            size_t i = 0;
            while (i < m && tolower((unsigned char)p[i]) == tolower((unsigned char)word[i])) {
                i++;
            }
            if (i == m) {
                return 1;
            }
        }
        p += n;
    }
    return 0;
}

/* Case-insensitive: does s (sl bytes) start with pat (pl bytes)? */
static int prefix_ci(const char *s, size_t sl, const char *pat, size_t pl)
{
    if (pl == 0 || pl > sl) {
        return 0;
    }
    for (size_t k = 0; k < pl; ++k) {
        if (tolower((unsigned char)s[k]) != tolower((unsigned char)pat[k])) {
            return 0;
        }
    }
    return 1;
}

/* Does a component of buf start with a pattern of the comma-separated list?
 * buf holds NUL-separated arguments (/proc/<pid>/cmdline) or one string
 * (comm); a component starts at the beginning of an argument or right after
 * a '/', so "/usr/bin/contextfriend-box", "python3 -m contextfriend.app" and
 * "/opt/context-friend/main.py" match while a shell string that merely
 * mentions the name ("bash -c 'grep contextfriend log'") does not. A
 * trailing '*' of a glob is dropped ('contextfriend*' -> 'contextfriend').
 * The matched pattern is copied to hit. */
static int buf_matches_list(const char *buf, size_t n, const char *list, char *hit, size_t hit_len)
{
    const char *p = list;
    while (p != NULL && *p != '\0') {
        while (*p == ',' || *p == ' ') {
            p++;
        }
        size_t m = strcspn(p, ",");
        const size_t adv = m;
        while (m > 0 && (p[m - 1] == ' ' || p[m - 1] == '*')) {
            m--;
        }
        if (m > 0) {
            for (size_t i = 0; i < n; ++i) {
                if (i > 0 && buf[i - 1] != '\0' && buf[i - 1] != '/') {
                    continue;
                }
                if (prefix_ci(buf + i, n - i, p, m)) {
                    if (hit_len > 0) {
                        const size_t cl = m < hit_len - 1u ? m : hit_len - 1u;
                        memcpy(hit, p, cl);
                        hit[cl] = '\0';
                    }
                    return 1;
                }
            }
        }
        p += adv;
    }
    return 0;
}

/* 1 when this host is protected: its hostname is listed, or a process runs
 * whose comm, or a path component of one of its arguments, starts with a
 * protected service pattern (see buf_matches_list).
 * `why` gets a human-readable reason. Kernel threads (empty cmdline) and the
 * pinner itself are ignored. Inside a container only that container's
 * processes are visible, which is the intended scope. */
static int protected_host(char *why, size_t why_len)
{
    const char *env_hosts = getenv("PIGEMMA_PROTECTED_HOSTS");
    const char *env_svcs = getenv("PIGEMMA_PROTECTED_SERVICES");
    char host[256] = "";
    if (gethostname(host, sizeof(host) - 1u) == 0) {
        host[sizeof(host) - 1u] = '\0';
        if (list_has_word(DEFAULT_PROTECTED_HOSTS, host) ||
            (env_hosts != NULL && list_has_word(env_hosts, host))) {
            if (snprintf(why, why_len, "hostname '%.200s' is a protected host", host) < 0 &&
                why_len > 0) {
                why[0] = '\0';
            }
            return 1;
        }
    }
    DIR *d = opendir("/proc");
    if (d == NULL) {
        return 0;
    }
    const long self = (long)getpid();
    int found = 0;
    struct dirent *de;
    while (!found && (de = readdir(d)) != NULL) {
        const char *nm = de->d_name;
        if (nm[0] < '1' || nm[0] > '9') {
            continue;
        }
        char *end = NULL;
        const long pid = strtol(nm, &end, 10);
        if (end == NULL || *end != '\0' || pid == self) {
            continue;
        }
        char path[64];
        char buf[4096];
        size_t n = 0;
        const int w = snprintf(path, sizeof(path), "/proc/%ld/cmdline", pid);
        if (w < 0 || (size_t)w >= sizeof(path)) {
            continue;
        }
        FILE *f = fopen(path, "r");
        if (f == NULL) {
            continue; /* exited meanwhile, or not visible */
        }
        n = fread(buf, 1, sizeof(buf), f);
        fclose(f);
        if (n == 0) {
            continue; /* kernel thread */
        }
        char hit[128] = "";
        int match = buf_matches_list(buf, n, DEFAULT_PROTECTED_SERVICES, hit, sizeof(hit)) ||
                    (env_svcs != NULL && buf_matches_list(buf, n, env_svcs, hit, sizeof(hit)));
        if (!match) {
            char comm[64] = "";
            const int w2 = snprintf(path, sizeof(path), "/proc/%ld/comm", pid);
            if (w2 > 0 && (size_t)w2 < sizeof(path)) {
                FILE *fc = fopen(path, "r");
                if (fc != NULL) {
                    const size_t cn = fread(comm, 1, sizeof(comm) - 1u, fc);
                    fclose(fc);
                    comm[cn] = '\0';
                    match = buf_matches_list(comm, cn, DEFAULT_PROTECTED_SERVICES, hit, sizeof(hit)) ||
                            (env_svcs != NULL &&
                             buf_matches_list(comm, cn, env_svcs, hit, sizeof(hit)));
                }
            }
        }
        if (match) {
            for (size_t i = 0; i < n; ++i) {
                if (buf[i] == '\0') {
                    buf[i] = ' ';
                }
            }
            buf[n < sizeof(buf) ? n : sizeof(buf) - 1u] = '\0';
            if (snprintf(why, why_len, "protected service pattern '%s' is running (pid %ld: %.120s)",
                         hit, pid, buf) < 0 &&
                why_len > 0) {
                why[0] = '\0';
            }
            found = 1;
        }
    }
    closedir(d);
    return found;
}

/* ------------------------------------------------------------------------ */
/* Ranges                                                                    */
/* ------------------------------------------------------------------------ */

enum { RS_PENDING = 0, RS_LOCKED, RS_FAILED, RS_SKIPPED };
enum { FC_NONE = 0, FC_RLIMIT, FC_MEMORY, FC_TRANSIENT, FC_OTHER };

typedef struct {
    uint64_t off;   /* as written in the file */
    uint64_t len;
    uint64_t a0;    /* page-aligned [a0, a1) */
    uint64_t a1;
    uint64_t lk1;   /* [a0, lk1) is mlocked by us; lk1 == a0 when nothing */
    unsigned line;
    int kind;       /* index into kinds[], -1 = no kind comment */
    int state;
    int err;
} range_t;

typedef struct {
    int present;
    char gguf[GGUF_NAME_LEN];
    int has_size, has_page, has_budget;
    uint64_t size, page, budget;
    unsigned line;
} header_t;

typedef struct {
    range_t *r;
    size_t n;
    size_t cap;
    char kinds[MAX_KINDS][KIND_LEN];
    int nkinds;
    int any_kind;
    header_t hdr;
    uint64_t page;
    uint64_t file_size;
    unsigned char *base;  /* the MAP_SHARED mapping, NULL before mmap */
    int fd;               /* O_RDONLY descriptor of the file (pread warm-up) */
    int mincore_ok;
} pin_t;

static int kind_index(pin_t *P, const char *s, size_t n)
{
    if (n >= KIND_LEN) {
        n = KIND_LEN - 1;
    }
    for (int i = 0; i < P->nkinds; ++i) {
        if (strlen(P->kinds[i]) == n && strncmp(P->kinds[i], s, n) == 0) {
            return i;
        }
    }
    if (P->nkinds < MAX_KINDS - 1) {
        memcpy(P->kinds[P->nkinds], s, n);
        P->kinds[P->nkinds][n] = '\0';
        return P->nkinds++;
    }
    /* Too many distinct kinds: the rest share one bucket. */
    if (P->nkinds == MAX_KINDS - 1) {
        memcpy(P->kinds[MAX_KINDS - 1], "(other)", 8);
        P->nkinds = MAX_KINDS;
    }
    return MAX_KINDS - 1;
}

/* Finds "kind=<word>" at a word start inside a range-line comment. */
static int extract_kind(pin_t *P, const char *c)
{
    const char *q = c;
    while (*q != '\0') {
        while (*q == ' ' || *q == '\t' || *q == '#') {
            q++;
        }
        const char *t = q;
        while (*q != '\0' && *q != ' ' && *q != '\t') {
            q++;
        }
        const size_t tl = (size_t)(q - t);
        if (tl > 5 && strncmp(t, "kind=", 5) == 0) {
            P->any_kind = 1;
            return kind_index(P, t + 5, tl - 5);
        }
    }
    return -1;
}

/* Header: "# pigemma-ranges v1 gguf=.. size=.. page=.. budget=..".
 * Returns 1 when the line is a header, 0 when it is an ordinary comment,
 * -1 on error (message printed). */
static int parse_header(const char *p, header_t *h, const char *path, unsigned ln)
{
    if (strncmp(p, HEADER_TAG, HEADER_TAG_LEN) != 0) {
        return 0;
    }
    const char *q = p + HEADER_TAG_LEN;
    if (strncmp(q, " v1", 3) != 0 || (q[3] != '\0' && q[3] != ' ' && q[3] != '\t')) {
        err_msg("%s:%u: unsupported ranges header (expected '# pigemma-ranges v1 ...'): %.120s",
                path, ln, p);
        return -1;
    }
    q += 3;
    h->present = 1;
    h->line = ln;
    for (;;) {
        while (*q == ' ' || *q == '\t') {
            q++;
        }
        if (*q == '\0') {
            break;
        }
        const char *t = q;
        while (*q != '\0' && *q != ' ' && *q != '\t') {
            q++;
        }
        const char *eq = memchr(t, '=', (size_t)(q - t));
        if (eq == NULL) {
            err_msg("%s:%u: malformed header token '%.*s' (expected key=value)", path, ln,
                    (int)(q - t), t);
            return -1;
        }
        const size_t kl = (size_t)(eq - t);
        const char *v = eq + 1;
        const size_t vl = (size_t)(q - v);
        if (kl == 4 && strncmp(t, "gguf", 4) == 0) {
            const size_t cl = vl < GGUF_NAME_LEN - 1 ? vl : GGUF_NAME_LEN - 1;
            memcpy(h->gguf, v, cl);
            h->gguf[cl] = '\0';
        } else if ((kl == 4 && strncmp(t, "size", 4) == 0) ||
                   (kl == 4 && strncmp(t, "page", 4) == 0) ||
                   (kl == 6 && strncmp(t, "budget", 6) == 0)) {
            uint64_t num = 0;
            if (!parse_u64_span(v, vl, &num)) {
                err_msg("%s:%u: header value '%.*s' is not an unsigned decimal", path, ln,
                        (int)(q - t), t);
                return -1;
            }
            if (t[0] == 's') {
                h->size = num;
                h->has_size = 1;
            } else if (t[0] == 'p') {
                h->page = num;
                h->has_page = 1;
            } else {
                h->budget = num;
                h->has_budget = 1;
            }
        } /* unknown keys: ignored (forward compatible) */
    }
    return 1;
}

static int append_range(pin_t *P, uint64_t off, uint64_t len, int kind, unsigned ln)
{
    if (P->n >= MAX_RANGES) {
        err_msg("too many ranges (limit %zu)", MAX_RANGES);
        return -1;
    }
    if (P->n == P->cap) {
        const size_t nc = P->cap == 0 ? 1024u : P->cap * 2u;
        range_t *nr = realloc(P->r, nc * sizeof(*nr));
        if (nr == NULL) {
            err_msg("out of memory reading ranges");
            return -1;
        }
        P->r = nr;
        P->cap = nc;
    }
    range_t *r = &P->r[P->n++];
    memset(r, 0, sizeof(*r));
    r->off = off;
    r->len = len;
    r->kind = kind;
    r->line = ln;
    r->state = RS_PENDING;
    return 0;
}

static void malformed(const char *path, unsigned ln, const char *why, const char *line)
{
    err_msg("%s:%u: malformed range line (%s; expected '<offset> <length> "
            "[# kind=<kind> layer=<l> expert=<e>]'): '%.120s'",
            path, ln, why, line);
}

/* Returns 0 on success, -1 on error (printed). */
static int parse_ranges(const char *path, pin_t *P)
{
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        err_msg("cannot open ranges file %s: %s", path, strerror(errno));
        return -1;
    }
    char *line = NULL;
    size_t cap = 0;
    ssize_t got;
    unsigned ln = 0;
    int seen_nonblank = 0;
    int rc = 0;
    while ((got = getline(&line, &cap, f)) != -1) {
        ln++;
        size_t n = (size_t)got;
        if (n > 0 && line[n - 1] == '\n') {
            line[--n] = '\0';
        }
        if (n > 0 && line[n - 1] == '\r') {
            line[--n] = '\0';
        }
        if (strlen(line) != n) {
            malformed(path, ln, "NUL byte in line", line);
            rc = -1;
            break;
        }
        const char *p = line;
        if (ln == 1 && n >= 3 && (unsigned char)p[0] == 0xEFu && (unsigned char)p[1] == 0xBBu &&
            (unsigned char)p[2] == 0xBFu) {
            p += 3; /* UTF-8 BOM */
        }
        while (*p == ' ' || *p == '\t') {
            p++;
        }
        if (*p == '\0') {
            continue;
        }
        const int first = !seen_nonblank;
        seen_nonblank = 1;
        if (*p == '#') {
            if (first) {
                if (parse_header(p, &P->hdr, path, ln) < 0) {
                    rc = -1;
                    break;
                }
            } else if (strncmp(p, HEADER_TAG " ", HEADER_TAG_LEN + 1) == 0) {
                warn_msg("%s:%u: '# pigemma-ranges' header not on the first line; ignored",
                         path, ln);
            }
            continue;
        }
        uint64_t off = 0, len = 0;
        int s = scan_u64(&p, &off);
        if (s != 1) {
            malformed(path, ln,
                      s < 0 ? "offset does not fit 64 bits" : "offset is not an unsigned decimal",
                      line);
            rc = -1;
            break;
        }
        if (*p != ' ' && *p != '\t') {
            malformed(path, ln, "offset must be followed by whitespace and a length", line);
            rc = -1;
            break;
        }
        while (*p == ' ' || *p == '\t') {
            p++;
        }
        s = scan_u64(&p, &len);
        if (s != 1) {
            malformed(path, ln,
                      s < 0 ? "length does not fit 64 bits" : "length is not an unsigned decimal",
                      line);
            rc = -1;
            break;
        }
        if (*p != '\0' && *p != ' ' && *p != '\t' && *p != '#') {
            malformed(path, ln, "unexpected characters after the length", line);
            rc = -1;
            break;
        }
        while (*p == ' ' || *p == '\t') {
            p++;
        }
        int kind = -1;
        if (*p == '#') {
            kind = extract_kind(P, p + 1);
        } else if (*p != '\0') {
            malformed(path, ln, "a third field must be a '#' comment", line);
            rc = -1;
            break;
        }
        if (len == 0) {
            malformed(path, ln, "length must be > 0", line);
            rc = -1;
            break;
        }
        if (off > UINT64_MAX - len) {
            malformed(path, ln, "offset + length overflows 64 bits", line);
            rc = -1;
            break;
        }
        if (append_range(P, off, len, kind, ln) != 0) {
            rc = -1;
            break;
        }
    }
    if (rc == 0 && ferror(f)) {
        err_msg("error reading %s: %s", path, strerror(errno));
        rc = -1;
    }
    free(line);
    fclose(f);
    return rc;
}

/* ------------------------------------------------------------------------ */
/* Unions of page ranges and mincore                                         */
/* ------------------------------------------------------------------------ */

typedef struct {
    uint64_t s;
    uint64_t e;
} ival_t;

static int cmp_ival(const void *a, const void *b)
{
    const ival_t *x = a;
    const ival_t *y = b;
    if (x->s != y->s) {
        return x->s < y->s ? -1 : 1;
    }
    if (x->e != y->e) {
        return x->e < y->e ? -1 : 1;
    }
    return 0;
}

/* Selector: 1 and [*s, *e) when range r belongs to the set. */
typedef int (*sel_fn)(const range_t *r, int arg, uint64_t *s, uint64_t *e);

static int sel_all(const range_t *r, int arg, uint64_t *s, uint64_t *e)
{
    (void)arg;
    *s = r->a0;
    *e = r->a1;
    return 1;
}

static int sel_kind(const range_t *r, int arg, uint64_t *s, uint64_t *e)
{
    if (r->kind != arg) {
        return 0;
    }
    return sel_all(r, arg, s, e);
}

static int sel_locked(const range_t *r, int arg, uint64_t *s, uint64_t *e)
{
    (void)arg;
    if (r->lk1 <= r->a0) {
        return 0;
    }
    *s = r->a0;
    *e = r->lk1;
    return 1;
}

static int sel_locked_kind(const range_t *r, int arg, uint64_t *s, uint64_t *e)
{
    if (r->kind != arg) {
        return 0;
    }
    return sel_locked(r, arg, s, e);
}

static unsigned char g_vec[MINCORE_CHUNK_PAGES];

/* Resident bytes of the page range [s, e) of the mapping. */
static int mincore_bytes(const pin_t *P, uint64_t s, uint64_t e, uint64_t *res)
{
    uint64_t total = 0;
    const uint64_t chunk = (uint64_t)MINCORE_CHUNK_PAGES * P->page;
    for (uint64_t pos = s; pos < e; pos += chunk) {
        const uint64_t len = (e - pos) < chunk ? (e - pos) : chunk;
        const size_t npages = (size_t)((len + P->page - 1u) / P->page);
        int rc = 0;
        int tries = 0;
        do {
            rc = mincore(P->base + pos, (size_t)len, g_vec);
        } while (rc != 0 && errno == EAGAIN && ++tries < 5);
        if (rc != 0) {
            return -1;
        }
        for (size_t i = 0; i < npages; ++i) {
            if (g_vec[i] & 1u) {
                total += P->page;
            }
        }
    }
    *res = total;
    return 0;
}

/* Union of the selected page ranges: total bytes, merged segment count and
 * (optionally) resident bytes. Returns 0, or -1 with errno set. */
static int union_stats(const pin_t *P, sel_fn sel, int arg, int want_res, uint64_t *bytes,
                       uint64_t *nseg, uint64_t *resident)
{
    *bytes = 0;
    if (nseg != NULL) {
        *nseg = 0;
    }
    if (resident != NULL) {
        *resident = 0;
    }
    if (P->n == 0) {
        return 0;
    }
    ival_t *iv = malloc(P->n * sizeof(*iv));
    if (iv == NULL) {
        errno = ENOMEM;
        return -1;
    }
    size_t m = 0;
    for (size_t i = 0; i < P->n; ++i) {
        uint64_t s = 0, e = 0;
        if (sel(&P->r[i], arg, &s, &e) && e > s) {
            iv[m].s = s;
            iv[m].e = e;
            m++;
        }
    }
    qsort(iv, m, sizeof(*iv), cmp_ival);
    uint64_t total = 0, segs = 0, res = 0;
    size_t i = 0;
    while (i < m) {
        const uint64_t cs = iv[i].s;
        uint64_t ce = iv[i].e;
        i++;
        while (i < m && iv[i].s <= ce) {
            if (iv[i].e > ce) {
                ce = iv[i].e;
            }
            i++;
        }
        total += ce - cs;
        segs++;
        if (want_res) {
            uint64_t r = 0;
            if (mincore_bytes(P, cs, ce, &r) != 0) {
                const int e = errno;
                free(iv);
                errno = e;
                return -1;
            }
            res += r;
        }
    }
    free(iv);
    *bytes = total;
    if (nseg != NULL) {
        *nseg = segs;
    }
    if (resident != NULL) {
        *resident = res;
    }
    return 0;
}

/* Number of ranges whose ORIGINAL bytes overlap another range, and whether
 * the file order is sorted by offset. */
static size_t count_overlaps(const pin_t *P, int *sorted)
{
    *sorted = 1;
    for (size_t i = 1; i < P->n; ++i) {
        if (P->r[i].off < P->r[i - 1].off) {
            *sorted = 0;
            break;
        }
    }
    ival_t *iv = malloc(P->n * sizeof(*iv));
    if (iv == NULL) {
        return 0;
    }
    for (size_t i = 0; i < P->n; ++i) {
        iv[i].s = P->r[i].off;
        iv[i].e = P->r[i].off + P->r[i].len;
    }
    qsort(iv, P->n, sizeof(*iv), cmp_ival);
    size_t ov = 0;
    uint64_t maxe = 0;
    for (size_t i = 0; i < P->n; ++i) {
        if (i > 0 && iv[i].s < maxe) {
            ov++;
        }
        if (iv[i].e > maxe) {
            maxe = iv[i].e;
        }
    }
    free(iv);
    return ov;
}

/* ------------------------------------------------------------------------ */
/* Pidfile                                                                   */
/* ------------------------------------------------------------------------ */

static long read_pidfile(const char *path)
{
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return -1;
    }
    char buf[64];
    const char *got = fgets(buf, sizeof(buf), f);
    fclose(f);
    if (got == NULL) {
        return -1;
    }
    const char *p = buf;
    while (*p == ' ' || *p == '\t') {
        p++;
    }
    uint64_t v = 0;
    if (scan_u64(&p, &v) != 1 || v == 0 || v > (uint64_t)INT_MAX) {
        return -1;
    }
    return (long)v;
}

/* 1 when pid is alive and is (or may be) a pinner: comm starts with "pinner"
 * (an unreadable comm counts as yes, to be safe). */
static int pid_is_live_pinner(long pid)
{
    if (kill((pid_t)pid, 0) != 0 && errno != EPERM) {
        return 0;
    }
    char path[64];
    const int w = snprintf(path, sizeof(path), "/proc/%ld/comm", pid);
    if (w < 0 || (size_t)w >= sizeof(path)) {
        return 1;
    }
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return 1; /* alive, comm unreadable: be conservative */
    }
    char comm[64];
    const char *got = fgets(comm, sizeof(comm), f);
    fclose(f);
    if (got == NULL) {
        return 1;
    }
    comm[strcspn(comm, "\r\n")] = '\0';
    return strncmp(comm, "pinner", 6) == 0; /* also pinner_asan, pinner-x, ... */
}

/* Atomic write (tmp + rename) of "<pid>\n". */
static int pidfile_write(const char *path)
{
    char tmp[4200];
    const int w = snprintf(tmp, sizeof(tmp), "%s.tmp.%ld", path, (long)getpid());
    if (w < 0 || (size_t)w >= sizeof(tmp)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    const int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        return -1;
    }
    char buf[32];
    const int n = snprintf(buf, sizeof(buf), "%ld\n", (long)getpid());
    int saved = 0;
    if (n < 0 || (size_t)n >= sizeof(buf)) {
        saved = EINVAL;
    } else {
        const ssize_t wr = write(fd, buf, (size_t)n);
        if (wr != (ssize_t)n) {
            saved = wr < 0 ? errno : EIO;
        }
    }
    if (close(fd) != 0 && saved == 0) {
        saved = errno;
    }
    if (saved == 0 && rename(tmp, path) != 0) {
        saved = errno;
    }
    if (saved != 0) {
        if (unlink(tmp) != 0) {
            /* the original error is what gets reported */
        }
        errno = saved;
        return -1;
    }
    return 0;
}

static void pidfile_remove(const char *path)
{
    if (read_pidfile(path) != (long)getpid()) {
        return; /* not ours (any more): leave it alone */
    }
    if (unlink(path) != 0) {
        warn_msg("cannot remove pidfile %s: %s", path, strerror(errno));
    }
}

/* ------------------------------------------------------------------------ */
/* Signals                                                                   */
/* ------------------------------------------------------------------------ */

static const int k_stop_signals[3] = {SIGTERM, SIGINT, SIGHUP};

static const char *sig_name(int s)
{
    if (s == SIGTERM) {
        return "SIGTERM";
    }
    if (s == SIGINT) {
        return "SIGINT";
    }
    if (s == SIGHUP) {
        return "SIGHUP";
    }
    return "a signal";
}

/* Builds the set of stop signals and blocks it. SIGINT/SIGHUP are left alone
 * when they are ignored at start (nohup, background jobs of a
 * non-interactive shell): blocking an ignored signal would make sigwait
 * deliver it and defeat the caller's intent. */
static int setup_signals(sigset_t *ws)
{
    if (sigemptyset(ws) != 0) {
        return -1;
    }
    for (unsigned i = 0; i < 3u; ++i) {
        const int s = k_stop_signals[i];
        if (s != SIGTERM) {
            struct sigaction old;
            if (sigaction(s, NULL, &old) == 0 && old.sa_handler == SIG_IGN) {
                continue;
            }
        }
        if (sigaddset(ws, s) != 0) {
            return -1;
        }
    }
    return sigprocmask(SIG_BLOCK, ws, NULL);
}

static int pending_stop(const sigset_t *ws)
{
    sigset_t pend;
    if (sigpending(&pend) != 0) {
        return 0;
    }
    for (unsigned i = 0; i < 3u; ++i) {
        const int s = k_stop_signals[i];
        if (sigismember(ws, s) == 1 && sigismember(&pend, s) == 1) {
            return s;
        }
    }
    return 0;
}

/* Waits for a stop signal: forever when hold_s <= 0, else at most hold_s.
 * Returns the signal, 0 on timeout, -1 on error. */
static int wait_stop(const sigset_t *ws, double hold_s)
{
    if (hold_s <= 0.0) {
        for (;;) {
            int sig = 0;
            const int rc = sigwait(ws, &sig);
            if (rc == 0) {
                return sig;
            }
            if (rc != EINTR) {
                errno = rc;
                return -1;
            }
        }
    }
    const double deadline = now_s() + hold_s;
    for (;;) {
        const double left = deadline - now_s();
        if (left <= 0.0) {
            return 0;
        }
        struct timespec ts;
        ts.tv_sec = (time_t)left;
        long ns = (long)((left - (double)ts.tv_sec) * 1e9);
        if (ns < 0) {
            ns = 0;
        }
        if (ns > 999999999L) {
            ns = 999999999L;
        }
        ts.tv_nsec = ns;
        const int sig = sigtimedwait(ws, NULL, &ts);
        if (sig > 0) {
            return sig;
        }
        if (errno == EAGAIN) {
            return 0;
        }
        if (errno != EINTR) {
            return -1;
        }
    }
}

/* ------------------------------------------------------------------------ */
/* Locking                                                                   */
/* ------------------------------------------------------------------------ */

typedef struct {
    int interrupted;
    int stop_signal;
    unsigned failed;     /* ranges attempted that failed */
    unsigned skipped;    /* ranges not (fully) attempted */
    unsigned n_eperm, n_enomem, n_eagain, n_other;
    int rlimit_hit;
    int memory_hit;
    unsigned first_line;
    int first_errno;
    double lock_s;
} lockres_t;

static int classify_mlock_errno(int e, const limits_t *L, uint64_t chunk_len)
{
    if (e == EPERM) {
        return FC_RLIMIT;
    }
    if (e == ENOMEM) {
        if (rlimit_binds(L)) {
            uint64_t kb = 0;
            const uint64_t lim = (uint64_t)L->rl.rlim_cur;
            if (!read_vmlck_kb(&kb) || kb * 1024u + chunk_len > lim) {
                return FC_RLIMIT;
            }
        }
        return FC_MEMORY;
    }
    if (e == EAGAIN) {
        return FC_TRANSIENT;
    }
    return FC_OTHER;
}

/* Reads one byte per page so every page of [s, s+len) is faulted in. */
static void touch_pages(const pin_t *P, uint64_t s, uint64_t len)
{
    const volatile unsigned char *b = P->base;
    unsigned char acc = 0;
    for (uint64_t off = 0; off < len; off += P->page) {
        acc = (unsigned char)(acc ^ b[s + off]);
    }
    volatile unsigned char sink = acc;
    (void)sink;
}

static uint64_t min_u64(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}

/* Brings [pos, pos+len) of the file into the page cache with large
 * sequential pread()s, unless mincore() says it is all resident already.
 * Without this, mlock() faults the pages in one at a time: MADV_RANDOM on
 * our mapping disables fault read-around, and MADV_WILLNEED / fadvise
 * readahead is capped by the kernel at the device readahead window.
 * MEASURED 2026-10-05 (Docker Desktop, overlayfs on the WSL2 disk), cold
 * 667.6 MiB: 30-33 MB/s without this vs ~490 MB/s with it. Returns 0, or -1
 * with errno set (the caller warns and lets mlock fault the pages in). */
static int warm_chunk(const pin_t *P, uint64_t pos, uint64_t len, unsigned char *buf,
                      size_t bufsz)
{
    if (P->mincore_ok) {
        uint64_t res = 0;
        if (mincore_bytes(P, pos, pos + len, &res) == 0 && res >= len) {
            return 0; /* already cached: nothing to read */
        }
    }
    uint64_t off = pos;
    const uint64_t end = (pos + len) < P->file_size ? (pos + len) : P->file_size;
    while (off < end) {
        const uint64_t left = end - off;
        const size_t want = (size_t)(left < (uint64_t)bufsz ? left : (uint64_t)bufsz);
        const ssize_t got = pread(P->fd, buf, want, (off_t)off);
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (got == 0) {
            break; /* file shrank under us; mlock will report it */
        }
        off += (uint64_t)got;
    }
    return 0;
}

static void lock_ranges(pin_t *P, const limits_t *L, const sigset_t *ws, int warm, lockres_t *R)
{
    memset(R, 0, sizeof(*R));
    uint64_t total = 0;
    for (size_t i = 0; i < P->n; ++i) {
        total += P->r[i].a1 - P->r[i].a0;
    }
    const double t0 = now_s();
    double last = t0;
    uint64_t done = 0;
    int stop = 0;
    int madv_warned = 0;
    int warm_warned = 0;
    unsigned char *buf = NULL;
    if (warm) {
        /* Per open file description only: does not touch llama.cpp's file. */
        const int fe = posix_fadvise(P->fd, 0, 0, POSIX_FADV_SEQUENTIAL);
        if (fe != 0) {
            note("posix_fadvise(SEQUENTIAL) on the pinner's own fd failed: %s (harmless)",
                 strerror(fe));
        }
        buf = malloc(WARM_BUF_BYTES);
        if (buf == NULL) {
            warn_msg("no memory for the %zu MiB warm-up buffer; mlock will fault pages in "
                     "one at a time (slower)",
                     WARM_BUF_BYTES >> 20);
        }
    }
    for (size_t i = 0; i < P->n; ++i) {
        range_t *r = &P->r[i];
        r->lk1 = r->a0;
        if (stop) {
            r->state = RS_SKIPPED;
            R->skipped++;
            continue;
        }
        uint64_t pos = r->a0;
        int failed = 0;
        while (pos < r->a1) {
            const uint64_t clen = min_u64(r->a1 - pos, LOCK_CHUNK_BYTES);
            const uint64_t npos = pos + clen;
            if (madvise(P->base + pos, (size_t)clen, MADV_WILLNEED) != 0 && !madv_warned) {
                warn_msg("madvise(MADV_WILLNEED) failed: %s (continuing)", strerror(errno));
                madv_warned = 1;
            }
            if (buf != NULL && warm_chunk(P, pos, clen, buf, WARM_BUF_BYTES) != 0 &&
                !warm_warned) {
                warn_msg("pread warm-up failed at offset %" PRIu64 ": %s (mlock faults the pages "
                         "in instead)",
                         pos, strerror(errno));
                warm_warned = 1;
            }
            int rc = mlock(P->base + pos, (size_t)clen);
            if (rc != 0 && errno == EAGAIN) {
                touch_pages(P, pos, clen);
                rc = mlock(P->base + pos, (size_t)clen);
            }
            if (rc != 0) {
                const int e = errno;
                const int cls = classify_mlock_errno(e, L, clen);
                failed = 1;
                r->err = e;
                if (e == EPERM) {
                    R->n_eperm++;
                } else if (e == ENOMEM) {
                    R->n_enomem++;
                } else if (e == EAGAIN) {
                    R->n_eagain++;
                } else {
                    R->n_other++;
                }
                if (R->first_line == 0) {
                    R->first_line = r->line;
                    R->first_errno = e;
                }
                if (cls == FC_RLIMIT) {
                    R->rlimit_hit = 1;
                    stop = 1;
                } else if (cls == FC_MEMORY) {
                    R->memory_hit = 1;
                    stop = 1;
                }
                if (R->failed < 5u) {
                    warn_msg("mlock failed for the range at line %u (offset %" PRIu64 ", %" PRIu64
                             " of %" PRIu64 " B locked): %s (%s)%s",
                             r->line, r->off, pos - r->a0, r->a1 - r->a0, strerror(e),
                             errno_name(e), stop ? "; stopping" : "; trying the next range");
                }
                break;
            }
            pos = npos;
            r->lk1 = pos;
            done += clen;
            const double t = now_s();
            if (t - last >= PROGRESS_EVERY_S) {
                note("locking: %.1f / %.1f MiB (%.1f%%) after %.1f s", mib(done), mib(total),
                     pct(done, total), t - t0);
                last = t;
            }
            const int sig = pending_stop(ws);
            if (sig != 0) {
                R->interrupted = 1;
                R->stop_signal = sig;
                stop = 1;
                break;
            }
        }
        if (failed) {
            r->state = RS_FAILED;
            R->failed++;
        } else if (pos >= r->a1) {
            r->state = RS_LOCKED;
        } else {
            r->state = RS_SKIPPED; /* interrupted half way */
            R->skipped++;
        }
    }
    free(buf);
    R->lock_s = now_s() - t0;
}

static void print_memory_hint(uint64_t nseg)
{
    memavail_t m;
    memory_available(&m);
    if (m.have) {
        hint_msg("the kernel could not fault in / lock more pages (ENOMEM); available now %" PRIu64
                 " B (%.1f MiB) per %s",
                 m.avail, mib(m.avail), m.where);
    } else {
        hint_msg("the kernel could not fault in / lock more pages (ENOMEM)");
    }
    uint64_t mmc = 0;
    if (read_single_u64("/proc/sys/vm/max_map_count", &mmc)) {
        hint_msg("each locked segment can cost up to 2 VMAs: %" PRIu64
                 " merged segments vs vm.max_map_count %" PRIu64,
                 nseg, mmc);
    }
    hint_msg("lower the hot-set budget (python -m pigemma.gguf_map hotset --budget-bytes ...) "
             "or free memory; what was locked stays locked (exit 2)");
}

/* ------------------------------------------------------------------------ */
/* CLI                                                                       */
/* ------------------------------------------------------------------------ */

typedef struct {
    const char *gguf;
    const char *ranges;
    const char *pidfile;
    int verify;
    int verify_only;
    int once;
    int json;
    int no_mem_check;
    int no_warm;
    int allow_protected;
    int best_effort;
    int allow_empty;
    int quiet;
    int have_max;
    uint64_t max_bytes;
    double hold_s;
} opts_t;

static void usage(FILE *to)
{
    fprintf(to,
            "usage: pinner --gguf FILE --ranges hotset.ranges.txt [--verify] [--once]\n"
            "              [--max-bytes N] [--pidfile P] [--json] [--hold-seconds S]\n"
            "              [--no-mem-check] [--best-effort] [--no-warm] [--quiet]\n"
            "              [--allow-protected-host] [--allow-empty]\n"
            "       pinner --gguf FILE --ranges hotset.ranges.txt --verify-only [--json]\n"
            "       pinner --help | --version\n"
            "\n"
            "Hot-set pinning daemon (pi-gemma4). Maps FILE MAP_SHARED and mlock()s the\n"
            "page-aligned byte ranges of the ranges file (written by `python -m\n"
            "pigemma.gguf_map hotset`), so the page cache keeps them resident for every\n"
            "mapping of FILE (llama.cpp -nr included) while cold experts stream.\n"
            "\n"
            "  --gguf FILE        model file to pin (any regular file works)\n"
            "  --ranges F         '# pigemma-ranges v1 ...' header, then '<offset> <length>'\n"
            "                     per line (decimal), optional '# kind=<k> layer=<l> expert=<e>'\n"
            "  --verify           also measure residency before locking, per-kind totals,\n"
            "                     and check that every locked page is resident (mincore)\n"
            "  --verify-only      no locking, no guards: report mincore residency, exit 0\n"
            "  --once             lock, report, exit (locks released: a warm-up)\n"
            "  --hold-seconds S   with --once: hold the locks S seconds before exiting;\n"
            "                     as a daemon: stop after S seconds even without a signal\n"
            "  --max-bytes N      refuse (exit 3) if the page-aligned union exceeds N bytes\n"
            "  --pidfile P        write the pid to P while holding (refuse if P names a\n"
            "                     live pinner); removed on exit\n"
            "  --json             one-line JSON summary on stdout\n"
            "  --no-mem-check     skip the MemAvailable/cgroup guard (requested + 512 MiB)\n"
            "  --no-warm          diagnostic: no sequential pread() warm-up before mlock\n"
            "                     (mlock then faults cold pages in one at a time)\n"
            "  --allow-protected-host  lock even on a protected host (context-friend-1, or\n"
            "                     a contextfriend*/context-friend* process running); ONLY with\n"
            "                     the owner's explicit approval (FACTS.md: no mlock of model\n"
            "                     weights there). --verify-only never needs it.\n"
            "  --best-effort      do not refuse when RLIMIT_MEMLOCK is too small: lock in\n"
            "                     file order until the limit is hit (exit 2)\n"
            "  --allow-empty      accept a ranges file without range lines: map the file,\n"
            "                     lock nothing, report and hold as usual (control arm)\n"
            "  --quiet            no notes/progress on stderr (warnings and errors stay)\n"
            "\n"
            "Daemon mode (no --once): holds until SIGTERM (or SIGINT/SIGHUP unless ignored).\n"
            "Exit: 0 all locked, 2 partial, 3 refused/permission (RLIMIT_MEMLOCK: fix with\n"
            "'ulimit -l unlimited', /etc/security/limits.d/90-pigemma.conf or CAP_IPC_LOCK),\n"
            "1 usage/IO error or stopped while still locking. Diagnostics go to stderr.\n");
}

enum {
    O_GGUF = 1,
    O_RANGES,
    O_VERIFY,
    O_VERIFY_ONLY,
    O_ONCE,
    O_MAX,
    O_PIDFILE,
    O_JSON,
    O_HOLD,
    O_NOMEM,
    O_NOWARM,
    O_ALLOW_PROT,
    O_BEST,
    O_ALLOW_EMPTY,
    O_QUIET,
    O_HELP,
    O_VERSION
};

typedef struct {
    const char *name;
    int has_arg;
    int id;
} optdef_t;

static const optdef_t k_opts[] = {
    {"--gguf", 1, O_GGUF},
    {"--ranges", 1, O_RANGES},
    {"--verify", 0, O_VERIFY},
    {"--verify-only", 0, O_VERIFY_ONLY},
    {"--once", 0, O_ONCE},
    {"--max-bytes", 1, O_MAX},
    {"--pidfile", 1, O_PIDFILE},
    {"--json", 0, O_JSON},
    {"--hold-seconds", 1, O_HOLD},
    {"--no-mem-check", 0, O_NOMEM},
    {"--no-warm", 0, O_NOWARM},
    {"--allow-protected-host", 0, O_ALLOW_PROT},
    {"--best-effort", 0, O_BEST},
    {"--allow-empty", 0, O_ALLOW_EMPTY},
    {"--quiet", 0, O_QUIET},
    {"--help", 0, O_HELP},
    {"-h", 0, O_HELP},
    {"--version", 0, O_VERSION},
};

/* Returns 0 ok, 1 usage error (printed), -1 when --help/--version ran. */
static int parse_args(int argc, char **argv, opts_t *o)
{
    memset(o, 0, sizeof(*o));
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        const char *val = NULL;
        size_t nlen = strlen(a);
        if (strncmp(a, "--", 2) == 0) {
            const char *eq = strchr(a, '=');
            if (eq != NULL) {
                nlen = (size_t)(eq - a);
                val = eq + 1;
            }
        }
        const optdef_t *d = NULL;
        for (size_t k = 0; k < sizeof(k_opts) / sizeof(k_opts[0]); ++k) {
            if (strlen(k_opts[k].name) == nlen && strncmp(k_opts[k].name, a, nlen) == 0) {
                d = &k_opts[k];
                break;
            }
        }
        if (d == NULL) {
            fprintf(stderr, "pinner: error: unknown argument '%s' (see --help)\n", a);
            return 1;
        }
        if (!d->has_arg && val != NULL) {
            fprintf(stderr, "pinner: error: %s takes no value\n", d->name);
            return 1;
        }
        if (d->has_arg && val == NULL) {
            if (i + 1 >= argc) {
                fprintf(stderr, "pinner: error: %s needs a value (see --help)\n", d->name);
                return 1;
            }
            val = argv[++i];
        }
        switch (d->id) {
        case O_GGUF:
            o->gguf = val;
            break;
        case O_RANGES:
            o->ranges = val;
            break;
        case O_VERIFY:
            o->verify = 1;
            break;
        case O_VERIFY_ONLY:
            o->verify_only = 1;
            break;
        case O_ONCE:
            o->once = 1;
            break;
        case O_MAX:
            if (!parse_u64_full(val, &o->max_bytes)) {
                fprintf(stderr,
                        "pinner: error: --max-bytes wants an unsigned decimal byte count, got "
                        "'%s'\n",
                        val);
                return 1;
            }
            o->have_max = 1;
            break;
        case O_PIDFILE:
            o->pidfile = val;
            break;
        case O_JSON:
            o->json = 1;
            break;
        case O_HOLD:
            if (!parse_seconds(val, &o->hold_s)) {
                fprintf(stderr, "pinner: error: --hold-seconds wants 0..%.0f seconds, got '%s'\n",
                        MAX_HOLD_S, val);
                return 1;
            }
            break;
        case O_NOMEM:
            o->no_mem_check = 1;
            break;
        case O_NOWARM:
            o->no_warm = 1;
            break;
        case O_ALLOW_PROT:
            o->allow_protected = 1;
            break;
        case O_BEST:
            o->best_effort = 1;
            break;
        case O_ALLOW_EMPTY:
            o->allow_empty = 1;
            break;
        case O_QUIET:
            o->quiet = 1;
            break;
        case O_HELP:
            usage(stdout);
            return -1;
        case O_VERSION:
            printf("pinner %s\n", PINNER_VERSION);
            return -1;
        default:
            return 1;
        }
    }
    if (o->gguf == NULL || o->gguf[0] == '\0' || o->ranges == NULL || o->ranges[0] == '\0') {
        fprintf(stderr, "pinner: error: --gguf FILE and --ranges FILE are required (see --help)\n");
        return 1;
    }
    if (o->pidfile != NULL && o->pidfile[0] == '\0') {
        fprintf(stderr, "pinner: error: --pidfile needs a non-empty path\n");
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Summary                                                                   */
/* ------------------------------------------------------------------------ */

typedef struct {
    uint64_t ranges;
    uint64_t requested;
    uint64_t locked;
    uint64_t resident;
    uint64_t resident_before;
} kstat_t;

#define KST_NONE MAX_KINDS /* kst[] slot of ranges without a kind comment */

typedef struct {
    const char *status;
    int exit_code;
    const char *mode;
    const char *reason;
    uint64_t requested;
    uint64_t requested_raw;
    uint64_t segments;
    uint64_t locked;
    int have_resident;
    uint64_t resident;
    int have_resident_before;
    uint64_t resident_before;
    unsigned failed;
    unsigned skipped;
    int have_vmlck;
    uint64_t vmlck_kb;
    memavail_t mem;
    int mem_known;
    int verify_state;   /* -1 n/a, 0 mismatch, 1 ok */
    int have_kinds;
    kstat_t kst[MAX_KINDS + 1];
    double lock_s;
    double hold_s;
    const lockres_t *lr;
} summary_t;

static void json_str(FILE *f, const char *s)
{
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p != '\0'; ++p) {
        if (*p == '"' || *p == '\\') {
            fputc('\\', f);
            fputc(*p, f);
        } else if (*p < 0x20u) {
            fprintf(f, "\\u%04x", (unsigned)*p);
        } else {
            fputc(*p, f);
        }
    }
    fputc('"', f);
}

static void json_u64_or_null(FILE *f, int have, uint64_t v)
{
    if (have) {
        fprintf(f, "%" PRIu64, v);
    } else {
        fputs("null", f);
    }
}

static void json_rlim(FILE *f, int have, rlim_t v)
{
    if (!have || v == RLIM_INFINITY) {
        fputs("null", f);
    } else {
        fprintf(f, "%llu", (unsigned long long)v);
    }
}

static const char *kind_name(const pin_t *P, int slot)
{
    return slot == KST_NONE ? "(none)" : P->kinds[slot];
}

static void print_json(FILE *f, const pin_t *P, const opts_t *o, const limits_t *L,
                       const summary_t *S)
{
    fprintf(f, "{\"tool\":\"pinner\",\"version\":\"%s\",\"status\":", PINNER_VERSION);
    json_str(f, S->status);
    fprintf(f, ",\"exit_code\":%d,\"mode\":", S->exit_code);
    json_str(f, S->mode);
    fprintf(f, ",\"pid\":%ld,\"gguf\":", (long)getpid());
    json_str(f, o->gguf);
    fputs(",\"ranges_file\":", f);
    json_str(f, o->ranges);
    fprintf(f, ",\"file_size\":%" PRIu64 ",\"header\":", P->file_size);
    if (P->hdr.present) {
        fputs("{\"gguf\":", f);
        if (P->hdr.gguf[0] != '\0') {
            json_str(f, P->hdr.gguf);
        } else {
            fputs("null", f);
        }
        fputs(",\"size\":", f);
        json_u64_or_null(f, P->hdr.has_size, P->hdr.size);
        fputs(",\"page\":", f);
        json_u64_or_null(f, P->hdr.has_page, P->hdr.page);
        fputs(",\"budget\":", f);
        json_u64_or_null(f, P->hdr.has_budget, P->hdr.budget);
        fputc('}', f);
    } else {
        fputs("null", f);
    }
    fprintf(f,
            ",\"page_size\":%" PRIu64 ",\"ranges\":%zu,\"segments\":%" PRIu64
            ",\"requested_bytes\":%" PRIu64 ",\"requested_raw_bytes\":%" PRIu64
            ",\"locked_bytes\":%" PRIu64 ",\"resident_bytes\":",
            P->page, P->n, S->segments, S->requested, S->requested_raw, S->locked);
    json_u64_or_null(f, S->have_resident, S->resident);
    fputs(",\"resident_before_bytes\":", f);
    json_u64_or_null(f, S->have_resident_before, S->resident_before);
    fprintf(f, ",\"failed\":%u,\"not_attempted\":%u", S->failed, S->skipped);
    const lockres_t *R = S->lr;
    fprintf(f, ",\"errors\":{\"EPERM\":%u,\"ENOMEM\":%u,\"EAGAIN\":%u,\"other\":%u}",
            R ? R->n_eperm : 0u, R ? R->n_enomem : 0u, R ? R->n_eagain : 0u,
            R ? R->n_other : 0u);
    fputs(",\"first_error\":", f);
    if (R != NULL && R->first_line != 0) {
        fprintf(f, "{\"line\":%u,\"errno\":", R->first_line);
        json_str(f, errno_name(R->first_errno));
        fputs(",\"message\":", f);
        json_str(f, strerror(R->first_errno));
        fputc('}', f);
    } else {
        fputs("null", f);
    }
    fputs(",\"memlock_limit\":", f);
    json_rlim(f, L->have_rl, L->rl.rlim_cur);
    fputs(",\"memlock_hard\":", f);
    json_rlim(f, L->have_rl, L->rl.rlim_max);
    fprintf(f, ",\"memlock_unlimited\":%s,\"cap_ipc_lock\":%s,\"vm_lck_kb\":",
            (L->have_rl && L->rl.rlim_cur == RLIM_INFINITY) ? "true" : "false",
            L->cap_ipc_lock ? "true" : "false");
    json_u64_or_null(f, S->have_vmlck, S->vmlck_kb);
    fputs(",\"mem_available_bytes\":", f);
    json_u64_or_null(f, S->mem_known && S->mem.have, S->mem.avail);
    fputs(",\"mem_available_source\":", f);
    if (S->mem_known && S->mem.have) {
        json_str(f, S->mem.where);
    } else {
        fputs("null", f);
    }
    fputs(",\"max_bytes\":", f);
    json_u64_or_null(f, o->have_max, o->max_bytes);
    fprintf(f, ",\"mincore_reliable\":%s,\"verify\":", P->mincore_ok ? "true" : "false");
    if (S->verify_state < 0) {
        fputs("null", f);
    } else {
        json_str(f, S->verify_state ? "ok" : "mismatch");
    }
    fputs(",\"kinds\":", f);
    if (S->have_kinds) {
        fputc('{', f);
        int first = 1;
        for (int k = 0; k <= KST_NONE; ++k) {
            const kstat_t *ks = &S->kst[k];
            if (ks->ranges == 0) {
                continue;
            }
            if (!first) {
                fputc(',', f);
            }
            first = 0;
            json_str(f, kind_name(P, k));
            fprintf(f,
                    ":{\"ranges\":%" PRIu64 ",\"requested_bytes\":%" PRIu64
                    ",\"locked_bytes\":%" PRIu64 ",\"resident_bytes\":",
                    ks->ranges, ks->requested, ks->locked);
            json_u64_or_null(f, S->have_resident, ks->resident);
            fputs(",\"resident_before_bytes\":", f);
            json_u64_or_null(f, S->have_resident_before, ks->resident_before);
            fputc('}', f);
        }
        fputc('}', f);
    } else {
        fputs("null", f);
    }
    fprintf(f, ",\"lock_s\":%.3f,\"hold_s\":%.3f,\"reason\":", S->lock_s, S->hold_s);
    if (S->reason != NULL) {
        json_str(f, S->reason);
    } else {
        fputs("null", f);
    }
    fputs("}\n", f);
}

static void u64_or_null_str(char *buf, size_t n, int have, uint64_t v)
{
    const int w = have ? snprintf(buf, n, "%" PRIu64, v) : snprintf(buf, n, "null");
    if (w < 0 && n > 0) {
        buf[0] = '\0';
    }
}

static void pct_str(char *buf, size_t n, int have, uint64_t part, uint64_t whole)
{
    /* one decimal, printed from integer tenths (bounded output length) */
    const unsigned long long tenths = (unsigned long long)(pct(part, whole) * 10.0 + 0.5);
    const int w = have ? snprintf(buf, n, "%llu.%llu", tenths / 10u, tenths % 10u)
                       : snprintf(buf, n, "n/a");
    if (w < 0 && n > 0) {
        buf[0] = '\0';
    }
}

static void print_text(FILE *f, const pin_t *P, const limits_t *L, const summary_t *S)
{
    char lim[32];
    if (!L->have_rl) {
        if (snprintf(lim, sizeof(lim), "unknown") < 0) {
            lim[0] = '\0';
        }
    } else if (L->rl.rlim_cur == RLIM_INFINITY) {
        if (snprintf(lim, sizeof(lim), "unlimited") < 0) {
            lim[0] = '\0';
        }
    } else if (snprintf(lim, sizeof(lim), "%llu", (unsigned long long)L->rl.rlim_cur) < 0) {
        lim[0] = '\0';
    }
    char res[32], vml[32], lpct[32], rpct[32];
    u64_or_null_str(res, sizeof(res), S->have_resident, S->resident);
    u64_or_null_str(vml, sizeof(vml), S->have_vmlck, S->vmlck_kb);
    pct_str(lpct, sizeof(lpct), 1, S->locked, S->requested);
    pct_str(rpct, sizeof(rpct), S->have_resident, S->resident, S->requested);
    fprintf(f,
            "pinner: status=%s exit=%d mode=%s requested_bytes=%" PRIu64 " locked_bytes=%" PRIu64
            " resident_bytes=%s ranges=%zu failed=%u not_attempted=%u page_size=%" PRIu64
            " memlock_limit=%s vm_lck_kb=%s segments=%" PRIu64
            " (requested %.1f MiB, locked %s%%, resident %s%%, lock %.2f s)\n",
            S->status, S->exit_code, S->mode, S->requested, S->locked, res, P->n, S->failed,
            S->skipped, P->page, lim, vml, S->segments, mib(S->requested), lpct, rpct,
            S->lock_s);
    if (S->have_kinds) {
        for (int k = 0; k <= KST_NONE; ++k) {
            const kstat_t *ks = &S->kst[k];
            if (ks->ranges == 0) {
                continue;
            }
            char kr[32], kb[32], kp[32];
            u64_or_null_str(kr, sizeof(kr), S->have_resident, ks->resident);
            u64_or_null_str(kb, sizeof(kb), S->have_resident_before, ks->resident_before);
            pct_str(kp, sizeof(kp), S->have_resident, ks->resident, ks->requested);
            fprintf(f,
                    "pinner: kind=%s ranges=%" PRIu64 " requested_bytes=%" PRIu64
                    " locked_bytes=%" PRIu64 " resident_bytes=%s resident_before_bytes=%s"
                    " resident_pct=%s\n",
                    kind_name(P, k), ks->ranges, ks->requested, ks->locked, kr, kb, kp);
        }
    }
    if (S->reason != NULL) {
        fprintf(f, "pinner: reason=%s\n", S->reason);
    }
}

static int emit_summary(const pin_t *P, const opts_t *o, const limits_t *L, const summary_t *S)
{
    if (o->json) {
        print_json(stdout, P, o, L, S);
    } else {
        print_text(stdout, P, L, S);
    }
    if (fflush(stdout) != 0 || ferror(stdout)) {
        err_msg("cannot write the summary to stdout: %s", strerror(errno));
        return -1;
    }
    return 0;
}

/* Per-kind totals: ranges + requested (+ residency when want_res) into
 * kst[].requested and *dst_res (resident or resident_before). */
static int kind_pass(const pin_t *P, summary_t *S, int want_res, int before)
{
    for (int k = -1; k < P->nkinds; ++k) {
        kstat_t *ks = &S->kst[k < 0 ? KST_NONE : k];
        uint64_t cnt = 0;
        for (size_t i = 0; i < P->n; ++i) {
            if (P->r[i].kind == k) {
                cnt++;
            }
        }
        ks->ranges = cnt;
        if (cnt == 0) {
            continue;
        }
        uint64_t res = 0;
        if (union_stats(P, sel_kind, k, want_res, &ks->requested, NULL, &res) != 0) {
            return -1;
        }
        if (before) {
            ks->resident_before = res;
        } else {
            ks->resident = res;
            if (union_stats(P, sel_locked_kind, k, 0, &ks->locked, NULL, NULL) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------ */
/* main                                                                      */
/* ------------------------------------------------------------------------ */

static const char *base_name(const char *p)
{
    const char *s = strrchr(p, '/');
    return s != NULL ? s + 1 : p;
}

int main(int argc, char **argv)
{
    opts_t o;
    const int pa = parse_args(argc, argv, &o);
    if (pa < 0) {
        return fflush(stdout) == 0 ? PX_OK : PX_ERROR;
    }
    if (pa > 0) {
        return PX_ERROR;
    }
    g_quiet = o.quiet;
    if (o.verify_only) {
        o.verify = 1;
        if (o.hold_s > 0.0 || o.pidfile != NULL || o.have_max || o.best_effort) {
            note("--verify-only: --hold-seconds/--pidfile/--max-bytes/--best-effort are ignored");
        }
    }
    if (signal(SIGPIPE, SIG_IGN) == SIG_ERR) {
        warn_msg("cannot ignore SIGPIPE: %s", strerror(errno));
    }
    sigset_t ws;
    if (setup_signals(&ws) != 0) {
        err_msg("cannot block the stop signals: %s", strerror(errno));
        return PX_ERROR;
    }
    const long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) {
        err_msg("sysconf(_SC_PAGESIZE) failed");
        return PX_ERROR;
    }

    /* Everything cleaned up at `out` is declared before the first goto. */
    pin_t P;
    memset(&P, 0, sizeof(P));
    P.page = (uint64_t)ps;
    P.fd = -1;
    int code = PX_ERROR;
    int fd = -1;
    size_t map_len = 0;
    int pidfile_written = 0;
    int locked_any = 0;
    summary_t S;
    memset(&S, 0, sizeof(S));
    S.verify_state = -1;
    S.mode = o.verify_only ? "verify-only" : (o.once ? "once" : "daemon");
    limits_t L;
    read_limits(&L);
    lockres_t LR;
    memset(&LR, 0, sizeof(LR));
    static char reason[640];
    reason[0] = '\0';

    if (parse_ranges(o.ranges, &P) != 0) {
        goto out;
    }
    if (P.n == 0 && !o.allow_empty) {
        err_msg("%s has no range lines; nothing to pin (--allow-empty accepts this)", o.ranges);
        goto out;
    }
    if (P.n == 0) {
        note("--allow-empty: %s has no range lines; mapping the file, locking nothing", o.ranges);
    }

    fd = open(o.gguf, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        err_msg("cannot open %s: %s", o.gguf, strerror(errno));
        goto out;
    }
    P.fd = fd;
    struct stat st;
    if (fstat(fd, &st) != 0) {
        err_msg("fstat %s: %s", o.gguf, strerror(errno));
        goto out;
    }
    if (!S_ISREG(st.st_mode)) {
        err_msg("%s is not a regular file", o.gguf);
        goto out;
    }
    if (st.st_size <= 0) {
        err_msg("%s is empty", o.gguf);
        goto out;
    }
    if ((uint64_t)st.st_size > (uint64_t)SIZE_MAX) {
        err_msg("%s is too large to map on this platform", o.gguf);
        goto out;
    }
    P.file_size = (uint64_t)st.st_size;

    if (P.hdr.present) {
        if (P.hdr.has_size && P.hdr.size != P.file_size) {
            err_msg("%s was generated for a file of %" PRIu64 " B (header size=), but %s has %" PRIu64
                    " B: wrong model or an incomplete download",
                    o.ranges, P.hdr.size, o.gguf, P.file_size);
            goto out;
        }
        if (!P.hdr.has_size) {
            warn_msg("%s: the header has no size=; file size not cross-checked", o.ranges);
        }
        if (P.hdr.gguf[0] != '\0' && strcmp(P.hdr.gguf, base_name(o.gguf)) != 0) {
            warn_msg("header gguf=%s differs from the file name %s (size matches; continuing)",
                     P.hdr.gguf, base_name(o.gguf));
        }
        if (P.hdr.has_page && P.hdr.page != P.page) {
            note("ranges were aligned for page=%" PRIu64 "; re-aligning to the runtime page size %" PRIu64,
                 P.hdr.page, P.page);
        }
    } else {
        warn_msg("%s has no '# pigemma-ranges v1' header on its first line; file size not "
                 "cross-checked",
                 o.ranges);
    }

    for (size_t i = 0; i < P.n; ++i) {
        range_t *r = &P.r[i];
        if (r->off + r->len > P.file_size) {
            err_msg("%s:%u: range %" PRIu64 "+%" PRIu64 " ends at %" PRIu64
                    ", beyond the end of %s (%" PRIu64 " B)",
                    o.ranges, r->line, r->off, r->len, r->off + r->len, o.gguf, P.file_size);
            goto out;
        }
        r->a0 = r->off / P.page * P.page;
        r->a1 = (r->off + r->len + P.page - 1u) / P.page * P.page;
        r->lk1 = r->a0;
        S.requested_raw += r->len;
    }
    {
        int sorted = 1;
        const size_t ov = count_overlaps(&P, &sorted);
        if (ov > 0) {
            warn_msg("%zu range(s) overlap other ranges; shared bytes are requested and locked once",
                     ov);
        }
        if (!sorted) {
            note("ranges are not sorted by offset; locking follows file order");
        }
    }
    if (union_stats(&P, sel_all, 0, 0, &S.requested, &S.segments, NULL) != 0) {
        err_msg("out of memory computing the range union");
        goto out;
    }
    P.mincore_ok = (st.st_uid == geteuid()) || L.cap_fowner ||
                   faccessat(AT_FDCWD, o.gguf, W_OK, AT_EACCESS) == 0;

    note("%s: %" PRIu64 " B; %zu ranges -> %" PRIu64 " B (%.1f MiB) page-aligned union in %" PRIu64
         " segments; page %" PRIu64 " B",
         o.gguf, P.file_size, P.n, S.requested, mib(S.requested), S.segments, P.page);

    /* ---- guards (not in --verify-only) -------------------------------- */
    if (!o.verify_only) {
        int refused = 0;
        {
            char why[400] = "";
            if (protected_host(why, sizeof(why))) {
                if (o.allow_protected) {
                    warn_msg("%s; --allow-protected-host given, locking anyway", why);
                } else {
                    refuse_msg("%s. Safety rule (docs/design/FACTS.md): no mlock of model weights "
                               "on this box, the page cache must stay reclaimable for its "
                               "services. Use --verify-only to inspect, or pass "
                               "--allow-protected-host only with the owner's explicit approval",
                               why);
                    if (snprintf(reason, sizeof(reason), "protected host: %.400s", why) < 0) {
                        reason[0] = '\0';
                    }
                    refused = 1;
                }
            }
        }
        if (!refused && o.pidfile != NULL) {
            const long other = read_pidfile(o.pidfile);
            if (other > 0 && other != (long)getpid() && pid_is_live_pinner(other)) {
                refuse_msg("pidfile %s names a running pinner (pid %ld); stop it first "
                           "(kill -TERM %ld) or inspect it with --verify-only",
                           o.pidfile, other, other);
                if (snprintf(reason, sizeof(reason),
                             "another pinner is running (pid %ld, pidfile %s)", other,
                             o.pidfile) < 0) {
                    reason[0] = '\0';
                }
                refused = 1;
            } else if (other > 0 && other != (long)getpid()) {
                note("stale pidfile %s (pid %ld is not a running pinner); it will be replaced",
                     o.pidfile, other);
            }
        }
        if (!refused && o.have_max && S.requested > o.max_bytes) {
            refuse_msg("requested %" PRIu64 " B (%.1f MiB, page-aligned union) > --max-bytes %" PRIu64
                       " B (%.1f MiB)",
                       S.requested, mib(S.requested), o.max_bytes, mib(o.max_bytes));
            if (snprintf(reason, sizeof(reason),
                         "requested %" PRIu64 " B > --max-bytes %" PRIu64 " B", S.requested,
                         o.max_bytes) < 0) {
                reason[0] = '\0';
            }
            refused = 1;
        }
        if (!refused && rlimit_binds(&L) && S.requested > (uint64_t)L.rl.rlim_cur) {
            print_rlimit_hint(&L, S.requested);
            if (o.best_effort) {
                warn_msg("--best-effort: locking in file order until RLIMIT_MEMLOCK (%llu B) is "
                         "reached",
                         (unsigned long long)L.rl.rlim_cur);
            } else {
                refuse_msg("RLIMIT_MEMLOCK %llu B < requested %" PRIu64
                           " B and no CAP_IPC_LOCK; raise the limit (see the hints) or pass "
                           "--best-effort",
                           (unsigned long long)L.rl.rlim_cur, S.requested);
                if (snprintf(reason, sizeof(reason),
                             "RLIMIT_MEMLOCK %llu B < requested %" PRIu64
                             " B; fix: ulimit -l unlimited / /etc/security/limits.d/"
                             "90-pigemma.conf / CAP_IPC_LOCK",
                             (unsigned long long)L.rl.rlim_cur, S.requested) < 0) {
                    reason[0] = '\0';
                }
                refused = 1;
            }
        }
        if (!refused) {
            memory_available(&S.mem);
            S.mem_known = 1;
            if (o.no_mem_check) {
                note("--no-mem-check: memory guard skipped");
            } else if (!S.mem.have) {
                warn_msg("memory guard skipped: neither /proc/meminfo MemAvailable nor a cgroup "
                         "limit is readable");
            } else if (S.requested + HEADROOM_BYTES > S.mem.avail) {
                refuse_msg("requested %" PRIu64 " B (%.1f MiB) + 512 MiB headroom > available %" PRIu64
                           " B (%.1f MiB, %s)",
                           S.requested, mib(S.requested), S.mem.avail, mib(S.mem.avail),
                           S.mem.where);
                if (S.mem.have_meminfo && S.mem.have_cgroup) {
                    note("MemAvailable %" PRIu64 " B, cgroup headroom %" PRIu64 " B",
                         S.mem.meminfo, S.mem.cgroup);
                }
                hint_msg("regenerate the hot set with a smaller --budget-bytes, free memory, or "
                         "pass --no-mem-check if you know better");
                if (snprintf(reason, sizeof(reason),
                             "requested %" PRIu64 " B + 512 MiB > available %" PRIu64 " B (%.300s)",
                             S.requested, S.mem.avail, S.mem.where) < 0) {
                    reason[0] = '\0';
                }
                refused = 1;
            }
        }
        if (refused) {
            S.status = "refused";
            S.exit_code = PX_REFUSED;
            S.reason = reason;
            S.have_vmlck = read_vmlck_kb(&S.vmlck_kb);
            code = emit_summary(&P, &o, &L, &S) == 0 ? PX_REFUSED : PX_ERROR;
            goto out;
        }
    }

    /* ---- pidfile (holding modes only), before any locking -------------- */
    const int wants_hold = !o.verify_only && (!o.once || o.hold_s > 0.0);
    if (o.pidfile != NULL && !o.verify_only) {
        if (wants_hold) {
            if (pidfile_write(o.pidfile) != 0) {
                err_msg("cannot write pidfile %s: %s", o.pidfile, strerror(errno));
                goto out;
            }
            pidfile_written = 1;
        } else {
            note("--pidfile is not written with --once and no --hold-seconds (nothing to stop)");
        }
    }

    /* ---- map ---------------------------------------------------------- */
    map_len = (size_t)P.file_size;
    void *m = mmap(NULL, map_len, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) {
        err_msg("mmap(MAP_SHARED) of %s failed: %s%s", o.gguf, strerror(errno),
                errno == ENODEV ? " (this file system does not support shared mappings; "
                                  "copy the model to a local disk or a docker volume)"
                                : "");
        goto out;
    }
    P.base = m;
    if (!o.verify_only && madvise(P.base, map_len, MADV_RANDOM) != 0) {
        note("madvise(MADV_RANDOM) on the pinner's own mapping failed: %s (harmless)",
             strerror(errno));
    }
    if (!P.mincore_ok) {
        warn_msg("mincore() is not reliable for %s (not the owner, no CAP_FOWNER, not writable): "
                 "Linux >= 5.0 then reports every page resident; resident values are null",
                 o.gguf);
    }

    /* ---- residency before locking (--verify) --------------------------- */
    if (o.verify) {
        if (P.mincore_ok) {
            uint64_t bytes = 0;
            if (union_stats(&P, sel_all, 0, 1, &bytes, NULL, &S.resident_before) != 0) {
                err_msg("mincore failed: %s", strerror(errno));
                goto out;
            }
            S.have_resident_before = 1;
            note("resident before locking: %" PRIu64 " B (%.1f MiB, %.1f%% of requested)",
                 S.resident_before, mib(S.resident_before), pct(S.resident_before, S.requested));
        }
        if (P.any_kind) {
            S.have_kinds = 1;
            if (kind_pass(&P, &S, P.mincore_ok, 1) != 0) {
                err_msg("per-kind statistics failed: %s", strerror(errno));
                goto out;
            }
        }
    }

    /* ---- lock ----------------------------------------------------------- */
    if (!o.verify_only) {
        lock_ranges(&P, &L, &ws, !o.no_warm, &LR);
        S.lr = &LR;
        S.lock_s = LR.lock_s;
    }
    if (union_stats(&P, sel_locked, 0, 0, &S.locked, NULL, NULL) != 0) {
        err_msg("out of memory computing the locked union");
        goto out;
    }
    locked_any = S.locked > 0;
    S.failed = LR.failed;
    S.skipped = LR.skipped;
    if (P.mincore_ok) {
        uint64_t bytes = 0;
        if (union_stats(&P, sel_all, 0, 1, &bytes, NULL, &S.resident) != 0) {
            err_msg("mincore failed: %s", strerror(errno));
            goto out;
        }
        S.have_resident = 1;
    }
    S.have_vmlck = read_vmlck_kb(&S.vmlck_kb);
    if (!o.verify_only && S.have_vmlck && S.vmlck_kb * 1024u < S.locked) {
        warn_msg("kernel VmLck %" PRIu64 " kB is below locked_bytes %" PRIu64
                 " B: mlock() looks intercepted (sanitizer or LD_PRELOAD); the pages are NOT "
                 "pinned",
                 S.vmlck_kb, S.locked);
    }

    if (o.verify) {
        if (S.have_kinds && kind_pass(&P, &S, P.mincore_ok, 0) != 0) {
            err_msg("per-kind statistics failed: %s", strerror(errno));
            goto out;
        }
        if (!o.verify_only && P.mincore_ok) {
            uint64_t lres = 0, lbytes = 0;
            if (union_stats(&P, sel_locked, 0, 1, &lbytes, NULL, &lres) != 0) {
                err_msg("mincore failed: %s", strerror(errno));
                goto out;
            }
            S.verify_state = lres == lbytes ? 1 : 0;
            if (S.verify_state) {
                note("verify: all %" PRIu64 " locked bytes are resident", lbytes);
            } else {
                warn_msg("verify: only %" PRIu64 " of %" PRIu64 " locked bytes are resident", lres,
                         lbytes);
            }
        }
    }

    /* ---- status / exit code ----------------------------------------- */
    if (o.verify_only) {
        S.status = "ok";
        S.exit_code = PX_OK;
    } else if (LR.interrupted) {
        S.status = "interrupted";
        S.exit_code = PX_ERROR;
        if (snprintf(reason, sizeof(reason), "stopped by %s while locking",
                     sig_name(LR.stop_signal)) < 0) {
            reason[0] = '\0';
        }
        S.reason = reason;
        warn_msg("%s; releasing", reason);
    } else if (LR.failed == 0 && LR.skipped == 0) {
        S.status = "ok";
        S.exit_code = PX_OK;
    } else if (S.locked == 0 && LR.rlimit_hit) {
        S.status = "refused";
        S.exit_code = PX_REFUSED;
        S.reason = "mlock not permitted (RLIMIT_MEMLOCK / EPERM); nothing locked";
    } else {
        S.status = "partial";
        S.exit_code = PX_PARTIAL;
        S.reason = LR.rlimit_hit    ? "RLIMIT_MEMLOCK reached before the whole hot set was locked"
                   : LR.memory_hit ? "out of memory (ENOMEM) before the whole hot set was locked"
                                    : "some ranges could not be locked";
    }
    if (!o.verify_only && !LR.interrupted && (LR.failed > 0 || LR.skipped > 0)) {
        warn_msg("%u of %zu ranges failed and %u were not attempted; locked %" PRIu64 " of %" PRIu64
                 " B (%.1f%%)",
                 LR.failed, P.n, LR.skipped, S.locked, S.requested, pct(S.locked, S.requested));
        if (LR.rlimit_hit) {
            print_rlimit_hint(&L, S.requested);
        }
        if (LR.memory_hit) {
            print_memory_hint(S.segments);
        }
    }

    /* 1.1.1: an empty list accepted with --allow-empty is HELD like a real pin set (the process
     * and its whole-file mapping stay until SIGTERM), so method B of the controlled harness has
     * the same helper alive for the whole cell as methods C and D. 1.1.0 exited at once here
     * ("nothing was locked; not holding"), which made B a 25 ms process launch. */
    const int hold_empty = o.allow_empty && P.n == 0;
    const int hold = wants_hold && !LR.interrupted && (S.locked > 0 || hold_empty);
    S.hold_s = hold ? (o.hold_s > 0.0 ? o.hold_s : -1.0) : 0.0;
    if (emit_summary(&P, &o, &L, &S) != 0) {
        code = PX_ERROR;
        goto out;
    }
    code = S.exit_code;

    if (hold) {
        if (o.hold_s > 0.0) {
            note("holding %" PRIu64 " locked bytes for %.1f s (pid %ld)", S.locked, o.hold_s,
                 (long)getpid());
        } else {
            note("holding %" PRIu64 " locked bytes until SIGTERM (pid %ld%s%s)", S.locked,
                 (long)getpid(), pidfile_written ? ", pidfile " : "",
                 pidfile_written ? o.pidfile : "");
        }
        const int sig = wait_stop(&ws, o.hold_s);
        if (sig < 0) {
            err_msg("waiting for a stop signal failed: %s", strerror(errno));
            code = PX_ERROR;
        } else if (sig > 0) {
            note("received %s; releasing the locks", sig_name(sig));
        } else {
            note("hold time over; releasing the locks");
        }
    } else if (wants_hold && S.locked == 0 && !LR.interrupted) {
        warn_msg("nothing was locked; not holding");
    }

out:
    if (locked_any && munlockall() != 0) {
        warn_msg("munlockall failed: %s", strerror(errno));
    }
    if (P.base != NULL && munmap(P.base, map_len) != 0) {
        warn_msg("munmap failed: %s", strerror(errno));
    }
    if (fd >= 0 && close(fd) != 0) {
        warn_msg("close failed: %s", strerror(errno));
    }
    if (pidfile_written) {
        pidfile_remove(o.pidfile);
    }
    free(P.r);
    return code;
}
