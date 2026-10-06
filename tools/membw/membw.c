/*
 * membw.c - memory-bandwidth micro-benchmark for pi-gemma4 (tools/membw).
 *
 * WHY
 *   Decode speed of a weight-streaming LLM on a CPU is bounded by how many
 *   bytes per second the cores can pull out of DRAM. The Pi in-RAM tok/s
 *   ceiling (pi/03_bench_io_mem.sh) and the desktop calibration
 *   (llama_efficiency = tg * bytes_per_token / read_gbps, used by
 *   pigemma/cache_sim.py) must use a MEASURED GB/s, not a guess. This tool
 *   produces that number.
 *
 * USAGE
 *   membw [--size-mb 512] [--threads 1,2,4] [--reps 5] [--json] [--no-mem-check]
 *   membw --help | --version
 *
 * METHOD (per thread count T of --threads, in the order given)
 *   1. Allocate ONE page-aligned arena of --size-mb MiB (1 MiB = 2^20 bytes)
 *      and let the T worker threads write (page-touch) their own contiguous
 *      chunk of it (first touch, untimed). The arena is freed before the next
 *      thread count ("size_mb buffer per thread-set").
 *   2. Three kernels, each timed --reps times; the BEST (shortest) pass wins:
 *        read  : sum of all 64-bit words of the arena, 8 independent
 *                accumulators; every thread's sum is folded into a volatile
 *                sink so the loads cannot be optimised away.
 *                bytes counted = size.
 *        copy  : memcpy() from the first half of the arena to the second half
 *                (a second buffer of size/2). bytes counted = 2 * size/2
 *                (read + write).
 *        triad : a[i] = b[i] + s*c[i] on doubles; three arrays of size/3 each
 *                (rounded down to 64 B). bytes counted = 3 * size/3
 *                (two reads + one write).
 *      Write-allocate (read-for-ownership) traffic is NOT counted, matching
 *      the STREAM convention. glibc memcpy may switch to non-temporal stores
 *      for large sizes; that is part of what "copy" measures.
 *   3. Every array is split into T equal contiguous chunks at 64-byte line
 *      granularity. All workers wait on a barrier, then each records its own
 *      CLOCK_MONOTONIC start/end; a pass lasts max(end) - min(start).
 *   4. GB/s = bytes counted / best pass seconds / 1e9 (decimal GB, 1e9 B/s).
 *   5. copy and triad results are spot-checked on sampled elements after
 *      their timed passes (a failed check is a runtime error, exit 3).
 *
 * OUTPUT (stdout; diagnostics go to stderr; nothing on stdout on failure)
 *   Text (default): a header line and one table row per thread count.
 *   --json: one JSON object, superset of the manifest contract
 *     {"size_mb":512,
 *      "results":[{"threads":4,"read_gbps":x,"copy_gbps":y,"triad_gbps":z}],
 *      "host":"<uname -m>"}
 *   Extra keys (stable, may be ignored by consumers): "tool", "version",
 *   "sysname", "release", "page_size", "nproc_online", "reps", "units",
 *   "bytes_counted":{"read","copy","triad"}, and per result
 *   "read_s","copy_s","triad_s" (best pass, seconds).
 *
 * EXIT CODES
 *   0 success; 1 allocation failure, or refused because the requested arena
 *   does not fit MemAvailable / the cgroup limit minus 256 MiB headroom
 *   (override with --no-mem-check); 2 usage error; 3 runtime error (thread /
 *   clock failure, failed result check, stdout write error).
 *
 * BUILD
 *   Normally via tools/CMakeLists.txt (target "membw"):
 *     Docker (x86): pc/docker_run.sh build-tools
 *     Pi (aarch64): pi/02_build_llama.sh --with-tools
 *   Stand-alone:
 *     gcc -std=c11 -Wall -Wextra -Werror -O2 -D_GNU_SOURCE -pthread \
 *         -o membw tools/membw/membw.c
 *   Builds with or without -D_GNU_SOURCE (falls back to _POSIX_C_SOURCE
 *   200809L), as -std=c11 or -std=gnu11 (tools/CMakeLists.txt uses gnu11
 *   -O3 -Wall -Wextra -Werror). Verified warning-free with gcc 13.3 (Ubuntu
 *   24.04 x86-64) and gcc 14.2 (Debian 13 aarch64). No dependencies beyond
 *   libc + pthreads; no libm. Uses its own mutex/condvar barrier instead of
 *   the optional pthread_barrier_t.
 *
 * DESIGN CHOICES (documented because the spec left them open)
 *   - "MB" in --size-mb is MiB; GB/s is decimal (1e9 B/s) as the spec says.
 *   - One arena per thread count holds all kernels' arrays (peak memory =
 *     --size-mb), so copy uses two halves and triad three thirds of it.
 *   - Before allocating, the arena (+256 MiB headroom) is checked against
 *     MemAvailable and, when present, the process's finite cgroup memory
 *     limit (v2 memory.max or v1 memory.limit_in_bytes, minus usage that is
 *     not inactive page cache), so the bench never pushes a busy Pi into the
 *     OOM killer. Inside a container /proc/meminfo shows the HOST, which is
 *     why the cgroup limit is consulted too (docker --memory caps).
 *   - The read kernel loads through memcpy() of 8 bytes (compiles to plain
 *     loads) so reading the double-initialised arena as 64-bit words is free
 *     of strict-aliasing undefined behaviour.
 *   - No thread pinning, no huge pages: the numbers reflect the default
 *     scheduler and page size, which is what llama.cpp gets too.
 *   - Thread counts above the online CPU count are allowed (with a warning
 *     on stderr) because oversubscription is sometimes what is being probed.
 */

#if !defined(_POSIX_C_SOURCE) && !defined(_GNU_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#define MEMBW_VERSION "1.0.0"

#define MIB_BYTES ((uint64_t)1 << 20)
#define LINE_BYTES ((size_t)64)
#define DBL_PER_LINE (LINE_BYTES / sizeof(double))
#define U64_PER_LINE (LINE_BYTES / sizeof(uint64_t))

#define DEFAULT_SIZE_MB 512u
#define DEFAULT_REPS 5u
#define MAX_SIZE_MB ((uint64_t)1 << 20) /* 1 TiB sanity cap */
#define MAX_REPS 1000u
#define MAX_THREAD_SETS 64u
#define MAX_THREADS 1024u
#define HEADROOM_MIB 256u
#define SMALL_SIZE_NOTE_MB 32u
#define TRIAD_SCALAR 3.0
#define N_SAMPLES 64u

enum {
    MEMBW_EXIT_OK = 0,
    MEMBW_EXIT_ALLOC = 1,
    MEMBW_EXIT_USAGE = 2,
    MEMBW_EXIT_RUNTIME = 3
};

typedef enum { OP_INIT, OP_READ, OP_COPY, OP_TRIAD, OP_QUIT } op_t;

/* ------------------------------------------------------------------------ */
/* Fatal path usable from any thread (stderr is unbuffered; _exit avoids     */
/* running exit handlers concurrently with other threads).                   */
/* ------------------------------------------------------------------------ */

static void fatal_rc(const char *what, int rc)
{
    fprintf(stderr, "membw: fatal: %s failed: %s (%d)\n", what, strerror(rc), rc);
    _exit(MEMBW_EXIT_RUNTIME);
}

/* ------------------------------------------------------------------------ */
/* Portable reusable barrier (mutex + condvar). pthread_barrier_t is an      */
/* optional POSIX feature; this keeps the tool buildable everywhere.         */
/* ------------------------------------------------------------------------ */

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    unsigned total;
    unsigned count;
    unsigned long generation;
} barrier_t;

static int barrier_init(barrier_t *b, unsigned total)
{
    int rc = pthread_mutex_init(&b->mutex, NULL);
    if (rc != 0) {
        return rc;
    }
    rc = pthread_cond_init(&b->cond, NULL);
    if (rc != 0) {
        (void)pthread_mutex_destroy(&b->mutex);
        return rc;
    }
    b->total = total;
    b->count = 0;
    b->generation = 0;
    return 0;
}

static void barrier_destroy(barrier_t *b)
{
    int rc = pthread_cond_destroy(&b->cond);
    if (rc != 0) {
        fprintf(stderr, "membw: warning: pthread_cond_destroy: %s\n", strerror(rc));
    }
    rc = pthread_mutex_destroy(&b->mutex);
    if (rc != 0) {
        fprintf(stderr, "membw: warning: pthread_mutex_destroy: %s\n", strerror(rc));
    }
}

static void barrier_wait(barrier_t *b)
{
    int rc = pthread_mutex_lock(&b->mutex);
    if (rc != 0) {
        fatal_rc("pthread_mutex_lock", rc);
    }
    const unsigned long gen = b->generation;
    b->count++;
    if (b->count >= b->total) {
        b->count = 0;
        b->generation++;
        rc = pthread_cond_broadcast(&b->cond);
        if (rc != 0) {
            fatal_rc("pthread_cond_broadcast", rc);
        }
    } else {
        while (gen == b->generation) {
            rc = pthread_cond_wait(&b->cond, &b->mutex);
            if (rc != 0) {
                fatal_rc("pthread_cond_wait", rc);
            }
        }
    }
    rc = pthread_mutex_unlock(&b->mutex);
    if (rc != 0) {
        fatal_rc("pthread_mutex_unlock", rc);
    }
}

/* Lower the participant count. Only used to unwind after pthread_create
 * failed part-way; the caller guarantees count < total (main has not
 * arrived yet, at most `total - 1` workers are waiting). */
static void barrier_set_total(barrier_t *b, unsigned total)
{
    int rc = pthread_mutex_lock(&b->mutex);
    if (rc != 0) {
        fatal_rc("pthread_mutex_lock", rc);
    }
    b->total = total;
    rc = pthread_mutex_unlock(&b->mutex);
    if (rc != 0) {
        fatal_rc("pthread_mutex_unlock", rc);
    }
}

/* ------------------------------------------------------------------------ */
/* Shared state and kernels                                                  */
/* ------------------------------------------------------------------------ */

typedef struct {
    barrier_t start;
    barrier_t done;
    op_t op;
    unsigned nthreads;
    unsigned char *arena;
    size_t arena_bytes;
    /* read view: whole arena as 64-bit words, in 64-byte lines */
    const unsigned char *rd;
    size_t rd_lines;
    /* copy view: first half -> second half */
    const unsigned char *cp_src;
    unsigned char *cp_dst;
    size_t cp_lines;
    /* triad view: three consecutive thirds */
    double *ta;
    const double *tb;
    const double *tc;
    size_t t_lines;
    double scalar;
} shared_t;

typedef struct {
    shared_t *sh;
    unsigned idx;
    pthread_t tid;
    double t_start;
    double t_end;
    uint64_t sink;
} worker_t;

typedef struct {
    unsigned threads;
    double read_s;
    double copy_s;
    double triad_s;
    double read_gbps;
    double copy_gbps;
    double triad_gbps;
} result_t;

static volatile uint64_t g_sink;

static double now_s(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        fatal_rc("clock_gettime(CLOCK_MONOTONIC)", errno);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Equal contiguous chunk [lo, hi) of `units` for worker idx of n. */
static void chunk_of(size_t units, unsigned n, unsigned idx, size_t *lo, size_t *hi)
{
    *lo = (size_t)(((uint64_t)units * (uint64_t)idx) / (uint64_t)n);
    *hi = (size_t)(((uint64_t)units * (uint64_t)(idx + 1u)) / (uint64_t)n);
}

#define LOAD_U64(dst, src) memcpy(&(dst), (src), sizeof(uint64_t))

static uint64_t kern_read(const unsigned char *p, size_t n_words)
{
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    size_t i = 0;
    for (; i + 8u <= n_words; i += 8u) {
        const unsigned char *q = p + i * sizeof(uint64_t);
        uint64_t v0, v1, v2, v3, v4, v5, v6, v7;
        LOAD_U64(v0, q);
        LOAD_U64(v1, q + 8);
        LOAD_U64(v2, q + 16);
        LOAD_U64(v3, q + 24);
        LOAD_U64(v4, q + 32);
        LOAD_U64(v5, q + 40);
        LOAD_U64(v6, q + 48);
        LOAD_U64(v7, q + 56);
        s0 += v0;
        s1 += v1;
        s2 += v2;
        s3 += v3;
        s4 += v4;
        s5 += v5;
        s6 += v6;
        s7 += v7;
    }
    for (; i < n_words; ++i) {
        uint64_t v;
        LOAD_U64(v, p + i * sizeof(uint64_t));
        s0 += v;
    }
    return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
}

static void kern_triad(double *restrict a, const double *restrict b,
                       const double *restrict c, double s, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        a[i] = b[i] + s * c[i];
    }
}

static void kern_init(double *x, size_t lo, size_t hi)
{
    /* Normal doubles in [1, 2): no denormal slow paths in triad. */
    for (size_t i = lo; i < hi; ++i) {
        x[i] = 1.0 + (double)(i & 1023u) / 1024.0;
    }
}

static void *worker_main(void *arg)
{
    worker_t *w = (worker_t *)arg;
    shared_t *sh = w->sh;
    for (;;) {
        barrier_wait(&sh->start);
        const op_t op = sh->op;
        if (op == OP_QUIT) {
            break;
        }
        size_t lo = 0, hi = 0;
        const double t0 = now_s();
        switch (op) {
        case OP_INIT: {
            const size_t lines = sh->arena_bytes / LINE_BYTES;
            chunk_of(lines, sh->nthreads, w->idx, &lo, &hi);
            kern_init((double *)(void *)sh->arena, lo * DBL_PER_LINE, hi * DBL_PER_LINE);
            break;
        }
        case OP_READ:
            chunk_of(sh->rd_lines, sh->nthreads, w->idx, &lo, &hi);
            w->sink += kern_read(sh->rd + lo * LINE_BYTES, (hi - lo) * U64_PER_LINE);
            break;
        case OP_COPY:
            chunk_of(sh->cp_lines, sh->nthreads, w->idx, &lo, &hi);
            if (hi > lo) {
                memcpy(sh->cp_dst + lo * LINE_BYTES, sh->cp_src + lo * LINE_BYTES,
                       (hi - lo) * LINE_BYTES);
            }
            break;
        case OP_TRIAD:
            chunk_of(sh->t_lines, sh->nthreads, w->idx, &lo, &hi);
            kern_triad(sh->ta + lo * DBL_PER_LINE, sh->tb + lo * DBL_PER_LINE,
                       sh->tc + lo * DBL_PER_LINE, sh->scalar, (hi - lo) * DBL_PER_LINE);
            break;
        case OP_QUIT:
        default:
            break;
        }
        const double t1 = now_s();
        w->t_start = t0;
        w->t_end = t1;
        barrier_wait(&sh->done);
    }
    return NULL;
}

/* One timed pass of `op` over all workers; returns max(end) - min(start). */
static double run_pass(shared_t *sh, const worker_t *w, op_t op)
{
    sh->op = op;
    barrier_wait(&sh->start);
    barrier_wait(&sh->done);
    double mn = w[0].t_start;
    double mx = w[0].t_end;
    for (unsigned i = 1; i < sh->nthreads; ++i) {
        if (w[i].t_start < mn) {
            mn = w[i].t_start;
        }
        if (w[i].t_end > mx) {
            mx = w[i].t_end;
        }
    }
    double el = mx - mn;
    if (!(el > 0.0)) {
        el = 1e-9; /* clock granularity guard; never expected at MiB sizes */
    }
    return el;
}

static double absd(double x)
{
    return x < 0.0 ? -x : x;
}

static int verify_copy(const shared_t *sh)
{
    const size_t bytes = sh->cp_lines * LINE_BYTES;
    if (bytes < sizeof(uint64_t)) {
        return 1;
    }
    const size_t last = bytes - sizeof(uint64_t);
    for (unsigned j = 0; j < N_SAMPLES; ++j) {
        size_t off = (size_t)(((uint64_t)last * j) / (N_SAMPLES - 1u));
        off -= off % sizeof(uint64_t);
        if (memcmp(sh->cp_src + off, sh->cp_dst + off, sizeof(uint64_t)) != 0) {
            fprintf(stderr, "membw: copy check failed at byte offset %zu\n", off);
            return 0;
        }
    }
    return 1;
}

static int verify_triad(const shared_t *sh)
{
    const size_t n = sh->t_lines * DBL_PER_LINE;
    if (n == 0) {
        return 1;
    }
    for (unsigned j = 0; j < N_SAMPLES; ++j) {
        const size_t k = (size_t)(((uint64_t)(n - 1u) * j) / (N_SAMPLES - 1u));
        const double expect = sh->tb[k] + sh->scalar * sh->tc[k];
        if (!(absd(sh->ta[k] - expect) <= 1e-12 * absd(expect))) {
            fprintf(stderr, "membw: triad check failed at element %zu (%.17g != %.17g)\n",
                    k, sh->ta[k], expect);
            return 0;
        }
    }
    return 1;
}

static size_t page_size(void)
{
    const long pg = sysconf(_SC_PAGESIZE);
    return pg > 0 ? (size_t)pg : (size_t)4096;
}

/* Runs all kernels for one thread count. Returns an exit code. */
static int run_thread_set(unsigned nthreads, size_t arena_bytes, unsigned reps, result_t *out)
{
    void *mem = NULL;
    int rc = posix_memalign(&mem, page_size(), arena_bytes);
    if (rc != 0 || mem == NULL) {
        fprintf(stderr, "membw: cannot allocate %zu bytes (%" PRIu64 " MiB) for %u thread(s): %s\n",
                arena_bytes, (uint64_t)arena_bytes / MIB_BYTES, nthreads,
                rc != 0 ? strerror(rc) : "posix_memalign returned NULL");
        return MEMBW_EXIT_ALLOC;
    }
    worker_t *w = (worker_t *)calloc(nthreads, sizeof(*w));
    if (w == NULL) {
        fprintf(stderr, "membw: cannot allocate worker table for %u threads\n", nthreads);
        free(mem);
        return MEMBW_EXIT_ALLOC;
    }

    shared_t sh;
    memset(&sh, 0, sizeof(sh));
    sh.nthreads = nthreads;
    sh.arena = (unsigned char *)mem;
    sh.arena_bytes = arena_bytes;
    sh.rd = sh.arena;
    sh.rd_lines = arena_bytes / LINE_BYTES;
    sh.cp_lines = (arena_bytes / 2u) / LINE_BYTES;
    sh.cp_src = sh.arena;
    sh.cp_dst = sh.arena + sh.cp_lines * LINE_BYTES;
    sh.t_lines = (arena_bytes / 3u) / LINE_BYTES;
    sh.ta = (double *)(void *)sh.arena;
    sh.tb = sh.ta + sh.t_lines * DBL_PER_LINE;
    sh.tc = sh.tb + sh.t_lines * DBL_PER_LINE;
    sh.scalar = TRIAD_SCALAR;

    rc = barrier_init(&sh.start, nthreads + 1u);
    if (rc != 0) {
        fprintf(stderr, "membw: barrier init failed: %s\n", strerror(rc));
        free(w);
        free(mem);
        return MEMBW_EXIT_RUNTIME;
    }
    rc = barrier_init(&sh.done, nthreads + 1u);
    if (rc != 0) {
        fprintf(stderr, "membw: barrier init failed: %s\n", strerror(rc));
        barrier_destroy(&sh.start);
        free(w);
        free(mem);
        return MEMBW_EXIT_RUNTIME;
    }

    unsigned created = 0;
    for (unsigned i = 0; i < nthreads; ++i) {
        w[i].sh = &sh;
        w[i].idx = i;
        rc = pthread_create(&w[i].tid, NULL, worker_main, &w[i]);
        if (rc != 0) {
            fprintf(stderr, "membw: pthread_create failed for thread %u of %u: %s\n",
                    i + 1u, nthreads, strerror(rc));
            break;
        }
        created++;
    }

    int status = MEMBW_EXIT_OK;
    if (created < nthreads) {
        status = MEMBW_EXIT_RUNTIME;
        barrier_set_total(&sh.start, created + 1u);
        barrier_set_total(&sh.done, created + 1u);
    } else {
        (void)run_pass(&sh, w, OP_INIT); /* page touch, untimed */

        const op_t ops[3] = {OP_READ, OP_COPY, OP_TRIAD};
        double best[3] = {-1.0, -1.0, -1.0};
        for (unsigned k = 0; k < 3u && status == MEMBW_EXIT_OK; ++k) {
            for (unsigned r = 0; r < reps; ++r) {
                const double el = run_pass(&sh, w, ops[k]);
                if (best[k] < 0.0 || el < best[k]) {
                    best[k] = el;
                }
            }
            if (ops[k] == OP_COPY && !verify_copy(&sh)) {
                status = MEMBW_EXIT_RUNTIME;
            }
            if (ops[k] == OP_TRIAD && !verify_triad(&sh)) {
                status = MEMBW_EXIT_RUNTIME;
            }
        }
        if (status == MEMBW_EXIT_OK) {
            const double rd_bytes = (double)(sh.rd_lines * LINE_BYTES);
            const double cp_bytes = 2.0 * (double)(sh.cp_lines * LINE_BYTES);
            const double tr_bytes = 3.0 * (double)(sh.t_lines * LINE_BYTES);
            out->threads = nthreads;
            out->read_s = best[0];
            out->copy_s = best[1];
            out->triad_s = best[2];
            out->read_gbps = rd_bytes / best[0] / 1e9;
            out->copy_gbps = cp_bytes / best[1] / 1e9;
            out->triad_gbps = tr_bytes / best[2] / 1e9;
        }
    }

    /* Release and join every worker that exists. */
    sh.op = OP_QUIT;
    barrier_wait(&sh.start);
    for (unsigned i = 0; i < created; ++i) {
        rc = pthread_join(w[i].tid, NULL);
        if (rc != 0) {
            fprintf(stderr, "membw: pthread_join failed: %s\n", strerror(rc));
            status = MEMBW_EXIT_RUNTIME;
        }
        g_sink ^= w[i].sink; /* consume the read sums */
    }
    barrier_destroy(&sh.done);
    barrier_destroy(&sh.start);
    free(w);
    free(mem);
    return status;
}

/* ------------------------------------------------------------------------ */
/* Memory-availability guard                                                 */
/* ------------------------------------------------------------------------ */

/* Reads "<key>[:] <value> [kB]" style entries (meminfo / memory.stat).
 * Returns 1 and the value (converted to bytes when suffixed kB). */
static int read_keyed_value(const char *path, const char *key, uint64_t *out)
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
        char *end = NULL;
        errno = 0;
        const unsigned long long v = strtoull(p, &end, 10);
        if (errno != 0 || end == p) {
            continue;
        }
        uint64_t val = (uint64_t)v;
        while (*end == ' ' || *end == '\t') {
            end++;
        }
        if (strncmp(end, "kB", 2) == 0) {
            val *= 1024u;
        }
        *out = val;
        found = 1;
        break;
    }
    if (fclose(f) != 0) {
        return 0;
    }
    return found;
}

/* Reads a file holding a single unsigned number. Returns 0 for "max",
 * missing files or garbage. */
static int read_single_u64(const char *path, uint64_t *out)
{
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return 0;
    }
    char buf[64];
    const char *got = fgets(buf, sizeof(buf), f);
    if (fclose(f) != 0 || got == NULL) {
        return 0;
    }
    if (buf[0] < '0' || buf[0] > '9') {
        return 0;
    }
    char *end = NULL;
    errno = 0;
    const unsigned long long v = strtoull(buf, &end, 10);
    if (errno != 0 || end == buf) {
        return 0;
    }
    *out = (uint64_t)v;
    return 1;
}

/* cgroup limits at or above this are "unlimited" (cgroup v1 reports
 * PAGE_COUNTER_MAX * page size, about 2^63, when no limit is set). */
#define CGROUP_UNLIMITED ((uint64_t)1 << 60)

/* Headroom of one cgroup directory: limit - (usage - inactive page cache).
 *   v2: memory.max, memory.current, memory.stat:inactive_file
 *   v1: memory.limit_in_bytes, memory.usage_in_bytes,
 *       memory.stat:total_inactive_file
 * Returns 1 when a finite limit was found there. */
static int cgroup_dir_available(const char *dir, int v1, uint64_t *avail, char *label,
                                size_t label_len)
{
    const char *f_max = v1 ? "memory.limit_in_bytes" : "memory.max";
    const char *f_cur = v1 ? "memory.usage_in_bytes" : "memory.current";
    const char *k_inact = v1 ? "total_inactive_file" : "inactive_file";
    char p_max[700], p_cur[700], p_stat[700];
    const int w1 = snprintf(p_max, sizeof(p_max), "%s/%s", dir, f_max);
    const int w2 = snprintf(p_cur, sizeof(p_cur), "%s/%s", dir, f_cur);
    const int w3 = snprintf(p_stat, sizeof(p_stat), "%s/memory.stat", dir);
    if (w1 < 0 || w2 < 0 || w3 < 0 || (size_t)w1 >= sizeof(p_max) ||
        (size_t)w2 >= sizeof(p_cur) || (size_t)w3 >= sizeof(p_stat)) {
        return 0;
    }
    uint64_t mx = 0, cur = 0, inact = 0;
    if (!read_single_u64(p_max, &mx) || mx >= CGROUP_UNLIMITED ||
        !read_single_u64(p_cur, &cur)) {
        return 0; /* no such file, "max", or unlimited */
    }
    if (!read_keyed_value(p_stat, k_inact, &inact)) {
        inact = 0;
    }
    const uint64_t used = cur > inact ? cur - inact : 0;
    *avail = mx > used ? mx - used : 0;
    /* Truncation of this label is harmless; the return value is used so
     * -Wformat-truncation does not fire under -Werror. */
    if (snprintf(label, label_len, "cgroup %s", p_max) < 0 && label_len > 0) {
        label[0] = '\0';
    }
    return 1;
}

/* 1 when the comma-separated list contains `tok`. */
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

/* Smallest finite cgroup headroom of this process. Looks at the process's
 * own cgroup (from /proc/self/cgroup) and at the mount roots, which under a
 * private cgroup namespace (containers) ARE the container's cgroup; handles
 * the v2 unified hierarchy and the v1 memory controller (Docker Desktop's
 * WSL2 kernel still mounts v1). Returns 1 when any finite limit exists. */
static int cgroup_available(uint64_t *avail, char *where, size_t where_len)
{
    char v2path[512] = "";
    char v1path[512] = "";
    FILE *f = fopen("/proc/self/cgroup", "r");
    if (f != NULL) {
        char line[1024];
        while (fgets(line, sizeof(line), f) != NULL) {
            line[strcspn(line, "\r\n")] = '\0';
            /* "hierarchy-id:controller-list:path" */
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
        (void)fclose(f);
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
        if (cgroup_dir_available(dirs[i], is_v1[i], &v, label, sizeof(label))) {
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

/* Returns 1 and the smallest known headroom; 0 when nothing is known. */
static int memory_available(uint64_t *avail, char *where, size_t where_len)
{
    uint64_t best = 0;
    int have = 0;
    uint64_t v = 0;
    if (read_keyed_value("/proc/meminfo", "MemAvailable", &v)) {
        best = v;
        have = 1;
        if (snprintf(where, where_len, "/proc/meminfo MemAvailable") < 0 && where_len > 0) {
            where[0] = '\0';
        }
    }
    char cg_where[720] = "";
    if (cgroup_available(&v, cg_where, sizeof(cg_where))) {
        if (!have || v < best) {
            best = v;
            if (snprintf(where, where_len, "%s", cg_where) < 0 && where_len > 0) {
                where[0] = '\0';
            }
        }
        have = 1;
    }
    *avail = best;
    return have;
}

/* ------------------------------------------------------------------------ */
/* CLI                                                                       */
/* ------------------------------------------------------------------------ */

static void usage(FILE *to)
{
    fprintf(to,
            "usage: membw [--size-mb N] [--threads LIST] [--reps N] [--json] [--no-mem-check]\n"
            "       membw --help | --version\n"
            "\n"
            "Memory-bandwidth micro-benchmark (pi-gemma4). For every thread count it\n"
            "allocates one page-touched arena of N MiB and measures, best of --reps:\n"
            "  read  : sum of 64-bit words over the arena          (bytes = N MiB)\n"
            "  copy  : memcpy first half -> second half            (bytes = 2 x N/2)\n"
            "  triad : a[i] = b[i] + s*c[i], three arrays of N/3   (bytes = 3 x N/3)\n"
            "GB/s = 1e9 bytes/s; write-allocate traffic is not counted (STREAM rule).\n"
            "\n"
            "options:\n"
            "  --size-mb N      arena size in MiB per thread set (default %u, 1..%" PRIu64 ")\n"
            "  --threads LIST   comma-separated thread counts, e.g. 1,2,4 (default 1,2,4;\n"
            "                   1..%u each, at most %u entries)\n"
            "  --reps N         timed passes per kernel; best is reported (default %u, 1..%u)\n"
            "  --json           print one JSON object instead of a table:\n"
            "                   {\"size_mb\":N,\"results\":[{\"threads\":T,\"read_gbps\":..,\n"
            "                    \"copy_gbps\":..,\"triad_gbps\":..}],\"host\":\"<uname -m>\",...}\n"
            "  --no-mem-check   skip the MemAvailable / cgroup headroom guard\n"
            "                   (default: refuse when N MiB + %u MiB does not fit)\n"
            "  -h, --help       show this help and exit\n"
            "  --version        print the version and exit\n"
            "\n"
            "exit codes: 0 ok, 1 allocation failure or not enough free memory,\n"
            "            2 usage error, 3 runtime error (threads/clock/check/output)\n",
            DEFAULT_SIZE_MB, MAX_SIZE_MB, MAX_THREADS, MAX_THREAD_SETS, DEFAULT_REPS,
            MAX_REPS, HEADROOM_MIB);
}

/* Parses a decimal unsigned integer in [lo, hi]; digits only. */
static int parse_uint(const char *s, uint64_t lo, uint64_t hi, uint64_t *out)
{
    if (s == NULL || s[0] < '0' || s[0] > '9') {
        return 0;
    }
    char *end = NULL;
    errno = 0;
    const unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        return 0;
    }
    if ((uint64_t)v < lo || (uint64_t)v > hi) {
        return 0;
    }
    *out = (uint64_t)v;
    return 1;
}

static int parse_threads(const char *s, unsigned *list, unsigned *count)
{
    if (s == NULL || *s == '\0') {
        return 0;
    }
    unsigned tmp[MAX_THREAD_SETS];
    unsigned n = 0;
    const char *p = s;
    for (;;) {
        const char *comma = strchr(p, ',');
        const size_t len = comma != NULL ? (size_t)(comma - p) : strlen(p);
        char tok[32];
        if (len == 0 || len >= sizeof(tok) || n >= MAX_THREAD_SETS) {
            return 0;
        }
        memcpy(tok, p, len);
        tok[len] = '\0';
        uint64_t v = 0;
        if (!parse_uint(tok, 1, MAX_THREADS, &v)) {
            return 0;
        }
        tmp[n++] = (unsigned)v;
        if (comma == NULL) {
            break;
        }
        p = comma + 1;
    }
    memcpy(list, tmp, n * sizeof(tmp[0]));
    *count = n;
    return 1;
}

/* Returns the option value for "--name V" or "--name=V"; NULL when argv[*i]
 * is not this option. Sets *missing when the value is absent. */
static const char *opt_value(int argc, char **argv, int *i, const char *name, int *missing)
{
    const char *a = argv[*i];
    const size_t n = strlen(name);
    if (strncmp(a, name, n) != 0) {
        return NULL;
    }
    if (a[n] == '=') {
        return a + n + 1;
    }
    if (a[n] != '\0') {
        return NULL;
    }
    if (*i + 1 >= argc) {
        *missing = 1;
        return NULL;
    }
    *i += 1;
    return argv[*i];
}

static void json_string(FILE *to, const char *s)
{
    fputc('"', to);
    for (const unsigned char *p = (const unsigned char *)s; *p != '\0'; ++p) {
        if (*p == '"' || *p == '\\') {
            fputc('\\', to);
            fputc((int)*p, to);
        } else if (*p < 0x20u || *p >= 0x7fu) {
            fprintf(to, "\\u%04x", (unsigned)*p);
        } else {
            fputc((int)*p, to);
        }
    }
    fputc('"', to);
}

int main(int argc, char **argv)
{
    uint64_t size_mb = DEFAULT_SIZE_MB;
    uint64_t reps = DEFAULT_REPS;
    unsigned threads[MAX_THREAD_SETS] = {1u, 2u, 4u};
    unsigned n_sets = 3u;
    int json = 0;
    int mem_check = 1;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        int missing = 0;
        const char *v = NULL;
        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            usage(stdout);
            return fflush(stdout) == 0 ? MEMBW_EXIT_OK : MEMBW_EXIT_RUNTIME;
        }
        if (strcmp(a, "--version") == 0) {
            printf("membw %s\n", MEMBW_VERSION);
            return fflush(stdout) == 0 ? MEMBW_EXIT_OK : MEMBW_EXIT_RUNTIME;
        }
        if (strcmp(a, "--json") == 0) {
            json = 1;
            continue;
        }
        if (strcmp(a, "--no-mem-check") == 0) {
            mem_check = 0;
            continue;
        }
        if ((v = opt_value(argc, argv, &i, "--size-mb", &missing)) != NULL || missing) {
            if (missing || !parse_uint(v, 1, MAX_SIZE_MB, &size_mb)) {
                fprintf(stderr, "membw: --size-mb needs an integer in 1..%" PRIu64 "\n",
                        MAX_SIZE_MB);
                usage(stderr);
                return MEMBW_EXIT_USAGE;
            }
            continue;
        }
        if ((v = opt_value(argc, argv, &i, "--threads", &missing)) != NULL || missing) {
            if (missing || !parse_threads(v, threads, &n_sets)) {
                fprintf(stderr,
                        "membw: --threads needs a comma list of integers in 1..%u"
                        " (max %u entries)\n",
                        MAX_THREADS, MAX_THREAD_SETS);
                usage(stderr);
                return MEMBW_EXIT_USAGE;
            }
            continue;
        }
        if ((v = opt_value(argc, argv, &i, "--reps", &missing)) != NULL || missing) {
            if (missing || !parse_uint(v, 1, MAX_REPS, &reps)) {
                fprintf(stderr, "membw: --reps needs an integer in 1..%u\n", MAX_REPS);
                usage(stderr);
                return MEMBW_EXIT_USAGE;
            }
            continue;
        }
        fprintf(stderr, "membw: unknown argument: %s\n", a);
        usage(stderr);
        return MEMBW_EXIT_USAGE;
    }

    if (size_mb > (uint64_t)SIZE_MAX / MIB_BYTES) {
        fprintf(stderr, "membw: --size-mb %" PRIu64 " does not fit this address space\n",
                size_mb);
        return MEMBW_EXIT_USAGE;
    }
    const size_t arena_bytes = (size_t)(size_mb * MIB_BYTES);

    if (mem_check) {
        uint64_t avail = 0;
        char where[300] = "";
        if (memory_available(&avail, where, sizeof(where))) {
            const uint64_t need = (uint64_t)arena_bytes + (uint64_t)HEADROOM_MIB * MIB_BYTES;
            if (need > avail) {
                fprintf(stderr,
                        "membw: refusing: --size-mb %" PRIu64 " needs %" PRIu64
                        " MiB incl. %u MiB headroom, but only %" PRIu64
                        " MiB are available (%s). Use a smaller --size-mb or --no-mem-check.\n",
                        size_mb, need / MIB_BYTES, HEADROOM_MIB, avail / MIB_BYTES, where);
                return MEMBW_EXIT_ALLOC;
            }
        } else {
            fprintf(stderr,
                    "membw: note: free-memory guard skipped (no /proc/meminfo or cgroup info)\n");
        }
    }

    long nproc = -1;
#ifdef _SC_NPROCESSORS_ONLN
    nproc = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    for (unsigned s = 0; s < n_sets; ++s) {
        if (nproc > 0 && (long)threads[s] > nproc) {
            fprintf(stderr, "membw: warning: %u threads > %ld online CPUs (oversubscribed)\n",
                    threads[s], nproc);
        }
    }
    if (size_mb < SMALL_SIZE_NOTE_MB) {
        fprintf(stderr,
                "membw: note: %" PRIu64 " MiB may fit in CPU caches;"
                " numbers can exceed DRAM bandwidth\n",
                size_mb);
    }

    struct utsname un;
    const char *machine = "unknown";
    const char *sysname = "unknown";
    const char *release = "unknown";
    if (uname(&un) == 0) {
        machine = un.machine;
        sysname = un.sysname;
        release = un.release;
    } else {
        fprintf(stderr, "membw: warning: uname failed: %s\n", strerror(errno));
    }

    result_t results[MAX_THREAD_SETS];
    memset(results, 0, sizeof(results));
    for (unsigned s = 0; s < n_sets; ++s) {
        const int rc = run_thread_set(threads[s], arena_bytes, (unsigned)reps, &results[s]);
        if (rc != MEMBW_EXIT_OK) {
            return rc;
        }
    }

    const size_t rd_bytes = (arena_bytes / LINE_BYTES) * LINE_BYTES;
    const size_t cp_half = ((arena_bytes / 2u) / LINE_BYTES) * LINE_BYTES;
    const size_t tr_third = ((arena_bytes / 3u) / LINE_BYTES) * LINE_BYTES;

    if (json) {
        printf("{\n  \"tool\": \"membw\",\n  \"version\": \"%s\",\n", MEMBW_VERSION);
        printf("  \"host\": ");
        json_string(stdout, machine);
        printf(",\n  \"sysname\": ");
        json_string(stdout, sysname);
        printf(",\n  \"release\": ");
        json_string(stdout, release);
        printf(",\n  \"page_size\": %zu,\n", page_size());
        printf("  \"nproc_online\": %ld,\n", nproc);
        printf("  \"size_mb\": %" PRIu64 ",\n", size_mb);
        printf("  \"reps\": %" PRIu64 ",\n", reps);
        printf("  \"units\": \"GB/s = 1e9 bytes/s, best of reps\",\n");
        printf("  \"bytes_counted\": {\"read\": %zu, \"copy\": %zu, \"triad\": %zu},\n",
               rd_bytes, 2u * cp_half, 3u * tr_third);
        printf("  \"results\": [");
        for (unsigned s = 0; s < n_sets; ++s) {
            const result_t *r = &results[s];
            printf("%s\n    {\"threads\": %u, \"read_gbps\": %.3f, \"copy_gbps\": %.3f,"
                   " \"triad_gbps\": %.3f, \"read_s\": %.6f, \"copy_s\": %.6f,"
                   " \"triad_s\": %.6f}",
                   s == 0 ? "" : ",", r->threads, r->read_gbps, r->copy_gbps, r->triad_gbps,
                   r->read_s, r->copy_s, r->triad_s);
        }
        printf("\n  ]\n}\n");
    } else {
        printf("membw %s  host=%s (%s %s)  page=%zu B  size=%" PRIu64 " MiB  reps=%" PRIu64 "\n",
               MEMBW_VERSION, machine, sysname, release, page_size(), size_mb, reps);
        printf("GB/s = 1e9 B/s, best of reps; bytes counted: read %zu, copy 2x%zu,"
               " triad 3x%zu\n",
               rd_bytes, cp_half, tr_third);
        printf("%7s  %11s  %11s  %11s\n", "threads", "read GB/s", "copy GB/s", "triad GB/s");
        for (unsigned s = 0; s < n_sets; ++s) {
            const result_t *r = &results[s];
            printf("%7u  %11.3f  %11.3f  %11.3f\n", r->threads, r->read_gbps, r->copy_gbps,
                   r->triad_gbps);
        }
    }
    if (fflush(stdout) != 0 || ferror(stdout)) {
        fprintf(stderr, "membw: error writing results to stdout\n");
        return MEMBW_EXIT_RUNTIME;
    }
    return MEMBW_EXIT_OK;
}
