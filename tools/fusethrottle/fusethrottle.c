/*
 * fusethrottle.c - read-only, bandwidth-limited FUSE passthrough of ONE file (pi-gemma4, P5).
 *
 * WHY
 *   The controlled-memory comparison (brief 2026-10-05 sections 2-4) must read the model at the
 *   Raspberry Pi microSD rate (91.6 MB/s, measured with O_DIRECT 4 MiB reads on the real Pi).
 *   The Docker Desktop VM kernel (5.15.167.4-microsoft-standard-WSL2) has no blk-throttling:
 *   no io.max, no blkio.throttle.*, and `docker run --device-read-bps` is silently discarded.
 *   This passthrough exposes the GGUF as <mountpoint>/<name>; llama-server and the pinner mmap
 *   THAT file, so the FUSE inode's page cache is the only cache of model bytes, and every page
 *   that is not cached costs a FUSE READ that this daemon serves at the emulated device rate.
 *
 * DEVICE MODEL (deliberately simple, recorded in .stats)
 *   One serial device of `rate` bytes/s (+ an optional fixed per-request latency): a request of
 *   n backing bytes reserves the time slot [max(now, vclock), +op_latency + n/rate) under a mutex,
 *   reads the bytes from the backing file, then sleeps until the end of its slot. Concurrent
 *   requests (libfuse is multi-threaded, the kernel sends async readahead) therefore complete at
 *   most `rate` bytes/s in aggregate. If the backing read itself overruns the slot, vclock is
 *   moved to the real completion time (no catch-up burst). rate 0 = unthrottled (still O_DIRECT).
 *   Bandwidth only: no queue-depth, IOPS or random-access penalty of a real SD card is modelled.
 *
 * NO SECOND COPY
 *   The backing file is opened with O_DIRECT (aligned bounce buffer, offsets/lengths rounded to
 *   --align), so its own page cache is not populated; at start the backing file's cached pages
 *   are dropped with posix_fadvise(DONTNEED) (targeted, never a global drop_caches). If O_DIRECT
 *   is refused the daemon exits 3 unless --allow-buffered is given ("direct_io": false then).
 *
 * FILES IN THE MOUNT
 *   /<name>   the backing file, mode 0444, same size; page cache kept across opens (kernel_cache)
 *   /.stats   4096-byte JSON snapshot of the counters (direct_io, never cached, never throttled)
 *
 * USAGE
 *   fusethrottle --backing FILE --mountpoint DIR [--name NAME] [--rate-bps N] [--op-latency-us U]
 *                [--align N] [--max-readahead N] [--max-read N] [--trace FILE]
 *                [--allow-buffered] [--no-kernel-cache] [--single-thread]
 *   Runs in the foreground until unmounted (fusermount3 -u DIR / umount DIR) or SIGTERM/SIGINT.
 *   One JSON line {"event":"ready",...} is printed on stdout once the kernel handshake is done.
 *   --trace FILE writes one CSV line per backing read:
 *     t_us,offset,length,queue_wait_us,pread_us,sleep_us   (t = slot start, relative to mount)
 *
 * BUILD (image pi-gemma4-build:fuse, docker/Dockerfile.fuse)
 *   gcc -std=c11 -D_GNU_SOURCE -Wall -Wextra -Werror -O2 -o fusethrottle \
 *       tools/fusethrottle/fusethrottle.c $(pkg-config --cflags --libs fuse3) -lpthread
 *
 * EXIT CODES  0 clean unmount; 1 usage / runtime error; 3 O_DIRECT refused (no --allow-buffered).
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define FUSE_USE_VERSION 31

#include <errno.h>
#include <fcntl.h>
#include <fuse.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define FT_VERSION "1.0.0"
#define STATS_NAME ".stats"
#define STATS_SIZE 4096

typedef struct {
    const char *backing;
    const char *mountpoint;
    char name[256];
    double rate_bps;          /* 0 = unthrottled */
    double op_latency_us;
    uint64_t align;
    uint64_t max_readahead;
    uint64_t max_read;        /* 0 = kernel/libfuse default */
    const char *trace;
    int allow_buffered;
    int kernel_cache;
    int single_thread;
} opts_t;

static opts_t g_o;
static int g_fd = -1;
static int g_direct = 0;
static uint64_t g_size = 0;
static struct stat g_st;
static int64_t g_t0_ns = 0;

static pthread_mutex_t g_mx = PTHREAD_MUTEX_INITIALIZER;
static int64_t g_vclock_ns = 0;
static FILE *g_trace = NULL;

/* counters (under g_mx) */
static uint64_t c_requests, c_bytes_served, c_backing_bytes, c_errors, c_short_reads;
static int64_t c_queue_wait_ns, c_sleep_ns, c_pread_ns, c_overrun_ns, c_max_queue_wait_ns;
static uint64_t c_hist[8]; /* <=4k, <=16k, <=64k, <=128k, <=256k, <=1M, <=4M, >4M */
static uint64_t c_stats_reads;
/* negotiated in init */
static unsigned n_max_read, n_max_readahead, n_max_background, n_max_write, n_proto_minor;
static int n_async_read;

static __thread void *t_buf = NULL;
static __thread size_t t_buf_len = 0;

static int64_t mono_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void sleep_until(int64_t t_ns)
{
    struct timespec ts;
    ts.tv_sec = (time_t)(t_ns / 1000000000LL);
    ts.tv_nsec = (long)(t_ns % 1000000000LL);
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) {
    }
}

static void die(int code, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "fusethrottle: error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(code);
}

static int hist_bin(uint64_t n)
{
    if (n <= 4096) return 0;
    if (n <= 16384) return 1;
    if (n <= 65536) return 2;
    if (n <= 131072) return 3;
    if (n <= 262144) return 4;
    if (n <= 1048576) return 5;
    if (n <= 4194304) return 6;
    return 7;
}

static int is_model(const char *path)
{
    return path[0] == '/' && strcmp(path + 1, g_o.name) == 0;
}

static int is_stats(const char *path)
{
    return strcmp(path, "/" STATS_NAME) == 0;
}

static int ft_getattr(const char *path, struct stat *st, struct fuse_file_info *fi)
{
    (void)fi;
    memset(st, 0, sizeof(*st));
    st->st_uid = 0;
    st->st_gid = 0;
    st->st_atim = g_st.st_atim;
    st->st_mtim = g_st.st_mtim;
    st->st_ctim = g_st.st_ctim;
    if (strcmp(path, "/") == 0) {
        st->st_mode = S_IFDIR | 0555;
        st->st_nlink = 2;
        return 0;
    }
    if (is_model(path)) {
        st->st_mode = S_IFREG | 0444;
        st->st_nlink = 1;
        st->st_size = (off_t)g_size;
        st->st_blksize = 4096;
        st->st_blocks = (blkcnt_t)((g_size + 511) / 512);
        return 0;
    }
    if (is_stats(path)) {
        st->st_mode = S_IFREG | 0444;
        st->st_nlink = 1;
        st->st_size = STATS_SIZE;
        return 0;
    }
    return -ENOENT;
}

static int ft_readdir(const char *path, void *buf, fuse_fill_dir_t filler, off_t off,
                      struct fuse_file_info *fi, enum fuse_readdir_flags flags)
{
    (void)off;
    (void)fi;
    (void)flags;
    if (strcmp(path, "/") != 0) return -ENOENT;
    filler(buf, ".", NULL, 0, 0);
    filler(buf, "..", NULL, 0, 0);
    filler(buf, g_o.name, NULL, 0, 0);
    filler(buf, STATS_NAME, NULL, 0, 0);
    return 0;
}

static int ft_open(const char *path, struct fuse_file_info *fi)
{
    if ((fi->flags & O_ACCMODE) != O_RDONLY) return -EROFS;
    if (is_model(path)) {
        fi->keep_cache = g_o.kernel_cache ? 1 : 0;
        return 0;
    }
    if (is_stats(path)) {
        fi->direct_io = 1;
        fi->keep_cache = 0;
        return 0;
    }
    return -ENOENT;
}

static int stats_json(char *out, size_t cap)
{
    pthread_mutex_lock(&g_mx);
    const int64_t now = mono_ns();
    int n = snprintf(out, cap,
        "{\"tool\":\"fusethrottle\",\"version\":\"" FT_VERSION "\",\"pid\":%d,"
        "\"backing\":\"%s\",\"name\":\"%s\",\"size\":%" PRIu64 ","
        "\"rate_bps\":%.0f,\"op_latency_us\":%.1f,\"align\":%" PRIu64 ",\"direct_io\":%s,"
        "\"kernel_cache\":%s,\"single_thread\":%s,\"max_readahead_opt\":%" PRIu64 ",\"max_read_opt\":%" PRIu64 ","
        "\"conn\":{\"max_read\":%u,\"max_readahead\":%u,\"max_background\":%u,\"max_write\":%u,"
        "\"proto_minor\":%u,\"async_read\":%s},"
        "\"t_mono_ns\":%" PRId64 ",\"uptime_s\":%.3f,"
        "\"requests\":%" PRIu64 ",\"bytes_served\":%" PRIu64 ",\"backing_bytes\":%" PRIu64 ","
        "\"short_reads\":%" PRIu64 ",\"errors\":%" PRIu64 ",\"stats_reads\":%" PRIu64 ","
        "\"queue_wait_ns\":%" PRId64 ",\"max_queue_wait_ns\":%" PRId64 ",\"sleep_ns\":%" PRId64 ","
        "\"pread_ns\":%" PRId64 ",\"overrun_ns\":%" PRId64 ","
        "\"size_hist\":{\"le4k\":%" PRIu64 ",\"le16k\":%" PRIu64 ",\"le64k\":%" PRIu64 ",\"le128k\":%" PRIu64
        ",\"le256k\":%" PRIu64 ",\"le1m\":%" PRIu64 ",\"le4m\":%" PRIu64 ",\"gt4m\":%" PRIu64 "}}",
        (int)getpid(), g_o.backing, g_o.name, g_size, g_o.rate_bps, g_o.op_latency_us, g_o.align,
        g_direct ? "true" : "false", g_o.kernel_cache ? "true" : "false",
        g_o.single_thread ? "true" : "false", g_o.max_readahead, g_o.max_read,
        n_max_read, n_max_readahead, n_max_background, n_max_write, n_proto_minor,
        n_async_read ? "true" : "false",
        now, (double)(now - g_t0_ns) / 1e9,
        c_requests, c_bytes_served, c_backing_bytes, c_short_reads, c_errors, c_stats_reads,
        c_queue_wait_ns, c_max_queue_wait_ns, c_sleep_ns, c_pread_ns, c_overrun_ns,
        c_hist[0], c_hist[1], c_hist[2], c_hist[3], c_hist[4], c_hist[5], c_hist[6], c_hist[7]);
    c_stats_reads++;
    pthread_mutex_unlock(&g_mx);
    return n;
}

static int ensure_buf(size_t need)
{
    if (t_buf_len >= need) return 0;
    free(t_buf);
    t_buf = NULL;
    t_buf_len = 0;
    size_t cap = need < (1u << 20) ? (1u << 20) : need;
    cap = (cap + g_o.align - 1) / g_o.align * g_o.align;
    if (posix_memalign(&t_buf, (size_t)g_o.align, cap) != 0) return -1;
    t_buf_len = cap;
    return 0;
}

static int ft_read(const char *path, char *buf, size_t size, off_t off, struct fuse_file_info *fi)
{
    (void)fi;
    if (is_stats(path)) {
        char tmp[STATS_SIZE + 1];
        int n = stats_json(tmp, sizeof(tmp));
        if (n < 0 || n >= STATS_SIZE) n = 0;
        memset(tmp + n, ' ', (size_t)(STATS_SIZE - n));
        tmp[STATS_SIZE - 1] = '\n';
        if (off >= STATS_SIZE) return 0;
        size_t take = (size_t)(STATS_SIZE - off) < size ? (size_t)(STATS_SIZE - off) : size;
        memcpy(buf, tmp + off, take);
        return (int)take;
    }
    if (!is_model(path)) return -ENOENT;
    if (off < 0) return -EINVAL;
    if ((uint64_t)off >= g_size || size == 0) return 0;
    if ((uint64_t)off + size > g_size) size = (size_t)(g_size - (uint64_t)off);

    const uint64_t a = g_o.align;
    const uint64_t a_off = (uint64_t)off / a * a;
    const uint64_t a_end = ((uint64_t)off + size + a - 1) / a * a;
    const size_t a_len = (size_t)(a_end - a_off);
    if (ensure_buf(a_len) != 0) return -ENOMEM;

    /* reserve a slot on the emulated serial device */
    int64_t now = mono_ns(), start = now, deadline = now;
    pthread_mutex_lock(&g_mx);
    if (g_o.rate_bps > 0) {
        start = g_vclock_ns > now ? g_vclock_ns : now;
        const int64_t dur = (int64_t)(g_o.op_latency_us * 1000.0) + (int64_t)((double)a_len * 1e9 / g_o.rate_bps);
        deadline = start + dur;
        g_vclock_ns = deadline;
    }
    const int64_t qwait = start - now;
    c_queue_wait_ns += qwait;
    if (qwait > c_max_queue_wait_ns) c_max_queue_wait_ns = qwait;
    pthread_mutex_unlock(&g_mx);

    /* the read itself (O_DIRECT: aligned offset, length and buffer) */
    const int64_t p0 = mono_ns();
    size_t got = 0;
    int err = 0;
    while (got < a_len) {
        ssize_t r = pread(g_fd, (char *)t_buf + got, a_len - got, (off_t)(a_off + got));
        if (r < 0) {
            if (errno == EINTR) continue;
            err = errno;
            break;
        }
        if (r == 0) break; /* EOF (the aligned tail may extend past the end) */
        got += (size_t)r;
    }
    const int64_t p1 = mono_ns();

    int64_t slept = 0;
    if (g_o.rate_bps > 0 && p1 < deadline) {
        sleep_until(deadline);
        slept = deadline - p1;
    }
    size_t avail = 0;
    if (!err && got > (uint64_t)off - a_off) {
        avail = got - (size_t)((uint64_t)off - a_off);
        if (avail > size) avail = size;
        memcpy(buf, (char *)t_buf + ((uint64_t)off - a_off), avail);
    }

    pthread_mutex_lock(&g_mx);
    if (g_o.rate_bps > 0 && p1 > deadline) {
        c_overrun_ns += p1 - deadline;
        if (p1 > g_vclock_ns) g_vclock_ns = p1; /* no catch-up burst after an overrun */
    }
    c_requests++;
    c_backing_bytes += got;
    c_bytes_served += avail;
    c_pread_ns += p1 - p0;
    c_sleep_ns += slept;
    c_hist[hist_bin(a_len)]++;
    if (err) c_errors++;
    if (!err && avail < size) c_short_reads++;
    if (g_trace) {
        fprintf(g_trace, "%" PRId64 ",%" PRIu64 ",%zu,%" PRId64 ",%" PRId64 ",%" PRId64 "\n",
                (start - g_t0_ns) / 1000, a_off, a_len, qwait / 1000, (p1 - p0) / 1000, slept / 1000);
    }
    pthread_mutex_unlock(&g_mx);
    if (err) return -err;
    return (int)avail;
}

static void *ft_init(struct fuse_conn_info *conn, struct fuse_config *cfg)
{
    cfg->kernel_cache = g_o.kernel_cache;
    cfg->auto_cache = 0;
    cfg->direct_io = 0;
    cfg->entry_timeout = 86400.0;
    cfg->attr_timeout = 86400.0;
    cfg->negative_timeout = 0.0;
    cfg->use_ino = 0;
    cfg->nullpath_ok = 0;
    if (g_o.max_readahead > 0) conn->max_readahead = (unsigned)g_o.max_readahead;
    if (g_o.max_read > 0) conn->max_read = (unsigned)g_o.max_read;
    n_max_read = conn->max_read;
    n_max_readahead = conn->max_readahead;
    n_max_background = conn->max_background;
    n_max_write = conn->max_write;
    n_proto_minor = conn->proto_minor;
    n_async_read = (conn->want & FUSE_CAP_ASYNC_READ) ? 1 : 0;
    g_t0_ns = mono_ns();
    printf("{\"event\":\"ready\",\"pid\":%d,\"mountpoint\":\"%s\",\"name\":\"%s\",\"size\":%" PRIu64
           ",\"rate_bps\":%.0f,\"direct_io\":%s,\"max_readahead\":%u,\"max_read\":%u,\"max_background\":%u}\n",
           (int)getpid(), g_o.mountpoint, g_o.name, g_size, g_o.rate_bps, g_direct ? "true" : "false",
           n_max_readahead, n_max_read, n_max_background);
    fflush(stdout);
    return NULL;
}

static void ft_destroy(void *data)
{
    (void)data;
    pthread_mutex_lock(&g_mx);
    if (g_trace) {
        fflush(g_trace);
        fclose(g_trace);
        g_trace = NULL;
    }
    pthread_mutex_unlock(&g_mx);
    char tmp[STATS_SIZE + 1];
    if (stats_json(tmp, sizeof(tmp)) > 0) {
        printf("{\"event\":\"exit\",\"stats\":%s}\n", tmp);
        fflush(stdout);
    }
}

static const struct fuse_operations ft_ops = {
    .getattr = ft_getattr,
    .readdir = ft_readdir,
    .open = ft_open,
    .read = ft_read,
    .init = ft_init,
    .destroy = ft_destroy,
};

static void usage(FILE *f)
{
    fprintf(f,
        "usage: fusethrottle --backing FILE --mountpoint DIR [--name NAME] [--rate-bps N]\n"
        "                    [--op-latency-us U] [--align N] [--max-readahead N] [--max-read N]\n"
        "                    [--trace FILE] [--allow-buffered] [--no-kernel-cache] [--single-thread]\n"
        "  --rate-bps N        emulated device read rate in bytes/s (default 91600000; 0 = unthrottled)\n"
        "  --op-latency-us U   fixed latency added per backing request (default 0)\n"
        "  --align N           O_DIRECT alignment of offset/length/buffer (default 4096)\n"
        "  --max-readahead N   FUSE max_readahead in bytes (default 131072; the kernel can only lower\n"
        "                      the 128 KiB default bdi readahead)\n"
        "  --max-read N        FUSE max_read in bytes (default: libfuse/kernel default)\n"
        "  --trace FILE        CSV per backing read: t_us,offset,length,queue_wait_us,pread_us,sleep_us\n");
}

static uint64_t parse_u64(const char *s, const char *what)
{
    char *e = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &e, 10);
    if (errno || e == s || *e) die(1, "%s wants an unsigned integer, got '%s'", what, s);
    return (uint64_t)v;
}

static double parse_dbl(const char *s, const char *what)
{
    char *e = NULL;
    errno = 0;
    double v = strtod(s, &e);
    if (errno || e == s || *e || v < 0) die(1, "%s wants a number >= 0, got '%s'", what, s);
    return v;
}

int main(int argc, char **argv)
{
    memset(&g_o, 0, sizeof(g_o));
    g_o.rate_bps = 91600000.0;
    g_o.align = 4096;
    g_o.max_readahead = 131072;
    g_o.kernel_cache = 1;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
#define NEEDV() do { if (!v) die(1, "%s needs a value", a); i++; } while (0)
        if (!strcmp(a, "--backing")) { NEEDV(); g_o.backing = v; }
        else if (!strcmp(a, "--mountpoint")) { NEEDV(); g_o.mountpoint = v; }
        else if (!strcmp(a, "--name")) { NEEDV(); snprintf(g_o.name, sizeof(g_o.name), "%s", v); }
        else if (!strcmp(a, "--rate-bps")) { NEEDV(); g_o.rate_bps = parse_dbl(v, a); }
        else if (!strcmp(a, "--op-latency-us")) { NEEDV(); g_o.op_latency_us = parse_dbl(v, a); }
        else if (!strcmp(a, "--align")) { NEEDV(); g_o.align = parse_u64(v, a); }
        else if (!strcmp(a, "--max-readahead")) { NEEDV(); g_o.max_readahead = parse_u64(v, a); }
        else if (!strcmp(a, "--max-read")) { NEEDV(); g_o.max_read = parse_u64(v, a); }
        else if (!strcmp(a, "--trace")) { NEEDV(); g_o.trace = v; }
        else if (!strcmp(a, "--allow-buffered")) g_o.allow_buffered = 1;
        else if (!strcmp(a, "--no-kernel-cache")) g_o.kernel_cache = 0;
        else if (!strcmp(a, "--single-thread")) g_o.single_thread = 1;
        else if (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(stdout); return 0; }
        else if (!strcmp(a, "--version")) { printf("fusethrottle " FT_VERSION "\n"); return 0; }
        else { usage(stderr); die(1, "unknown argument '%s'", a); }
#undef NEEDV
    }
    if (!g_o.backing || !g_o.mountpoint) {
        usage(stderr);
        die(1, "--backing and --mountpoint are required");
    }
    if (g_o.align < 512 || (g_o.align & (g_o.align - 1))) die(1, "--align must be a power of two >= 512");
    if (!g_o.name[0]) {
        const char *s = strrchr(g_o.backing, '/');
        snprintf(g_o.name, sizeof(g_o.name), "%s", s ? s + 1 : g_o.backing);
    }
    if (strchr(g_o.name, '/') || !strcmp(g_o.name, STATS_NAME)) die(1, "invalid --name '%s'", g_o.name);

    g_fd = open(g_o.backing, O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (g_fd >= 0) {
        g_direct = 1;
    } else {
        const int e = errno;
        if (!g_o.allow_buffered) die(3, "O_DIRECT open of %s failed: %s (use --allow-buffered to accept a "
                                        "buffered backing read, which caches a second copy)", g_o.backing, strerror(e));
        g_fd = open(g_o.backing, O_RDONLY | O_CLOEXEC);
        if (g_fd < 0) die(1, "cannot open %s: %s", g_o.backing, strerror(errno));
        fprintf(stderr, "fusethrottle: warning: O_DIRECT refused (%s); buffered backing reads\n", strerror(e));
    }
    if (fstat(g_fd, &g_st) != 0) die(1, "fstat %s: %s", g_o.backing, strerror(errno));
    if (!S_ISREG(g_st.st_mode)) die(1, "%s is not a regular file", g_o.backing);
    g_size = (uint64_t)g_st.st_size;
    /* drop any cached pages of the backing file (targeted; never global) */
    {
        int bfd = open(g_o.backing, O_RDONLY | O_CLOEXEC);
        if (bfd >= 0) {
            (void)posix_fadvise(bfd, 0, 0, POSIX_FADV_DONTNEED);
            close(bfd);
        }
    }
    if (g_o.trace) {
        g_trace = fopen(g_o.trace, "w");
        if (!g_trace) die(1, "cannot open --trace %s: %s", g_o.trace, strerror(errno));
        setvbuf(g_trace, NULL, _IOFBF, 1 << 20);
        fprintf(g_trace, "t_us,offset,length,queue_wait_us,pread_us,sleep_us\n");
    }

    /* fuse_main argv: foreground, read-only, fsname for /proc/mounts */
    char *fargv[16];
    int fargc = 0;
    static char optbuf[512];
    snprintf(optbuf, sizeof(optbuf), "ro,fsname=fusethrottle,subtype=fusethrottle%s",
             g_o.max_read ? "" : "");
    if (g_o.max_read) {
        size_t l = strlen(optbuf);
        snprintf(optbuf + l, sizeof(optbuf) - l, ",max_read=%" PRIu64, g_o.max_read);
    }
    fargv[fargc++] = argv[0];
    fargv[fargc++] = (char *)g_o.mountpoint;
    fargv[fargc++] = "-f";
    if (g_o.single_thread) fargv[fargc++] = "-s";
    fargv[fargc++] = "-o";
    fargv[fargc++] = optbuf;
    fargv[fargc] = NULL;
    const int rc = fuse_main(fargc, fargv, &ft_ops, NULL);
    close(g_fd);
    return rc == 0 ? 0 : 1;
}
