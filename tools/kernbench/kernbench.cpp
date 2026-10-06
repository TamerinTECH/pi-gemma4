/*
 * kernbench.cpp - per-quantization-type matrix x vector throughput through the REAL ggml CPU code path
 * (pi-gemma4, comparisons workflow stage X4 "resident bottleneck", 2026-10-06).
 *
 * WHY
 *   Resident decode of Gemma 4 26B-A4B on the Pi 5 reads ~3.0 GB of weights per token (UD-Q3_K_M) and runs at
 *   ~3.1-3.3 tok/s, i.e. ~9.7 GB/s. To say whether that is the DRAM bandwidth or the arithmetic of a kernel
 *   (IQ3_XXS / IQ4_NL / Q8_0 / Q6_K ...), each kernel has to be timed twice on the same board:
 *     stream : the weight matrix is far larger than the caches (default 512 MiB)  -> what decode sees
 *     cached : the weight matrix fits in the caches (default 384 KiB), repeated   -> arithmetic only
 *   If stream == cached the kernel is compute-bound; if stream << cached and stream ~ membw "read" it is
 *   memory-bound. A third regime (cached, 32 columns) approximates prefill, where each row is reused.
 *
 * METHOD
 *   The tensors are handed to ggml_mul_mat() and executed with ggml_graph_compute_with_ctx(), i.e. the same
 *   dispatch llama.cpp uses with -nr (no repack): the activation is converted to the type's vec_dot_type, then
 *   llamafile_sgemm (tinyBLAS: Q8_0 / Q4_0 on NEON+dotprod) or the per-row ggml_vec_dot_* kernel
 *   (IQ3_XXS, IQ4_NL, IQ4_XS, Q6_K on ARM) runs on n_threads threads (OpenMP, as in llama-server).
 *   Weights: 32 rows of N(0,1) floats are quantized with ggml_quantize_chunk() and tiled to fill the matrix
 *   (IQ quantization is slow; kernel cost does not depend on the values). Row length = --n-in (2816 = the
 *   model's hidden size; "TYPE@704" selects another row length for one type).
 *   Every (type, regime, threads) cell: 1 untimed pass + --reps timed passes; best and median are reported.
 *   GB/s = bytes of the weight matrix processed per second (decimal, 1e9), the same unit as tools/membw.
 *   The llama.cpp default-repack kernels (CPU_REPACK buffer) are NOT covered: that conversion is internal to
 *   the CPU backend's buffer type.
 *
 * SAFETY
 *   Refuses to run when MemAvailable - stream arena < --min-avail-mb (default 3072 MiB): the real Pi runs
 *   services that must not be squeezed. No mlock, no huge pages, no thread pinning.
 *
 * BUILD (aarch64, against the static libs of the llama.cpp a7b94df build in Docker volume pigemma-arm64-llama)
 *   pc/x4/build_kernbench_arm64.sh        -> tools/build-pi-arm64/kernbench
 * USAGE
 *   kernbench [--types Q8_0,Q6_K,IQ3_XXS,IQ4_NL,IQ4_NL@704,IQ4_XS,Q4_0] [--threads 1,2,3,4]
 *             [--stream-mb 512] [--cache-kb 384] [--cache-total-mb 256] [--n-in 2816] [--batch-cols 32]
 *             [--reps 5] [--min-avail-mb 3072] [--json]
 * EXIT CODES  0 ok; 1 refused (memory) / allocation failure; 2 usage; 3 runtime error.
 */
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#define KERNBENCH_VERSION "1.0.0"

struct type_spec {
    std::string name;
    ggml_type   type;
    int         n_in;
};

struct cell_result {
    std::string type;
    std::string regime;
    std::string vec_dot_type;
    int         n_in;
    int         threads;
    int64_t     n_rows;
    int         n_cols;
    int         n_nodes;
    size_t      row_bytes;
    double      bytes_per_pass;
    double      pairs_per_pass; // weight x activation products per pass
    double      best_s;
    double      median_s;
    double      checksum;
};

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

static bool parse_type(const std::string & s, ggml_type & out) {
    static const struct { const char * n; ggml_type t; } tbl[] = {
        {"F32", GGML_TYPE_F32},       {"F16", GGML_TYPE_F16},     {"Q4_0", GGML_TYPE_Q4_0},
        {"Q8_0", GGML_TYPE_Q8_0},     {"Q4_K", GGML_TYPE_Q4_K},   {"Q5_K", GGML_TYPE_Q5_K},
        {"Q6_K", GGML_TYPE_Q6_K},     {"Q3_K", GGML_TYPE_Q3_K},   {"IQ3_XXS", GGML_TYPE_IQ3_XXS},
        {"IQ3_S", GGML_TYPE_IQ3_S},   {"IQ4_NL", GGML_TYPE_IQ4_NL}, {"IQ4_XS", GGML_TYPE_IQ4_XS},
    };
    for (const auto & e : tbl) {
        if (s == e.n) {
            out = e.t;
            return true;
        }
    }
    return false;
}

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t                   a = 0;
    while (a <= s.size()) {
        size_t b = s.find(sep, a);
        if (b == std::string::npos) {
            b = s.size();
        }
        if (b > a) {
            out.push_back(s.substr(a, b - a));
        }
        a = b + 1;
    }
    return out;
}

static long mem_available_kb() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) {
        return -1;
    }
    char line[256];
    long v = -1;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %ld kB", &v) == 1) {
            break;
        }
    }
    fclose(f);
    return v;
}

// One cell: n_nodes x mul_mat(A[n_in, n_rows] of `type`, B[n_in, n_cols] F32) in one graph, timed reps times.
static bool run_cell(ggml_type type, void * a_data, int n_in, int64_t n_rows, int n_cols, int n_nodes, int n_threads,
                     int reps, const std::vector<float> & act, cell_result & res) {
    const size_t result_bytes = (size_t) n_rows * (size_t) n_cols * sizeof(float);
    size_t       mem          = (size_t) 32 << 20;
    mem += (size_t) n_nodes * (ggml_tensor_overhead() + result_bytes + 512);
    mem += ggml_graph_overhead_custom((size_t) n_nodes + 16, false);
    mem += (size_t) (reps + 2) * ((size_t) 8 << 20); // work buffers: one per graph_compute_with_ctx call
    mem += (size_t) n_in * (size_t) n_cols * sizeof(float);

    ggml_init_params ip;
    ip.mem_size   = mem;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        fprintf(stderr, "kernbench: ggml_init failed (%zu bytes)\n", mem);
        return false;
    }
    ggml_set_no_alloc(ctx, true);
    ggml_tensor * A = ggml_new_tensor_2d(ctx, type, n_in, n_rows);
    A->data         = a_data;
    ggml_set_no_alloc(ctx, false);
    ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, n_cols);
    {
        float * b = (float *) B->data;
        for (int c = 0; c < n_cols; c++) {
            for (int i = 0; i < n_in; i++) {
                b[(size_t) c * n_in + i] = act[((size_t) c * 131 + i) % act.size()];
            }
        }
    }
    ggml_cgraph * gf   = ggml_new_graph_custom(ctx, (size_t) n_nodes + 16, false);
    ggml_tensor * last = nullptr;
    for (int i = 0; i < n_nodes; i++) {
        last = ggml_mul_mat(ctx, A, B);
        ggml_build_forward_expand(gf, last);
    }

    std::vector<double> times;
    for (int r = 0; r <= reps; r++) {
        const double t0 = now_s();
        if (ggml_graph_compute_with_ctx(ctx, gf, n_threads) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "kernbench: graph compute failed\n");
            ggml_free(ctx);
            return false;
        }
        const double t1 = now_s();
        if (r > 0) {
            times.push_back(t1 - t0);
        }
    }
    double        sum = 0.0;
    const float * out = (const float *) last->data;
    for (int64_t i = 0; i < n_rows * n_cols; i++) {
        sum += out[i];
    }
    std::sort(times.begin(), times.end());
    res.best_s         = times.front();
    res.median_s       = times[times.size() / 2];
    res.checksum       = sum;
    res.n_rows         = n_rows;
    res.n_cols         = n_cols;
    res.n_nodes        = n_nodes;
    res.threads        = n_threads;
    res.n_in           = n_in;
    res.row_bytes      = ggml_row_size(type, n_in);
    res.bytes_per_pass = (double) n_nodes * (double) n_rows * (double) res.row_bytes;
    res.pairs_per_pass = (double) n_nodes * (double) n_rows * (double) n_in * (double) n_cols;
    res.type           = ggml_type_name(type);
    res.vec_dot_type   = ggml_type_name(ggml_get_type_traits_cpu(type)->vec_dot_type);
    ggml_free(ctx);
    return true;
}

static void usage() {
    fprintf(stderr,
            "usage: kernbench [--types Q8_0,Q6_K,IQ3_XXS,IQ4_NL,IQ4_NL@704,IQ4_XS,Q4_0] [--threads 1,2,3,4]\n"
            "                 [--stream-mb 512] [--cache-kb 384] [--cache-total-mb 256] [--n-in 2816]\n"
            "                 [--batch-cols 32] [--reps 5] [--min-avail-mb 3072] [--json]\n");
}

int main(int argc, char ** argv) {
    std::string types_arg     = "Q8_0,Q6_K,IQ3_XXS,IQ4_NL,IQ4_NL@704,IQ4_XS,Q4_0";
    std::string threads_arg   = "1,2,3,4";
    long        stream_mb     = 512;
    long        cache_kb      = 384;
    long        cache_total_mb = 256;
    int         n_in_default  = 2816;
    int         batch_cols    = 32;
    int         reps          = 5;
    long        min_avail_mb  = 3072;
    bool        json          = false;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto              need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "kernbench: %s needs a value\n", name);
                usage();
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--types") {
            types_arg = need("--types");
        } else if (a == "--threads") {
            threads_arg = need("--threads");
        } else if (a == "--stream-mb") {
            stream_mb = atol(need("--stream-mb"));
        } else if (a == "--cache-kb") {
            cache_kb = atol(need("--cache-kb"));
        } else if (a == "--cache-total-mb") {
            cache_total_mb = atol(need("--cache-total-mb"));
        } else if (a == "--n-in") {
            n_in_default = atoi(need("--n-in"));
        } else if (a == "--batch-cols") {
            batch_cols = atoi(need("--batch-cols"));
        } else if (a == "--reps") {
            reps = atoi(need("--reps"));
        } else if (a == "--min-avail-mb") {
            min_avail_mb = atol(need("--min-avail-mb"));
        } else if (a == "--json") {
            json = true;
        } else if (a == "--version") {
            printf("kernbench %s\n", KERNBENCH_VERSION);
            return 0;
        } else {
            usage();
            return a == "--help" || a == "-h" ? 0 : 2;
        }
    }
    if (stream_mb < 1 || cache_kb < 16 || reps < 1 || n_in_default < 256 || n_in_default % 256 != 0 || batch_cols < 0 ||
        cache_total_mb < 1) {
        fprintf(stderr, "kernbench: bad sizes (n-in must be a multiple of 256)\n");
        return 2;
    }

    std::vector<type_spec> specs;
    for (const auto & tok : split(types_arg, ',')) {
        type_spec    sp;
        const size_t at = tok.find('@');
        sp.name         = tok.substr(0, at);
        sp.n_in         = at == std::string::npos ? n_in_default : atoi(tok.substr(at + 1).c_str());
        if (!parse_type(sp.name, sp.type) || sp.n_in < 256 || sp.n_in % 256 != 0) {
            fprintf(stderr, "kernbench: unknown type or bad row length '%s'\n", tok.c_str());
            return 2;
        }
        specs.push_back(sp);
    }
    std::vector<int> threads;
    for (const auto & tok : split(threads_arg, ',')) {
        const int t = atoi(tok.c_str());
        if (t < 1 || t > 256) {
            fprintf(stderr, "kernbench: bad thread count '%s'\n", tok.c_str());
            return 2;
        }
        threads.push_back(t);
    }
    if (specs.empty() || threads.empty()) {
        usage();
        return 2;
    }

    const size_t stream_bytes = (size_t) stream_mb << 20;
    const long   avail_kb     = mem_available_kb();
    if (avail_kb >= 0 && avail_kb - (long) (stream_bytes >> 10) < min_avail_mb * 1024) {
        fprintf(stderr, "kernbench: refused: MemAvailable %ld kB - arena %zu kB < floor %ld kB\n", avail_kb,
                stream_bytes >> 10, min_avail_mb * 1024);
        return 1;
    }

    ggml_time_init();
    ggml_cpu_init();

    std::mt19937                    rng(12345);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float>              act(8192);
    for (auto & v : act) {
        v = gauss(rng);
    }

    void * arena = nullptr;
    if (posix_memalign(&arena, 64, stream_bytes) != 0 || arena == nullptr) {
        fprintf(stderr, "kernbench: cannot allocate %zu bytes\n", stream_bytes);
        return 1;
    }

    std::vector<cell_result> results;
    const int                gen_rows = 32;
    for (const auto & sp : specs) {
        const size_t row_bytes = ggml_row_size(sp.type, sp.n_in);
        // 1. quantize gen_rows random rows, 2. tile them over the whole arena (also first-touches every page)
        std::vector<float> src((size_t) gen_rows * sp.n_in);
        for (auto & v : src) {
            v = gauss(rng);
        }
        std::vector<uint8_t> gen((size_t) gen_rows * row_bytes);
        const double         tq0 = now_s();
        const size_t         got = ggml_quantize_chunk(sp.type, src.data(), gen.data(), 0, gen_rows, sp.n_in, nullptr);
        const double         tq1 = now_s();
        if (got != gen.size()) {
            fprintf(stderr, "kernbench: quantize %s gave %zu bytes, expected %zu\n", sp.name.c_str(), got, gen.size());
            free(arena);
            return 3;
        }
        const int64_t rows_stream = (int64_t) (stream_bytes / row_bytes);
        for (int64_t r = 0; r < rows_stream; r += gen_rows) {
            const int64_t n = std::min<int64_t>(gen_rows, rows_stream - r);
            memcpy((uint8_t *) arena + (size_t) r * row_bytes, gen.data(), (size_t) n * row_bytes);
        }
        int64_t rows_cache = (int64_t) (((size_t) cache_kb << 10) / row_bytes);
        rows_cache         = std::max<int64_t>(16, rows_cache - rows_cache % 16);
        const double cache_bytes = (double) rows_cache * (double) row_bytes;
        int          nodes_cache = (int) std::min<double>(1024.0, std::max(8.0, ((double) cache_total_mb * 1048576.0) / cache_bytes));
        fprintf(stderr, "kernbench: %s n_in=%d row=%zu B quantize %.2f s; stream rows %" PRId64 ", cache rows %" PRId64 " x %d nodes\n",
                sp.name.c_str(), sp.n_in, row_bytes, tq1 - tq0, rows_stream, rows_cache, nodes_cache);

        for (int t : threads) {
            cell_result r;
            r.regime = "stream";
            if (!run_cell(sp.type, arena, sp.n_in, rows_stream, 1, 1, t, reps, act, r)) {
                free(arena);
                return 3;
            }
            results.push_back(r);
            r.regime = "cached";
            if (!run_cell(sp.type, arena, sp.n_in, rows_cache, 1, nodes_cache, t, reps, act, r)) {
                free(arena);
                return 3;
            }
            results.push_back(r);
            if (batch_cols > 1) {
                const int nodes_b = std::max(2, nodes_cache / batch_cols);
                r.regime          = "cached_batch";
                if (!run_cell(sp.type, arena, sp.n_in, rows_cache, batch_cols, nodes_b, t, reps, act, r)) {
                    free(arena);
                    return 3;
                }
                results.push_back(r);
            }
        }
    }
    free(arena);

    struct utsname un;
    memset(&un, 0, sizeof(un));
    uname(&un);
    if (json) {
        printf("{\n \"tool\": \"kernbench\", \"version\": \"%s\", \"host\": \"%s\", \"release\": \"%s\", \"page_size\": %ld,\n",
               KERNBENCH_VERSION, un.machine, un.release, sysconf(_SC_PAGESIZE));
        printf(" \"nproc_online\": %ld, \"mem_available_kb_at_start\": %ld,\n", sysconf(_SC_NPROCESSORS_ONLN), avail_kb);
        printf(" \"cpu_features\": {\"neon\": %d, \"dotprod\": %d, \"matmul_int8\": %d, \"sve\": %d, \"avx2\": %d, \"avx512\": %d},\n",
               ggml_cpu_has_neon(), ggml_cpu_has_dotprod(), ggml_cpu_has_matmul_int8(), ggml_cpu_has_sve(),
               ggml_cpu_has_avx2(), ggml_cpu_has_avx512());
        printf(" \"stream_mb\": %ld, \"cache_kb\": %ld, \"cache_total_mb\": %ld, \"batch_cols\": %d, \"reps\": %d,\n", stream_mb,
               cache_kb, cache_total_mb, batch_cols, reps);
        printf(" \"units\": \"GB/s = 1e9 bytes of the weight matrix per second; pairs = weight x activation products\",\n");
        printf(" \"path\": \"ggml_mul_mat via ggml_graph_compute_with_ctx (no repack): llamafile_sgemm or ggml_vec_dot_*\",\n");
        printf(" \"results\": [\n");
        for (size_t i = 0; i < results.size(); i++) {
            const auto & r = results[i];
            printf("  {\"type\": \"%s\", \"n_in\": %d, \"vec_dot_type\": \"%s\", \"regime\": \"%s\", \"threads\": %d, "
                   "\"n_rows\": %" PRId64 ", \"n_cols\": %d, \"n_nodes\": %d, \"row_bytes\": %zu, \"bytes_per_pass\": %.0f, "
                   "\"best_s\": %.6f, \"median_s\": %.6f, \"gbps_best\": %.3f, \"gbps_median\": %.3f, "
                   "\"pairs_per_s_best\": %.4e, \"checksum\": %.6e}%s\n",
                   r.type.c_str(), r.n_in, r.vec_dot_type.c_str(), r.regime.c_str(), r.threads, r.n_rows, r.n_cols, r.n_nodes,
                   r.row_bytes, r.bytes_per_pass, r.best_s, r.median_s, r.bytes_per_pass / r.best_s / 1e9,
                   r.bytes_per_pass / r.median_s / 1e9, r.pairs_per_pass / r.best_s, r.checksum,
                   i + 1 < results.size() ? "," : "");
        }
        printf(" ]\n}\n");
    } else {
        printf("kernbench %s on %s %s: stream %ld MiB, cache %ld KiB, reps %d (GB/s of weight bytes, best pass)\n",
               KERNBENCH_VERSION, un.machine, un.release, stream_mb, cache_kb, reps);
        printf("%-9s %5s %-13s %3s %12s %12s %14s\n", "type", "n_in", "regime", "thr", "GB/s best", "GB/s median", "pairs/s best");
        for (const auto & r : results) {
            printf("%-9s %5d %-13s %3d %12.3f %12.3f %14.4e\n", r.type.c_str(), r.n_in, r.regime.c_str(), r.threads,
                   r.bytes_per_pass / r.best_s / 1e9, r.bytes_per_pass / r.median_s / 1e9, r.pairs_per_pass / r.best_s);
        }
    }
    return 0;
}
