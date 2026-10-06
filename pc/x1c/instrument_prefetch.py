#!/usr/bin/env python3
"""X1c: add TRACE-ONLY instrumentation to a COPY of ik_llama.cpp's ggml/src/ggml-moe-prefetch.cpp.

The result is an EVIDENCE build (third_party/ik_llama.cpp-x1c-instrumented), never an engine: it is
used once to record, per graph evaluation, which expert ids the prefetcher read at the moment it
enqueued work (scheduler hook) and which ids the MoE kernel then really used (kernel hook), plus every
worker job. No control flow is changed: every inserted statement is a log line (or the bookkeeping for
one) behind `if (x1c_trace())`, which is null unless GGML_MOE_PREFETCH_TRACE=<file> is set.

    instrument_prefetch.py <path to ggml-moe-prefetch.cpp>

Every replacement must match exactly once, otherwise the script fails (so a different source revision
cannot be instrumented by accident).
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "\r\n" in src:
    raise SystemExit("refusing: source has CRLF line endings (use the LF clone)")
if "x1c_trace" in src:
    raise SystemExit("already instrumented")

REPL: list[tuple[str, str]] = []


def rep(old: str, new: str) -> None:
    REPL.append((old, new))


# 1) helpers, placed right after the anonymous-namespace opening
rep("""namespace {

struct mapping_entry {""", """namespace {

// ---- X1c instrumentation (evidence build only) ----------------------------------------------
static FILE * x1c_trace() {
    static FILE * f = [] () -> FILE * {
        const char * p = getenv("GGML_MOE_PREFETCH_TRACE");
        if (!p || !*p) return (FILE *) nullptr;
        FILE * fp = fopen(p, "a");
        if (fp) setvbuf(fp, nullptr, _IOLBF, 0);
        return fp;
    }();
    return f;
}
static double x1c_now() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec*1e-9;
}
static thread_local int    x1c_caller   = 0;  // 0 = scheduler hook (compute_splits), 1 = kernel hook
static thread_local int    x1c_last_enq = -1; // 0 off, 1 not mapped, 2 same epoch (no-op), 3 dropped, 4 enqueued, 5 empty
static thread_local size_t x1c_last_jobs = 0;
// ---------------------------------------------------------------------------------------------

struct mapping_entry {""")

# 2) worker: log every job
rep("""            n_jobs.fetch_add(1, std::memory_order_relaxed);
            bool done_read = false;
            // skip when every page is already resident; keeps the drain rate
            // high when the cache is warm
            if (chunk_resident(j.addr, j.len)) {
                n_skipped.fetch_add(1, std::memory_order_relaxed);
            } else {
                const uintptr_t astart = j.addr & ~(uintptr_t)(page - 1);
                const size_t    alen   = ((j.addr + j.len + page - 1) & ~(uintptr_t)(page - 1)) - astart;
                if (madvise((void *)astart, alen, MADV_POPULATE_READ) == 0) {
                    bytes_populated.fetch_add(alen, std::memory_order_relaxed);
                    done_read = true;
                } // on failure the fault path takes over
            }
""", """            n_jobs.fetch_add(1, std::memory_order_relaxed);
            bool done_read = false;
            const double x1c_t0 = x1c_trace() ? x1c_now() : 0.0;
            int x1c_res = 0; // 0 skipped (resident), 1 populated, 2 madvise failed
            // skip when every page is already resident; keeps the drain rate
            // high when the cache is warm
            if (chunk_resident(j.addr, j.len)) {
                n_skipped.fetch_add(1, std::memory_order_relaxed);
            } else {
                const uintptr_t astart = j.addr & ~(uintptr_t)(page - 1);
                const size_t    alen   = ((j.addr + j.len + page - 1) & ~(uintptr_t)(page - 1)) - astart;
                if (madvise((void *)astart, alen, MADV_POPULATE_READ) == 0) {
                    bytes_populated.fetch_add(alen, std::memory_order_relaxed);
                    done_read = true;
                    x1c_res = 1;
                } else { // on failure the fault path takes over
                    x1c_res = 2;
                }
            }
            if (x1c_trace()) {
                fprintf(x1c_trace(), "J %.6f %.6f addr=%llx len=%u res=%d epoch=%llu\\n", x1c_t0, x1c_now(),
                        (unsigned long long) j.addr, (unsigned) j.len, x1c_res,
                        (unsigned long long) (j.tk ? j.tk->epoch : 0));
            }
""")

# 3) enqueue_ranges: remember why it returned
rep("""    if (!pool_sp) return false;

    if (!is_mapped(w->data, ggml_nbytes(w))) return false;
""", """    if (!pool_sp) { x1c_last_enq = 0; return false; }

    if (!is_mapped(w->data, ggml_nbytes(w))) { x1c_last_enq = 1; return false; }
""")
rep("""    if (pool.shutdown) return false;
""", """    if (pool.shutdown) { x1c_last_enq = 0; return false; }
""")
rep("""    if (tk->epoch == cur_epoch) {
        return true; // already enqueued for this scheduler pass
    }
    if (!urgent && pool.queue.size() > GGML_MOE_PREFETCH_MAX_QUEUE) {
        return true; // drop lookahead under backpressure; do not mark the epoch
    }
""", """    if (tk->epoch == cur_epoch) {
        x1c_last_enq = 2;
        return true; // already enqueued for this scheduler pass
    }
    if (!urgent && pool.queue.size() > GGML_MOE_PREFETCH_MAX_QUEUE) {
        x1c_last_enq = 3;
        return true; // drop lookahead under backpressure; do not mark the epoch
    }
""")
rep("""    if (jobs.empty()) return true;

    tk->pending.fetch_add((int)jobs.size(), std::memory_order_acq_rel);
""", """    if (jobs.empty()) { x1c_last_enq = 5; return true; }

    x1c_last_enq  = 4;
    x1c_last_jobs = jobs.size();
    if (x1c_trace()) {
        fprintf(x1c_trace(), "Q %.6f epoch=%llu urgent=%d jobs=%zu queue_before=%zu w=%s\\n", x1c_now(),
                (unsigned long long) cur_epoch, urgent ? 1 : 0, jobs.size(), pool.queue.size(), w->name);
    }
    tk->pending.fetch_add((int)jobs.size(), std::memory_order_acq_rel);
""")

# 4) mapping registration + pool start
rep("""    s.mappings.push_back({(uintptr_t)base, size});
}
""", """    s.mappings.push_back({(uintptr_t)base, size});
    if (x1c_trace()) {
        fprintf(x1c_trace(), "M %.6f base=%llx size=%zu\\n", x1c_now(), (unsigned long long) (uintptr_t) base, size);
    }
}
""")
rep("""    s.pool = std::make_shared<prefetch_pool>();
    s.pool->start(n_threads);
}
""", """    s.pool = std::make_shared<prefetch_pool>();
    s.pool->start(n_threads);
    if (x1c_trace()) {
        fprintf(x1c_trace(), "P %.6f n_threads=%d page=%ld\\n", x1c_now(), n_threads, sysconf(_SC_PAGESIZE));
    }
}
""")

# 5) epoch
rep("""void ggml_moe_prefetch_new_epoch(void) {
    state().epoch.fetch_add(1, std::memory_order_relaxed);
}
""", """void ggml_moe_prefetch_new_epoch(void) {
    const uint64_t x1c_e = state().epoch.fetch_add(1, std::memory_order_relaxed) + 1;
    if (x1c_trace()) {
        fprintf(x1c_trace(), "E %.6f epoch=%llu\\n", x1c_now(), (unsigned long long) x1c_e);
    }
}
""")

# 6) selective enqueue: log the ids the prefetcher reads and what enqueue_ranges did with them
rep("""    std::vector<std::pair<size_t, size_t>> ranges;
    collect_selected_ranges(w0, ids, ranges);
    enqueue_ranges(w0, ranges, /*urgent =*/ true, /*track_reads =*/ false);
    if (w1) {
        ranges.clear();
        collect_selected_ranges(w1, ids, ranges);
        enqueue_ranges(w1, ranges, /*urgent =*/ true, /*track_reads =*/ false);
    }
}
""", """    std::vector<std::pair<size_t, size_t>> ranges;
    collect_selected_ranges(w0, ids, ranges);
    x1c_last_enq = -1; x1c_last_jobs = 0;
    enqueue_ranges(w0, ranges, /*urgent =*/ true, /*track_reads =*/ false);
    if (x1c_trace()) {
        char buf[1024]; int n = 0;
        const int64_t tot = ids->ne[0]*ids->ne[1];
        for (int64_t i1 = 0; i1 < ids->ne[1] && n < 900; ++i1) {
            for (int64_t i0 = 0; i0 < ids->ne[0] && n < 900; ++i0) {
                const int32_t id = *(const int32_t *)((const char *)ids->data + i1*ids->nb[1] + i0*ids->nb[0]);
                n += snprintf(buf + n, sizeof(buf) - n, "%s%d", (i0 == 0 && i1 == 0) ? "" : (i0 == 0 ? ";" : ","), id);
            }
        }
        buf[n] = 0;
        size_t bytes = 0;
        for (const auto & r : ranges) bytes += r.second;
        fprintf(x1c_trace(), "N %.6f epoch=%llu caller=%c op=%s w=%s wdata=%llx stride=%zu ids_ne=%lldx%lld n_ids=%lld ids_ptr=%llx ids=%s nranges=%zu bytes=%zu enq=%d jobs=%zu\\n",
                x1c_now(), (unsigned long long) state().epoch.load(std::memory_order_relaxed),
                x1c_caller ? 'K' : 'S', ggml_op_name(node->op), w0->name, (unsigned long long) (uintptr_t) w0->data,
                (size_t) w0->nb[2], (long long) ids->ne[0], (long long) ids->ne[1], (long long) tot,
                (unsigned long long) (uintptr_t) ids->data, buf, ranges.size(), bytes, x1c_last_enq, x1c_last_jobs);
    }
    if (w1) {
        ranges.clear();
        collect_selected_ranges(w1, ids, ranges);
        enqueue_ranges(w1, ranges, /*urgent =*/ true, /*track_reads =*/ false);
    }
}
""")

# 7) full-tensor lookahead
rep("""    std::vector<std::pair<size_t, size_t>> ranges;
    ranges.emplace_back(0, ggml_nbytes(w));
    enqueue_ranges(w, ranges, /*urgent =*/ false, /*track_reads =*/ true);
}
""", """    std::vector<std::pair<size_t, size_t>> ranges;
    ranges.emplace_back(0, ggml_nbytes(w));
    x1c_last_enq = -1; x1c_last_jobs = 0;
    enqueue_ranges(w, ranges, /*urgent =*/ false, /*track_reads =*/ true);
    if (x1c_trace()) {
        fprintf(x1c_trace(), "T %.6f epoch=%llu w=%s wdata=%llx bytes=%zu enq=%d jobs=%zu\\n", x1c_now(),
                (unsigned long long) state().epoch.load(std::memory_order_relaxed), w->name,
                (unsigned long long) (uintptr_t) w->data, ggml_nbytes(w), x1c_last_enq, x1c_last_jobs);
    }
}
""")

# 8) MADV_COLD pass
rep("""    pool->n_colded.fetch_add(chunks.size(), std::memory_order_relaxed);
}
""", """    pool->n_colded.fetch_add(chunks.size(), std::memory_order_relaxed);
    if (x1c_trace()) {
        size_t bytes = 0;
        for (const auto & c : chunks) bytes += c.second;
        fprintf(x1c_trace(), "C %.6f w=%s chunks=%zu bytes=%zu\\n", x1c_now(), w->name, chunks.size(), bytes);
    }
}
""")

# 9) kernel hook: mark the caller
rep("""    // in the same expert-index order.
    ggml_moe_prefetch_node(node);
}
""", """    // in the same expert-index order.
    x1c_caller = 1;
    ggml_moe_prefetch_node(node);
    x1c_caller = 0;
}
""")

for old, new in REPL:
    c = src.count(old)
    if c != 1:
        raise SystemExit(f"replacement target found {c} times (expected 1):\n{old[:200]}")
    src = src.replace(old, new)
if "#include <ctime>" not in src:
    src = src.replace("#include <cstdio>\n", "#include <cstdio>\n#include <ctime>\n", 1)
open(path, "w", encoding="utf-8", newline="\n").write(src)
print(f"instrumented {path}: {len(REPL)} replacements")
