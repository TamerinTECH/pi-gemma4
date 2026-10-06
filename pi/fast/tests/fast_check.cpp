// pigemma-fastcheck: API-level regression scenarios for the pigemma-fast patches (layer skip, padded tail, chunks).
//
// One run executes fixed scenarios against one model and writes every observable result to a file:
//   - full logits of every token that has the output flag,
//   - a digest of the persistent KV state of each sequence (llama_state_seq_get_data),
//   - per-token embeddings where they are requested,
//   - the committed token sequences of the decoding scenarios.
// The environment switches (PIGEMMA_STRIP_PREFILL, PIGEMMA_PAD_TAIL, PIGEMMA_CHUNK_KB) are read once at start-up
// by the library, so a comparison is two runs of this tool (or of the tool built against the reference tree)
// followed by `--compare A B`.
//
//   pigemma-fastcheck -m MODEL.gguf --text corpus.txt --out run.bin [-t 4] [--only s1,s3]
//   pigemma-fastcheck --compare a.bin b.bin [--require-kv-equal] [--tol-kl 1e-3] [--tol-abs X] [--json out.json]
//   pigemma-fastcheck --pair a.bin NAME_A b.bin NAME_B      (one logits record against another)
//
// Scenarios (docs/fast/LAYER_SKIP.md explains what each one protects):
//   s1 plain        prompt in one batch (output on the last token), then 12 greedy tokens
//   s2 chunk        the same 45-token prompt in caller batches of 1,2,3,4,5,7,8,9 tokens (no output until the end)
//   s3 boundary     prompts of 511, 512 and 513 tokens in one call; 513 also as 511+2 and 512+1
//   s4 reuse        prompt A, 4 generated tokens, cut the context back to 40 tokens, continue with other text;
//                   and the same final context built from scratch
//   s5 multiseq     two sequences in one batch, output on the last token of each, then one token each;
//                   once with a unified KV buffer and once with per-sequence streams
//   s6 midrows      30-token batch with output flags on rows 3, 7, 16 and 29
//   s7 rollback     draft-and-verify by hand: [t0,d1,d2,d3] all with output, accept the matching prefix, remove
//                   the rejected rows from the KV state, repeat; the committed tokens must equal plain greedy
//   s8 embeddings   per-token embeddings of a 12-token batch (every row is an output)
//   s9 widths       batches of 1,2,3,4,5,7,8,9 tokens with EVERY row flagged as output (verification shape)

#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

enum rec_kind : uint32_t { K_LOGITS = 0, K_KVHASH = 1, K_TOKENS = 2, K_EMBD = 3, K_INFO = 4 };

struct record {
    std::string          name;
    uint32_t             kind = 0;
    std::vector<float>   f;    // logits / embeddings
    std::vector<int64_t> i;    // tokens, or {hash1, hash2, n_bytes} for a KV digest
};

struct writer {
    FILE * fp = nullptr;
    void put(const record & r) {
        const uint32_t magic = 0x50474643;  // "PGFC"
        const uint32_t nlen  = (uint32_t) r.name.size();
        const uint64_t nf = r.f.size(), ni = r.i.size();
        fwrite(&magic, 4, 1, fp);
        fwrite(&r.kind, 4, 1, fp);
        fwrite(&nlen, 4, 1, fp);
        fwrite(r.name.data(), 1, nlen, fp);
        fwrite(&nf, 8, 1, fp);
        fwrite(&ni, 8, 1, fp);
        if (nf) fwrite(r.f.data(), sizeof(float), nf, fp);
        if (ni) fwrite(r.i.data(), sizeof(int64_t), ni, fp);
        fflush(fp);
    }
};

bool read_all(const std::string & path, std::vector<record> & out) {
    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) return false;
    for (;;) {
        uint32_t magic = 0, kind = 0, nlen = 0;
        uint64_t nf = 0, ni = 0;
        if (fread(&magic, 4, 1, fp) != 1) break;
        if (magic != 0x50474643 || fread(&kind, 4, 1, fp) != 1 || fread(&nlen, 4, 1, fp) != 1 || nlen > 4096) { fclose(fp); return false; }
        record r;
        r.kind = kind;
        r.name.resize(nlen);
        if (nlen && fread(&r.name[0], 1, nlen, fp) != nlen) { fclose(fp); return false; }
        if (fread(&nf, 8, 1, fp) != 1 || fread(&ni, 8, 1, fp) != 1 || nf > (1u << 28) || ni > (1u << 28)) { fclose(fp); return false; }
        r.f.resize(nf);
        r.i.resize(ni);
        if (nf && fread(r.f.data(), sizeof(float), nf, fp) != nf) { fclose(fp); return false; }
        if (ni && fread(r.i.data(), sizeof(int64_t), ni, fp) != ni) { fclose(fp); return false; }
        out.push_back(std::move(r));
    }
    fclose(fp);
    return true;
}

uint64_t fnv1a(const uint8_t * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t k = 0; k < n; ++k) { h ^= p[k]; h *= 1099511628211ull; }
    return h;
}
uint64_t poly(const uint8_t * p, size_t n) {  // independent second digest
    uint64_t h = 0x9E3779B97F4A7C15ull;
    for (size_t k = 0; k < n; ++k) { h = (h ^ (uint64_t) p[k]) * 0xFF51AFD7ED558CCDull; h ^= h >> 29; }
    return h;
}

struct entry { llama_token id; llama_pos pos; llama_seq_id seq; bool out; };

struct harness {
    llama_model *       model = nullptr;
    const llama_vocab * vocab = nullptr;
    int                 n_vocab = 0;
    int                 n_threads = 4;
    int                 flash = -1;      // --flash auto|on|off (llama_flash_attn_type)
    bool                kv_q8 = false;   // --kv-type q8_0: quantized KV cache (needs flash attention)
    writer              w;
    std::vector<llama_token> text;  // tokenised corpus, BOS first

    llama_context * make_ctx(int n_ctx, int n_batch, int n_seq_max, bool unified, bool embeddings) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = n_ctx;
        cp.n_batch         = n_batch;
        cp.n_ubatch        = 512;
        cp.n_seq_max       = n_seq_max;
        cp.kv_unified      = unified;
        cp.n_threads       = n_threads;
        cp.n_threads_batch = n_threads;
        cp.embeddings      = embeddings;
        cp.no_perf         = true;
        cp.flash_attn_type = (llama_flash_attn_type) flash;
        if (kv_q8) { cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0; }
        if (embeddings) cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "fastcheck: context creation failed\n"); exit(3); }
        return ctx;
    }

    void process(llama_context * ctx, const std::vector<entry> & es) {
        common_batch b(ctx);
        for (const auto & e : es) b.add(e.id, e.pos, e.seq, e.out);
        const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get());
        if (rc != 0) { fprintf(stderr, "fastcheck: llama_process failed rc=%d (n=%zu)\n", rc, es.size()); exit(3); }
    }

    const float * logits_at(llama_context * ctx, int idx) {
        const float * l = llama_get_logits_ith(ctx, idx);
        if (!l) { fprintf(stderr, "fastcheck: no logits for batch index %d\n", idx); exit(3); }
        return l;
    }
    int argmax(const float * l) const { return (int) (std::max_element(l, l + n_vocab) - l); }

    void rec_logits(const std::string & name, const float * l) {
        record r; r.name = name; r.kind = K_LOGITS; r.f.assign(l, l + n_vocab); w.put(r);
    }
    void rec_tokens(const std::string & name, const std::vector<llama_token> & t) {
        record r; r.name = name; r.kind = K_TOKENS; r.i.assign(t.begin(), t.end()); w.put(r);
    }
    void rec_kv(const std::string & name, llama_context * ctx, llama_seq_id seq) {
        const size_t n = llama_state_seq_get_size(ctx, seq);
        std::vector<uint8_t> buf(n);
        const size_t got = llama_state_seq_get_data(ctx, buf.data(), n, seq);
        record r; r.name = name; r.kind = K_KVHASH;
        r.i = { (int64_t) fnv1a(buf.data(), got), (int64_t) poly(buf.data(), got), (int64_t) got };
        w.put(r);
    }
    void rec_info(const std::string & name, const std::string & text_value) {
        record r; r.name = name + "=" + text_value; r.kind = K_INFO; w.put(r);
    }

    // tokens [a, b) of the corpus as a prompt starting at position p0 (BOS only if a == 0)
    std::vector<entry> span(int a, int b, llama_pos p0, llama_seq_id seq, bool out_last) const {
        std::vector<entry> es;
        for (int k = a; k < b; ++k) es.push_back({ text[k], p0 + (k - a), seq, false });
        if (out_last && !es.empty()) es.back().out = true;
        return es;
    }

    std::vector<llama_token> greedy(llama_context * ctx, const std::string & tag, llama_pos pos, llama_seq_id seq, int last_idx, int n, bool record_logits) {
        std::vector<llama_token> out;
        int idx = last_idx;
        for (int g = 0; g < n; ++g) {
            const float * l = logits_at(ctx, idx);
            if (record_logits) rec_logits(tag + ".gen" + std::to_string(g), l);
            const llama_token t = argmax(l);
            out.push_back(t);
            process(ctx, { { t, pos + g, seq, true } });
            idx = 0;
        }
        return out;
    }

    // ---------------------------------------------------------------------------------------------------------
    void s1_plain() {
        llama_context * ctx = make_ctx(1024, 1024, 1, true, false);
        auto p = span(0, 40, 0, 0, true);
        process(ctx, p);
        rec_logits("s1.prompt_last", logits_at(ctx, 39));
        rec_kv("s1.kv_after_prompt", ctx, 0);
        auto g = greedy(ctx, "s1", 40, 0, 39, 12, true);
        rec_tokens("s1.greedy", g);
        rec_kv("s1.kv_end", ctx, 0);
        llama_free(ctx);
    }

    void s2_chunk() {
        for (int b : { 45, 1, 2, 3, 4, 5, 7, 8, 9 }) {
            llama_context * ctx = make_ctx(1024, 1024, 1, true, false);
            const std::string tag = "s2.b" + std::to_string(b);
            int done = 0, last_idx = -1;
            while (done < 45) {
                const int n = std::min(b, 45 - done);
                auto es = span(done, done + n, done, 0, done + n == 45);
                process(ctx, es);
                last_idx = n - 1;
                done += n;
            }
            rec_logits(tag + ".last", logits_at(ctx, last_idx));
            rec_kv(tag + ".kv", ctx, 0);
            llama_free(ctx);
        }
    }

    void s3_boundary() {
        struct split { int n; std::vector<int> parts; };
        const std::vector<split> cases = { { 511, { 511 } }, { 512, { 512 } }, { 513, { 513 } }, { 513, { 511, 2 } }, { 513, { 512, 1 } } };
        for (const auto & c : cases) {
            if ((int) text.size() < c.n) { rec_info("s3.skipped", "corpus shorter than " + std::to_string(c.n)); return; }
            llama_context * ctx = make_ctx(1024, 1024, 1, true, false);
            std::string tag = "s3.n" + std::to_string(c.n) + ".";
            for (size_t k = 0; k < c.parts.size(); ++k) tag += (k ? "+" : "") + std::to_string(c.parts[k]);
            int done = 0, last_idx = -1;
            for (int n : c.parts) {
                auto es = span(done, done + n, done, 0, done + n == c.n);
                process(ctx, es);
                last_idx = n - 1;
                done += n;
            }
            rec_logits(tag + ".last", logits_at(ctx, last_idx));
            rec_kv(tag + ".kv", ctx, 0);
            llama_free(ctx);
        }
    }

    void s4_reuse() {
        const int cut = 40, alt0 = 200, alt_n = 15;
        {
            llama_context * ctx = make_ctx(1024, 1024, 1, true, false);
            process(ctx, span(0, 60, 0, 0, true));
            greedy(ctx, "s4.reuse", 60, 0, 59, 4, false);
            llama_memory_seq_rm(llama_get_memory(ctx), 0, cut, -1);   // keep positions [0, cut)
            process(ctx, span(alt0, alt0 + alt_n, cut, 0, true));
            rec_logits("s4.reused.last", logits_at(ctx, alt_n - 1));
            rec_kv("s4.reused.kv", ctx, 0);
            llama_free(ctx);
        }
        {
            llama_context * ctx = make_ctx(1024, 1024, 1, true, false);
            process(ctx, span(0, cut, 0, 0, false));
            process(ctx, span(alt0, alt0 + alt_n, cut, 0, true));
            rec_logits("s4.fresh.last", logits_at(ctx, alt_n - 1));
            rec_kv("s4.fresh.kv", ctx, 0);
            llama_free(ctx);
        }
    }

    void s5_multiseq() {
        for (int unified = 1; unified >= 0; --unified) {
            llama_context * ctx = make_ctx(2048, 1024, 2, unified != 0, false);
            const std::string tag = unified ? "s5.unified" : "s5.streams";
            std::vector<entry> es = span(0, 20, 0, 0, true);
            auto b = span(100, 125, 0, 1, true);
            es.insert(es.end(), b.begin(), b.end());
            process(ctx, es);
            const float * l0 = logits_at(ctx, 19);
            rec_logits(tag + ".seq0.last", l0);
            const llama_token t0 = argmax(l0);
            const float * l1 = logits_at(ctx, 44);
            rec_logits(tag + ".seq1.last", l1);
            const llama_token t1 = argmax(l1);
            rec_kv(tag + ".seq0.kv", ctx, 0);
            rec_kv(tag + ".seq1.kv", ctx, 1);
            process(ctx, { { t0, 20, 0, true }, { t1, 25, 1, true } });
            rec_logits(tag + ".seq0.step", logits_at(ctx, 0));
            rec_logits(tag + ".seq1.step", logits_at(ctx, 1));
            llama_free(ctx);
        }
    }

    void s6_midrows() {
        llama_context * ctx = make_ctx(1024, 1024, 1, true, false);
        auto es = span(0, 30, 0, 0, false);
        for (int k : { 3, 7, 16, 29 }) es[k].out = true;
        process(ctx, es);
        for (int k : { 3, 7, 16, 29 }) rec_logits("s6.row" + std::to_string(k), logits_at(ctx, k));
        rec_kv("s6.kv", ctx, 0);
        llama_free(ctx);
    }

    void s7_rollback() {
        const int n_prompt = 30, n_commit = 24;
        // plain greedy continuation as the truth for this run
        std::vector<llama_token> plain;
        {
            llama_context * ctx = make_ctx(1024, 1024, 1, true, false);
            process(ctx, span(0, n_prompt, 0, 0, true));
            plain = greedy(ctx, "s7.plain", n_prompt, 0, n_prompt - 1, n_commit, false);
            rec_kv("s7.plain.kv", ctx, 0);
            llama_free(ctx);
        }
        rec_tokens("s7.plain.tokens", plain);
        // draft-and-verify: drafts are the next corpus tokens (sometimes right, mostly wrong) or, every third
        // cycle, the true continuation (all accepted), so both acceptance and rejection with rollback occur
        llama_context * ctx = make_ctx(1024, 1024, 1, true, false);
        process(ctx, span(0, n_prompt, 0, 0, true));
        std::vector<llama_token> done;
        llama_token t0 = argmax(logits_at(ctx, n_prompt - 1));
        int cycle = 0, n_acc_total = 0, n_rej_cycles = 0;
        while ((int) done.size() < n_commit) {
            const llama_pos p = n_prompt + (llama_pos) done.size();
            std::vector<entry> es = { { t0, p, 0, true } };
            for (int d = 0; d < 3; ++d) {
                const size_t at = done.size() + 1 + d;
                llama_token dt = (cycle % 3 == 2 && at < plain.size()) ? plain[at] : text[(n_prompt + (int) done.size() + 1 + d) % text.size()];
                es.push_back({ dt, p + 1 + d, 0, true });
            }
            process(ctx, es);
            for (int k = 0; k < 4; ++k) rec_logits("s7.c" + std::to_string(cycle) + ".row" + std::to_string(k), logits_at(ctx, k));
            done.push_back(t0);
            int acc = 0;
            llama_token next = argmax(logits_at(ctx, 0));
            while (acc < 3 && (int) done.size() < n_commit && es[1 + acc].id == next) {
                done.push_back(es[1 + acc].id);
                ++acc;
                next = argmax(logits_at(ctx, acc));
            }
            // remove everything after the last accepted row from the persistent state
            llama_memory_seq_rm(llama_get_memory(ctx), 0, p + 1 + acc, -1);
            n_acc_total += acc;
            if (acc < 3) ++n_rej_cycles;
            t0 = next;
            ++cycle;
        }
        done.resize(n_commit);
        rec_tokens("s7.spec.tokens", done);
        rec_info("s7.cycles", std::to_string(cycle) + " accepted_drafts=" + std::to_string(n_acc_total) + " cycles_with_rejection=" + std::to_string(n_rej_cycles));
        rec_info("s7.spec_equals_plain", done == plain ? "yes" : "NO");
        llama_free(ctx);
    }

    void s8_embeddings() {
        llama_context * ctx = make_ctx(1024, 1024, 1, true, true);
        auto es = span(0, 12, 0, 0, false);
        for (auto & e : es) e.out = true;
        process(ctx, es);
        const int n_embd = llama_model_n_embd(model);
        for (int k = 0; k < 12; ++k) {
            const float * e = llama_get_embeddings_ith(ctx, k);
            if (!e) { rec_info("s8.missing_row", std::to_string(k)); continue; }
            record r; r.name = "s8.row" + std::to_string(k); r.kind = K_EMBD; r.f.assign(e, e + n_embd); w.put(r);
        }
        llama_free(ctx);
    }

    void s9_widths() {
        for (int b : { 1, 2, 3, 4, 5, 7, 8, 9 }) {
            llama_context * ctx = make_ctx(1024, 1024, 1, true, false);
            process(ctx, span(0, 24, 0, 0, false));
            auto es = span(24, 24 + b, 24, 0, false);
            for (auto & e : es) e.out = true;
            process(ctx, es);
            for (int k = 0; k < b; ++k) rec_logits("s9.b" + std::to_string(b) + ".row" + std::to_string(k), logits_at(ctx, k));
            rec_kv("s9.b" + std::to_string(b) + ".kv", ctx, 0);
            llama_free(ctx);
        }
    }
};

int compare(const std::string & pa, const std::string & pb, bool require_kv, double tol_abs, double tol_kl, const std::string & json_path) {
    std::vector<record> a, b;
    if (!read_all(pa, a) || !read_all(pb, b)) { fprintf(stderr, "fastcheck: cannot read the result files\n"); return 3; }
    std::map<std::string, const record *> mb;
    for (const auto & r : b) mb[r.name] = &r;
    int n_float = 0, n_bitequal = 0, n_argmax_diff = 0, n_tol_fail = 0, n_kv = 0, n_kv_diff = 0, n_tok = 0, n_tok_diff = 0, n_missing = 0;
    double worst_abs = 0, worst_kl = 0;
    std::string worst_name;
    std::ostringstream js;
    js << "{\"a\":\"" << pa << "\",\"b\":\"" << pb << "\",\"records\":[";
    bool first = true;
    for (const auto & ra : a) {
        if (ra.kind == K_INFO) { printf("info A  %s\n", ra.name.c_str()); continue; }
        auto it = mb.find(ra.name);
        if (it == mb.end()) { ++n_missing; printf("MISSING in B: %s\n", ra.name.c_str()); continue; }
        const record & rb = *it->second;
        if (!first) js << ",";
        first = false;
        js << "{\"name\":\"" << ra.name << "\",\"kind\":" << ra.kind;
        if (ra.kind == K_LOGITS || ra.kind == K_EMBD) {
            ++n_float;
            const size_t n = std::min(ra.f.size(), rb.f.size());
            const bool bits = ra.f.size() == rb.f.size() && memcmp(ra.f.data(), rb.f.data(), n * sizeof(float)) == 0;
            double mx = 0, kl = 0;
            size_t am_a = 0, am_b = 0;
            for (size_t k = 0; k < n; ++k) {
                mx = std::max(mx, (double) std::fabs(ra.f[k] - rb.f[k]));
                if (ra.f[k] > ra.f[am_a]) am_a = k;
                if (rb.f[k] > rb.f[am_b]) am_b = k;
            }
            if (ra.kind == K_LOGITS && !bits) {
                double za = 0, zb = 0;
                for (size_t k = 0; k < n; ++k) { za += std::exp((double) ra.f[k] - ra.f[am_a]); zb += std::exp((double) rb.f[k] - rb.f[am_b]); }
                const double la = std::log(za) + ra.f[am_a], lb = std::log(zb) + rb.f[am_b];
                for (size_t k = 0; k < n; ++k) {
                    const double p = std::exp((double) ra.f[k] - la);
                    if (p > 1e-12) kl += p * (((double) ra.f[k] - la) - ((double) rb.f[k] - lb));
                }
            }
            if (bits) ++n_bitequal;
            if (am_a != am_b) ++n_argmax_diff;
            if (mx > tol_abs || kl > tol_kl) ++n_tol_fail;
            if (mx > worst_abs) { worst_abs = mx; worst_name = ra.name; }
            worst_kl = std::max(worst_kl, kl);
            js << ",\"bit_identical\":" << (bits ? "true" : "false") << ",\"max_abs\":" << mx << ",\"kl\":" << kl << ",\"argmax_equal\":" << (am_a == am_b ? "true" : "false");
            if (!bits) printf("%-34s max|d|=%.3e  KL=%.3e  argmax %s\n", ra.name.c_str(), mx, kl, am_a == am_b ? "same" : "DIFFERENT");
        } else if (ra.kind == K_KVHASH) {
            ++n_kv;
            const bool eq = ra.i == rb.i;
            if (!eq) { ++n_kv_diff; printf("%-34s KV state digest differs (%" PRId64 " vs %" PRId64 " bytes)\n", ra.name.c_str(), ra.i.size() > 2 ? ra.i[2] : -1, rb.i.size() > 2 ? rb.i[2] : -1); }
            js << ",\"kv_equal\":" << (eq ? "true" : "false");
        } else if (ra.kind == K_TOKENS) {
            ++n_tok;
            const bool eq = ra.i == rb.i;
            if (!eq) { ++n_tok_diff; printf("%-34s token sequence DIFFERS\n", ra.name.c_str()); }
            js << ",\"tokens_equal\":" << (eq ? "true" : "false");
        }
        js << "}";
    }
    for (const auto & rb2 : b) if (rb2.kind == K_INFO) printf("info B  %s\n", rb2.name.c_str());
    const bool pass = n_missing == 0 && n_argmax_diff == 0 && n_tok_diff == 0 && n_tol_fail == 0 && (!require_kv || n_kv_diff == 0);
    js << "],\"summary\":{\"float_records\":" << n_float << ",\"bit_identical\":" << n_bitequal << ",\"argmax_different\":" << n_argmax_diff
       << ",\"over_tolerance\":" << n_tol_fail << ",\"worst_max_abs\":" << worst_abs << ",\"worst_kl\":" << worst_kl << ",\"worst_record\":\"" << worst_name
       << "\",\"kv_records\":" << n_kv << ",\"kv_different\":" << n_kv_diff << ",\"token_records\":" << n_tok << ",\"token_sequences_different\":" << n_tok_diff
       << ",\"missing\":" << n_missing << ",\"tol_abs\":" << tol_abs << ",\"tol_kl\":" << tol_kl << ",\"require_kv_equal\":" << (require_kv ? "true" : "false") << ",\"pass\":" << (pass ? "true" : "false") << "}}";
    printf("SUMMARY float=%d bit_identical=%d argmax_different=%d over_tolerance=%d worst_max_abs=%.3e (%s) worst_kl=%.3e | kv=%d kv_different=%d | token_seqs=%d different=%d | missing=%d | %s\n",
           n_float, n_bitequal, n_argmax_diff, n_tol_fail, worst_abs, worst_name.c_str(), worst_kl, n_kv, n_kv_diff, n_tok, n_tok_diff, n_missing, pass ? "PASS" : "FAIL");
    if (!json_path.empty()) { std::ofstream o(json_path); o << js.str() << "\n"; }
    return pass ? 0 : 1;
}

// compare one named logits record of file A with one of file B (control: the reference against itself
// for two ways of feeding the same tokens)
int pair(const std::string & pa, const std::string & na, const std::string & pb, const std::string & nb) {
    std::vector<record> a, b;
    if (!read_all(pa, a) || !read_all(pb, b)) { fprintf(stderr, "fastcheck: cannot read the result files\n"); return 3; }
    const record * ra = nullptr, * rb = nullptr;
    for (const auto & r : a) if (r.name == na) ra = &r;
    for (const auto & r : b) if (r.name == nb) rb = &r;
    if (!ra || !rb || ra->f.size() != rb->f.size() || ra->f.empty()) { fprintf(stderr, "fastcheck: record not found or size mismatch\n"); return 3; }
    const size_t n = ra->f.size();
    size_t am_a = 0, am_b = 0;
    double mx = 0;
    for (size_t k = 0; k < n; ++k) {
        mx = std::max(mx, (double) std::fabs(ra->f[k] - rb->f[k]));
        if (ra->f[k] > ra->f[am_a]) am_a = k;
        if (rb->f[k] > rb->f[am_b]) am_b = k;
    }
    double za = 0, zb = 0, kl = 0;
    for (size_t k = 0; k < n; ++k) { za += std::exp((double) ra->f[k] - ra->f[am_a]); zb += std::exp((double) rb->f[k] - rb->f[am_b]); }
    const double la = std::log(za) + ra->f[am_a], lb = std::log(zb) + rb->f[am_b];
    for (size_t k = 0; k < n; ++k) {
        const double p = std::exp((double) ra->f[k] - la);
        if (p > 1e-12) kl += p * (((double) ra->f[k] - la) - ((double) rb->f[k] - lb));
    }
    const bool bits = memcmp(ra->f.data(), rb->f.data(), n * sizeof(float)) == 0;
    printf("PAIR %s | %s : bit_identical=%s max|d|=%.3e KL=%.3e argmax %s (top logit %.4f vs %.4f)\n", na.c_str(), nb.c_str(),
           bits ? "yes" : "no", mx, kl, am_a == am_b ? "same" : "DIFFERENT", ra->f[am_a], rb->f[am_b]);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model_path, text_path, out_path, only, cmp_a, cmp_b, json_path;
    int    threads = 4;
    int    flash = -1;
    bool   kv_q8 = false;
    bool   require_kv = false;
    double tol_abs = 1e30;   // absolute logit tolerance is off by default: see --tol-kl
    double tol_kl  = 1e-3;   // KL(reference || candidate) of the next-token distribution
    std::vector<std::string> pair_args;
    for (int k = 1; k < argc; ++k) {
        const std::string a = argv[k];
        auto next = [&]() -> std::string { if (k + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++k]; };
        if (a == "-m") model_path = next();
        else if (a == "--text") text_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "-t") threads = atoi(next().c_str());
        else if (a == "--only") only = next();
        else if (a == "--compare") { cmp_a = next(); cmp_b = next(); }
        else if (a == "--require-kv-equal") require_kv = true;
        else if (a == "--tol-abs") tol_abs = atof(next().c_str());
        else if (a == "--tol-kl") tol_kl = atof(next().c_str());
        else if (a == "--pair") { for (int q = 0; q < 4; ++q) pair_args.push_back(next()); }
        else if (a == "--json") json_path = next();
        else if (a == "--flash") { const std::string v = next(); flash = v == "on" ? 1 : v == "off" ? 0 : -1; }
        else if (a == "--kv-type") { kv_q8 = next() == "q8_0"; }
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (pair_args.size() == 4) return pair(pair_args[0], pair_args[1], pair_args[2], pair_args[3]);
    if (!cmp_a.empty()) return compare(cmp_a, cmp_b, require_kv, tol_abs, tol_kl, json_path);
    if (model_path.empty() || text_path.empty() || out_path.empty()) { fprintf(stderr, "usage: see the header of fast_check.cpp\n"); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    harness h;
    h.n_threads = threads;
    h.flash     = flash;
    h.kv_q8     = kv_q8;
    h.model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!h.model) { fprintf(stderr, "fastcheck: cannot load %s\n", model_path.c_str()); return 3; }
    h.vocab   = llama_model_get_vocab(h.model);
    h.n_vocab = llama_vocab_n_tokens(h.vocab);
    std::ifstream tf(text_path);
    std::stringstream ss;
    ss << tf.rdbuf();
    h.text = common_tokenize(h.vocab, ss.str(), true, false);
    if (h.text.size() < 260) { fprintf(stderr, "fastcheck: corpus too short (%zu tokens)\n", h.text.size()); return 2; }
    h.w.fp = fopen(out_path.c_str(), "wb");
    if (!h.w.fp) { fprintf(stderr, "fastcheck: cannot write %s\n", out_path.c_str()); return 3; }
    auto env = [](const char * n) { const char * e = getenv(n); return std::string(e ? e : ""); };
    h.rec_info("env", "STRIP=" + env("PIGEMMA_STRIP_PREFILL") + " PAD=" + env("PIGEMMA_PAD_TAIL") + " CHUNK_KB=" + env("PIGEMMA_CHUNK_KB"));
    h.rec_info("corpus_tokens", std::to_string(h.text.size()));
    h.rec_info("n_vocab", std::to_string(h.n_vocab));
    auto want = [&](const char * s) { return only.empty() || ("," + only + ",").find(std::string(",") + s + ",") != std::string::npos; };
    if (want("s1")) { h.s1_plain();      fprintf(stderr, "fastcheck: s1 done\n"); }
    if (want("s2")) { h.s2_chunk();      fprintf(stderr, "fastcheck: s2 done\n"); }
    if (want("s3")) { h.s3_boundary();   fprintf(stderr, "fastcheck: s3 done\n"); }
    if (want("s4")) { h.s4_reuse();      fprintf(stderr, "fastcheck: s4 done\n"); }
    if (want("s5")) { h.s5_multiseq();   fprintf(stderr, "fastcheck: s5 done\n"); }
    if (want("s6")) { h.s6_midrows();    fprintf(stderr, "fastcheck: s6 done\n"); }
    if (want("s7")) { h.s7_rollback();   fprintf(stderr, "fastcheck: s7 done\n"); }
    if (want("s8")) { h.s8_embeddings(); fprintf(stderr, "fastcheck: s8 done\n"); }
    if (want("s9")) { h.s9_widths();     fprintf(stderr, "fastcheck: s9 done\n"); }
    fclose(h.w.fp);
    llama_model_free(h.model);
    llama_backend_free();
    fprintf(stderr, "fastcheck: FASTCHECK_RUN_DONE %s\n", out_path.c_str());
    return 0;
}
