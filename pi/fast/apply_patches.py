#!/usr/bin/env python3
"""pigemma-fast: source patches for a COPY of the pinned llama.cpp (a7b94df). Never run it on the reference tree.

Every change is OFF unless its environment variable is set at run time, so the patched binaries behave like the
reference build by default.

  PIGEMMA_STRIP_PREFILL=1  Gemma 4 E-models: layers >= n_layer_kv_from_start reuse the KV cache of earlier layers
                           (shared-KV layers). A token that needs no output never has to pass through them, so the
                           rows are stripped right after the last KV layer (the upstream "TODO @ngxson" in
                           src/models/gemma4.cpp). Only when no per-token hidden state is exported (no MTP drafter).
  PIGEMMA_PAD_TAIL=1       repacked matmul: 2 or 3 token rows beyond a multiple of 4 go through ONE zero-padded
                           gemm sweep instead of one gemv sweep (= one more pass over the weights) per row.
  PIGEMMA_CHUNK_KB=<n>     repacked matmul: weight chunks of at most n KiB when more than 4 token rows are
                           evaluated, so later row groups re-read the chunk from the L2 cache instead of DRAM.

Usage: apply_patches.py <path-to-llama.cpp-copy>
"""
import pathlib
import sys


def sub(path, old, new, count=1):
    code = path.read_text(encoding="utf-8")
    n = code.count(old)
    if n != count:
        raise SystemExit(f"{path}: expected {count} occurrence(s), found {n}:\n{old[:120]}")
    path.write_text(code.replace(old, new), encoding="utf-8", newline="\n")


def patch_graph(root):
    h = root / "src/llama-graph.h"
    sub(h, """            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_k_iswa * build_attn_inp_k_iswa() const;""",
        """            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il,
            ggml_tensor * q_out_ids = nullptr) const; // PIGEMMA: q_cur holds only these batch rows

    llm_graph_input_attn_k_iswa * build_attn_inp_k_iswa() const;""")
    c = root / "src/llama-graph.cpp"
    sub(c, """        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;
    auto * v_rot = is_swa ? inp->self_v_rot_swa : inp->self_v_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }
    if (v_rot) {
        if (v_cur) {
            v_cur = llama_mul_mat_hadamard(ctx0, v_cur, v_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    const auto * mctx_iswa = inp->mctx;""",
        """        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il,
        ggml_tensor * q_out_ids) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;
    auto * v_rot = is_swa ? inp->self_v_rot_swa : inp->self_v_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }
    if (v_rot) {
        if (v_cur) {
            v_cur = llama_mul_mat_hadamard(ctx0, v_cur, v_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    const auto * mctx_iswa = inp->mctx;""")
    # the mask: one row per query token; with stripped queries keep only the rows of the output tokens
    sub(c, """    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = use_kv_cur ? k_cur : mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = use_kv_cur ? v_cur : mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_iswa * inp,""",
        """    ggml_tensor * kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();
    if (q_out_ids) {
        // PIGEMMA: [n_kv, n_tokens] -> [n_kv, n_outputs]; ggml_get_rows returns F32, flash attention wants F16
        GGML_ASSERT(k_cur == nullptr && v_cur == nullptr);
        GGML_ASSERT(kq_mask->ne[2] == 1 && kq_mask->ne[3] == 1);
        kq_mask = ggml_get_rows(ctx0, kq_mask, q_out_ids);
        if (cparams.flash_attn) {
            kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F16);
        }
    }

    ggml_tensor * q = q_cur;
    ggml_tensor * k = use_kv_cur ? k_cur : mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = use_kv_cur ? v_cur : mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_iswa * inp,""")


def patch_gemma4(root):
    p = root / "src/models/gemma4.cpp"
    sub(p, '#include "models.h"\n', '#include "models.h"\n\n#include <cstdlib>\n')
    sub(p, """    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));
""",
        """    // PIGEMMA (upstream "TODO @ngxson" below): layers >= n_layer_kv_from_start reuse the KV cache of earlier
    // layers, so a token that needs no output never has to pass through them. Strip those rows before the first
    // such layer. Not possible while a drafter reads the hidden state of every token (embeddings_nextn).
    static const bool pg_strip_env = []() { const char * e = getenv("PIGEMMA_STRIP_PREFILL"); return e && atoi(e) != 0; }();
    const int  pg_il_strip = hparams.n_layer_kv_from_start;
    const bool pg_strip    = pg_strip_env && inp_out_ids != nullptr && !cparams.embeddings_nextn &&
                             pg_il_strip > 0 && pg_il_strip < n_layer &&
                             (cparams.kv_unified || ubatch.n_seqs_unq == 1) && hparams.n_pos_per_embd() == 1;
    ggml_tensor * pg_pos       = inp_pos;
    int64_t       pg_n_tok     = n_tokens;
    bool          pg_stripped  = false;

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));

        if (pg_strip && il == pg_il_strip) {
            inpL        = ggml_get_rows(ctx0, inpL, inp_out_ids);
            pg_pos      = ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, inp_pos, 1, n_tokens), inp_out_ids), n_outputs);
            pg_n_tok    = n_outputs;
            pg_stripped = true;
            cb(inpL, "pg_stripped_inp", il);
        }
""")
    sub(p, """            Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv_fused, q_dim, n_tokens, qkv_fused->nb[1], 0));""",
           """            Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv_fused, q_dim, pg_n_tok, qkv_fused->nb[1], 0));""")
    sub(p, """            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);""",
           """            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, pg_n_tok);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, pg_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);""")
    sub(p, """            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr, model.layers[il].wo_s,
                    Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        }

        // TODO @ngxson : strip unused token right after the last KV layer to speed up prompt processing
        // keep all rows when extracting unmasked nextn embeddings (MTP target needs the hidden state for every token)
        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {""",
           """            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr, model.layers[il].wo_s,
                    Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il,
                    pg_stripped ? inp_out_ids : nullptr);
        }

        // TODO @ngxson : strip unused token right after the last KV layer to speed up prompt processing
        // keep all rows when extracting unmasked nextn embeddings (MTP target needs the hidden state for every token)
        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked && !pg_stripped) {""")
    sub(p, """            // TODO @ngxson : improve this
            if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
                inp_this_layer = ggml_get_rows(ctx0, inp_this_layer, inp_out_ids);
            }""",
           """            // TODO @ngxson : improve this
            if (pg_stripped || (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked)) {
                inp_this_layer = ggml_get_rows(ctx0, inp_this_layer, inp_out_ids);
            }""")
    sub(p, """    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);""",
           """    if (!cparams.embeddings_nextn_masked && inp_out_ids && !pg_stripped) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);""")


def patch_repack(root):
    p = root / "ggml/src/ggml-cpu/repack.cpp"
    sub(p, "#include <cstdio>  // for GGML_ASSERT\n", "#include <cstdio>  // for GGML_ASSERT\n#include <cstdlib>\n#include <vector>\n")
    sub(p, """        // If there are more than three rows in src1, use gemm; otherwise, use gemv.
        if (nrows > 3) {
            gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, (float *) (dst_ptr) + src0_start, nb1 / nb0,
                                                             src0_ptr + src0_start * nb01, src1_ptr,
                                                             nrows - (nrows % 4), ncols);
        }
        for (int iter = nrows - (nrows % 4); iter < nrows; iter++) {""",
           """        // If there are more than three rows in src1, use gemm; otherwise, use gemv.
        if (nrows > 3) {
            gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, (float *) (dst_ptr) + src0_start, nb1 / nb0,
                                                             src0_ptr + src0_start * nb01, src1_ptr,
                                                             nrows - (nrows % 4), ncols);
        }
        // PIGEMMA: 2 or 3 leftover token rows -> ONE gemm sweep over this weight chunk with zero rows as padding
        // (upstream: one gemv sweep, i.e. one more pass over the weights, per leftover row).
        static const bool pg_pad_tail = []() { const char * e = getenv("PIGEMMA_PAD_TAIL"); return e && atoi(e) != 0; }();
        const int64_t pg_tail = nrows % 4;
        if (pg_pad_tail && pg_tail >= 2 && ncols > 0) {
            static thread_local std::vector<float> pg_x;    // 4 float rows: the tail rows + zeros
            static thread_local std::vector<char>  pg_q;    // the same 4 rows, quantized and interleaved
            static thread_local std::vector<float> pg_out;  // [4, ncols]
            pg_x.assign((size_t) 4 * ne10, 0.0f);
            pg_q.resize((size_t) 4 * src1_col_stride);
            pg_out.resize((size_t) 4 * ncols);
            const char * src1_f32 = (const char *) src1->data + i12 * nb12;
            for (int64_t t = 0; t < pg_tail; t++) {
                memcpy(pg_x.data() + t * ne10, src1_f32 + (i11 + nrows - pg_tail + t) * nb11, (size_t) ne10 * sizeof(float));
            }
            ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>(pg_x.data(), (void *) pg_q.data(), 4, ne10);
            gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, pg_out.data(), (size_t) ncols,
                                                             src0_ptr + src0_start * nb01, pg_q.data(), 4, ncols);
            for (int64_t t = 0; t < pg_tail; t++) {
                memcpy((float *) (dst_ptr + (nrows - pg_tail + t) * nb1) + src0_start, pg_out.data() + t * ncols,
                       (size_t) ncols * sizeof(float));
            }
            return;
        }
        for (int iter = nrows - (nrows % 4); iter < nrows; iter++) {""")
    sub(p, """        int     nth_scaled  = nth * 4;
        int64_t chunk_size0 = (nr0 + nth_scaled - 1) / nth_scaled;
        int64_t nchunk0     = (nr0 + chunk_size0 - 1) / chunk_size0;""",
           """        int     nth_scaled  = nth * 4;
        int64_t chunk_size0 = (nr0 + nth_scaled - 1) / nth_scaled;
        {
            // PIGEMMA: cache-sized weight chunks when several groups of token rows sweep the same chunk
            static const int64_t pg_chunk_bytes = []() { const char * e = getenv("PIGEMMA_CHUNK_KB"); return e ? (int64_t) atol(e) * 1024 : (int64_t) 0; }();
            if (pg_chunk_bytes > 0 && ne11 > 4) {
                const int64_t pg_row_bytes = (int64_t) ggml_row_size(src0->type, ne00);
                int64_t pg_rows = pg_chunk_bytes / (pg_row_bytes > 0 ? pg_row_bytes : 1);
                pg_rows -= pg_rows % NB_COLS;
                if (pg_rows < NB_COLS) {
                    pg_rows = NB_COLS;
                }
                if (pg_rows < chunk_size0) {
                    chunk_size0 = pg_rows;
                }
            }
        }
        int64_t nchunk0     = (nr0 + chunk_size0 - 1) / chunk_size0;""")


def patch_v2_mtp_strip(root):
    """v2: let PIGEMMA_STRIP_PREFILL work together with the Gemma 4 MTP drafter.

    The Gemma 4 drafter shares the target's KV cache (is_mem_shared), so it never runs a catch-up decode over the
    prompt; from the target it only reads the hidden row of the LAST prompt token and the rows of a verification
    batch (where every token has the output flag). Asking the target for masked rows (output tokens only) is
    therefore enough, and then the target can skip its shared-KV layers for the other prompt tokens.
    """
    g = root / "src/models/gemma4.cpp"
    sub(g, "pg_strip_env && inp_out_ids != nullptr && !cparams.embeddings_nextn &&",
           "pg_strip_env && inp_out_ids != nullptr && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked) &&")
    s = root / "common/speculative.cpp"
    sub(s, """    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared
""", """    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared
    bool    pg_masked_tgt = false;   // PIGEMMA: target exports hidden rows of output tokens only
""")
    sub(s, """        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;
""", """        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

        // PIGEMMA: a drafter that shares the target KV (gemma4) only reads target hidden rows of tokens that
        // have the output flag, so the target may skip its shared-KV layers for all other prompt tokens
        {
            const char * pg_e = getenv("PIGEMMA_STRIP_PREFILL");
            pg_masked_tgt = pg_e && atoi(pg_e) != 0 && is_mem_shared;
        }
        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ pg_masked_tgt);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);
""")
    sub(s, """            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);
""", """            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            if (pg_masked_tgt) {
                // PIGEMMA: rows exist only for tokens with the output flag (a prompt batch: its last token or none;
                // a verification batch: every token, in order)
                verify_h[seq_id].resize((size_t) n_rows * n_embd);
                int32_t n_out = 0;
                for (int32_t i = 0; i < n_rows; ++i) {
                    const int32_t k = i_batch_beg[seq_id] + i;
                    if (!batch_in.tokens[k].output || batch_in.tokens[k].seq_id != seq_id) {
                        continue;
                    }
                    const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, k);
                    if (h == nullptr) {
                        continue;
                    }
                    std::memcpy(verify_h[seq_id].data() + (size_t) n_out * n_embd, h, row_bytes);
                    n_out++;
                }
                verify_h_rows[seq_id] = n_out;
                if (n_out > 0) {
                    std::memcpy(pending_h[seq_id].data(),
                            verify_h[seq_id].data() + (size_t) (n_out - 1) * n_embd, row_bytes);
                }
                continue;
            }
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);
""")


def patch_v3_pad_once(root):
    """v3: PIGEMMA_PAD_TAIL quantized the padded tail rows again for every weight chunk (measured: no gain, the
    re-quantization cost as much as the saved sweep). Quantize them once per call (thread 0, before the barrier)."""
    p = root / "ggml/src/ggml-cpu/repack.cpp"
    sub(p, """    void forward_mul_mat_one_chunk(ggml_compute_params * params,
                                   ggml_tensor *         op,""",
           """    // PIGEMMA: PIGEMMA_PAD_TAIL switch and the per-call buffer with the zero-padded, quantized tail rows.
    // One buffer per kernel family; assumes one graph computation at a time (true for llama-server/llama-bench).
    static bool pg_pad_tail_on() {
        static const bool on = []() { const char * e = getenv("PIGEMMA_PAD_TAIL"); return e && atoi(e) != 0; }();
        return on;
    }
    static std::vector<char> & pg_tail_buf() {
        static std::vector<char> buf;
        return buf;
    }

    void forward_mul_mat_one_chunk(ggml_compute_params * params,
                                   ggml_tensor *         op,""")
    sub(p, """        static const bool pg_pad_tail = []() { const char * e = getenv("PIGEMMA_PAD_TAIL"); return e && atoi(e) != 0; }();
        const int64_t pg_tail = nrows % 4;
        if (pg_pad_tail && pg_tail >= 2 && ncols > 0) {
            static thread_local std::vector<float> pg_x;    // 4 float rows: the tail rows + zeros
            static thread_local std::vector<char>  pg_q;    // the same 4 rows, quantized and interleaved
            static thread_local std::vector<float> pg_out;  // [4, ncols]
            pg_x.assign((size_t) 4 * ne10, 0.0f);
            pg_q.resize((size_t) 4 * src1_col_stride);
            pg_out.resize((size_t) 4 * ncols);
            const char * src1_f32 = (const char *) src1->data + i12 * nb12;
            for (int64_t t = 0; t < pg_tail; t++) {
                memcpy(pg_x.data() + t * ne10, src1_f32 + (i11 + nrows - pg_tail + t) * nb11, (size_t) ne10 * sizeof(float));
            }
            ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>(pg_x.data(), (void *) pg_q.data(), 4, ne10);
            gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, pg_out.data(), (size_t) ncols,
                                                             src0_ptr + src0_start * nb01, pg_q.data(), 4, ncols);""",
           """        const int64_t pg_tail = nrows % 4;
        if (pg_pad_tail_on() && pg_tail >= 2 && ncols > 0 && i11 == 0 && nrows == ne11) {
            static thread_local std::vector<float> pg_out;  // [4, ncols]
            pg_out.resize((size_t) 4 * ncols);
            // the 4 padded rows of this plane were quantized once in forward_mul_mat
            const char * pg_q = pg_tail_buf().data() + (size_t) i12 * 4 * src1_col_stride;
            gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, pg_out.data(), (size_t) ncols,
                                                             src0_ptr + src0_start * nb01, pg_q, 4, ncols);""")
    sub(p, """        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        const int64_t nr0 = ggml_nrows(op->src[0]);
""", """        // PIGEMMA: zero-padded copy of the 2-3 leftover token rows, quantized once (thread 0) for all chunks;
        // the other threads only read it after the barrier below
        if (pg_pad_tail_on() && ith == 0 && (ne11 % 4) >= 2) {
            const int64_t       pg_tail = ne11 % 4;
            std::vector<char> & pg_q    = pg_tail_buf();
            pg_q.resize((size_t) ne12 * 4 * nbw1);
            std::vector<float> pg_x((size_t) 4 * ne10);
            for (int64_t i12 = 0; i12 < ne12; i12++) {
                std::fill(pg_x.begin(), pg_x.end(), 0.0f);
                const char * pg_src = (const char *) src1->data + i12 * nb12;
                for (int64_t t = 0; t < pg_tail; t++) {
                    memcpy(pg_x.data() + t * ne10, pg_src + (ne11 - pg_tail + t) * nb11, (size_t) ne10 * sizeof(float));
                }
                ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>(pg_x.data(), (void *) (pg_q.data() + (size_t) i12 * 4 * nbw1), 4, ne10);
            }
        }

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        const int64_t nr0 = ggml_nrows(op->src[0]);
""")
    sub(p, "#include <cstdlib>\n#include <vector>\n", "#include <algorithm>\n#include <cstdlib>\n#include <vector>\n")


def patch_v4_layer_inp_guard(root):
    """v4: never strip rows when a caller asked for the per-token input of a stripped layer
    (llama_set_embeddings_layer_inp, used by some drafters); those buffers expect one row per token."""
    g = root / "src/models/gemma4.cpp"
    sub(g, '#include "models.h"\n\n#include <cstdlib>\n', '#include "models.h"\n\n#include <algorithm>\n#include <cstdlib>\n')
    sub(g, """                             (cparams.kv_unified || ubatch.n_seqs_unq == 1) && hparams.n_pos_per_embd() == 1;
""", """                             (cparams.kv_unified || ubatch.n_seqs_unq == 1) && hparams.n_pos_per_embd() == 1 &&
                             std::none_of(cparams.embeddings_layer_inp.begin() +
                                              std::min<size_t>((size_t) std::max(pg_il_strip, 0), cparams.embeddings_layer_inp.size()),
                                          cparams.embeddings_layer_inp.end(), [](bool b) { return b; });
""")


UPGRADES = [("v2", patch_v2_mtp_strip), ("v3", patch_v3_pad_once), ("v4", patch_v4_layer_inp_guard)]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    upgrade = "--upgrade" in sys.argv[1:]
    if len(args) != 1:
        raise SystemExit(__doc__)
    root = pathlib.Path(args[0])
    if not (root / "src/models/gemma4.cpp").is_file():
        raise SystemExit(f"{root} is not a llama.cpp tree")
    marker = root / "PIGEMMA_FAST_PATCHED"
    if upgrade:
        # bring an already patched tree up to date in place, so its build directory can be reused
        if not marker.exists():
            raise SystemExit("--upgrade needs a patched tree")
        done = " ".join(marker.read_text().split()) + " "
        if "v1" not in done:
            done = "v1 " + ("v2 " if "v2" in done else "")  # markers written by the first two script versions
    else:
        if marker.exists():
            raise SystemExit("already patched (use --upgrade)")
        patch_graph(root)
        patch_gemma4(root)
        patch_repack(root)
        done = "v1 "
    for name, fn in UPGRADES:
        if name not in done.split():
            fn(root)
            done += name + " "
    marker.write_text(done.strip() + "\n")
    print("patched:", root, "|", done.strip())


if __name__ == "__main__":
    main()
