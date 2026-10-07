#!/usr/bin/env python3
"""Upstream-style versions of the two changes, for a clean tree of current ggml-org/llama.cpp master
(written against 51ce9c11a6f2dfa895696c0048c4333e8953b728, 2026-10-06). No environment switches, no project names.

  port_to_master.py layer-skip <tree>    src/models/gemma4.cpp, src/llama-graph.{h,cpp}, common/speculative.cpp
  port_to_master.py pad-tail   <tree>    ggml/src/ggml-cpu/repack.cpp

The two changes are independent; apply each to its own copy of the tree to get two separate diffs.
"""
import pathlib
import sys


def sub(path, old, new, count=1):
    code = path.read_text(encoding="utf-8")
    n = code.count(old)
    if n != count:
        raise SystemExit(f"{path}: expected {count} occurrence(s), found {n}:\n{old[:160]}")
    path.write_text(code.replace(old, new), encoding="utf-8", newline="\n")


def layer_skip(root):
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
            ggml_tensor * q_out_ids = nullptr) const; // optional: q_cur has only these rows of the batch

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
""", """        ggml_tensor * sinks,
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
""")
    sub(c, """    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = use_kv_cur ? k_cur : mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = use_kv_cur ? v_cur : mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
    }
""", """    ggml_tensor * kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();
    if (q_out_ids) {
        // q_cur has only the output rows, take the same rows of the mask
        GGML_ASSERT(k_cur == nullptr && v_cur == nullptr);
        GGML_ASSERT(kq_mask->ne[2] == 1 && kq_mask->ne[3] == 1);
        kq_mask = ggml_get_rows(ctx0, kq_mask, q_out_ids); // result is F32
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
""")
    g = root / "src/models/gemma4.cpp"
    sub(g, '#include "llama-impl.h"\n', '#include "llama-impl.h"\n\n#include <algorithm>\n')
    sub(g, """    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));
""", """    // layers >= n_layer_kv_from_start store no KV, so tokens without output can skip them
    // keep all rows if something reads one row per token from these layers, or if the kq mask has a stream dim
    const int  il_crop = hparams.n_layer_kv_from_start;
    const bool crop_at_shared_kv =
            crop_before_nextn(inp_out_ids) && il_crop > 0 && il_crop < n_layer &&
            (cparams.kv_unified || ubatch.n_seqs_unq == 1) && hparams.n_pos_per_embd() == 1 &&
            cparams.cb_eval == nullptr && !cparams.training && ubatch.token != nullptr &&
            std::none_of(cparams.embeddings_layer_inp.begin() +
                             std::min<size_t>((size_t) std::max(il_crop, 0), cparams.embeddings_layer_inp.size()),
                         cparams.embeddings_layer_inp.end(), [](bool b) { return b; });
    ggml_tensor * inp_pos_q  = inp_pos;  // positions of the remaining rows
    int64_t       n_tokens_q = n_tokens; // number of remaining rows
    bool          cropped    = false;

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));

        if (crop_at_shared_kv && il == il_crop) {
            inpL       = ggml_get_rows(ctx0, inpL, inp_out_ids);
            inp_pos_q  = ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, inp_pos, 1, n_tokens), inp_out_ids), n_outputs);
            n_tokens_q = n_outputs;
            cropped    = true;
        }
""")
    sub(g, """            Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv_fused, q_dim, n_tokens, qkv_fused->nb[1], 0));""",
           """            Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv_fused, q_dim, n_tokens_q, qkv_fused->nb[1], 0));""")
    sub(g, """            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);""",
           """            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens_q);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos_q, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);""")
    sub(g, """            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr, model.layers[il].wo_s,
                    Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        }

        // TODO @ngxson : strip unused token right after the last KV layer to speed up prompt processing
        // keep all rows when extracting unmasked nextn embeddings (MTP target needs the hidden state for every token)
        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {""",
           """            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr, model.layers[il].wo_s,
                    Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il,
                    cropped ? inp_out_ids : nullptr);
        }

        // keep all rows when extracting unmasked nextn embeddings (MTP target needs the hidden state for every token)
        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids) && !cropped) {""")
    sub(g, """            // TODO @ngxson : improve this
            if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
                inp_this_layer = ggml_get_rows(ctx0, inp_this_layer, inp_out_ids);
            }""",
           """            // TODO @ngxson : improve this
            if (cropped || (il == n_layer - 1 && crop_before_nextn(inp_out_ids))) {
                inp_this_layer = ggml_get_rows(ctx0, inp_this_layer, inp_out_ids);
            }""")
    s = root / "common/speculative.cpp"
    sub(s, """    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared
""", """    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared
    bool    masked_tgt    = false;   // target exports hidden rows of output tokens only (is_mem_shared)
""")
    sub(s, """        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;
""", """        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

        // with shared KV the draft never replays the prompt, it only reads target rows of output tokens
        masked_tgt = is_mem_shared;
        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ masked_tgt);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);
""")
    sub(s, """            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);
""", """            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            if (masked_tgt) {
                // only output tokens have a row
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


def pad_tail(root):
    p = root / "ggml/src/ggml-cpu/repack.cpp"
    sub(p, "#include <cstdio>  // for GGML_ASSERT\n", "#include <cstdio>  // for GGML_ASSERT\n#include <vector>\n")
    sub(p, """                                   int64_t               src1_start,
                                   int64_t               src1_end) {
        const ggml_tensor * src0 = op->src[0];""",
           """                                   int64_t               src1_start,
                                   int64_t               src1_end,
                                   const char *          src1_tail = nullptr) {
        const ggml_tensor * src0 = op->src[0];""")
    sub(p, """                                                             nrows - (nrows % 4), ncols);
        }
        for (int iter = nrows - (nrows % 4); iter < nrows; iter++) {""",
           """                                                             nrows - (nrows % 4), ncols);
        }
        // 2-3 leftover rows: one gemm over the zero-padded block instead of one gemv per row
        if (src1_tail != nullptr && ncols > 0) {
            const int64_t n_tail = nrows % 4;
            static thread_local std::vector<float> tail_out; // reused, one per thread
            tail_out.resize((size_t) 4 * ncols);
            gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, tail_out.data(), (size_t) ncols,
                                                             src0_ptr + src0_start * nb01,
                                                             src1_tail + (size_t) i12 * 4 * src1_col_stride, 4, ncols);
            for (int64_t t = 0; t < n_tail; t++) {
                memcpy((float *) (dst_ptr + (nrows - n_tail + t) * nb1) + src0_start, tail_out.data() + t * ncols,
                       (size_t) ncols * sizeof(float));
            }
            return;
        }
        for (int iter = nrows - (nrows % 4); iter < nrows; iter++) {""")
    sub(p, """        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();
""", """        // 2-3 leftover rows: thread 0 also stores them as one zero-padded block of 4, behind the row data in wdata
        char *     wdata_tail = wdata + nbw2 * ne12;
        const bool pad_tail   = ne11 % 4 >= 2 && params->wsize >= (nbw2 + 4 * nbw1) * ne12;
        if (pad_tail && ith == 0) {
            const int64_t      n_tail = ne11 % 4;
            std::vector<float> tail_x((size_t) 4 * ne10, 0.0f);
            for (int64_t i12 = 0; i12 < ne12; i12++) {
                const char * data_ptr = (const char *) src1->data + i12 * nb12 + (ne11 - n_tail) * nb11;
                for (int64_t t = 0; t < n_tail; t++) {
                    memcpy(tail_x.data() + t * ne10, data_ptr + t * nb11, (size_t) ne10 * sizeof(float));
                }
                ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>(tail_x.data(), (void *) (wdata_tail + i12 * 4 * nbw1), 4, ne10);
            }
        }
        // the other threads read the block only after the barrier below
        const char * src1_tail = pad_tail ? wdata_tail : nullptr;

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();
""")
    sub(p, "forward_mul_mat_one_chunk(params, dst, src0_start, src0_end, src1_start, src1_end);",
           "forward_mul_mat_one_chunk(params, dst, src0_start, src0_end, src1_start, src1_end, src1_tail);")
    # work buffer: room for the padded block (MUL_MAT only; the first of the two identical size lines)
    code = p.read_text(encoding="utf-8")
    old = """            case GGML_OP_MUL_MAT:
                {
                    size = ggml_row_size(PARAM_TYPE, ggml_nelements(op->src[1]));
                    return true;
                }"""
    new = """            case GGML_OP_MUL_MAT:
                {
                    size = ggml_row_size(PARAM_TYPE, ggml_nelements(op->src[1]));
                    // + one zero-padded block of 4 rows per plane for 2-3 leftover rows
                    size += 4 * ggml_row_size(PARAM_TYPE, op->src[1]->ne[0]) * op->src[1]->ne[2];
                    return true;
                }"""
    if code.count(old) != 1:
        raise SystemExit(f"{p}: work_size block: expected 1 occurrence, found {code.count(old)}")
    p.write_text(code.replace(old, new), encoding="utf-8", newline="\n")


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("layer-skip", "pad-tail"):
        raise SystemExit(__doc__)
    root = pathlib.Path(sys.argv[2])
    (layer_skip if sys.argv[1] == "layer-skip" else pad_tail)(root)
    print("ported", sys.argv[1], "into", root)


if __name__ == "__main__":
    main()
