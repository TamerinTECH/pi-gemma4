#!/usr/bin/env bash
# X1b LOCAL ADAPTATION (not HotPin's original): one-hunk fix of HotPin's --dump-routing callback so
# multi-token evaluations are read with the tensor's row stride (ffn_moe_topk is a strided view of the
# 128-wide argsort at a6647b1). Builds an EXTRA binary llama-simple-chat-stridefix from the HotPin
# tree on volume pigemma-x1b-hotpin-a6647b1, then restores the tree to HotPin's unmodified patch state.
# The diff is a modification of AGPL-3.0 code and stays under third_party/hotpin-llm-build-a6647b1/.
#   build_stridefix_in_container.sh <SRC_DIR> <BUILD_DIR> <OUT_DIR>
set -euo pipefail
SRC="${1:?src}"; BUILD="${2:?build}"; OUT="${3:?out}"
F="$SRC/examples/simple-chat/simple-chat.cpp"
mkdir -p "$OUT/local_adaptations" "$OUT/build-linux-x86/bin"
git config --global --add safe.directory '*'
cp -p "$F" /tmp/simple-chat.hotpin.cpp
ORIG_SHA=$(sha256sum "$F" | cut -d' ' -f1)
python3 - "$F" <<'PY'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = """    if (ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(ids.data(), t->data, ids.size() * sizeof(int32_t));
    } else {
        ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));
    }
"""
new = """    // pigemma X1b local adaptation: ffn_moe_topk is a VIEW {k, n_tokens} of the argsort {n_expert, n_tokens};
    // token `tok` starts at byte offset tok * nb[1] (n_expert * 4), not tok * k * 4.
    for (int tok = 0; tok < ne1; tok++) {
        const size_t off = (size_t) tok * t->nb[1];
        if (ggml_backend_buffer_is_host(t->buffer)) {
            memcpy(ids.data() + (size_t) tok * ne0, (const char *) t->data + off, (size_t) ne0 * sizeof(int32_t));
        } else {
            ggml_backend_tensor_get(t, ids.data() + (size_t) tok * ne0, off, (size_t) ne0 * sizeof(int32_t));
        }
    }
"""
if s.count(old) != 1:
    sys.exit("stride-fix anchor not found exactly once")
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
PY
( cd "$SRC" && diff -u --label a/examples/simple-chat/simple-chat.cpp --label b/examples/simple-chat/simple-chat.cpp \
    /tmp/simple-chat.hotpin.cpp "$F" > "$OUT/local_adaptations/dump_routing_stride_fix.diff" ) || true
wc -l "$OUT/local_adaptations/dump_routing_stride_fix.diff"
nice -n 10 cmake --build "$BUILD" -j 4 --target llama-simple-chat > "$OUT/local_adaptations/build_stridefix.log" 2>&1 \
  || { tail -30 "$OUT/local_adaptations/build_stridefix.log"; cp -p /tmp/simple-chat.hotpin.cpp "$F"; echo "STRIDEFIX BUILD FAILED"; exit 11; }
cp -f "$BUILD/bin/llama-simple-chat" "$OUT/build-linux-x86/bin/llama-simple-chat-stridefix"
FIX_SHA=$(sha256sum "$OUT/build-linux-x86/bin/llama-simple-chat-stridefix" | cut -d' ' -f1)
# restore HotPin's unmodified patch state and rebuild, so the volume matches PATCHED_TREE.diff again
cp -p /tmp/simple-chat.hotpin.cpp "$F"
touch "$F"
nice -n 10 cmake --build "$BUILD" -j 4 --target llama-simple-chat >> "$OUT/local_adaptations/build_stridefix.log" 2>&1
REST_SRC_SHA=$(sha256sum "$F" | cut -d' ' -f1)
REST_BIN_SHA=$(sha256sum "$BUILD/bin/llama-simple-chat" | cut -d' ' -f1)
SHIPPED_BIN_SHA=$(sha256sum "$OUT/build-linux-x86/bin/llama-simple-chat" | cut -d' ' -f1)
git -C "$SRC" -c core.fileMode=false diff > /tmp/tree_after.diff
cat > "$OUT/local_adaptations/STRIDEFIX_INFO.json" <<JSON
{
 "schema": "pigemma-x1b-stridefix v1",
 "what": "LOCAL ADAPTATION of HotPin's routing dump (examples/simple-chat/simple-chat.cpp, routing_dump_cb): read each token's top-k row with the view stride nb[1]; nothing else changed",
 "diff": "local_adaptations/dump_routing_stride_fix.diff (against the HotPin-patched file)",
 "binary": "build-linux-x86/bin/llama-simple-chat-stridefix",
 "binary_sha256": "$FIX_SHA",
 "hotpin_patched_source_sha256_before": "$ORIG_SHA",
 "source_sha256_after_restore": "$REST_SRC_SHA",
 "source_restored": $([ "$ORIG_SHA" = "$REST_SRC_SHA" ] && echo true || echo false),
 "rebuilt_original_llama_simple_chat_sha256": "$REST_BIN_SHA",
 "shipped_original_llama_simple_chat_sha256": "$SHIPPED_BIN_SHA",
 "rebuild_of_original_is_byte_identical": $([ "$REST_BIN_SHA" = "$SHIPPED_BIN_SHA" ] && echo true || echo false),
 "tree_diff_equals_PATCHED_TREE_diff": $(cmp -s /tmp/tree_after.diff "$OUT/build-linux-x86/PATCHED_TREE.diff" && echo true || echo false),
 "license": "modification of AGPL-3.0 code (HotPin); comparison use only, never copy into product code"
}
JSON
cat "$OUT/local_adaptations/STRIDEFIX_INFO.json"
echo "STRIDEFIX_DONE"
