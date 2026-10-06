"""Create a disposable pinned llama source overlay; leave reference builds intact."""
import argparse
import io
import pathlib
import subprocess
import tarfile

PIN = "a7b94df2c616bc1f62a73b964b4a71cb0dcc488e"


def prepare(source, out):
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise ValueError(f"reference revision mismatch: {revision}")
    if out.exists():
        raise ValueError(f"overlay already exists: {out}; choose a new output directory")
    archive = subprocess.check_output(["git", "-C", str(source), "archive", PIN])
    out.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        tf.extractall(out, filter="data")
    path = out / "ggml/src/ggml-cpu/ggml-cpu.c"
    code = path.read_text(encoding="utf-8")
    code = '#include "adapter.h"\n' + code
    old = "                ggml_compute_forward_mul_mat_id(params, tensor);"
    assert code.count(old) == 1
    code = code.replace(old, """                if (!cf_compute_mul_mat_id(params, tensor)) {
                    ggml_compute_forward_mul_mat_id(params, tensor);
                }
                cf_validate_after(params, tensor);""")
    old = "        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);"
    assert code.count(old) == 1
    code = code.replace(old, "        cf_profile_begin(&params, node);\n" + old)
    old = """        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }"""
    assert code.count(old) == 1
    code = code.replace(old, """        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
        cf_profile_end(&params, node, n_fused);
    }""")
    path.write_text(code, encoding="utf-8", newline="\n")
    (out / "CF_REFERENCE_REVISION").write_text(PIN + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=pathlib.Path, default=pathlib.Path("third_party/llama.cpp"))
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("cf-engine/vendor-llama"))
    a = p.parse_args()
    prepare(a.source, a.out)
