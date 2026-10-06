"""Unit tests for pigemma.results (stdlib + pytest; no models, no Docker; Windows and Linux).

Manifest test field: missing `kind` rejected; projected without assumptions rejected; summarize
renders a fixture dir with one measured + one projected file into markdown containing both
section headers and 'not run'. Plus: auto-filled fields, normalisation rules, loader skipping,
validate/write/summarize CLIs and their exit codes.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from pigemma import paths, results

REPO = Path(__file__).resolve().parent.parent

MODEL_26B = {
    "id": "26b-q4",
    "file": "models/google__gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf",
    "quant": "Q4_0",
    "bytes": 14439363584,
    "sha_prefix": None,
}
HOST = {
    "hostname": "testhost", "os": "Linux", "kernel": "6.18", "cpu": "Cortex-A76",
    "n_logical": 4, "n_physical": 4, "ram_total_bytes": 16_000_000_000, "page_size": 16384,
    "in_container": False, "cgroup_mem_max": None,
}


def measured_fields(**over: Any) -> dict[str, Any]:
    f: dict[str, Any] = dict(
        experiment="E0", kind="measured", host=dict(HOST), llama_build=11398, llama_commit="a7b94df",
        model=dict(MODEL_26B), variant="norepack", flags=["-nr"], threads=6, n_ctx=4096,
        tg={"tok_s": 13.52, "sd": 0.11, "n": 3},
    )
    f.update(over)
    return f


def projected_fields(**over: Any) -> dict[str, Any]:
    f: dict[str, Any] = dict(
        experiment="E4", kind="projected", host=dict(HOST), llama_build=11398, llama_commit="a7b94df",
        model=dict(MODEL_26B), variant="lru@pi5-8gb-nvme-gen3", flags=[], threads=4, n_ctx=4096,
        tg={"tok_s": 2.7},
        assumptions={
            "mem_bw_gbps": {"value": 10.0, "unit": "GB/s", "source": "assumed: research summary"},
            "llama_efficiency": {"value": 0.65, "source": "assumed: default"},
        },
        inputs=["results/h1/sim.json", "traces/26b/all.moetrace"],
    )
    f.update(over)
    return f


def _run_cli(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pigemma.results", *args], cwd=REPO, capture_output=True,
        text=True, encoding="utf-8", input=stdin, timeout=120,
    )


# ------------------------------------------------------------------------------- required keys
def test_missing_kind_rejected(tmp_path: Path) -> None:
    f = measured_fields()
    del f["kind"]
    with pytest.raises(ValueError) as ei:
        results.write_result(tmp_path / "x.json", **f)
    assert "kind" in str(ei.value)
    assert "missing required keys" in str(ei.value)
    assert isinstance(ei.value, results.ResultSchemaError)
    assert ei.value.missing == ["kind"]
    assert not (tmp_path / "x.json").exists()


def test_missing_keys_all_listed() -> None:
    with pytest.raises(ValueError) as ei:
        results.build_result(kind="measured")
    msg = str(ei.value)
    for k in ("experiment", "llama_build", "llama_commit", "model", "variant", "flags", "threads", "n_ctx"):
        assert k in msg, k
    # auto-filled ones are not reported missing
    for k in results.AUTO_FILLED:
        assert k not in ei.value.missing


def test_invalid_kind_rejected() -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        results.build_result(**measured_fields(kind="guessed"))


def test_projected_without_assumptions_rejected(tmp_path: Path) -> None:
    f = projected_fields()
    del f["assumptions"]
    with pytest.raises(ValueError, match="assumptions"):
        results.write_result(tmp_path / "p.json", **f)
    with pytest.raises(ValueError, match="non-empty"):
        results.build_result(**projected_fields(assumptions={}))
    with pytest.raises(ValueError, match="inputs"):
        results.build_result(**projected_fields(inputs=[]))
    f = projected_fields()
    del f["inputs"]
    with pytest.raises(ValueError, match="inputs"):
        results.build_result(**f)
    with pytest.raises(ValueError, match="source"):
        results.build_result(**projected_fields(assumptions={"x_bytes": {"value": 1}}))
    assert not (tmp_path / "p.json").exists()


def test_assumed_needs_assumptions_but_inputs_may_be_empty() -> None:
    with pytest.raises(ValueError, match="assumptions"):
        results.build_result(**projected_fields(kind="assumed", assumptions={}))
    doc = results.build_result(**projected_fields(kind="assumed", inputs=[]))
    assert doc["kind"] == "assumed"


# ------------------------------------------------------------------------- build / write output
def test_write_measured_autofill_and_roundtrip(tmp_path: Path) -> None:
    f = measured_fields()
    del f["host"]
    p = results.write_result(tmp_path / "sub" / "r.json", **f)
    assert p == tmp_path / "sub" / "r.json" and p.is_file()
    raw = p.read_bytes()
    assert b"\r\n" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    doc = json.loads(raw.decode("utf-8"))
    assert uuid.UUID(doc["run_id"]).version == 4
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", doc["created_at"])
    host = doc["host"]
    for k in results.HOST_KEYS:
        assert k in host, k
    assert list(doc)[: len(results.REQUIRED_KEYS)] == list(results.REQUIRED_KEYS)
    assert doc["tg"] == {"tok_s": 13.52, "sd": 0.11, "n": 3}
    assert results.validate_result(doc, strict=True) == []


def test_write_into_directory_uses_result_filename(tmp_path: Path) -> None:
    p1 = results.write_result(tmp_path, **measured_fields())
    p2 = results.write_result(str(tmp_path) + "/new/", **measured_fields(variant="default"))
    p3 = results.write_result(tmp_path / "fresh" / "dir", **measured_fields())  # no suffix, absent
    assert p1.parent == tmp_path and p2.parent == tmp_path / "new"
    assert p3.parent == tmp_path / "fresh" / "dir" and p3.suffix == ".json"
    assert re.match(r"^E0_26b-q4_norepack_6_\d{8}T\d{6}Z_[0-9a-f]{8}\.json$", p1.name), p1.name
    assert p2.name.startswith("E0_26b-q4_default_6_")
    assert p1.name != results.write_result(tmp_path, **measured_fields()).name  # unique per run


def test_normalisation_rules(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="pigemma.results")
    doc = results.build_result(**measured_fields(
        experiment="E0-desktop-baseline-and-repack-gate", kind="Measured",
        flags="-nr -t 6 --temp 0", threads="6", notes="one note",
        output_sample="x" * 2500, load_s=float("nan"), my_custom_key={"a": 1},
        inputs=[REPO / "results" / "x.json"],
    ))
    assert doc["experiment"] == "E0" and doc["kind"] == "measured"
    assert doc["flags"] == ["-nr", "-t", "6", "--temp", "0"]
    assert doc["threads"] == 6
    assert len(doc["output_sample"]) == results.OUTPUT_SAMPLE_MAX
    assert doc["load_s"] is None
    assert doc["notes"][0] == "one note"
    assert any("truncated" in n for n in doc["notes"])
    assert any("non-finite" in n and "load_s" in n for n in doc["notes"])
    assert doc["extra"] == {"my_custom_key": {"a": 1}} and "my_custom_key" not in doc
    assert "moved under 'extra'" in caplog.text
    assert doc["inputs"] == ["results/x.json"]
    json.dumps(doc, allow_nan=False)  # strictly serialisable


def test_experiment_values() -> None:
    assert results.normalize_experiment("e3") == "E3"
    assert results.normalize_experiment("ADHOC") == "adhoc"
    for bad in ("E9", "E10", "", "X0", None):
        with pytest.raises(ValueError):
            results.normalize_experiment(bad)
    with pytest.raises(ValueError, match="experiment"):
        results.build_result(**measured_fields(experiment="E9"))


def test_created_at_must_be_utc() -> None:
    with pytest.raises(ValueError, match="created_at"):
        results.build_result(**measured_fields(created_at="2026-10-05T10:00:00+03:00"))
    with pytest.raises(ValueError, match="created_at"):
        results.build_result(**measured_fields(created_at="yesterday"))
    for ok in ("2026-10-05T10:00:00Z", "2026-10-05T10:00:00.123456+00:00"):
        assert results.build_result(**measured_fields(created_at=ok))["created_at"] == ok


def test_run_id_must_be_uuid4() -> None:
    with pytest.raises(ValueError, match="uuid4"):
        results.build_result(**measured_fields(run_id="abc"))
    with pytest.raises(ValueError, match="uuid4"):
        results.build_result(**measured_fields(run_id=str(uuid.uuid1())))
    rid = uuid.uuid4()
    assert results.build_result(**measured_fields(run_id=rid))["run_id"] == str(rid)


def test_type_errors_reported() -> None:
    bad: list[dict[str, Any]] = [
        {"tg": {"tok_s": "fast"}}, {"tg": {"sd": 1.0}}, {"accept_rate": 1.5}, {"coherent": "yes"},
        {"flags": [object()]}, {"kld": {"same_top_p": 99.0}}, {"qa_score": 17, "qa_total": 16},
        {"draft_n": 5, "draft_n_accepted": 6}, {"threads": 0}, {"variant": ""}, {"host": {"os": "x"}},
        {"model": {"id": None, "file": None, "quant": None, "bytes": None, "sha_prefix": None}},
        {"extra": [1]}, {"notes": [1]},
    ]
    for over in bad:
        with pytest.raises(ValueError):
            results.build_result(**measured_fields(**over))
    # flags given as numbers are stringified, not rejected
    assert results.build_result(**measured_fields(flags=["-t", 6]))["flags"] == ["-t", "6"]


def test_model_from_path_and_partial_model(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    g = tmp_path / "tiny-Q4_0.gguf"
    g.write_bytes(b"GGUF" + b"\0" * 100)
    doc = results.build_result(**measured_fields(model=str(g)))
    assert doc["model"]["bytes"] == 104 and doc["model"]["quant"] == "Q4_0"
    assert isinstance(doc["model"]["sha_prefix"], str) and len(doc["model"]["sha_prefix"]) == 64
    caplog.set_level(logging.WARNING, logger="pigemma.results")
    doc = results.build_result(**measured_fields(model={"id": "26b-q4"}))
    assert doc["model"] == {"id": "26b-q4", "file": None, "quant": None, "bytes": None, "sha_prefix": None}
    assert "model block lacks" in caplog.text
    assert results.build_result(**measured_fields(model=None))["model"] is None


def test_numpy_values_are_converted() -> None:
    np = pytest.importorskip("numpy")
    doc = results.build_result(**measured_fields(
        threads=np.int64(6), read_bytes_per_token=np.float32(1.5e8),
        extra={"curve": np.array([0.1, 0.2]), "n": np.int32(3)},
    ))
    assert doc["threads"] == 6 and type(doc["threads"]) is int
    assert doc["extra"]["curve"] == pytest.approx([0.1, 0.2])
    assert type(doc["extra"]["n"]) is int
    json.dumps(doc, allow_nan=False)


def test_assumption_helpers() -> None:
    a = results.normalize_assumptions({
        "access_ms": {"value": 0.1, "source": "assumed: catalog"},
        "kv_bytes": (123, "B", "simulated: n_ctx 4096"),
        "foo_bytes": {"value": 1, "source": "assumed"},
        "bar": {"value": 1, "source": "assumed"},
    })
    assert a["access_ms"] == {"value": 0.1, "unit": "ms", "source": "assumed: catalog"}
    assert a["kv_bytes"] == {"value": 123, "unit": "B", "source": "simulated: n_ctx 4096"}
    assert a["foo_bytes"]["unit"] == "B" and a["bar"]["unit"] is None
    assert results.assumption(1, "x", "s") == {"value": 1, "unit": "x", "source": "s"}


def test_new_result_fills_pinned_build() -> None:
    f = measured_fields()
    del f["llama_build"], f["llama_commit"], f["experiment"], f["kind"]
    doc = results.new_result("E1", "measured", **f)
    assert doc["llama_build"] == paths.LLAMA_BUILD and doc["llama_commit"] == paths.LLAMA_COMMIT


def test_measured_with_assumptions_warns(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="pigemma.results")
    results.build_result(**measured_fields(assumptions={"x_s": {"value": 1, "unit": "s", "source": "a"}}))
    assert "measured result carries assumptions" in caplog.text


# --------------------------------------------------------------------------------------- loader
def test_load_results_skips_unparsable_and_non_envelopes(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    results.write_result(tmp_path / "a.json", **measured_fields())
    results.write_result(tmp_path / "deep" / "b.json", **projected_fields())
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "bench_raw.json").write_text("[{\"avg_ts\": 1.0}]", encoding="utf-8")
    (tmp_path / "sim.json").write_text("{\"runs\": []}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    # PowerShell 5.1 writes UTF-8 with a BOM
    bom = json.dumps(results.build_result(**measured_fields(variant="bom"))).encode("utf-8")
    (tmp_path / "bom.json").write_bytes(b"\xef\xbb\xbf" + bom)
    caplog.set_level(logging.INFO, logger="pigemma.results")
    docs = results.load_results(tmp_path)
    assert sorted(d["variant"] for d in docs) == ["bom", "lru@pi5-8gb-nvme-gen3", "norepack"]
    assert all(Path(d["_path"]).is_file() for d in docs)
    assert [d["experiment"] for d in docs] == ["E0", "E0", "E4"]  # sorted by experiment
    assert "skipping unparsable results file" in caplog.text and "broken.json" in caplog.text
    assert "not results envelopes" in caplog.text
    # glob + non-recursive + list input + duplicates loaded once
    assert len(results.load_results(str(tmp_path / "*.json"))) == 2
    assert len(results.load_results(tmp_path, recursive=False)) == 2
    assert len(results.load_results([tmp_path, tmp_path / "a.json"])) == 3


def test_load_results_validate_and_duplicate_run_id(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    good = results.build_result(**measured_fields())
    (tmp_path / "1.json").write_text(json.dumps(good), encoding="utf-8")
    (tmp_path / "2.json").write_text(json.dumps(good), encoding="utf-8")  # same run_id
    bad = dict(good, run_id=str(uuid.uuid4()), kind="projected")  # no assumptions
    (tmp_path / "3.json").write_text(json.dumps(bad), encoding="utf-8")
    caplog.set_level(logging.WARNING, logger="pigemma.results")
    assert len(results.load_results(tmp_path)) == 2
    assert "duplicate run_id" in caplog.text
    assert len(results.load_results(tmp_path, validate=True)) == 1


# ------------------------------------------------------------------------------------ summarize
def _fixture_dir(tmp_path: Path) -> Path:
    d = tmp_path / "fixture"
    results.write_result(d / "m.json", **measured_fields(notes=["first measured run"]))
    results.write_result(d / "p.json", **projected_fields())
    return d


def test_summarize_measured_and_projected(tmp_path: Path) -> None:
    d = _fixture_dir(tmp_path)
    out = tmp_path / "docs" / "SUMMARY.md"
    md = results.summarize([d], out)
    assert out.read_text(encoding="utf-8") == md
    assert "### MEASURED (kind=measured)" in md
    assert "### PROJECTED (kind=projected)" in md
    assert "### ASSUMED (kind=assumed)" in md
    assert results.NOT_RUN in md  # e.g. pp of the E0 row, load_s, buffers ...
    assert "**Assumptions used**" in md
    assert "| mem_bw_gbps | 10.00 | GB/s | assumed: research summary | P1 |" in md
    assert "| llama_efficiency | 0.65 | fraction | assumed: default | P1 |" in md
    assert "results/h1/sim.json" in md
    assert "13.52 ± 0.11 (n=3)" in md and "2.70" in md
    assert "M1: first measured run" in md
    assert "_No projected results for E0._" in md and "_No measured results for E4._" in md
    # the measured E0 section must not contain the projected row
    e0 = md.split("## E0:")[1].split("## E4:")[0]
    assert "lru@pi5-8gb-nvme-gen3" not in e0 and "norepack" in e0
    # every markdown table row has the same number of cells as its header
    for block in re.split(r"\n\n", md):
        rows = [ln for ln in block.splitlines() if ln.startswith("|")]
        if rows:
            widths = {len(re.split(r"(?<!\\)\|", r)) for r in rows}
            assert len(widths) == 1, block


def test_summarize_experiment_filter_and_skipped(tmp_path: Path) -> None:
    d = _fixture_dir(tmp_path)
    bad = results.build_result(**measured_fields(variant="broken"))
    bad["kind"] = "projected"  # invalid: no assumptions/inputs
    (d / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    md = results.summarize(d, experiments=["E4"])
    assert "## E4:" in md and "## E0:" not in md
    assert "## Skipped results" in md and "bad.json" in md
    md_all = results.summarize(d)
    assert "broken" not in md_all.split("## Skipped results")[0]


def test_render_emulated_host_and_na() -> None:
    h = dict(HOST, emulated=True, in_container=True)
    doc = results.build_result(**measured_fields(host=h, accept_rate=None, experiment="E1"))
    md = results.render_markdown([doc])
    assert "testhost [emulated, container]" in md
    assert "| n/a |" in md and results.NOT_RUN in md


# ------------------------------------------------------------------------------------------ CLI
def test_cli_summarize(tmp_path: Path) -> None:
    d = _fixture_dir(tmp_path)
    out = tmp_path / "S.md"
    r = _run_cli("summarize", str(d), "--md", str(out), "--experiment", "E0,E4")
    assert r.returncode == 0, r.stderr
    md = out.read_text(encoding="utf-8")
    assert "MEASURED" in md and "PROJECTED" in md and "not run" in md
    r = _run_cli("summarize", str(tmp_path / "does-not-exist"))
    assert r.returncode == paths.EXIT_MISSING
    r = _run_cli("summarize", str(d), "--experiment", "E9")
    assert r.returncode == paths.EXIT_USAGE


def test_cli_validate(tmp_path: Path) -> None:
    good = results.write_result(tmp_path / "good.json", **measured_fields())
    r = _run_cli("validate", str(good))
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.startswith("OK")
    doc = json.loads(good.read_text(encoding="utf-8"))
    del doc["kind"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    r = _run_cli("validate", str(good), str(bad))
    assert r.returncode == 1
    assert "missing required key: kind" in r.stdout
    (tmp_path / "garbage.json").write_text("nope", encoding="utf-8")
    r = _run_cli("validate", str(tmp_path / "garbage.json"), "--json")
    assert r.returncode == 1 and json.loads(r.stdout)["ok"] is False
    # unknown top-level key: warning by default, error with --strict
    doc = json.loads(good.read_text(encoding="utf-8"))
    doc["surprise"] = 1
    odd = tmp_path / "odd.json"
    odd.write_text(json.dumps(doc), encoding="utf-8")
    assert _run_cli("validate", str(odd)).returncode == 0
    assert _run_cli("validate", str(odd), "--strict").returncode == 1
    assert _run_cli("validate", str(tmp_path / "missing.json")).returncode == paths.EXIT_MISSING


def test_cli_write(tmp_path: Path) -> None:
    f = measured_fields()
    del f["kind"]
    fields = tmp_path / "fields.json"
    fields.write_text(json.dumps(f), encoding="utf-8")
    r = _run_cli("write", "--out", str(tmp_path / "w.json"), "--fields", str(fields))
    assert r.returncode == 1 and "kind" in r.stderr
    r = _run_cli("write", "--out", str(tmp_path / "w.json"), "--fields", str(fields),
                 "--set", "kind=measured", "--set", "tg.tok_s=2.44", "--set", "notes=[\"from cli\"]")
    assert r.returncode == 0, r.stderr
    doc = json.loads((tmp_path / "w.json").read_text(encoding="utf-8"))
    assert doc["kind"] == "measured" and doc["tg"]["tok_s"] == 2.44 and doc["notes"] == ["from cli"]
    # stdin + pinned llama + dir output
    g = measured_fields()
    del g["llama_build"], g["llama_commit"], g["host"]
    r = _run_cli("write", "--out", str(tmp_path), "--fields", "-", "--pinned-llama", stdin=json.dumps(g))
    assert r.returncode == 0, r.stderr
    written = Path(r.stdout.strip())
    assert written.parent == tmp_path and json.loads(written.read_text(encoding="utf-8"))["llama_build"] == 11398


def test_cli_help() -> None:
    r = _run_cli("--help")
    assert r.returncode == 0 and "summarize" in r.stdout and "validate" in r.stdout
    assert _run_cli().returncode == paths.EXIT_USAGE
