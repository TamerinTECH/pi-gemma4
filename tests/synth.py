"""Synthetic inputs for the pigemma unit tests: routing traces, tiny GGUFs, activation dumps, facts.

Owner: docs/design/manifest.json files[] entry "tests/conftest.py (+ tests/synth.py, tests/test_*.py
per module)". Everything here runs on Windows and Linux without model files, Docker or llama.cpp
binaries, in milliseconds. tests/conftest.py wraps these builders as session fixtures; test modules
may also import them directly:

    import synth                                   # tests/ is on sys.path (pytest prepend mode)
    st = synth.synth_trace(T=2000, L=4, K=8, E=16, alpha=1.2, layer_corr=0.0, seed=0)
    st = synth.write_synth_trace(tmp_path / "x.moetrace", T=500, layer_corr=0.9)
    g = synth.write_tiny_gguf(tmp_path / "moe.gguf")          # gemma4-shaped MoE, manifest values
    path, offsets = g                                         # TinyGguf unpacks to (path, offsets)

Use ``import synth`` (not ``import tests.synth``): both work, but they create two module objects and
isinstance() checks between them fail.

Contents
--------
* Routing traces: ``synth_trace``, ``write_synth_trace``, ``trace_from_ids``, ``cyclic_trace``,
  ``constant_trace`` and the ``SynthTrace`` container (dict-style access like the generator in
  tests/test_trace.py: st["ids"], st["weights"], st["phase"], st["chunk"], st["pos"], st["token_ids"]).
* Tiny GGUFs (gguf-py's GGUFWriter): ``write_tiny_gguf`` (gemma4 MoE or dense, manifest shapes),
  ``write_tiny_dense_gguf``, ``write_tiny_llama_gguf`` (a *runnable* llama-architecture model with a
  byte-level SentencePiece vocabulary; used to capture the real-binary fixtures in tests/fixtures and
  by integration tests that need llama.cpp to load something in seconds). Every builder returns a
  ``TinyGguf`` that knows the writer's absolute tensor offsets, byte sizes, kinds and source arrays.
* Activation dumps: ``write_synth_acts_dir`` (ffn_norm / ffn_geglu / ffn_out that satisfy
  ffn_out = W_down @ geglu for the tiny dense GGUF's real dequantised weights).
* Pi facts fixtures: ``FACTS_NAMES``, ``facts_path``, ``load_facts``.

Design decisions (also listed as open issues in the build report)
-----------------------------------------------------------------
* Zipf routing: expert popularity per layer is rank^-alpha over a per-layer random permutation of
  the expert ids. K distinct experts per (token, layer) are drawn with the Gumbel-top-k trick (exact
  sampling without replacement proportional to popularity). Routing weights are the softmax of the
  same perturbed logits, restricted to the K winners and renormalised, so ids are in descending
  weight order exactly like ffn_moe_topk + ffn_moe_weights_norm; the optional probs block is the
  top-P of the full softmax, so its first min(K, P) ids equal the selected ids.
* layer_corr: with probability layer_corr a token's set at layer j is the image of its set at layer
  j-1 under a fixed per-layer permutation (``SynthTrace.layer_maps``). layer_corr=1.0 with
  stickiness=0 gives fully deterministic cross-layer routing (a perfect cross-layer predictor exists).
* stickiness (extra knob, default 0): with that probability a token re-uses the previous token's
  routing at the same layer (temporal locality, never across a chunk boundary).
* Tiny gemma4 GGUF: the manifest asks for Q4_0 everywhere, but Q4_0 needs ne[0] % 32 == 0. With the
  manifest's feed_forward_length 48 the dense ffn_down (ne=[48,64]) cannot be Q4_0, so any tensor whose
  ne[0] is not a multiple of the block size falls back to F16 (recorded in ``TinyGguf.params
  ["type_fallbacks"]``; pass n_ff=64 or 96 for an all-Q4_0 file). Global (non-sliding) layers have no
  attn_v, like the real 26B/31B files (K == V); the extra norms, router scale, rope_freqs and
  layer_output_scale of the real files are included so every map.json kind occurs.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

import pigemma  # noqa: F401  (puts third_party/llama.cpp/gguf-py on sys.path when gguf is not installed)
from pigemma import trace as tr

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
FIXTURES_DIR = TESTS_DIR / "fixtures"

#: Fixed timestamp used in synthetic headers/meta so outputs are byte-for-byte reproducible.
SYNTH_CREATED_AT = "2026-10-05T00:00:00Z"
#: Gemma 4 BOS id (tokenizer.ggml.bos_token_id of the real GGUFs); first token of every synthetic chunk.
GEMMA4_BOS_ID = 2
GEMMA4_VOCAB = 262144
DEFAULT_SOURCES = ("en_synth", "he_synth")

__all__ = [
    "TESTS_DIR", "REPO_ROOT", "FIXTURES_DIR", "SYNTH_CREATED_AT", "DEFAULT_SOURCES",
    "SynthTrace", "synth_trace", "write_synth_trace", "trace_from_ids", "cyclic_trace", "constant_trace",
    "TinyTensor", "TinyGguf", "write_tiny_gguf", "write_tiny_dense_gguf", "write_tiny_llama_gguf",
    "TINY_MOE_SPEC", "TINY_DENSE_SPEC", "expected_kind", "gelu_tanh",
    "SynthActs", "write_synth_acts_dir",
    "FACTS_NAMES", "FACTS_KEYS", "FACTS_OPTIONAL_KEYS", "FACTS_STORAGE_KEYS", "facts_path", "load_facts",
    "GEMMA4_BOS_ID", "GEMMA4_VOCAB",
    "fixture_path", "read_fixture_text", "read_fixture_json",
]


# ==================================================================================================
# Routing traces
# ==================================================================================================
@dataclass(repr=False, eq=False)
class SynthTrace:
    """A synthetic routing trace held in memory (and optionally written to ``path``).

    Arrays (T tokens, L MoE layers, K experts per token, E experts, P probs entries):
      ids        uint8   [T, L, K]  selected experts, descending routing weight
      weights    float32 [T, L, K]  normalised routing weights (sum 1), descending
      phase      uint8   [T]        0 = prefill, 1 = generated (tail of each chunk)
      chunk      int64   [T]        full chunk index (records store it mod 256)
      pos        int64   [T]        position inside the chunk
      token_ids  int32   [T]        input token ids (BOS at every chunk start)
      probs_ids  uint8   [T, L, P]  or None: top-P experts of the full softmax, descending
      probs      float32 [T, L, P]  or None
      popularity float64 [L, E]     Zipf probability of each expert id per layer (sums to 1)
      layer_maps int64   [L, E]     layer_maps[j][e]: the layer-j expert that follows expert e of
                                    layer j-1 under cross-layer correlation (row 0 = identity)
      corr_mask  bool    [T, L]     True where the set was forced by the cross-layer map
      sticky_mask bool   [T, L]     True where the set was copied from the previous token
    """

    ids: np.ndarray
    weights: np.ndarray
    phase: np.ndarray
    chunk: np.ndarray
    pos: np.ndarray
    token_ids: np.ndarray
    n_expert: int
    popularity: np.ndarray
    layer_maps: np.ndarray
    corr_mask: np.ndarray
    sticky_mask: np.ndarray
    probs_ids: np.ndarray | None = None
    probs: np.ndarray | None = None
    sources: list[str] = field(default_factory=lambda: list(DEFAULT_SOURCES))
    params: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    _KEYS = ("ids", "weights", "phase", "chunk", "pos", "token_ids", "probs_ids", "probs",
             "popularity", "layer_maps", "corr_mask", "sticky_mask")

    # -- shape ---------------------------------------------------------------------------------
    @property
    def T(self) -> int:
        return int(self.ids.shape[0])

    @property
    def L(self) -> int:
        return int(self.ids.shape[1])

    @property
    def K(self) -> int:
        return int(self.ids.shape[2])

    @property
    def E(self) -> int:
        return int(self.n_expert)

    @property
    def P(self) -> int:
        return 0 if self.probs_ids is None else int(self.probs_ids.shape[2])

    @property
    def n_chunks(self) -> int:
        return int(self.chunk.max()) + 1 if self.chunk.size else 0

    @property
    def n_gen(self) -> int:
        return int(np.count_nonzero(self.phase == 1))

    @property
    def n_prefill(self) -> int:
        return self.T - self.n_gen

    @property
    def moe_layers(self) -> list[int]:
        return list(range(self.L))

    @property
    def weights_f16(self) -> np.ndarray:
        """Weights as a reader sees them after the f16 round trip (float32 [T, L, K])."""
        return self.weights.astype(np.float16).astype(np.float32)

    # -- path-like once written; compact repr -----------------------------------------------------
    def __fspath__(self) -> str:
        """The written .moetrace path, so ``TraceReader(write_synth_trace(...))`` works."""
        if self.path is None:
            raise TypeError("SynthTrace has not been written (use write_synth_trace); it has no path")
        return str(self.path)

    def __repr__(self) -> str:
        return (f"SynthTrace(T={self.T}, L={self.L}, K={self.K}, E={self.E}, P={self.P}, "
                f"chunks={self.n_chunks}, path={str(self.path) if self.path else None!r}, "
                f"params={ {k: v for k, v in self.params.items() if k not in ('T', 'L', 'K', 'E', 'P')} })")

    # -- dict-style access (compatible with the local generator in tests/test_trace.py) ----------
    def __getitem__(self, key: str) -> Any:
        if key in self._KEYS:
            return getattr(self, key)
        raise KeyError(key)

    def keys(self) -> list[str]:
        return [k for k in self._KEYS if getattr(self, k) is not None]

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.keys()}

    # -- helpers ---------------------------------------------------------------------------------
    def chunk_slice(self, c: int) -> slice:
        rows = np.flatnonzero(self.chunk == c)
        if rows.size == 0:
            raise IndexError(f"chunk {c} has no records")
        return slice(int(rows[0]), int(rows[-1]) + 1)

    def chunk_source(self, c: int) -> str:
        return self.sources[c % len(self.sources)] if self.sources else "synth"

    def chunks(self) -> list[dict[str, Any]]:
        """Header-style chunk list: [{chunk_id, source, n_prompt, n_gen}]."""
        out = []
        for c in range(self.n_chunks):
            sl = self.chunk_slice(c)
            n_gen = int(np.count_nonzero(self.phase[sl] == 1))
            out.append({"chunk_id": c, "source": self.chunk_source(c),
                        "n_prompt": (sl.stop - sl.start) - n_gen, "n_gen": n_gen})
        return out

    def freq(self) -> np.ndarray:
        """Selection counts int64 [L, E]."""
        out = np.zeros((self.L, self.E), dtype=np.int64)
        for j in range(self.L):
            out[j] = np.bincount(self.ids[:, j, :].ravel().astype(np.int64), minlength=self.E)[: self.E]
        return out

    def items(self, item: str = "layer-expert") -> np.ndarray:
        """Access sequence in (t, j, k) order, same encoding as pigemma.trace.access_items."""
        return tr.access_items(self.ids, self.E, item)[0]

    def header(self, **fields: Any) -> dict[str, Any]:
        """A complete .moetrace JSON header for this trace (``fields`` override the defaults)."""
        h: dict[str, Any] = {
            "model_name": "synthetic",
            "gguf_path": "synthetic.gguf",
            "gguf_size_bytes": 0,
            "n_embd": 64,
            "probs_top": self.P,
            "corpus_id": "synth",
            "created_at": SYNTH_CREATED_AT,
            "n_ctx": 4096,
            "n_batch": 1024,
            "n_ubatch": 512,
            "n_threads": 1,
            "flags": ["tests/synth.py"],
            "notes": ["synthetic routing trace generated by tests/synth.py; not model output"],
            "synthetic": {k: v for k, v in self.params.items() if _jsonable(v)},
        }
        h.update(fields)
        return tr.make_header(n_expert=self.E, n_expert_used=self.K, moe_layers=self.moe_layers, **h)


def _jsonable(v: Any) -> bool:
    try:
        json.dumps(v, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False


def _chunk_layout(T: int, n_chunks: int, gen_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(chunk, pos, phase) for T tokens split into n_chunks equal chunks with a generated tail."""
    n_chunks = max(1, min(int(n_chunks), T)) if T > 0 else 1
    bounds = np.linspace(0, T, n_chunks + 1).astype(np.int64)
    chunk = np.zeros(T, dtype=np.int64)
    pos = np.zeros(T, dtype=np.int64)
    phase = np.zeros(T, dtype=np.uint8)
    for c in range(n_chunks):
        a, b = int(bounds[c]), int(bounds[c + 1])
        chunk[a:b] = c
        pos[a:b] = np.arange(b - a)
        n_gen = int((b - a) * float(gen_frac))
        if n_gen > 0:
            phase[b - n_gen:b] = 1
    return chunk, pos, phase


def synth_trace(
    T: int = 2000,
    L: int = 4,
    K: int = 8,
    E: int = 16,
    alpha: float = 1.2,
    layer_corr: float = 0.0,
    seed: int = 0,
    *,
    P: int = 0,
    n_chunks: int = 4,
    gen_frac: float = 0.2,
    stickiness: float = 0.0,
    temperature: float = 1.0,
    vocab: int = GEMMA4_VOCAB,
    sources: Sequence[str] | None = None,
) -> SynthTrace:
    """Zipf expert popularity per layer, optional cross-layer correlation, phase vector.

    Deterministic for a given argument set. See the module docstring for the generative model.
    """
    T, L, K, E, P = int(T), int(L), int(K), int(E), int(P)
    if T < 1 or L < 1:
        raise ValueError("T and L must be >= 1")
    if not 1 <= K <= E <= tr.MOETRACE_MAX_EXPERTS:
        raise ValueError(f"need 1 <= K <= E <= {tr.MOETRACE_MAX_EXPERTS}, got K={K} E={E}")
    if not 0 <= P <= E:
        raise ValueError(f"need 0 <= P <= E, got P={P}")
    if alpha < 0 or not 0.0 <= layer_corr <= 1.0 or not 0.0 <= stickiness <= 1.0 or temperature <= 0:
        raise ValueError("need alpha >= 0, 0 <= layer_corr <= 1, 0 <= stickiness <= 1, temperature > 0")
    rng = np.random.default_rng(seed)

    ranks = np.arange(1, E + 1, dtype=np.float64) ** (-float(alpha))
    ranks /= ranks.sum()
    popularity = np.empty((L, E), dtype=np.float64)
    for j in range(L):
        popularity[j, rng.permutation(E)] = ranks  # expert perm[0] is the most popular at layer j
    layer_maps = np.empty((L, E), dtype=np.int64)
    layer_maps[0] = np.arange(E)
    for j in range(1, L):
        layer_maps[j] = rng.permutation(E)

    chunk, pos, phase = _chunk_layout(T, n_chunks, gen_frac)
    chunk_start = pos == 0

    ids = np.empty((T, L, K), dtype=np.uint8)
    weights = np.empty((T, L, K), dtype=np.float32)
    probs_ids = np.empty((T, L, P), dtype=np.uint8) if P else None
    probs = np.empty((T, L, P), dtype=np.float32) if P else None
    corr_mask = np.zeros((T, L), dtype=bool)
    sticky_mask = np.zeros((T, L), dtype=bool)
    margin = 0.25
    for j in range(L):
        g = np.log(popularity[j])[None, :] + rng.gumbel(size=(T, E))
        if j > 0 and layer_corr > 0:
            cm = rng.random(T) < layer_corr
            if cm.any():
                forced = layer_maps[j][ids[cm, j - 1, :].astype(np.int64)]  # [n, K], distinct per row
                rows = g[cm]
                fv = np.take_along_axis(rows, forced, axis=1)
                shift = rows.max(axis=1, keepdims=True) - fv.min(axis=1, keepdims=True) + margin
                np.put_along_axis(rows, forced, fv + shift, axis=1)  # forced set becomes the top K
                g[cm] = rows
            corr_mask[:, j] = cm
        if stickiness > 0:
            sm = (rng.random(T) < stickiness) & ~chunk_start
            sm[0] = False
            src = np.where(sm, 0, np.arange(T))
            src = np.maximum.accumulate(src)
            g = g[src]
            corr_mask[:, j] = corr_mask[src, j]
            sticky_mask[:, j] = sm
        order = np.argsort(-g, axis=1, kind="stable")
        z = g / float(temperature)
        z -= z.max(axis=1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=1, keepdims=True)
        top = order[:, :K]
        qk = np.take_along_axis(q, top, axis=1)
        ids[:, j, :] = top.astype(np.uint8)
        weights[:, j, :] = (qk / qk.sum(axis=1, keepdims=True)).astype(np.float32)
        if P:
            assert probs_ids is not None and probs is not None
            probs_ids[:, j, :] = order[:, :P].astype(np.uint8)
            probs[:, j, :] = np.take_along_axis(q, order[:, :P], axis=1).astype(np.float32)

    token_ids = rng.integers(3, max(int(vocab), 4), size=T).astype(np.int32)
    token_ids[chunk_start] = GEMMA4_BOS_ID
    params = {"generator": "synth_trace", "T": T, "L": L, "K": K, "E": E, "alpha": float(alpha),
              "layer_corr": float(layer_corr), "seed": int(seed), "P": P, "n_chunks": int(chunk.max()) + 1,
              "gen_frac": float(gen_frac), "stickiness": float(stickiness), "temperature": float(temperature)}
    return SynthTrace(ids=ids, weights=weights, phase=phase, chunk=chunk, pos=pos, token_ids=token_ids,
                      n_expert=E, popularity=popularity, layer_maps=layer_maps, corr_mask=corr_mask,
                      sticky_mask=sticky_mask, probs_ids=probs_ids, probs=probs,
                      sources=list(sources) if sources else list(DEFAULT_SOURCES), params=params)


def trace_from_ids(
    ids: Any,
    weights: Any = None,
    *,
    E: int | None = None,
    phase: Any = None,
    chunk: Any = None,
    pos: Any = None,
    token_ids: Any = None,
    sources: Sequence[str] | None = None,
) -> SynthTrace:
    """Wrap explicit routing ids [T, L, K] into a SynthTrace (weights default to uniform 1/K)."""
    ids_a = np.asarray(ids)
    if ids_a.ndim != 3:
        raise ValueError(f"ids must be [T, L, K], got shape {ids_a.shape}")
    T, L, K = ids_a.shape
    if T < 1:
        raise ValueError("need at least one token")
    E = int(E) if E is not None else int(ids_a.max()) + 1
    if ids_a.min() < 0 or ids_a.max() >= E or E > tr.MOETRACE_MAX_EXPERTS:
        raise ValueError(f"ids must be in 0..E-1 with E <= {tr.MOETRACE_MAX_EXPERTS}")
    w = np.full((T, L, K), 1.0 / K, dtype=np.float32) if weights is None else np.asarray(weights, np.float32)
    if w.shape != ids_a.shape:
        raise ValueError(f"weights shape {w.shape} != ids shape {ids_a.shape}")
    ch = np.zeros(T, np.int64) if chunk is None else np.asarray(chunk, np.int64)
    if pos is None:
        p = np.arange(T, dtype=np.int64)
        brk = np.flatnonzero(ch[1:] != ch[:-1]) + 1
        starts = np.zeros(T, np.int64)
        starts[brk] = brk
        pos_a = p - np.maximum.accumulate(starts)
    else:
        pos_a = np.asarray(pos, np.int64)
    ph = np.zeros(T, np.uint8) if phase is None else np.asarray(phase, np.uint8)
    tok = (np.arange(T, dtype=np.int32) % 1000 + 3) if token_ids is None else np.asarray(token_ids, np.int32)
    counts = np.zeros((L, E))
    for j in range(L):
        counts[j] = np.bincount(ids_a[:, j, :].ravel().astype(np.int64), minlength=E)[:E]
    pop = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return SynthTrace(ids=ids_a.astype(np.uint8), weights=w, phase=ph, chunk=ch, pos=pos_a, token_ids=tok,
                      n_expert=E, popularity=pop, layer_maps=np.tile(np.arange(E), (L, 1)),
                      corr_mask=np.zeros((T, L), bool), sticky_mask=np.zeros((T, L), bool),
                      sources=list(sources) if sources else list(DEFAULT_SOURCES),
                      params={"generator": "trace_from_ids", "T": T, "L": L, "K": K, "E": E})


def cyclic_trace(T: int, period: int, *, L: int = 1, K: int = 1, E: int | None = None) -> SynthTrace:
    """Token t uses experts ((t % period) * K + k) at every layer: ``period`` distinct sets cycle.

    With L=1, K=1 the item sequence is 0, 1, ..., period-1, 0, 1, ...: an LRU cache with fewer than
    ``period`` items never hits; Belady (offline optimal) does.
    """
    if period < 1 or K < 1 or L < 1 or T < 1:
        raise ValueError("T, period, L and K must be >= 1")
    E = int(E) if E is not None else period * K
    if E < period * K:
        raise ValueError(f"E={E} < period*K={period * K}")
    base = (np.arange(T) % period)[:, None] * K + np.arange(K)[None, :]  # [T, K]
    ids = np.repeat(base[:, None, :], L, axis=1)
    st = trace_from_ids(ids, E=E)
    st.params.update({"generator": "cyclic_trace", "period": int(period)})
    return st


def constant_trace(T: int, *, L: int = 4, K: int = 8, E: int = 16, experts: Sequence[int] | None = None) -> SynthTrace:
    """Every token routes to the same K experts at every layer (``experts`` default 0..K-1).

    Weights are geometric-like and strictly descending (sum 1). Single hot expert: K=1.
    """
    ex = np.arange(K) if experts is None else np.asarray(experts, np.int64)
    if ex.shape != (K,) or len(set(ex.tolist())) != K:
        raise ValueError("experts must be K distinct ids")
    ids = np.broadcast_to(ex[None, None, :], (T, L, K)).copy()
    raw = 0.5 ** np.arange(K, dtype=np.float64)
    w = np.broadcast_to((raw / raw.sum()).astype(np.float32), (T, L, K)).copy()
    st = trace_from_ids(ids, w, E=E)
    st.params.update({"generator": "constant_trace", "experts": [int(x) for x in ex]})
    return st


def write_synth_trace(
    path: str | os.PathLike[str],
    trace: SynthTrace | None = None,
    *,
    header: Mapping[str, Any] | None = None,
    write_gen: bool = False,
    write_meta: bool = False,
    strict: bool = True,
    **synth_kwargs: Any,
) -> SynthTrace:
    """Write a synthetic trace with pigemma.trace.TraceWriter and return it (``.path`` set).

    ``trace`` defaults to ``synth_trace(**synth_kwargs)``. One TraceWriter chunk per synthetic chunk
    (source = ``trace.sources`` cycled). ``write_gen`` adds ``<trace>.gen.txt`` (one separator plus a
    short English sentence per chunk with generated tokens); ``write_meta`` adds ``<trace>.meta.json``
    via pigemma.trace.build_meta (coherent=None, no sha, a fixed fake host).
    """
    if trace is not None and synth_kwargs:
        raise TypeError("pass either a SynthTrace or synth_trace() keyword arguments, not both")
    st = trace if trace is not None else synth_trace(**synth_kwargs)
    p = Path(os.fspath(path))
    hdr = st.header(**dict(header or {}))
    with tr.TraceWriter(p, hdr, strict=strict) as w:
        for c in range(st.n_chunks):
            sl = st.chunk_slice(c)
            n = sl.stop - sl.start
            w.begin_chunk(st.chunk_source(c))
            probs = None
            if st.P:
                assert st.probs_ids is not None and st.probs is not None
                probs = (st.probs_ids[sl], st.probs[sl])
            w.write_many(st.ids[sl], st.weights[sl], phase=st.phase[sl], pos=st.pos[sl],
                         token_ids=st.token_ids[sl], chunk_ids=np.full(n, c, dtype=np.int64), probs=probs)
    if write_gen:
        parts = []
        for c in range(st.n_chunks):
            sl = st.chunk_slice(c)
            parts.append(tr.GEN_CHUNK_SEPARATOR.format(i=c))
            if np.any(st.phase[sl] == 1):
                parts.append(f"Synthetic generated text for chunk {c}: the experts were routed as planned.")
        tr.gen_path(p).write_text("".join(parts), encoding="utf-8", newline="")
    if write_meta:
        host = {"hostname": "synth-host", "os": "synthetic", "kernel": "0", "cpu": "synthetic",
                "n_logical": 1, "n_physical": 1, "ram_total_bytes": 0, "ram_avail_bytes": 0,
                "page_size": 4096, "in_container": False, "cgroup_mem_max": None, "python": "3"}
        meta = tr.build_meta(p, argv=["tests/synth.py"], wall_ms=0.0, coherent=None, host=host,
                             notes=["synthetic trace (tests/synth.py)"], with_sha=False)
        tr.write_meta(p, meta)
    st.path = p
    return st


# ==================================================================================================
# Tiny GGUFs
# ==================================================================================================
_TYPE_IDS = {"F32": 0, "F16": 1, "Q4_0": 2, "Q8_0": 8}
_BLOCK = {"F32": (1, 4), "F16": (1, 2), "Q4_0": (32, 18), "Q8_0": (32, 34)}

#: The manifest's tiny MoE spec (files[] tests/conftest.py).
TINY_MOE_SPEC: dict[str, Any] = {
    "arch": "gemma4", "n_layer": 2, "n_embd": 64, "n_ff": 48, "n_ff_exp": 32, "n_expert": 4,
    "n_expert_used": 2, "n_head": 2, "head_count_kv": (2, 1), "key_length": 32, "key_length_swa": 16,
    "sliding_window": 8, "sliding_pattern": (True, False), "n_vocab": 16, "context_length": 128,
}
#: The manifest's tiny dense spec: 2 layers, no experts, ffn gate/up/down Q4_0 [64,256]/[256,64].
TINY_DENSE_SPEC: dict[str, Any] = dict(TINY_MOE_SPEC, n_ff=256, n_expert=0, n_expert_used=0, n_ff_exp=0)


def expected_kind(name: str) -> tuple[str, int | None]:
    """Independent re-statement of the map.json tensor kinds (TRACE_FORMAT 8) for synthetic files."""
    m = re.match(r"blk\.(\d+)\.", name)
    layer = int(m.group(1)) if m else None
    if re.fullmatch(r"blk\.\d+\.ffn_(gate_up|gate|up|down)_exps\.weight", name):
        kind = "expert"
    elif re.fullmatch(r"blk\.\d+\.ffn_(gate_up|gate|up|down)_exps\.scale", name):
        kind = "expert_scale"
    elif re.fullmatch(r"blk\.\d+\.ffn_gate_inp\.(weight|scale)", name):
        kind = "router"
    elif name.startswith("token_embd."):
        kind = "token_embd"
    elif name == "output.weight":
        kind = "output"
    elif re.fullmatch(r"blk\.\d+\.attn_(q|k|v|output)\.weight", name):
        kind = "attn"
    elif re.fullmatch(r"blk\.\d+\.ffn_(gate|up|down)\.weight", name):
        kind = "ffn_dense"
    elif "norm" in name or name.endswith("layer_output_scale.weight"):
        kind = "norm"
    else:
        kind = "other"
    return kind, layer


@dataclass(repr=False, eq=False)
class TinyTensor:
    """One tensor as the writer laid it out."""

    name: str
    type: str                 # 'F32' | 'F16' | 'Q4_0' | 'Q8_0'
    shape: list[int]          # GGUF ne order (ne[0] fastest)
    n_bytes: int
    offset: int               # absolute file offset of the first data byte
    rel_offset: int           # offset inside the data section
    kind: str
    layer: int | None
    stored: np.ndarray        # array handed to GGUFWriter.add_tensor (float, or uint8 blocks)
    source: np.ndarray        # float32 values before quantisation, numpy shape = reversed(ne)

    @property
    def raw(self) -> bytes:
        return self.stored.tobytes()

    def __repr__(self) -> str:
        return (f"TinyTensor({self.name!r}, {self.type}, ne={self.shape}, n_bytes={self.n_bytes}, "
                f"offset={self.offset}, kind={self.kind}, layer={self.layer})")

    @property
    def n_elements(self) -> int:
        return int(np.prod(self.shape))

    def values(self) -> np.ndarray:
        """float32 values as llama.cpp sees them (dequantised for Q types), numpy shape reversed(ne)."""
        if self.type in ("F32", "F16"):
            return np.asarray(self.stored, dtype=np.float32)
        import gguf

        return np.asarray(gguf.quants.dequantize(self.stored, gguf.GGMLQuantizationType[self.type]),
                          dtype=np.float32).reshape(tuple(reversed(self.shape)))

    def matrix(self) -> np.ndarray:
        """2-D weight as W [n_out, n_in] with y = W @ x (GGUF ne = [n_in, n_out])."""
        if len(self.shape) != 2:
            raise ValueError(f"{self.name} is not a matrix: ne={self.shape}")
        return self.values()


@dataclass(repr=False, eq=False)
class TinyGguf(os.PathLike):
    """A synthetic GGUF plus the writer's view of it. Usable as a path and unpackable.

    ``Path(g)``, ``open(g)``, ``os.fspath(g)`` and ``str(g)`` give the file path;
    ``path, offsets = g`` gives (Path, {tensor name: absolute offset}).
    """

    path: Path
    arch: str
    kv: dict[str, Any]
    tensors: dict[str, TinyTensor]
    alignment: int
    data_offset: int
    file_size: int
    params: dict[str, Any]

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)

    def __iter__(self) -> Iterator[Any]:
        return iter((self.path, self.offsets))

    def __repr__(self) -> str:
        return (f"TinyGguf({str(self.path)!r}, kind={self.params.get('kind')}, n_tensors={len(self.tensors)}, "
                f"file_size={self.file_size}, data_offset={self.data_offset}, alignment={self.alignment})")

    # -- writer view ----------------------------------------------------------------------------
    @property
    def offsets(self) -> dict[str, int]:
        return {n: t.offset for n, t in self.tensors.items()}

    @property
    def n_bytes(self) -> dict[str, int]:
        return {n: t.n_bytes for n, t in self.tensors.items()}

    @property
    def kinds(self) -> dict[str, str]:
        return {n: t.kind for n, t in self.tensors.items()}

    @property
    def n_layer(self) -> int:
        return int(self.params["n_layer"])

    @property
    def n_expert(self) -> int:
        return int(self.params.get("n_expert") or 0)

    @property
    def n_expert_used(self) -> int:
        return int(self.params.get("n_expert_used") or 0)

    @property
    def n_embd(self) -> int:
        return int(self.params["n_embd"])

    @property
    def n_ff(self) -> int:
        return int(self.params["n_ff"])

    @property
    def n_ff_exp(self) -> int:
        return int(self.params.get("n_ff_exp") or 0)

    @property
    def is_moe(self) -> bool:
        return self.n_expert > 0

    def tensor(self, name: str) -> TinyTensor:
        return self.tensors[name]

    def raw_bytes(self, name: str) -> bytes:
        """Bytes of one tensor read back from the file (for slab checks)."""
        t = self.tensors[name]
        with open(self.path, "rb") as f:
            f.seek(t.offset)
            return f.read(t.n_bytes)

    def expert_layers(self) -> list[dict[str, Any]]:
        """map.json-style expert_layers: [{layer, gate_up|gate|up|down: {offset, n_bytes, nb_expert}, n_expert, unit_bytes}]."""
        out = []
        for il in range(self.n_layer):
            parts: dict[str, Any] = {"layer": il}
            unit = 0
            for part in ("gate_up", "gate", "up", "down"):
                t = self.tensors.get(f"blk.{il}.ffn_{part}_exps.weight")
                if t is None:
                    continue
                nb = t.n_bytes // t.shape[2]
                parts[part] = {"offset": t.offset, "n_bytes": t.n_bytes, "nb_expert": nb}
                unit += nb
            if unit:
                parts["n_expert"] = self.n_expert
                parts["unit_bytes"] = unit
                out.append(parts)
        return out

    def expert_ranges(self, il: int, e: int) -> list[tuple[int, int]]:
        """(offset, length) of expert ``e`` of layer ``il`` in every expert tensor (gate_up, then down)."""
        el = next((x for x in self.expert_layers() if x["layer"] == il), None)
        if el is None:
            raise KeyError(f"layer {il} has no expert tensors")
        if not 0 <= e < self.n_expert:
            raise IndexError(f"expert {e} outside 0..{self.n_expert - 1}")
        return [(el[p]["offset"] + e * el[p]["nb_expert"], el[p]["nb_expert"])
                for p in ("gate_up", "gate", "up", "down") if p in el]

    def summary(self) -> dict[str, Any]:
        """The map.json summary numbers that do not depend on page size or KV-cache rules."""
        by_kind: dict[str, int] = {}
        for t in self.tensors.values():
            by_kind[t.kind] = by_kind.get(t.kind, 0) + t.n_bytes
        el = self.expert_layers()
        units = [x["unit_bytes"] for x in el]
        expert_bytes = by_kind.get("expert", 0)
        total = sum(t.n_bytes for t in self.tensors.values())
        return {
            "nonexpert_bytes": total - expert_bytes,
            "expert_bytes": expert_bytes,
            "expert_unit_bytes": int(sum(units) // len(units)) if units else 0,
            "n_moe_layers": len(el),
            "expert_bytes_per_token_worst": self.n_expert_used * sum(units),
            "tensor_bytes_total": total,
            "bytes_per_kind": by_kind,
        }


def _pad(n: int, align: int) -> int:
    return -(-int(n) // align) * align


class _Builder:
    """Collects tensors for GGUFWriter and records the writer's layout."""

    def __init__(self, path: Path, arch: str, seed: int, alignment: int | None) -> None:
        import gguf

        self.gguf = gguf
        self.path = path
        self.arch = arch
        self.rng = np.random.default_rng(seed)
        self.writer = gguf.GGUFWriter(str(path), arch)
        self.alignment = int(alignment) if alignment else 32
        if alignment:
            self.writer.add_custom_alignment(int(alignment))
        self.kv: dict[str, Any] = {"general.architecture": arch}
        if alignment:
            self.kv["general.alignment"] = int(alignment)
        self.order: list[TinyTensor] = []
        self.fallbacks: list[str] = []

    # KV ------------------------------------------------------------------------------------------
    def u32(self, key: str, v: int) -> None:
        self.writer.add_uint32(key, int(v))
        self.kv[key] = int(v)

    def f32(self, key: str, v: float) -> None:
        self.writer.add_float32(key, float(v))
        self.kv[key] = float(v)

    def string(self, key: str, v: str) -> None:
        self.writer.add_string(key, v)
        self.kv[key] = v

    def boolean(self, key: str, v: bool) -> None:
        self.writer.add_bool(key, bool(v))
        self.kv[key] = bool(v)

    def array(self, key: str, v: Sequence[Any]) -> None:
        self.writer.add_array(key, list(v))
        self.kv[key] = list(v)

    # tensors ---------------------------------------------------------------------------------------
    def tensor(self, name: str, ne: Sequence[int], typ: str, *, scale: float = 0.05, values: np.ndarray | None = None,
               ones: bool = False) -> None:
        ne = [int(x) for x in ne]
        np_shape = tuple(reversed(ne))
        if values is not None:
            src = np.asarray(values, dtype=np.float32).reshape(np_shape)
        elif ones:
            src = np.ones(np_shape, dtype=np.float32)
        else:
            src = (self.rng.standard_normal(np_shape) * scale).astype(np.float32)
        bs, ts = _BLOCK[typ]
        if ne[0] % bs:
            self.fallbacks.append(f"{name}: {typ} needs ne[0] % {bs} == 0, ne={ne} -> F16")
            typ = "F16"
            bs, ts = _BLOCK[typ]
        if typ == "F32":
            stored = src
        elif typ == "F16":
            stored = src.astype(np.float16)
        else:
            qt = self.gguf.GGMLQuantizationType[typ]
            stored = self.gguf.quants.quantize(src, qt)
        n_bytes = int(np.prod(ne)) // bs * ts
        assert stored.nbytes == n_bytes, (name, stored.nbytes, n_bytes)
        if typ in ("F32", "F16"):
            self.writer.add_tensor(name, stored)
        else:
            self.writer.add_tensor(name, stored, raw_dtype=self.gguf.GGMLQuantizationType[typ])
        kind, layer = expected_kind(name)
        self.order.append(TinyTensor(name=name, type=typ, shape=ne, n_bytes=n_bytes, offset=-1, rel_offset=-1,
                                     kind=kind, layer=layer, stored=stored, source=src))

    def finish(self, params: dict[str, Any]) -> TinyGguf:
        w = self.writer
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        size = self.path.stat().st_size
        padded = [_pad(t.n_bytes, self.alignment) for t in self.order]
        data_offset = size - sum(padded)  # GGUFWriter pads every tensor, the last one included
        rel = 0
        for t, pb in zip(self.order, padded):
            t.rel_offset = rel
            t.offset = data_offset + rel
            rel += pb
        params = dict(params)
        params["type_fallbacks"] = list(self.fallbacks)
        return TinyGguf(path=self.path, arch=self.arch, kv=self.kv, tensors={t.name: t for t in self.order},
                        alignment=self.alignment, data_offset=data_offset, file_size=size, params=params)


def write_tiny_gguf(
    path: str | os.PathLike[str],
    *,
    moe: bool = True,
    n_layer: int = 2,
    n_embd: int = 64,
    n_ff: int | None = None,
    n_ff_exp: int = 32,
    n_expert: int = 4,
    n_expert_used: int = 2,
    n_head: int = 2,
    head_count_kv: Sequence[int] = (2, 1),
    key_length: int = 32,
    key_length_swa: int = 16,
    sliding_window: int = 8,
    sliding_pattern: Sequence[bool] = (True, False),
    n_vocab: int = 16,
    context_length: int = 128,
    qtype: str = "Q4_0",
    alignment: int | None = None,
    drop_keys: Sequence[str] = (),
    with_output: bool = False,
    seed: int = 0,
) -> TinyGguf:
    """Write a tiny gemma4-shaped GGUF with gguf.GGUFWriter (header + tensors, no tokenizer).

    Defaults are the manifest's ``tiny_moe_gguf`` (n_ff 48). ``moe=False`` drops experts, router and
    MoE norms and defaults n_ff to 256 (``tiny_dense_gguf``). ``head_count_kv`` and
    ``sliding_pattern`` are cycled to n_layer entries. ``drop_keys`` omits KV keys (suffixes after
    'gemma4.' or full names) to test missing-key errors. Not loadable by llama.cpp (no vocabulary).
    """
    p = Path(os.fspath(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    if n_ff is None:
        n_ff = 48 if moe else 256
    if not moe:
        n_expert = n_expert_used = n_ff_exp = 0
    hkv = [int(head_count_kv[i % len(head_count_kv)]) for i in range(n_layer)]
    swa = [bool(sliding_pattern[i % len(sliding_pattern)]) for i in range(n_layer)]
    a = "gemma4"
    b = _Builder(p, a, seed, alignment)
    drop = {k if "." in k and k.split(".")[0] in ("general", a, "tokenizer") else f"{a}.{k}" for k in drop_keys}

    def kv(kind: str, key: str, val: Any) -> None:
        if key in drop:
            return
        getattr(b, kind)(key, val)

    kv("string", "general.name", "pigemma-tiny-" + ("moe" if moe else "dense"))
    kv("u32", "general.file_type", 2 if qtype == "Q4_0" else (7 if qtype == "Q8_0" else 1))
    kv("u32", "general.quantization_version", 2)
    kv("u32", f"{a}.block_count", n_layer)
    kv("u32", f"{a}.context_length", context_length)
    kv("u32", f"{a}.embedding_length", n_embd)
    kv("u32", f"{a}.feed_forward_length", n_ff)
    kv("u32", f"{a}.attention.head_count", n_head)
    kv("array", f"{a}.attention.head_count_kv", hkv)
    kv("f32", f"{a}.rope.freq_base", 1_000_000.0)
    kv("f32", f"{a}.rope.freq_base_swa", 10_000.0)
    kv("f32", f"{a}.attention.layer_norm_rms_epsilon", 1e-6)
    if moe:
        kv("u32", f"{a}.expert_count", n_expert)
        kv("u32", f"{a}.expert_used_count", n_expert_used)
        kv("u32", f"{a}.expert_feed_forward_length", n_ff_exp)
    kv("u32", f"{a}.attention.key_length", key_length)
    kv("u32", f"{a}.attention.value_length", key_length)
    kv("f32", f"{a}.final_logit_softcapping", 30.0)
    kv("u32", f"{a}.attention.sliding_window", sliding_window)
    kv("u32", f"{a}.attention.shared_kv_layers", 0)
    kv("array", f"{a}.attention.sliding_window_pattern", swa)
    kv("u32", f"{a}.attention.key_length_swa", key_length_swa)
    kv("u32", f"{a}.attention.value_length_swa", key_length_swa)
    kv("u32", f"{a}.rope.dimension_count", key_length)
    kv("u32", f"{a}.rope.dimension_count_swa", key_length_swa)

    b.tensor("token_embd.weight", [n_embd, n_vocab], "F16", scale=1.0)
    b.tensor("output_norm.weight", [n_embd], "F32", ones=True)
    b.tensor("rope_freqs.weight", [key_length // 2], "F32", ones=True)
    if with_output:
        b.tensor("output.weight", [n_embd, n_vocab * 2], qtype)
    for il in range(n_layer):
        hd = key_length_swa if swa[il] else key_length
        pre = f"blk.{il}."
        b.tensor(pre + "attn_norm.weight", [n_embd], "F32", ones=True)
        b.tensor(pre + "attn_q.weight", [n_embd, n_head * hd], qtype)
        b.tensor(pre + "attn_k.weight", [n_embd, hkv[il] * hd], qtype)
        if swa[il]:  # global layers share K and V in Gemma 4 (no attn_v), like the real files
            b.tensor(pre + "attn_v.weight", [n_embd, hkv[il] * hd], qtype)
        b.tensor(pre + "attn_output.weight", [n_head * hd, n_embd], qtype)
        b.tensor(pre + "attn_q_norm.weight", [hd], "F32", ones=True)
        b.tensor(pre + "attn_k_norm.weight", [hd], "F32", ones=True)
        b.tensor(pre + "post_attention_norm.weight", [n_embd], "F32", ones=True)
        b.tensor(pre + "ffn_norm.weight", [n_embd], "F32", ones=True)
        b.tensor(pre + "ffn_gate.weight", [n_embd, n_ff], qtype)
        b.tensor(pre + "ffn_up.weight", [n_embd, n_ff], qtype)
        b.tensor(pre + "ffn_down.weight", [n_ff, n_embd], qtype)
        if moe:
            b.tensor(pre + "ffn_gate_inp.weight", [n_embd, n_expert], "F32", scale=1.0)
            b.tensor(pre + "ffn_gate_inp.scale", [n_embd], "F32", ones=True)
            b.tensor(pre + "pre_ffw_norm_2.weight", [n_embd], "F32", ones=True)
            b.tensor(pre + "ffn_gate_up_exps.weight", [n_embd, 2 * n_ff_exp, n_expert], qtype)
            b.tensor(pre + "ffn_down_exps.weight", [n_ff_exp, n_embd, n_expert], qtype)
            b.tensor(pre + "ffn_down_exps.scale", [n_expert], "F32", ones=True)
            b.tensor(pre + "post_ffw_norm_1.weight", [n_embd], "F32", ones=True)
            b.tensor(pre + "post_ffw_norm_2.weight", [n_embd], "F32", ones=True)
        b.tensor(pre + "post_ffw_norm.weight", [n_embd], "F32", ones=True)
        b.tensor(pre + "layer_output_scale.weight", [1], "F32", ones=True)
    params = {"kind": "moe" if moe else "dense", "arch": a, "n_layer": n_layer, "n_embd": n_embd, "n_ff": n_ff,
              "n_ff_exp": n_ff_exp, "n_expert": n_expert, "n_expert_used": n_expert_used, "n_head": n_head,
              "head_count_kv": hkv, "key_length": key_length, "key_length_swa": key_length_swa,
              "sliding_window": sliding_window, "sliding_pattern": swa, "n_vocab": n_vocab,
              "context_length": context_length, "qtype": qtype, "seed": seed, "dropped_keys": sorted(drop)}
    return b.finish(params)


def write_tiny_dense_gguf(path: str | os.PathLike[str], **kwargs: Any) -> TinyGguf:
    """``tiny_dense_gguf``: 2 layers, no experts, ffn gate/up/down Q4_0 [64,256]/[256,64]."""
    kwargs.setdefault("n_ff", 256)
    return write_tiny_gguf(path, moe=False, **kwargs)


def _spm_byte_vocab() -> tuple[list[str], list[float], list[int]]:
    """<unk> <s> </s>, the 256 byte-fallback tokens, then U+2581 and printable ASCII (SentencePiece)."""
    tokens = ["<unk>", "<s>", "</s>"]
    scores = [0.0, 0.0, 0.0]
    types = [2, 3, 3]  # UNKNOWN, CONTROL, CONTROL
    for i in range(256):
        tokens.append(f"<0x{i:02X}>")
        scores.append(0.0)
        types.append(6)  # BYTE
    normal = ["▁"] + [chr(c) for c in range(33, 127)]
    for i, t in enumerate(normal):
        tokens.append(t)
        scores.append(-float(i + 1))
        types.append(1)  # NORMAL
    return tokens, scores, types


def write_tiny_llama_gguf(
    path: str | os.PathLike[str],
    *,
    qtype: str = "F32",
    n_layer: int = 2,
    n_embd: int = 64,
    n_head: int = 4,
    n_head_kv: int = 2,
    n_ff: int = 128,
    context_length: int = 256,
    seed: int = 0,
    name: str = "pigemma-tiny-llama",
) -> TinyGguf:
    """A tiny *runnable* llama-architecture GGUF (random weights, byte-level SPM vocabulary, 354 tokens).

    The same ``seed`` gives the same underlying float weights for every ``qtype`` (F32, F16, Q4_0,
    Q8_0), so a Q4_0 file is the quantised version of the F32 file (useful for a real
    ``llama-perplexity --kl-divergence`` run). Norm weights stay F32. Output is gibberish but
    deterministic under --temp 0. Loads with llama.cpp b11398 (llama-bench, llama-completion,
    llama-perplexity, llama-server) in well under a second.
    """
    p = Path(os.fspath(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    tokens, scores, types = _spm_byte_vocab()
    n_vocab = len(tokens)
    a = "llama"
    b = _Builder(p, a, seed, None)
    b.string("general.name", name)
    b.u32("general.file_type", {"F32": 0, "F16": 1, "Q4_0": 2, "Q8_0": 7}[qtype])
    b.u32("general.quantization_version", 2)
    b.u32(f"{a}.vocab_size", n_vocab)
    b.u32(f"{a}.context_length", context_length)
    b.u32(f"{a}.embedding_length", n_embd)
    b.u32(f"{a}.block_count", n_layer)
    b.u32(f"{a}.feed_forward_length", n_ff)
    b.u32(f"{a}.rope.dimension_count", n_embd // n_head)
    b.u32(f"{a}.attention.head_count", n_head)
    b.u32(f"{a}.attention.head_count_kv", n_head_kv)
    b.f32(f"{a}.attention.layer_norm_rms_epsilon", 1e-5)
    b.f32(f"{a}.rope.freq_base", 10000.0)
    b.string("tokenizer.ggml.model", "llama")
    b.array("tokenizer.ggml.tokens", tokens)
    b.array("tokenizer.ggml.scores", scores)
    b.array("tokenizer.ggml.token_type", types)
    b.u32("tokenizer.ggml.unknown_token_id", 0)
    b.u32("tokenizer.ggml.bos_token_id", 1)
    b.u32("tokenizer.ggml.eos_token_id", 2)
    b.boolean("tokenizer.ggml.add_bos_token", True)
    b.boolean("tokenizer.ggml.add_eos_token", False)
    hd = n_embd // n_head
    b.tensor("token_embd.weight", [n_embd, n_vocab], qtype, scale=0.5)
    b.tensor("output_norm.weight", [n_embd], "F32", ones=True)
    b.tensor("output.weight", [n_embd, n_vocab], qtype, scale=0.2)
    for il in range(n_layer):
        pre = f"blk.{il}."
        b.tensor(pre + "attn_norm.weight", [n_embd], "F32", ones=True)
        b.tensor(pre + "attn_q.weight", [n_embd, n_embd], qtype, scale=0.1)
        b.tensor(pre + "attn_k.weight", [n_embd, n_head_kv * hd], qtype, scale=0.1)
        b.tensor(pre + "attn_v.weight", [n_embd, n_head_kv * hd], qtype, scale=0.1)
        b.tensor(pre + "attn_output.weight", [n_embd, n_embd], qtype, scale=0.1)
        b.tensor(pre + "ffn_norm.weight", [n_embd], "F32", ones=True)
        b.tensor(pre + "ffn_gate.weight", [n_embd, n_ff], qtype, scale=0.1)
        b.tensor(pre + "ffn_up.weight", [n_embd, n_ff], qtype, scale=0.1)
        b.tensor(pre + "ffn_down.weight", [n_ff, n_embd], qtype, scale=0.1)
    params = {"kind": "llama", "arch": a, "n_layer": n_layer, "n_embd": n_embd, "n_ff": n_ff, "n_head": n_head,
              "n_head_kv": n_head_kv, "n_vocab": n_vocab, "context_length": context_length, "qtype": qtype,
              "seed": seed, "n_expert": 0, "n_expert_used": 0}
    return b.finish(params)


# ==================================================================================================
# Activation dumps (H2)
# ==================================================================================================
def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """GELU with the tanh approximation (Gemma's activation), float32."""
    x = np.asarray(x, dtype=np.float32)
    return (0.5 * x * (1.0 + np.tanh(np.float32(math.sqrt(2.0 / math.pi)) * (x + np.float32(0.044715) * x ** 3)))
            ).astype(np.float32)


@dataclass
class SynthActs:
    """What ``write_synth_acts_dir`` wrote, with the float32 arrays and weights used."""

    path: Path
    layers: list[int]
    x: dict[int, np.ndarray]           # ffn_norm  [T, n_embd]  (as stored, f16 -> f32)
    geglu: dict[int, np.ndarray]       # ffn_geglu [T, n_ff]
    out: dict[int, np.ndarray]         # ffn_out   [T, n_embd]
    W: dict[tuple[int, str], np.ndarray]  # (layer, 'gate'|'up'|'down') -> W [n_out, n_in] float32
    phase: np.ndarray
    token_ids: np.ndarray


def write_synth_acts_dir(
    directory: str | os.PathLike[str],
    tiny: TinyGguf | None = None,
    *,
    layers: Sequence[int] = (0, 1),
    T: int = 32,
    n_embd: int = 64,
    n_ff: int = 256,
    gen_frac: float = 0.25,
    seed: int = 0,
    meta: Mapping[str, Any] | None = None,
) -> SynthActs:
    """Write an act-dump directory (TRACE_FORMAT 7) whose activations obey the dense FFN algebra.

    x = ffn_norm (random), geglu = gelu_tanh(W_gate x) * (W_up x), ffn_out = W_down @ geglu, where
    geglu is rounded to f16 *before* the down projection so ``ffn_out == W_down @ stored_geglu`` up
    to the final f16 rounding. With ``tiny`` (a dense or MoE TinyGguf) the dequantised
    blk.<il>.ffn_{gate,up,down} weights of that file are used, so an oracle can re-read them through
    gguf_map / h2.decompose; otherwise random W of shape [n_ff, n_embd] / [n_embd, n_ff].
    """
    d = Path(os.fspath(directory))
    rng = np.random.default_rng(seed)
    layers = [int(x) for x in layers]
    if tiny is not None:
        n_embd, n_ff = tiny.n_embd, tiny.n_ff
    W: dict[tuple[int, str], np.ndarray] = {}
    for il in layers:
        for m in ("gate", "up", "down"):
            if tiny is not None:
                W[(il, m)] = tiny.tensor(f"blk.{il}.ffn_{m}.weight").matrix()
            else:
                shape = (n_embd, n_ff) if m == "down" else (n_ff, n_embd)
                W[(il, m)] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
    arrays: dict[tuple[int, str], np.ndarray] = {}
    xs, gs, outs = {}, {}, {}
    for il in layers:
        x = rng.standard_normal((T, n_embd)).astype(np.float16).astype(np.float32)
        gate = x @ W[(il, "gate")].T
        up = x @ W[(il, "up")].T
        geglu = (gelu_tanh(gate) * up).astype(np.float16).astype(np.float32)
        out = (geglu @ W[(il, "down")].T).astype(np.float32)
        arrays[(il, "ffn_norm")] = x
        arrays[(il, "ffn_geglu")] = geglu
        arrays[(il, "ffn_out")] = out
        xs[il], gs[il], outs[il] = x, geglu, out.astype(np.float16).astype(np.float32)
    n_gen = int(T * gen_frac)
    phase = np.zeros(T, np.uint8)
    if n_gen:
        phase[T - n_gen:] = 1
    toks = rng.integers(3, GEMMA4_VOCAB, size=T).astype(np.int32)
    toks[0] = GEMMA4_BOS_ID
    m: dict[str, Any] = {
        "model": tiny.path.name if tiny is not None else "synthetic",
        "gguf_path": str(tiny.path) if tiny is not None else "synthetic.gguf",
        "flags": ["tests/synth.py"],
        "chunks": [{"chunk_id": 0, "source": "synth", "n_prompt": T - n_gen, "n_gen": n_gen}],
        "corpus_id": "synth",
        "created_at": SYNTH_CREATED_AT,
        "notes": ["synthetic activations (tests/synth.py): ffn_out = W_down @ ffn_geglu"],
    }
    m.update(meta or {})
    tr.write_acts_dir(d, arrays, phase=phase, token_ids=toks, meta=m)
    return SynthActs(path=d, layers=layers, x=xs, geglu=gs, out=outs, W=W, phase=phase, token_ids=toks)


# ==================================================================================================
# Pi facts fixtures and checked-in sample files
# ==================================================================================================
#: Names of tests/fixtures/facts_<name>.json. The first five are the manifest's list.
FACTS_NAMES = (
    "pi5-4gb-microsd",        # the originally probed context-friend-1 (4 GB, microSD, Hailo HAT, protected services)
    "pi5-8gb-usb3",
    "pi5-8gb-nvme-gen3",
    "pi5-16gb-nvme-gen3",
    "generic-linux-nodisk",   # x86-64 container, storage unknown, 1.5 GB free -> NO_DISK
    "pi5-16gb-microsd-real",  # the real target board behind context-friend-1 since 2026-10-05 (FACTS UPDATE)
    "pi5-8gb-nvme-busy",      # load average 3.6 on 4 cores -> busy host refusal
)
#: Top-level keys of data_formats "facts.json" (every fixture has exactly these plus the optional ones).
FACTS_KEYS = ("hostname", "kernel", "page_size", "cpu_model", "cpu_flags", "nproc", "ram_total_kb",
              "ram_available_kb", "swap_total_kb", "loadavg1", "model_dir", "storage", "free_disk_bytes",
              "running_services", "hat_product", "pciex1_gen_cfg", "governor", "memlock_limit", "tools",
              "override", "collected_at")
FACTS_OPTIONAL_KEYS = ("emulated", "in_container", "notes", "device", "cgroup_mem_max")
FACTS_STORAGE_KEYS = ("dev", "type", "rotational", "pcie_link_speed", "pcie_link_width", "read_ahead_kb")


def fixture_path(name: str) -> Path:
    """tests/fixtures/<name> (no existence check)."""
    return FIXTURES_DIR / name


def read_fixture_text(name: str) -> str:
    return fixture_path(name).read_text(encoding="utf-8")


def read_fixture_json(name: str) -> Any:
    return json.loads(fixture_path(name).read_text(encoding="utf-8"))


def facts_path(name: str) -> Path:
    """tests/fixtures/facts_<name>.json; FileNotFoundError naming the known fixtures if absent."""
    p = FIXTURES_DIR / f"facts_{name}.json"
    if not p.is_file():
        raise FileNotFoundError(f"no facts fixture {p.name}; known: {', '.join(FACTS_NAMES)}")
    return p


def load_facts(name: str, **overrides: Any) -> dict[str, Any]:
    """A fresh (deep-copied) facts dict. ``overrides`` replace top-level keys; a dict value for a
    dict key (``storage``, ``tools``) is merged into it instead of replacing it."""
    facts = json.loads(facts_path(name).read_text(encoding="utf-8"))
    for k, v in overrides.items():
        if isinstance(v, Mapping) and isinstance(facts.get(k), dict):
            facts[k] = {**facts[k], **copy.deepcopy(dict(v))}
        else:
            facts[k] = copy.deepcopy(v)
    return facts
