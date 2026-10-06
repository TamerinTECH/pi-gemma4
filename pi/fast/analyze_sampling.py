#!/usr/bin/env python3
"""Sampling audit of the drafter vocabulary shortcut (temperature > 0), desktop runs of mtp_ab.py.

  seeded  six prompts, temperature 0.8, fixed seed per prompt: with the default verification (the target samples,
          the draft is accepted while it equals the sample) the text must equal the run without drafter.
  digits  400 seeds of one "four random digits" prompt, temperature 1.0: digit histograms per position for
          plain sampling, match verification and rejection-sampling verification (--spec-draft-sampling
          probabilistic), each with the dense and the cluster head; chi-square test of homogeneity against plain.

Usage: analyze_sampling.py [--root results/desktop-fast/sampling] [--out FILE.json]
"""
import argparse
import collections
import json
import math
import os
import re
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["config"]: r for r in json.load(f)["runs"]}


def gammaincc(a, x):
    """Regularized upper incomplete gamma Q(a, x)."""
    if x <= 0:
        return 1.0
    if x < a + 1:  # series for P
        term = 1.0 / a
        total = term
        n = a
        for _ in range(500):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1 - a  # continued fraction for Q
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_homogeneity(h1, h2):
    keys = sorted(set(h1) | set(h2))
    n1, n2 = sum(h1.values()), sum(h2.values())
    if not n1 or not n2:
        return None
    # pool categories with a small expected count
    rows = [(h1.get(k, 0), h2.get(k, 0)) for k in keys]
    big = [(a, b) for a, b in rows if (a + b) * min(n1, n2) / (n1 + n2) >= 5]
    rest = [(a, b) for a, b in rows if (a + b) * min(n1, n2) / (n1 + n2) < 5]
    if rest:
        big.append((sum(a for a, _ in rest), sum(b for _, b in rest)))
    big = [(a, b) for a, b in big if a + b > 0]
    stat = 0.0
    for a, b in big:
        e1, e2 = (a + b) * n1 / (n1 + n2), (a + b) * n2 / (n1 + n2)
        stat += (a - e1) ** 2 / e1 + (b - e2) ** 2 / e2
    dof = len(big) - 1
    return {"chi2": round(stat, 2), "dof": dof, "p": round(gammaincc(dof / 2.0, stat / 2.0), 4) if dof > 0 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/desktop-fast/sampling")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    summary = {}
    p = os.path.join(a.root, "seeded", "ab.json")
    if os.path.exists(p):
        runs = load(p)
        base = {r["id"]: r for r in runs["plain"]["requests"]}
        print("== seeded (temperature 0.8, one seed per prompt): text equal to plain sampling with the same seed")
        summary["seeded"] = {}
        for name, run in runs.items():
            reqs = run.get("requests") or []
            same = sum(1 for r in reqs if r["text_sha256"] == base[r["id"]]["text_sha256"])
            gen = sum(r["gen_n"] or 0 for r in reqs)
            acc = sum(r.get("draft_n_accepted") or 0 for r in reqs)
            dr = sum(r.get("draft_n") or 0 for r in reqs)
            # first position where the text leaves the plain text (characters), for the runs that differ
            div = []
            for r in reqs:
                x, y = r["content"], base[r["id"]]["content"]
                if x != y:
                    k = next((i for i, (c1, c2) in enumerate(zip(x, y)) if c1 != c2), min(len(x), len(y)))
                    div.append((r["id"], k, len(y)))
            summary["seeded"][name] = {"same_text": f"{same}/{len(reqs)}", "generated": gen, "accepted": acc, "drafted": dr,
                                       "tokens_per_cycle": round(gen / (gen - acc), 3) if gen > acc else None, "diverged_at_char": div}
            print(f"  {name:11s} same text {same}/{len(reqs)} | drafted {dr}, accepted {acc} ({(acc / dr if dr else 0):.3f}) | "
                  f"tokens per cycle {summary['seeded'][name]['tokens_per_cycle']} | error {run.get('error')}")
    p = os.path.join(a.root, "digits", "ab.json")
    if os.path.exists(p):
        runs = load(p)
        print("== digits (temperature 1.0, 400 seeds): digit histograms per position, chi-square against plain sampling")
        summary["digits"] = {}
        hist = {}
        for name, run in runs.items():
            reqs = run.get("requests") or []
            hs = [collections.Counter() for _ in range(4)]
            wellformed = 0
            for r in reqs:
                d = re.findall(r"\d", r["content"])
                if re.fullmatch(r"\s*\d \d \d \d\s*", r["content"] or ""):
                    wellformed += 1
                for i in range(min(4, len(d))):
                    hs[i][d[i]] += 1
            hist[name] = hs
            gen = sum(r["gen_n"] or 0 for r in reqs)
            acc = sum(r.get("draft_n_accepted") or 0 for r in reqs)
            dr = sum(r.get("draft_n") or 0 for r in reqs)
            summary["digits"][name] = {"requests": len(reqs), "well_formed": wellformed, "generated": gen, "drafted": dr, "accepted": acc,
                                       "hist": [dict(sorted(h.items())) for h in hs]}
        base_reqs = {r["id"]: r["content"] for r in runs["plain"]["requests"]}
        for name, run in runs.items():
            s = summary["digits"][name]
            same = sum(1 for r in run.get("requests") or [] if r["content"] == base_reqs.get(r["id"]))
            tests = [chi2_homogeneity(hist["plain"][i], hist[name][i]) for i in range(4)] if name != "plain" else []
            s["same_string_as_plain_same_seed"] = same
            s["chi2_vs_plain"] = tests
            print(f"  {name:11s} n={s['requests']} well-formed {s['well_formed']} | same string as plain for the same seed {same} | "
                  f"drafted {s['drafted']}, accepted {s['accepted']} | " +
                  " ".join(f"pos{i + 1}: chi2 {t['chi2']} (dof {t['dof']}, p {t['p']})" for i, t in enumerate(tests) if t))
            print("              " + " | ".join("pos%d %s" % (i + 1, " ".join(f"{k}:{v}" for k, v in sorted(hist[name][i].items()))) for i in range(4)))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
