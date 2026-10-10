#!/usr/bin/env python3
"""
centroid_law.py -- collapse the P3 grid onto a single predictor.

P3 varies two knobs independently: block size B (8..128) and sub-centroids per
block m (1..8). The raw grid looks like two interacting effects. It is not:
once you divide them, B/m = TOKENS PER CENTROID is the only variable that
matters. This script regroups the grid by that ratio and shows how tight the
collapse is.

Usage: python benchmarks/centroid_law.py [path/to/real_kv_audit_og_v2_*.json]
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def main(path: str | None = None):
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        cands = sorted(glob.glob(os.path.join(here, "..", "docs",
                                              "real_kv_audit_og_v2_*.json")))
        if not cands:
            sys.exit("no v2 result json found")
        path = cands[-1]
    data = json.load(open(path))
    print(f"source: {os.path.basename(path)}\n")

    for model, v in data.items():
        p3 = v.get("P3_blocksize_curve")
        if not p3 or "per_block" not in p3:
            print(f"{model}: no P3 data\n")
            continue
        print(f"=== {model}  (layer {p3['layer']}) ===")
        for frac in p3["fracs"]:
            groups: dict[int, list[tuple[str, float]]] = defaultdict(list)
            for bkey, entry in p3["per_block"].items():
                B = int(bkey.replace("block", ""))
                for k, agg in entry.items():
                    if "|" not in k:
                        continue
                    mk, fk = k.split("|")
                    if abs(float(fk) - float(frac)) > 1e-12:
                        continue
                    m = int(mk[1:])
                    tpc = B // m                    # tokens per centroid
                    groups[tpc].append((f"B{B}/m{m}", agg["mean"]))
            if not groups:
                continue
            print(f"\n  selection fraction {frac}:")
            print(f"    {'tok/centroid':>13} | {'mean':>6} {'spread':>7} | cells")
            for tpc in sorted(groups):
                vals = [x[1] for x in groups[tpc]]
                names = ", ".join(n for n, _ in groups[tpc])
                spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
                print(f"    {tpc:>13} | {np.mean(vals):6.3f} {spread:7.3f} | {names}")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
