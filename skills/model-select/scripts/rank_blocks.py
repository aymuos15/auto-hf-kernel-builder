from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import model_slug  # noqa: E402

CONTAINERS = {"ModuleList", "Sequential", "ModuleDict", "ParameterList", "Identity", "Dropout"}
WRAPPER_PCT = 0.95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    slug = model_slug(cfg["model"]["id"])
    prof = json.loads(Path(f"targets/{slug}/profile.json").read_text())
    by_cls, root = prof["modules_by_class"], prof["root_class"]
    denom = max((v["incl_ms"] for v in by_cls.values()), default=1.0) or 1.0
    ranked = []
    for cls, v in by_cls.items():
        pct = v["incl_ms"] / denom
        reasons = ([("root", cls == root), ("container", cls in CONTAINERS),
                    ("primitive_op", v.get("primitive_op")),
                    ("whole-model_wrapper", pct >= WRAPPER_PCT)])
        reasons = [r for r, hit in reasons if hit]
        eligible = not reasons
        # fraction-of-runtime dominates; reuse is a sub-linear tiebreak.
        score = pct * (1.0 + 0.05 * v["instances"]) if eligible else 0.0
        ranked.append({"block": cls, "example_path": v["example"],
                       "pct_runtime": round(pct, 4), "instances": v["instances"],
                       "calls": v["calls"], "score": round(score, 4),
                       "eligible": eligible, "excluded_because": reasons or None})
    ranked.sort(key=lambda r: r["score"], reverse=True)
    winner = next((r for r in ranked if r["eligible"]), None)
    Path(f"targets/{slug}/selection.json").write_text(json.dumps({
        "model": cfg["model"]["id"], "winner_class": winner["block"],
        "winner_example_path": winner["example_path"], "ranking": ranked,
        "attribution": "REAL module-tree (forward-hook inclusive CUDA time per class)"}, indent=2))
    print(f"winner = {winner['block']} ({winner['pct_runtime']:.1%} runtime, "
          f"{winner['instances']} instances)")


if __name__ == "__main__":
    main()
