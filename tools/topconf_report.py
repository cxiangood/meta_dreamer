#!/usr/bin/env python3
"""
Aggregate top-conference ablation runs and compute statistics.

Expected run layout:
  dreamer/logs_metadrive/<run_dir>/
    - config.yaml
    - scores.jsonl

Run naming convention (recommended):
  <timestamp>_<variant>_s<seed>

Example:
  python tools/topconf_report.py \
    --logdir /share/home/u23516/code/meta_dreamer-main/dreamer/logs_metadrive \
    --outdir /share/home/u23516/code/meta_dreamer-main/result/topconf_report
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
from typing import Dict, List, Optional, Tuple


def _read_jsonl_scores(path: str) -> List[float]:
    scores = []
    if not os.path.isfile(path):
        return scores
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "episode/score" in obj:
                try:
                    scores.append(float(obj["episode/score"]))
                except Exception:
                    pass
    return scores


def _read_simple_yaml_value(lines: List[str], key: str) -> Optional[str]:
    prefix = f"{key}:"
    for line in lines:
        s = line.strip()
        if s.startswith(prefix):
            return s[len(prefix):].strip()
    return None


def _load_config_fields(config_path: str) -> Dict[str, Optional[float]]:
    out = {
        "seed": None,
        "expert_heads": None,
        "expert_modes": None,
        "risk_threshold": None,
        "action_threshold": None,
    }
    if not os.path.isfile(config_path):
        return out

    with open(config_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Lightweight parser for only a few scalar keys.
    mappings = {
        "seed": "seed",
        "expert_heads": "expert_heads",
        "expert_modes": "expert_modes",
        "risk_threshold": "risk_threshold",
        "action_threshold": "action_threshold",
    }

    for raw_key, dst_key in mappings.items():
        val = _read_simple_yaml_value(lines, raw_key)
        if val is None:
            continue
        try:
            out[dst_key] = float(val)
        except ValueError:
            pass

    return out


def _infer_variant(run_name: str, cfg: Dict[str, Optional[float]]) -> str:
    name = run_name.lower()
    for tag in ["baseline", "risk_only", "multitraj_only", "disagreement_only", "full"]:
        if f"_{tag}_" in name or name.endswith(f"_{tag}") or name.startswith(f"{tag}_"):
            return tag

    heads = int(cfg["expert_heads"] or 1)
    modes = int(cfg["expert_modes"] or 1)
    tokens = heads * modes
    risk_th = float(cfg["risk_threshold"] if cfg["risk_threshold"] is not None else 1.0)
    act_th = float(cfg["action_threshold"] if cfg["action_threshold"] is not None else 0.0)

    risk_on = risk_th < 0.999
    multi_on = tokens > 1
    disagree_on = act_th > 1e-8

    if risk_on and multi_on and disagree_on:
        return "full"
    if risk_on and (not multi_on) and (not disagree_on):
        return "risk_only"
    if (not risk_on) and multi_on and (not disagree_on):
        return "multitraj_only"
    if (not risk_on) and (not multi_on) and disagree_on:
        return "disagreement_only"
    return "baseline"


def _parse_seed(run_name: str, cfg: Dict[str, Optional[float]]) -> int:
    parts = run_name.split("_s")
    if len(parts) >= 2:
        tail = parts[-1]
        digits = []
        for ch in tail:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if digits:
            return int("".join(digits))
    if cfg.get("seed") is not None:
        return int(cfg["seed"])
    return -1


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _bootstrap_ci(values: List[float], alpha: float = 0.05, n_boot: int = 5000) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]

    rng = random.Random(0)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(0, n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot)
    hi_idx = min(hi_idx, n_boot - 1)
    return means[lo_idx], means[hi_idx]


def _permutation_pvalue(a: List[float], b: List[float], n_perm: int = 10000) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    rng = random.Random(0)
    observed = abs(_mean(a) - _mean(b))
    pool = a + b
    na = len(a)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        pa = pool[:na]
        pb = pool[na:]
        diff = abs(_mean(pa) - _mean(pb))
        if diff >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def collect_runs(logdir: str, tail_episodes: int) -> List[Dict[str, object]]:
    rows = []
    if not os.path.isdir(logdir):
        return rows

    for run_name in sorted(os.listdir(logdir)):
        run_path = os.path.join(logdir, run_name)
        if not os.path.isdir(run_path):
            continue

        config_path = os.path.join(run_path, "config.yaml")
        scores_path = os.path.join(run_path, "scores.jsonl")
        scores = _read_jsonl_scores(scores_path)
        if not scores:
            continue

        tail = scores[-tail_episodes:] if len(scores) >= tail_episodes else scores
        final_score = _mean(tail)

        cfg = _load_config_fields(config_path)
        variant = _infer_variant(run_name, cfg)
        seed = _parse_seed(run_name, cfg)

        rows.append({
            "run_name": run_name,
            "run_path": run_path,
            "variant": variant,
            "seed": seed,
            "episodes": len(scores),
            "final_score": final_score,
            "last_score": scores[-1],
        })

    return rows


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[float]] = {}
    for r in rows:
        grouped.setdefault(str(r["variant"]), []).append(float(r["final_score"]))

    order = ["baseline", "risk_only", "multitraj_only", "disagreement_only", "full"]
    variants = [v for v in order if v in grouped] + [v for v in sorted(grouped.keys()) if v not in order]

    baseline = grouped.get("baseline", [])
    out = []
    for v in variants:
        vals = grouped[v]
        mu = _mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        ci_lo, ci_hi = _bootstrap_ci(vals)

        delta = float("nan")
        pval = float("nan")
        if baseline and v != "baseline":
            delta = mu - _mean(baseline)
            pval = _permutation_pvalue(vals, baseline)

        out.append({
            "variant": v,
            "n_seeds": len(vals),
            "mean_final_score": mu,
            "std_final_score": sd,
            "ci95_low": ci_lo,
            "ci95_high": ci_hi,
            "delta_vs_baseline": delta,
            "pvalue_vs_baseline": pval,
        })

    return out


def write_markdown(path: str, summary_rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# TopConf Ablation Report\n\n")
        f.write("Columns: mean score over the last evaluation episodes per run, aggregated across seeds.\\\n")
        f.write("CI is bootstrap 95%. P-value is a two-sided permutation test against baseline.\n\n")
        f.write("| Variant | Seeds | Mean | Std | 95% CI | Delta vs Baseline | P-value |\n")
        f.write("| --- | ---: | ---: | ---: | --- | ---: | ---: |\n")
        for r in summary_rows:
            ci = f"[{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]"
            delta = "nan" if math.isnan(float(r["delta_vs_baseline"])) else f"{float(r['delta_vs_baseline']):.3f}"
            pval = "nan" if math.isnan(float(r["pvalue_vs_baseline"])) else f"{float(r['pvalue_vs_baseline']):.4f}"
            f.write(
                f"| {r['variant']} | {r['n_seeds']} | {float(r['mean_final_score']):.3f} | "
                f"{float(r['std_final_score']):.3f} | {ci} | {delta} | {pval} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="Root log directory")
    parser.add_argument("--outdir", required=True, help="Output directory for tables")
    parser.add_argument("--tail-episodes", type=int, default=20, help="Tail window for final score")
    args = parser.parse_args()

    rows = collect_runs(args.logdir, args.tail_episodes)
    if not rows:
        print("[WARN] No valid runs found.")
        return

    details_csv = os.path.join(args.outdir, "run_details.csv")
    write_csv(
        details_csv,
        rows,
        ["run_name", "variant", "seed", "episodes", "final_score", "last_score", "run_path"],
    )

    summary_rows = summarize(rows)
    summary_csv = os.path.join(args.outdir, "summary.csv")
    write_csv(
        summary_csv,
        summary_rows,
        [
            "variant",
            "n_seeds",
            "mean_final_score",
            "std_final_score",
            "ci95_low",
            "ci95_high",
            "delta_vs_baseline",
            "pvalue_vs_baseline",
        ],
    )

    summary_md = os.path.join(args.outdir, "summary.md")
    write_markdown(summary_md, summary_rows)

    print(f"[OK] Saved: {details_csv}")
    print(f"[OK] Saved: {summary_csv}")
    print(f"[OK] Saved: {summary_md}")


if __name__ == "__main__":
    main()
