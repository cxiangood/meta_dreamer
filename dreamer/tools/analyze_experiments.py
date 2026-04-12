"""
Aggregate metrics from multiple experiment runs.

Supported inputs per run directory:
- scores.jsonl (Dreamer logger)
- metrics.jsonl (Dreamer logger)
- metadrive_eval_rewards.csv (episode-level reward log)

Outputs:
- run_summary.csv
- group_summary.csv
"""

import argparse
import csv
import json
import math
import pathlib
import re
from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple


def _to_float(x):
    if isinstance(x, (int, float)):
        v = float(x)
        if math.isfinite(v):
            return v
    return None


def flatten_numeric(d: dict, prefix: str = "") -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_numeric(v, key))
        else:
            fv = _to_float(v)
            if fv is not None:
                out[key] = fv
    return out


def read_jsonl(path: pathlib.Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_scores(path: pathlib.Path) -> Dict[str, float]:
    rows = read_jsonl(path)
    if not rows:
        return {}

    series = []
    steps = []
    for r in rows:
        flat = flatten_numeric(r)
        score = None
        for k in ("episode/score", "score", "ys"):
            if k in flat:
                score = flat[k]
                break
        step = None
        for k in ("step", "xs", "global_step"):
            if k in flat:
                step = flat[k]
                break
        if score is not None:
            series.append(score)
            if step is not None:
                steps.append(step)

    if not series:
        return {}

    tail = series[-50:] if len(series) > 50 else series
    out = {
        "episodes": float(len(series)),
        "score_final": float(series[-1]),
        "score_best": float(max(series)),
        "score_mean": float(mean(series)),
        "score_tail50_mean": float(mean(tail)),
    }
    if steps:
        out["step_final"] = float(steps[-1])
    return out


def parse_metrics(path: pathlib.Path, extra_keys: Iterable[str]) -> Dict[str, float]:
    rows = read_jsonl(path)
    if not rows:
        return {}

    latest = {}
    all_numeric = defaultdict(list)
    for r in rows:
        flat = flatten_numeric(r)
        for k, v in flat.items():
            latest[k] = v
            all_numeric[k].append(v)

    out = {}
    default_candidates = [
        "eval/score",
        "eval/length",
        "train/loss",
        "loss",
        "fps/policy",
        "usage/gpu_mem_gb",
        "risk_score",
        "episode/score",
    ]
    for k in default_candidates:
        if k in latest:
            out[f"metric_last::{k}"] = latest[k]
    for k in extra_keys:
        if k in latest:
            out[f"metric_last::{k}"] = latest[k]
            vals = all_numeric[k]
            if vals:
                out[f"metric_mean::{k}"] = float(mean(vals))
    return out


def parse_eval_rewards_csv(path: pathlib.Path) -> Dict[str, float]:
    rewards = []
    lengths = []
    reasons = Counter()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rewards.append(float(row.get("total_reward", 0.0)))
            except Exception:
                pass
            try:
                lengths.append(float(row.get("length", 0.0)))
            except Exception:
                pass
            reason = str(row.get("reason", "")).strip()
            if reason:
                reasons[reason] += 1
    if not rewards and not reasons:
        return {}
    n = max(1, sum(reasons.values()) if reasons else len(rewards))
    out = {
        "csv_episodes": float(len(rewards)),
        "csv_reward_mean": float(mean(rewards)) if rewards else 0.0,
        "csv_reward_best": float(max(rewards)) if rewards else 0.0,
        "csv_length_mean": float(mean(lengths)) if lengths else 0.0,
    }
    for k, v in reasons.items():
        out[f"reason_count::{k}"] = float(v)
        out[f"reason_rate::{k}"] = float(v) / float(n)
    return out


def infer_seed(name: str) -> Optional[int]:
    m = re.search(r"(?:^|[_-])s(\d+)(?:$|[_-])", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|[_-])seed(\d+)(?:$|[_-])", name)
    if m:
        return int(m.group(1))
    return None


def infer_group(name: str) -> str:
    g = re.sub(r"(?:[_-])s\d+(?:$|[_-].*)?", "", name)
    g = re.sub(r"(?:[_-])seed\d+(?:$|[_-].*)?", "", g)
    return g if g else name


def find_runs(roots: List[str]) -> List[pathlib.Path]:
    runs = set()
    for root in roots:
        rp = pathlib.Path(root)
        if not rp.exists():
            continue
        for p in rp.rglob("scores.jsonl"):
            runs.add(p.parent)
        for p in rp.rglob("metrics.jsonl"):
            runs.add(p.parent)
        for p in rp.rglob("metadrive_eval_rewards.csv"):
            runs.add(p.parent)
    return sorted(runs)


def aggregate_by_group(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["group"]].append(r)

    out = []
    for g, items in sorted(grouped.items()):
        row = {"group": g, "runs": len(items)}
        keys = set()
        for it in items:
            keys.update(k for k, v in it.items() if isinstance(v, (int, float)))
        for k in sorted(keys):
            vals = [float(it[k]) for it in items if isinstance(it.get(k), (int, float))]
            if not vals:
                continue
            row[f"{k}__mean"] = float(mean(vals))
            row[f"{k}__std"] = float(pstdev(vals)) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def write_csv(path: pathlib.Path, rows: List[Dict[str, object]]):
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: pathlib.Path, payload: Dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_markdown(path: pathlib.Path, runs: List[Dict[str, object]], groups: List[Dict[str, object]], topk: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Experiment Analysis\n\n")
        f.write(f"- Runs: {len(runs)}\n")
        f.write(f"- Groups: {len(groups)}\n\n")

        ranked = []
        for r in runs:
            score = None
            for k in ("score_tail50_mean", "score_final", "metric_last::eval/score", "csv_reward_mean"):
                if isinstance(r.get(k), (int, float)):
                    score = float(r[k])
                    break
            if score is not None:
                ranked.append((score, r))
        ranked.sort(key=lambda x: x[0], reverse=True)

        f.write("## Top Runs\n\n")
        f.write("| Rank | Run | Group | Key Score |\n")
        f.write("|---:|---|---|---:|\n")
        for i, (score, r) in enumerate(ranked[:topk], 1):
            f.write(f"| {i} | {r.get('run_name','')} | {r.get('group','')} | {score:.6g} |\n")
        f.write("\n")

        f.write("## Group Summary\n\n")
        if not groups:
            f.write("_No groups found._\n")
            return
        headers = ["group", "runs", "score_tail50_mean__mean", "score_final__mean", "csv_reward_mean__mean"]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for g in groups:
            row = [_fmt(g.get(h, "")) for h in headers]
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n")


def write_html(path: pathlib.Path, runs: List[Dict[str, object]], groups: List[Dict[str, object]], topk: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"runs": runs, "groups": groups, "topk": topk}, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Experiment Analysis</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      margin: 24px;
      color: #111;
      background: #f7f7f8;
    }}
    h1, h2 {{ margin: 8px 0 12px 0; }}
    .card {{
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 14px 16px;
      margin-bottom: 14px;
    }}
    .meta {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      font-size: 14px;
      color: #374151;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #eceff3;
      padding: 8px 6px;
      text-align: left;
      vertical-align: top;
      word-break: break-word;
    }}
    th {{
      background: #f9fafb;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .small {{ font-size: 12px; color: #6b7280; }}
  </style>
</head>
<body>
  <h1>Experiment Analysis</h1>
  <div class="card">
    <div class="meta" id="meta"></div>
  </div>
  <div class="card">
    <h2>Top Runs</h2>
    <table id="top-table"></table>
  </div>
  <div class="card">
    <h2>Group Summary</h2>
    <table id="group-table"></table>
  </div>
  <div class="card">
    <h2>Run Summary</h2>
    <div class="small">Tip: open CSV/JSON for full columns.</div>
    <table id="run-table"></table>
  </div>

  <script>
    const data = {payload};
    const runs = data.runs || [];
    const groups = data.groups || [];
    const topk = data.topk || 10;

    function firstScore(r) {{
      for (const k of ["score_tail50_mean", "score_final", "metric_last::eval/score", "csv_reward_mean"]) {{
        if (typeof r[k] === "number") return r[k];
      }}
      return null;
    }}

    function renderTable(el, columns, rows) {{
      let html = "<thead><tr>" + columns.map(c => `<th>${{c}}</th>`).join("") + "</tr></thead><tbody>";
      for (const row of rows) {{
        html += "<tr>" + columns.map(c => {{
          const v = row[c];
          if (typeof v === "number") return `<td>${{Number.isFinite(v) ? v.toFixed(6) : ""}}</td>`;
          return `<td>${{v === undefined ? "" : String(v)}}</td>`;
        }}).join("") + "</tr>";
      }}
      html += "</tbody>";
      el.innerHTML = html;
    }}

    document.getElementById("meta").innerHTML =
      `<div>Runs: <b>${{runs.length}}</b></div><div>Groups: <b>${{groups.length}}</b></div>`;

    const ranked = runs
      .map(r => ({{...r, __score: firstScore(r)}}))
      .filter(r => r.__score !== null)
      .sort((a, b) => b.__score - a.__score)
      .slice(0, topk)
      .map((r, i) => ({{rank: i + 1, run_name: r.run_name, group: r.group, key_score: r.__score}}));
    renderTable(document.getElementById("top-table"), ["rank", "run_name", "group", "key_score"], ranked);

    const gcols = ["group", "runs", "score_tail50_mean__mean", "score_final__mean", "csv_reward_mean__mean"];
    renderTable(document.getElementById("group-table"), gcols, groups);

    const rcols = ["run_name", "group", "seed", "score_tail50_mean", "score_final", "csv_reward_mean", "run_dir"];
    renderTable(document.getElementById("run-table"), rcols, runs);
  </script>
</body>
</html>
"""
    with path.open("w", encoding="utf-8") as f:
        f.write(html)


def summarize(rows: List[Dict[str, object]], topk: int):
    if not rows:
        print("[Summary] no runs found.")
        return
    print(f"[Summary] runs={len(rows)} groups={len(set(r['group'] for r in rows))}")
    ranked = []
    for r in rows:
        score = None
        for k in ("score_tail50_mean", "score_final", "metric_last::eval/score", "csv_reward_mean"):
            if isinstance(r.get(k), (int, float)):
                score = float(r[k])
                break
        if score is not None:
            ranked.append((score, r))
    ranked.sort(key=lambda x: x[0], reverse=True)
    for i, (score, r) in enumerate(ranked[:topk], 1):
        print(f"  Top{i}: {r['run_name']}  score={score:.4f}  group={r['group']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=str, nargs="+", default=["dreamer", "logs", "result"])
    parser.add_argument("--extra_keys", type=str, nargs="*", default=[])
    parser.add_argument("--outdir", type=str, default="result/analysis")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    runs = find_runs(args.roots)
    rows = []
    for run in runs:
        row = {
            "run_dir": str(run.resolve()),
            "run_name": run.name,
            "group": infer_group(run.name),
        }
        seed = infer_seed(run.name)
        if seed is not None:
            row["seed"] = seed

        score_file = run / "scores.jsonl"
        metric_file = run / "metrics.jsonl"
        csv_file = run / "metadrive_eval_rewards.csv"

        if score_file.exists():
            row.update(parse_scores(score_file))
        if metric_file.exists():
            row.update(parse_metrics(metric_file, args.extra_keys))
        if csv_file.exists():
            row.update(parse_eval_rewards_csv(csv_file))
        rows.append(row)

    outdir = pathlib.Path(args.outdir)
    run_csv = outdir / "run_summary.csv"
    group_csv = outdir / "group_summary.csv"
    out_json = outdir / "summary.json"
    out_md = outdir / "summary.md"
    out_html = outdir / "summary.html"
    group_rows = aggregate_by_group(rows)
    write_csv(run_csv, rows)
    write_csv(group_csv, group_rows)
    write_json(
        out_json,
        {
            "num_runs": len(rows),
            "num_groups": len(group_rows),
            "runs": rows,
            "groups": group_rows,
        },
    )
    write_markdown(out_md, rows, group_rows, args.topk)
    write_html(out_html, rows, group_rows, args.topk)
    summarize(rows, args.topk)
    print(f"[Saved] {run_csv}")
    print(f"[Saved] {group_csv}")
    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_md}")
    print(f"[Saved] {out_html}")


if __name__ == "__main__":
    main()
