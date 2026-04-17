"""
Post-run report generator for the full GRPO training pipeline.

Reads:
  - output/metrics.jsonl       (streamed by MetricsCallback during training)
  - results/baseline.json      (pre-training benchmark)
  - results/post_rlvr.json     (post-training benchmark)

Writes:
  - results/grpo_report.html   (standalone, dark-theme Plotly report)

All inputs are optional — the script degrades gracefully so a partial run
still produces something usable for the presentation.

Usage:
    python3 make_report.py
    python3 make_report.py --metrics output/metrics.jsonl \
        --baseline results/baseline.json --post results/post_rlvr.json \
        --out results/grpo_report.html

AI-assisted code — generated with Claude Code (Anthropic).
See README.md for full attribution. Maintainer: Tom Juzek <tjuzek@fsu.edu>.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from html import escape
from pathlib import Path

ROOT = Path(__file__).parent

ACCENT = "#4d8eff"
SUCCESS = "#22c55e"
DANGER = "#ef4444"
WARN = "#f59e0b"
MUTED = "#a1a1aa"

PLOT_LAYOUT_JS = """{
  paper_bgcolor: '#09090b',
  plot_bgcolor: '#141418',
  font: { color: '#a1a1aa', family: 'JetBrains Mono, monospace', size: 12 },
  xaxis: { gridcolor: '#27272a' },
  yaxis: { gridcolor: '#27272a' },
  margin: { l: 60, r: 20, t: 20, b: 50 },
  showlegend: true,
  legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: '#a1a1aa' } },
}"""


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def series(rows: list[dict], key: str) -> tuple[list, list]:
    """Extract (step, value) pairs for rows that have `key`."""
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None and "step" in r:
            xs.append(r["step"])
            ys.append(r[key])
    return xs, ys


def rolling_mean(ys: list[float], window: int = 5) -> list[float]:
    if len(ys) < window:
        return ys[:]
    out = []
    for i in range(len(ys)):
        lo = max(0, i - window + 1)
        window_slice = ys[lo:i + 1]
        out.append(sum(window_slice) / len(window_slice))
    return out


def fmt_pct(x: float | None) -> str:
    return f"{x:.1f}%" if x is not None else "—"


def fmt_delta(x: float | None) -> str:
    if x is None:
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.1f}%"


def fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 7200:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def compute_flips(baseline: dict | None, post: dict | None) -> dict:
    """Return per-problem flip counts."""
    if not baseline or not post:
        return {}
    b_map = {r["task_id"]: r["passed"] for r in baseline.get("results", [])}
    p_map = {r["task_id"]: r["passed"] for r in post.get("results", [])}
    common = set(b_map) & set(p_map)
    fail_to_pass = sum(1 for t in common if not b_map[t] and p_map[t])
    pass_to_fail = sum(1 for t in common if b_map[t] and not p_map[t])
    still_pass = sum(1 for t in common if b_map[t] and p_map[t])
    still_fail = sum(1 for t in common if not b_map[t] and not p_map[t])
    return {
        "fail_to_pass": fail_to_pass,
        "pass_to_fail": pass_to_fail,
        "still_pass": still_pass,
        "still_fail": still_fail,
        "total": len(common),
    }


def render_stat_cards(baseline: dict | None, post: dict | None,
                      metrics_rows: list[dict]) -> str:
    b_acc = baseline["accuracy"] if baseline else None
    p_acc = post["accuracy"] if post else None
    delta = (p_acc - b_acc) if (b_acc is not None and p_acc is not None) else None

    # Training time: sum of step wall-clock if available, else fall back
    training_secs = None
    if metrics_rows:
        steps_with_runtime = [r for r in metrics_rows if "train_runtime" in r]
        if steps_with_runtime:
            training_secs = steps_with_runtime[-1]["train_runtime"]
        else:
            # Best-effort: last step's epoch * some marker isn't available,
            # so fall back to total wallclock between benchmark timestamps.
            pass

    if training_secs is None and baseline and post:
        try:
            b_ts = datetime.fromisoformat(baseline["timestamp"])
            p_ts = datetime.fromisoformat(post["timestamp"])
            training_secs = (p_ts - b_ts).total_seconds()
        except (KeyError, ValueError):
            pass

    n_steps = max((r.get("step", 0) for r in metrics_rows), default=0)

    cards = [
        ("Baseline pass@1", fmt_pct(b_acc), ACCENT),
        ("Post-RLVR pass@1", fmt_pct(p_acc), SUCCESS),
        ("Absolute delta", fmt_delta(delta),
         SUCCESS if (delta or 0) >= 0 else DANGER),
        ("Training steps", f"{n_steps}", ACCENT),
        ("Wall time", fmt_time(training_secs), MUTED),
    ]
    return "\n".join(
        f'<div class="stat"><div class="stat-value" style="color:{c}">{v}</div>'
        f'<div class="stat-label">{l}</div></div>'
        for l, v, c in cards
    )


def render_reward_chart(rows: list[dict]) -> str:
    xs, ys = series(rows, "reward")
    if len(xs) < 3:
        return ""
    ys_smooth = rolling_mean(ys, window=5)
    raw = (f"{{x: {json.dumps(xs)}, y: {json.dumps(ys)}, "
           f"type: 'scatter', mode: 'lines+markers', name: 'Per-log reward', "
           f"line: {{color: '{ACCENT}', width: 2}}, "
           f"marker: {{size: 5, color: '{ACCENT}'}}, opacity: 0.55}}")
    smooth = (f"{{x: {json.dumps(xs)}, y: {json.dumps(ys_smooth)}, "
              f"type: 'scatter', mode: 'lines', name: 'Rolling mean (5)', "
              f"line: {{color: '{SUCCESS}', width: 3}}}}")
    return f"""
<h2>Average reward (fraction of generations passing tests)</h2>
<div id="reward-chart" class="chart"></div>
<script>
Plotly.newPlot('reward-chart', [{raw}, {smooth}],
  {{...PLOT, xaxis: {{...PLOT.xaxis, title: 'Training step'}},
    yaxis: {{...PLOT.yaxis, title: 'Reward', range: [0, 1]}}}},
  {{displayModeBar: false, responsive: true}});
</script>"""


def render_passat1_chart(baseline: dict | None, post: dict | None) -> str:
    if not baseline or not post:
        return ""
    b = baseline["accuracy"]
    p = post["accuracy"]
    data = (f"[{{x: ['Baseline', 'Post-RLVR'], y: [{b:.2f}, {p:.2f}], "
            f"type: 'bar', marker: {{color: ['{ACCENT}', '{SUCCESS}']}}, "
            f"text: ['{b:.1f}%', '{p:.1f}%'], textposition: 'outside', "
            f"textfont: {{color: '#e4e4e7', size: 16}}}}]")
    return f"""
<h2>Pass@1 on held-out test set: before vs. after GRPO</h2>
<div id="passat1-chart" class="chart"></div>
<script>
Plotly.newPlot('passat1-chart', {data},
  {{...PLOT, showlegend: false,
    yaxis: {{...PLOT.yaxis, title: 'pass@1 (%)', range: [0, 100]}}}},
  {{displayModeBar: false, responsive: true}});
</script>"""


def render_kl_chart(rows: list[dict]) -> str:
    xs, ys = series(rows, "kl")
    if len(xs) < 3:
        return ""
    trace = (f"{{x: {json.dumps(xs)}, y: {json.dumps(ys)}, "
             f"type: 'scatter', mode: 'lines+markers', name: 'KL', "
             f"line: {{color: '{WARN}', width: 2}}, marker: {{size: 5}}}}")
    return f"""
<h2>KL divergence from reference policy</h2>
<p style="color:{MUTED};margin-top:-0.5rem;font-size:0.85rem">
  Low = the trained policy stays close to the reference model. If this
  explodes, the model has drifted and outputs will degrade.
</p>
<div id="kl-chart" class="chart"></div>
<script>
Plotly.newPlot('kl-chart', [{trace}],
  {{...PLOT, showlegend: false,
    xaxis: {{...PLOT.xaxis, title: 'Training step'}},
    yaxis: {{...PLOT.yaxis, title: 'KL'}}}},
  {{displayModeBar: false, responsive: true}});
</script>"""


def render_loss_chart(rows: list[dict]) -> str:
    xs_loss, ys_loss = series(rows, "loss")
    xs_gn, ys_gn = series(rows, "grad_norm")
    if len(xs_loss) < 3 and len(xs_gn) < 3:
        return ""
    traces = []
    if len(xs_loss) >= 3:
        traces.append(
            f"{{x: {json.dumps(xs_loss)}, y: {json.dumps(ys_loss)}, "
            f"type: 'scatter', mode: 'lines', name: 'Loss', "
            f"line: {{color: '{ACCENT}', width: 2}}, yaxis: 'y'}}"
        )
    if len(xs_gn) >= 3:
        traces.append(
            f"{{x: {json.dumps(xs_gn)}, y: {json.dumps(ys_gn)}, "
            f"type: 'scatter', mode: 'lines', name: 'Grad norm', "
            f"line: {{color: '{DANGER}', width: 2, dash: 'dot'}}, "
            f"yaxis: 'y2'}}"
        )
    if not traces:
        return ""
    return f"""
<h2>Policy loss and gradient norm</h2>
<div id="loss-chart" class="chart"></div>
<script>
Plotly.newPlot('loss-chart', [{", ".join(traces)}],
  {{...PLOT,
    xaxis: {{...PLOT.xaxis, title: 'Training step'}},
    yaxis: {{...PLOT.yaxis, title: 'Loss', side: 'left'}},
    yaxis2: {{gridcolor: '#27272a', title: 'Grad norm',
             overlaying: 'y', side: 'right', color: '{DANGER}'}}}},
  {{displayModeBar: false, responsive: true}});
</script>"""


def render_flip_chart(flips: dict) -> str:
    if not flips:
        return ""
    cats = ["Fixed (fail→pass)", "Regressed (pass→fail)",
            "Still passing", "Still failing"]
    vals = [flips["fail_to_pass"], flips["pass_to_fail"],
            flips["still_pass"], flips["still_fail"]]
    colors = [SUCCESS, DANGER, ACCENT, MUTED]
    trace = (f"{{x: {json.dumps(cats)}, y: {json.dumps(vals)}, "
             f"type: 'bar', marker: {{color: {json.dumps(colors)}}}, "
             f"text: {json.dumps(vals)}, textposition: 'outside', "
             f"textfont: {{color: '#e4e4e7', size: 14}}}}")
    return f"""
<h2>Per-problem flip analysis ({flips['total']} common problems)</h2>
<p style="color:{MUTED};margin-top:-0.5rem;font-size:0.85rem">
  What happened at the problem level, not just in aggregate. Big green bar
  is the fix rate — how many previously-failing problems the RLVR-trained
  model now solves.
</p>
<div id="flip-chart" class="chart"></div>
<script>
Plotly.newPlot('flip-chart', [{trace}],
  {{...PLOT, showlegend: false,
    yaxis: {{...PLOT.yaxis, title: 'Problems'}}}},
  {{displayModeBar: false, responsive: true}});
</script>"""


def render_flip_examples(baseline: dict | None, post: dict | None,
                         n: int = 2) -> str:
    """Show a few task diffs: one fixed, one stayed-broken."""
    if not baseline or not post:
        return ""
    b_map = {r["task_id"]: r for r in baseline.get("results", [])}
    p_map = {r["task_id"]: r for r in post.get("results", [])}
    common = [t for t in b_map if t in p_map]
    fixed = [t for t in common if not b_map[t]["passed"] and p_map[t]["passed"]]

    blocks = []
    for task_id in fixed[:n]:
        b_code = escape((b_map[task_id].get("generated_code") or "").strip())
        p_code = escape((p_map[task_id].get("generated_code") or "").strip())
        blocks.append(f"""
<div class="example fail">
  <div class="label">Task {task_id} — baseline FAIL</div>
  <pre>{b_code[:800]}</pre>
</div>
<div class="example pass">
  <div class="label">Task {task_id} — post-RLVR PASS</div>
  <pre>{p_code[:800]}</pre>
</div>
<br>""")
    if not blocks:
        return ""
    return "<h2>Example fixes — problems the model learned to solve</h2>" + "".join(blocks)


def render_html(metrics_rows: list[dict], baseline: dict | None,
                post: dict | None, model_name: str) -> str:
    stat_cards = render_stat_cards(baseline, post, metrics_rows)
    reward = render_reward_chart(metrics_rows)
    passat1 = render_passat1_chart(baseline, post)
    kl = render_kl_chart(metrics_rows)
    loss = render_loss_chart(metrics_rows)
    flips = compute_flips(baseline, post)
    flip_chart = render_flip_chart(flips)
    flip_examples = render_flip_examples(baseline, post)

    available_count = sum(1 for s in [reward, passat1, kl, loss, flip_chart] if s)
    if available_count == 0:
        body_warning = (
            f'<p style="color:{DANGER}">No data found. '
            f'Expected output/metrics.jsonl, results/baseline.json, '
            f'and/or results/post_rlvr.json.</p>'
        )
    else:
        body_warning = ""

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html>
<head>
<title>RLVR GRPO Training Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{
    font-family: 'JetBrains Mono', 'SF Mono', 'Consolas', monospace;
    background: #09090b;
    color: #e4e4e7;
    margin: 0;
    padding: 2rem;
    max-width: 900px;
  }}
  h1 {{ color: {ACCENT}; font-size: 1.8rem; margin-bottom: 0.25rem; }}
  h2 {{ color: {MUTED}; font-size: 1.1rem; font-weight: normal;
        margin-top: 2.5rem; margin-bottom: 0.5rem; }}
  .subtitle {{ color: #71717a; font-size: 0.9rem; margin-bottom: 2rem; }}
  .chart {{ width: 100%; max-width: 820px; height: 360px; margin: 0.5rem 0; }}
  .stats {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
  }}
  .stat {{
    background: #141418;
    border: 1px solid #27272a;
    border-radius: 8px;
    padding: 1rem;
  }}
  .stat-value {{ font-size: 1.9rem; color: {ACCENT}; }}
  .stat-label {{ font-size: 0.8rem; color: #71717a;
                 text-transform: uppercase; letter-spacing: 0.05em; }}
  .example {{
    background: #141418;
    border: 1px solid #27272a;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
  }}
  .pass {{ border-left: 3px solid {SUCCESS}; }}
  .fail {{ border-left: 3px solid {DANGER}; }}
  pre {{ margin: 0.5rem 0; white-space: pre-wrap; color: {MUTED}; }}
  .label {{ color: #71717a; font-size: 0.75rem; text-transform: uppercase;
            letter-spacing: 0.05em; }}
  footer {{ color: #52525b; font-size: 0.75rem; margin-top: 3rem;
            padding-top: 1rem; border-top: 1px solid #27272a; }}
</style>
</head>
<body>

<h1>RLVR — GRPO Training Report</h1>
<div class="subtitle">Model: {escape(model_name)} · Rendered {ts}</div>

{body_warning}

<div class="stats">
{stat_cards}
</div>

<script>
const PLOT = {PLOT_LAYOUT_JS};
</script>

{reward}
{passat1}
{kl}
{loss}
{flip_chart}
{flip_examples}

<footer>
  Generated by make_report.py from output/metrics.jsonl +
  results/baseline.json + results/post_rlvr.json
</footer>

</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Generate GRPO training report")
    parser.add_argument("--metrics", type=Path,
                        default=ROOT / "output" / "metrics.jsonl")
    parser.add_argument("--baseline", type=Path,
                        default=ROOT / "results" / "baseline.json")
    parser.add_argument("--post", type=Path,
                        default=ROOT / "results" / "post_rlvr.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "grpo_report.html")
    args = parser.parse_args()

    metrics_rows = load_jsonl(args.metrics)
    baseline = load_json(args.baseline)
    post = load_json(args.post)

    model_name = (baseline or post or {}).get("model", "unknown")

    html = render_html(metrics_rows, baseline, post, model_name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)

    print(f"Report written to: {args.out}")
    print(f"  Metrics rows:  {len(metrics_rows)}")
    print(f"  Baseline:      {'yes' if baseline else 'no'}")
    print(f"  Post-RLVR:     {'yes' if post else 'no'}")


if __name__ == "__main__":
    main()
