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

Authored by Anthropic's Claude Opus 4.7 via the Claude Code CLI.
See README.md for full attribution. Maintainer: Tommie Juzek <tjuzek@fsu.edu>.
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


def render_sparse_reward_note(baseline: dict | None, post: dict | None,
                              metrics_rows: list[dict],
                              num_gen_override: int | None = None) -> str:
    """Callout explaining why a low-baseline GRPO run shows modest gains.

    At a low baseline pass rate p, the probability a group of G completions
    is all-zero is (1 - p)^G. All-zero groups have zero reward variance, so
    GRPO's advantage is zero, so the gradient contribution is zero — the
    model learns from only the minority of groups that contain at least one
    passing sample. This is the load-bearing caveat for interpreting the
    number, and it's worth showing the audience directly.
    """
    if not baseline:
        return ""
    b_acc = baseline.get("accuracy")
    if b_acc is None or b_acc <= 0:
        return ""
    p = b_acc / 100.0

    num_gen = num_gen_override
    if num_gen is None:
        cfg_path = ROOT / "output" / "training_config.json"
        if cfg_path.exists():
            try:
                num_gen = json.loads(cfg_path.read_text()).get("num_generations")
            except json.JSONDecodeError:
                pass
    if num_gen is None:
        num_gen = 4  # sane fallback

    all_zero = (1 - p) ** num_gen * 100
    all_zero_8 = (1 - p) ** 8 * 100
    all_zero_16 = (1 - p) ** 16 * 100

    return f"""
<h2>Why the absolute gain is modest: sparse rewards at low baseline</h2>
<div class="callout">
  <p>
    The baseline model solves only <b style="color:{ACCENT}">{b_acc:.1f}%</b> of
    problems. For GRPO to learn, each group of <i>G</i> completions needs
    <b>at least one passing sample</b> — otherwise every sample gets reward 0,
    the within-group advantage is 0, and the gradient contribution is 0.
  </p>
  <p>
    With <b>G = {num_gen}</b> generations per prompt and baseline pass rate
    <i>p</i> = {p:.3f}, the fraction of groups that are entirely zero-reward is
    <span class="math">(1 − p)<sup>G</sup> = (1 − {p:.3f})<sup>{num_gen}</sup>
    ≈ <b style="color:{DANGER}">{all_zero:.1f}%</b></span>.
  </p>
  <p>
    In other words: <b>~{all_zero:.0f}% of optimizer steps carry no learning
    signal at all</b>. Only the remaining {100 - all_zero:.0f}% contribute
    gradient. The run is doing real RL, but on a fraction of the compute it
    looks like it's doing.
  </p>
  <p style="margin-bottom: 0;">
    Counter-moves: <b>more generations per prompt</b>
    (G = 8 → {all_zero_8:.0f}% zero-reward · G = 16 → {all_zero_16:.0f}%),
    <b>filter training set</b> to problems the base model solves ≥ sometimes
    (cheapest, biggest lift), or <b>warm-start</b> with SFT to raise the
    baseline first. Tulu 3 does all three.
  </p>
</div>"""


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


def render_sibling_note(sibling_label: str | None,
                        sibling_href: str | None) -> str:
    if not sibling_label or not sibling_href:
        return ""
    return f"""
<div class="sibling-note">
  <b>Companion report:</b>
  <a href="{escape(sibling_href)}" style="color:{ACCENT}">{escape(sibling_label)}</a>
  — same baseline, different GRPO configuration. Read both to see why low-baseline
  RL sits close to the noise floor.
</div>"""


def render_html(metrics_rows: list[dict], baseline: dict | None,
                post: dict | None, model_name: str,
                label: str | None = None,
                num_gen_override: int | None = None,
                sibling_label: str | None = None,
                sibling_href: str | None = None) -> str:
    stat_cards = render_stat_cards(baseline, post, metrics_rows)
    sparse_note = render_sparse_reward_note(
        baseline, post, metrics_rows, num_gen_override=num_gen_override
    )
    sibling_note = render_sibling_note(sibling_label, sibling_href)
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
  .callout {{
    background: #141418;
    border: 1px solid #27272a;
    border-left: 3px solid {WARN};
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0 1rem 0;
    font-size: 0.9rem;
    line-height: 1.55;
  }}
  .callout p {{ margin: 0.35rem 0; color: #d4d4d8; }}
  .callout b {{ color: #e4e4e7; }}
  .math {{ background: #09090b; padding: 0.15rem 0.4rem;
           border-radius: 4px; font-size: 0.92rem; }}
  .sibling-note {{
    background: #141418;
    border: 1px solid #27272a;
    border-left: 3px solid {ACCENT};
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0 0 1.5rem 0;
    font-size: 0.85rem;
    color: #d4d4d8;
  }}
  pre {{ margin: 0.5rem 0; white-space: pre-wrap; color: {MUTED}; }}
  .label {{ color: #71717a; font-size: 0.75rem; text-transform: uppercase;
            letter-spacing: 0.05em; }}
  footer {{ color: #52525b; font-size: 0.75rem; margin-top: 3rem;
            padding-top: 1rem; border-top: 1px solid #27272a; }}
</style>
</head>
<body>

<h1>RLVR — GRPO Training Report{f" · {escape(label)}" if label else ""}</h1>
<div class="subtitle">Model: {escape(model_name)} · Rendered {ts}</div>

{sibling_note}

{body_warning}

<div class="stats">
{stat_cards}
</div>

{sparse_note}

<script>
const PLOT = {PLOT_LAYOUT_JS};
</script>

{passat1}
{reward}
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
    parser.add_argument("--label", type=str, default=None,
                        help="Run label shown in the report title (e.g. 'Run 1 · G=4')")
    parser.add_argument("--num-generations", type=int, default=None,
                        help="Override G for the sparse-reward callout")
    parser.add_argument("--sibling-label", type=str, default=None,
                        help="Label for the companion-report link")
    parser.add_argument("--sibling-href", type=str, default=None,
                        help="Relative href for the companion-report link")
    args = parser.parse_args()

    metrics_rows = load_jsonl(args.metrics)
    baseline = load_json(args.baseline)
    post = load_json(args.post)

    model_name = (baseline or post or {}).get("model", "unknown")

    html = render_html(
        metrics_rows, baseline, post, model_name,
        label=args.label,
        num_gen_override=args.num_generations,
        sibling_label=args.sibling_label,
        sibling_href=args.sibling_href,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)

    print(f"Report written to: {args.out}")
    print(f"  Metrics rows:  {len(metrics_rows)}")
    print(f"  Baseline:      {'yes' if baseline else 'no'}")
    print(f"  Post-RLVR:     {'yes' if post else 'no'}")


if __name__ == "__main__":
    main()
