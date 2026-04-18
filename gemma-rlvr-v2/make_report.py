"""
Post-run report generator for the Gemma RLVR experiment on GSM8K.

This is the third experiment in the rlvr-demo repo. It targets GRPO's
"reward variance sweet spot" by picking a model whose GSM8K baseline
lands in the middle band (Gemma-2-2B-IT, ~51%) and bumping G=8 and
temperature=1.0 to keep groups mixed-reward.

Reads:
  - output/metrics.jsonl        (streamed by MetricsCallback during training)
  - output/training_config.json (hyperparameters used for this run)
  - results/baseline.json       (pre-training benchmark on GSM8K test)
  - results/post_rlvr.json      (post-training benchmark on GSM8K test)

Writes:
  - results/gemma_gsm8k_report.html  (standalone, dark-theme Plotly report)

All inputs are optional — the script degrades gracefully so a partial run
still produces something usable.
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
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(20,20,24,0.45)',
  font: { color: '#c7c7d1', family: 'JetBrains Mono, ui-monospace, monospace', size: 15 },
  xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)', linecolor: 'rgba(255,255,255,0.12)', tickfont: { size: 14 } },
  yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)', linecolor: 'rgba(255,255,255,0.12)', tickfont: { size: 14 } },
  margin: { l: 78, r: 28, t: 28, b: 62 },
  showlegend: true,
  legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: '#c7c7d1', size: 14 } },
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
<h2>Average reward (fraction of generations producing the correct answer)</h2>
<div class="chart-frame"><div id="reward-chart" class="chart"></div></div>
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
            f"textfont: {{color: '#e4e4e7', size: 18}}}}]")
    return f"""
<h2>Accuracy on GSM8K test (1,319 problems): before vs. after GRPO</h2>
<div class="chart-frame"><div id="passat1-chart" class="chart"></div></div>
<script>
Plotly.newPlot('passat1-chart', {data},
  {{...PLOT, showlegend: false,
    yaxis: {{...PLOT.yaxis, title: 'pass@1 (%)', range: [0, 100]}}}},
  {{displayModeBar: false, responsive: true}});
</script>"""


def render_variance_chart(rows: list[dict]) -> str:
    """Fraction of GRPO groups with zero reward variance, over training.

    This is the single most diagnostic signal for whether GRPO has signal
    to learn from. At `frac_reward_zero_std = 1.0`, every group has all
    identical rewards (all-pass or all-fail) and the advantage term is
    zero — the step contributes no gradient. The whole recipe (model
    choice, G, temperature) is tuned to push this number down.
    """
    xs, ys = series(rows, "frac_reward_zero_std")
    if len(xs) < 3:
        return ""
    ys_pct = [y * 100 for y in ys]
    mean_pct = sum(ys_pct) / len(ys_pct) if ys_pct else 0.0
    trace = (f"{{x: {json.dumps(xs)}, y: {json.dumps(ys_pct)}, "
             f"type: 'scatter', mode: 'lines+markers', name: 'Zero-variance groups', "
             f"line: {{color: '{DANGER}', width: 2}}, marker: {{size: 5}}, "
             f"fill: 'tozeroy', fillcolor: 'rgba(248,113,113,0.08)'}}")
    return f"""
<h2>Reward-variance diagnostic · <code>frac_reward_zero_std</code></h2>
<p class="chart-hint">
  Percent of GRPO groups where all {{G}} completions got the same reward
  (all-pass or all-fail). These groups have zero advantage and contribute
  <b>no gradient</b>. Low values = healthy variance = real RL signal.
  Run average: <b style="color:{DANGER}">{mean_pct:.1f}%</b>.
</p>
<div class="chart-frame"><div id="variance-chart" class="chart"></div></div>
<script>
Plotly.newPlot('variance-chart', [{trace}],
  {{...PLOT, showlegend: false,
    xaxis: {{...PLOT.xaxis, title: 'Training step'}},
    yaxis: {{...PLOT.yaxis, title: 'Zero-variance groups (%)', range: [0, 100]}}}},
  {{displayModeBar: false, responsive: true}});
</script>"""


def render_recipe_panel(config: dict | None, baseline: dict | None,
                        post: dict | None) -> str:
    """Prominent hyperparameter summary — the knobs that define this run."""
    if not config:
        return ""
    rows = [
        ("Base model", str(config.get("model", "?"))),
        ("Generations per prompt (G)", str(config.get("num_generations", "?"))),
        ("Temperature", f"{config.get('temperature', '?')}"),
        ("Training steps", str(config.get("max_steps", "?"))),
        ("Learning rate", f"{config.get('learning_rate', '?')}"),
        ("KL coefficient", f"{config.get('kl_coeff', '?')}"),
        ("LoRA r / α", f"{config.get('lora_r', '?')} / {config.get('lora_alpha', '?')}"),
        ("Max completion tokens", str(config.get("max_new_tokens", "?"))),
    ]
    if baseline and "accuracy" in baseline:
        b = baseline["accuracy"]
        g = config.get("num_generations")
        if isinstance(b, (int, float)) and isinstance(g, int) and g > 0:
            p = b / 100.0 if b > 1 else b
            allzero = ((1 - p) ** g) * 100
            allone = (p ** g) * 100
            variance_band = 100 - allzero - allone
            rows.append((
                "Variance band (predicted)",
                f"{variance_band:.1f}% of groups are mixed-reward"
            ))
    cells = "\n".join(
        f'<div class="recipe-label">{escape(k)}</div>'
        f'<div class="recipe-value"><code>{escape(v)}</code></div>'
        for k, v in rows
    )
    return f"""
<h2>Recipe</h2>
<div class="recipe-grid">
{cells}
</div>"""


def render_kl_chart(rows: list[dict]) -> str:
    xs, ys = series(rows, "kl")
    if len(xs) < 3:
        return ""
    trace = (f"{{x: {json.dumps(xs)}, y: {json.dumps(ys)}, "
             f"type: 'scatter', mode: 'lines+markers', name: 'KL', "
             f"line: {{color: '{WARN}', width: 2}}, marker: {{size: 5}}}}")
    return f"""
<h2>KL divergence from reference policy</h2>
<p class="chart-hint">
  Low = the trained policy stays close to the reference model. If this
  explodes, the model has drifted and outputs will degrade.
</p>
<div class="chart-frame"><div id="kl-chart" class="chart"></div></div>
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
<div class="chart-frame"><div id="loss-chart" class="chart"></div></div>
<script>
Plotly.newPlot('loss-chart', [{", ".join(traces)}],
  {{...PLOT,
    xaxis: {{...PLOT.xaxis, title: 'Training step'}},
    yaxis: {{...PLOT.yaxis, title: 'Loss', side: 'left'}},
    yaxis2: {{gridcolor: 'rgba(255,255,255,0.06)', title: 'Grad norm',
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
             f"textfont: {{color: '#e4e4e7', size: 16}}}}")
    return f"""
<h2>Per-problem flip analysis ({flips['total']} common problems)</h2>
<p class="chart-hint">
  What happened at the problem level, not just in aggregate. Big green bar
  is the fix rate — how many previously-failing problems the RLVR-trained
  model now solves.
</p>
<div class="chart-frame"><div id="flip-chart" class="chart"></div></div>
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
<div class="example-pair">
  <div class="example fail">
    <div class="label">Task {task_id} — baseline FAIL</div>
    <pre>{b_code[:800]}</pre>
  </div>
  <div class="example pass">
    <div class="label">Task {task_id} — post-RLVR PASS</div>
    <pre>{p_code[:800]}</pre>
  </div>
</div>""")
    if not blocks:
        return ""
    return "<h2>Example fixes — problems the model learned to solve after RLVR</h2>" + "".join(blocks)


def render_sibling_note(sibling_label: str | None,
                        sibling_href: str | None) -> str:
    if not sibling_label or not sibling_href:
        return ""
    return f"""
<div class="sibling-note">
  <b>Companion report:</b>
  <a href="{escape(sibling_href)}">{escape(sibling_label)}</a>
</div>"""


def render_context_callout() -> str:
    return f"""
<div class="sibling-note" style="border-left-color: {WARN};">
  <b>About this run — hunting for GRPO's variance band.</b>
  The two prior experiments in this repo landed outside the band where
  GRPO can learn: <b>code-rlvr</b> (MBPP, baseline 2.7%) had groups that were
  almost all-fail, and <b>math-rlvr</b> (OLMo-2-7B-Instruct on GSM8K, baseline 82.6%)
  had groups that were mostly all-pass. Both deltas were within noise.
  This run picks <b>Gemma-2-2B-IT</b> (GSM8K baseline ~51%) and bumps
  <b>G = 8</b> with <b>temperature 1.0</b> — a configuration designed
  to keep the fraction of mixed-reward groups as high as possible.
  <br><br>
  Companion reports:
  <a href="../math-rlvr/results/gsm8k_update.html">math-rlvr (OLMo-2-7B · G=4)</a> ·
  <a href="../code-rlvr/results/grpo_report.html">code-rlvr (MBPP)</a>
</div>"""


def render_html(metrics_rows: list[dict], baseline: dict | None,
                post: dict | None, model_name: str,
                config: dict | None = None,
                label: str | None = None,
                num_gen_override: int | None = None,
                sibling_label: str | None = None,
                sibling_href: str | None = None) -> str:
    stat_cards = render_stat_cards(baseline, post, metrics_rows)
    recipe_panel = render_recipe_panel(config, baseline, post)
    sparse_note = render_sparse_reward_note(
        baseline, post, metrics_rows, num_gen_override=num_gen_override
    )
    sibling_note = render_sibling_note(sibling_label, sibling_href)
    context_callout = render_context_callout()
    reward = render_reward_chart(metrics_rows)
    variance = render_variance_chart(metrics_rows)
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
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemma-2-2B RLVR on GSM8K v2 — Unlocked update budget</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --bg-base:      #09090b;
    --bg-primary:   #0c0c0f;
    --bg-secondary: #141418;
    --bg-card:      #19191f;
    --bg-elevated:  #1f1f27;
    --text-primary:   #f0f0f2;
    --text-secondary: #c7c7d1;
    --text-muted:     #8b8b9a;
    --text-faint:     #65657a;
    --accent:  {ACCENT};
    --success: #34d399;
    --danger:  #f87171;
    --warn:    #fbbf24;
    --purple:  #a78bfa;
    --accent-glow:  rgba(77, 142, 255, 0.18);
    --success-glow: rgba(52, 211, 153, 0.15);
    --danger-glow:  rgba(248, 113, 113, 0.15);
    --warn-glow:    rgba(251, 191, 36, 0.14);
    --border:       rgba(255, 255, 255, 0.07);
    --border-light: rgba(255, 255, 255, 0.12);
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', 'Consolas', ui-monospace, monospace;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --shadow-card: 0 4px 16px -2px rgba(0, 0, 0, 0.35), 0 1px 0 0 rgba(255, 255, 255, 0.03) inset;
  }}

  * {{ box-sizing: border-box; }}

  html, body {{
    margin: 0;
    padding: 0;
    background: var(--bg-base);
  }}

  body {{
    font-family: var(--font-sans);
    color: var(--text-primary);
    font-size: 20px;
    line-height: 1.6;
    min-height: 100vh;
    /* Soft radial aura — rhymes with presentation viewport */
    background-color: var(--bg-base);
    background-image:
      radial-gradient(ellipse 70% 45% at 50% 0%, rgba(77, 142, 255, 0.055) 0%, transparent 65%),
      radial-gradient(ellipse 50% 35% at 85% 100%, rgba(167, 139, 250, 0.035) 0%, transparent 60%);
    background-attachment: fixed;
    font-feature-settings: 'ss01' 1, 'cv11' 1;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }}

  .container {{
    max-width: 1440px;
    margin: 0 auto;
    padding: 3.5rem 4rem 4rem;
  }}

  /* ── Header ───────────────────────────────────────────────── */
  .report-header {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: baseline;
    gap: 2rem;
    padding-bottom: 1.75rem;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid var(--border);
  }}
  .eyebrow {{
    font-family: var(--font-mono);
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--accent);
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-bottom: 0.65rem;
    opacity: 0.9;
  }}
  h1 {{
    font-family: var(--font-sans);
    color: var(--text-primary);
    font-size: 2.85rem;
    font-weight: 700;
    letter-spacing: -0.022em;
    line-height: 1.12;
    margin: 0 0 0.5rem 0;
    /* Subtle blue→purple gradient on the title, matching title slide */
    background: linear-gradient(135deg, #4d8eff 0%, #a78bfa 55%, #4d8eff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .subtitle {{
    color: var(--text-muted);
    font-size: 1.05rem;
    font-family: var(--font-mono);
    letter-spacing: 0.01em;
    margin: 0;
  }}
  .render-meta {{
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--text-faint);
    text-align: right;
    letter-spacing: 0.04em;
    white-space: nowrap;
  }}

  /* ── H2 Section Titles — accent-underline sweep, matches slides ── */
  h2 {{
    font-family: var(--font-sans);
    color: var(--text-primary);
    font-size: 1.55rem;
    font-weight: 600;
    letter-spacing: -0.008em;
    line-height: 1.3;
    margin: 3rem 0 1rem 0;
    padding-bottom: 0.55rem;
    background-image: linear-gradient(to right, var(--accent), transparent 35%);
    background-size: 100% 2px;
    background-position: bottom left;
    background-repeat: no-repeat;
  }}

  /* Opening H2 doesn't need the huge top margin after the header */
  h2:first-of-type {{ margin-top: 2rem; }}

  p {{ margin: 0.5rem 0 0.9rem; color: var(--text-secondary); }}

  /* ── Sibling companion link ──────────────────────────────── */
  .sibling-note {{
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(77, 142, 255, 0.04) 100%);
    border: 1px solid var(--border-light);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.4rem;
    margin: 0 0 2rem 0;
    font-size: 1.05rem;
    color: var(--text-secondary);
    box-shadow: var(--shadow-card);
  }}
  .sibling-note b {{ color: var(--text-primary); font-weight: 600; }}
  .sibling-note a {{
    color: var(--accent);
    text-decoration: none;
    font-weight: 500;
    border-bottom: 1px solid transparent;
    transition: border-color 200ms ease;
  }}
  .sibling-note a:hover {{ border-bottom-color: var(--accent); }}

  /* ── Stat cards — wide grid, accent glow on top edge ─────── */
  .stats {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0 2.5rem 0;
  }}
  .stat {{
    position: relative;
    background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.5rem 1.65rem;
    box-shadow: var(--shadow-card);
    overflow: hidden;
  }}
  .stat::before {{
    content: '';
    position: absolute;
    top: 0; left: 10%; right: 10%;
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.18), transparent);
  }}
  .stat-value {{
    font-family: var(--font-mono);
    font-size: 2.4rem;
    font-weight: 600;
    color: var(--accent);
    line-height: 1.1;
    letter-spacing: -0.02em;
    font-feature-settings: 'tnum' 1, 'zero' 1;
  }}
  .stat-label {{
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: var(--text-faint);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.5rem;
  }}

  /* ── Recipe panel ─────────────────────────────────────────── */
  .recipe-grid {{
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.5rem 1.25rem;
    background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    margin: 0.5rem 0 2rem;
    box-shadow: var(--shadow-card);
    font-size: 0.95rem;
  }}
  .recipe-label {{
    font-family: var(--font-mono);
    color: var(--text-faint);
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    align-self: center;
  }}
  .recipe-value {{
    color: var(--text-primary);
    align-self: center;
  }}
  .recipe-value code {{
    background: rgba(77, 142, 255, 0.08);
    color: var(--accent);
    padding: 0.15rem 0.5rem;
    border-radius: var(--radius-sm);
    font-size: 0.92rem;
  }}

  /* ── Charts ──────────────────────────────────────────────── */
  .chart-frame {{
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.25rem 0.75rem;
    margin: 0.75rem 0 1.5rem;
    box-shadow: var(--shadow-card);
  }}
  .chart {{
    width: 100%;
    max-width: 100%;
    height: 420px;
    margin: 0;
  }}
  .chart-hint {{
    color: var(--text-muted);
    font-size: 0.98rem;
    margin: -0.4rem 0 0.7rem;
  }}

  /* ── Callouts — the sparse-reward math explanation card ──── */
  .callout {{
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(251, 191, 36, 0.035) 100%);
    border: 1px solid var(--border-light);
    border-left: 3px solid var(--warn);
    border-radius: var(--radius-md);
    padding: 1.4rem 1.65rem;
    margin: 0.75rem 0 1.75rem;
    font-size: 1.1rem;
    line-height: 1.65;
    box-shadow: var(--shadow-card);
  }}
  .callout p {{ margin: 0.45rem 0; color: var(--text-secondary); }}
  .callout p:first-child {{ margin-top: 0; }}
  .callout p:last-child {{ margin-bottom: 0; }}
  .callout b {{ color: var(--text-primary); font-weight: 600; }}
  .callout i {{ color: var(--purple); font-style: italic; }}
  .math {{
    font-family: var(--font-mono);
    background: var(--bg-base);
    padding: 0.25rem 0.55rem;
    border-radius: var(--radius-sm);
    font-size: 0.92em;
    border: 1px solid var(--border);
    color: var(--text-primary);
  }}

  /* ── Example code cards (fixes) ──────────────────────────── */
  .example-pair {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
    gap: 1.25rem;
    margin: 1rem 0;
  }}
  .example {{
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.25rem;
    margin: 0;
    box-shadow: var(--shadow-card);
    overflow: hidden;
  }}
  .pass {{
    border-left: 3px solid var(--success);
    background: linear-gradient(135deg, var(--bg-secondary) 0%, rgba(52, 211, 153, 0.035) 100%);
  }}
  .fail {{
    border-left: 3px solid var(--danger);
    background: linear-gradient(135deg, var(--bg-secondary) 0%, rgba(248, 113, 113, 0.035) 100%);
  }}
  pre {{
    margin: 0.7rem 0 0;
    white-space: pre-wrap;
    font-family: var(--font-mono);
    font-size: 0.98rem;
    line-height: 1.55;
    color: var(--text-secondary);
    font-feature-settings: 'zero' 1;
  }}
  .label {{
    font-family: var(--font-mono);
    color: var(--text-faint);
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 500;
  }}
  .pass .label {{ color: var(--success); }}
  .fail .label {{ color: var(--danger); }}

  /* ── Footer ──────────────────────────────────────────────── */
  footer {{
    color: var(--text-faint);
    font-family: var(--font-mono);
    font-size: 0.88rem;
    margin-top: 4rem;
    padding-top: 1.25rem;
    border-top: 1px solid var(--border);
    letter-spacing: 0.02em;
  }}

  /* ── Responsive ──────────────────────────────────────────── */
  @media (max-width: 1100px) {{
    .container {{ padding: 2.5rem 2rem 3rem; }}
    .report-header {{ grid-template-columns: 1fr; }}
    .render-meta {{ text-align: left; }}
    h1 {{ font-size: 2.2rem; }}
  }}
</style>
</head>
<body>
<div class="container">

<header class="report-header">
  <div>
    <div class="eyebrow">Experiment 4 · RLVR · GSM8K · Gemma-2-2B · v2</div>
    <h1>Unlocked update budget{f" · {escape(label)}" if label else ""}</h1>
    <div class="subtitle">Base model: {escape(model_name)} · Held-out: openai/gsm8k test (1,319)</div>
  </div>
  <div class="render-meta">Rendered {ts}</div>
</header>

{context_callout}
{sibling_note}

{body_warning}

<div class="stats">
{stat_cards}
</div>

{recipe_panel}

{sparse_note}

<script>
const PLOT = {PLOT_LAYOUT_JS};
</script>

{passat1}
{reward}
{variance}
{kl}
{loss}
{flip_chart}
{flip_examples}

<footer>
  Generated by gemma-rlvr/make_report.py · source:
  <a href="https://github.com/tjuzek/rlvr-demo" style="color:var(--text-muted);text-decoration:underline;">github.com/tjuzek/rlvr-demo</a>
</footer>

</div>
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
                        default=ROOT / "results" / "gemma_gsm8k_report.html")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "output" / "training_config.json")
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
    config = load_json(args.config)

    model_name = (baseline or post or {}).get("model", "unknown")

    html = render_html(
        metrics_rows, baseline, post, model_name,
        config=config,
        label=args.label,
        num_gen_override=args.num_generations or (config or {}).get("num_generations"),
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
