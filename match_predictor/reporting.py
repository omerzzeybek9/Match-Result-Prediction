"""Render recorded results; never reruns selection on the test set."""
import json
from pathlib import Path

from .data import LEAGUES, ROOT


def write_benchmark(directory=None, output=None):
    directory = Path(directory or ROOT / "artifacts")
    output = Path(output or ROOT / "reports/benchmark.md")
    reports = [json.loads(path.read_text()) for path in sorted(directory.glob('*_report.json'))]
    if not reports:
        raise ValueError("No reports found; train at least one league first")
    lines = ["# Independent chronological benchmark", "",
        "These are recorded held-out results, not training scores. Recipe selection used earlier validation periods only.", "",
        "| League | Test matches | Accuracy | League-mean accuracy | Rolling accuracy | Log loss | Rolling log loss | Exact score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        s=r['selected']; b=r['baselines']; rolling=b['rolling_poisson']
        lines.append(f"| {LEAGUES[r['league']][0]} | {s['matches']} | {s['accuracy']:.1%} | {b['league_mean']['accuracy']:.1%} | {rolling['accuracy']:.1%} | {s['log_loss']:.3f} | {rolling['log_loss']:.3f} | {s['exact_score_accuracy']:.1%} |")
    domestic=[r for r in reports if r['league']!='cl']
    n=sum(r['selected']['matches'] for r in domestic)
    if n:
        accuracy=sum(r['selected']['accuracy']*r['selected']['matches'] for r in domestic)/n
        loss=sum(r['selected']['log_loss']*r['selected']['matches'] for r in domestic)/n
        reference=sum(r['baselines']['rolling_poisson']['log_loss']*r['selected']['matches'] for r in domestic)/n
        lines += ["",f"Domestic leagues, match-weighted: **{n:,} test matches; {accuracy:.1%} accuracy; {loss:.3f} log loss vs {reference:.3f} rolling reference.**"]
    lines += ["", "Lower log loss is better. The model is selected for probability quality, so higher accuracy is not guaranteed.",
        "The old random-split notebook scores are not a valid comparison because their feature construction contains future information.",
        "", "## Evidence by league", ""]
    for r in reports:
        interval=r['log_loss_difference_vs_rolling']['day_bootstrap_95_interval']
        lines += [f"### {LEAGUES[r['league']][0]}", "",
            f"- History: {r['data']['from']} to {r['data']['to']}; {r['data']['matches']:,} matches.",
            f"- Test: {r['test_period']['from']} to {r['test_period']['to']}.",
            f"- Protocol: {r['protocol']}.",
            f"- Log-loss difference against rolling reference: {r['log_loss_difference_vs_rolling']['difference']:+.4f}; paired day-bootstrap 95% interval [{interval[0]:+.4f}, {interval[1]:+.4f}]. Negative favors the selected model; intervals crossing zero do not establish improvement.",
            f"- Selected mixture: {', '.join(f'{name}={weight:.3f}' for name,weight in r['weights'].items())}; temperature={r['temperature']}.", ""]
    lines += ["## Limits and interpretation", "",
        "- Six domestic leagues have multi-season history. Champions League is a separate small-data legacy experiment.",
        "- Model coefficients stay fixed during the test; prior match-day results update causal features.",
        "- Production artifacts are refitted afterwards on all results; their future accuracy remains unknown.",
        "- No lineup/injury/transfer/xG feed, bookmaker benchmark, or demonstrated betting profitability.",
        "- Confidence intervals describe sampling uncertainty, not guarantees against future distribution shifts.",
        "- Per-match predictions, calibration buckets, periods, versions and data hashes are available beside each model.",
        "- Tests: future-result invariance, prefix invariance, same-day isolation, training/live parity, temporal separation, current partial-season handling, valid probability mass, score/outcome consistency, invalid-input rejection and opponent sensitivity.", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines))
    return output
