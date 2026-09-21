"""Render recorded results; never reruns selection on the test set."""
import json
from pathlib import Path
from .data import LEAGUES, ROOT

def _pct(value):
    return "—" if value is None else f"{value:.1%}"

def write_benchmark(directory=None, output=None):
    directory=Path(directory or ROOT/"artifacts")
    output=Path(output or ROOT/"reports/benchmark.md")
    reports=[json.loads(path.read_text()) for path in sorted(directory.glob('*_report.json'))
             if path.name != "selective_report.json"]
    if not reports: raise ValueError("No reports found; train at least one league first")
    selective_path=directory/"selective_report.json"
    selective=json.loads(selective_path.read_text()) if selective_path.exists() else None
    lines=["# Independent chronological benchmark","",
      "Model recipes and strong-prediction thresholds were chosen with older validation periods. The confirmation season and latest partial-season audit were evaluated afterwards.","",
      "| League | Confirmation matches | Stats accuracy | Market-assisted accuracy | Stats log loss | Market-assisted log loss |",
      "|---|---:|---:|---:|---:|---:|"]
    domestic=[]
    for r in reports:
        s=r['selected'];m=r.get('market_assisted',s)
        lines.append(f"| {LEAGUES[r['league']][0]} | {s['matches']} | {s['accuracy']:.1%} | {m['accuracy']:.1%} | {s['log_loss']:.3f} | {m['log_loss']:.3f} |")
        if r['league']!='cl': domestic.append(r)
    n=sum(r['selected']['matches'] for r in domestic)
    if n:
        for key,label in [('selected','stats'),('market_assisted','market-assisted')]:
            accuracy=sum(r[key]['accuracy']*r[key]['matches'] for r in domestic)/n
            loss=sum(r[key]['log_loss']*r[key]['matches'] for r in domestic)/n
            lines.append(f"- Domestic {label}, match-weighted: **{n:,} matches; {accuracy:.1%} accuracy; {loss:.3f} log loss.**")
    if selective:
        lines += ["","## Every-match forecast: double chance","",
          "For every fixture, the two highest-probability 1-X-2 outcomes are retained. This is an easier target than exact 1-X-2 and must not be compared as if they were the same task.","",
          "| Mode | Confirmation matches | Coverage | Accuracy | Best constant pick | Uplift | 95% interval |",
          "|---|---:|---:|---:|---:|---:|---:|"]
        for mode,label in [('stats','Statistics only'),('assisted','Market assisted')]:
            result=selective['modes'][mode].get('all_match_double_chance')
            if result:
                interval=result['accuracy_95_interval']
                lines.append(f"| {label} | {result['total_matches']} | {_pct(result['coverage'])} | {_pct(result['accuracy'])} | {_pct(result['best_constant_accuracy'])} | {_pct(result['improvement_vs_best_constant'])} | {_pct(interval[0])}–{_pct(interval[1])} |")
        lines += ["", "The rule excludes the outcome with the lowest predicted probability. It emits a pick for every fixture and was scored on the untouched confirmation season.",
          "","## 70% target: selective exact 1-X-2 forecasts","",
          "The system may abstain. Accuracy without coverage is misleading, so both are always reported.","",
          "| Mode | Validation-selected threshold | Confirmation selected | Confirmation coverage | Confirmation accuracy | Latest audit selected | Latest audit accuracy |",
          "|---|---:|---:|---:|---:|---:|---:|"]
        for mode,label in [('stats','Statistics only'),('assisted','Market assisted')]:
            detail=selective['modes'][mode];policy=detail['policy'];c=detail['confirmation'];a=detail.get('latest_partial_season_audit')
            lines.append(f"| {label} | {_pct(policy.get('threshold'))} | {c['selected_matches']} | {_pct(c['coverage'])} | {_pct(c['accuracy'])} | {a['selected_matches'] if a else '—'} | {_pct(a['accuracy']) if a else '—'} |")
        lines += ["", "The threshold search used expanding validation predictions only. Confirmation and audit outcomes did not change the threshold."]
    lines += ["","## Evidence by league",""]
    for r in reports:
        interval=r['log_loss_difference_vs_rolling']['day_bootstrap_95_interval']
        lines += [f"### {LEAGUES[r['league']][0]}","",
          f"- History: {r['data']['from']} to {r['data']['to']}; {r['data']['matches']:,} matches.",
          f"- Confirmation: {r['test_period']['from']} to {r['test_period']['to']}; {r['test_period']['matches']} matches.",
          f"- Protocol: {r['protocol']}.",
          f"- Statistical log-loss difference against rolling reference: {r['log_loss_difference_vs_rolling']['difference']:+.4f}; paired day-bootstrap 95% interval [{interval[0]:+.4f}, {interval[1]:+.4f}]. Negative favors the selected model.",
          f"- Statistical mixture: {', '.join(f'{name}={weight:.3f}' for name,weight in r['weights'].items())}; temperature={r['temperature']}; market weight={r.get('market_weight','—')}.",""]
    lines += ["## Interpretation limits","",
      "- Double chance covers two of three outcomes and is therefore materially easier than exact 1-X-2. Its measured accuracy is not an exact-result accuracy.",
      "- A 70% selective result is a historical group rate, not a guarantee for one match.",
      "- Market-assisted mode requires three contemporaneous pre-match decimal odds. The historical evaluation uses the source's closing or available market-average snapshot.",
      "- The system has no dated lineup, injury, transfer or xG feed yet.",
      "- Goal totals and both-teams-to-score outputs have not been separately calibrated or confirmed.",
      "- No betting profitability, transaction-cost edge or superiority to the market is established.",
      "- Production artifacts are refitted on all available results after evaluation; their future performance remains unknown.",""]
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text('\n'.join(lines));return output
