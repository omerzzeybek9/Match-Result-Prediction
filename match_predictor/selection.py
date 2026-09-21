"""Selective 1X2 forecasts: coverage is always reported beside accuracy."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


DOUBLE_CHANCE_CODES = {0: "X2", 1: "12", 2: "1X"}


def double_chance_pick(probabilities):
    """Return the two-outcome pick obtained by excluding the least likely 1X2 result."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (3,) or not np.isfinite(probabilities).all():
        raise ValueError("Expected three finite home/draw/away probabilities")
    if (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
        raise ValueError("Home/draw/away probabilities must be non-negative and sum to one")
    excluded = int(probabilities.argmin())
    included = [index for index in range(3) if index != excluded]
    return {"code": DOUBLE_CHANCE_CODES[excluded],
            "probability": float(probabilities[included].sum()),
            "included_classes": included, "excluded_class": excluded}


def double_chance_metrics(frame, prefix):
    """Measure a 100%-coverage top-two result forecast on a labelled prediction table."""
    columns = [f"p_{prefix}_home", f"p_{prefix}_draw", f"p_{prefix}_away"]
    probabilities = frame[columns].to_numpy(dtype=float)
    excluded = probabilities.argmin(axis=1)
    actual = frame.actual_class.to_numpy(dtype=int)
    correct = excluded != actual
    successes, n = int(correct.sum()), len(frame)
    codes = np.array([DOUBLE_CHANCE_CODES[int(value)] for value in excluded])
    constant_baselines = {DOUBLE_CHANCE_CODES[value]: float((actual != value).mean()) if n else None
                          for value in range(3)}
    best_constant = max(value for value in constant_baselines.values() if value is not None) if n else None
    return {"total_matches": n, "selected_matches": n, "coverage": 1.0 if n else 0.0,
            "correct": successes, "accuracy": float(successes/n) if n else None,
            "accuracy_95_interval": wilson_interval(successes, n),
            "pick_counts": {code: int((codes == code).sum()) for code in ["1X", "12", "X2"]},
            "constant_baselines": constant_baselines, "best_constant_accuracy": best_constant,
            "improvement_vs_best_constant": float(successes/n-best_constant) if n else None,
            "rule": "Exclude the least likely result and keep the other two; no outcome labels are used to choose the pick."}


def wilson_interval(successes, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = successes/n
    centre = (p + z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return [float(max(0, centre-radius)), float(min(1, centre+radius))]


def selection_metrics(frame, mask):
    mask = np.asarray(mask, dtype=bool)
    chosen = frame.loc[mask]
    correct = chosen.predicted_class.eq(chosen.actual_class)
    n, successes = len(chosen), int(correct.sum())
    return {"total_matches": len(frame), "selected_matches": n,
            "coverage": float(n/len(frame)) if len(frame) else 0.0,
            "correct": successes, "accuracy": float(successes/n) if n else None,
            "accuracy_95_interval": wilson_interval(successes,n)}


def fit_policy(frame, target=0.70, min_matches=120):
    """Use policy-development labels only; conservative multiple-threshold bound.

    This selection bound is not an independent performance claim. Confirmation
    must be reported on a later untouched period, including abstention coverage.
    """
    thresholds = np.round(np.arange(.50, .851, .025),3)
    agreement_options = [False, True] if "agreement" in frame and not frame.agreement.all() else [False]
    z = float(norm.ppf(1-.05/(len(thresholds)*len(agreement_options))))
    qualified = (frame.home_experience >= 10) & (frame.away_experience >= 10)
    if "eligible" in frame:
        qualified &= frame.eligible.astype(bool)
    evaluations=[]
    for require_agreement in agreement_options:
        for threshold in thresholds:
            mask=qualified & (frame.confidence >= threshold)
            if require_agreement:
                mask &= frame.agreement.astype(bool)
            result=selection_metrics(frame,mask)
            result.update(threshold=float(threshold),require_agreement=require_agreement,
                conservative_lower_bound=wilson_interval(result['correct'],result['selected_matches'],z=z)[0])
            evaluations.append(result)
    eligible=[e for e in evaluations if e['selected_matches']>=min_matches and e['conservative_lower_bound']>=target]
    chosen=max(eligible,key=lambda e:e['selected_matches']) if eligible else None
    return {"target":target,"threshold":chosen['threshold'] if chosen else None,
            "require_agreement":chosen['require_agreement'] if chosen else None,
            "status":"validation_supported" if chosen else "insufficient_validation_evidence",
            "min_team_history":10,"min_validation_matches":min_matches,
            "selection_bound":"One-sided Wilson, Bonferroni over the fixed threshold grid; approximate and not a guarantee",
            "validation":chosen,"threshold_search":evaluations}


def apply_policy(frame, policy):
    if policy.get('threshold') is None:
        return np.zeros(len(frame),dtype=bool)
    eligible = frame.eligible.astype(bool) if "eligible" in frame else True
    agreement = frame.agreement.astype(bool) if policy.get('require_agreement') and "agreement" in frame else True
    return (eligible & agreement & (frame.confidence >= policy['threshold']) &
            (frame.home_experience >= policy['min_team_history']) &
            (frame.away_experience >= policy['min_team_history'])).to_numpy()


def policy_verdict(policy, confidence, home_history, away_history, data_age,
                   agreement=True, experimental=False):
    reasons=[]
    if experimental:
        reasons.append("Bu lig için seçim kuralı doğrulanmadı.")
    if not policy or policy.get('threshold') is None:
        reasons.append("%70 hedefi için geliştirme döneminde yeterli kanıt bulunamadı.")
    elif confidence < policy['threshold']:
        reasons.append(f"Tahmin güveni %{100*confidence:.1f}; seçme eşiği %{100*policy['threshold']:.1f}.")
    if policy and policy.get('require_agreement') and not agreement:
        reasons.append("İstatistik modeli ile piyasa aynı sonucu seçmiyor.")
    if min(home_history,away_history)<10:
        reasons.append("Takımlardan en az birinin maç geçmişi yetersiz.")
    if data_age>14:
        reasons.append(f"Veri maç tarihinden {data_age} gün eski; güncelleme gerekli.")
    return {"selected":not reasons,"label":"Seçim ölçütlerini karşılıyor" if not reasons else "Pas",
            "reasons":reasons,"target":.70,"threshold":policy.get('threshold') if policy else None,
            "note":"%70 hedefi, seçilmiş maçların geçmiş toplu başarısı içindir; bu maç için garanti değildir."}
