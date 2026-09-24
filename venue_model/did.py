#!/usr/bin/env python3
"""
Difference-in-differences with staggered treatment — the estimator.

WHAT THIS IS FOR

Layer 2 can say that cities with bigger rooms get picked more often. It cannot
say that the room caused it, because cities build big rooms where promoters
already expected demand. The only way out of that with this data is a natural
experiment: watch what happens to a city when a room actually opens, and
compare it with cities where nothing opened.

WHY NOT JUST PUT UNIT AND YEAR FIXED EFFECTS IN A REGRESSION

Because it is wrong when treatment is staggered, which is exactly our case --
venues open in different years. The two-way fixed effects estimator makes
comparisons between every pair of units, and some of those pairs use an
ALREADY-TREATED city as the control for a later-treated one. If the effect of a
room grows or fades over time, those comparisons subtract a trend that is
itself a treatment effect, and the coefficient can come out with the wrong
sign even when every individual effect is positive. That is the Goodman-Bacon
decomposition, and it is not a subtlety: it is the main reason the
difference-in-differences literature was rewritten after 2018.

WHAT THIS DOES INSTEAD

Callaway and Sant'Anna's group-time approach. For each cohort g (the year its
room opened) and each year t, it estimates one clean comparison:

    ATT(g,t) = [ Y_t - Y_{g-1} ]  for cities treated in year g
             - [ Y_t - Y_{g-1} ]  for cities NOT YET treated by then

The control group is only ever cities that have not been treated at or before
max(g, t). An already-treated city never acts as a control. Those ATT(g,t) are
then averaged into an event study -- effect by years since opening -- which is
the output worth reading, because it shows the pre-trend as well as the effect.

THE PRE-TREND IS THE POINT, NOT A ROBUSTNESS CHECK

The whole design rests on an assumption that cannot be tested directly: absent
the new room, treated and control cities would have moved in parallel. What CAN
be seen is whether they were moving in parallel BEFORE the room opened. If the
event study shows treated cities already pulling away two years ahead of the
opening, the design is dead -- the room was built because the city was already
taking off, which is the very selection story Layer 2 could not rule out.

So the leads are not decoration. A Layer 3 result with a visible pre-trend is
worth less than the honest Layer 2 interval it was meant to replace.
"""

import numpy as np
import pandas as pd


def _weighted_mean(v, w=None):
    v = np.asarray(v, dtype=float)
    if not len(v):
        return np.nan
    if w is None:
        return float(v.mean())
    w = np.asarray(w, dtype=float)
    return float((v * w).sum() / w.sum()) if w.sum() else np.nan


def att_gt(panel, unit="unit", time="time", outcome="y", cohort="cohort",
           control="notyet"):
    """
    One clean comparison per (cohort, year).

    `cohort` is the period a unit was first treated; 0 or NaN means never
    treated. `control` picks who the comparison is against:

        "notyet"  units not treated at or before max(g, t). Uses more data and
                  is the default, because never-treated units are scarce here.
        "never"   only units never treated at all. Cleaner, weaker.

    Returns one row per (g, t) with the estimate, the group sizes, and the base
    period used -- so any single number on the event study can be traced back
    to the two differences it came from.
    """
    d = panel[[unit, time, outcome, cohort]].copy()
    d.columns = ["unit", "time", "y", "g"]
    d["g"] = pd.to_numeric(d["g"], errors="coerce").fillna(0).astype(int)

    wide = d.pivot_table(index="unit", columns="time", values="y")
    g_of = d.groupby("unit")["g"].first()
    return _att_gt_wide(wide, g_of, control)


def _att_gt_wide(wide, g_of, control="notyet"):
    """
    The same estimator on an already-pivoted matrix.

    Split out so the bootstrap can resample rows of `wide` directly. The first
    version rebuilt a long DataFrame for every resample -- 921 city filters
    times 200 reps -- which turned a two-minute job into an overnight one for
    no gain, since the pivot is the only thing that needed doing once.
    """
    times = sorted(wide.columns)
    cohorts = sorted(c for c in g_of.unique() if c > 0)

    rows = []
    for g in cohorts:
        base = g - 1
        if base not in wide.columns:
            # No pre-period for this cohort: it cannot be identified, and
            # dropping it silently would overstate how much evidence there is.
            rows.append({"cohort": g, "time": np.nan, "att": np.nan,
                         "n_treated": int((g_of == g).sum()), "n_control": 0,
                         "base_period": base,
                         "note": "no pre-period in the panel"})
            continue
        treated_units = g_of[g_of == g].index
        for t in times:
            if t == base:
                continue
            if control == "never":
                ctrl_units = g_of[g_of == 0].index
            else:
                cutoff = max(g, t)
                ctrl_units = g_of[(g_of == 0) | (g_of > cutoff)].index
            if not len(ctrl_units) or not len(treated_units):
                continue

            dt = wide.loc[treated_units, t] - wide.loc[treated_units, base]
            dc = wide.loc[ctrl_units, t] - wide.loc[ctrl_units, base]
            dt, dc = dt.dropna(), dc.dropna()
            if len(dt) < 1 or len(dc) < 2:
                continue
            rows.append({"cohort": g, "time": t,
                         "att": float(dt.mean() - dc.mean()),
                         "n_treated": int(len(dt)), "n_control": int(len(dc)),
                         "base_period": base, "note": ""})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["event_time"] = out["time"] - out["cohort"]
    return out


def event_study(gt, min_treated=1):
    """
    Effect by years since the room opened, averaged across cohorts.

    Weighted by how many treated cities each cohort contributes, so a year in
    which one city opened a room does not carry the same weight as one in which
    twelve did.
    """
    d = gt.dropna(subset=["att", "event_time"])
    d = d[d["n_treated"] >= min_treated]
    if d.empty:
        return pd.DataFrame()
    rows = []
    for e, grp in d.groupby("event_time"):
        rows.append({"years since opening": int(e),
                     "effect": _weighted_mean(grp["att"], grp["n_treated"]),
                     "cohorts": int(grp["cohort"].nunique()),
                     "treated cities": int(grp["n_treated"].sum())})
    return pd.DataFrame(rows).sort_values("years since opening")


def overall_att(gt):
    """The single post-opening average, weighted by cohort size."""
    d = gt.dropna(subset=["att", "event_time"])
    post = d[d["event_time"] >= 0]
    if post.empty:
        return np.nan
    return _weighted_mean(post["att"], post["n_treated"])


def pretrend_test(gt, leads=3):
    """
    Were treated and control cities moving together BEFORE the opening?

    Returns the average effect over the `leads` years before treatment, which
    should be indistinguishable from zero if the design holds. It is reported
    as a number rather than a pass/fail because the honest reading is a matter
    of degree: a pre-trend a tenth the size of the post effect is a caveat, one
    the same size is a refutation.

    MIND THE SIGN. Everything is measured against the base period g-1, so a
    treated group that was ALREADY CLIMBING before its room opened produces
    NEGATIVE leads: the earlier years sit below an inflated base. The direction
    is the opposite of the trend it reveals, which is why `size` below is the
    number to judge on and the signed mean is kept only for reading the shape.
    """
    d = gt.dropna(subset=["att", "event_time"])
    pre = d[(d["event_time"] < 0) & (d["event_time"] >= -leads)]
    if pre.empty:
        return {"leads_tested": 0, "mean_pre_effect": np.nan}
    signed = _weighted_mean(pre["att"], pre["n_treated"])
    return {"leads_tested": int(pre["event_time"].nunique()),
            "mean_pre_effect": signed,
            "size": abs(signed),
            "worst_lead": float(pre["att"].abs().max())}


def bootstrap(panel, reps=300, seed=0, unit="unit", time="time", outcome="y",
              cohort="cohort", control="notyet"):
    """
    Confidence intervals by resampling whole cities.

    Cities, not city-years: observations within a city are obviously correlated
    across years, and resampling rows would treat fifteen years of one city as
    fifteen independent facts. That would produce intervals far too narrow, in
    the same way unclustered standard errors do in Layer 2.

    The panel is pivoted once and each resample takes rows of that matrix, so
    the cost per replication is the estimator itself and nothing else.
    """
    rng = np.random.default_rng(seed)
    d = panel[[unit, time, outcome, cohort]].copy()
    d.columns = ["unit", "time", "y", "g"]
    d["g"] = pd.to_numeric(d["g"], errors="coerce").fillna(0).astype(int)
    wide = d.pivot_table(index="unit", columns="time", values="y")
    g_of = d.groupby("unit")["g"].first().reindex(wide.index)

    n = len(wide)
    overall, es = [], []
    for _ in range(int(reps)):
        pick = rng.integers(0, n, size=n)
        # A city drawn twice needs a distinct label, or the pivot index would
        # collapse the duplicates and quietly shrink the sample.
        w = wide.iloc[pick].copy()
        g = g_of.iloc[pick].copy()
        labels = [f"{i}" for i in range(n)]
        w.index, g.index = labels, labels

        gt = _att_gt_wide(w, g, control)
        if gt.empty:
            continue
        overall.append(overall_att(gt))
        e = event_study(gt)
        if not e.empty:
            es.append(e.set_index("years since opening")["effect"])

    overall = np.array([v for v in overall if np.isfinite(v)])
    band = pd.DataFrame(es).quantile([0.05, 0.95]).T if es else pd.DataFrame()
    return {"overall": overall,
            "overall_ci": (float(np.percentile(overall, 5)),
                           float(np.percentile(overall, 95)))
            if len(overall) else (np.nan, np.nan),
            "event_band": band}


# ---------------------------------------------------------------------------
# Self-test
#
# An estimator that is subtly wrong produces output indistinguishable from one
# that is right, so it is checked against data built with a known answer --
# including the case it exists to handle, where the effect grows over time and
# naive two-way fixed effects goes astray.
# ---------------------------------------------------------------------------

def _simulate(n_units=220, t0=2012, t1=2026, effect=4.0, growing=False,
              pretrend=0.0, seed=0):
    rng = np.random.default_rng(seed)
    years = list(range(t0, t1 + 1))
    # a third never treated; the rest open a room in a staggered fashion
    cohorts = rng.choice([0, 2016, 2018, 2020, 2022], size=n_units,
                         p=[0.35, 0.16, 0.16, 0.16, 0.17])
    unit_fx = rng.normal(20, 6, n_units)
    year_fx = {y: 0.4 * (y - t0) for y in years}

    rows = []
    for i in range(n_units):
        g = cohorts[i]
        for y in years:
            e = (y - g) if g else -999
            eff = 0.0
            if g and e >= 0:
                eff = effect * (1 + 0.35 * e) if growing else effect
            # a pre-trend, when asked for: treated cities already climbing
            if g and -4 <= e < 0:
                eff += pretrend * (4 + e)
            rows.append({"unit": f"c{i}", "time": y, "cohort": g,
                         "y": unit_fx[i] + year_fx[y] + eff
                              + rng.normal(0, 1.5)})
    return pd.DataFrame(rows)


def _twfe(panel):
    """Naive two-way fixed effects, for contrast only."""
    d = panel.copy()
    d["post"] = ((d["cohort"] > 0) & (d["time"] >= d["cohort"])).astype(float)
    u = pd.get_dummies(d["unit"], drop_first=True, dtype=float)
    t = pd.get_dummies(d["time"], drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(d)), d["post"].values, u.values, t.values])
    beta, *_ = np.linalg.lstsq(X, d["y"].values, rcond=None)
    return float(beta[1])


def _self_test():
    print("1. constant effect of +4.0, staggered openings")
    p = _simulate(effect=4.0, seed=1)
    gt = att_gt(p)
    got = overall_att(gt)
    print(f"   Callaway-Sant'Anna  {got:+.2f}   (truth +4.00)")
    print(f"   naive TWFE          {_twfe(p):+.2f}")
    assert abs(got - 4.0) < 0.5, f"constant-effect case off: {got}"

    print("\n2. effect GROWING with time since opening -- the case TWFE breaks on")
    p2 = _simulate(effect=4.0, growing=True, seed=2)
    gt2 = att_gt(p2)
    cs, tw = overall_att(gt2), _twfe(p2)
    es = event_study(gt2)
    print(f"   Callaway-Sant'Anna  {cs:+.2f}")
    print(f"   naive TWFE          {tw:+.2f}   <- biased downward by "
          f"already-treated controls")
    print(es.to_string(index=False))
    assert cs > tw + 0.5, ("TWFE should be biased downward here; if it is not, "
                           "the staggered design is not being exercised")
    # effect at e=0 should be near the truth of 4.0
    at0 = es.loc[es["years since opening"] == 0, "effect"].iloc[0]
    assert abs(at0 - 4.0) < 0.8, f"effect at opening off: {at0}"

    print("\n3. NO effect -- the estimator must not invent one")
    p3 = _simulate(effect=0.0, seed=3)
    got3 = overall_att(att_gt(p3))
    print(f"   Callaway-Sant'Anna  {got3:+.2f}   (truth 0.00)")
    assert abs(got3) < 0.5, f"invented an effect: {got3}"

    print("\n4. a PRE-TREND must show up in the leads")
    p4 = _simulate(effect=4.0, pretrend=1.2, seed=4)
    gt4 = att_gt(p4)
    pre = pretrend_test(gt4)
    print(f"   mean effect in the 3 years BEFORE opening  "
          f"{pre['mean_pre_effect']:+.2f}  (size {pre['size']:.2f})")
    clean = pretrend_test(att_gt(_simulate(effect=4.0, seed=5)))
    print(f"   same, on data with no pre-trend            "
          f"{clean['mean_pre_effect']:+.2f}  (size {clean['size']:.2f})")
    print("   the sign is negative because everything is measured against the")
    print("   base period g-1, which a rising pre-trend has already inflated.")
    # Judged on SIZE, not sign: see pretrend_test. The first version of this
    # test asserted a positive value and failed on correct output.
    assert pre["size"] > 1.0, "failed to detect a real pre-trend"
    assert clean["size"] < 0.6, "saw a pre-trend that is not there"

    print("\nPASS")


if __name__ == "__main__":
    _self_test()


def detrend(gt, leads=5):
    """
    The effect after removing the trend the treated cities were ALREADY on.

    A pre-trend does not automatically kill a design, but it does mean the raw
    effect double-counts: part of the post-opening rise is the continuation of
    something that was happening anyway. The standard sensitivity is to fit a
    straight line through the pre-period effects, extrapolate it across the
    post period, and subtract.

    This is a SENSITIVITY, NOT A CORRECTION, and the difference matters. The
    extrapolation assumes the pre-trend would have continued linearly, which is
    exactly as unverifiable as the parallel-trends assumption it is patching.
    If the raw and detrended numbers agree, the pre-trend was not doing much
    work; if they disagree, the honest answer is that the data cannot separate
    the room from the trend, and the detrended figure is the more conservative
    of two guesses rather than the right one.
    """
    d = gt.dropna(subset=["att", "event_time"])
    pre = d[(d["event_time"] < 0) & (d["event_time"] >= -leads)]
    if pre.empty or pre["event_time"].nunique() < 3:
        return {"ok": False,
                "why": f"needs at least 3 lead years within {leads}; "
                       f"the panel has {pre['event_time'].nunique()}"}

    es = event_study(gt)
    x = pre["event_time"].astype(float).values
    y = pre["att"].astype(float).values
    w = pre["n_treated"].astype(float).values
    # weighted least squares on the leads
    W = np.diag(w)
    A = np.column_stack([np.ones_like(x), x])
    slope_fit = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ y, rcond=None)[0]
    intercept, slope = float(slope_fit[0]), float(slope_fit[1])

    out = es.copy()
    e = out["years since opening"].astype(float)
    out["pre-trend line"] = (intercept + slope * e).round(3)
    out["detrended effect"] = (out["effect"] - out["pre-trend line"]).round(3)

    post = out[out["years since opening"] >= 0]
    raw = _weighted_mean(post["effect"], post["treated cities"])
    adj = _weighted_mean(post["detrended effect"], post["treated cities"])
    return {"ok": True, "slope_per_year": slope, "intercept": intercept,
            "raw_att": raw, "detrended_att": adj, "table": out}
