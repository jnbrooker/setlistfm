#!/usr/bin/env python3
"""
Layer 2 — what actually drives the choice of city, and what a new room changes.

WHAT THIS PRODUCES

A conditional logit over tour-city choices, and from it the answer to the
question Layer 1 cannot reach: if this market gained a room of size X, how many
more tours would choose it?

    python layer2.py --market Naples --capacity 13500 --kind indoor
    python layer2.py --market Naples --sweep          # size against payoff

WHAT A COEFFICIENT HERE MEANS, AND WHAT IT DOES NOT

The model is fitted on which cities tours actually picked, holding constant
everything about the act -- the conditional logit differences that out for
free. So a coefficient on capacity says: among cities a given tour was choosing
between, the ones with bigger rooms were picked more often, by this much.

It does NOT say that building a room causes tours to come. Cities have big
arenas because promoters expected demand there. Capacity is correlated with
every unobserved thing that makes a city attractive -- a strong local promoter,
a habit of selling out, a university, a motorway junction -- and the
coefficient carries all of it.

HOW THIS FILE HANDLES THAT, RATHER THAN IGNORING IT

Two specifications are fitted, and the honest answer is the interval between
them.

  A  catchment, income, capacity, routing.
     Capacity carries its own effect PLUS every unobserved reason the city is
     attractive. Its coefficient is an UPPER BOUND on the causal effect.

  B  the same, plus how many shows the market already hosts.
     Past activity is the clearest proxy available for those unobserved
     reasons -- but it is also partly the thing capacity delivers, so
     controlling for it strips out some of the real effect too. Its
     coefficient is a LOWER BOUND.

Neither is "the" answer. Reporting the pair is the closest this data gets to
honesty, and the gap between them is a direct measure of how much the question
is unsettled. If the two bounds agree, the finding is robust to the worry. If
they are far apart, the data cannot settle it and no amount of further
modelling on the same data will.

Layer 3 -- difference-in-differences around real venue openings -- is what
would narrow the interval, because a venue that opened in 2018 gives a before
and an after for the same city. It does not exist yet.
"""

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import choice                                            # noqa: E402
import logit                                             # noqa: E402
from gap import latest_extract, load_extract             # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# The design matrix
# ---------------------------------------------------------------------------

# What each variable is, in one sentence, for every place the model is
# reported. Kept beside the code that builds the column so the two cannot
# drift apart.
VARIABLE_NOTES = {
    "log catchment": (
        "Log of the people for whom this is the nearest market hosting shows, "
        "within 60 km, in millions. A coefficient of 1 means doubling the "
        "catchment doubles the odds of being picked."),
    "log income per head": (
        "Log of disposable spending power per resident. Separates 'lots of "
        "people' from 'people with money'."),
    "log fit gap": (
        "How far the market's CLOSEST room is from what this act plays, in "
        "log seats. Zero is a perfect fit. It exists because the ceiling alone "
        "made a market whose only room is a 20,000 arena score 'big enough' "
        "for an act that plays to 800 -- true, and useless. This is what a "
        "promoter is actually judging: not whether the city has a big room, "
        "but whether it has the RIGHT room. Adding it lifted out-of-sample "
        "fit more than any other variable in the model, and pulled the raw "
        "size coefficient down by a third, which is the tell that size was "
        "standing in for fit all along."),
    "log largest room": (
        "Log of the biggest room of the kind this act plays -- indoor acts "
        "against the indoor ceiling, outdoor against outdoor."),
    "has a room big enough": (
        "1 if the market's biggest room of the right kind is at least as large "
        "as the room this act uses elsewhere. THE POLICY VARIABLE: it is the "
        "only one a new building moves in a way that could not already be "
        "achieved by the city getting bigger."),
    "log km from the tour's other stops": (
        "Distance to the nearest other city this tour played in the country, "
        "leaving the focal visit out. It measures SPREAD, not routing "
        "efficiency, and its coefficient is positive: tours deliberately "
        "space their dates across a country rather than clustering, so the "
        "city that gets picked is typically the one furthest from the ones "
        "already booked. It is in the model as a control for that spacing "
        "pattern, not as a finding."),
    "log shows already hosted": (
        "How many shows the market ran, minus this tour's own. Specification B "
        "only. It proxies everything unobserved that makes a city attractive, "
        "and in doing so absorbs part of what capacity delivers."),
    "spread x log stops in country": (
        "Does the spreading pattern depend on how many cities the tour plays? "
        "On the full sample: NO. The coefficient is indistinguishable from "
        "zero in both specifications, so a two-stop run and an eight-stop run "
        "space their dates the same way. Fitted on Italy alone it came out "
        "strongly positive, which was 292 tours' worth of noise -- a useful "
        "reminder of what a single country buys you."),
    "spread x log room needed": (
        "Does it depend on how big the act is? YES. Positive and significant "
        "in both specifications: bigger acts space their dates further apart, "
        "which is what you would expect when three arena shows have to cover a "
        "whole country without competing for the same audience. This is why "
        "spread cannot be a single number -- a stadium run and a club run are "
        "not playing the same game."),
    "room big enough x log room needed": (
        "Does having a big enough room matter MORE for bigger acts? Positive, "
        "and significant once past activity is controlled for. The capacity "
        "constraint binds hardest exactly where the rooms are largest -- which "
        "is the case that matters for deciding what to build."),
}


def _fitgap_median(rows):
    """Median fit gap, for filling markets with no room of a known size."""
    best = pd.to_numeric(rows.get("best_room"), errors="coerce")
    need = pd.to_numeric(rows.get("room_needed"), errors="coerce")
    if best is None or best.notna().sum() == 0:
        return 0.75
    g = (np.log(best.clip(lower=50)) - np.log(need.clip(lower=50))).abs()
    return float(g.median()) if g.notna().any() else 0.75


def design_constants(rows):
    """
    The six numbers design() takes from the WHOLE sample rather than from the
    row in front of it: two medians and a percentile used to fill gaps, and the
    two means the interactions are centred on.

    They are pulled into their own function for one reason, and it is not tidiness.

    A counterfactual rebuilds the frame with one market's ceiling raised. If the
    constants were re-derived from that rebuilt frame they would move -- the
    1st-percentile ceiling most obviously, since it is the value stood in for
    markets with no room of the right kind. That would shift the `log largest
    room` column for every OTHER market on the menu as well, and the change in
    predicted visits would no longer be attributable to the one intervention,
    which is the whole reason for doing it this way rather than by refitting.

    So the constants are computed once, from the world as it is, and frozen.
    They are also saved with the fitted coefficients, because a coefficient on a
    centred column is only meaningful against the centring it was fitted with:
    apply the full-sample betas to a differently-centred column and the answer
    is quietly wrong rather than loudly broken.
    """
    inc = pd.to_numeric(rows["income"], errors="coerce")
    ceil = pd.to_numeric(rows["ceiling"], errors="coerce")
    need = pd.to_numeric(rows["room_needed"], errors="coerce")
    km = pd.to_numeric(rows["routing_km"], errors="coerce")
    stops = np.log(pd.to_numeric(rows["stops_in_country"],
                                 errors="coerce").fillna(2).clip(lower=1))
    big = np.log(need.fillna(need.median()).clip(lower=100) / 1000.0)
    return {
        "income_median": float(inc.median()) if inc.notna().any() else 1e4,
        "ceiling_floor": (float(np.nanpercentile(ceil.dropna(), 1))
                          if ceil.notna().any() else 500.0),
        "need_median": float(need.median()) if need.notna().any() else 1000.0,
        "fitgap_median": _fitgap_median(rows),
        "km_median": float(km.median()) if km.notna().any() else 0.0,
        "stops_mean": float(stops.mean()),
        "big_mean": float(big.mean()),
    }


def design(rows, spec="A", consts=None):
    """
    Build X from the choice table.

    A NOTE ON WHAT CAN AND CANNOT GO IN HERE, because it catches people out.

    In a conditional logit, anything constant across a menu vanishes. The room
    an act needs is a property of the act, so it is the same for all ninety
    Italian markets on that occasion -- put it in on its own and it contributes
    exactly nothing, because it cancels top and bottom of the softmax.

    This is why `has a room big enough` is the interesting variable and
    `log largest room` is not sufficient on its own. The first is a comparison
    between an act-level number and a market-level number: it varies across the
    menu precisely because different markets clear the same bar differently.
    `log largest room` measures size in general; `has a room big enough`
    measures whether the size is enough for THIS act. Only the second is what a
    promoter is actually deciding on, and only the second responds to a
    building in a way that a bigger city would not also deliver.
    """
    k = consts if consts is not None else design_constants(rows)

    cat = pd.to_numeric(rows["catchment"], errors="coerce")
    inc = pd.to_numeric(rows["income"], errors="coerce")
    ceil = pd.to_numeric(rows["ceiling"], errors="coerce")
    need = pd.to_numeric(rows["room_needed"], errors="coerce")
    km = pd.to_numeric(rows["routing_km"], errors="coerce")

    cols, names = [], []

    cols.append(np.log(cat.clip(lower=1_000) / 1e6))
    names.append("log catchment")

    cols.append(np.log(inc.fillna(k["income_median"]).clip(lower=1_000) / 1e4))
    names.append("log income per head")

    # A market with no known room of that kind is given the smallest room on
    # the menu rather than dropped. Dropping it would quietly remove the very
    # alternatives the model should be learning are unattractive.
    floor = k["ceiling_floor"]
    cols.append(np.log(ceil.fillna(floor).clip(lower=floor)))
    names.append("log largest room")

    cols.append(((ceil.fillna(0) >= need) & need.notna()).astype(float))
    names.append("has a room big enough")

    # HOW WELL THE MARKET'S LADDER FITS THIS ACT, as opposed to how big its
    # biggest room is. See VARIABLE_NOTES for why the ceiling alone was not
    # enough. Absolute log distance, so being half the right size and twice
    # the right size are equally poor fits -- which is the honest reading: an
    # act does not want a room it cannot fill any more than one it cannot
    # get into.
    best = pd.to_numeric(rows.get("best_room"), errors="coerce") \
        if "best_room" in rows else pd.Series(np.nan, index=rows.index)
    gap = (np.log(best.clip(lower=50)) - np.log(need.clip(lower=50))).abs()
    cols.append(gap.fillna(k.get("fitgap_median", 0.75)))
    names.append("log fit gap")

    # NOT a routing-efficiency measure, despite the obvious reading.
    #
    # The first fit returned +0.96 on this and I took it for a leak, which it
    # partly was -- the tour's own other stops were sitting on the menu at a
    # distance of zero to themselves while never being the chosen alternative.
    # Removing them from the menu (see choice.build) cut the coefficient to
    # +0.66 but did not flip it, because the remaining signal is real: acts
    # spread their dates across a country. A tour playing Milan and Rome is
    # 475 km apart on purpose, so on the Milan occasion the chosen city is the
    # one FURTHEST from Rome, and every candidate clustered near Rome was
    # passed over.
    #
    # True routing -- would this date fit between the two either side of it --
    # needs the order the dates were played in, which means working at date
    # grain rather than city grain. That is a Layer 3 refinement. What this
    # column does here is hold the spacing pattern constant so it cannot be
    # picked up by catchment or capacity, and that job it does.
    spread = np.log1p(km.fillna(k["km_median"]))
    cols.append(spread)
    names.append("log km from the tour's other stops")

    # --- interactions -----------------------------------------------------
    #
    # A single spread coefficient assumes every tour spaces its dates the same
    # way, and that is plainly wrong: a three-city stadium run covers a country
    # by hitting its corners, while an eight-city club run fills it in. Both
    # terms below are a chooser-level number multiplied by an
    # alternative-varying one, which is the only way an artist characteristic
    # can enter a conditional logit at all -- on its own it would cancel.
    #
    # They are centred so the main effect stays readable: with the multiplier
    # at zero for an average tour, the plain `log km` coefficient is still the
    # spread effect for a typical act rather than for a meaningless act of zero
    # size.
    stops = np.log(pd.to_numeric(rows["stops_in_country"],
                                 errors="coerce").fillna(2).clip(lower=1))
    stops_c = stops - k["stops_mean"]
    big = np.log(need.fillna(k["need_median"]).clip(lower=100) / 1000.0)
    big_c = big - k["big_mean"]

    cols.append(spread * stops_c)
    names.append("spread x log stops in country")

    cols.append(spread * big_c)
    names.append("spread x log room needed")

    cols.append(cols[3] * big_c)          # `has a room big enough` x act size
    names.append("room big enough x log room needed")

    if spec.upper() == "B":
        # Subtract this tour's own dates before logging: without that, the
        # market's show count literally contains the outcome being predicted.
        prior = (pd.to_numeric(rows["market_events"], errors="coerce").fillna(0)
                 - pd.to_numeric(rows["own_dates_in_market"], errors="coerce").fillna(0))
        cols.append(np.log1p(prior.clip(lower=0)))
        names.append("log shows already hosted")

    X = np.column_stack([c.values.astype(float) for c in cols])
    if not np.isfinite(X).all():
        bad = [names[i] for i in range(X.shape[1]) if not np.isfinite(X[:, i]).all()]
        raise ValueError(f"non-finite values in: {bad}")
    return X, names


def prepare(rows):
    """Sort into contiguous occasions and hand back what the estimator needs."""
    rows = rows.sort_values(["countryCode", "occasion"], kind="stable").reset_index(drop=True)
    codes, _ = pd.factorize(rows["occasion"], sort=False)
    rows["_occ"] = codes
    offsets = logit.group_offsets(codes)
    y = rows["chosen"].values.astype(float)
    return rows, offsets, y


def fit_spec(rows, offsets, y, spec, verbose=False, consts=None):
    consts = consts if consts is not None else design_constants(rows)
    X, names = design(rows, spec, consts)
    res = logit.fit(X, y, offsets, cluster=rows["tour"].values, names=names,
                    verbose=verbose)
    res["spec"] = spec
    res["X"] = X
    res["consts"] = consts
    return res


# ---------------------------------------------------------------------------
# Reading the coefficients out loud
# ---------------------------------------------------------------------------

def interpret(res):
    """Turn each coefficient into a sentence a non-modeller can check."""
    out = []
    for name, b, se in zip(res["names"], res["beta"], res["se"]):
        sig = "" if abs(b) > 1.96 * se else "  (not distinguishable from zero)"
        if " x " in name:
            # An interaction is a slope on a slope, and the doubling phrasing
            # used below would be actively misleading for it. Say what it does.
            first, second = name.split(" x ", 1)
            direction = "strengthens" if b > 0 else "weakens"
            out.append(f"{name}: the effect of '{first}' {direction} as "
                       f"{second.replace('log ', '')} rises "
                       f"(by {abs(b):.3f} per log unit).{sig}")
        elif name.startswith("log "):
            # A coefficient on a logged variable is an elasticity, which almost
            # nobody reads fluently. Doubling is the unit people can picture.
            eff = 2 ** b - 1
            out.append(f"{name}: doubling it multiplies the odds of being "
                       f"picked by {2 ** b:.2f} ({eff:+.0%}).{sig}")
        elif " x " in name:
            out.append(f"{name}: each doubling of the second term multiplies "
                       f"the first term's effect by {2 ** b:.2f}.{sig}")
        else:
            out.append(f"{name}: having it multiplies the odds of being picked "
                       f"by {np.exp(b):.2f} ({np.exp(b) - 1:+.0%}).{sig}")
    return out


# ---------------------------------------------------------------------------
# Saving the fitted model
#
# The coefficients are the whole model -- five or six numbers and their
# standard errors. Writing them to a small JSON file means the front end can
# load a fitted model in milliseconds instead of refitting 1.3 million rows on
# every page load, and it means the model that produced a given screenshot can
# be pointed at and read.
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(HERE, "models")


def save_model(results, stamp, n_rows, n_occasions, n_tours, countries, consts):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"layer2_{stamp}.json")
    blob = {
        "fitted": dt.datetime.now().isoformat(timespec="seconds"),
        "extract": stamp,
        "countries": sorted(countries),
        "n_rows": int(n_rows), "n_occasions": int(n_occasions),
        "n_tours": int(n_tours),
        # Saved because a coefficient on a centred or gap-filled column only
        # means anything against the centring it was fitted with. The app
        # builds one country's menu at a time and must not re-derive these from
        # that subset -- see design_constants.
        "design_constants": consts,
        "variable_notes": VARIABLE_NOTES,
        "specs": {
            spec: {
                "names": r["names"],
                "beta": [float(b) for b in r["beta"]],
                "se": [float(e) for e in r["se"]],
                # The full covariance, not just its diagonal. Standard errors
                # alone cannot propagate uncertainty through a counterfactual:
                # the coefficients are correlated -- capacity and "big enough"
                # especially, since they measure overlapping things -- and
                # drawing them independently would overstate the spread badly.
                # Eight numbers become sixty-four, which is nothing to store
                # and the difference between a simulation that means something
                # and one that does not.
                "cov": [[float(v) for v in row] for row in r["cov"]],
                "loglik": r["loglik"], "pseudo_r2": r["pseudo_r2"],
                "se_kind": r["se_kind"], "iterations": r["iterations"],
            } for spec, r in results.items()},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    return path


def load_model(stamp=None):
    """Newest fitted model, or the one matching an extract stamp."""
    if not os.path.isdir(MODEL_DIR):
        return None
    files = sorted(f for f in os.listdir(MODEL_DIR) if f.startswith("layer2_"))
    if not files:
        return None
    want = f"layer2_{stamp}.json" if stamp else None
    name = want if want in files else files[-1]
    with open(os.path.join(MODEL_DIR, name), encoding="utf-8") as fh:
        blob = json.load(fh)
    # Record whether this is actually the model for the extract that was asked
    # for. The fallback to the newest file is convenient and dangerous: a model
    # fitted on a 2023-onward extract applied to a 2020-onward one would pass
    # every column check, because the variables are identical, and be wrong in
    # a way nothing downstream could detect.
    blob["loaded_file"] = name
    blob["stamp_requested"] = stamp
    blob["stamp_matched"] = bool(stamp is None or want == name)
    return blob


# ---------------------------------------------------------------------------
# The counterfactual
# ---------------------------------------------------------------------------

def counterfactual(rows, offsets, res, spec, market, capacity, kind="indoor",
                   country=None):
    """
    Rebuild the menu with a new room in one market, and re-predict.

    Only two columns move: the market's ceiling for acts of the matching kind,
    and whether that ceiling now clears the bar for each act. Everything else
    -- catchment, income, routing, the rest of the menu -- is held exactly as
    it was. That is the point of doing it this way rather than by re-fitting:
    the change in predicted visits is attributable to one intervention and
    nothing else.

    WHAT THE RESULT IS AND IS NOT

    It is the model's expected number of tour-visits, summed over every
    occasion in the country, before and after. Because the coefficients carry
    the selection problem described at the top of this file, the 'after' figure
    is what a city with that room and otherwise identical characteristics would
    get -- not what THIS city would get after building one. Those differ by
    however much the room is a symptom of demand rather than a cause of it.
    """
    target = rows["market"].astype(str).str.casefold() == market.strip().casefold()
    if country:
        target &= rows["countryCode"].str.upper() == country.upper()
    if not target.any():
        raise SystemExit(f"no market called {market!r} on any menu")
    code = rows.loc[target, "countryCode"].iloc[0]
    in_country = rows["countryCode"] == code
    target = target & in_country

    after = rows.copy()
    # The new room only helps acts of the matching kind. An indoor arena does
    # nothing for an act that plays festivals, and saying otherwise is the
    # commonest way this sort of projection is inflated.
    if kind in ("indoor", "outdoor"):
        applies = target & (after["act_plays"].isin([kind, "either"]))
    else:
        applies = target
    after.loc[applies, "ceiling"] = np.maximum(
        pd.to_numeric(after.loc[applies, "ceiling"], errors="coerce").fillna(0),
        capacity)

    X_after, _ = design(after, spec, res.get("consts"))
    beta = res["beta"]

    # restrict to this country's occasions, since nothing else can change
    occ = rows["_occ"].values
    mask = in_country.values
    sub_occ = occ[mask]
    sub_off = logit.group_offsets(sub_occ)

    p0 = logit.choice_probabilities(res["X"][mask] @ beta, sub_off)
    p1 = logit.choice_probabilities(X_after[mask] @ beta, sub_off)

    t = target.values[mask]
    before_visits = float(p0[t].sum())
    after_visits = float(p1[t].sum())

    # per-tour detail: which acts move most
    sub = rows.loc[mask].copy()
    sub["p_before"], sub["p_after"] = p0, p1
    hits = sub[t].copy()
    hits["gain"] = hits["p_after"] - hits["p_before"]
    per_tour = (hits.groupby(["tour", "category", "act_plays", "room_needed"],
                             as_index=False)
                .agg(occasions=("gain", "size"),
                     p_before=("p_before", "sum"),
                     p_after=("p_after", "sum"),
                     gain=("gain", "sum"))
                .sort_values("gain", ascending=False))

    return {
        "market": market, "countryCode": code, "capacity": capacity, "kind": kind,
        "expected_visits_before": before_visits,
        "expected_visits_after": after_visits,
        "extra_visits": after_visits - before_visits,
        "occasions": int(len(sub_off)),
        "per_tour": per_tour,
        "already_clears": float((pd.to_numeric(rows.loc[target, "ceiling"],
                                               errors="coerce").fillna(0)
                                 >= pd.to_numeric(rows.loc[target, "room_needed"],
                                                  errors="coerce")).mean()),
    }


def decompose(rows, res, spec, market, capacity, kind="indoor", country=None,
              top=25):
    """
    WHY the model expects more tours -- variable by variable, tour by tour.

    A counterfactual that reports only "+5.4 tours" is a number to be taken on
    trust. This returns the arithmetic that produced it.

    HOW THE DECOMPOSITION WORKS, AND WHY IT IS EXACT

    Adding a room changes exactly two things about the market: how big its
    biggest room is, and whether that room now clears each act's bar. Every
    other column -- catchment, income, spread -- is untouched. So the change in
    the market's utility for a given tour is

        du  =  sum over variables of  (change in that variable) x (its coefficient)

    and that sum is exact, not an approximation or an attribution heuristic.
    Each term is one variable's contribution in utility units, and they add up
    to the total with nothing left over.

    Probabilities are then the softmax of utility, which is NOT additive -- a
    given utility gain moves a market from 2% to 4% but from 40% to only 45%,
    because it has to take the probability from somewhere. So this returns both:
    the exact additive story in utility, and the actual probability change that
    follows from it. Reporting only the first would overstate; reporting only
    the second would hide the reason.
    """
    target = rows["market"].astype(str).str.casefold() == market.strip().casefold()
    if country:
        target &= rows["countryCode"].str.upper() == country.upper()
    if not target.any():
        raise SystemExit(f"no market called {market!r} on any menu")
    code = rows.loc[target, "countryCode"].iloc[0]
    in_country = (rows["countryCode"] == code).values
    target = (target & (rows["countryCode"] == code)).values

    after = rows.copy()
    if kind in ("indoor", "outdoor"):
        applies = target & after["act_plays"].isin([kind, "either"]).values
    else:
        applies = target
    after.loc[applies, "ceiling"] = np.maximum(
        pd.to_numeric(after.loc[applies, "ceiling"], errors="coerce").fillna(0),
        capacity)

    X_before = res["X"]
    X_after, names = design(after, spec, res.get("consts"))
    beta = res["beta"]

    occ = rows["_occ"].values
    sub_off = logit.group_offsets(occ[in_country])
    p0 = logit.choice_probabilities(X_before[in_country] @ beta, sub_off)
    p1 = logit.choice_probabilities(X_after[in_country] @ beta, sub_off)

    t = target[in_country]
    # per-variable utility change, on the target market's rows only
    dX = (X_after[target] - X_before[target])
    contrib = dX * beta[None, :]

    sub = rows.loc[in_country].copy()
    sub["p_before"], sub["p_after"] = p0, p1
    hit = sub[t].copy()
    hit["gain"] = hit["p_after"] - hit["p_before"]
    hit["utility_gain"] = contrib.sum(axis=1)
    for j, nm in enumerate(names):
        if np.any(np.abs(contrib[:, j]) > 1e-12):
            hit[f"from: {nm}"] = contrib[:, j]

    moved = hit[hit["gain"] > 1e-6].copy()
    per_tour = (moved.sort_values("gain", ascending=False)
                .head(top)
                [[c for c in ["tour", "category", "act_plays", "room_needed",
                              "ceiling", "p_before", "p_after", "gain",
                              "utility_gain"] if c in moved.columns]
                 + [c for c in moved.columns if c.startswith("from: ")]])

    # aggregate: how much of the TOTAL utility gain each variable supplied,
    # weighted by how much each tour's probability actually moved -- so a
    # variable that only helps tours the market was never going to get does
    # not get credit for the headline number.
    w = hit["gain"].clip(lower=0).values
    share = {}
    for j, nm in enumerate(names):
        c = contrib[:, j]
        if np.any(np.abs(c) > 1e-12):
            share[nm] = float(np.sum(c * w))
    total = sum(share.values())
    share_tbl = pd.DataFrame(
        [{"variable": k,
          "utility contributed (gain-weighted)": round(v, 4),
          "share of the reason": round(v / total, 3) if total else None,
          "what it means": VARIABLE_NOTES.get(k, "")}
         for k, v in sorted(share.items(), key=lambda kv: -abs(kv[1]))])

    return {
        "market": market, "capacity": capacity, "kind": kind,
        "extra_visits": float(p1[t].sum() - p0[t].sum()),
        "tours_moved": int((hit["gain"] > 0.01).sum()),
        "per_tour": per_tour,
        "why": share_tbl,
        "median_gain_pp": float(moved["gain"].median() * 100) if len(moved) else 0.0,
    }


def capacity_sweep(rows, offsets, res, spec, market, kind="indoor",
                   sizes=None, country=None):
    """
    Expected extra visits at each candidate room size.

    The shape of this curve is the actual answer to "how big should it be".
    It is a step function, not a smooth one, because what the model responds to
    is clearing each act's bar -- so the gains arrive in jumps as the proposed
    room passes the sizes that touring acts actually use, and the flat stretches
    between them are capacity that buys nothing.
    """
    if sizes is None:
        sizes = [1_500, 2_500, 3_500, 5_000, 6_500, 8_000, 10_000, 12_000,
                 15_000, 18_000, 20_000, 25_000]
    out = []
    for cap in sizes:
        cf = counterfactual(rows, offsets, res, spec, market, cap, kind, country)
        out.append({"capacity": cap,
                    "expected extra visits": round(cf["extra_visits"], 2),
                    "tours materially helped": int(
                        (cf["per_tour"]["gain"] > 0.01).sum())})
    df = pd.DataFrame(out)
    df["extra per 1,000 seats"] = (
        df["expected extra visits"].diff()
        / (df["capacity"].diff() / 1000)).round(3)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", default=None)
    ap.add_argument("--countries", nargs="*", default=None,
                    help="restrict the menus to these country codes")
    ap.add_argument("--market", default=None, help="market for the counterfactual")
    ap.add_argument("--country", default=None, help="disambiguate --market")
    ap.add_argument("--capacity", type=int, default=None)
    ap.add_argument("--kind", default="indoor", choices=["indoor", "outdoor"])
    ap.add_argument("--sweep", action="store_true",
                    help="expected gain at a range of room sizes")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--save", action="store_true",
                    help="write the fitted coefficients to models/ so the app "
                         "can load them without refitting")
    a = ap.parse_args()

    ex = load_extract(a.extract or latest_extract())
    log("building choice occasions ...")
    rows, cand, n_tours = choice.build(ex, countries=a.countries)
    rows, offsets, y = prepare(rows)
    log(f"{len(rows):,} rows · {len(offsets):,} occasions · {n_tours:,} tours · "
        f"{int(y.sum()):,} chosen")

    consts = design_constants(rows)
    results = {}
    for spec in ("A", "B"):
        log(f"fitting specification {spec} ...")
        results[spec] = fit_spec(rows, offsets, y, spec, a.verbose, consts)
        r = results[spec]
        log(f"   converged in {r['iterations']} iterations · "
            f"pseudo-R2 {r['pseudo_r2']:.3f} · SEs {r['se_kind']}")

    for spec, r in results.items():
        title = ("A — capacity carries its own effect plus every unobserved "
                 "reason the city is attractive (UPPER BOUND)" if spec == "A"
                 else "B — also controls for shows already hosted, which "
                      "absorbs part of the real effect (LOWER BOUND)")
        print(f"\n{'=' * 78}\nSpecification {title}\n{'=' * 78}")
        print(logit.summary(r).to_string(index=False))
        print()
        for line in interpret(r):
            print("  " + line)

    ca = results["A"]["beta"][results["A"]["names"].index("has a room big enough")]
    cb = results["B"]["beta"][results["B"]["names"].index("has a room big enough")]
    print(f"\n{'-' * 78}")
    print(f"HAVING A ROOM BIG ENOUGH multiplies the odds of being chosen by "
          f"between {np.exp(min(ca, cb)):.2f} and {np.exp(max(ca, cb)):.2f}.")
    print("The lower figure is what survives once past activity is controlled "
          "for;\nthe upper is what the raw association gives. The truth is "
          "inside that interval,\nand this data cannot say where.")
    print("-" * 78)

    if a.save:
        stamp = os.path.basename(a.extract or latest_extract())
        path = save_model(results, stamp, len(rows), len(offsets), n_tours,
                          set(rows["countryCode"]), consts)
        log(f"wrote {path}")

    if a.market:
        for spec in ("A", "B"):
            r = results[spec]
            if a.sweep:
                print(f"\n=== {a.market}: room size against expected gain "
                      f"({a.kind}, specification {spec}) ===")
                print(capacity_sweep(rows, offsets, r, spec, a.market,
                                     a.kind, country=a.country).to_string(index=False))
            elif a.capacity:
                cf = counterfactual(rows, offsets, r, spec, a.market,
                                    a.capacity, a.kind, a.country)
                print(f"\n=== {a.market}: a {a.capacity:,}-capacity {a.kind} room "
                      f"(specification {spec}) ===")
                print(f"expected tour-visits now      {cf['expected_visits_before']:.1f}")
                print(f"expected tour-visits after    {cf['expected_visits_after']:.1f}")
                print(f"difference                    {cf['extra_visits']:+.1f} "
                      f"over {cf['occasions']:,} choice occasions")
                print("\nTours whose probability moves most:")
                print(cf["per_tour"].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
