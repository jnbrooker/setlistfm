#!/usr/bin/env python3
"""
Putting a building that does not exist onto the menu, and reading what happens.

WHAT THIS FILE IS FOR

layer2.py can already answer "what if this market had a room of size X". It
answers it the thorough way: rebuild the whole choice table with the new room
in it and re-predict every occasion in the country. That is correct and it is
slow -- a copy of six hundred thousand rows per question -- which is fine for a
command line and hopeless for a screen with a slider on it.

This file is the same arithmetic arranged so a person can move the slider. It
is NOT a faster approximation. It returns the same numbers, and there is a
self-test at the bottom that checks it against layer2 to two decimal places,
because a shortcut that is nearly right is worse than no shortcut at all.

WHY IT CAN BE FASTER WITHOUT CUTTING A CORNER

Two facts about the conditional logit make this exact.

  1. ADDING A ROOM TOUCHES ONE MARKET'S ROWS. Catchment, income, routing and
     every other market's ceiling are untouched, so their utilities are
     unchanged. Recomputing them is arithmetic whose answer is already known.

  2. WITH THE DESIGN CONSTANTS FROZEN, EVERY COLUMN IS ROW-WISE. Once the
     medians, the centring means and the ceiling floor are fixed (see
     layer2.design_constants), each row's X depends on nothing but that row.
     So the design matrix for a subset of rows is exactly the corresponding
     slice of the design matrix for all of them.

Together they mean the whole update is: recompute X for the few thousand rows
belonging to the target market, splice their utilities into the vector, and
take the softmax again. Everything else is reused.

WHAT THE NUMBERS COMING OUT OF HERE MEAN

`expected visits` is the model's expected number of tour-visits to the market,
summed over every choice occasion in the country. Its "before" value is
therefore checkable against a fact -- how many times the market was actually
picked -- and `calibration()` returns exactly that comparison, because a model
whose before is wrong has no business being asked about an after.

The "after" value is what a city with that room AND OTHERWISE IDENTICAL
CHARACTERISTICS would be expected to get. It is not a forecast for this city.
Cities have big arenas because promoters already expected demand, so the
capacity coefficient carries both the room's own effect and every unobserved
reason the city was attractive enough to get one. That is why nothing here ever
returns a single number: every answer is the interval between specification A
(which lets capacity keep the credit) and specification B (which hands as much
of it as possible to past activity). The truth is inside. This data cannot say
where.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import choice                                   # noqa: E402
import layer2                                   # noqa: E402
import logit                                    # noqa: E402

# The sizes the sweep tries. Chosen to straddle the rungs European touring
# actually uses -- club, theatre, small hall, the 10-12k arena that is the
# commonest new build, and the stadium end -- rather than to be evenly spaced,
# because the curve is a step function and even spacing would land the steps in
# arbitrary places.
SWEEP_SIZES = [1_000, 1_500, 2_000, 3_000, 4_000, 5_000, 6_500, 8_000,
               10_000, 12_000, 15_000, 18_000, 20_000, 25_000, 35_000, 50_000]


# ---------------------------------------------------------------------------
# Loading the fitted model and building one country's menu
# ---------------------------------------------------------------------------

def fitted_model(stamp=None):
    """
    The saved coefficients, with the check that makes them usable.

    A model saved before design_constants existed has no record of what its
    centred columns were centred on. Applying its betas to a menu built here
    would then be wrong in a way that produces plausible numbers, which is the
    worst kind of wrong. So it is refused rather than guessed at.
    """
    blob = layer2.load_model(stamp)
    if blob is None:
        return None, ("No fitted model in models/. Run "
                      "`python layer2.py --save` to produce one.")
    if "design_constants" not in blob:
        return None, ("The saved model predates the design constants being "
                      "recorded with it, so its coefficients cannot safely be "
                      "applied to a single country's menu. Re-run "
                      "`python layer2.py --save`.")
    # A stamp mismatch is NOT fatal -- the coefficients still work and the app
    # still runs -- so it goes on the blob rather than into the error slot.
    # Callers treat a non-None error as fatal and abort, and a warning that
    # kills the page is worse than the thing it warns about.
    blob["warning"] = None
    if not blob.get("stamp_matched", True):
        blob["warning"] = (
            f"This extract is **{blob.get('stamp_requested')}**, but the only "
            f"fitted model available is `{blob.get('loaded_file')}`, fitted on "
            f"the **{blob.get('extract')}** extract. The coefficients come "
            f"from a different sample of tours than the menus they are applied "
            f"to, and nothing downstream can detect that. Re-run "
            f"`python layer2.py --save` against this extract.")
    return blob, None


def countries_available(ex):
    """Country codes with enough coverage to build a menu for."""
    cand = choice.candidate_markets(ex)
    return sorted(cand["countryCode"].unique())


def build_menu(ex, code, blob):
    """
    Everything needed to answer questions about one country, computed once.

    The design matrix and the utility vector are built here rather than per
    question because they are the expensive part and they never change: the
    only thing a proposed building alters is a slice of them.
    """
    rows, cand, n_tours = choice.build(ex, countries=[code], progress=False)
    rows, offsets, y = layer2.prepare(rows)
    rows["headliner"] = rows["tour"].map(
        ex["tours"].set_index("tour")["headliner"])

    consts = blob["design_constants"]
    specs = {}
    for spec, saved in blob["specs"].items():
        X, names = layer2.design(rows, spec, consts)
        if names != saved["names"]:
            raise ValueError(
                f"specification {spec} was fitted on columns {saved['names']} "
                f"but this build produced {names}. The extract and the model "
                f"are out of step -- refit with `python layer2.py --save`.")
        beta = np.asarray(saved["beta"], dtype=float)
        cov = saved.get("cov")
        specs[spec] = {"X": X, "beta": beta, "se": np.asarray(saved["se"]),
                       "cov": np.asarray(cov) if cov is not None else None,
                       "names": names, "v": X @ beta, "consts": consts,
                       "pseudo_r2": saved.get("pseudo_r2")}

    return {"countryCode": code, "warning": blob.get("warning"),
            "rows": rows, "offsets": offsets, "y": y,
            "markets": sorted(rows["market"].unique()), "specs": specs,
            "n_tours": int(rows["tour"].nunique()),
            "n_occasions": int(len(offsets)), "consts": consts}


def calibration(menu, spec):
    """
    Expected visits against actual visits, per market.

    The single most useful thing to be able to show before any counterfactual:
    the model's "before" is a prediction about a world we can see. Where it is
    badly wrong for a market, its "after" for that market should not be
    believed either, and this puts that judgement in the reader's hands rather
    than in a footnote.
    """
    rows, s = menu["rows"], menu["specs"][spec]
    p = logit.choice_probabilities(s["v"], menu["offsets"])
    out = (rows.assign(p=p)
           .groupby("market", as_index=False)
           .agg(expected=("p", "sum"), actual=("chosen", "sum")))
    out["actual"] = out["actual"].astype(int)
    out["expected"] = out["expected"].round(1)
    out["model over/under"] = (out["expected"] - out["actual"]).round(1)
    return out.sort_values("actual", ascending=False)


# ---------------------------------------------------------------------------
# The proposed building
# ---------------------------------------------------------------------------

def _target_mask(menu, market):
    rows = menu["rows"]
    t = rows["market"].astype(str).str.casefold() == str(market).strip().casefold()
    if not t.any():
        raise KeyError(
            f"{market!r} is not on the {menu['countryCode']} menu. A market "
            f"needs {choice.MIN_MARKET_EVENTS_PER_YEAR:g} shows a year and a measured catchment "
            f"to be somewhere a touring act was realistically weighing.")
    return t.values


def _rebuilt_utility(menu, spec, target, applies, capacity):
    """
    The utility vector with the new room in it.

    Only `applies` rows are recomputed -- see the module docstring for why that
    is exact rather than a shortcut -- and the design matrix slice is returned
    alongside, because the decomposition needs the before/after difference
    column by column.
    """
    rows, s = menu["rows"], menu["specs"][spec]
    sub = rows.loc[applies].copy()
    sub["ceiling"] = np.maximum(
        pd.to_numeric(sub["ceiling"], errors="coerce").fillna(0.0), float(capacity))

    # THE NEW ROOM JOINS THE LADDER, not just the ceiling.
    #
    # Without this, a proposed 20,000 arena and a proposed 3,000 hall would
    # score identically for an act that plays 3,000 -- both raise the ceiling
    # past its bar, and nothing else would move. The whole point of carrying
    # the ladder is that a RIGHT-SIZED room helps more than an oversized one,
    # and that only shows up if the counterfactual can make the new room the
    # closest rung.
    if "best_room" in sub:
        need = pd.to_numeric(sub["room_needed"], errors="coerce").clip(lower=50)
        old = pd.to_numeric(sub["best_room"], errors="coerce")
        d_old = (np.log(old.clip(lower=50)) - np.log(need)).abs()
        d_new = (np.log(max(float(capacity), 50.0)) - np.log(need)).abs()
        sub["best_room"] = np.where(d_new < d_old.fillna(np.inf),
                                    float(capacity), old)
    X_sub, _ = layer2.design(sub, spec, s["consts"])

    v_after = s["v"].copy()
    v_after[applies] = X_sub @ s["beta"]
    return v_after, X_sub


def impact(menu, spec, market, capacity, kind="indoor"):
    """
    What the model expects a room of this size and kind to change.

    `kind` matters and is not a formality. An indoor arena does nothing for an
    act that plays festival fields, and quietly crediting it with those acts is
    the commonest way a projection like this gets inflated. Acts whose indoor
    and outdoor dates tie are counted as helped by either, which is the
    generous reading and is flagged as such in the output.
    """
    rows, s = menu["rows"], menu["specs"][spec]
    target = _target_mask(menu, market)
    if kind in ("indoor", "outdoor"):
        applies = target & rows["act_plays"].isin([kind, "either"]).values
    else:
        applies = target

    v_after, X_sub = _rebuilt_utility(menu, spec, target, applies, capacity)
    p0 = logit.choice_probabilities(s["v"], menu["offsets"])
    p1 = logit.choice_probabilities(v_after, menu["offsets"])

    hit = rows.loc[target, ["tour", "headliner", "category", "act_plays",
                            "room_needed", "ceiling", "chosen"]].copy()
    hit["p_before"] = p0[target]
    hit["p_after"] = p1[target]
    hit["gain"] = hit["p_after"] - hit["p_before"]

    per_tour = (hit.groupby(["tour", "headliner", "category", "act_plays"],
                            as_index=False)
                .agg(occasions=("gain", "size"), room_it_uses=("room_needed", "max"),
                     p_before=("p_before", "sum"), p_after=("p_after", "sum"),
                     gain=("gain", "sum"))
                .sort_values("gain", ascending=False))

    # How often the market's existing room is already big enough for the act
    # in front of it, and how often it would be with the proposal built.
    # Measured over the SAME rows in both cases -- the occasions the proposed
    # room could possibly help -- because a before and an after computed over
    # different denominators is not a comparison.
    ceil_now = pd.to_numeric(rows.loc[applies, "ceiling"], errors="coerce").fillna(0.0)
    need = pd.to_numeric(rows.loc[applies, "room_needed"], errors="coerce")
    testable = need.notna()
    if testable.any():
        before_share = float((ceil_now[testable] >= need[testable]).mean())
        after_share = float((np.maximum(ceil_now[testable], float(capacity))
                             >= need[testable]).mean())
    else:
        before_share = after_share = float("nan")

    return {
        "market": market, "countryCode": menu["countryCode"],
        "capacity": int(capacity), "kind": kind, "spec": spec,
        "expected_before": float(p0[target].sum()),
        "expected_after": float(p1[target].sum()),
        "extra_visits": float(p1[target].sum() - p0[target].sum()),
        "actual_visits": int(rows.loc[target, "chosen"].sum()),
        "occasions": int(target.sum()),
        "occasions_helped": int(applies.sum()),
        "wrong_kind": int(target.sum() - applies.sum()),
        "either_kind": int((target & (rows["act_plays"] == "either").values).sum()),
        "cleared_before": before_share,
        "cleared_after": after_share,
        "per_tour": per_tour,
        "tours_moved": int((per_tour["gain"] > 0.01).sum()),
        "_target": target, "_applies": applies, "_X_sub": X_sub,
        "_p0": p0, "_p1": p1,
    }


def why(menu, spec, imp, top=20):
    """
    The arithmetic behind the headline, variable by variable and tour by tour.

    HOW THIS DECOMPOSITION WORKS, AND WHY IT IS EXACT RATHER THAN ATTRIBUTED

    Adding a room changes two things about a market: how big its biggest room
    is, and whether that room now clears each act's bar. Nothing else moves. So
    the change in the market's utility for a given tour is

        du  =  sum over variables of (change in that variable) x (its coefficient)

    and that sum is the total exactly, with nothing left over. It is not a
    heuristic attribution of the kind that has to be apologised for.

    Probability is the softmax of utility and is NOT additive: the same utility
    gain takes a market from 2% to 4% but from 40% to only 45%, because it has
    to take the probability from somewhere. So both are returned -- the exact
    additive story in utility, and the probability change that actually
    follows from it. Showing only the first overstates; only the second hides
    the reason.
    """
    rows, s = menu["rows"], menu["specs"][spec]
    target, applies = imp["_target"], imp["_applies"]

    dX = np.zeros((int(target.sum()), len(s["names"])))
    # rows that changed, expressed as positions within the target block
    moved_in_target = applies[target]
    dX[moved_in_target] = imp["_X_sub"] - s["X"][applies]
    contrib = dX * s["beta"][None, :]

    hit = rows.loc[target, ["tour", "headliner", "category", "act_plays",
                            "room_needed", "ceiling"]].copy()
    hit["p_before"] = imp["_p0"][target]
    hit["p_after"] = imp["_p1"][target]
    hit["gain"] = hit["p_after"] - hit["p_before"]
    hit["utility gain"] = contrib.sum(axis=1)
    for j, nm in enumerate(s["names"]):
        if np.any(np.abs(contrib[:, j]) > 1e-12):
            hit[f"from: {nm}"] = contrib[:, j]

    # Gain-weighted, so a variable that only moves tours the market was never
    # going to get does not take credit for the headline number.
    w = hit["gain"].clip(lower=0).values
    share = {nm: float(np.sum(contrib[:, j] * w))
             for j, nm in enumerate(s["names"])
             if np.any(np.abs(contrib[:, j]) > 1e-12)}
    total = sum(share.values())
    tbl = pd.DataFrame(
        [{"variable": k,
          "utility contributed": round(v, 4),
          "share of the reason": (v / total) if total else None,
          "what it is": layer2.VARIABLE_NOTES.get(k, "")}
         for k, v in sorted(share.items(), key=lambda kv: -abs(kv[1]))])

    movers = (hit[hit["gain"] > 1e-6]
              .sort_values("gain", ascending=False)
              .head(top))
    return {"why": tbl, "movers": movers,
            "median_gain_pp": float(hit.loc[hit["gain"] > 1e-6, "gain"].median() * 100)
            if (hit["gain"] > 1e-6).any() else 0.0}


def sweep(menu, spec, market, kind="indoor", sizes=None):
    """
    Expected gain at each candidate size -- the actual answer to "how big".

    The curve is a STEP function, and that is the finding rather than an
    artefact. What the model responds to is clearing each act's bar, so the
    gains arrive in jumps as the proposed room passes the sizes touring acts
    really use, and the flat stretches between are seats that buy nothing. A
    smooth curve here would mean the model had been asked the wrong question.
    """
    sizes = sizes or SWEEP_SIZES
    rows, s = menu["rows"], menu["specs"][spec]
    target = _target_mask(menu, market)
    if kind in ("indoor", "outdoor"):
        applies = target & rows["act_plays"].isin([kind, "either"]).values
    else:
        applies = target
    p0 = logit.choice_probabilities(s["v"], menu["offsets"])
    base = float(p0[target].sum())

    # Gains are counted per TOUR, not per occasion. A tour that plays four
    # cities in the country generates four occasions, and a room that nudges
    # each of them by half a point has moved one act, not four.
    tours = rows.loc[target, "tour"].values

    out = []
    for cap in sizes:
        v_after, _ = _rebuilt_utility(menu, spec, target, applies, cap)
        p1 = logit.choice_probabilities(v_after, menu["offsets"])
        gain = pd.Series(p1[target] - p0[target]).groupby(tours).sum()
        out.append({"capacity": int(cap),
                    "expected extra visits": float(p1[target].sum() - base),
                    "tours materially helped": int((gain > 0.01).sum())})
    df = pd.DataFrame(out)
    df["extra per 1,000 seats"] = (
        df["expected extra visits"].diff() / (df["capacity"].diff() / 1000))
    return df


# ---------------------------------------------------------------------------
# The Layer 1 half: which blocked tours a room of this size would clear
#
# Kept deliberately separate from everything above, and computed by counting
# rather than by the model. It answers a narrower question -- of the tours the
# descriptive layer found were blocked by room size, how many would this room
# physically accommodate -- and it would survive the model being wrong, which
# is exactly why it is worth having beside a model output.
# ---------------------------------------------------------------------------

def clears(blocked, capacity, kind=None):
    """
    Of the blocked tours, which this room is physically big enough for.

    `room_it_needs` is the median room the act actually played elsewhere in the
    country, so "clears" means the act has demonstrably worked in a room this
    size. It does NOT mean the act would come: removing the obstacle is
    necessary, not sufficient, and the model above is the only part of this
    that speaks to sufficiency.
    """
    if blocked is None or blocked.empty:
        return pd.DataFrame(), 0, 0
    b = blocked.copy()
    if kind in ("indoor", "outdoor"):
        b = b[b["plays"].isin([kind, "either"])]
    if b.empty:
        return b, 0, 0
    need = pd.to_numeric(b["room_it_needs"], errors="coerce")
    b["fits in the proposed room"] = need <= float(capacity)
    b["seats short"] = (need - float(capacity)).clip(lower=0).round(0)
    return (b.sort_values("room_it_needs", ascending=False),
            int(b["fits in the proposed room"].sum()), int(len(b)))


# ---------------------------------------------------------------------------
# Reading an interval out loud
# ---------------------------------------------------------------------------

def interval(a, b, unit="tour-visits"):
    """
    The pair of specifications as one sentence.

    Never a midpoint, and never a single number. The gap between A and B is not
    noise to be averaged away -- it is a direct measure of how much of the
    association is credibly the building and how much is the city that built
    it. Collapsing it to a point would throw away the only honest thing this
    model has to say.
    """
    lo, hi = (a, b) if a <= b else (b, a)
    if hi <= 0.05:
        return f"no material change ({lo:+.1f} to {hi:+.1f} {unit})"
    spread = (hi / lo) if lo > 0.05 else np.inf
    if np.isfinite(spread) and spread < 2:
        strength = ("The two specifications broadly agree, so the finding "
                    "survives the selection worry.")
    elif np.isfinite(spread) and spread < 4:
        strength = ("The two specifications differ by a factor of "
                    f"{spread:.1f}, so how much of this is the room and how "
                    "much is the city is genuinely unsettled.")
    else:
        strength = ("The two specifications are far apart, which means almost "
                    "all of the apparent effect could be the city rather than "
                    "the room. Treat the upper figure as an upper bound only.")
    return f"{lo:+.1f} to {hi:+.1f} {unit}. {strength}"


# ---------------------------------------------------------------------------
# Saying what the counterfactual actually means
#
# "+5.9 to +20.8 tour-visits" is not an answer anyone can use. It has no time
# frame, no baseline, and "tour-visit" is a word this project invented. The
# three functions below turn the same arithmetic into a sentence and two small
# tables, which is what the number was always for.
# ---------------------------------------------------------------------------

def years_covered(ex):
    """How long the extract spans, for turning totals into a rate."""
    try:
        lo, hi = ex["manifest"]["event_date_range"]
        return max((pd.Timestamp(hi) - pd.Timestamp(lo)).days / 365.25, 0.25)
    except Exception:
        return None


def describe(a, b, years=None):
    """
    The headline in words, with a time frame and a baseline.

    A total over three and a half years means nothing without both. The rate
    per year is what a reader can hold, and the comparison against what the
    market ACTUALLY got is what makes the size of the claim visible.
    """
    lo, hi = sorted((a["extra_visits"], b["extra_visits"]))
    actual = a["actual_visits"]
    out = {"lo": lo, "hi": hi, "actual": actual}

    if hi <= 0.05:
        out["headline"] = "No material change"
        out["sub"] = ("A room this size does not clear any bar the market does "
                      "not already clear, so the model expects nothing from it.")
        return out

    out["headline"] = (f"Between {lo:.0f} and {hi:.0f} more touring acts would "
                       f"have come")
    bits = []
    if years:
        bits.append(f"over the {years:.1f} years this data covers — roughly "
                    f"**{lo / years:.0f} to {hi / years:.0f} a year**")
    if actual:
        bits.append(f"against the **{actual:,}** that actually came in that "
                    f"time, so a lift of **{lo / actual:+.0%} to "
                    f"{hi / actual:+.0%}**")
    out["sub"] = ", ".join(bits) + "." if bits else ""
    out["per_year"] = (lo / years, hi / years) if years else None
    out["lift"] = (lo / actual, hi / actual) if actual else None
    return out


# Short, readable labels for the model's columns. The full sentences live in
# layer2.VARIABLE_NOTES; these are what fits in a table cell.
SHORT_NOTES = {
    "log largest room":
        "Bigger rooms get picked more often, whoever is touring.",
    "has a room big enough":
        "The room clears the size THIS act works in. The policy variable.",
    "room big enough x log room needed":
        "Clearing the bar matters more for bigger acts.",
}


def moving_parts(menu, spec, imp):
    """
    Two tables: what the building changes, and what that does to the odds.

    Both are read off the design matrix rather than recomputed, so they cannot
    drift from the number in the headline. The first is in seats and shares --
    things a person can picture. The second is the model's response to them.
    """
    s = menu["specs"][spec]
    rows = menu["rows"]
    applies = imp["_applies"]

    dX = imp["_X_sub"] - s["X"][applies]
    beta = s["beta"]

    # --- what the building physically changes ------------------------------
    # The `ceiling` column holds the ceiling of the kind the ACT plays, so on
    # an "either" act's row it is the market's overall ceiling -- which for
    # Naples is a 52,530 football stadium. Taking the max across all applicable
    # rows would therefore report the stadium as the indoor room. Restrict to
    # acts of exactly this kind, for whom the column is the ceiling asked about.
    ceil = pd.to_numeric(rows.loc[applies, "ceiling"], errors="coerce").fillna(0.0)
    exact = (rows.loc[applies, "act_plays"] == imp["kind"]).values
    ceil_now = ceil[exact].max() if exact.any() else np.nan

    physical = pd.DataFrame([
        {"what changes": f"Biggest {imp['kind']} room",
         "before": f"{ceil_now:,.0f}" if np.isfinite(ceil_now) else "none",
         "after": f"{imp['capacity']:,}"},
        {"what changes": "Big enough for the act",
         "before": f"{imp['cleared_before']:.0%}",
         "after": f"{imp['cleared_after']:.0%}"},
    ])

    # --- what the model does with them -------------------------------------
    out = []
    for j, name in enumerate(s["names"]):
        d = dX[:, j]
        if not np.any(np.abs(d) > 1e-12):
            continue
        moved = d[np.abs(d) > 1e-12]
        typical = float(np.mean(moved))
        out.append({
            "variable": name,
            "coefficient": round(float(beta[j]), 3),
            "typical change": round(typical, 3),
            "odds multiplier": round(float(np.exp(beta[j] * typical)), 3),
            "applies to": f"{len(moved):,} of {len(d):,} occasions",
            "what it is": SHORT_NOTES.get(name, ""),
        })
    response = pd.DataFrame(out).sort_values("odds multiplier", ascending=False)

    unchanged = [n for n in s["names"]
                 if not np.any(np.abs(dX[:, s["names"].index(n)]) > 1e-12)]
    return {"physical": physical, "response": response, "unchanged": unchanged}


# ---------------------------------------------------------------------------
# Is this place getting more or fewer tours than it should?
#
# THE QUESTION, AND WHY SPECIFICATION A IS THE ONLY ONE THAT CAN ANSWER IT
#
# The model predicts how often each market gets picked from its demographics
# (catchment, spending power), its infrastructure (the biggest room of the kind
# each act plays, and whether that room is big enough for them) and where it
# sits relative to the tour's other stops. Compare that prediction against what
# actually happened and the residual is the answer: a market that gets fewer
# visits than its fundamentals predict is under-toured, and one that gets more
# is doing better than its fundamentals explain.
#
# This only works on specification A. Specification B adds `log shows already
# hosted`, which is very nearly the outcome being predicted -- it would fit the
# actual count well by construction and leave a residual that means nothing.
# So everything below is A, and A alone.
#
# WHAT A LARGE RESIDUAL DOES AND DOES NOT MEAN
#
# It means one of two things, and this data cannot separate them:
#
#   the market is genuinely under-served -- the audience and the rooms are
#   there and the tours are going elsewhere; or
#
#   the model is missing something about it -- no local promoter, a hostile
#   calendar, a border, a rail line that does not exist, an audience that does
#   not buy tickets for this kind of show.
#
# Both are worth knowing and neither is settled here. A residual is a question
# to take to someone who knows the market, not a finding.
# ---------------------------------------------------------------------------

PERF_SPEC = "A"


def performance(menu, spec=PERF_SPEC):
    """
    Expected against actual tour-visits for every market on the menu.

    `expected` is what the demographics, the infrastructure and the routing
    predict. `actual` is what happened. Everything else is those two read
    different ways.
    """
    rows, s = menu["rows"], menu["specs"][spec]
    p = logit.choice_probabilities(s["v"], menu["offsets"])

    g = (rows.assign(p=p)
         .groupby("market", as_index=False)
         .agg(expected=("p", "sum"), actual=("chosen", "sum"),
              on_the_menu=("p", "size"),
              catchment=("catchment", "first"),
              income=("income", "first"),
              indoor_ceiling=("ceiling_indoor", "first"),
              shows_hosted=("market_events", "first")))
    g["actual"] = g["actual"].astype(int)
    g["gap"] = g["actual"] - g["expected"]
    # Ratio as well as difference: a shortfall of 20 means something very
    # different for a market expected to get 300 than for one expected to get 25.
    g["actual vs expected"] = g["actual"] / g["expected"].where(g["expected"] > 0)
    g["verdict"] = np.select(
        [g["actual vs expected"] >= 1.5, g["actual vs expected"] >= 1.15,
         g["actual vs expected"] <= 0.67, g["actual vs expected"] <= 0.87],
        ["far more than expected", "more than expected",
         "far fewer than expected", "fewer than expected"],
        default="about as expected")
    return g.sort_values("actual vs expected")


VERDICT_ORDER = ["far fewer than expected", "fewer than expected",
                 "about as expected", "more than expected",
                 "far more than expected"]


def why_expected(menu, market, spec=PERF_SPEC):
    """
    What makes the model expect this many visits, variable by variable.

    Read as a comparison against the average alternative on the same menus.
    A market is picked more often than the average candidate city to the extent
    that its attributes are better than that city's, so the honest
    decomposition is (this market's mean attribute - the menu's mean attribute)
    x the coefficient. The exponential of each term is how much that one
    variable multiplies this market's odds relative to a typical rival.

    The terms add up, in utility, to the market's total advantage. They do not
    add up in probability, because probability is a softmax and has to take
    from somewhere -- which is why the odds column is labelled a multiplier and
    not a share.
    """
    s = menu["specs"][spec]
    t = _target_mask(menu, market)
    X = s["X"]

    here = X[t].mean(axis=0)
    rival = X.mean(axis=0)
    diff = here - rival
    contrib = diff * s["beta"]

    tbl = pd.DataFrame({
        "variable": s["names"],
        "this market": np.round(here, 3),
        "average rival": np.round(rival, 3),
        "coefficient": np.round(s["beta"], 3),
        "utility": np.round(contrib, 3),
        "odds x": np.round(np.exp(contrib), 2),
        "what it is": [SHORT_NOTES.get(n, layer2.VARIABLE_NOTES.get(n, "")[:90])
                       for n in s["names"]],
    })
    return tbl.sort_values("utility", key=lambda c: c.abs(), ascending=False)


# ---------------------------------------------------------------------------
# Layer 4, in the one form this data actually supports
#
# The coefficients are estimates with a covariance, so every number derived
# from them inherits a distribution. Drawing coefficients from that covariance
# and re-running the arithmetic turns a point estimate into a spread.
#
# WHAT THIS SIMULATION COVERS, AND WHAT IT EMPHATICALLY DOES NOT
#
# It covers ESTIMATION uncertainty only: how much the answer would move if the
# same model were fitted to another sample of tours drawn the same way. That is
# the narrowest of the uncertainties in this project and by some distance the
# least important.
#
# It does NOT cover the selection problem -- whether capacity causes shows or
# merely accompanies them -- which is what the gap between specifications A and
# B measures, and which is much wider. Nor does it cover being wrong about the
# form of the model, the choice set, or the data.
#
# So a tight band here means "the sample is large", not "we are confident".
# Anywhere this is shown, the A-to-B interval is shown beside it, and the
# difference in width between the two is the point.
# ---------------------------------------------------------------------------

def _market_codes(menu):
    if "_codes" not in menu:
        codes, uniq = pd.factorize(menu["rows"]["market"], sort=True)
        menu["_codes"], menu["_code_names"] = codes, list(uniq)
    return menu["_codes"], menu["_code_names"]


def simulate_expected(menu, draws=300, spec=PERF_SPEC, seed=0):
    """
    Expected visits per market under `draws` samples of the coefficients.

    Returns (names, matrix) with one row per draw and one column per market,
    so any percentile of interest can be taken afterwards.
    """
    s = menu["specs"][spec]
    if s.get("cov") is None:
        raise ValueError(
            "the saved model has no covariance matrix, so uncertainty cannot "
            "be propagated. Re-run `python layer2.py --save`.")
    rng = np.random.default_rng(seed)
    B = rng.multivariate_normal(np.asarray(s["beta"]), np.asarray(s["cov"]),
                                size=int(draws))
    codes, names = _market_codes(menu)
    n = len(names)

    out = np.empty((int(draws), n))
    for i, b in enumerate(B):
        p = logit.choice_probabilities(s["X"] @ b, menu["offsets"])
        out[i] = np.bincount(codes, weights=p, minlength=n)
    return names, out


def expected_bands(menu, draws=300, spec=PERF_SPEC, seed=0, lo=5, hi=95):
    """
    Per-market expected visits with a band, and whether actual falls outside it.

    A market whose actual count sits outside the band is off-trend by more than
    the coefficients' own wobble. That is a necessary condition for calling it
    under- or over-toured, not a sufficient one -- see the note at the top of
    this section.
    """
    names, sims = simulate_expected(menu, draws, spec, seed)
    perf = performance(menu, spec).set_index("market")
    band = pd.DataFrame({
        "market": names,
        "expected": sims.mean(axis=0).round(1),
        f"p{lo}": np.percentile(sims, lo, axis=0).round(1),
        f"p{hi}": np.percentile(sims, hi, axis=0).round(1),
    })
    band["actual"] = band["market"].map(perf["actual"]).astype(int)
    band["outside the band"] = np.where(
        band["actual"] < band[f"p{lo}"], "fewer than the model can explain",
        np.where(band["actual"] > band[f"p{hi}"],
                 "more than the model can explain", "inside"))
    band["actual vs expected"] = (band["actual"]
                                  / band["expected"].where(band["expected"] > 0))
    return band.sort_values("actual vs expected")


def simulate_impact(menu, market, capacity, kind="indoor", draws=300,
                    spec="A", seed=0):
    """
    The distribution of extra visits a proposed room would bring.

    Same shortcut as `impact`: the design matrix is rebuilt once for the rows
    the building touches, and only the coefficients are redrawn. So the cost is
    one softmax per draw rather than one rebuild per draw.
    """
    s = menu["specs"][spec]
    if s.get("cov") is None:
        raise ValueError("the saved model has no covariance matrix; re-run "
                         "`python layer2.py --save`.")
    rows = menu["rows"]
    target = _target_mask(menu, market)
    if kind in ("indoor", "outdoor"):
        applies = target & rows["act_plays"].isin([kind, "either"]).values
    else:
        applies = target

    sub = rows.loc[applies].copy()
    sub["ceiling"] = np.maximum(
        pd.to_numeric(sub["ceiling"], errors="coerce").fillna(0.0), float(capacity))
    X_sub, _ = layer2.design(sub, spec, s["consts"])

    rng = np.random.default_rng(seed)
    B = rng.multivariate_normal(np.asarray(s["beta"]), np.asarray(s["cov"]),
                                size=int(draws))
    out = np.empty(int(draws))
    for i, b in enumerate(B):
        v0 = s["X"] @ b
        v1 = v0.copy()
        v1[applies] = X_sub @ b
        p0 = logit.choice_probabilities(v0, menu["offsets"])
        p1 = logit.choice_probabilities(v1, menu["offsets"])
        out[i] = float(p1[target].sum() - p0[target].sum())
    return out


# ---------------------------------------------------------------------------
# Self-test
#
# The claim this file makes is that its shortcut is exact. That claim is worth
# nothing unless it is checked against the slow path it replaces, so:
#
#     python whatif.py
#
# builds one country, runs the same counterfactual both ways, and fails loudly
# if they disagree.
# ---------------------------------------------------------------------------

def _self_test(code="IT", market="Naples", capacity=13_500, kind="indoor"):
    from gap import latest_extract, load_extract

    ex = load_extract(latest_extract())
    blob, err = fitted_model()
    if err:
        raise SystemExit(err)
    menu = build_menu(ex, code, blob)
    print(f"{code}: {len(menu['rows']):,} rows · {menu['n_occasions']:,} "
          f"occasions · {menu['n_tours']:,} tours")

    for spec in ("A", "B"):
        s = menu["specs"][spec]
        fast = impact(menu, spec, market, capacity, kind)

        slow_res = {"beta": s["beta"], "X": s["X"], "names": s["names"],
                    "consts": s["consts"]}
        slow = layer2.counterfactual(menu["rows"], menu["offsets"], slow_res,
                                     spec, market, capacity, kind, code)

        d_before = abs(fast["expected_before"] - slow["expected_visits_before"])
        d_extra = abs(fast["extra_visits"] - slow["extra_visits"])
        print(f"  spec {spec}: fast {fast['extra_visits']:+.4f} · "
              f"slow {slow['extra_visits']:+.4f} · "
              f"diff {d_extra:.2e} (before {d_before:.2e})")
        assert d_before < 1e-6, "the 'before' figures disagree"
        assert d_extra < 1e-6, "the shortcut is not exact"

        w = why(menu, spec, fast)
        recon = w["why"]["utility contributed"].abs().sum()
        assert recon > 0, "decomposition found no moving variable"

    a = impact(menu, "A", market, capacity, kind)["extra_visits"]
    b = impact(menu, "B", market, capacity, kind)["extra_visits"]
    print(f"\n  {market}, a {capacity:,}-seat {kind} room: {interval(a, b)}")
    return True


if __name__ == "__main__":
    print("checking the fast path against layer2's full rebuild\n")
    _self_test()
    print("\nPASS — identical to the slow path")
