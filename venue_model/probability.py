#!/usr/bin/env python3
"""
Probability as the thing on the screen, rather than counts derived from it.

WHY THIS FILE EXISTS SEPARATELY

Every other screen in this project reports a COUNT -- expected visits, extra
tour-visits, seats below peers. Counts are what a promoter argues about, but
they are all sums of the same underlying quantity, and summing first hides what
the model actually produces:

    P(this tour picks this city) = exp(x'B) / sum over every city it could pick

That probability is the model. A count is what you get by adding a few hundred
of them together, and the addition throws away the shape -- whether a city's
twelve expected visits are twelve near-certainties or two hundred long shots.
Those are very different propositions for anyone deciding what to build, and
only the distribution tells them apart.

So this file exposes the probabilities themselves, three ways:

    per tour, for one city      which acts are actually in play here
    per city, for one tour      where this act is likely to go instead
    before and after a room     how the probability moves, act by act

WHAT DOES NOT CHANGE BY LOOKING AT IT THIS WAY

The identification problem. A probability is no more causal than the count it
sums to: cities have big rooms because promoters expected demand, so the
capacity coefficient carries both effects and every number here inherits that.
Specification A and B bracket it exactly as they do everywhere else, and
nothing in this file collapses the two into one figure.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import layer2                                                    # noqa: E402
import logit                                                     # noqa: E402
import whatif                                                    # noqa: E402

def _modal_kind(series):
    """
    The kind of room an act mostly plays, across its occasions.

    `act_plays` is computed leave-one-out, so it can differ between a tour's
    own occasions -- drop a different stop and the indoor/outdoor tally can
    flip. It therefore cannot be a grouping key; it has to be reduced to one
    value per tour.
    """
    v = series.value_counts()
    if v.empty:
        return "either"
    top = v.max()
    tied = set(v[v == top].index)
    return "either" if len(tied) > 1 else v.idxmax()


# The columns that genuinely identify a tour. NOT room_needed or act_plays:
# both are leave-one-out figures that vary between a tour's own occasions, and
# grouping on them split Simple Minds' Global Tour 2024 into four separate rows
# with four different probabilities -- one tour appearing four times in a list
# that is supposed to hold one row per act.
TOUR_KEYS = ["tour", "headliner", "category"]


# A tour below this probability is not realistically choosing this city. Kept
# as a named constant because it decides what "in play" means on every screen
# that uses it, and a threshold buried in a filter is a threshold nobody can
# argue with.
IN_PLAY = 0.02


# ---------------------------------------------------------------------------
# One city, every tour
# ---------------------------------------------------------------------------

def tours_for_city(menu, spec, market, min_p=0.0):
    """
    Every tour's probability of picking this city, and what it would need.

    A tour that toured the country twice has two choice occasions, so its
    probability of appearing here at all is summed across them -- which is why
    the column can exceed 1 and is named `expected visits` rather than
    `probability`. `best occasion` keeps the single highest, which is the
    honest "chance this act comes at all".
    """
    rows, s = menu["rows"], menu["specs"][spec]
    target = whatif._target_mask(menu, market)
    p = logit.choice_probabilities(s["v"], menu["offsets"])

    sub = rows.loc[target].copy()
    sub["p"] = p[target]
    out = (sub.groupby(TOUR_KEYS, as_index=False)
           .agg(**{"expected visits": ("p", "sum"),
                   "best occasion": ("p", "max"),
                   "occasions": ("p", "size"),
                   "room_needed": ("room_needed", "median"),
                   "act_plays": ("act_plays", _modal_kind)}))
    out["in play"] = out["best occasion"] >= IN_PLAY
    return out.sort_values("expected visits", ascending=False)


def probability_shape(tours):
    """
    How a city's expected visits are made up -- a few likely acts or many long shots.

    Two cities can expect the same number of visits from completely different
    distributions, and the difference matters: twelve visits from fifteen
    near-certain acts is a healthy market, twelve from four hundred long shots
    is a city that occasionally gets lucky. The count alone cannot tell them
    apart, which is the whole argument for this file.
    """
    p = pd.to_numeric(tours["best occasion"], errors="coerce").dropna()
    total = float(tours["expected visits"].sum())
    bands = [(0.50, 1.01, "better than even"), (0.20, 0.50, "one in two to one in five"),
             (0.05, 0.20, "one in five to one in twenty"),
             (IN_PLAY, 0.05, "long shots"), (0.0, IN_PLAY, "not in play")]
    rows = []
    for lo, hi, label in bands:
        sel = tours[(p >= lo) & (p < hi)]
        rows.append({"likelihood": label, "tours": len(sel),
                     "expected visits": round(float(sel["expected visits"].sum()), 2)})
    d = pd.DataFrame(rows)
    d["share of expected visits"] = (
        d["expected visits"] / total).round(3) if total else np.nan
    return d


# ---------------------------------------------------------------------------
# One tour, every city
# ---------------------------------------------------------------------------

def cities_for_tour(menu, spec, tour, limit=25):
    """
    Where this act is likely to play, ranked -- the competitive picture.

    This is the view a promoter recognises: not "what are our chances" but
    "who are we losing to". Because the alternatives on a menu share one
    denominator, a city can only gain probability by taking it from the others,
    and this shows exactly who currently holds it.

    RANKED ON THE PER-OCCASION CHANCE, NOT ON SUMMED EXPECTED VISITS, and the
    difference is not cosmetic. Each occasion's menu excludes the tour's other
    stops, so a city the act PLAYED appears on exactly one menu while a city it
    skipped appears on all of them. Summing therefore rewards cities the act
    never went to: on a 25-stop Italian tour it put Bari first on 0.21 and Rome
    fifth on 0.73, which is precisely backwards. The mean over the menus a city
    actually appeared on is comparable across cities.

    One asymmetry survives and is worth knowing. A city the act played is
    measured on the single occasion it won, so its figure is conditioned on
    winning and reads high. `played` marks those rows; the directly
    interpretable numbers are the ones where it is False -- the act had this
    chance and did not take it.
    """
    rows, s = menu["rows"], menu["specs"][spec]
    p = logit.choice_probabilities(s["v"], menu["offsets"])
    sel = (rows["tour"] == tour).values
    if not sel.any():
        raise KeyError(f"{tour!r} is not on the {menu['countryCode']} menu")

    sub = rows.loc[sel].copy()
    sub["p"] = p[sel]
    out = (sub.groupby("market", as_index=False)
           .agg(chance=("p", "mean"), best=("p", "max"),
                menus=("p", "size"),
                catchment=("catchment", "first"),
                ceiling=("ceiling", "first"),
                played=("chosen", "max")))
    out["played"] = out["played"].astype(bool)
    return out.sort_values("chance", ascending=False).head(limit)


def tour_detail(menu, tour):
    """The act's own numbers, which are constant across the menu it faces."""
    rows = menu["rows"]
    sub = rows[rows["tour"] == tour]
    if sub.empty:
        raise KeyError(tour)
    r = sub.iloc[0]
    return {
        "tour": tour, "headliner": r.get("headliner"),
        "category": r.get("category"), "plays": r.get("act_plays"),
        "room_needed": r.get("room_needed"),
        "stops_in_country": int(r.get("stops_in_country") or 0),
        "occasions": int(sub["occasion"].nunique()),
        "cities_on_the_menu": int(sub["market"].nunique()),
    }


# ---------------------------------------------------------------------------
# Before and after a proposed room
# ---------------------------------------------------------------------------

def with_new_room(menu, spec, market, capacity, kind="indoor"):
    """
    Probabilities before and after, for every tour, in one frame.

    Returns the same per-tour shape as tours_for_city with a matching `after`
    column, so a chart can show the movement rather than two separate totals.
    Uses whatif's exact rebuild, so these are the same numbers the counting
    screens report -- not a second implementation that could drift.
    """
    rows, s = menu["rows"], menu["specs"][spec]
    target = whatif._target_mask(menu, market)
    if kind in ("indoor", "outdoor"):
        applies = target & rows["act_plays"].isin([kind, "either"]).values
    else:
        applies = target

    v_after, _ = whatif._rebuilt_utility(menu, spec, target, applies, capacity)
    p0 = logit.choice_probabilities(s["v"], menu["offsets"])
    p1 = logit.choice_probabilities(v_after, menu["offsets"])

    sub = rows.loc[target].copy()
    sub["p_before"], sub["p_after"] = p0[target], p1[target]
    out = (sub.groupby(TOUR_KEYS, as_index=False)
           .agg(before=("p_before", "sum"), after=("p_after", "sum"),
                best_before=("p_before", "max"), best_after=("p_after", "max"),
                room_needed=("room_needed", "median"),
                act_plays=("act_plays", _modal_kind)))
    out["change"] = out["after"] - out["before"]
    # Where the room clears the act's bar. This is what the model is actually
    # responding to, so it explains every non-trivial row in the chart.
    need = pd.to_numeric(out["room_needed"], errors="coerce")
    out["room clears its bar"] = need <= float(capacity)
    return out.sort_values("change", ascending=False)


def tour_with_new_room(menu, spec, market, tour, capacity, kind="indoor"):
    """One act's probability of this city, before and after."""
    d = with_new_room(menu, spec, market, capacity, kind)
    hit = d[d["tour"] == tour]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return {"before": float(r["before"]), "after": float(r["after"]),
            "change": float(r["change"]),
            "best_before": float(r["best_before"]),
            "best_after": float(r["best_after"]),
            "clears": bool(r["room clears its bar"])}


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

def simulate_city(menu, market, spec="A", capacity=None, kind="indoor",
                  draws=300, seed=0):
    """
    Expected visits to one city across coefficient draws, before and after.

    WHAT THE SPREAD HERE COVERS, AND WHAT IT LEAVES OUT

    Only estimation uncertainty: how far the answer would move if the same
    model were fitted to another sample of tours drawn the same way. That is
    the narrowest uncertainty in this project and by some way the least
    important.

    It does NOT cover the selection problem -- whether capacity causes shows or
    merely accompanies them -- which is what the gap between specifications A
    and B measures and which is much wider. A tight band here means the sample
    is large, not that the answer is safe, and any screen showing this band
    should show the A-to-B interval beside it.
    """
    s = menu["specs"][spec]
    if s.get("cov") is None:
        raise ValueError("the saved model has no covariance matrix; re-run "
                         "`python layer2.py --save`.")
    rows = menu["rows"]
    target = whatif._target_mask(menu, market)

    X_sub, applies = None, None
    if capacity:
        if kind in ("indoor", "outdoor"):
            applies = target & rows["act_plays"].isin([kind, "either"]).values
        else:
            applies = target
        sub = rows.loc[applies].copy()
        sub["ceiling"] = np.maximum(
            pd.to_numeric(sub["ceiling"], errors="coerce").fillna(0.0),
            float(capacity))
        X_sub, _ = layer2.design(sub, spec, s["consts"])

    rng = np.random.default_rng(seed)
    B = rng.multivariate_normal(np.asarray(s["beta"]), np.asarray(s["cov"]),
                                size=int(draws))
    before = np.empty(int(draws))
    after = np.empty(int(draws)) if capacity else None
    for i, b in enumerate(B):
        v0 = s["X"] @ b
        p0 = logit.choice_probabilities(v0, menu["offsets"])
        before[i] = float(p0[target].sum())
        if capacity:
            v1 = v0.copy()
            v1[applies] = X_sub @ b
            p1 = logit.choice_probabilities(v1, menu["offsets"])
            after[i] = float(p1[target].sum())
    return before, after


def drivers(menu, spec, market):
    """
    What makes the model expect this many visits, variable by variable.

    A thin wrapper over whatif.why_expected so the probability screen reads
    from the same decomposition the performance screen does -- two
    implementations of "why" would drift, and the first symptom would be two
    screens quietly disagreeing about the same city.
    """
    return whatif.why_expected(menu, market, spec)


# ---------------------------------------------------------------------------
# What the act actually did, and where a city would now sit in its plans
# ---------------------------------------------------------------------------

def tour_itinerary(ex, tour, code=None):
    """
    The cities this tour really played, in order.

    Fact, not model output. It belongs beside the probabilities because the
    two answer different questions and are easy to confuse: the model says
    where an act was LIKELY to go, this says where it went. A city with a high
    probability and no visit is the interesting case, and it is only visible
    when both are on the same screen.
    """
    tc = ex["tour_city"]
    sel = tc["tour"] == tour
    if code:
        sel &= tc["countryCode"] == code
    d = tc[sel].copy()
    if d.empty:
        return d
    city_to_market = ex["cities"].set_index(["countryCode", "city"])["market"]
    d["market"] = pd.MultiIndex.from_arrays(
        [d["countryCode"], d["city"]]).map(city_to_market)
    cols = ["first_event", "city", "market", "country", "venue", "events",
            "largest_capacity_played"]
    return (d[[c for c in cols if c in d.columns]]
            .sort_values("first_event").reset_index(drop=True))


def cities_for_tour_change(menu, spec, market, tour, capacity, kind="indoor",
                           limit=20):
    """
    Where this act is likely to play, before and after a proposed room, with ranks.

    The rank is the point. A city moving from ninth to fourth in an act's
    running order is a far more legible statement than a probability moving
    from 0.031 to 0.068, and it is the form a promoter argues in: not "our
    chances went up" but "we are now ahead of Bologna".

    Only the target city's rows are rebuilt, so every other city's utility is
    untouched -- their ranks move only because the softmax has to take the
    probability from somewhere, which is exactly the competitive effect worth
    showing.
    """
    rows, s = menu["rows"], menu["specs"][spec]
    target = whatif._target_mask(menu, market)
    if kind in ("indoor", "outdoor"):
        applies = target & rows["act_plays"].isin([kind, "either"]).values
    else:
        applies = target
    v_after, _ = whatif._rebuilt_utility(menu, spec, target, applies, capacity)

    p0 = logit.choice_probabilities(s["v"], menu["offsets"])
    p1 = logit.choice_probabilities(v_after, menu["offsets"])

    sel = (rows["tour"] == tour).values
    if not sel.any():
        raise KeyError(f"{tour!r} is not on the {menu['countryCode']} menu")
    sub = rows.loc[sel].copy()
    sub["before"], sub["after"] = p0[sel], p1[sel]

    # Mean over the menus a city appeared on, for the reason set out in
    # cities_for_tour: summing would rank cities the act skipped above the ones
    # it played, because a played city is on only one menu.
    out = (sub.groupby("market", as_index=False)
           .agg(chance_before=("before", "mean"), chance_after=("after", "mean"),
                menus=("before", "size"), ceiling=("ceiling", "first"),
                played=("chosen", "max")))
    out["played"] = out["played"].astype(bool)
    out["change"] = out["chance_after"] - out["chance_before"]
    out["rank before"] = out["chance_before"].rank(ascending=False,
                                                   method="min").astype(int)
    out["rank after"] = out["chance_after"].rank(ascending=False,
                                                 method="min").astype(int)
    out["rank move"] = out["rank before"] - out["rank after"]
    return out.sort_values("chance_after", ascending=False).head(limit)


def ordinal(n):
    """1st, 2nd, 3rd, 4th -- because "Bari stays 1th" undermines everything
    around it, however right the arithmetic behind it is."""
    n = int(n)
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def rank_sentence(change_table, market):
    """One line saying where the city now sits in this act's running order."""
    hit = change_table[change_table["market"] == market]
    if hit.empty:
        return None
    r = hit.iloc[0]
    rb, ra = int(r["rank before"]), int(r["rank after"])
    chance = f"{r['chance_before']:.1%} to {r['chance_after']:.1%}"
    if ra < rb:
        return (f"**{market} moves from {ordinal(rb)} to {ordinal(ra)}** in "
                f"this act's running order, and its chance goes from {chance}.")
    if ra == rb and r["change"] > 1e-6:
        return (f"**{market} stays {ordinal(ra)}**, but its chance rises from "
                f"{chance} — the cities above it are too far ahead to pass.")
    return (f"**{market} stays {ordinal(ra)}** and its chance is unchanged at "
            f"{r['chance_before']:.1%}: the room does nothing for this act.")


# ---------------------------------------------------------------------------
# How much of a city's calendar this model can see at all
# ---------------------------------------------------------------------------

def touring_share(ex, market, code):
    """
    The share of a market's shows that come from multi-city tours.

    WHY THIS BELONGS NEXT TO EVERY PROBABILITY ON THE SCREEN

    The choice model only ever sees tours that played two or more cities in a
    country, because only those involved a choice between cities. A one-off
    local gig, a residency or a single-date festival had no menu: nothing was
    traded off, so there is nothing to estimate, and including them would
    dilute every probability while teaching the model nothing.

    That exclusion is right, and it is also a limit worth putting a number on.
    Across Italy the model sees 49% of shows. By market it ranges from 64% in
    Padua to 9% in Sanremo, whose calendar is one festival rather than a
    touring circuit. A figure that describes half a city's year should say so,
    and one that describes a tenth of it should say so loudly.

    Nothing here corrects the model. It states its coverage.
    """
    tc, cities, tours = ex["tour_city"], ex["cities"], ex["tours"]
    members = set(cities.loc[(cities["market"] == market)
                             & (cities["countryCode"] == code), "city"])
    here = tc[(tc["countryCode"] == code) & (tc["city"].isin(members))]
    if here.empty:
        return None
    n_cities = tours.set_index("tour")["cities"]
    here = here.assign(tour_cities=here["tour"].map(n_cities))
    seen = float(pd.to_numeric(
        here.loc[here["tour_cities"] >= 2, "events"], errors="coerce").sum())

    row = ex["markets"]
    row = row[(row["city"] == market) & (row["countryCode"] == code)]
    total = float(pd.to_numeric(row["events"], errors="coerce").iloc[0]) if len(row) else 0.0
    if not total:
        return None
    share = seen / total
    if share >= 0.55:
        verdict = "the model sees most of this city's calendar"
    elif share >= 0.3:
        verdict = "the model sees about half of this city's calendar"
    else:
        verdict = ("MOST OF THIS CITY'S SHOWS ARE NOT TOURING DATES, so the "
                   "model is describing a small corner of its year")
    return {"touring_dates": seen, "all_events": total, "share": share,
            "verdict": verdict}
