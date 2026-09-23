#!/usr/bin/env python3
"""
Turning the two layers into a plain answer.

WHAT THIS FILE IS FOR

Layer 1 produces verdicts and Layer 2 produces coefficients. Neither is an
answer to the question anyone actually asks, which is:

    Should this city build a room, how big, indoor or outdoor, and what would
    it get back?

This assembles that answer from the pieces, and -- because the whole project
turns on being able to say where a number came from -- carries the evidence for
each part of it alongside.

WHAT EACH PART OF THE ANSWER RESTS ON

  what size      Layer 1 only. The median room the blocked tours actually used
                 elsewhere. No model: these are rooms that were really played.
  indoor or out  Layer 1 only. Which kind of act the blocked tours are.
  why            Layer 1 only. Counts of tours and the ceiling they exceeded.
  how many more  Layer 2. This is the one part that is a model output, and it
                 is reported as an interval between two specifications rather
                 than a point, for the reasons set out in layer2.py.

Keeping that split visible matters. The first three would survive the model
being wrong; the fourth would not.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gap import (BORDERLINE_MULTIPLE, GAP_MULTIPLE, capacity_ladder,   # noqa: E402
                 capacity_test, ceilings_from_ladder, tours_that_skipped)

# Below this many blocked tours there is no case worth making. A market with
# three blocked tours has a coincidence, not a capacity problem.
MIN_BLOCKED_TOURS = 8

# Round recommendations to something a building could actually be. Nobody
# commissions a 13,475-seat arena.
ROUNDING = 500


def _round_to(x, step=ROUNDING):
    if x is None or not np.isfinite(x):
        return None
    return int(round(float(x) / step) * step)


def recommend(ex, market_row, min_dates=3, gap_mult=GAP_MULTIPLE,
              borderline_mult=BORDERLINE_MULTIPLE,
              min_blocked=MIN_BLOCKED_TOURS):
    """
    The Layer 1 half of the answer: size, kind, and the evidence for both.

    Every threshold that can change the verdict is an argument with a default,
    not a constant read from the module. That is deliberate: the app exposes
    all three as controls, and a finding that only exists at one setting of
    them should be visibly fragile rather than quietly asserted.

    Returns a dict with a `case` field that is one of:
      "build"      enough blocked tours of one kind to point at a size
      "no case"    tours skip this market, but not for want of a room
      "too few"    some evidence, below the threshold worth acting on
      "unknown"    cannot be judged -- no ceiling, or nothing skipped it
    """
    ladder = capacity_ladder(ex, market_row)
    ceilings = ceilings_from_ladder(ladder)
    skipped = tours_that_skipped(ex, market_row, min_dates)

    base = {
        "market": market_row["city"], "country": market_row["country"],
        "countryCode": market_row["countryCode"],
        "ladder": ladder, "ceilings": ceilings, "skipped": skipped,
        "min_dates": min_dates, "gap_mult": gap_mult,
        "borderline_mult": borderline_mult, "min_blocked": min_blocked,
    }
    if skipped.empty:
        return {**base, "case": "unknown", "tested": pd.DataFrame(),
                "reason": f"No tour with {min_dates} or more dates in "
                          f"{market_row['country']} skipped this market."}

    tested = capacity_test(skipped, market_row, ceilings, gap_mult,
                           borderline_mult)
    blocked = tested[tested["verdict"].str.startswith("capacity gap")].copy()
    base["tested"] = tested
    base["blocked"] = blocked

    if blocked.empty:
        return {**base, "case": "no case",
                "reason": (f"{len(tested)} tours skipped this market, and none "
                           f"of them was blocked by room size. They all play "
                           f"rooms it can already match, so whatever kept them "
                           f"away, a building would not fix it.")}

    # --- indoor or outdoor -------------------------------------------------
    #
    # Decided by counting which kind of act is actually blocked, not by which
    # is cheaper or more fashionable. "either" acts are attributed to neither:
    # they are tours whose indoor and outdoor dates tie, so using them to break
    # the tie would be circular.
    counts = blocked["plays"].value_counts()
    n_in, n_out = int(counts.get("indoor", 0)), int(counts.get("outdoor", 0))
    kind = "indoor" if n_in >= n_out else "outdoor"
    of_kind = blocked[blocked["plays"].isin([kind, "either"])]

    if len(blocked) < min_blocked:
        return {**base, "case": "too few", "kind": kind,
                "blocked_indoor": n_in, "blocked_outdoor": n_out,
                "blocked_tours": len(blocked), "skipped_tours": len(tested),
                "reason": (f"Only {len(blocked)} tours were blocked by room "
                           f"size, below the {min_blocked} this treats "
                           f"as the minimum worth acting on. That is a handful "
                           f"of acts, not a pattern.")}

    need = pd.to_numeric(of_kind["room_it_needs"], errors="coerce").dropna()
    ceiling_now = ceilings.get(kind)

    return {
        **base,
        "case": "build",
        "kind": kind,
        "blocked_indoor": n_in,
        "blocked_outdoor": n_out,
        "blocked_tours": len(blocked),
        "skipped_tours": len(tested),
        "ceiling_now": ceiling_now,
        # The median clears half the blocked tours; the upper quartile clears
        # three in four. Both are shown because the choice between them is a
        # judgement about ambition, not a fact the data settles.
        "size_median": _round_to(need.median()),
        "size_upper": _round_to(need.quantile(0.75)),
        "size_max": _round_to(need.max()),
        "of_kind": of_kind,
        "reason": (
            f"{len(blocked)} of the {len(tested)} tours that played "
            f"{min_dates}+ dates in {market_row['country']} and skipped "
            f"{market_row['city']} were blocked by room size: they "
            f"consistently played rooms larger than anything "
            f"{market_row['city']} has of the kind they use."
            + (f" Its biggest {kind} room holds {int(ceiling_now):,}."
               if ceiling_now and np.isfinite(ceiling_now)
               else f" It has no {kind} room with a recorded capacity.")),
    }


def where_blocked_tours_went(ex, market_row, blocked, limit=15):
    """
    The markets that absorbed the dates this one lost to room size.

    Deliberately narrower than Layer 1's version, which uses every tour that
    skipped. Restricting to the BLOCKED tours answers the question actually
    being asked -- where are the shows this city cannot host going -- rather
    than the much vaguer "where does everyone else play".
    """
    tc = ex["tour_city"]
    code = market_row["countryCode"]
    want = set(blocked["tour"])
    e = tc[(tc["countryCode"] == code) & (tc["tour"].isin(want))]
    if e.empty:
        return pd.DataFrame()
    city_to_market = ex["cities"].set_index(["countryCode", "city"])["market"]
    e = e.assign(market=pd.MultiIndex.from_arrays(
        [e["countryCode"], e["city"]]).map(city_to_market))
    g = (e.groupby("market")
         .agg(tours=("tour", "nunique"), dates=("events", "sum"),
              biggest_room_used=("largest_capacity_played", "max"))
         .sort_values("tours", ascending=False)
         .reset_index())
    g = g[g["market"] != market_row["city"]]
    g["share of the blocked tours"] = (g["tours"] / len(want)).round(3)
    return g.head(limit)


# ---------------------------------------------------------------------------
# The reasoning, written out
#
# A recommendation that arrives as a verdict and a number is a recommendation
# that has to be taken on trust, and this project's whole premise is that
# nothing here should have to be. So the verdict is also available as an
# ordered chain: each step is a question, the answer, the arithmetic that
# produced the answer, and which layer licensed it.
#
# The chain is deliberately the SAME chain the code walked. It is not a
# narrative written afterwards to justify a number -- each step names the
# figures that decided it, and a reader who disagrees with a step can move the
# threshold behind it in the app and watch the rest of the chain change.
# ---------------------------------------------------------------------------

CASE_HEADLINE = {
    "build": "Build",
    "no case": "No case for a building",
    "too few": "Not enough evidence to act on",
    "unknown": "Cannot be judged",
}

CASE_MEANING = {
    "build": ("A building would remove a real, measured obstacle. It does "
              "not follow that the acts would then come."),
    "no case": ("Tours do skip this market, but they play rooms it can already "
                "match. Whatever is keeping them away, a building is not it."),
    "too few": ("There is some evidence of a capacity obstacle, but from too "
                "few acts to separate from coincidence."),
    "unknown": ("The test cannot be run here: nothing skipped this market, or "
                "there is no capacity on record to judge against."),
}


def article(n):
    """
    "a" or "an" for a number, taken from how it is said rather than spelt.

    Eleven, eighteen and eight open with a vowel sound, so "a 11,500-seat room"
    reads as a typo to anyone who hears the sentence in their head. Small
    thing; this text is meant to be read aloud in a meeting.
    """
    lead = str(int(n)).lstrip("0")
    return "an" if lead[:2] in ("11", "18") or lead[:1] == "8" else "a"


def headline(rec):
    """One line, the way it would be said out loud."""
    case = rec["case"]
    if case != "build":
        return f"{CASE_HEADLINE[case]} — {rec['market']}"
    return (f"Build {article(rec['size_median'])} {rec['size_median']:,}-seat "
            f"{rec['kind']} room in {rec['market']} "
            f"({rec['blocked_tours']} tours blocked today)")


def narrative(rec, market_row=None, kept=None, caveat=None):
    """
    The recommendation as a chain of question, answer and arithmetic.

    Returns a list of dicts with keys: step, question, answer, because, layer.
    `layer` is the honest label on each step -- "counting" for everything the
    descriptive layer establishes, "model" for anything that needed a
    coefficient -- so a reader can see at a glance how much of the case rests
    on a model and how much would survive the model being wrong.
    """
    steps = []
    m = rec["market"]
    ceilings = rec.get("ceilings") or {}
    ladder = rec.get("ladder")
    tested = rec.get("tested")
    n_venues = 0 if ladder is None else len(ladder)

    def add(question, answer, because, layer="counting"):
        steps.append({"step": len(steps) + 1, "question": question,
                      "answer": answer, "because": because, "layer": layer})

    # --- 1. what is there now ---------------------------------------------
    indoor, outdoor = ceilings.get("indoor"), ceilings.get("outdoor")
    add("What rooms does it have?",
        (f"Biggest indoor room {int(indoor):,}" if indoor and np.isfinite(indoor)
         else "No indoor room with a recorded capacity")
        + (f"; biggest outdoor {int(outdoor):,}." if outdoor and np.isfinite(outdoor)
           else "; no outdoor room with a recorded capacity."),
        f"From the {n_venues} venues in {m} that have actually hosted a show "
        f"since 2023. A hall that never books music is not a room a promoter "
        f"can use.")

    # --- 2. how many people is it for -------------------------------------
    if market_row is not None:
        raw = market_row.get("population_60km")
        excl = market_row.get("exclusive_population_60km")
        if raw and excl:
            add("How many people is that room for?",
                f"{int(excl):,} for whom this is the nearest market that hosts "
                f"shows, out of {int(raw):,} within 60 km"
                + (f" ({kept:.0%} kept)." if kept is not None else "."),
                "A plain radius credits a town beside a big city with that "
                "city's population. This keeps only residents with no closer "
                "market."
                + (" Below 0.3 that is a satellite, not a market."
                   if kept is not None and kept < 0.3 else ""))

    # --- 3. who skipped ----------------------------------------------------
    n_skipped = 0 if tested is None else len(tested)
    if rec["case"] == "unknown":
        add("Who toured the country and skipped it?", "Nobody, on this filter.",
            rec.get("reason", ""))
        add("So?", CASE_HEADLINE["unknown"], CASE_MEANING["unknown"])
        return steps

    add("Who toured the country and skipped it?",
        f"{n_skipped:,} tours played {rec['min_dates']}+ dates in "
        f"{rec['country']} without coming here.",
        f"Acts with fewer than {rec['min_dates']} dates are excluded \u2014 "
        f"they were never choosing between cities.")

    # --- 4. was the room the reason ---------------------------------------
    blocked = rec.get("blocked")
    n_blocked = 0 if blocked is None else len(blocked)
    add("Of those, how many were blocked by room size?",
        f"{n_blocked:,} of {n_skipped:,}"
        + (f" ({n_blocked / n_skipped:.0%})." if n_skipped else "."),
        f"Blocked means the act's usual room is more than "
        f"{rec.get('gap_mult', 1.2):g}x the biggest room of the same kind "
        f"here. Ambiguous acts are judged against the larger of the two "
        f"ceilings, so a missing label can never invent a gap.")

    if rec["case"] == "no case":
        add("So?", CASE_HEADLINE["no case"], rec.get("reason", CASE_MEANING["no case"]))
        return steps

    # --- 5. indoor or outdoor ---------------------------------------------
    n_in = rec.get("blocked_indoor", 0)
    n_out = rec.get("blocked_outdoor", 0)
    add("Indoor or outdoor?",
        f"{rec['kind'].capitalize()} — {n_in} of the blocked tours are indoor "
        f"acts and {n_out} are outdoor.",
        "Counted from which kind of act is blocked, not from which is "
        "cheaper to build. Acts whose dates tie are attributed to neither.")

    if rec["case"] == "too few":
        add("Is that enough to act on?", CASE_HEADLINE["too few"],
            rec.get("reason", CASE_MEANING["too few"]))
        return steps

    # --- 6. how big --------------------------------------------------------
    ceiling_now = rec.get("ceiling_now")
    add("How big?",
        f"{rec['size_median']:,} seats clears half of them; "
        f"{rec['size_upper']:,} clears three in four.",
        f"Rooms these acts actually played elsewhere in {rec['country']} — "
        f"median and upper quartile, no model. Median, not mean: one stadium "
        f"act would otherwise drag it into fantasy. The largest any of them "
        f"needs is {rec['size_max']:,}."
        + (f" Today's {rec['kind']} ceiling is {int(ceiling_now):,}, so this "
           f"is {rec['size_median'] - int(ceiling_now):,} seats more."
           if ceiling_now and np.isfinite(ceiling_now) else ""))

    # --- 7. the verdict ----------------------------------------------------
    add("So?", headline(rec), CASE_MEANING["build"])

    if caveat:
        add("What would overturn this?", "A single mislabelled venue.",
            f"{caveat} If that room is playable indoors, the ceiling every "
            f"verdict above used is wrong and the case goes with it.")
    return steps


def what_would_change_it(rec, market_row=None, kept=None, caveat=None):
    """
    The explicit list of things that would overturn the recommendation.

    Written out rather than left implied, because a finding whose falsifiers
    are not stated is not a finding anyone can argue with -- and the point of
    this project is to produce numbers that survive being argued with.
    """
    out = []
    if caveat:
        out.append(f"**A mislabelled venue.** {caveat}")
    if kept is not None and kept < 0.3:
        out.append(
            f"**It may not be a market.** Only {kept:.0%} of the people within "
            f"60 km have no closer market \u2014 the rest already have a "
            f"bigger room nearer.")
    ceilings = rec.get("ceilings") or {}
    unl = int(ceilings.get("unlabelled_venues") or 0)
    biggest = ceilings.get("largest_unlabelled")
    if unl and biggest is not None and np.isfinite(biggest):
        out.append(
            f"**{unl} venues carry no indoor/outdoor label**, the largest "
            f"holding {int(biggest):,}. If it is indoors, the ceiling is too "
            f"low.")
    if rec.get("case") == "build":
        out.append(
            f"**The threshold.** Blocked means "
            f"{rec.get('gap_mult', 1.2):g}x the ceiling. If the case does not "
            f"survive raising that to 1.4, it is thin.")
        out.append(
            "**Removing the obstacle is not filling the room.** A building "
            "removes one reason for staying away. Routing, guarantees and "
            "promoter relationships are reasons this data never sees.")
    return out


# ---------------------------------------------------------------------------
# Every market at once
#
# The per-market function above is the honest one: it re-runs the capacity test
# from the extract. Running it for all 735 markets takes minutes, which is what
# screen.py is for -- it does exactly that and writes the result to a workbook.
#
# So the Europe-wide view reads that workbook and applies the SAME rule to it,
# rather than re-deriving anything. If the two ever disagreed, the per-market
# page would be right and this would be stale; the app therefore labels this
# view with the screening run it came from.
# ---------------------------------------------------------------------------

RANKINGS = {
    "seats below comparable markets": (
        "ceiling_below_peers",
        "How many seats smaller the biggest indoor room is than the median for "
        "markets with a similar catchment. The default, because it is the only "
        "one of these that is adjusted for how big the market is."),
    "tours blocked by room size": (
        "gap_tours",
        "The raw count of tours that skipped and needed a bigger room. Reads "
        "naturally and ranks badly: it is largely a measure of how SMALL the "
        "biggest room is, so villages with an 800-seat hall top it. Useful for "
        "comparing markets within a country, misleading across them."),
    "share of skipping tours blocked": (
        "gap_share",
        "The same count as a share of everyone who skipped. Less sensitive to "
        "how many acts tour the country, but still rewards a small ceiling."),
    "catchment": (
        "exclusive_population_60km",
        "People for whom this is the nearest market hosting shows. Not a "
        "finding at all -- it is here so a reader can see the size of the "
        "place behind every other ranking."),
}


def rank_all(screen, min_blocked=MIN_BLOCKED_TOURS, min_kept=0.3,
             require_assessable=True, require_below_peers=True,
             rank_by="seats below comparable markets"):
    """
    A verdict and a one-line reason for every market in a screening run.

    THE THREE GATES, AND WHY EACH IS THERE

    A market earns "build" only if all three hold, and each exists because
    without it the ranking fills with something that is not a finding:

      enough tours blocked      below MIN_BLOCKED_TOURS it is a coincidence
      small room FOR ITS SIZE   without this, ranking by blocked tours simply
                                finds the markets with the smallest rooms --
                                Frome, with an 850-seat hall, tops the raw
                                count with 978 "blocked" tours, which says
                                nothing except that Frome is a village
      a market, not a satellite Bristol keeps 25% of its 60 km radius and
                                Steventon 7%; most of the people a room there
                                would serve already have a bigger one nearer

    All three are arguments, not facts, which is why all three are sliders in
    the app rather than constants here.

    Markets that cannot be judged are labelled "unknown" rather than dropped:
    "we cannot tell" and "there is no gap" are different findings and must not
    be averaged together later.
    """
    s = screen.copy()

    gap = pd.to_numeric(s.get("gap_tours"), errors="coerce").fillna(0)
    skipped = pd.to_numeric(s.get("skipped_tours"), errors="coerce").fillna(0)
    kept = pd.to_numeric(s.get("catchment_kept"), errors="coerce")
    ceiling = pd.to_numeric(s.get("indoor_ceiling"), errors="coerce")
    suggested = pd.to_numeric(s.get("suggested_capacity"), errors="coerce")
    below = pd.to_numeric(s.get("ceiling_below_peers"), errors="coerce")
    assessable = (s["assessable"].astype(bool) if "assessable" in s
                  else pd.Series(True, index=s.index))

    judgeable = assessable | (not require_assessable)
    satellite = kept.notna() & (kept < min_kept)
    blocked_enough = gap >= min_blocked
    bigger_than_now = suggested > ceiling.fillna(0)
    small_for_size = (below > 0) if require_below_peers else pd.Series(
        True, index=s.index)

    s["case"] = np.select(
        [~judgeable, skipped <= 0, satellite,
         blocked_enough & bigger_than_now & small_for_size,
         gap > 0],
        ["unknown", "unknown", "satellite", "build", "too few"],
        default="no case")

    s["seats to add"] = (suggested - ceiling.fillna(0)).where(s["case"] == "build")
    s["blocked share"] = (gap / skipped.replace(0, np.nan)).round(3)

    def reason(r):
        case = r["case"]
        mk, cty = r["market"], r.get("country", "")
        g, sk = int(r.get("gap_tours") or 0), int(r.get("skipped_tours") or 0)
        c, pm = r.get("indoor_ceiling"), r.get("peer_median_ceiling")
        ceil_s = f"{int(c):,}" if pd.notna(c) else "no measured indoor room"
        if case == "build":
            peer = (f" — {int(r['ceiling_below_peers']):,} seats below the "
                    f"{int(pm):,} typical of markets its size"
                    if pd.notna(pm) and pd.notna(r.get("ceiling_below_peers"))
                    else "")
            return (f"{g} of the {sk} tours that toured {cty} without playing "
                    f"{mk} need a bigger indoor room than its {ceil_s}{peer}. "
                    f"They typically play "
                    f"{int(r['suggested_capacity']):,}.")
        if case == "satellite":
            return (f"{mk} keeps only {r['catchment_kept']:.0%} of the people "
                    f"within 60 km — the rest are nearer a bigger market. It "
                    f"is a satellite, not a market, whatever the counts say.")
        if case == "too few":
            if pd.notna(r.get("ceiling_below_peers")) and r["ceiling_below_peers"] <= 0:
                return (f"{g} tours were blocked by room size, but {mk}'s "
                        f"{ceil_s} is already normal or large for a market of "
                        f"its catchment. The blocked count reflects its size, "
                        f"not a shortfall.")
            return (f"Only {g} of {sk} skipping tours were blocked by room "
                    f"size, below the {min_blocked} treated as the minimum "
                    f"worth acting on.")
        if case == "no case":
            return (f"{sk} tours skipped {mk} and none was blocked by room "
                    f"size — they all play rooms its {ceil_s} can match.")
        if not bool(r.get("assessable", True)):
            return (f"Not assessable: {mk} has too little indoor activity or "
                    f"too small a catchment for the indoor test to mean "
                    f"anything.")
        return f"Too little evidence to judge {mk}."

    s["why"] = s.apply(reason, axis=1)

    col = RANKINGS.get(rank_by, RANKINGS["seats below comparable markets"])[0]
    if col not in s.columns:
        col = "gap_tours"
    order = {"build": 0, "too few": 1, "no case": 2, "satellite": 3, "unknown": 4}
    s["_o"] = s["case"].map(order)
    s["_r"] = pd.to_numeric(s[col], errors="coerce").fillna(-np.inf)
    return (s.sort_values(["_o", "_r"], ascending=[True, False])
            .drop(columns=["_o", "_r"]))
