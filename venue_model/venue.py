#!/usr/bin/env python3
"""
One room at a time — what a venue pulls, and which tours it should expect.

THE HONEST STARTING POINT

Layer 2 does not model venues. Its choice set is cities: a tour picks Naples or
it picks Rome, and the only thing it knows about the rooms in either is the
capacity of the biggest one of the kind that act plays. Nothing in the fitted
model distinguishes the Palapartenope from any other 6,541-seat hall.

Everything here is therefore built on top of the market model, and the parts
rest on very different foundations. They are kept apart throughout, because
mixing them is how a venue screen ends up more confident than its weakest
ingredient.

  SOLID -- pure model arithmetic, no extra assumption
    MARKET PULL       Does this room bring tours to the CITY that would
                      otherwise not come? Delete the venue, let the market's
                      ceiling fall to whatever is left, and ask the model
                      again. Non-zero only when the venue is the biggest of its
                      kind, because only then does removing it change anything
                      an act is tested against.
    CAN IT HOST       Whether the act's usual room fits inside this one, and
                      whether the kind matches. A comparison of two numbers
                      and a label.

  SHAKY -- needs an assumption about which of several adequate rooms is used
    WOULD IT HOST     Of the tours the city gets, which land in THIS room
                      rather than a neighbouring one.

WHAT THE ASSUMPTION IS, AND HOW WELL IT ACTUALLY DOES

    NEAREST-CAPACITY RULE. A tour plays the room whose capacity is closest, in
    log terms, to the room it uses elsewhere.

The obvious alternative -- the smallest room that will HOLD the act -- was
tried first and is much worse, because it assumes acts never scale down. They
do, constantly: a tour averaging 9,881 elsewhere played Naples' 6,541-seat
Palapartenope, and one averaging 76,903 played the 52,530 stadium. A floor rule
calls both "nothing here fits" and matched the real room 11% of the time in
Naples against 51% for nearest-capacity.

But nearest-capacity is not good everywhere, and `validate_rightsizing()`
measures it per market against a deliberately stupid baseline -- always guess
the busiest room -- so the comparison is honest:

    Naples   13 rooms    51%  vs  29% baseline     usable
    Bari      7 rooms    76%  vs  24% baseline     good
    Rome     27 rooms    22%  vs  22% baseline     no better than guessing
    Milan    35 rooms    11%  vs  18% baseline     WORSE than guessing

The pattern is not subtle. Where a market has a handful of rooms the rule
works; where it has thirty-five at overlapping sizes, nothing in this data
distinguishes them and the rule adds nothing. So the accuracy and the baseline
travel with the output and are meant to be shown beside it. In a market where
the rule loses to the baseline, the room split is decoration and the market
pull is the only number on the screen worth reading.
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
from gap import capacity_ladder, resolve_market                  # noqa: E402

# A tour's probability of the city has to clear this before the venue is said
# to "expect" it. Below this the act is not realistically in play, and a long
# tail of near-zero rows buries the ones that matter.
MIN_P = 0.02

# Capacities are clipped here before logging, so a room recorded as holding
# four people cannot dominate a log distance.
MIN_CAP = 50.0


# ---------------------------------------------------------------------------
# The rooms in a market
# ---------------------------------------------------------------------------

def rooms(ex, market_row):
    """
    Every room in the market with a known capacity, largest first.

    Built on Layer 1's ladder, so the indoor/outdoor labels are the same ones
    the capacity verdicts used -- including those inferred from the venue name,
    which `io_source` flags.
    """
    lad = capacity_ladder(ex, market_row)
    lad = lad[pd.to_numeric(lad["capacity"], errors="coerce").notna()].copy()
    lad["capacity"] = pd.to_numeric(lad["capacity"], errors="coerce")
    return lad.sort_values("capacity", ascending=False).reset_index(drop=True)


def ceilings_without(lad, venue):
    """
    What the market's ceilings become if this venue did not exist.

    The entire mechanism behind market pull. If the answer is unchanged, the
    venue cannot be bringing anyone to the city who was not already coming.
    """
    rest = lad[lad["venue"] != venue]
    f = lambda s: float(s.max()) if len(s) and s.notna().any() else np.nan
    return (f(rest.loc[rest["io"] == "inside", "capacity"]),
            f(rest.loc[rest["io"] == "outside", "capacity"]),
            f(rest["capacity"]))


def _pool(lad, kind):
    """Rooms of the right kind, falling back to all of them if none match."""
    if kind == "either":
        return lad
    want = "inside" if kind == "indoor" else "outside"
    sub = lad[lad["io"] == want]
    return sub if len(sub) else lad


def assign_room(lad, need, kind):
    """
    The room whose capacity is closest to what the act plays elsewhere.

    Log distance rather than absolute, because the difference between 1,000 and
    2,000 seats matters far more than between 41,000 and 42,000. Returns
    (venue, capacity) or (None, nan) when the act's size is unknown.
    """
    if need is None or not np.isfinite(need):
        return None, np.nan
    pool = _pool(lad, kind)
    if pool.empty:
        return None, np.nan
    d = (np.log(pool["capacity"].clip(lower=MIN_CAP))
         - np.log(max(float(need), MIN_CAP))).abs()
    r = pool.loc[d.idxmin()]
    return r["venue"], float(r["capacity"])


def _act_sizes(ex, market_row):
    """What each tour plays ELSEWHERE in the country, and which kind it is."""
    tc = ex["tour_city"]
    code = market_row["countryCode"]
    members = set(ex["cities"].loc[
        (ex["cities"]["market"] == market_row["city"])
        & (ex["cities"]["countryCode"] == code), "city"])
    other = tc[(tc["countryCode"] == code) & (~tc["city"].isin(members))]
    pt = other.groupby("tour").agg(
        need=("largest_capacity_played", "median"),
        indoor=("indoor_events", "sum"), outdoor=("outdoor_events", "sum"))
    pt["kind"] = np.select(
        [pt["indoor"] > pt["outdoor"], pt["outdoor"] > pt["indoor"]],
        ["indoor", "outdoor"], default="either")
    return pt, members


def validate_rightsizing(ex, market_row, lad):
    """
    How often the rule names the room that was actually used, against a baseline.

    THIS IS THE NUMBER THAT LICENSES THE ROOM SPLIT, so it is measured rather
    than asserted, and measured against something deliberately stupid -- always
    guessing the market's busiest room. A rule that cannot beat that is not
    adding information, however plausible it sounds.

    Two weaknesses, both of which push the measured agreement DOWN rather than
    flattering it:

      - the extract records one venue per tour-city pair, taken as the
        alphabetically last where a tour used several rooms in one city, so
        multi-room visits can be scored wrong through no fault of the rule.
      - an act's size is a median over its other stops, which for a two-city
        tour rests on a single observation.
    """
    tc = ex["tour_city"]
    code = market_row["countryCode"]
    pt, members = _act_sizes(ex, market_row)
    here = tc[(tc["countryCode"] == code) & (tc["city"].isin(members))
              & tc["venue"].notna()].copy()
    here = here.join(pt, on="tour")
    here = here[pd.to_numeric(here["need"], errors="coerce").notna()]

    blank = {"checked": 0, "agreement": np.nan, "baseline": np.nan,
             "beats_baseline": False, "rooms": int(len(lad)),
             "detail": pd.DataFrame()}
    if here.empty or lad.empty:
        return blank

    here["predicted room"] = [assign_room(lad, n, k)[0]
                              for n, k in zip(here["need"], here["kind"])]
    here["actual room"] = here["venue"]
    here["match"] = here["predicted room"] == here["actual room"]

    busiest = lad.sort_values("events", ascending=False).iloc[0]["venue"]
    agreement = float(here["match"].mean())
    baseline = float((here["actual room"] == busiest).mean())

    return {
        "checked": int(len(here)),
        "agreement": agreement,
        "baseline": baseline,
        "baseline_room": busiest,
        "beats_baseline": agreement > baseline,
        "rooms": int(len(lad)),
        "multi_date_share": float((pd.to_numeric(here["events"],
                                                 errors="coerce") > 1).mean()),
        "detail": here[["tour", "need", "kind", "predicted room",
                        "actual room", "match", "events"]]
        .sort_values("match"),
    }


def reliability(val):
    """One sentence on whether the room split on this screen is worth reading."""
    if not val or not val.get("checked"):
        return ("No visit to this market can be checked, so the room split is "
                "unverified. Read the market pull only.")
    a, b, n = val["agreement"], val["baseline"], val["checked"]
    head = (f"The rule names the right room {a:.0%} of the time across {n} "
            f"checkable visits, against {b:.0%} for always guessing "
            f"{val['baseline_room']}.")
    if a > b + 0.15:
        return head + (" It is clearly adding information, so the room split "
                       "below is worth reading.")
    if a > b:
        return head + (" It only just beats guessing, so treat the split "
                       "between similarly sized rooms as weak.")
    return head + (f" IT DOES NOT BEAT GUESSING. With {val['rooms']} rooms at "
                   f"overlapping sizes, nothing in this data says which one an "
                   f"act picks. Ignore the room split and read only the market "
                   f"pull and what the room can physically host.")


# ---------------------------------------------------------------------------
# Market pull: delete the venue and ask the model again
# ---------------------------------------------------------------------------

def _utility_without(menu, spec, target, ceil_in, ceil_out, ceil_any):
    """
    The utility vector with the target market's ceilings lowered.

    The mirror of whatif._rebuilt_utility, which raises them. Kept separate
    rather than generalised because the two differ in a way worth seeing:
    adding a room can only RAISE a ceiling, so that function takes a maximum,
    while removing one re-derives the ceiling from what is left and must
    therefore be handed all three values.
    """
    rows, s = menu["rows"], menu["specs"][spec]
    sub = rows.loc[target].copy()
    kind = sub["act_plays"].values
    sub["ceiling"] = np.where(kind == "indoor", ceil_in,
                              np.where(kind == "outdoor", ceil_out, ceil_any))
    X_sub, _ = layer2.design(sub, spec, s["consts"])
    v = s["v"].copy()
    v[target] = X_sub @ s["beta"]
    return v


def market_pull(menu, spec, ex, market_row, venue, lad=None):
    """
    Expected tour-visits to the CITY with this venue, and without it.

    The difference is the venue's market pull. `binding` is False when removing
    the venue leaves every ceiling unchanged -- the common case, and not a
    failure. It means the room is not what makes acts choose this city, and the
    honest answer to "what does it pull" is zero.
    """
    if lad is None:
        lad = rooms(ex, market_row)
    target = whatif._target_mask(menu, market_row["city"])
    s = menu["specs"][spec]

    ins = lad.loc[lad["io"] == "inside", "capacity"]
    out = lad.loc[lad["io"] == "outside", "capacity"]
    cur = (float(ins.max()) if len(ins) else np.nan,
           float(out.max()) if len(out) else np.nan,
           float(lad["capacity"].max()) if len(lad) else np.nan)
    without = ceilings_without(lad, venue)

    p_now = logit.choice_probabilities(s["v"], menu["offsets"])
    expected_now = float(p_now[target].sum())

    same = all((np.isnan(a) and np.isnan(b)) or a == b
               for a, b in zip(cur, without))
    if same:
        return {"binding": False, "expected_with": expected_now,
                "expected_without": expected_now, "pull": 0.0,
                "ceilings_now": cur, "ceilings_without": without,
                "per_tour": pd.DataFrame()}

    v_without = _utility_without(menu, spec, target, *without)
    p_without = logit.choice_probabilities(v_without, menu["offsets"])
    expected_without = float(p_without[target].sum())

    sub = menu["rows"].loc[target].copy()
    sub["p_with"], sub["p_without"] = p_now[target], p_without[target]
    sub["pull"] = sub["p_with"] - sub["p_without"]
    per_tour = (sub.groupby(["tour", "headliner", "category", "act_plays",
                             "room_needed"], as_index=False)
                .agg(p_with=("p_with", "sum"), p_without=("p_without", "sum"),
                     pull=("pull", "sum"))
                .sort_values("pull", ascending=False))

    return {"binding": True, "expected_with": expected_now,
            "expected_without": expected_without,
            "pull": expected_now - expected_without,
            "ceilings_now": cur, "ceilings_without": without,
            "per_tour": per_tour}


# ---------------------------------------------------------------------------
# Which tours the room should and should not expect
# ---------------------------------------------------------------------------

# Two separate judgements, deliberately not collapsed into one column.
#
# CAN_HOST is a comparison of two numbers and a label. It does not depend on
# the assignment rule and is as reliable as the capacity data itself.
#
# WOULD_HOST asks which of several adequate rooms an act picks, and is only as
# good as the nearest-capacity rule -- which in a market like Milan is not good
# at all. Anything reading WOULD_HOST should show reliability() beside it.

CAN_HOST_MEANING = {
    "fits": "The act's usual room fits inside this one, and the kind matches.",
    "too big for this room": "The act plays rooms larger than this venue holds. "
                             "It can only come if it scales down, which acts do "
                             "but not indefinitely.",
    "far too small for the act": "This room is a small fraction of what the act "
                                 "normally plays. Taking it would mean turning "
                                 "away most of the demand.",
    "wrong kind of room": "An indoor act and an outdoor venue, or the reverse.",
    "size unknown": "No capacity recorded for the rooms this act plays "
                    "elsewhere, so it cannot be placed.",
}

# Below this ratio of venue capacity to the act's usual room, the venue is not
# a plausible substitute even allowing for scaling down. A third is generous:
# acts really do play a third of their usual size in a secondary market.
TOO_SMALL_RATIO = 1 / 3


def expectations(menu, spec, ex, market_row, venue, lad=None, min_p=MIN_P):
    """
    Every tour that toured the country, and what this room should expect of it.

    One row per tour, carrying three things in separate columns because they
    have three different levels of trustworthiness:

      p_city     the model's probability the CITY gets this tour.
      can_host   whether this room could physically take it.
      would_host whether the assignment rule puts it here rather than in a
                 neighbouring room.
    """
    if lad is None:
        lad = rooms(ex, market_row)
    hit = lad[lad["venue"] == venue]
    if hit.empty:
        raise KeyError(f"no venue called {venue!r} with a known capacity in "
                       f"{market_row['city']}")
    cap = float(hit.iloc[0]["capacity"])
    vkind = hit.iloc[0]["io"]

    target = whatif._target_mask(menu, market_row["city"])
    s = menu["specs"][spec]
    p = logit.choice_probabilities(s["v"], menu["offsets"])

    sub = menu["rows"].loc[target].copy()
    sub["p"] = p[target]
    per = (sub.groupby(["tour", "headliner", "category", "act_plays",
                        "room_needed"], as_index=False)
           .agg(p_city=("p", "sum"), occasions=("p", "size")))

    need = pd.to_numeric(per["room_needed"], errors="coerce")
    kinds = per["act_plays"]

    assigned = [assign_room(lad, n, k) for n, k in zip(need, kinds)]
    per["room the rule picks"] = [a[0] for a in assigned]
    per["that room holds"] = [a[1] for a in assigned]

    kind_ok = (kinds == "either") | (
        ((kinds == "indoor") & (vkind == "inside"))
        | ((kinds == "outdoor") & (vkind == "outside")))

    per["can_host"] = np.select(
        [need.isna(), ~kind_ok, need > cap, need * TOO_SMALL_RATIO > cap],
        ["size unknown", "wrong kind of room", "too big for this room",
         "far too small for the act"],
        default="fits")
    # "too big" is checked before "far too small" above, so restore the order:
    # an act needing 40,000 in a 1,000-seat room is far too small, not too big.
    per["can_host"] = np.where(
        (need.notna()) & kind_ok & (need * TOO_SMALL_RATIO > cap),
        "far too small for the act", per["can_host"])

    per["would_host"] = (per["room the rule picks"] == venue)
    per["likely to come"] = per["p_city"] >= min_p

    # Expected visits landing in THIS room. Model probability filtered by the
    # assignment rule, so it inherits that rule's reliability -- which is why
    # reliability() is meant to be shown next to it.
    per["expected here"] = np.where(per["would_host"], per["p_city"], 0.0)
    per["ratio to this room"] = (need / cap).round(2)
    return per.sort_values(["expected here", "p_city"], ascending=False), cap, vkind


def summarise(per, cap, venue, market_row, pull, val):
    """The handful of numbers a venue screen leads with."""
    likely = per[per["likely to come"]]
    can = likely["can_host"].value_counts()
    return {
        "venue": venue, "capacity": cap,
        "market": market_row["city"], "country": market_row["country"],
        "market_pull": pull["pull"], "binding": pull["binding"],
        "expected_city": pull["expected_with"],
        "expected_here": float(per["expected here"].sum()),
        "in_play": int(len(likely)),
        "fits": int(can.get("fits", 0)),
        "too_big": int(can.get("too big for this room", 0)),
        "too_small": int(can.get("far too small for the act", 0)),
        "wrong_kind": int(can.get("wrong kind of room", 0)),
        "would_host": int(likely["would_host"].sum()),
        "rule_agreement": val.get("agreement"),
        "rule_baseline": val.get("baseline"),
        "rule_trustworthy": bool(val.get("beats_baseline")),
    }


def actual_at_venue(ex, market_row, venue):
    """
    Tours that really played this room.

    The check on everything above. Subject to the one-venue-per-tour-city
    limitation described in validate_rightsizing, so it undercounts rooms
    sharing a city with an alphabetically later neighbour.
    """
    tc = ex["tour_city"]
    code = market_row["countryCode"]
    members = set(ex["cities"].loc[
        (ex["cities"]["market"] == market_row["city"])
        & (ex["cities"]["countryCode"] == code), "city"])
    here = tc[(tc["countryCode"] == code) & (tc["city"].isin(members))
              & (tc["venue"] == venue)]
    if here.empty:
        return pd.DataFrame()
    meta = ex["tours"].set_index("tour")[["headliner", "category"]]
    return (here.join(meta, on="tour")[
        ["tour", "headliner", "category", "events", "first_event",
         "largest_capacity_played"]]
        .sort_values("first_event", ascending=False))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test(code="IT", markets=("Naples", "Bari", "Milan")):
    from gap import latest_extract, load_extract
    ex = load_extract(latest_extract())
    blob, err = whatif.fitted_model()
    if err:
        raise SystemExit(err)
    menu = whatif.build_menu(ex, code, blob)

    for market in markets:
        row = resolve_market(ex, market)
        lad = rooms(ex, row)
        val = validate_rightsizing(ex, row, lad)
        print(f"\n=== {market} ({len(lad)} rooms) ===")
        print("  " + reliability(val))
        for v in lad["venue"].head(3):
            pull = market_pull(menu, "A", ex, row, v, lad)
            per, cap, vkind = expectations(menu, "A", ex, row, v, lad)
            s = summarise(per, cap, v, row, pull, val)
            print(f"  {v[:44]:46s} {cap:>7,.0f} {vkind:7s} "
                  f"pull {s['market_pull']:+6.2f}  "
                  f"fits {s['fits']:>3d}  too big {s['too_big']:>3d}  "
                  f"would host {s['would_host']:>3d}")

        biggest = lad.iloc[0]["venue"]
        assert market_pull(menu, "A", ex, row, biggest, lad)["pull"] >= -1e-9, \
            "removing the largest room should not increase expected visits"
    print("\nPASS")


if __name__ == "__main__":
    _self_test()
