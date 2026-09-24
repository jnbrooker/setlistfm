#!/usr/bin/env python3
"""
Building the choice dataset — turning tour histories into menus.

THE QUESTION THIS SETS UP

Layer 1 counts what happened. Layer 2 asks why: given that a tour played four
cities in Italy, why those four and not the other ninety it could have picked?

To ask that, every real visit has to be paired with the alternatives that were
passed over. That is what this file builds. Each row is one (occasion,
candidate market) pair, and one row per occasion is marked as the one actually
chosen.

WHAT COUNTS AS A CHOICE OCCASION

One per city a tour visited in a country. A tour that played Milan, Rome and
Turin generates three occasions, each offering the full Italian menu.

This is an approximation and worth being explicit about. Real routing is a
joint decision -- an act picks a set of cities that work as a run, not three
independent draws. Modelling the set jointly would mean a choice set of every
possible subset, which for ninety markets is a number with twenty-six digits.
The standard compromise is to model each visit as its own draw, and the cost is
that the model treats the three picks as independent when they are not. Two
consequences follow, and neither is hidden:

  - Standard errors would be far too small if the occasions were treated as
    independent, so they are clustered by tour throughout.
  - The model has no memory of the ORDER a tour picked its cities in, so it
    cannot say anything about sequencing.

What it does handle is that a tour cannot play the same city twice: each
occasion's menu excludes the markets the tour played at its other stops. That
is both realistic and necessary -- see the note on the menu in build() for the
leak that results from leaving them in.

WHICH TOURS ARE USABLE

Only tours that played at least two cities in the country. A tour with a single
date was never choosing between cities -- it had a reason to be in exactly one
place -- so including it would add rows that carry no information about
trade-offs while inflating the sample.

THE LEAVE-ONE-OUT RULE, AND WHY IT MATTERS MORE THAN IT SOUNDS

Two of the attributes below describe the tour: the size of room it needs, and
where its other dates are. Both are built from the tour's own history, which
includes the very visit being explained. Computed naively, a market would look
attractive partly because the tour went there -- the outcome leaking into the
predictor, which produces a model that fits beautifully and means nothing.

So both are computed leaving the focal visit out, and, critically, the
reference set is THE SAME for every alternative on a menu. If the tour played
Milan, Rome and Turin, then on the Rome occasion every candidate city is
measured against {Milan, Turin} -- including Rome itself. Otherwise the chosen
city would be the only one measuring its distance to itself, and the routing
coefficient would be measuring the leak rather than the routing.
"""

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from catchment import COVERED, EARTH_RADIUS_KM   # noqa: E402
from gap import latest_extract, load_extract     # noqa: E402

# A market must host shows at this RATE to appear on a menu at all. Candidate
# cities with a gig a year are not places a touring act was realistically
# weighing, and including several hundred of them dilutes every probability in
# the model toward zero while teaching it nothing.
#
# A RATE, NOT A COUNT, and that distinction was a real bug. The threshold used
# to be a flat 20 events. Over a 3.7-year extract that meant 5.4 shows a year --
# a working market. Over a 14.7-year one the same 20 means 1.4 shows a year, so
# the Italian menu swelled from 39 markets to 90 and every probability fell,
# purely because the window got longer. The model was answering a different
# question at each window length without saying so.
MIN_MARKET_EVENTS_PER_YEAR = 5.0

# An absolute floor as well, for very short extracts where the rate alone would
# admit a market on two observations.
MIN_MARKET_EVENTS_FLOOR = 12

# A tour must have played at least this many cities in the country. Two is the
# minimum that makes the leave-one-out rule above possible at all: with one
# city, removing it leaves nothing to measure against.
MIN_CITIES = 2

# Radius whose catchment the model uses. 60 km matches the screening pass.
RADIUS_KM = 60


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def pairwise_km(lat, lon):
    """Great-circle distance between every pair of points, in kilometres."""
    la = np.radians(np.asarray(lat, dtype=float))
    lo = np.radians(np.asarray(lon, dtype=float))
    dlat = la[:, None] - la[None, :]
    dlon = lo[:, None] - lo[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def years_covered(ex):
    """How long the extract spans, so a rate can be turned into a count."""
    try:
        lo, hi = ex["manifest"]["event_date_range"]
        return max((pd.Timestamp(hi) - pd.Timestamp(lo)).days / 365.25, 0.25)
    except Exception:
        return 1.0


def candidate_markets(ex, min_events=None,
                      per_year=MIN_MARKET_EVENTS_PER_YEAR):
    """
    The menu, by country.

    Restricted to countries with demographic coverage, because catchment is the
    single most important control and a model fitted where it is missing would
    be attributing to capacity whatever population explains.

    The activity threshold is a rate scaled by the extract's own span, so the
    menu means the same thing whether the window is four years or fifteen.
    Passing `min_events` overrides it with a flat count, which is only there
    for callers that want to reproduce an older run.
    """
    m = ex["markets"].copy()
    m = m[m["countryCode"].isin(COVERED)]
    if min_events is None:
        min_events = max(per_year * years_covered(ex), MIN_MARKET_EVENTS_FLOOR)
    m = m[pd.to_numeric(m["events"], errors="coerce").fillna(0) >= min_events]
    cat = f"exclusive_population_{RADIUS_KM}km"
    m = m[pd.to_numeric(m[cat], errors="coerce").fillna(0) > 0]
    m = m.reset_index(drop=True)
    m["market_id"] = np.arange(len(m))
    return m


def market_ladders(ex, cand):
    """
    Every room capacity in each market, split by kind and sorted.

    WHY THE CEILING ALONE IS NOT ENOUGH

    The model's capacity variables were both derived from the biggest room:
    `log largest room` and `has a room big enough`. That makes a market whose
    ONLY room is a 20,000 arena score "big enough" for an act that plays to
    800 people -- which is plainly wrong. The act cannot fill it and would not
    book it. The ceiling says what a city can host at most, not whether it can
    host THIS act well.

    What actually matters is the fit between the act and the ladder of rooms
    the market has. So the ladder comes through to the choice table, and
    `best_room` below records the closest rung.
    """
    v = ex["venues"].copy()
    v["capacity"] = pd.to_numeric(v["capacity"], errors="coerce")
    city_to_market = ex["cities"].set_index(["countryCode", "city"])["market"]
    v["market"] = pd.MultiIndex.from_arrays(
        [v["countryCode"], v["city"]]).map(city_to_market)
    v = v.dropna(subset=["capacity", "market"])
    io = v["outside_inside"].astype(str).str.lower().str.strip()

    out = {}
    for (mk, cc), g in v.groupby(["market", "countryCode"]):
        gi = io.loc[g.index]
        out[(cc, mk)] = {
            "indoor": np.sort(g.loc[gi == "inside", "capacity"].values),
            "outdoor": np.sort(g.loc[gi == "outside", "capacity"].values),
            "either": np.sort(g["capacity"].values),
        }
    return out


def nearest_room(ladder, need, kind):
    """
    The rung closest to what the act plays, measured in LOG capacity.

    Log distance because the gap between 1,000 and 2,000 seats matters far more
    than between 41,000 and 42,000 -- the same reasoning as the venue-level
    assignment rule. Falls back to every room when the market has none of the
    matching kind, which is the generous reading used throughout.
    """
    if ladder is None or need is None or not np.isfinite(need):
        return np.nan
    arr = ladder.get(kind if kind in ("indoor", "outdoor") else "either")
    if arr is None or not len(arr):
        arr = ladder.get("either")
    if arr is None or not len(arr):
        return np.nan
    i = int(np.argmin(np.abs(np.log(np.clip(arr, 50, None))
                             - np.log(max(float(need), 50)))))
    return float(arr[i])


def tour_market_visits(ex, cand):
    """
    Which markets each tour actually played, with the rooms it used there.

    tour_city is at city grain; markets fold near-neighbour cities together
    (Assago into Milan), so visits are summed up to market grain first. A tour
    that played Assago and central Milan visited the Milan market once, not
    twice, and counting it twice would tell the model that Milan is picked more
    often than it is.
    """
    tc = ex["tour_city"].copy()
    city_to_market = ex["cities"].set_index(["countryCode", "city"])["market"]
    tc["market"] = pd.MultiIndex.from_arrays(
        [tc["countryCode"], tc["city"]]).map(city_to_market)
    tc = tc.dropna(subset=["market"])

    g = (tc.groupby(["tour", "countryCode", "market"], as_index=False)
         .agg(events=("events", "sum"),
              indoor_events=("indoor_events", "sum"),
              outdoor_events=("outdoor_events", "sum"),
              room_played=("largest_capacity_played", "max"),
              indoor_room=("largest_indoor_played", "max"),
              outdoor_room=("largest_outdoor_played", "max")))
    # keep only visits to markets that are on the menu
    keys = cand.set_index(["countryCode", "city"])["market_id"]
    g["market_id"] = pd.MultiIndex.from_arrays(
        [g["countryCode"], g["market"]]).map(keys)
    return g.dropna(subset=["market_id"]).astype({"market_id": int})


def build(ex, min_cities=MIN_CITIES, min_market_events=None,
          countries=None, progress=True):
    """
    Assemble the full (occasion x alternative) table.

    Returns (rows, markets, tours_used). `rows` carries one line per candidate
    market per occasion, with `chosen` marking the real visit.
    """
    cand = candidate_markets(ex, min_market_events)
    if countries:
        cand = cand[cand["countryCode"].isin([c.upper() for c in countries])]
        cand = cand.reset_index(drop=True)
        cand["market_id"] = np.arange(len(cand))
    visits = tour_market_visits(ex, cand)
    visits = visits[visits["countryCode"].isin(set(cand["countryCode"]))]
    ladders = market_ladders(ex, cand)

    tour_meta = ex["tours"].set_index("tour")[["headliner", "category"]]

    frames = []
    for code, menu in cand.groupby("countryCode"):
        menu = menu.reset_index(drop=True)
        n_alts = len(menu)
        # position of each market_id within this country's menu
        pos = pd.Series(np.arange(n_alts), index=menu["market_id"].values)
        D = pairwise_km(menu["lat"].values, menu["lon"].values)

        cat_col = f"exclusive_population_{RADIUS_KM}km"
        catchment = pd.to_numeric(menu[cat_col], errors="coerce").values
        income = pd.to_numeric(
            menu[f"purchasing_power_per_head_eur_{RADIUS_KM}km"],
            errors="coerce").values
        ceil_in = pd.to_numeric(menu["largest_indoor_capacity"],
                                errors="coerce").values
        ceil_out = pd.to_numeric(menu["largest_outdoor_capacity"],
                                 errors="coerce").values
        ceil_any = pd.to_numeric(menu["largest_venue_capacity"],
                                 errors="coerce").values
        market_events = pd.to_numeric(menu["events"], errors="coerce").values
        menu_lads = [ladders.get((code, m)) for m in menu["city"].values]

        here = visits[visits["countryCode"] == code]
        n_done = 0
        for tour, grp in here.groupby("tour", sort=False):
            if len(grp) < min_cities:
                continue
            idx = pos.reindex(grp["market_id"].values).values
            rooms = pd.to_numeric(grp["room_played"], errors="coerce").values
            indoor = pd.to_numeric(grp["indoor_events"], errors="coerce").fillna(0).values
            outdoor = pd.to_numeric(grp["outdoor_events"], errors="coerce").fillna(0).values
            dates = pd.to_numeric(grp["events"], errors="coerce").fillna(0).values
            n_vis = len(idx)

            for k in range(n_vis):
                # --- the leave-one-out reference set: this tour's OTHER stops
                others = np.delete(idx, k)
                others = others[~pd.isna(others)]
                if not len(others):
                    continue
                others = others.astype(int)

                # distance from every candidate to the nearest other stop.
                # Same reference set for all alternatives, including the one
                # actually chosen -- see the module docstring.
                routing = D[:, others].min(axis=1)

                # THE MENU EXCLUDES CITIES THIS TOUR ALREADY PLAYED.
                #
                # This is both the correct formulation and the fix for a leak
                # that produced a flatly wrong answer on the first run. The
                # tour's other stops are themselves markets, so they sit on the
                # menu -- and their distance to the nearest other stop is zero,
                # because the nearest other stop IS them. They are also never
                # the chosen alternative on this occasion, since the chosen one
                # is the focal city. So "routing distance of zero" became a
                # perfect predictor of "not chosen", and the fitted routing
                # coefficient came out at +0.96: the model cheerfully reported
                # that acts prefer cities FAR from their other dates.
                #
                # Removing them is right on its own terms too. A tour choosing
                # its third Italian city is not weighing the two it has already
                # booked; the real choice set is the cities still available.
                avail = np.ones(n_alts, dtype=bool)
                avail[others] = False

                # the size of room this act works in, from its other stops
                other_rooms = np.delete(rooms, k)
                other_rooms = other_rooms[np.isfinite(other_rooms)]
                room_needed = (float(np.median(other_rooms))
                               if len(other_rooms) else np.nan)

                # indoor or outdoor act, again from its other stops
                oi, oo = np.delete(indoor, k).sum(), np.delete(outdoor, k).sum()
                kind = "indoor" if oi > oo else ("outdoor" if oo > oi else "either")
                ceiling = {"indoor": ceil_in, "outdoor": ceil_out}.get(kind, ceil_any)

                # The closest rung in each market to what this act plays.
                # Recomputed per occasion because `room_needed` is
                # leave-one-out and therefore differs between a tour's own
                # stops.
                best = np.array([nearest_room(L, room_needed, kind)
                                 for L in menu_lads])

                frames.append(pd.DataFrame({
                    "tour": tour,
                    "countryCode": code,
                    "occasion": f"{tour}|{code}|{k}",
                    "market_id": menu["market_id"].values[avail],
                    "market": menu["city"].values[avail],
                    "chosen": (np.arange(n_alts) == idx[k]).astype(float)[avail],
                    "catchment": catchment[avail],
                    "income": income[avail],
                    "ceiling": ceiling[avail],
                    "best_room": best[avail],
                    "ceiling_indoor": ceil_in[avail],
                    "market_events": market_events[avail],
                    "routing_km": routing[avail],
                    "room_needed": room_needed,
                    "act_plays": kind,
                    # How many cities this tour played in the country. A
                    # chooser-level number, so it cannot enter the model on its
                    # own -- but interacted with an alternative-varying
                    # attribute it can, and it is the natural way to ask
                    # whether a three-city stadium run and an eight-city club
                    # run weigh the same things.
                    "stops_in_country": n_vis,
                    "dates_here": dates[k],
                    # How many cities this tour played in this country. A
                    # chooser-level number, so it cannot enter the model on its
                    # own -- it is the same for every alternative on the menu
                    # and would cancel. It earns its place only as an
                    # interaction: a three-city stadium run and an eight-city
                    # club run do not space their dates the same way.
                    "stops_in_country": n_vis,
                    # the tour's own dates in this market, so specification B
                    # can subtract them and avoid regressing the outcome on
                    # itself (see layer2.py)
                    "own_dates_in_market": np.where(
                        np.arange(n_alts) == idx[k], dates[k], 0.0)[avail],
                }))
            n_done += 1
            if progress and n_done % 200 == 0:
                log(f"   {code}: {n_done:,} tours")

    if not frames:
        raise SystemExit("no usable choice occasions")
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.join(rows["tour"].map(tour_meta["category"]).rename("category"))
    return rows, cand, rows["tour"].nunique()


def describe(rows, cand):
    """A plain-language account of what the sample actually contains."""
    occ = rows["occasion"].nunique()
    per = rows.groupby("countryCode")["occasion"].nunique().sort_values(ascending=False)
    menu = rows.groupby("countryCode")["market_id"].nunique()
    out = pd.DataFrame({"choice occasions": per, "markets on the menu": menu})
    out["rows"] = out["choice occasions"] * out["markets on the menu"]
    out.loc["ALL"] = [occ, len(cand), len(rows)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", default=None)
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--min-cities", type=int, default=MIN_CITIES)
    ap.add_argument("--min-market-events", type=int, default=None,
                    help="flat event count instead of the per-year rate")
    a = ap.parse_args()

    ex = load_extract(a.extract or latest_extract())
    log("building choice occasions ...")
    rows, cand, n_tours = build(ex, a.min_cities, a.min_market_events, a.countries)
    log(f"{len(rows):,} rows, {rows['occasion'].nunique():,} occasions, "
        f"{n_tours:,} tours")
    print(describe(rows, cand).to_string())


if __name__ == "__main__":
    main()
