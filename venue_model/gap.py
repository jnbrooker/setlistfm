#!/usr/bin/env python3
"""
Layer 1 -- the descriptive picture. No model, just counting.

WHAT THIS IS FOR

Before fitting anything, establish whether there is a gap worth modelling. If
the counting does not show one, no amount of conditional logit will manufacture
it, and the honest answer to "should we build here" is already available.

Everything here is arithmetic on observed shows. Nothing is predicted, nothing
is inferred, and every figure can be traced to rows you can list.

THE FIVE QUESTIONS IT ANSWERS

  1. What rooms does this market actually have?          capacity_ladder()
  2. Where are the holes in that ladder?                 ladder_gaps()
  3. Which tours played the country and skipped here?    tours_that_skipped()
  4. For each, WAS CAPACITY THE REASON?                  capacity_test()
  5. Which markets caught the dates this one missed?     where_skippers_went()

Plus a benchmark: peer_markets() finds markets of similar catchment and shows
what they have and what they get, which is the closest thing to a controlled
comparison available without a model.

QUESTION 4 IS THE ONE THAT MATTERS

It is easy to show that tours skip a city. It is much harder, and much more
useful, to say whether they skipped it *because the rooms are too small*.

The test: for each tour that skipped, look at the size of room it used
elsewhere. If it played 9,000-seat halls across the country and this market's
biggest indoor room holds 10,800, then capacity was not the obstacle -- the
detour, the catchment or the routing was. Only tours that consistently played
rooms bigger than anything available here are evidence of a capacity gap.

Without this test, "198 tours skipped Bari" is a number that sounds like an
argument and is not one.

USAGE
    python gap.py Bari
    python gap.py Naples --min-dates 4
    python gap.py Milan --country IT --out reports
"""

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))       # reuse the pipeline's labeller

try:
    from venue_io import infer_from_name
except ImportError:                              # keep gap.py standalone-able
    infer_from_name = lambda _name: None

# Room-size bands used to describe a capacity ladder. Chosen to match how the
# industry talks about rooms (club / theatre / mid / arena / big arena /
# stadium) rather than round numbers for their own sake.
LADDER_BANDS = [
    (0, 1_500, "club (under 1,500)"),
    (1_500, 3_500, "theatre (1,500-3,499)"),
    (3_500, 8_000, "mid-scale (3,500-7,999)"),
    (8_000, 15_000, "arena (8,000-14,999)"),
    (15_000, 30_000, "large arena (15,000-29,999)"),
    (30_000, 10 ** 9, "stadium (30,000+)"),
]


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------ loading ------

def latest_extract(base=None):
    """Most recent extract folder, so the CLI needs no path in normal use."""
    base = base or os.path.join(HERE, "extracts")
    tags = sorted(d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d)))
    if not tags:
        raise SystemExit(f"no extracts under {base} - run extract.py first")
    return os.path.join(base, tags[-1])


def load_extract(folder):
    """
    Read an extract into memory.

    Returns a dict of DataFrames plus the manifest, so anything using this can
    state which extract produced its numbers. That is not decoration: a figure
    that cannot name its source cannot be defended six months later.
    """
    db = os.path.join(folder, "extract.db")
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        ex = {n: pd.read_sql(f"SELECT * FROM {n}", con) for n in names}
    with open(os.path.join(folder, "MANIFEST.json"), encoding="utf-8") as f:
        ex["manifest"] = json.load(f)
    return ex


def resolve_market(ex, name, country=None):
    """
    Find the market, and refuse to guess between same-named ones.

    Accepts either a market name or any city folded into one, so asking for
    "Assago" quietly gets you Milan -- and says so.
    """
    markets = ex["markets"]
    m = markets["city"].str.casefold() == name.strip().casefold()
    if country:
        m &= markets["countryCode"].str.upper() == country.upper()
    hit = markets[m]

    if hit.empty:                       # maybe they named a merged-in city
        members = ex["cities"]
        mm = members["city"].str.casefold() == name.strip().casefold()
        if country:
            mm &= members["countryCode"].str.upper() == country.upper()
        if mm.any():
            row = members[mm].iloc[0]
            log(f"'{name}' is part of the {row['market']} market "
                f"({row['market_distance_km']:g} km away) - reporting on {row['market']}")
            return resolve_market(ex, row["market"], row["countryCode"])
        near = markets[markets["city"].str.contains(name.strip(), case=False, na=False)]
        log(f"no market called {name!r}")
        for r in near.head(8).itertuples(index=False):
            log(f"   did you mean: {r.city}, {r.country} ({r.events} shows)")
        raise SystemExit(2)

    if hit["countryCode"].nunique() > 1:
        log(f"{name!r} exists in several countries - pass --country:")
        for r in hit.itertuples(index=False):
            log(f"   {r.countryCode}  {r.city}, {r.country} ({r.events} shows)")
        raise SystemExit(2)
    return hit.iloc[0]


# --------------------------------------------------- 1 & 2: the ladder -----

def capacity_ladder(ex, market_row):
    """
    Every room the market has actually used, largest first.

    Only venues that have hosted a show appear: this describes the working
    inventory, not the building stock. A hall that exists but never books music
    is not a room a promoter can use.
    """
    v = ex["venues"]
    members = set(ex["cities"].loc[
        (ex["cities"]["market"] == market_row["city"]) &
        (ex["cities"]["countryCode"] == market_row["countryCode"]), "city"])
    here = v[(v["city"].isin(members)) &
             (v["countryCode"] == market_row["countryCode"])].copy()
    here["capacity"] = pd.to_numeric(here["capacity"], errors="coerce")
    here["band"] = here["capacity"].map(band_of)

    # Fill missing indoor/outdoor from the venue name.
    #
    # This is not cosmetic. Bari's Stadio San Nicola holds 55,109 and carries no
    # indoor/outdoor label in the source data, so without this the market's
    # outdoor ceiling reads 15,670 -- and the capacity test then reports a
    # "gap" for every act that plays 46,000-seat outdoor shows, when Bari can
    # already host them. A missing label was being read as a missing building.
    #
    # `io_source` records which venues were labelled this way so a reader can
    # discount them; the rule is the same one the main pipeline uses.
    lab = here["outside_inside"].astype(str).str.strip().str.lower()
    known = lab.isin(["inside", "outside"])
    here["io"] = np.where(known, lab, here["venue"].map(infer_from_name))
    here["io_source"] = np.where(known, "database",
                                 np.where(here["io"].notna(), "venue name", "unlabelled"))
    cols = ["venue", "city", "capacity", "band", "io", "io_source", "outside_inside",
            "venue_type", "events", "headliners", "capacity_source",
            "first_event", "last_event"]
    return here[cols].sort_values("capacity", ascending=False, na_position="last")


def ceilings_from_ladder(ladder):
    """
    The biggest room of each kind this market actually has.

    Taken from the ladder rather than the extract so it benefits from the
    name-based fill above, and so the number the verdict is judged against is
    the same number shown on screen. Returns indoor, outdoor and overall, plus
    how much capacity is still sitting behind an unlabelled venue -- because a
    market with a large unlabelled room deserves a caveat, not a verdict.
    """
    cap = pd.to_numeric(ladder["capacity"], errors="coerce")
    inside = cap.where(ladder["io"] == "inside")
    outside = cap.where(ladder["io"] == "outside")
    unl = cap.where(ladder["io_source"] == "unlabelled")
    return {
        "indoor": float(inside.max()) if inside.notna().any() else None,
        "outdoor": float(outside.max()) if outside.notna().any() else None,
        "overall": float(cap.max()) if cap.notna().any() else None,
        "largest_unlabelled": float(unl.max()) if unl.notna().any() else None,
        "unlabelled_venues": int((ladder["io_source"] == "unlabelled").sum()),
    }


def band_of(capacity):
    if capacity is None or pd.isna(capacity):
        return "capacity unknown"
    for lo, hi, label in LADDER_BANDS:
        if lo <= capacity < hi:
            return label
    return LADDER_BANDS[-1][2]


def ladder_gaps(ladder):
    """
    Which rungs of the ladder are missing, indoor and outdoor separately.

    Separately because they are not substitutes. An outdoor site cannot host a
    November show in northern Europe, so a market with a 40,000 field and no
    10,000 hall has a gap for eight months of the year even though its headline
    capacity looks ample.
    """
    rows = []
    for io in ("inside", "outside"):
        sub = ladder[ladder["io"] == io]
        for lo, hi, label in LADDER_BANDS:
            n = int(((sub["capacity"] >= lo) & (sub["capacity"] < hi)).sum())
            rows.append({
                "type": io, "band": label, "venues": n,
                "events": int(sub.loc[(sub["capacity"] >= lo) &
                                      (sub["capacity"] < hi), "events"].sum()),
                "status": "none" if n == 0 else ("thin" if n == 1 else "served"),
            })
    return pd.DataFrame(rows)


# ------------------------------------------- 3 & 4: who skipped, and why ---

def tours_that_skipped(ex, market_row, min_dates=3):
    """
    Tours that toured this country properly and never came here.

    `min_dates` filters out acts that played one date in the capital and went
    home -- those tell you nothing about this market, because they were never
    choosing between cities in the first place.
    """
    tc = ex["tour_city"]
    country = market_row["countryCode"]
    in_country = tc[tc["countryCode"] == country]

    members = set(ex["cities"].loc[
        (ex["cities"]["market"] == market_row["city"]) &
        (ex["cities"]["countryCode"] == country), "city"])

    per_tour = in_country.groupby("tour").agg(
        country_dates=("events", "sum"),
        cities=("city", "nunique"),
        largest_room_played=("largest_capacity_played", "max"),
        typical_room_played=("largest_capacity_played", "median"),
        indoor_dates=("indoor_events", "sum"),
        outdoor_dates=("outdoor_events", "sum"),
        typical_indoor_room=("largest_indoor_played", "median"),
        typical_outdoor_room=("largest_outdoor_played", "median"),
        where=("city", lambda s: ", ".join(sorted(set(s))[:10])))
    # Which kind of room this act actually works in. Ties and unlabelled tours
    # fall to "either", and are then tested against the market's larger ceiling
    # -- the generous reading, so the test cannot manufacture a capacity gap
    # out of missing labels.
    per_tour["plays"] = np.select(
        [per_tour["indoor_dates"] > per_tour["outdoor_dates"],
         per_tour["outdoor_dates"] > per_tour["indoor_dates"]],
        ["indoor", "outdoor"], default="either")
    came = set(in_country.loc[in_country["city"].isin(members), "tour"])
    per_tour["played_here"] = per_tour.index.isin(came)

    skipped = per_tour[(~per_tour["played_here"]) &
                       (per_tour["country_dates"] >= min_dates)]
    meta = ex["tours"].set_index("tour")[["headliner", "category"]]
    return (skipped.join(meta).drop(columns=["played_here"])
            .sort_values("country_dates", ascending=False).reset_index())


# The two multiples that decide every verdict. Defaults, not constants: both
# are arguments to capacity_test and sliders in the app. See its docstring.
GAP_MULTIPLE = 1.2
BORDERLINE_MULTIPLE = 0.8


def capacity_test(skipped, market_row, ceilings=None, gap_mult=GAP_MULTIPLE,
                  borderline_mult=BORDERLINE_MULTIPLE):
    """
    THE CENTRAL DIAGNOSTIC: was the room size actually the obstacle?

    For each tour that skipped, compare the room it typically used elsewhere
    against the biggest room of THE SAME KIND this market has. Four verdicts:

      capacity gap        it consistently played bigger than anything here of
                          that kind. A new room that size removes the obstacle.
      borderline          close to our ceiling; the evidence is weak.
      not capacity        it played rooms we can already match. Something else
                          kept it away -- distance, catchment, routing, or the
                          promoter's judgement. A building would not fix it.
      unknown             no capacity known for the rooms it used.

    MATCHING THE KIND OF ROOM MATTERS. An act that plays 5,000-seat halls
    cannot use a 55,000 football stadium, so testing it against a market's
    overall ceiling would wrongly report that the market could already host it.
    Indoor acts are tested against the indoor ceiling, outdoor against outdoor,
    and anything ambiguous against the larger of the two -- the generous
    reading, so a missing label can never invent a capacity gap.

    THE TWO MULTIPLES ARE ARGUMENTS, NOT FACTS, which is why they are
    arguments to this function. At the 1.2 default an act must typically play
    rooms a fifth larger than anything here of its kind before room size is
    blamed; at 0.8 anything closer than that is reported as inconclusive
    rather than counted either way. Nothing in the data fixes either number.
    The app puts both on sliders so that anyone who suspects a finding is an
    artefact of a cut-off can move the cut-off and watch.

    The honest framing: only the first group is evidence for building, and it
    is usually far smaller than the raw skip count implies. The distance
    between "198 tours skipped us" and "n skipped us for a reason a building
    would fix" is the entire point of this function.
    """
    num = lambda v: pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    ceilings = ceilings or {}
    # Prefer ceilings derived from the ladder (which fills unlabelled venues by
    # name); fall back to the extract's own figures when called without them.
    ceil_in = num(ceilings.get("indoor", market_row.get("largest_indoor_capacity")))
    ceil_out = num(ceilings.get("outdoor", market_row.get("largest_outdoor_capacity")))
    ceil_any = num(ceilings.get("overall", market_row.get("largest_venue_capacity")))

    out = skipped.copy()
    # the room this act uses, and the ceiling it should be judged against
    typical = np.where(out["plays"] == "indoor", out["typical_indoor_room"],
                       np.where(out["plays"] == "outdoor", out["typical_outdoor_room"],
                                out["typical_room_played"]))
    typical = pd.to_numeric(pd.Series(typical, index=out.index), errors="coerce")
    typical = typical.fillna(pd.to_numeric(out["typical_room_played"], errors="coerce"))

    ceiling = pd.Series(
        np.where(out["plays"] == "indoor", ceil_in,
                 np.where(out["plays"] == "outdoor", ceil_out, ceil_any)),
        index=out.index).astype(float)
    # If we have no ceiling of that kind at all, the market simply cannot host
    # that sort of show -- which is itself a capacity gap, not an unknown.
    no_ceiling = ceiling.isna()

    out["room_it_needs"] = typical.round(0)
    out["our_ceiling_of_that_kind"] = ceiling.round(0)
    # Shown as well as used: the ratio is the whole verdict, and a reader who
    # can see it can recompute any row by hand against the two multiples.
    out["ratio"] = (typical / ceiling).round(2)
    out["verdict"] = np.select(
        [typical.isna(),
         no_ceiling,
         typical > ceiling * gap_mult,
         typical > ceiling * borderline_mult],
        ["unknown", "capacity gap (none of this kind)", "capacity gap", "borderline"],
        default="not capacity")
    out["headroom_needed"] = (typical - ceiling).where(typical > ceiling).round(0)
    return out


def capacity_test_summary(tested):
    """One row per verdict -- the headline of the whole layer."""
    g = (tested.groupby("verdict")
         .agg(tours=("tour", "count"),
              country_dates=("country_dates", "sum"),
              median_room_needed=("room_it_needs", "median"),
              our_ceiling=("our_ceiling_of_that_kind", "median"))
         .sort_values("tours", ascending=False).reset_index())
    total = g["tours"].sum()
    g["share of skipping tours"] = (g["tours"] / total).round(4) if total else None
    return g


# ------------------------------------------------- 5: where they went ------

def where_skippers_went(ex, market_row, skipped):
    """
    Which markets absorbed the dates this one did not get.

    Reads as a competitive picture: the markets at the top are the ones this
    one is losing to, and their capacity ladder shows what they offer that this
    market does not.
    """
    tc = ex["tour_city"]
    country = market_row["countryCode"]
    e = tc[(tc["countryCode"] == country) & (tc["tour"].isin(set(skipped["tour"])))]
    city_to_market = ex["cities"].set_index("city")["market"]
    e = e.assign(market=e["city"].map(city_to_market))
    g = (e.groupby("market")
         .agg(tours_caught=("tour", "nunique"), dates=("events", "sum"),
              largest_room_used=("largest_capacity_played", "max"))
         .sort_values("tours_caught", ascending=False).reset_index())
    n = skipped["tour"].nunique()
    g["share of the skipping tours"] = (g["tours_caught"] / n).round(4) if n else None
    return g


# ------------------------------------------------------- the benchmark ----

def peer_markets(ex, market_row, radius_km=60, tolerance=0.35, limit=12):
    """
    Markets with a similar catchment, and what they have and get.

    The nearest thing to a controlled comparison without a model: hold the
    population roughly constant and look at how the capacity ladder and the
    show count differ. If markets this size typically have a 15,000 arena and
    three times the shows, that is evidence. If they look just like this one,
    the gap is not in the building.

    `tolerance` is the half-width of the population band, so 0.35 means +/-35%.
    Wide, because there are only a few hundred markets and a tight band would
    leave too few to compare against.
    """
    col = f"population_{radius_km}km"
    m = ex["markets"].copy()
    pop = pd.to_numeric(m[col], errors="coerce")
    mine = pd.to_numeric(pd.Series([market_row.get(col)]), errors="coerce").iloc[0]
    if mine is None or pd.isna(mine):
        return pd.DataFrame()
    lo, hi = mine * (1 - tolerance), mine * (1 + tolerance)

    peers = m[(pop >= lo) & (pop <= hi)].copy()
    peers["shows_per_million"] = (peers["events"] /
                                  (pd.to_numeric(peers[col], errors="coerce") / 1e6)).round(1)
    peers["this market"] = np.where(
        (peers["city"] == market_row["city"]) &
        (peers["countryCode"] == market_row["countryCode"]), "<<<", "")
    cols = ["city", "country", "events", "shows_per_million", col,
            "largest_indoor_capacity", "largest_outdoor_capacity",
            "events_indoor", "events_outdoor", "venues", "this market"]
    return peers[cols].sort_values("events", ascending=False).head(limit)


# -------------------------------------------------------------- the CLI ---

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("market")
    ap.add_argument("--country", default=None, help="ISO-2 code, if the name is ambiguous")
    ap.add_argument("--extract", default=None, help="extract folder (default: newest)")
    ap.add_argument("--min-dates", type=int, default=3,
                    help="country dates a tour needs before skipping here counts")
    ap.add_argument("--peer-radius", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(HERE, "reports"))
    a = ap.parse_args()

    folder = a.extract or latest_extract()
    ex = load_extract(folder)
    log(f"extract {ex['manifest']['tag']} -- {ex['manifest']['rows']['markets']:,} markets, "
        f"shows since {ex['manifest']['since']}")

    row = resolve_market(ex, a.market, a.country)
    log(f"{row['city']}, {row['country']}: {row['events']} shows, "
        f"{row['venues']} venues, catchment {row.get('population_60km'):,} at 60km")

    ladder = capacity_ladder(ex, row)
    ceilings = ceilings_from_ladder(ladder)
    gaps = ladder_gaps(ladder)
    if ceilings["unlabelled_venues"]:
        log(f"   {ceilings['unlabelled_venues']} venue(s) still have no indoor/outdoor "
            f"label; largest is {ceilings['largest_unlabelled']}")
    skipped = tours_that_skipped(ex, row, a.min_dates)
    tested = capacity_test(skipped, row, ceilings)
    summary = capacity_test_summary(tested)
    went = where_skippers_went(ex, row, skipped)
    peers = peer_markets(ex, row, a.peer_radius)

    os.makedirs(a.out, exist_ok=True)
    safe = "".join(ch for ch in row["city"] if ch.isalnum() or ch in " -_").replace(" ", "_")
    out = os.path.join(a.out, f"{safe}_gap_{ex['manifest']['tag']}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Verdict", index=False)
        ladder.to_excel(xw, sheet_name="Capacity ladder", index=False)
        gaps.to_excel(xw, sheet_name="Ladder gaps", index=False)
        tested.to_excel(xw, sheet_name="Tours that skipped", index=False)
        went.to_excel(xw, sheet_name="Where they went", index=False)
        peers.to_excel(xw, sheet_name="Peer markets", index=False)
        pd.DataFrame([ceilings]).to_excel(xw, sheet_name="Ceilings used", index=False)
        ex["features"].to_excel(xw, sheet_name="Feature dictionary", index=False)
        pd.DataFrame({"note": ex["notes"]["note"].tolist() + [
            f"Extract: {ex['manifest']['tag']}, built {ex['manifest']['built_at']}.",
            f"A tour counts as having skipped this market if it played at least "
            f"{a.min_dates} dates in {row['country']} and none here.",
            "The capacity verdict compares the MEDIAN room a tour used elsewhere with the "
            "largest room OF THE SAME KIND this market has -- indoor acts against the indoor "
            "ceiling, outdoor against outdoor. Above 120% of it is a capacity gap, 80-120% is "
            "borderline, below is not a capacity problem. Tours with no clear indoor/outdoor "
            "lean are tested against the larger ceiling, so missing labels cannot invent a gap.",
            "Venues with no indoor/outdoor label in the database are labelled from their "
            "name (see the io_source column on the capacity ladder). Without this a large "
            "unlabelled stadium makes the outdoor ceiling look far lower than it is.",
            "Layer 1 is descriptive. Nothing here is a forecast and nothing is causal.",
        ]}).to_excel(xw, sheet_name="Methodology", index=False)

    log(f"wrote {out}")
    print(f"\n{row['city']} -- why tours skipped ({len(tested):,} tours with "
          f">= {a.min_dates} {row['country']} dates)\n")
    print(summary.to_string(index=False))
    print(f"\nCeilings this market is judged against: indoor {ceilings['indoor']}, "
          f"outdoor {ceilings['outdoor']}, overall {ceilings['overall']}")
    print(tested.groupby(["plays", "verdict"]).size().rename("tours").reset_index().to_string(index=False))
    print("\nWhere those dates went instead:")
    print(went.head(8).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
