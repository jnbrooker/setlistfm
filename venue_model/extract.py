#!/usr/bin/env python3
"""
Build a versioned extract for the venue model.

WHY AN EXTRACT AND NOT THE LIVE DATABASE

`setlistfm.db` is 5.7 GB and changes continuously while scrapers and the
enrichment crawler run. Four consequences:

  * a figure quoted in a report cannot be reproduced later,
  * the app is slow, because every interaction rescans 3.5M events,
  * the app breaks during a pipeline run, when `venues` is briefly dropped,
  * licensed Pollstar data has to travel with it.

So this writes a small, self-contained SQLite file plus a MANIFEST that records
exactly what went into it. A chart can then cite its own provenance, which is
the same discipline as the `capacity_source` columns in the main database.

WHAT IT PRODUCES

  extracts/<date>/extract.db
      cities      one row per city: coordinates, event counts, and catchment
                  demographics at five radii
      venues      one row per venue: capacity, type, indoor/outdoor, and where
                  each of those facts came from
      tours       one row per tour: headliner, tier, dates, countries, cities
      tour_city   one row per (tour, city it played) -- the raw material for
                  the choice model in layer 2
      features    a data dictionary: every column, its units and its caveats
      notes       method limitations, carried so the UI can display them

  extracts/<date>/MANIFEST.json
      row counts, date range, source database sizes and modification times

USAGE
    python extract.py
    python extract.py --since 2015-01-01 --categories A B C D E
    python extract.py --countries IT ES --out-dir extracts
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # so `paths` resolves

import paths                                        # noqa: E402
from catchment import (COVERED, DEFAULT_RADII_KM, FEATURE_NOTES,   # noqa: E402
                       METHOD_NOTES, catchment_for_cities, exclusive_catchment,
                       load_postcodes)

DEMO_DB = r"C:\database-creation\demographicdata.db"

# Which artist tiers count as "a show" for this model. A/B/C are the acts with
# a measurable audience; D and E are mostly small local bookings and including
# them would drown the signal we are looking for. Overridable, and whatever is
# chosen is recorded in the manifest, because it materially changes every count.
# Artist tiers the model is fitted on.
#
# D is included, and it is the largest of the four by tour count: 4,348 tours
# against 897 for A and 2,745 for B. Those acts are almost never the ones a new
# arena is built for -- they average 3,057 capacity -- but they are exactly the
# traffic that says whether a market's MID-SIZE rooms fit, which is what the
# `log fit gap` variable measures.
#
# E is deliberately excluded. At 21,182 tours averaging 1,680 capacity it would
# swamp the sample with acts no venue decision turns on, and every probability
# in the model would be diluted by menus full of cities competing for club
# shows.
DEFAULT_CATEGORIES = ("Category A", "Category B", "Category C", "Category D")

# A city needs at least this many shows before it is worth modelling. Below it
# the counts are too noisy to support any statement.
MIN_CITY_EVENTS = 5

# A market must host at least this many shows before it can claim territory in
# the exclusive catchment. Otherwise a village with three gigs would take a
# slice of population away from the city next door, which is not how anybody
# chooses where to see a band.
MIN_MARKET_EVENTS_FOR_CATCHMENT = 20

# Cities closer than this to a bigger neighbour are folded into it as one MARKET.
#
# Why this exists: the Unipol Forum is in Assago, 7.9 km from Milan, and the
# Unipol Arena is in Casalecchio di Reno, 5.4 km from Bologna. Left alone, a
# tour playing the Unipol Forum counts as "did not play Milan" -- nonsense,
# and it would teach the choice model that Milan gets skipped far more often
# than it does.
#
# The threshold is a judgement, not a fact, so it is a parameter and every
# merge it makes is written to the market_members table for inspection. At
# 15 km it catches Assago, Casalecchio and Segrate while leaving Caserta
# (28.7 km from Naples) and Molfetta (23.1 km from Bari) separate, which
# matches how a promoter sees them: a tour might play Naples AND Caserta, but
# never Milan AND Assago.
DEFAULT_MARKET_RADIUS_KM = 15


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def assign_markets(cities, radius_km):
    """
    Fold near-neighbour cities into a single market.

    Greedy and deterministic: work through cities busiest first; each either
    joins the nearest already-established market within `radius_km`, or founds
    a new one. The busiest city names the market, so the Milan market is called
    Milan rather than Assago.

    Greedy rather than a clustering algorithm on purpose -- it explains in two
    sentences, gives the same answer every run, and the anchor is always the
    city a reader expects. k-means on coordinates would be none of those.

    Returns (cities with `market` and `market_distance_km`, a table of merges).
    """
    cities = cities.sort_values("events", ascending=False).reset_index(drop=True)
    market_of = []
    anchors = {}          # country code -> [(market name, lat, lon), ...]

    for r in cities.itertuples(index=False):
        best, best_km = None, None
        for name, alat, alon in anchors.get(r.countryCode, []):
            la1, lo1, la2, lo2 = map(np.radians, [alat, alon, r.lat, r.lon])
            d = 6371.0 * 2 * np.arcsin(np.sqrt(
                np.sin((la2 - la1) / 2) ** 2
                + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
            if d <= radius_km and (best_km is None or d < best_km):
                best, best_km = name, float(d)
        if best is None:
            anchors.setdefault(r.countryCode, []).append((r.city, r.lat, r.lon))
            market_of.append((r.city, 0.0))
        else:
            market_of.append((best, round(best_km, 1)))

    cities["market"] = [m for m, _ in market_of]
    cities["market_distance_km"] = [d for _, d in market_of]
    merges = (cities[cities["market"] != cities["city"]]
              [["market", "city", "countryCode", "market_distance_km", "events"]]
              .sort_values(["countryCode", "market", "events"], ascending=[True, True, False]))
    return cities, merges


def aggregate_markets(cities):
    """
    Market-level totals -- the grain the choice model works at.

    Events and venues sum across member cities. Capacities take the maximum,
    because what matters to a promoter is the biggest room anywhere in the
    market. Coordinates come from the anchor city rather than a centroid of the
    members, so catchment is measured from the place a reader recognises.
    """
    anchor = (cities[cities["market_distance_km"] == 0]
              .set_index(["countryCode", "market"])[["lat", "lon", "country"]])
    sums = ["events", "venues", "headliners", "tours", "events_cat_a", "events_cat_b",
            "events_cat_c", "events_indoor", "events_outdoor"]
    maxes = ["largest_venue_capacity", "largest_indoor_capacity", "largest_outdoor_capacity"]
    g = cities.groupby(["countryCode", "market"]).agg(
        **{c: (c, "sum") for c in sums},
        **{c: (c, "max") for c in maxes},
        cities_merged=("city", "nunique"),
        member_cities=("city", lambda s: ", ".join(sorted(set(s)))),
        first_event=("first_event", "min"), last_event=("last_event", "max"))
    return g.join(anchor).reset_index().rename(columns={"market": "city"})


# --------------------------------------------------------------- the pulls ---

def pull_cities(con, since, categories, countries):
    """
    One row per city, with its coordinate and what happens there now.

    The coordinate is the mean of its events' coordinates. In the source data
    most venues carry a city-centroid coordinate rather than their own, which
    is a problem for venue-level distance work but is exactly right here: we
    want the middle of the city, and that is what we get.
    """
    marks = ",".join("?" * len(categories))
    cmarks = ",".join("?" * len(countries))
    sql = f"""
        SELECT city, country, countryCode,
               AVG(CAST(latitude  AS REAL)) AS lat,
               AVG(CAST(longitude AS REAL)) AS lon,
               COUNT(*)                        AS events,
               COUNT(DISTINCT venue_uid)       AS venues,
               COUNT(DISTINCT headliner)       AS headliners,
               COUNT(DISTINCT NULLIF(tour,'')) AS tours,
               SUM(category = 'Category A')    AS events_cat_a,
               SUM(category = 'Category B')    AS events_cat_b,
               SUM(category = 'Category C')    AS events_cat_c,
               SUM(io_inside)                  AS events_indoor,
               SUM(io_outside)                 AS events_outdoor,
               MAX(venue_capacity)             AS largest_venue_capacity,
               MAX(CASE WHEN io_inside  THEN venue_capacity END) AS largest_indoor_capacity,
               MAX(CASE WHEN io_outside THEN venue_capacity END) AS largest_outdoor_capacity,
               MIN(date_iso) AS first_event, MAX(date_iso) AS last_event
        FROM (
            SELECT e.*,
                   LOWER(TRIM(COALESCE(e.venue_outside_inside, e.arena_outside_inside, ''))) = 'inside'  AS io_inside,
                   LOWER(TRIM(COALESCE(e.venue_outside_inside, e.arena_outside_inside, ''))) = 'outside' AS io_outside
            FROM events e
            WHERE e.date_iso >= ?
              AND e.category IN ({marks})
              AND e.countryCode IN ({cmarks})
              AND e.latitude IS NOT NULL AND TRIM(e.latitude) <> ''
        )
        GROUP BY city, country, countryCode
        HAVING events >= ?
    """
    params = [since, *categories, *countries, MIN_CITY_EVENTS]
    return pd.read_sql(sql, con, params=params)


def pull_venues(con, since, categories, countries):
    """
    One row per venue, carrying the provenance of every fact about it.

    `capacity_source` and `outside_inside_source` travel with the numbers
    because a capacity from the curated arena sheet and one inferred from a
    single Pollstar row are not equally trustworthy, and the model should be
    able to say which it used.
    """
    marks = ",".join("?" * len(categories))
    cmarks = ",".join("?" * len(countries))
    sql = f"""
        SELECT e.venue_uid, MAX(e.venue) AS venue, e.city, e.country, e.countryCode,
               COUNT(*) AS events,
               COUNT(DISTINCT e.headliner) AS headliners,
               MAX(e.venue_capacity) AS capacity,
               MAX(e.venue_capacity_source) AS capacity_source,
               MAX(e.venue_type) AS venue_type,
               MAX(COALESCE(e.venue_outside_inside, e.arena_outside_inside)) AS outside_inside,
               MAX(e.arena_id) AS arena_id,
               MIN(e.date_iso) AS first_event, MAX(e.date_iso) AS last_event,
               AVG(e.pollstar_tickets_sold)  AS avg_tickets_sold,
               AVG(e.pollstar_capacity_pct)  AS avg_capacity_pct
        FROM events e
        WHERE e.date_iso >= ? AND e.category IN ({marks}) AND e.countryCode IN ({cmarks})
        GROUP BY e.venue_uid, e.city, e.country, e.countryCode
    """
    v = pd.read_sql(sql, con, params=[since, *categories, *countries])
    # Pull the enrichment source across so the UI can mark a Wikidata capacity
    # differently from a curated one.
    try:
        en = pd.read_sql("""SELECT venue_uid, confidence AS enrichment_confidence,
                                   wikidata_id, opened_year
                            FROM ref_venue_enrichment WHERE status='matched'""", con)
        v = v.merge(en, how="left", on="venue_uid")
    except Exception:
        v["enrichment_confidence"] = None
        v["wikidata_id"] = None
        v["opened_year"] = None
    return v


def pull_tours(con, since, categories, countries):
    """
    One row per tour, and one row per (tour, city) it visited.

    The pair table is the raw material for layer 2: a tour that played five
    Italian cities *chose* those five from every city in the country, and that
    choice is what the model learns from.
    """
    marks = ",".join("?" * len(categories))
    cmarks = ",".join("?" * len(countries))
    # The indoor/outdoor flag has to come along, because a tour that only plays
    # arenas cannot use a football stadium: comparing it against a market's
    # overall capacity ceiling would wrongly conclude the market could host it.
    base = f"""
        FROM (
            SELECT e.*,
                   LOWER(TRIM(COALESCE(e.venue_outside_inside, e.arena_outside_inside,''))) = 'inside'  AS io_inside,
                   LOWER(TRIM(COALESCE(e.venue_outside_inside, e.arena_outside_inside,''))) = 'outside' AS io_outside
            FROM events e
            WHERE e.date_iso >= ? AND e.category IN ({marks}) AND e.countryCode IN ({cmarks})
              AND TRIM(COALESCE(e.tour,'')) <> ''
        ) e
    """
    params = [since, *categories, *countries]
    tours = pd.read_sql(f"""
        SELECT e.tour, MAX(e.headliner) AS headliner, MAX(e.category) AS category,
               COUNT(*) AS events, COUNT(DISTINCT e.city) AS cities,
               COUNT(DISTINCT e.country) AS countries,
               MIN(e.date_iso) AS first_event, MAX(e.date_iso) AS last_event
        {base} GROUP BY e.tour""", con, params=params)
    tour_city = pd.read_sql(f"""
        SELECT e.tour, e.city, e.country, e.countryCode,
               COUNT(*) AS events, MIN(e.date_iso) AS first_event,
               MAX(e.venue_capacity) AS largest_capacity_played,
               SUM(e.io_inside)  AS indoor_events,
               SUM(e.io_outside) AS outdoor_events,
               MAX(CASE WHEN e.io_inside  THEN e.venue_capacity END) AS largest_indoor_played,
               MAX(CASE WHEN e.io_outside THEN e.venue_capacity END) AS largest_outdoor_played,
               MAX(e.venue) AS venue
        {base} GROUP BY e.tour, e.city, e.country, e.countryCode""", con, params=params)
    return tours, tour_city


# ------------------------------------------------------------ the assembly ---

def build_feature_dictionary(radii):
    """
    Every derived column, what it means, and what it does not capture.

    Written into the extract so the front end can explain any figure without a
    human duplicating the caption. If a column is not in here, it should not be
    on screen.
    """
    rows = [
        ("cities", "events", "count", "Shows in the selected artist tiers since the cutoff.", ""),
        ("cities", "events_indoor", "count", "Shows at a venue labelled indoor.",
         "Labels are missing for many venues; indoor + outdoor < events."),
        ("cities", "events_outdoor", "count", "Shows at a venue labelled outdoor.",
         "As above."),
        ("cities", "largest_indoor_capacity", "seats",
         "Biggest indoor room the city has actually hosted a show in.",
         "Venue capacity, not the configuration used on the night. Absent where no capacity is known."),
        ("cities", "largest_outdoor_capacity", "seats",
         "Biggest outdoor site the city has hosted a show at.", "As above."),
        ("venues", "capacity", "seats", "Best available capacity for the venue.",
         "See capacity_source: curated sheet > Wikidata > largest Pollstar-reported figure."),
        ("venues", "avg_capacity_pct", "percent", "Mean reported sell-through.",
         "Only where Pollstar reported box office, which is a minority of shows."),
        ("tour_city", "largest_capacity_played", "seats",
         "Biggest room this tour used in this city.",
         "Used to test whether capacity, rather than distance, was the binding constraint."),
    ]
    for km in radii:
        for field, note in FEATURE_NOTES.items():
            rows.append(("markets", f"{field}_{km}km",
                         "count" if field != "purchasing_power_per_head_eur" else "EUR",
                         f"{note} Within {km} km.",
                         "Straight-line radius; stops at the national border."))
    return pd.DataFrame(rows, columns=["table", "column", "unit", "meaning", "caveat"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=paths.DB)
    ap.add_argument("--demo-db", default=DEMO_DB)
    ap.add_argument("--since", default="2023-01-01")
    ap.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    ap.add_argument("--countries", nargs="+", default=list(COVERED),
                    help="ISO-2 codes; defaults to the demographic coverage")
    ap.add_argument("--radii", nargs="+", type=int, default=list(DEFAULT_RADII_KM))
    ap.add_argument("--market-radius", type=float, default=DEFAULT_MARKET_RADIUS_KM,
                    help="cities closer than this to a bigger neighbour merge into it")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "extracts"))
    ap.add_argument("--tag", default=dt.date.today().isoformat())
    a = ap.parse_args()

    out_dir = os.path.join(a.out_dir, a.tag)
    os.makedirs(out_dir, exist_ok=True)
    out_db = os.path.join(out_dir, "extract.db")

    # Read-only, so this can run while the scrapers and the enrichment crawler
    # are writing. A read-only connection never takes a write lock, so it can
    # neither block them nor be blocked.
    log(f"reading {a.db} (read-only)")
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True, timeout=120)

    log(f"cities: shows since {a.since} in {', '.join(a.categories)}")
    cities = pull_cities(con, a.since, a.categories, a.countries)
    log(f"   {len(cities):,} cities with >= {MIN_CITY_EVENTS} shows")

    cities, merges = assign_markets(cities, a.market_radius)
    log(f"   {cities['market'].nunique():,} markets after folding in "
        f"{len(merges):,} neighbour cities within {a.market_radius:g} km")
    for r in merges.head(8).itertuples(index=False):
        log(f"      {r.city} -> {r.market} ({r.market_distance_km:g} km, {r.events} shows)")

    log("venues ...")
    venues = pull_venues(con, a.since, a.categories, a.countries)
    log(f"   {len(venues):,} venues")

    log("tours ...")
    tours, tour_city = pull_tours(con, a.since, a.categories, a.countries)
    log(f"   {len(tours):,} tours, {len(tour_city):,} tour-city pairs")
    con.close()

    log(f"catchment demographics at {a.radii} km from {a.demo_db}")
    pc = load_postcodes(a.demo_db)
    log(f"   {len(pc):,} postcode districts loaded")
    # Catchment is computed once per MARKET, from the anchor city coordinate.
    # Summing member cities would double-count the overlap, which for Milan and
    # Assago is very nearly all of it.
    markets = aggregate_markets(cities)
    markets = catchment_for_cities(
        pc, markets, a.radii,
        progress=lambda i, n: log(f"   {i:,}/{n:,} markets"))
    log(f"   catchment attached to {len(markets):,} markets")

    # ... and the exclusive version, which is the one the screening pass ranks
    # on. See catchment.exclusive_catchment for why a plain radius misleads.
    markets = exclusive_catchment(pc, markets, min_events=MIN_MARKET_EVENTS_FOR_CATCHMENT)
    got = markets["exclusive_population"].notna().sum()
    log(f"   exclusive catchment for {got:,} markets "
        f"(those with >= {MIN_MARKET_EVENTS_FOR_CATCHMENT} shows claim territory)")

    features = build_feature_dictionary(a.radii)
    notes = pd.DataFrame({"note": METHOD_NOTES + [
        f"Artist tiers included: {', '.join(a.categories)}.",
        f"Shows on or after {a.since}.",
        "Counts are shows, not attendance.",
        f"Cities within {a.market_radius:g} km of a bigger neighbour are merged into it as "
        f"one market (Assago into Milan, Casalecchio di Reno into Bologna). Every merge is "
        f"listed in the market_members table.",
        "This extract is a snapshot; the source database changes continuously.",
    ]})

    if os.path.exists(out_db):
        os.remove(out_db)
    with sqlite3.connect(out_db) as out:
        for name, frame in (("markets", markets), ("cities", cities),
                            ("market_members", merges), ("venues", venues),
                            ("tours", tours), ("tour_city", tour_city),
                            ("features", features), ("notes", notes)):
            frame.to_sql(name, out, index=False)
            log(f"   wrote {name}: {len(frame):,} rows")
        out.execute("CREATE INDEX ix_tc_tour ON tour_city(tour)")
        out.execute("CREATE INDEX ix_tc_city ON tour_city(city, countryCode)")
        out.execute("CREATE INDEX ix_v_city  ON venues(city, countryCode)")

    def stamp(path):
        return {"path": path, "bytes": os.path.getsize(path),
                "modified": dt.datetime.fromtimestamp(
                    os.path.getmtime(path)).isoformat(timespec="seconds")} \
            if os.path.exists(path) else {"path": path, "missing": True}

    manifest = {
        "tag": a.tag,
        "built_at": dt.datetime.now().isoformat(timespec="seconds"),
        "since": a.since,
        "categories": a.categories,
        "countries": a.countries,
        "radii_km": a.radii,
        "min_city_events": MIN_CITY_EVENTS,
        "market_radius_km": a.market_radius,
        "rows": {"markets": len(markets), "cities": len(cities),
                 "market_members": len(merges), "venues": len(venues),
                 "tours": len(tours), "tour_city": len(tour_city)},
        "event_date_range": [str(cities["first_event"].min()), str(cities["last_event"].max())],
        "sources": {"events": stamp(a.db), "demographics": stamp(a.demo_db)},
        "caveats": list(notes["note"]),
    }
    with open(os.path.join(out_dir, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log(f"extract written to {out_db}")
    log(f"   {len(markets):,} markets ({len(cities):,} cities) | "
        f"{len(venues):,} venues | {len(tours):,} tours")
    log(f"   quote this extract as: {a.tag}")


if __name__ == "__main__":
    sys.exit(main())
