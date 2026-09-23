#!/usr/bin/env python3
"""
Catchment demographics — who lives within reach of a point.

WHAT THIS ANSWERS

Given a coordinate (a city centre, or a proposed venue site), how many people
live within N km of it, how old are they, and how much money do they have.

WHY IT MATTERS TO THE MODEL

"Market size" is the variable most likely to be confounded with venue capacity:
big cities have big arenas, so a model that omits catchment will credit the
arena with everything the city would have got anyway. Getting this right is
what stops layer 2 being a very confident restatement of "large places are
large".

WHERE THE NUMBERS COME FROM

`demographicdata.db`:
  mbi_master          one row per postcode district, with population, age bands
                      and purchasing power
  postcode_locations  a latitude/longitude centroid for each of those districts

Coverage is seven countries: DE, GB, FR, IT, ES, PT, CH. Everything else
returns nulls rather than zeros, because "no data" and "nobody lives there" are
different statements and must not be confused downstream.

THE MAIN APPROXIMATION, STATED UP FRONT

A postcode district is counted as entirely inside or entirely outside the
radius, based on its centroid. There is no partial inclusion. For a 30km+
radius against districts a few km across this is a small error and it averages
out; at very small radii it would not, which is why 10km is the smallest
offered and why the returned `postcodes` count is exposed — a catchment built
from three districts deserves less trust than one built from four hundred.

Distances are straight-line (great-circle), not drive time. A coastal city
loses half its circle to the sea, and mountains are ignored entirely. Bari and
Milan are therefore not quite comparable on a raw radius, which is a real
limitation and is noted wherever catchment is used.
"""

import math
import sqlite3

import numpy as np
import pandas as pd

# The radii the extract computes for every city. Several, rather than one,
# because the right catchment differs by act: a club show draws from the city,
# a stadium show draws from the region.
DEFAULT_RADII_KM = (10, 30, 60, 90, 120)

# Countries mbi_master covers. Anything else gets nulls.
COVERED = ("DE", "GB", "FR", "IT", "ES", "PT", "CH")

EARTH_RADIUS_KM = 6371.0


def load_postcodes(demo_db):
    """
    Every postcode district we have both demographics and a location for.

    Returns one DataFrame held in memory (~43,000 rows, a few MB) because the
    alternative -- a SQL query per city -- would run 3,684 times. Loading once
    and slicing with NumPy is around a thousand times faster.
    """
    sql = """
        SELECT m.ctrycode, m.concated AS postcode, m.name AS area,
               m.p_t        AS population,
               m.hh_t       AS households,
               m.AGE_T1529  AS age_15_29,
               m.AGE_T3044  AS age_30_44,
               m.AGE_T4559  AS age_45_59,
               m.AGE_T60PL  AS age_60_plus,
               m.PP_EURO    AS purchasing_power_per_head_eur,
               m.PP_MIO     AS purchasing_power_total_meur,
               l.latitude   AS lat,
               l.longitude  AS lon
        FROM mbi_master m
        JOIN postcode_locations l ON l.postcode = m.concated
        WHERE l.latitude IS NOT NULL AND l.longitude IS NOT NULL
    """
    with sqlite3.connect(f"file:{demo_db}?mode=ro", uri=True, timeout=60) as con:
        pc = pd.read_sql(sql, con)

    # Pre-convert to radians once. Every distance calculation needs them, and
    # doing it here rather than per city saves 3,684 repetitions.
    pc["lat_rad"] = np.radians(pc["lat"])
    pc["lon_rad"] = np.radians(pc["lon"])
    return pc


def haversine_km(lat_deg, lon_deg, lat_rad_arr, lon_rad_arr):
    """
    Great-circle distance from one point to many, in kilometres.

    The haversine formula. Written out rather than imported so the arithmetic
    is inspectable:

        a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
        d = 2R * arcsin(sqrt(a))

    It treats the Earth as a sphere, which is wrong by up to ~0.5% -- far
    smaller than the error already introduced by using postcode centroids, so
    the extra complexity of an ellipsoidal formula would buy nothing here.

    Vectorised: `lat_rad_arr` is a NumPy array of many points and the whole
    array is computed in one pass, in C, with no Python loop.
    """
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    dlat = lat_rad_arr - lat1
    dlon = lon_rad_arr - lon1
    a = (np.sin(dlat / 2.0) ** 2
         + math.cos(lat1) * np.cos(lat_rad_arr) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def catchment_for_point(pc, lat, lon, country_code, radii_km=DEFAULT_RADII_KM):
    """
    Demographics within each radius of one point.

    `pc` is the frame from load_postcodes(). `country_code` restricts the
    search to one country -- both a large speed-up and a deliberate modelling
    choice: a catchment that reaches across a national border is usually not a
    real market, because touring, ticketing and media all stop at the border.
    Feeding a Swiss city's German hinterland into its catchment would overstate
    it. The cost is that genuinely cross-border cities (Basel, Aachen, Lille)
    are understated, which is noted rather than corrected.

    Returns a flat dict, one set of keys per radius, so it drops straight into
    a DataFrame row.
    """
    out = {}
    if country_code not in COVERED or lat is None or lon is None:
        # Nulls, not zeros. "We have no data for Poland" must never be read as
        # "nobody lives near this Polish city".
        for km in radii_km:
            for field in ("population", "age_15_29", "age_30_44", "households",
                          "purchasing_power_total_meur", "postcodes"):
                out[f"{field}_{km}km"] = None
            out[f"purchasing_power_per_head_eur_{km}km"] = None
        return out

    here = pc[pc["ctrycode"] == country_code]
    if here.empty:
        return catchment_for_point(pc, lat, lon, "__none__", radii_km)

    dist = haversine_km(lat, lon, here["lat_rad"].values, here["lon_rad"].values)

    for km in radii_km:
        inside = dist <= km
        sel = here[inside]
        pop = sel["population"].sum()
        out[f"population_{km}km"] = int(pop) if pop == pop else None
        out[f"age_15_29_{km}km"] = int(sel["age_15_29"].sum(skipna=True))
        out[f"age_30_44_{km}km"] = int(sel["age_30_44"].sum(skipna=True))
        out[f"households_{km}km"] = int(sel["households"].sum(skipna=True))
        # Total spending power in the catchment, in millions of euros. This is
        # the one to compare across cities -- per-head figures say how rich the
        # average resident is, which is a different question.
        out[f"purchasing_power_total_meur_{km}km"] = round(
            float(sel["purchasing_power_total_meur"].sum(skipna=True)), 1)
        # Population-weighted, NOT a plain mean of the districts: a mean would
        # let a tiny wealthy district count as much as a large poor one.
        out[f"purchasing_power_per_head_eur_{km}km"] = (
            round(float((sel["purchasing_power_per_head_eur"] * sel["population"]).sum()
                        / pop), 0) if pop else None)
        # How many districts the figure rests on. A catchment assembled from
        # three districts is a much weaker number than one from four hundred,
        # and the UI surfaces this.
        out[f"postcodes_{km}km"] = int(inside.sum())
    return out


def catchment_for_cities(pc, cities, radii_km=DEFAULT_RADII_KM, progress=None):
    """
    Run catchment_for_point over a DataFrame of cities.

    `cities` needs columns: city, countryCode, lat, lon. Returns the same rows
    with the catchment columns joined on.
    """
    rows = []
    for i, r in enumerate(cities.itertuples(index=False), start=1):
        rows.append(catchment_for_point(pc, r.lat, r.lon, r.countryCode, radii_km))
        if progress and i % 250 == 0:
            progress(i, len(cities))
    return pd.concat([cities.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# What every catchment column means, carried into the extract so the front end
# can explain any figure it shows without a human writing the caption twice.
FEATURE_NOTES = {
    "population": "Residents in postcode districts whose centroid is inside the radius.",
    "age_15_29": "Residents aged 15-29 -- the heaviest live-music attendance band.",
    "age_30_44": "Residents aged 30-44 -- highest spend per head on live events.",
    "households": "Households, a better denominator than population for spend.",
    "purchasing_power_total_meur": "Total disposable spending power in the catchment, EUR millions.",
    "purchasing_power_per_head_eur": "Disposable spending power per resident, population-weighted.",
    "postcodes": "How many postcode districts the figure is built from; low counts mean a fragile estimate.",
}

FEATURE_NOTES.update({
    "exclusive_population": ("Residents for whom this is the NEAREST market hosting shows. "
                            "Unlike the radius figures nobody is counted twice, so a town "
                            "beside a big city is not credited with that city's population."),
    "exclusive_mean_km": ("Average distance those residents live from the market. A "
                          "high value means the figure is built from people who would "
                          "have a long journey, and should be discounted accordingly."),
    "exclusive_population_60km": ("Residents for whom this is the nearest market hosting "
                                 "shows AND who live within 60 km. The figure to rank on: "
                                 "directly comparable to population_60km beside it, with a "
                                 "neighbouring city's residents removed."),
})

METHOD_NOTES = [
    "Exclusive catchment assigns each postcode to its single nearest market that hosts shows "
    "(Voronoi). It understates big cities and overstates small ones slightly, because "
    "people do travel past a small market to reach a large one; the radius figures are "
    "kept alongside so both readings are available.",
    "Straight-line radius, not drive time: coastal and mountain cities are understated.",
    "A postcode district counts fully in or fully out, by its centroid.",
    "Catchments stop at national borders, so cross-border cities are understated.",
    f"Demographics cover {', '.join(COVERED)} only; elsewhere the fields are null, not zero.",
]


# ---------------------------------------------------------------------------
# EXCLUSIVE catchment
# ---------------------------------------------------------------------------

def exclusive_catchment(pc, markets, min_events=20, radii_km=DEFAULT_RADII_KM,
                        max_km=150):
    """
    Population assigned to whichever qualifying market is nearest.

    WHY A PLAIN RADIUS IS NOT ENOUGH

    A radius counts everyone within N km regardless of what else is nearby.
    That credited Guildford with 13.8 million people -- almost all of London,
    40 km up the road -- and Hanley with 7.2 million of Manchester and
    Birmingham. Both then looked like enormous under-served markets, when in
    truth their residents already have far better venues a short train ride
    away. On the exclusive measure they hold 855,000 and 916,000.

    HOW IT WORKS

    Every postcode district is assigned to the single nearest market that
    actually hosts shows (at least `min_events` of them), and the assigned
    districts are then summed. Nobody is counted twice, and a town in a big
    city's shadow gets only the people for whom it genuinely is the closest
    option.

    It is a Voronoi allocation -- the simplest defensible model of competition,
    chosen because it explains in one sentence: people go to their nearest
    market. A gravity or Huff model would capture that a bigger city pulls from
    further away, at the cost of a distance-decay parameter nobody can observe
    and a number no reader could check by hand.

    TWO FIGURES, NOT ONE

    `exclusive_population` is everyone assigned, out to `max_km`. In a region
    with few qualifying markets that reaches a long way: Bari, the only such
    market for most of Puglia, claims 2.7 million at an average distance of
    76 km. True, but not a catchment anyone would plan a venue against.

    `exclusive_population_60km` and the other banded columns are assigned AND
    within that radius. These are the ones to rank on, because they are
    directly comparable to the plain `population_60km` column beside them --
    the only difference is that a neighbouring city's residents have been
    removed.

    Known consequence, stated rather than hidden: this UNDERSTATES big cities
    slightly (people do travel past a small market to reach a large one) and
    OVERSTATES small ones for the same reason. Both measures are kept so either
    reading is available.
    """
    qualifying = markets[pd.to_numeric(markets["events"], errors="coerce")
                         .fillna(0) >= min_events]
    out = {}

    for code, grp in qualifying.groupby("countryCode"):
        here = pc[pc["ctrycode"] == code]
        if here.empty or grp.empty:
            continue
        mlat = np.radians(pd.to_numeric(grp["lat"], errors="coerce").values)
        mlon = np.radians(pd.to_numeric(grp["lon"], errors="coerce").values)
        ok = np.isfinite(mlat) & np.isfinite(mlon)
        mlat, mlon, names = mlat[ok], mlon[ok], grp["city"].values[ok]
        if not len(mlat):
            continue

        plat = here["lat_rad"].values[:, None]      # postcodes x 1
        plon = here["lon_rad"].values[:, None]

        # every postcode against every market in this country, in one broadcast
        dlat = mlat[None, :] - plat
        dlon = mlon[None, :] - plon
        a = (np.sin(dlat / 2) ** 2
             + np.cos(plat) * np.cos(mlat[None, :]) * np.sin(dlon / 2) ** 2)
        dist = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

        nearest = dist.argmin(axis=1)
        nearest_km = dist.min(axis=1)
        keep = nearest_km <= max_km

        frame = here.loc[keep].copy()
        frame["market"] = names[nearest[keep]]
        frame["km_to_market"] = nearest_km[keep]

        g = frame.groupby("market").agg(
            exclusive_population=("population", "sum"),
            exclusive_age_15_29=("age_15_29", "sum"),
            exclusive_households=("households", "sum"),
            exclusive_pp_total_meur=("purchasing_power_total_meur", "sum"),
            exclusive_postcodes=("postcode", "count"),
            exclusive_mean_km=("km_to_market", "mean"))
        for market, row in g.iterrows():
            out[(code, market)] = {
                "exclusive_population": int(row["exclusive_population"]),
                "exclusive_age_15_29": int(row["exclusive_age_15_29"]),
                "exclusive_households": int(row["exclusive_households"]),
                "exclusive_pp_total_meur": round(float(row["exclusive_pp_total_meur"]), 1),
                "exclusive_postcodes": int(row["exclusive_postcodes"]),
                "exclusive_mean_km": round(float(row["exclusive_mean_km"]), 1),
            }

        # the same sums again, but capped at each radius
        for km in radii_km:
            band = frame[frame["km_to_market"] <= km].groupby("market").agg(
                pop=("population", "sum"), young=("age_15_29", "sum"),
                pp=("purchasing_power_total_meur", "sum"))
            for market in names:
                d = out.setdefault((code, market), {})
                if market in band.index:
                    r = band.loc[market]
                    d[f"exclusive_population_{km}km"] = int(r["pop"])
                    d[f"exclusive_age_15_29_{km}km"] = int(r["young"])
                    d[f"exclusive_pp_total_meur_{km}km"] = round(float(r["pp"]), 1)
                else:
                    d[f"exclusive_population_{km}km"] = 0
                    d[f"exclusive_age_15_29_{km}km"] = 0
                    d[f"exclusive_pp_total_meur_{km}km"] = 0.0

    rows = [out.get((r.countryCode, r.city), {}) for r in markets.itertuples(index=False)]
    return pd.concat([markets.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
