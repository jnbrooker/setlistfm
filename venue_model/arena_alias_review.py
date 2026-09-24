#!/usr/bin/env python3
"""
Which arenas are split across several names, and what aliases would fix them.

THE PROBLEM THIS EXISTS FOR

Arenas get renamed every few years, because the name is a sponsorship asset.
The Milan forum has traded as Forum di Assago, FilaForum, DatchForum,
Mediolanum Forum and now Unipol Forum. `venues` is keyed on the name, so that
one building is seven rows with 1,272 events split between them -- and the
capacity ladder, the peer comparison and every layer above them see five small
venues where there is one large one.

WHY venue_dedup.py DOES NOT CATCH THESE

venue_dedup generates candidates by NAME SIMILARITY, with a floor of 88. That
is the right tool for spelling variants and descriptive suffixes, and it found
189 of them. But a sponsorship rename replaces the distinctive part of the
name:

    mediolanum forum di assago   vs   unipol forum          score 52
    paris la defense arena       vs   plenitude arena       score 45

Those never reach the candidate stage, so no amount of tuning the date test
downstream will find them. The signal that does work is geography.

THE SIGNAL THIS USES INSTEAD

A building does not move. Every one of the Assago rows carries the identical
coordinate 45.408869, 9.125647; Plenitude Arena carries Paris La Defense
Arena's exact coordinate. Co-location is independent of the name, which is
precisely what is needed when the name is the thing that changed.

CO-LOCATION ALONE IS NOT ENOUGH, AND THE FAILURE IS OBVIOUS

An arena and the theatre attached to it share a coordinate. The O2 and indigo
at The O2 are one address and two rooms; merging them would destroy the ladder
this project is built on. So co-location only generates the candidate, and the
discriminator is the one venue_dedup already validated on 4,640 pairs:

    DISJOINT date ranges     the old name stopped when the new one started.
                             A RENAME. Propose an alias.
    OVERLAPPING date ranges  both names were in use at once. DIFFERENT ROOMS
                             in one complex. Never merge; reported separately
                             so the distinction is visible rather than assumed.

Capacity ratio is carried alongside as a second opinion, not as a rule: a
rename usually keeps the capacity within a few per cent, while an arena and its
side hall differ by a factor.

WHAT IT WRITES

    python arena_alias_review.py

A workbook in reports/, every sheet ranked by how many events are at stake so
the worst splits are read first:

    All candidates   the deliverable, in the exact shape
                     `build_events.py load-venue-aliases` reads. `should_merge`
                     is pre-set for the high-confidence rows only.
    needs_city_fix   one building split by a MISSING COUNTRY CODE rather than a
                     rename. No alias can repair these -- see the note on
                     blank_country_splits -- and they strand about as many
                     events as every rename here put together.
    alias_cannot_fix proposals whose two rows disagree on city or country, so
                     folding the name would not merge them either
    renames          the same proposals with all the evidence columns
    arena_duplicates the dashboard's own table holding one building twice
    same_complex     co-located but concurrent; explicitly NOT to be merged
    unlinked         arena-scale venues with no arena_id at all

AND IT TARGETS venue_aliases, NOT arena_aliases. That distinction is the whole
point: arena_aliases maps a spelling to a dashboard arena so an event can
inherit a verified capacity, but the identity of the BUILDING is venue_uid, and
build-venues derives that from venue_aliases alone. Adding rows to
arena_aliases leaves the events exactly as split as they were.

Nothing is written back to the database. This produces a review, and a human
decides -- the same contract venue_dedup works under.
"""

import argparse
import datetime as dt
import os
import sqlite3
import sys
import unicodedata

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_DB = os.path.join(os.path.dirname(HERE), "setlistfm.db")
OUT_DIR = os.path.join(HERE, "reports")

sys.path.insert(0, HERE)
from venue_dedup import (GENERIC_NAMES, SPONSORS,   # noqa: E402
                         SUBROOM_MARKERS, _overlap_fraction)

# How close two rows must be to count as the same address.
#
# 25 m, which is very nearly "identical", and the data is what forced it down.
# At 150 m every single one of the 577 proposals still came out at EXACTLY
# 0.0 m apart -- because a renamed building keeps the coordinate of the place
# record it was geocoded from, so a rename is a coordinate match rather than a
# proximity match. The wider radius therefore bought nothing and cost a great
# deal: clustering is transitive, so in a dense complex A-near-B and B-near-C
# chained genuinely separate arenas into one group. Berlin came out as a single
# cluster containing the Tempodrom, the Velodrom, the Max-Schmeling-Halle and
# the Uber Arena, which are four buildings.
RADIUS_M = 25

# Arena scale. Below this a split name costs little and the report fills with
# clubs, which venue_dedup already handles on name similarity.
MIN_CAPACITY = 4_000

# A venue needs some events before its date range means anything.
MIN_EVENTS = 2

# Share of the shorter span the two may overlap and still be called a rename.
# Not zero: a listing under the old name routinely trails the switch.
CONCURRENT_FRACTION = 0.25

_PUNCT = None


def norm_key(name):
    """
    The same aggressive key build_events uses, so an alias proposed here can be
    compared against arena_aliases.alias_norm without a second convention.
    """
    import re
    s = str(name or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("&", " and ").replace("$", "s")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[4:] if s.startswith("the ") else s


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(db=MAIN_DB, min_capacity=MIN_CAPACITY, min_events=MIN_EVENTS):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    v = pd.read_sql("""
        SELECT venue, venue_norm, city, city_norm, country, countryCode,
               latitude, longitude, events, capacity, arena_id,
               first_event, last_event, aliases
        FROM venues WHERE events >= ?""", con, params=(min_events,))
    a = pd.read_sql("SELECT * FROM arenas", con)
    al = pd.read_sql("SELECT alias_norm, arena_id, alias, source, ambiguous "
                     "FROM arena_aliases", con)
    va = pd.read_sql("SELECT alias_norm, city_norm, countryCode, canonical_norm "
                     "FROM venue_aliases", con)
    con.close()

    for c in ("latitude", "longitude", "capacity", "events"):
        v[c] = pd.to_numeric(v[c], errors="coerce")
    v["venue_norm"] = v["venue_norm"].fillna("").astype(str)
    v.loc[v["venue_norm"] == "", "venue_norm"] = v["venue"].map(norm_key)

    # Arena scale: a big capacity, OR already linked to a dashboard arena.
    # The second clause matters -- a room whose capacity never landed on this
    # particular spelling is exactly the case being hunted.
    big = (v["capacity"].fillna(0) >= min_capacity) | v["arena_id"].notna()
    return v[big].copy(), a, al, va


# ---------------------------------------------------------------------------
# Clustering by address
# ---------------------------------------------------------------------------

def coordinate_crowding(v):
    """
    How many DISTINCT venue names sit on each coordinate.

    The guard against city-centroid geocoding, which is widespread here and
    would otherwise make co-location meaningless. Every arena-scale venue in
    Berlin carries 52.516667, 13.4 -- the city centre, to three decimal places
    -- so Columbiahalle, the Waldbuehne, the Mercedes-Benz Arena and the
    Max-Schmeling-Halle all share an "address". Sixty-two London venues share
    one point.

    A real building address is shared only by that building's own names: the
    Assago forum's coordinate is carried by exactly its five spellings. So the
    count is reported beside every proposal as the reader's discount factor.
    It is deliberately NOT a filter -- Assago would survive a threshold of five
    and Duesseldorf's eleven would not, but so would genuine pairs in Berlin,
    and the date test already separates those.
    """
    g = v.dropna(subset=["latitude", "longitude"]).copy()
    g["_coord"] = (g["latitude"].round(6).astype(str) + "," +
                   g["longitude"].round(6).astype(str))
    n = g.groupby("_coord")["venue_norm"].nunique()
    return g["_coord"].map(n).reindex(v.index)


def clusters(v, radius_m=RADIUS_M):
    """
    Group venue rows that sit at the same address, country by country.

    Country by country because a KD-tree over the whole world would pair
    antipodes across the dateline, and because a cross-border match is never
    wanted: two names in different countries are different buildings, the same
    rule venue_dedup holds to.
    """
    out = []
    geo = v.dropna(subset=["latitude", "longitude"])
    for cc, g in geo.groupby("countryCode"):
        if len(g) < 2:
            continue
        lat = np.radians(g["latitude"].values)
        lon = np.radians(g["longitude"].values)
        # Equirectangular projection to metres. Fine at this scale: the error
        # over 150 m at European latitudes is centimetres.
        R = 6_371_000.0
        x = R * lon * np.cos(lat.mean())
        y = R * lat
        pairs = cKDTree(np.c_[x, y]).query_pairs(radius_m, output_type="ndarray")
        if not len(pairs):
            continue
        # union-find over the pairs
        idx = g.index.values
        parent = {i: i for i in range(len(g))}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in pairs:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[ri] = rj
        lab = {}
        for i in range(len(g)):
            lab.setdefault(find(i), []).append(idx[i])
        for k, members in lab.items():
            if len(members) > 1:
                out.append((f"{cc}:{k}", members))
    return out


def _span(row):
    return (str(row["first_event"] or ""), str(row["last_event"] or ""))


def metres_between(a, b):
    """Great-circle distance between two venue rows, in metres."""
    try:
        lat1, lon1 = float(a["latitude"]), float(a["longitude"])
        lat2, lon2 = float(b["latitude"]), float(b["longitude"])
    except (TypeError, ValueError):
        return np.nan
    if not all(np.isfinite([lat1, lon1, lat2, lon2])):
        return np.nan
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * 6_371_000.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1))))


def gap_days(a, b):
    """
    Days between one name going quiet and the other starting.

    THE DISCRIMINATOR THAT SEPARATES A RENAME FROM A REPLACEMENT BUILDING.

    Co-location plus disjoint dates describes a rename -- and equally describes
    a new arena built beside the one it replaced, which is a different building
    with its own capacity and history. Memphis has exactly that: FedExForum
    opened next to the Pyramid, and the naive test pairs them.

    A sponsorship rename is seamless: the last show under the old name and the
    first under the new one are days or weeks apart, because nothing physical
    happened. A replacement leaves a construction-shaped hole. The number is
    reported rather than thresholded, because a rebuild -- Dublin's Point
    closing for two years to reopen as 3Arena -- is genuinely both.
    """
    try:
        a1, a2 = pd.Timestamp(a["first_event"]), pd.Timestamp(a["last_event"])
        b1, b2 = pd.Timestamp(b["first_event"]), pd.Timestamp(b["last_event"])
    except (TypeError, ValueError):
        return np.nan
    if pd.isna(a1) or pd.isna(a2) or pd.isna(b1) or pd.isna(b2):
        return np.nan
    if b1 > a2:
        return float((b1 - a2).days)
    if a1 > b2:
        return float((a1 - b2).days)
    return 0.0


def grade(distance_m, ratio, gap):
    """
    How far a proposal deserves to be trusted, from the evidence beside it.

    Three coarse bands rather than a score: the inputs are a distance, a ratio
    and a gap, and combining them into a decimal would imply a precision none
    of them have.
    """
    near = np.isfinite(distance_m) and distance_m <= 25
    same_size = np.isfinite(ratio) and ratio <= 1.25
    # A gap of exactly zero is not a seamless handover -- it is the two names
    # RUNNING AT THE SAME TIME, which is the concurrency signal and argues
    # against a rename. It survives the 0.25 overlap tolerance upstream, so it
    # has to be caught here or Dallas proposes folding Reunion Arena into the
    # American Airlines Center, which are two buildings that briefly coexisted.
    seamless = np.isfinite(gap) and 0 < gap <= 400
    concurrent = np.isfinite(gap) and gap == 0
    if concurrent:
        return "low"
    if near and same_size and seamless:
        return "high"
    if near and (same_size or seamless):
        return "medium"
    return "low"


def looks_like_subroom(name_norm):
    toks = set(name_norm.split())
    return any(m in name_norm for m in SUBROOM_MARKERS) or bool(
        toks & {"2", "3", "ii", "iii"})


def classify_cluster(rows, concurrent_fraction=CONCURRENT_FRACTION):
    """
    Split a co-located cluster into a rename chain and everything else.

    The test is pairwise against the LARGEST row, which is the building the
    complex is named for. A row whose dates do not overlap it is a former or
    later name for it; a row that ran alongside it is a different room.
    """
    rows = rows.sort_values("events", ascending=False)
    anchor = rows.iloc[0]
    a1, a2 = _span(anchor)

    renames, concurrent = [], []
    for _, r in rows.iloc[1:].iterrows():
        b1, b2 = _span(r)
        if not (a1 and a2 and b1 and b2):
            concurrent.append((r, None))
            continue
        frac = _overlap_fraction(a1, a2, b1, b2)
        cap_a = anchor["capacity"] if pd.notna(anchor["capacity"]) else np.nan
        cap_b = r["capacity"] if pd.notna(r["capacity"]) else np.nan
        ratio = (max(cap_a, cap_b) / min(cap_a, cap_b)
                 if np.isfinite(cap_a) and np.isfinite(cap_b) and min(cap_a, cap_b) > 0
                 else np.nan)
        sub = looks_like_subroom(r["venue_norm"])
        if frac <= concurrent_fraction and not sub:
            renames.append((r, frac, ratio, metres_between(anchor, r),
                            gap_days(anchor, r)))
        else:
            concurrent.append((r, frac))
    return anchor, renames, concurrent


# ---------------------------------------------------------------------------
# The review
# ---------------------------------------------------------------------------

def review(v, arenas, aliases, venue_aliases, radius_m=RADIUS_M):
    v = v.assign(names_on_this_coord=coordinate_crowding(v))
    known_alias = set(aliases["alias_norm"])
    va_known = {(r.alias_norm, r.city_norm, r.countryCode)
                for r in venue_aliases.itertuples()}

    rename_rows, concurrent_rows, dup_rows = [], [], []
    for cid, members in clusters(v, radius_m):
        g = v.loc[members]
        if g["venue_norm"].nunique() < 2:
            continue
        anchor, renames, concurrent = classify_cluster(g)

        # the dashboard holding one building twice
        ids = sorted(set(g["arena_id"].dropna()))
        if len(ids) > 1:
            for aid in ids:
                nm = arenas.loc[arenas["arena_id"] == aid, "name"]
                dup_rows.append({
                    "cluster": cid,
                    "arena_id": aid,
                    "arena name": nm.iloc[0] if len(nm) else "",
                    "venue names at this address": " | ".join(
                        sorted(g.loc[g["arena_id"] == aid, "venue"].unique())),
                    "events": int(g.loc[g["arena_id"] == aid, "events"].sum()),
                    "city": anchor["city"], "country": anchor["country"],
                })

        for r, frac, ratio, dist, gap in renames:
            already = (r["venue_norm"] in known_alias
                       and anchor["venue_norm"] in known_alias
                       and r["arena_id"] == anchor["arena_id"]
                       and pd.notna(r["arena_id"]))
            same_arena = (pd.notna(r["arena_id"])
                          and r["arena_id"] == anchor["arena_id"])
            covered_va = (r["venue_norm"], r["city_norm"],
                          r["countryCode"]) in va_known
            rename_rows.append({
                "action": "add alias",
                "alias (fold this away)": r["venue"],
                "alias_norm": r["venue_norm"],
                "canonical (keep this)": anchor["venue"],
                "canonical_norm": anchor["venue_norm"],
                "city": anchor["city"], "countryCode": anchor["countryCode"],
                "country": anchor["country"],
                "arena_id": anchor["arena_id"] if pd.notna(anchor["arena_id"])
                            else r["arena_id"],
                "events at stake": int(r["events"]),
                "canonical events": int(anchor["events"]),
                "alias dates": f"{r['first_event']} .. {r['last_event']}",
                "canonical dates": f"{anchor['first_event']} .. {anchor['last_event']}",
                "date overlap": round(float(frac), 3),
                "alias capacity": r["capacity"],
                "canonical capacity": anchor["capacity"],
                "capacity ratio": round(float(ratio), 2) if np.isfinite(ratio) else None,
                "metres apart": (round(dist, 1) if np.isfinite(dist) else None),
                "gap between runs (days)": (int(gap) if np.isfinite(gap) else None),
                "confidence": grade(dist, ratio, gap),
                "names on this coordinate": (
                    int(anchor["names_on_this_coord"])
                    if pd.notna(anchor.get("names_on_this_coord")) else None),
                # An alias can only ever fire inside the city and country it
                # was recorded for, because venue_uid embeds both. When the two
                # rows disagree on either, folding the NAME will not merge them.
                "same city and country": bool(
                    (r["city_norm"] or "") == (anchor["city_norm"] or "")
                    and (r["countryCode"] or "") == (anchor["countryCode"] or "")),
                "already same arena_id": bool(same_arena),
                "already in venue_aliases": bool(covered_va),
                "already covered": bool(already or covered_va),
            })

        for r, frac in concurrent:
            concurrent_rows.append({
                "cluster": cid,
                "venue A (bigger)": anchor["venue"],
                "venue B": r["venue"],
                "city": anchor["city"], "countryCode": anchor["countryCode"],
                "A events": int(anchor["events"]), "B events": int(r["events"]),
                "A capacity": anchor["capacity"], "B capacity": r["capacity"],
                "date overlap": (round(float(frac), 3)
                                 if frac is not None else None),
                "why not merged": ("dates overlap - both names in use at once"
                                   if frac is not None and frac > CONCURRENT_FRACTION
                                   else ("sub-room wording"
                                         if frac is not None
                                         else "no usable dates")),
            })

    ren = pd.DataFrame(rename_rows)
    if len(ren):
        order = {"high": 0, "medium": 1, "low": 2}
        ren["_g"] = ren["confidence"].map(order)
        ren = (ren.sort_values(["already covered", "_g", "events at stake"],
                               ascending=[True, True, False])
               .drop(columns="_g"))
    con = (pd.DataFrame(concurrent_rows)
           .sort_values("B events", ascending=False)
           if concurrent_rows else pd.DataFrame())
    dup = (pd.DataFrame(dup_rows).sort_values(["cluster", "events"],
                                              ascending=[True, False])
           if dup_rows else pd.DataFrame())
    return ren, con, dup


def loadable(ren):
    """
    The proposals in the exact shape `build_events.py load-venue-aliases` reads.

    THIS IS THE SHEET THAT DOES THE WORK, AND venue_aliases IS THE RIGHT TABLE.

    arena_aliases does NOT merge anything. It maps a spelling to a dashboard
    arena so an event can inherit a verified capacity; the identity of the
    BUILDING is venue_uid, and build-venues computes that as

        COALESCE(venue_aliases.canonical_norm, venue_norm) |city_norm|countryCode

    so a name only folds onto another when there is a venue_aliases row for it.
    Adding rows to arena_aliases would leave the events exactly as split as
    they are now.

    `should_merge` is pre-set from the confidence grade -- high is proposed,
    everything else is left for a human to turn on. The loader honours that
    column, so overruling any verdict is editing one cell.
    """
    keep = ren[ren["same city and country"]].copy()
    out = pd.DataFrame({
        "keep": keep["canonical (keep this)"],
        "keep_norm": keep["canonical_norm"],
        "fold": keep["alias (fold this away)"],
        "fold_norm": keep["alias_norm"],
        "city": keep["city"],
        "countryCode": keep["countryCode"],
        "verdict": keep["confidence"].map(
            {"high": "rename (co-located, seamless, same size)",
             "medium": "rename (co-located, one check weak)",
             "low": "unconfirmed - review before merging"}),
        "should_merge": keep["confidence"].eq("high"),
        "events at stake": keep["events at stake"],
        "metres apart": keep["metres apart"],
        "gap between runs (days)": keep["gap between runs (days)"],
        "capacity ratio": keep["capacity ratio"],
        "names on this coordinate": keep["names on this coordinate"],
        "already covered": keep["already covered"],
    })
    return out[~out["already covered"]].drop(columns="already covered")


def not_fixable_by_alias(ren):
    """
    Splits an alias CANNOT close, because venue_uid embeds city and country.

    venue_uid is `name|city_norm|countryCode`. A venue_aliases row rewrites
    only the name part, and it is itself keyed on (alias, city, country) -- so
    when two rows for one building disagree on the city or carry a blank
    country code, folding the name changes nothing. The Assago forum has both
    problems at once: `unipol forum|assago|IT` and `unipol forum|milan|`
    are the same room, the same spelling, and two buildings as far as the
    pipeline is concerned.

    These need the city and country fixed on the events, not an alias. They are
    reported separately so the two remedies are not confused.
    """
    bad = ren[~ren["same city and country"]].copy()
    if bad.empty:
        return bad
    bad["why an alias will not fix it"] = np.where(
        (bad["countryCode"].fillna("") == ""),
        "blank countryCode on one side",
        "the two rows disagree on the city")
    cols = ["alias (fold this away)", "canonical (keep this)", "city",
            "countryCode", "events at stake", "confidence",
            "why an alias will not fix it"]
    return bad[cols].sort_values("events at stake", ascending=False)


def arena_alias_rows(ren, aliases, arenas):
    """
    The same renames, aimed at arena_aliases instead of venue_aliases.

    THE TWO TABLES DO DIFFERENT JOBS AND BOTH ARE WANTED.

        venue_aliases   makes the EVENTS collapse onto one venue_uid, so a
                        building's history stops being split five ways.
        arena_aliases   is the DIRECTORY: spelling -> arena_id. It is what lets
                        a lookup of any name return the building's id, and a
                        lookup of that id return every name the building has
                        traded under. It is also how an event inherits the
                        dashboard's verified capacity.

    Only renames go in here, never the concurrent co-located pairs. Aliasing a
    side room onto its arena's id would hand that room the arena's capacity,
    which is precisely the failure the `festhalle` entry caused when one alias
    reached fifty-two towns and gave a village hall Festhalle Frankfurt's
    15,000 seats.

    Output matches arena_aliases_manual.csv exactly -- alias, arena_id, city,
    country -- with the evidence columns after them. The loader reads by name
    with csv.DictReader, so the extra columns are ignored by the pipeline and
    kept for the human.
    """
    if not len(ren):
        return pd.DataFrame()
    known = {(r.alias_norm, r.arena_id) for r in aliases.itertuples()}
    name_of = arenas.set_index("arena_id")["name"].to_dict()

    d = ren[ren["arena_id"].notna() & ~ren["already covered"]].copy()
    d = d[[(a, i) not in known
           for a, i in zip(d["alias_norm"], d["arena_id"])]]
    if d.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "alias": d["alias (fold this away)"],
        "arena_id": d["arena_id"],
        "city": d["city"],
        "country": d["country"],
        "arena name": d["arena_id"].map(name_of),
        "confidence": d["confidence"],
        "events at stake": d["events at stake"],
        "gap between runs (days)": d["gap between runs (days)"],
        "capacity ratio": d["capacity ratio"],
    })
    order = {"high": 0, "medium": 1, "low": 2}
    return (out.assign(_g=out["confidence"].map(order))
            .sort_values(["_g", "events at stake"], ascending=[True, False])
            .drop(columns="_g"))


def blank_country_splits(v):
    """
    One building split in two by a MISSING COUNTRY CODE, not by a rename.

    THE DEFECT NO ALIAS CAN REPAIR, AND THE LARGER OF THE TWO.

    venue_uid is `venue_norm|city_norm|countryCode`. Where some events carry a
    country code and others do not, the same spelling in the same city becomes
    two buildings:

        royal albert hall|london|GB     2,122 events
        royal albert hall|london|         295 events

    venue_aliases cannot close that. It rewrites the NAME part of the uid and
    is itself keyed on (alias, city, country), so with the name already
    identical there is nothing for it to rewrite and no key that matches both
    sides. The remedy is upstream: backfill countryCode on the events, which is
    unambiguous precisely because the name and the city already agree.

    Reported at every capacity, not just arena scale, because the fix is one
    UPDATE rather than a judgement per building.
    """
    w = v.copy()
    w["cc"] = w["countryCode"].fillna("")
    rows = []
    for (vn, cn), g in w.groupby(["venue_norm", "city_norm"], dropna=False):
        if g["cc"].nunique() < 2 or not (g["cc"] == "").any():
            continue
        blank, known = g[g["cc"] == ""], g[g["cc"] != ""]
        rows.append({
            "venue": known["venue"].iloc[0] if len(known) else blank["venue"].iloc[0],
            "venue_norm": vn, "city": cn,
            "countryCode present": "/".join(sorted(set(known["cc"]))),
            "events stranded (blank code)": int(blank["events"].sum()),
            "events on the coded row": int(known["events"].sum()),
            "capacity": known["capacity"].max(),
            "fix": "backfill countryCode on the events; an alias cannot do it",
        })
    d = pd.DataFrame(rows)
    return (d.sort_values("events stranded (blank code)", ascending=False)
            if len(d) else d)


def unlinked(v, min_capacity=MIN_CAPACITY):
    """Arena-scale venues with no dashboard arena behind them at all."""
    u = v[v["arena_id"].isna() & (v["capacity"].fillna(0) >= min_capacity)]
    return (u[["venue", "city", "country", "countryCode", "capacity", "events",
               "first_event", "last_event"]]
            .sort_values("events", ascending=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=MAIN_DB)
    ap.add_argument("--radius", type=float, default=RADIUS_M,
                    help="metres; two rows closer than this share an address")
    ap.add_argument("--min-capacity", type=int, default=MIN_CAPACITY)
    ap.add_argument("--min-events", type=int, default=MIN_EVENTS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    log("loading ...")
    v, arenas, aliases, venue_aliases = load(a.db, a.min_capacity, a.min_events)
    log(f"{len(v):,} arena-scale venue rows, {len(arenas):,} dashboard arenas, "
        f"{len(aliases):,} existing arena aliases")

    log(f"clustering by address at {a.radius:g} m ...")
    ren, con, dup = review(v, arenas, aliases, venue_aliases, a.radius)
    unl = unlinked(v, a.min_capacity)

    new = ren[~ren["already covered"]] if len(ren) else ren
    print(f"\n{'=' * 78}")
    print(f"PROPOSED ALIASES        {len(new):,} name(s) to fold, "
          f"{int(new['events at stake'].sum()) if len(new) else 0:,} events "
          f"currently detached")
    print(f"ALREADY COVERED         {int(ren['already covered'].sum()) if len(ren) else 0:,}")
    print(f"SAME COMPLEX, NOT MERGED {len(con):,}  (concurrent dates)")
    print(f"DASHBOARD DUPLICATES    {dup['cluster'].nunique() if len(dup) else 0:,} "
          f"building(s) held under more than one arena_id")
    print(f"UNLINKED ARENA-SCALE    {len(unl):,} venues with no arena_id")
    print("=" * 78)

    if len(new):
        print("\nBY CONFIDENCE")
        print(new.groupby("confidence")
              .agg(proposals=("alias_norm", "size"),
                   events=("events at stake", "sum"))
              .reindex(["high", "medium", "low"]).fillna(0).astype(int)
              .to_string())
        cols = ["alias (fold this away)", "canonical (keep this)", "city",
                "countryCode", "events at stake", "metres apart",
                "gap between runs (days)", "capacity ratio"]
        print("\nHIGH CONFIDENCE, WORST SPLITS FIRST\n")
        print(new[new["confidence"] == "high"][cols].head(25)
              .to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = a.out or os.path.join(
        OUT_DIR, f"arena_alias_review_{dt.date.today():%Y-%m-%d}.xlsx")
    load_sheet = loadable(ren) if len(ren) else pd.DataFrame()
    nofix = not_fixable_by_alias(ren) if len(ren) else pd.DataFrame()
    blanks = blank_country_splits(v)
    arena_rows = arena_alias_rows(ren, aliases, arenas)
    if len(arena_rows):
        hi = arena_rows[arena_rows["confidence"].isin(["high", "medium"])]
        csv_path = os.path.join(
            OUT_DIR, f"arena_aliases_manual_proposed_{dt.date.today():%Y-%m-%d}.csv")
        os.makedirs(OUT_DIR, exist_ok=True)
        hi[["alias", "arena_id", "city", "country"]].to_csv(csv_path, index=False)
        print("")
        print(f"ARENA DIRECTORY         {len(arena_rows):,} spelling(s) not yet "
              f"in arena_aliases; {len(hi):,} written in the")
        print(f"                        arena_aliases_manual.csv format to "
              f"{os.path.basename(csv_path)}")
    if len(blanks):
        print("")
        print(f"BLANK COUNTRY SPLITS    {len(blanks):,} buildings split by a "
              f"missing countryCode, {int(blanks['events stranded (blank code)'].sum()):,} "
              f"events stranded")
        print("                        no alias can fix these - see "
              "needs_city_fix")

    n_merge = int(load_sheet["should_merge"].sum()) if len(load_sheet) else 0
    print("")
    print(f"LOADABLE                {len(load_sheet):,} proposals in the "
          f"venue_aliases contract, {n_merge:,} pre-marked should_merge")
    print(f"ALIAS CANNOT FIX        {len(nofix):,} split(s) where the two rows "
          f"disagree on city or country")

    with pd.ExcelWriter(out, engine="openpyxl") as w:
        # named to match what `build_events.py load-venue-aliases` reads
        (load_sheet if len(load_sheet) else pd.DataFrame({"note": ["none"]})
         ).to_excel(w, sheet_name="All candidates", index=False)
        (arena_rows if len(arena_rows) else pd.DataFrame({"note": ["none"]})
         ).to_excel(w, sheet_name="arena_aliases_manual", index=False)
        (blanks if len(blanks) else pd.DataFrame({"note": ["none"]})).to_excel(
            w, sheet_name="needs_city_fix", index=False)
        (nofix if len(nofix) else pd.DataFrame({"note": ["none"]})).to_excel(
            w, sheet_name="alias_cannot_fix", index=False)
        (ren if len(ren) else pd.DataFrame({"note": ["none found"]})).to_excel(
            w, sheet_name="renames", index=False)
        (dup if len(dup) else pd.DataFrame({"note": ["none found"]})).to_excel(
            w, sheet_name="arena_duplicates", index=False)
        (con if len(con) else pd.DataFrame({"note": ["none found"]})).to_excel(
            w, sheet_name="same_complex", index=False)
        unl.to_excel(w, sheet_name="unlinked", index=False)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
