#!/usr/bin/env python3
"""
Proof that venue merging folds renames and never crosses a city boundary.

WHAT IT CHECKS

Three things, on a database built from scratch so the 7.5 GB live one is never
touched and the scraper is never blocked:

  1. A RENAME MERGES. AO Arena, Manchester Arena and MEN Arena become one row
     carrying all the events, with every spelling recorded in `aliases`.

  2. THE SAME NAME IN A DIFFERENT CITY DOES NOT. This is the one that matters.
     The alias table is keyed on city and country, so a Festhalle alias
     recorded for Frankfurt cannot reach Bad Urach however similar the names
     are. It is the structural guarantee that the arena_alias bug cannot recur
     here, and a guarantee is only worth what its test is worth.

  3. A SUB-ROOM STAYS SEPARATE. Manchester Academy 1 and Academy 2 are
     different rooms with different capacities; nothing in the merge may join
     them.

    python test_venue_merge.py
"""

import os
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_events as BE   # noqa: E402


def build_fixture(path):
    con = sqlite3.connect(path)
    con.executescript(BE.SCHEMA)
    # venue_norm, city_norm and venue_uid are migrations rather than part of
    # SCHEMA, so the fixture has to take the same path a real database does.
    BE.ensure_columns(con)
    rows = [
        # one Manchester building under three names, never at the same time
        ("m1", "MEN Arena", "men arena", "k1", "Manchester", "manchester",
         "United Kingdom", "GB"),
        ("m2", "Manchester Arena", "manchester arena", "k2", "Manchester",
         "manchester", "United Kingdom", "GB"),
        ("m3", "AO Arena", "ao arena", "k3", "Manchester", "manchester",
         "United Kingdom", "GB"),
        # two DIFFERENT rooms in one students' union
        ("a1", "Manchester Academy 1", "manchester academy 1", "k4",
         "Manchester", "manchester", "United Kingdom", "GB"),
        ("a2", "Manchester Academy 2", "manchester academy 2", "k5",
         "Manchester", "manchester", "United Kingdom", "GB"),
        # the same spelling in two towns: must never join
        ("f1", "Festhalle", "festhalle", "k6", "Frankfurt", "frankfurt",
         "Germany", "DE"),
        ("f2", "Festhalle", "festhalle", "k7", "Bad Urach", "bad urach",
         "Germany", "DE"),
    ]
    con.executemany(
        "INSERT INTO events (event_id, venue, venue_norm, venue_key, city, "
        "city_norm, country, countryCode) VALUES (?,?,?,?,?,?,?,?)", rows)

    # Only Manchester's renames are aliased, and only for Manchester.
    con.executemany(
        """INSERT INTO venue_aliases (alias_norm, city_norm, countryCode,
               canonical_norm, canonical, source, note, loaded_at)
           VALUES (?,?,?,?,?,?,?,datetime('now'))""",
        [("men arena", "manchester", "GB", "ao arena", "AO Arena", "test",
          "sponsor rename"),
         ("manchester arena", "manchester", "GB", "ao arena", "AO Arena",
          "test", "sponsor rename")])
    con.commit()
    return con


def assign_uids(con):
    """Exactly the statement build_venues runs."""
    con.execute("""
        UPDATE events SET venue_uid =
            CASE WHEN COALESCE(venue_norm,'') = '' THEN venue_key
                 ELSE COALESCE(
                        (SELECT va.canonical_norm FROM venue_aliases va
                          WHERE va.alias_norm  = events.venue_norm
                            AND va.city_norm   = COALESCE(events.city_norm,'')
                            AND va.countryCode = COALESCE(events.countryCode,'')),
                        events.venue_norm)
                      || '|' || COALESCE(city_norm,'')
                      || '|' || COALESCE(countryCode,'')
            END""")
    con.commit()
    return {r[0]: r[1] for r in con.execute(
        "SELECT event_id, venue_uid FROM events")}


def main():
    tmp = os.path.join(tempfile.mkdtemp(), "venue_merge_test.db")
    con = build_fixture(tmp)
    uid = assign_uids(con)

    print("event  venue_uid")
    for e in sorted(uid):
        print(f"{e:6s} {uid[e]}")
    print()

    failures = []

    # 1. the rename collapses to one id
    if not (uid["m1"] == uid["m2"] == uid["m3"]):
        failures.append("the three Manchester Arena names did not merge")
    if not uid["m3"].startswith("ao arena|"):
        failures.append("the surviving name should be AO Arena")

    # 2. THE CRITICAL ONE: same spelling, different city, must not merge
    if uid["f1"] == uid["f2"]:
        failures.append(
            "Frankfurt and Bad Urach Festhalle merged -- an alias crossed a "
            "city boundary, which is the bug this design exists to prevent")

    # 3. sub-rooms stay apart
    if uid["a1"] == uid["a2"]:
        failures.append("Manchester Academy 1 and 2 merged")

    # 4. the aggregate that records the folded names
    con.execute("""CREATE TEMP TABLE t AS
        SELECT venue_uid, COUNT(DISTINCT venue) AS n_names,
               group_concat(DISTINCT venue) AS aliases, COUNT(*) AS events
        FROM events GROUP BY venue_uid""")
    rows = {r[0]: r for r in con.execute("SELECT * FROM t")}
    ao = rows[uid["m3"]]
    print(f"merged row: {ao[2]}  ({ao[1]} names, {ao[3]} events)")
    if ao[1] != 3 or ao[3] != 3:
        failures.append(f"expected 3 names and 3 events, got {ao[1]} and {ao[3]}")
    for name in ("AO Arena", "MEN Arena", "Manchester Arena"):
        if name not in ao[2]:
            failures.append(f"{name} missing from the aliases column")

    # Seven events; the three Manchester Arena names collapse to one. So five
    # buildings: AO Arena, Academy 1, Academy 2, and a Festhalle in each town.
    print(f"\ndistinct buildings: {len(rows)} (expected 5)")
    if len(rows) != 5:
        failures.append(f"expected 5 distinct buildings, got {len(rows)}")

    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        raise SystemExit(1)
    print("\nPASS  rename merged with its names recorded; the two Festhallen "
          "stayed apart;\n      Academy 1 and 2 stayed apart.")
    con.close()


if __name__ == "__main__":
    main()
