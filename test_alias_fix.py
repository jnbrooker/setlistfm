#!/usr/bin/env python3
"""
Proof that a generic venue name no longer matches one arena worldwide.

WHY THIS EXISTS AS A FILE RATHER THAN A ONE-OFF CHECK

The bug it guards against was invisible for as long as it existed: every
affected row looked perfectly well-formed, and the only symptom was a village
hall in Konken recorded as holding fifteen thousand people. Nothing failed,
nothing logged a warning, and the capacity was plausible enough to survive
every downstream check. A fix for that class of bug is worth a test that can be
re-run, not a query someone remembers running once.

It builds a small database from scratch -- the real one is 7.5 GB and usually
has a scraper writing to it -- containing exactly the shape that broke:

    arenas          one Festhalle, in Frankfurt, holding 15,000
    events          Festhalle in five cities and two countries, plus a
                    genuinely unambiguous arena as a control

and then asserts the two things that matter:

    1. only the Frankfurt Festhalle gets the arena_id
    2. the control venue, whose name is not generic, still matches

The second assertion is the one that keeps the fix honest. Turning the matcher
off entirely would pass the first test and destroy the pipeline.

    python test_alias_fix.py
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
    con.executescript("""
        CREATE TABLE arenas (
            arena_id TEXT PRIMARY KEY, name TEXT, city TEXT, country TEXT,
            concert_capacity INTEGER, arena_type TEXT, outside_inside TEXT);
        CREATE TABLE arena_aliases (
            alias_norm TEXT, arena_id TEXT, alias TEXT, source TEXT,
            city_norm TEXT, country TEXT, ambiguous INTEGER DEFAULT 0,
            generic INTEGER DEFAULT 0, n_cities INTEGER,
            PRIMARY KEY (alias_norm, arena_id));
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY, venue TEXT, venue_norm TEXT,
            city TEXT, city_norm TEXT, country TEXT, countryCode TEXT,
            arena_id TEXT, arena_name TEXT, arena_capacity INTEGER,
            arena_type TEXT, arena_outside_inside TEXT);
    """)

    con.executemany(
        "INSERT INTO arenas VALUES (?,?,?,?,?,?,?)",
        [("A-FFM", "Festhalle Messe Frankfurt", "Frankfurt", "Germany",
          15000, "Arena", "Inside"),
         ("A-O2", "The O2 Arena", "London", "United Kingdom",
          20000, "Arena", "Inside")])

    # Aliases exactly as build_arena_aliases would write them.
    con.executemany(
        "INSERT INTO arena_aliases (alias_norm, arena_id, alias, source, "
        "city_norm, country) VALUES (?,?,?,?,?,?)",
        [("festhalle", "A-FFM", "Festhalle", "also_known_as",
          "frankfurt", "Germany"),
         ("festhalle messe frankfurt", "A-FFM", "Festhalle Messe Frankfurt",
          "name", "frankfurt", "Germany"),
         ("o2 arena", "A-O2", "The O2 Arena", "name", "london",
          "United Kingdom")])

    # Five Festhallen in two countries; one real, four not. Plus the control.
    rows = [
        ("e1", "Festhalle", "festhalle", "Frankfurt", "frankfurt", "Germany", "DE"),
        ("e2", "Festhalle", "festhalle", "Bad Urach", "bad urach", "Germany", "DE"),
        ("e3", "Festhalle", "festhalle", "Konken", "konken", "Germany", "DE"),
        ("e4", "Festhalle", "festhalle", "Bern", "bern", "Switzerland", "CH"),
        ("e5", "Festhalle", "festhalle", "Takamatsu", "takamatsu", "Japan", "JP"),
        ("e6", "The O2 Arena", "o2 arena", "London", "london",
         "United Kingdom", "GB"),
    ]
    con.executemany(
        "INSERT INTO events (event_id, venue, venue_norm, city, city_norm, "
        "country, countryCode) VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    return con


def resolve(con):
    """The alias passes and the attach, exactly as resolve_arenas runs them."""
    BE.flag_generic_aliases(con)
    con.execute("UPDATE events SET arena_id = NULL")
    BE.resolve_by_alias(con, "events")
    con.execute(BE.ATTACH_ARENA)
    con.commit()
    return {r[0]: (r[1], r[2]) for r in con.execute(
        "SELECT city, arena_id, arena_capacity FROM events")}


def main():
    tmp = os.path.join(tempfile.mkdtemp(), "alias_fix_test.db")
    con = build_fixture(tmp)
    print(f"fixture: {tmp}\n")

    got = resolve(con)
    print(f"{'city':12s} {'arena_id':8s} capacity")
    for city, (aid, cap) in sorted(got.items()):
        print(f"{city:12s} {str(aid or '-'):8s} {cap if cap else '-'}")

    print()
    failures = []

    # 1. the real one still matches
    if got["Frankfurt"][0] != "A-FFM":
        failures.append("Frankfurt's Festhalle should still match A-FFM")

    # 2. nowhere else does, in any country
    for city in ("Bad Urach", "Konken", "Bern", "Takamatsu"):
        if got[city][0] is not None:
            failures.append(
                f"{city} matched {got[city][0]} -- a generic name escaped the "
                f"city check")
        if got[city][1]:
            failures.append(
                f"{city} inherited a capacity of {got[city][1]}")

    # 3. THE CONTROL: a specific name must still match with no city check.
    #    Without this, deleting the matcher would pass every test above.
    if got["London"][0] != "A-O2":
        failures.append(
            "the control failed: a specific, unambiguous name no longer "
            "matches, so the fix has broken ordinary resolution")
    if got["London"][1] != 20000:
        failures.append("the control did not inherit its arena capacity")

    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        raise SystemExit(1)

    print("PASS  Frankfurt matched; Bad Urach, Konken, Bern and Takamatsu did "
          "not;\n      and the unambiguous control still resolves.")
    con.close()


if __name__ == "__main__":
    main()
