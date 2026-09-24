#!/usr/bin/env python3
"""
Proof that Pollstar and fixture events get a countryCode, and the right one.

WHY THIS MATTERS

venue_uid is `name|city|countryCode`. setlist.fm rows carry the code; Pollstar
and fixture rows arrived with only the country name, so every one of them
(125,023 events) got a venue_uid ending in an empty code. A sport or family
show at a building therefore came out as a DIFFERENT building from the concerts
there -- `verizon center|washington|` against `verizon center|washington|US`.

WHAT IS CHECKED

  1. a Pollstar row at a setlist.fm building gets the same code, so the two
     share a venue_uid
  2. the three names Pollstar spells differently still resolve (fallbacks)
  3. an existing code is never overwritten, even one that looks wrong
  4. a name with NO dominant code is left empty rather than guessed at --
     the case the first version got wrong, by letting one stray code win a tie
  5. an unmappable country is left empty rather than invented

    python test_country_backfill.py
"""

import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_events as BE   # noqa: E402


def main():
    con = sqlite3.connect(":memory:")
    con.executescript(BE.SCHEMA)
    BE.ensure_columns(con)

    rows = []
    # ten clean setlist.fm rows define "united states" -> US ...
    for i in range(10):
        rows.append((f"s{i}", "setlistfm", "Verizon Center", "verizon center",
                     "washington", "United States", "US"))
    # ... and one stray bad code, which must NOT win (10 of 11 = 91% dominant)
    rows.append(("sX", "setlistfm", "Odd", "odd", "washington",
                 "United States", "XX"))
    rows += [
        ("s20", "setlistfm", "O2 Arena", "o2 arena", "prague", "Czechia", "CZ"),
        # a name split evenly between two codes: nothing dominates
        ("t1", "setlistfm", "A", "a", "x", "Ruritania", "RU"),
        ("t2", "setlistfm", "B", "b", "x", "Ruritania", "RT"),
        # Pollstar rows: name only
        ("p1", "pollstar", "Verizon Center", "verizon center", "washington",
         "United States", None),
        ("p2", "pollstar", "O2 Arena", "o2 arena", "prague", "Czech Republic", None),
        ("p3", "pollstar", "Coliseum", "coliseum", "hong kong", "Hong Kong", ""),
        ("p4", "pollstar", "Nowhere Hall", "nowhere hall", "atlantis", "Atlantis", None),
        ("p5", "pollstar", "C", "c", "x", "Ruritania", None),
    ]
    con.executemany(
        "INSERT INTO events (event_id, source, venue, venue_norm, city_norm, "
        "country, countryCode) VALUES (?,?,?,?,?,?,?)", rows)

    BE.backfill_country_codes(con)
    got = dict(con.execute("SELECT event_id, countryCode FROM events"))

    expect = {
        "p1": "US",   # 1. same code as the setlist.fm building
        "p2": "CZ",   # 2. fallback: Czech Republic -> Czechia's code
        "p3": "HK",   # 2. fallback, from an empty string rather than NULL
        "sX": "XX",   # 3. an existing code is never overwritten
        "p5": None,   # 4. no dominant code -> left empty, not guessed
        "p4": None,   # 5. unmappable -> left empty, not invented
    }
    failures = [f"{k}: got {got[k]!r}, expected {v!r}"
                for k, v in expect.items() if got[k] != v]

    uid = lambda e: con.execute(   # noqa: E731
        "SELECT venue_norm||'|'||city_norm||'|'||COALESCE(countryCode,'') "
        "FROM events WHERE event_id=?", (e,)).fetchone()[0]
    if uid("s0") != uid("p1"):
        failures.append(f"concert and Pollstar rows at one building still "
                        f"differ: {uid('s0')} vs {uid('p1')}")

    for k in sorted(expect):
        print(f"{k:4s} {str(got[k]):6s} (expected {expect[k]})")
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        raise SystemExit(1)
    print(f"\nPASS  Pollstar row now shares a venue_uid with the concerts "
          f"({uid('p1')}); fallbacks resolve; existing codes kept; ambiguous "
          f"and unknown countries left empty.")


if __name__ == "__main__":
    main()
