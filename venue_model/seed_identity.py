#!/usr/bin/env python3
"""
Filling venue_identity, the one table a human edits.

WHAT THIS IS FOR

venue_aliases and arena_aliases are derived tables: `load-arenas` and
`load-venue-aliases` empty and rebuild them on every pipeline run. Until now
they were rebuilt from two separate hand-edited inputs -- a dedup workbook and
arena_aliases_manual.csv -- which could disagree with each other and which no
single query could search.

venue_identity replaces both. This seeds it from everything already known,
then adds whatever the review has found, so that after one run there is a
single table where a spelling can be looked up and a building's every name
recovered.

    python seed_identity.py                     # show what would be written
    python seed_identity.py --apply             # write it
    python seed_identity.py --apply --min-confidence medium

WHERE THE ROWS COME FROM, IN ORDER OF TRUST

  dashboard   the arenas sheet: `name` and each `also_known_as` entry, carrying
              the arena_id. Hand-verified, so it wins any collision.
  dedup       the 189 decisions venue_dedup already had loaded into
              venue_aliases, which were reviewed when they were made.
  manual      arena_aliases_manual.csv, the file the old flow told you to edit.
  geo         the co-location review in arena_alias_review.py -- the new rows,
              and the only ones carrying a confidence grade below `high`.

Earlier sources are never overwritten by later ones. A `geo` proposal that
contradicts a `dashboard` row is dropped and counted, not applied, because the
whole point of one table is that it has one answer per key.

WHAT IT DOES NOT DO

It does not decide anything. Every row it writes is one a human already
approved -- the dashboard, the loaded dedup decisions, the manual CSV -- or one
the review graded `high` by an explicit rule you can read in
arena_alias_review.grade. Everything below that grade is written only when you
ask for it.
"""

import argparse
import csv
import datetime as dt
import os
import sqlite3
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_DB = os.path.join(os.path.dirname(HERE), "setlistfm.db")
MANUAL_CSV = os.path.join(os.path.dirname(HERE), "arena_aliases_manual.csv")
OUT_DIR = os.path.join(HERE, "reports")

sys.path.insert(0, HERE)
from arena_alias_review import norm_key   # noqa: E402

# Trust order. A key already claimed by an earlier source is never rewritten.
SOURCE_RANK = {"dashboard": 0, "dedup": 1, "manual": 2, "geo": 3}


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def ensure_table(con):
    con.executescript("""
    CREATE TABLE IF NOT EXISTS venue_identity (
        alias_norm     TEXT NOT NULL,
        city_norm      TEXT NOT NULL,
        countryCode    TEXT NOT NULL,
        canonical_norm TEXT NOT NULL,
        canonical      TEXT,
        alias          TEXT,
        arena_id       TEXT,
        country        TEXT,
        source         TEXT,
        confidence     TEXT,
        evidence       TEXT,
        note           TEXT,
        loaded_at      TEXT,
        PRIMARY KEY (alias_norm, city_norm, countryCode)
    );
    CREATE INDEX IF NOT EXISTS ix_identity_canon
        ON venue_identity(canonical_norm, city_norm, countryCode);
    CREATE INDEX IF NOT EXISTS ix_identity_arena ON venue_identity(arena_id);
    """)


# ---------------------------------------------------------------------------
# The sources
# ---------------------------------------------------------------------------

def from_dashboard(con):
    """
    The arenas sheet: the canonical name and every also_known_as spelling.

    These carry an arena_id and are hand-verified, so they seed both lookups at
    once and outrank everything below. The canonical is the arena's own `name`,
    which is the spelling the dashboard considers current.
    """
    a = pd.read_sql("SELECT name, also_known_as, city, country, arena_id "
                    "FROM arenas WHERE arena_id IS NOT NULL", con)
    rows = []
    for r in a.itertuples(index=False):
        canon, city = str(r.name or "").strip(), str(r.city or "").strip()
        if not canon:
            continue
        cc = country_code(con, r.country)
        spellings = [canon] + [p.strip() for p in
                               str(r.also_known_as or "").split(",") if p.strip()]
        for sp in spellings:
            rows.append({
                "alias_norm": norm_key(sp), "city_norm": norm_key(city),
                "countryCode": cc or "", "canonical_norm": norm_key(canon),
                "canonical": canon, "alias": sp, "arena_id": r.arena_id,
                "country": r.country, "source": "dashboard",
                "confidence": "high", "evidence": None,
                "note": "arenas sheet",
            })
    return pd.DataFrame(rows)


_CC_CACHE = {}


def country_code(con, country):
    """
    The two-letter code for a dashboard country name.

    The arenas sheet stores 'Italy' while events store 'IT', and venue_identity
    is keyed on the code because venue_uid is. Resolved from the events table
    rather than a hard-coded map so it follows whatever the data actually uses.
    """
    if not country:
        return ""
    key = str(country).strip().lower()
    if key in _CC_CACHE:
        return _CC_CACHE[key]
    row = con.execute("SELECT countryCode FROM events WHERE lower(country)=? "
                      "AND countryCode IS NOT NULL AND countryCode <> '' "
                      "LIMIT 1", (key,)).fetchone()
    _CC_CACHE[key] = row[0] if row else ""
    return _CC_CACHE[key]


def from_venue_aliases(con):
    """The decisions venue_dedup already had loaded -- reviewed when made."""
    d = pd.read_sql("""SELECT alias_norm, city_norm, countryCode,
                              canonical_norm, canonical, note
                       FROM venue_aliases""", con)
    if d.empty:
        return d
    d["alias"] = d["alias_norm"]
    d["arena_id"] = None
    d["country"] = None
    d["source"] = "dedup"
    d["confidence"] = "high"
    d["evidence"] = None
    return d


def from_manual_csv(path=MANUAL_CSV):
    """arena_aliases_manual.csv -- the file the old flow told you to edit."""
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            alias = (r.get("alias") or "").strip()
            aid = (r.get("arena_id") or "").strip()
            if not alias or not aid:
                continue
            rows.append({
                "alias_norm": norm_key(alias), "city_norm": norm_key(r.get("city")),
                "countryCode": "", "canonical_norm": norm_key(alias),
                "canonical": alias, "alias": alias, "arena_id": aid,
                "country": r.get("country"), "source": "manual",
                "confidence": "high", "evidence": None,
                "note": "arena_aliases_manual.csv",
            })
    return pd.DataFrame(rows)


def from_review(path=None, min_confidence="high"):
    """
    The co-location review's proposals.

    Read from the workbook rather than recomputed, so what lands in the table
    is exactly what was on screen when it was reviewed -- and so editing a
    confidence cell in the workbook is enough to change what gets seeded.
    """
    if path is None:
        hits = sorted(f for f in os.listdir(OUT_DIR)
                      if f.startswith("arena_alias_review_")) if os.path.isdir(OUT_DIR) else []
        if not hits:
            return pd.DataFrame()
        path = os.path.join(OUT_DIR, hits[-1])
    d = pd.read_excel(path, sheet_name="renames")
    allowed = {"high": {"high"}, "medium": {"high", "medium"},
               "low": {"high", "medium", "low"}}[min_confidence]
    d = d[d["confidence"].isin(allowed) & ~d["already covered"]
          & d["same city and country"]]
    if d.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "alias_norm": d["alias_norm"],
        "city_norm": d["city"].map(norm_key),
        "countryCode": d["countryCode"].fillna(""),
        "canonical_norm": d["canonical_norm"],
        "canonical": d["canonical (keep this)"],
        "alias": d["alias (fold this away)"],
        "arena_id": d["arena_id"],
        "country": d.get("country"),
        "source": "geo",
        "confidence": d["confidence"],
        "evidence": ("co-located " + d["metres apart"].astype(str) + " m, gap "
                     + d["gap between runs (days)"].astype(str) + " d, capacity x"
                     + d["capacity ratio"].astype(str)),
        "note": os.path.basename(path),
    })


# ---------------------------------------------------------------------------
# Assembling
# ---------------------------------------------------------------------------

def assemble(con, min_confidence="high", review_path=None):
    """
    Everything, with earlier sources winning every collision.

    Collisions are counted rather than silently resolved: a geo proposal that
    contradicts the dashboard is the review being wrong about a building
    somebody has already verified, and that is worth seeing.
    """
    parts = []
    for name, frame in (("dashboard", from_dashboard(con)),
                        ("dedup", from_venue_aliases(con)),
                        ("manual", from_manual_csv()),
                        ("geo", from_review(review_path, min_confidence))):
        if len(frame):
            log(f"   {name:10s} {len(frame):6,} rows")
            parts.append(frame)
    if not parts:
        return pd.DataFrame(), pd.DataFrame()

    d = pd.concat(parts, ignore_index=True)
    for c in ("alias_norm", "city_norm", "countryCode", "canonical_norm"):
        d[c] = d[c].fillna("").astype(str)
    d = d[d["alias_norm"] != ""]
    d["_rank"] = d["source"].map(SOURCE_RANK)

    key = ["alias_norm", "city_norm", "countryCode"]
    d = d.sort_values("_rank", kind="stable")
    winners = d.drop_duplicates(key, keep="first")
    # a losing row that disagrees with the winner, rather than merely repeating it
    merged = d.merge(winners[key + ["canonical_norm", "source"]], on=key,
                     suffixes=("", "_win"))
    clashes = merged[(merged["_rank"] > merged["source_win"].map(SOURCE_RANK))
                     & (merged["canonical_norm"] != merged["canonical_norm_win"])]
    return resolve_chains(winners.drop(columns="_rank")), clashes


def resolve_chains(d):
    """
    Make every name point DIRECTLY at its final survivor.

    THE BUG THIS EXISTS TO PREVENT, WHICH IS ALREADY DOCUMENTED IN THE PIPELINE

    Sources disagree about which spelling is current, and the winner of each
    disagreement is picked per key. That alone produces chains:

        motorpoint arena -> trent fm arena nottingham -> capital fm arena nottingham

    venue_uid resolves with a SINGLE lookup, so the first name would land on a
    spelling that itself folds away and no longer survives -- and would end up
    as its own building after all, which is the exact failure this whole table
    is meant to remove. load_venue_aliases solves the same problem the same
    way; without it here, the derived table would reintroduce it.

    The survivor of a group is the member from the most trusted source, because
    that is the spelling somebody has actually verified. Within one source it
    is whichever name nothing else folds onto -- a root rather than a leaf.
    """
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Components are built WITHIN one (city, country); two names in different
    # cities are different buildings, which is the guard the whole schema rests
    # on and is not relaxed here.
    node = lambda n, c, k: (n, c, k)   # noqa: E731
    for r in d.itertuples():
        union(node(r.alias_norm, r.city_norm, r.countryCode),
              node(r.canonical_norm, r.city_norm, r.countryCode))

    # THE DASHBOARD'S OWN `name` IS THE CANONICAL SPELLING, ALWAYS.
    #
    # A row where the dashboard lists a name as its own canonical is that
    # building's CURRENT spelling, hand-verified. Everything else -- including
    # a dedup decision made when the old name still carried more events -- is
    # describing history. Without this the survivor came out as whichever
    # spelling happened to weigh most, which is nearly always the OLD sponsor:
    # Nottingham resolved onto `capital fm arena nottingham` when the dashboard
    # plainly calls it Motorpoint Arena Nottingham.
    dash_names = set(zip(
        d.loc[(d["source"] == "dashboard")
              & (d["alias_norm"] == d["canonical_norm"]), "canonical_norm"],
        d.loc[(d["source"] == "dashboard")
              & (d["alias_norm"] == d["canonical_norm"]), "city_norm"],
        d.loc[(d["source"] == "dashboard")
              & (d["alias_norm"] == d["canonical_norm"]), "countryCode"]))

    # Best (lowest) source rank over every row where a node acts as canonical.
    # A minimum, not a last-one-wins dict: a name can be the canonical of rows
    # from several sources, and the most trusted of those is what it is worth.
    canon_rank = {}
    for r in d.itertuples():
        n = node(r.canonical_norm, r.city_norm, r.countryCode)
        rk = SOURCE_RANK.get(r.source, 9)
        if rk < canon_rank.get(n, 99):
            canon_rank[n] = rk

    folded = set(zip(d.loc[d["alias_norm"] != d["canonical_norm"], "alias_norm"],
                     d.loc[d["alias_norm"] != d["canonical_norm"], "city_norm"],
                     d.loc[d["alias_norm"] != d["canonical_norm"], "countryCode"]))

    best = {}
    for r in d.itertuples():
        for n in (node(r.alias_norm, r.city_norm, r.countryCode),
                  node(r.canonical_norm, r.city_norm, r.countryCode)):
            root = find(n)
            score = (0 if n in dash_names else 1,   # the dashboard's current name
                     canon_rank.get(n, 9),          # else the most trusted source
                     n in folded)                   # else a root, not a leaf
            if root not in best or score < best[root][0]:
                best[root] = (score, n)

    out = d.copy()
    survivors = [best[find(node(r.alias_norm, r.city_norm, r.countryCode))][1]
                 for r in d.itertuples()]
    out["canonical_norm"] = [s[0] for s in survivors]
    readable = dict(zip(zip(d["canonical_norm"], d["city_norm"], d["countryCode"]),
                        d["canonical"]))
    readable.update(dict(zip(zip(d["alias_norm"], d["city_norm"], d["countryCode"]),
                             d["alias"])))
    out["canonical"] = [readable.get(s, s[0]) for s in survivors]
    return out


def write(con, d):
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    cols = ["alias_norm", "city_norm", "countryCode", "canonical_norm",
            "canonical", "alias", "arena_id", "country", "source",
            "confidence", "evidence", "note"]
    payload = [tuple(r) + (now,) for r in d[cols].where(pd.notna(d[cols]), None)
               .itertuples(index=False, name=None)]
    con.execute("DELETE FROM venue_identity")
    con.executemany(
        f"INSERT OR REPLACE INTO venue_identity ({','.join(cols)}, loaded_at) "
        f"VALUES ({','.join('?' * (len(cols) + 1))})", payload)
    con.commit()
    return len(payload)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=MAIN_DB)
    ap.add_argument("--review", default=None, help="review workbook to seed from")
    ap.add_argument("--min-confidence", default="high",
                    choices=["high", "medium", "low"])
    ap.add_argument("--apply", action="store_true",
                    help="write to venue_identity; without it, only report")
    a = ap.parse_args()

    con = sqlite3.connect(a.db)
    ensure_table(con)
    log("assembling venue_identity ...")
    d, clashes = assemble(con, a.min_confidence, a.review)
    if d.empty:
        raise SystemExit("nothing to seed")

    print(f"\n{'=' * 70}")
    print(f"venue_identity would hold {len(d):,} rows")
    print(d.groupby("source").size().rename("rows").to_string())
    folds = int((d["alias_norm"] != d["canonical_norm"]).sum())
    print(f"\n  folds (alias -> a different canonical) {folds:,}")
    print(f"  registrations (alias is its own canonical) {len(d) - folds:,}")
    print(f"  carrying an arena_id                     "
          f"{int(d['arena_id'].notna().sum()):,}")
    if len(clashes):
        print(f"\n  !! {len(clashes):,} lower-trust row(s) contradicted a "
              f"higher-trust one and were dropped")
        print(clashes[["alias_norm", "city_norm", "source", "canonical_norm",
                       "source_win", "canonical_norm_win"]].head(10)
              .to_string(index=False))
    print("=" * 70)

    if not a.apply:
        print("\nnothing written. re-run with --apply")
        return
    n = write(con, d)
    log(f"wrote {n:,} rows to venue_identity")
    log("now run:  python build_events.py load-arenas   (rebuilds both lookups)")


if __name__ == "__main__":
    main()
