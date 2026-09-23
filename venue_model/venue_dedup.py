#!/usr/bin/env python3
"""
One row per building — finding the venues that are the same place twice.

THE PROBLEM

`venues` is keyed on venue_uid = venue_norm|city_norm|countryCode, so a
building that changed its name is several rows. AO Arena, Manchester Arena and
MEN Arena are one room and three rows, with the events split between them. Any
capacity ladder built on that under-counts a venue's activity and can miss its
size entirely if the capacity only ever landed on one of the names.

WHY NAME SIMILARITY ALONE CANNOT SOLVE IT

The obvious fix -- merge venues in the same city whose names look alike -- is
actively dangerous, and the data says so loudly. In Manchester:

    manchester academy 1                    5,050 capacity
    manchester academy 2                    1,500
    manchester academy 3                      553
    club academy                              908

Those score 94-98 against each other on token similarity and they are four
different rooms in one students' union. Merging them would destroy the ladder
they are supposed to describe. The same trap catches `fillmore` against `lounge
at the fillmore`, and `showbox` against `showbox sodo`.

THE DISCRIMINATOR THAT DOES WORK

A rename replaces a name; a sub-room coexists with one. So:

    DISJOINT date ranges     -> the old name stopped when the new one started.
                                A rename.
    OVERLAPPING date ranges  -> both names were in use at once. Different
                                rooms, and they must not be merged.

Across 4,640 similar-name pairs within the same city this splits 1,192 renames
from 3,448 sub-rooms, and an entirely independent signal agrees: the median
capacity ratio is 1.08 for the disjoint pairs against 1.42 for the concurrent
ones. Same building versus different rooms.

THE RULE THAT MAKES THE WHOLE THING SAFE

Candidates are only ever generated WITHIN a single (city, country). That is not
an optimisation, it is the guard: it makes it structurally impossible to repeat
the arena_alias bug, where one `festhalle` alias was applied to fifty-two towns
in four countries and gave a village hall in Konken the 15,000 capacity of
Festhalle Messe Frankfurt. Two names in different cities are different
buildings. There is no exception this file will ever make.
"""

import argparse
import datetime as dt
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_DB = os.path.join(os.path.dirname(HERE), "setlistfm.db")
OUT_DIR = os.path.join(HERE, "reports")

# Token-similarity floor for a pair to be considered at all. 88 is loose on
# purpose: this stage is a candidate generator, and the date test below does
# the real discriminating. Missing a rename here cannot be recovered later.
NAME_SCORE = 88

# Share of the shorter venue's active span that the two must overlap before
# they are treated as concurrent. Not zero, because a rename is rarely clean --
# a listing under the old name can trail the switch by a few weeks.
CONCURRENT_FRACTION = 0.25

# A venue needs this many events before it is worth deduping. Below it the
# date range is a point, so the disjoint test has nothing to work with.
MIN_EVENTS = 3

# Cities with more rooms than this are skipped: the pair count grows as the
# square, and a city with 400+ distinct venues is a data-quality problem of a
# different kind.
MAX_VENUES_PER_CITY = 400

# Sponsor and operator tokens. A name that differs from another ONLY by these
# is a rebrand, which is the single commonest rename in live music.
#
# This list is domain knowledge, not something derivable from the data, so it
# lives here where it can be read and argued with rather than inside a scoring
# function.
SPONSORS = {
    "o2", "ao", "men", "carling", "3", "3arena", "3olympia", "citizens",
    "vodafone", "movistar", "barclaycard", "barclays", "santander", "utilita",
    "motorpoint", "ovo", "co op", "coop", "bonus", "first direct", "m and s",
    "m and s bank", "resorts world", "mercedes benz", "telekom", "lanxess",
    "sap", "uber", "crypto com", "chase", "t mobile", "xfinity", "toyota",
    "honda", "prudential", "capital one", "rocket mortgage", "little caesars",
    "fiserv", "delta", "footprint", "climate pledge", "moda", "paycom",
    "frost bank", "at and t", "golden 1", "bridgestone", "nationwide",
    "keybank", "wells fargo", "cfg bank", "amalie", "kaseya", "amerant",
    "gainbridge", "ball", "verizon", "sprint", "enterprise", "ppg",
    "smoothie king", "target", "united center", "scotiabank", "canadian tire",
    "rogers", "bell", "videotron", "place bell", "avicii", "friends",
    "tele2", "hovet", "globen", "jyske bank", "royal", "forum", "zenith",
    "accor", "bercy", "paris la defense", "adidas", "puma", "emirates",
    "etihad", "eventim", "mitsubishi electric", "wizink", "riyadh air",
    "utilita", "swansea com", "principality", "cardiff city",
}

# Names too vague to be a building. A one-word generic like "club" or "hall"
# is a data-quality artifact -- it is what a listing says when nobody recorded
# the venue -- and merging anything into it invents a room. Amsterdam had
# "Club" (1,500 capacity, no dates that overlap anything) queued to absorb
# "Club 11", which is a real venue it has nothing to do with.
GENERIC_NAMES = {
    "club", "hall", "theatre", "theater", "arena", "stadium", "stadion",
    "venue", "the venue", "auditorium", "upstairs", "downstairs", "basement",
    "bar", "cafe", "centre", "center", "pavilion", "festival", "park",
    "square", "church", "cathedral", "museum", "gallery", "studio", "studios",
    "tent", "big top", "main stage", "outdoor", "indoor", "unknown",
}

# Words that mark a SMALLER ROOM inside a bigger venue. Their presence is
# strong evidence against merging even when the dates look disjoint, because a
# side room often has sparse listings that happen not to overlap.
SUBROOM_MARKERS = {
    "lounge", "bar", "club", "cafe", "basement", "upstairs", "downstairs",
    "studio", "hall 2", "hall 3", "room 2", "second stage", "small",
    "kleine", "klein", "petite", "sala 2", "annex", "parish", "foundry",
    "attic", "loft", "garage", "backstage", "gallery", "terrace", "rooftop",
    "grounds", "park", "plaza", "arena 2", "academy 2", "academy 3",
}


# Words that describe what a building IS rather than which building it is.
# When two names differ only by these -- or by the city they are already in --
# they are the same place written at different lengths: "Brooklyn Paramount"
# against "Brooklyn Paramount Theatre", "Effenaar" against "Poppodium
# Effenaar", "Ludlow Garage" against "Ludlow Garage Cincinnati".
DESCRIPTIVE = {
    "theatre", "theater", "centre", "center", "coliseum", "stadium", "arena",
    "hall", "poppodium", "concert", "venue", "campground", "convention",
    "auditorium", "playhouse", "complex", "the", "at", "and", "of", "in",
    "s", "zaal", "sala", "salle", "teatro", "teatre", "halle", "palais",
    "pavilion", "amphitheater", "amphitheatre", "music",
    # "palace" is deliberately NOT here. It reads like a descriptor but forms
    # distinct venue identities -- Pasadena's The Rose and Rose Palace are two
    # different rooms, and with "palace" in this set they merged.
}

# Words that pick one room or site out of several. Their presence means the
# two names are DISTINGUISHING themselves from each other, which is the exact
# opposite of a rename -- The Fillmore against Fillmore West, Debaser Slussen
# against Debaser Slussen Terrassen, The Regency Ballroom against The Grand
# Ballroom at the Regency Center.
DISTINGUISHERS = {
    "west", "east", "north", "south", "upper", "lower", "grand", "main",
    "big", "little", "old", "new", "annex", "terrassen", "terrace", "side",
    "bayside", "front", "back", "red", "blue", "green", "black", "white",
    "one", "two", "three", "a", "b", "i", "ii", "iii",
}


# Pairs reviewed by hand, where the rules get it wrong or cannot tell.
#
# Rules handle the bulk, but a few hundred pairs turn on knowing what a place
# actually is, and no amount of string comparison recovers that. These were
# read individually, worst-impact first; the reason is recorded so a later
# reader can disagree with the judgement rather than just the outcome.
#
# Keyed on (keep_norm, fold_norm) as the candidate generator orders them:
# busier venue first. NOTE the keys carry no leading "the": norm_key strips it,
# so an override written as ("the fillmore", ...) matches nothing. Ten of the
# first twenty-six did exactly that and silently did nothing, which is why
# check_overrides() below now refuses to let it happen quietly.
OVERRIDES = {
    # --- genuine renames the rules were too cautious about ----------------
    ("olympia theatre", "3olympia theatre"):
        (True, "Dublin's Olympia took the 3 sponsorship in 2022"),
    ("lka longhorn", "longhorn"): (True, "same Stuttgart club, LKA prefix"),
    ("de helling", "tivoli de helling"): (True, "Utrecht, Tivoli operator prefix"),
    ("gabe s", "gabe s oasis"): (True, "Iowa City, same room renamed"),
    ("nycb theatre at westbury", "theatre at westbury"):
        (True, "NYCB was the sponsor"),
    ("kyocera dome osaka", "osaka dome"): (True, "Kyocera naming rights"),
    ("cannery ballroom", "cannery"): (True, "same Nashville building"),
    ("hedon grote zaal", "hedon"): (True, "Zwolle, main room of the venue"),
    ("skeppet", "skeppet eira"): (True, "Gothenburg, same room"),
    ("ferret", "mad ferret"): (True, "Preston, renamed"),
    ("allas sea pool", "allas pool"): (True, "Helsinki, same site"),
    ("great woods center for the performing arts",
     "tweeter center for the performing arts"):
        (True, "Mansfield amphitheatre, successive sponsors"),
    ("le studio td", "le studio"): (True, "Montreal, TD naming rights"),
    ("lantarenvenster", "lantaren venster"):
        (True, "Rotterdam, punctuation only"),

    # --- different places the rules wanted to merge -----------------------
    ("rose", "rose palace"):
        (False, "Pasadena has both, and they are not the same room"),
    ("o2 academy liverpool", "academy"):
        (False, "'The Academy' is ambiguous and carries only 5 events"),
    ("brighton music hall", "music hall"):
        (False, "Boston: 'The Music Hall' is not Brighton Music Hall"),
    ("mgm music hall at fenway", "music hall"):
        (False, "same, a generic name against a specific one"),
    ("halle", "rudi sedlmayer halle"):
        (False, "Munich: 'Halle' is a placeholder, and 1,500 against 6,534"),
    ("zepp namba osaka", "zepp osaka"):
        (False, "two different Zepp rooms in Osaka"),
    ("horn", "horn of plenty"): (False, "St Albans, unrelated pubs"),
    ("coca cola roxy", "roxy"):
        (False, "Atlanta: the Coca-Cola Roxy is not the old Buckhead Roxy"),
    ("once", "once boynton yards"):
        (False, "Somerville: a relocation to a different address"),
    ("centro de convenciones festiva", "centro de convenciones claro"):
        (False, "Lima, two different convention centres"),
    ("regency ballroom", "grand ballroom at the regency center"):
        (False, "two rooms in the same San Francisco building"),
    ("fillmore", "fillmore west"):
        (False, "different San Francisco venues, different eras"),
}


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def load_venues(db=MAIN_DB, min_events=MIN_EVENTS):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=180)
    try:
        return pd.read_sql(
            """SELECT venue_uid, venue, venue_norm, city, city_norm, country,
                      countryCode, events, capacity, capacity_source,
                      outside_inside, arena_id, first_event, last_event
               FROM venues
               WHERE events >= ? AND TRIM(COALESCE(venue_norm,'')) <> ''""",
            con, params=[min_events])
    finally:
        con.close()


def _overlap_fraction(a1, a2, b1, b2):
    """
    How much of the shorter venue's life the two ranges share.

    Returns a negative number when they do not overlap at all, so the sign
    alone separates "ran at the same time" from "one replaced the other".
    """
    try:
        a1, a2, b1, b2 = (pd.Timestamp(a1), pd.Timestamp(a2),
                          pd.Timestamp(b1), pd.Timestamp(b2))
    except (ValueError, TypeError):
        return np.nan
    lo, hi = max(a1, b1), min(a2, b2)
    days = (hi - lo).days
    shorter = max(min((a2 - a1).days, (b2 - b1).days), 1)
    return days / shorter


def _residual_tokens(a, b):
    """The words that differ between two names, ignoring shared ones."""
    ta, tb = set(a.split()), set(b.split())
    return (ta - tb) | (tb - ta)


def candidates(v, score_floor=NAME_SCORE):
    """
    Every within-city pair of venues whose names are similar.

    Deliberately generous. The decision is made downstream; this only has to
    avoid missing anything, and a pair it never generates can never be merged.
    """
    rows = []
    groups = v.groupby(["countryCode", "city_norm"])
    for n, ((cc, city), g) in enumerate(groups, 1):
        if len(g) < 2 or len(g) > MAX_VENUES_PER_CITY:
            continue
        r = g.to_dict("records")
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                a, b = r[i], r[j]
                s = fuzz.token_set_ratio(a["venue_norm"], b["venue_norm"])
                if s < score_floor:
                    continue
                # order so `a` is always the busier of the two: the surviving
                # name should be the one the data actually uses
                if b["events"] > a["events"]:
                    a, b = b, a
                rows.append({
                    "countryCode": cc, "city": a["city"],
                    "keep": a["venue"], "fold_in": b["venue"],
                    "keep_norm": a["venue_norm"], "fold_norm": b["venue_norm"],
                    "score": round(s, 1),
                    "keep_events": a["events"], "fold_events": b["events"],
                    "keep_capacity": a["capacity"], "fold_capacity": b["capacity"],
                    "keep_dates": f"{a['first_event']} to {a['last_event']}",
                    "fold_dates": f"{b['first_event']} to {b['last_event']}",
                    "overlap_frac": round(_overlap_fraction(
                        a["first_event"], a["last_event"],
                        b["first_event"], b["last_event"]), 2),
                    "keep_uid": a["venue_uid"], "fold_uid": b["venue_uid"],
                    "differing_words": " / ".join(
                        sorted(_residual_tokens(a["venue_norm"],
                                                b["venue_norm"]))) or "(none)",
                })
        if n % 4000 == 0:
            log(f"   scanned {n:,} cities, {len(rows):,} candidate pairs")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

def classify(c):
    """
    A verdict and a reason for every pair.

    Each rule is a separate, nameable test rather than a score, so a reader can
    disagree with one pair without having to distrust the whole file. The
    reason column carries which rule fired.
    """
    c = c.copy()
    cap_a = pd.to_numeric(c["keep_capacity"], errors="coerce")
    cap_b = pd.to_numeric(c["fold_capacity"], errors="coerce")
    both_cap = cap_a.notna() & cap_b.notna()
    ratio = (np.maximum(cap_a, cap_b) / np.minimum(cap_a, cap_b)).where(both_cap)
    c["capacity_ratio"] = ratio.round(2)

    concurrent = c["overlap_frac"] > CONCURRENT_FRACTION
    diff_words = c["differing_words"].str.split(" / ")
    # A BARE DIGIT IS NOT A SPONSOR, and treating it as one was a real bug.
    # It queued Liverpool's "O2 Academy 2" to merge into "The Academy": the
    # names differ by {o2, 2}, and counting "2" as noise left only a sponsor
    # token. But a bare number is how a building names its SECOND ROOM --
    # Academy 2, ABC2, Hall 2, Sala 2 -- so the digit is the very thing that
    # distinguishes them.
    only_sponsor = diff_words.apply(
        lambda ws: bool(ws) and ws != ["(none)"]
        and all(w in SPONSORS for w in ws if w))
    numbered = diff_words.apply(
        lambda ws: any(w.isdigit() for w in ws if w))
    distinguishing = diff_words.apply(
        lambda ws: any(w in DISTINGUISHERS for w in ws if w))
    # City tokens count as descriptive: a venue naming its own city is not
    # naming a different building.
    city_tokens = c["city"].astype(str).str.lower().str.split().apply(set)
    only_descriptive = [
        bool(ws) and ws != ["(none)"]
        and all(w in DESCRIPTIVE or w in SPONSORS or w in ct
                for w in ws if w)
        for ws, ct in zip(diff_words, city_tokens)]
    only_descriptive = pd.Series(only_descriptive, index=c.index)
    generic = (c["keep_norm"].isin(GENERIC_NAMES)
               | c["fold_norm"].isin(GENERIC_NAMES))
    has_subroom = diff_words.apply(
        lambda ws: any(w in SUBROOM_MARKERS for w in ws if w))
    much_smaller = both_cap & (ratio >= 2.0)

    c["verdict"] = np.select(
        [
            generic,
            distinguishing,
            numbered,
            has_subroom & concurrent,
            concurrent & much_smaller,
            concurrent,
            has_subroom,
            only_sponsor & ~concurrent,
            only_descriptive & ~concurrent,
            (~concurrent) & both_cap & (ratio < 1.5),
            (~concurrent) & ~both_cap,
        ],
        [
            "keep separate: one name is too generic to be a building",
            "keep separate: names differ by a word that picks one room or "
            "site from several",
            "keep separate: names differ by a number, so a second room",
            "keep separate: sub-room, ran at the same time",
            "keep separate: ran at the same time, and one is half the size",
            "keep separate: both names in use at once",
            "keep separate: name marks a smaller room inside the venue",
            "MERGE: sponsor rename, dates do not overlap",
            "MERGE: same name plus descriptive words only",
            "MERGE: dates do not overlap and capacities agree",
            "review: dates do not overlap but capacity is unknown",
        ],
        default="review: unclear")
    # NOT called "merge": that shadows DataFrame.merge, so `df.merge` silently
    # returns the method instead of the column and every filter written the
    # natural way fails with a bare KeyError: False.
    # Hand-reviewed pairs win over any rule.
    for i, row in c.iterrows():
        hit = OVERRIDES.get((row["keep_norm"], row["fold_norm"]))
        if hit is None:
            hit = OVERRIDES.get((row["fold_norm"], row["keep_norm"]))
        if hit is not None:
            keep, why = hit
            c.at[i, "verdict"] = (f"MERGE: reviewed by hand -- {why}" if keep
                                  else f"keep separate: reviewed by hand -- {why}")

    c["should_merge"] = c["verdict"].str.startswith("MERGE")
    return c.sort_values(["should_merge", "keep_events"],
                         ascending=[False, False])


def check_overrides(c):
    """
    Every hand-reviewed pair must match a real candidate.

    A judgement that matches nothing is worse than no judgement: it looks like
    the case was handled. Ten of the first batch missed because norm_key strips
    a leading "the" and the keys were written with it, so this is now checked
    rather than assumed.
    """
    seen = set(zip(c["keep_norm"], c["fold_norm"]))
    seen |= set(zip(c["fold_norm"], c["keep_norm"]))
    missing = [k for k in OVERRIDES if k not in seen]
    return missing


def write(c, path=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = path or os.path.join(
        OUT_DIR, f"venue_dedup_{dt.date.today():%Y-%m-%d}.xlsx")
    cols = ["countryCode", "city", "keep", "fold_in", "verdict", "should_merge",
            "score", "overlap_frac", "capacity_ratio",
            "keep_capacity", "fold_capacity", "keep_events", "fold_events",
            "keep_dates", "fold_dates", "differing_words",
            "keep_norm", "fold_norm", "keep_uid", "fold_uid"]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        c[cols].to_excel(xw, sheet_name="All candidates", index=False)
        c[c["should_merge"]][cols].to_excel(xw, sheet_name="To merge", index=False)
        c[c["verdict"].str.startswith("review")][cols].to_excel(
            xw, sheet_name="Needs a human", index=False)
        c[c["verdict"].str.startswith("keep separate")][cols].to_excel(
            xw, sheet_name="Deliberately not merged", index=False)
        pd.DataFrame({"note": [
            "Candidates are only ever generated WITHIN one city and country. "
            "Two names in different cities are different buildings, always.",
            f"A pair is concurrent when their date ranges overlap by more than "
            f"{CONCURRENT_FRACTION:.0%} of the shorter venue's span. Concurrent "
            f"means two rooms, not one renamed room.",
            "'Deliberately not merged' is kept so these pairs are not "
            "re-proposed every run, and so a reader can check what was "
            "rejected as well as what was accepted.",
            "Edit the `should_merge` column to overrule any verdict, then "
            "load the file back in.",
        ]}).to_excel(xw, sheet_name="Method", index=False)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=MAIN_DB)
    ap.add_argument("--min-events", type=int, default=MIN_EVENTS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    log("loading venues ...")
    v = load_venues(a.db, a.min_events)
    log(f"   {len(v):,} venue rows with {a.min_events}+ events")
    log("finding within-city near-duplicates ...")
    c = candidates(v)
    log(f"   {len(c):,} candidate pairs")
    c = classify(c)

    missing = check_overrides(c)
    if missing:
        log(f"   !! {len(missing)} hand-reviewed override(s) matched no "
            f"candidate pair and did nothing:")
        for k in missing:
            log(f"        {k}")
    else:
        log(f"   all {len(OVERRIDES)} hand-reviewed overrides matched a pair")

    counts = c["verdict"].value_counts()
    print("\nVerdicts:")
    for k, n in counts.items():
        print(f"  {n:6,}  {k}")
    path = write(c, a.out)
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
