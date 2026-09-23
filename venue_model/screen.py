#!/usr/bin/env python3
"""
Layer 1, applied to every market at once -- a screening pass.

WHAT THIS IS FOR

`gap.py` answers "is there a case in this market?". This asks the inverse:
"across every market we have, where are the cases?" -- so the tool proposes
candidates instead of waiting to be asked about one.

It runs exactly the same arithmetic as gap.py for all 735 markets and lays the
components side by side. Nothing new is estimated; this is the same counting,
done everywhere.

DELIBERATELY NOT A SINGLE SCORE

It would be easy to blend the columns into one "opportunity index" and sort by
it. That would be worse, because a single number hides which part of it is
doing the work, and the weights would be mine rather than yours.

Instead each component stands on its own and the default ordering is by ONE
named quantity -- `gap_dates`, the number of show-dates that went elsewhere
held by tours whose stated reason was room size. Sort by a different column and
you get a different, equally defensible, ranking. The UI is expected to let you
do exactly that.

THE COLUMNS THAT MATTER

  gap_tours          tours that skipped and consistently played rooms bigger
                     than anything of that kind here
  gap_dates          how many show-dates those tours represent -- the volume
                     of displaced demand a building could plausibly compete for
  suggested_capacity the median room those tours actually used. If you built
                     for this market, this is the size the evidence points to
  indoor_ceiling     the biggest indoor room the market has today
  headroom           suggested_capacity minus the ceiling: how far short it is
  shows_per_million  current activity, normalised for catchment
  peer_median_spm    the same figure for markets of similar catchment
  vs_peers           ratio of the two. Above 1 means the market is ALREADY
                     over-performing for its size, which is evidence against a
                     capacity constraint, not for one

READ `vs_peers` BEFORE `gap_dates`

A market can show a large gap simply because it is large and busy. The peer
ratio is the check: a market already doing more shows per head than its
comparators is not obviously starved of capacity, however many tours skipped
it. Bari scores 86th percentile among its peers, which is exactly the sort of
context a raw skip count buries.

USAGE
    python screen.py
    python screen.py --country IT --min-dates 3
    python screen.py --sort gap_dates --limit 40
    python screen.py --min-events 40 --out reports
"""

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gap import (capacity_ladder, capacity_test, ceilings_from_ladder,   # noqa: E402
                 latest_extract, load_extract, tours_that_skipped)

# Catchment radius used for the peer comparison. 60 km is a reasonable
# afternoon's travel for a show and is the radius most stable across the
# seven countries; it is a parameter because nothing about it is sacred.
PEER_RADIUS_KM = 60

# Rank on the EXCLUSIVE catchment, not the raw radius.
#
# A 60 km circle round Guildford contains 13.8 million people, because London
# is 40 km away. A circle round Hanley contains 7.2 million, because Manchester
# and Birmingham are. Ranked on the raw radius both looked like enormous
# under-served markets, when their residents already have far better venues a
# short train ride away.
#
# The exclusive figure counts only residents for whom this market is the
# NEAREST one hosting shows. Guildford holds 855,000 of its 13.8 million and
# Hanley 916,000 of its 7.2 million; London keeps 8.0 million of 15.6 and Bari,
# with no competitor inside 60 km, keeps all 1.67 million. See
# catchment.exclusive_catchment for how the assignment is done and what it
# trades away.
CATCHMENT_COL = f"exclusive_population_{PEER_RADIUS_KM}km"

# Half-width of the peer population band. +/-35% is wide, because there are
# only a few hundred markets and a tight band leaves too few comparators for
# the median to mean anything.
PEER_TOLERANCE = 0.35

# A market is only ASSESSABLE for an indoor capacity gap if it already hosts
# indoor shows. Without this the ranking fills with festival fields: Pilton
# (Glastonbury), Bramham (Leeds Festival) and Silverstone have no indoor venue
# at all, so every tour that skipped them counted as blocked by capacity --
# technically true, analytically useless. "This village has no arena" is not a
# finding. A market that already runs indoor shows and cannot run big ones is.
MIN_INDOOR_EVENTS = 10

# And it needs enough people to fill a room. Without this the ranking fills
# with festival villages -- Belladrum, Frome, Steventon -- whose biggest indoor
# room seats 800, so virtually every touring act "needs something bigger" and
# they score a thousand capacity gaps apiece. A market of 40,000 people is not
# short of an arena; it is short of people.
MIN_CATCHMENT_60KM = 250_000

# City names in the source data that are placeholders rather than places.
JUNK_MARKETS = {"unknown city", "unknown", "n/a", ""}

# Names that suggest a room could be played indoors even though the database
# calls it outdoor -- or leaves it unlabelled.
#
# This exists because of Lille. Its Stade Pierre-Mauroy is a retractable-roof
# stadium that converts to a 27,000-seat indoor hall and trades as "Decathlon
# Arena"; the database labels it Outside, so Lille's indoor ceiling reads 7,000
# (the Zenith) and Lille ranks second on the whole screen. The finding is an
# artefact of one label.
#
# Rather than guess at reclassifying such venues -- most stadiums called Arena
# really are open to the sky -- the screen raises a CAVEAT: it names the venue
# and lets a reader judge. A flagged market is not wrong, it is unverified.
# Graded, because the two groups are not equally informative.
#
# STRONG: words that in practice only ever name a roofed building. A venue
# called Palacio de los Deportes or Palasport is an indoor sports hall; if the
# database says Outside, the database is very likely wrong. This group caught a
# real error on the first run -- Malaga's Palacio de los Deportes Martin
# Carpena, an 8,950-seat indoor arena, labelled Outside, which is why Malaga
# appeared to have a 1,535-seat ceiling and ranked tenth on the whole screen.
#
# WEAK: "arena" alone, which is genuinely ambiguous. Lille's Decathlon Arena is
# a retractable-roof hall that really does host 27,000 indoors, but the
# Allianz Arena, Veltins-Arena and MHPArena are open-air football grounds
# correctly labelled Outside. A weak hit means look, not fix.
STRONG_INDOOR_HINTS = ("palacio de los deportes", "palais des sports", "palasport",
                       "palazzo dello sport", "palacongressi", "sporthalle",
                       "messehalle", "stadthalle", "festhalle", "olympiahalle")
WEAK_INDOOR_HINTS = ("arena", "forum", "dome", "halle", "hall", "pala")

# No size cap is applied, and that is a deliberate reversal.
#
# Capping weak hits at 30,000 seats did remove the football grounds -- but it
# also removed Lille's Decathlon Arena at 53,344, which is the one genuinely
# convertible venue in the top of the screen and the reason the flag was
# written. A rule that silences the noise by also silencing the finding is
# worse than no rule. Roughly fifteen markets carry a weak flag; reading
# fifteen venue names is cheap, and a false positive costs a glance whereas a
# false negative costs a wrong ranking.


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


# Windows consoles default to a codepage that cannot print Malaga or Osnabruck
# with their accents, and would mangle them to "M?laga". Market names are data;
# printing them wrong in the summary invites someone to search for the wrong
# string later.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def peer_table(markets, radius_km=PEER_RADIUS_KM, tolerance=PEER_TOLERANCE):
    """
    For every market, the median shows-per-million of markets its own size.

    Computed once for all markets rather than per market, because the inner
    comparison is the same work repeated. 735 x 735 is small enough to do
    exhaustively, which keeps it simple and exact rather than approximate.
    """
    col = CATCHMENT_COL if CATCHMENT_COL in markets.columns else f"population_{radius_km}km"
    pop = pd.to_numeric(markets[col], errors="coerce").values
    events = pd.to_numeric(markets["events"], errors="coerce").values
    with np.errstate(divide="ignore", invalid="ignore"):
        spm = np.where(pop > 0, events / (pop / 1e6), np.nan)

    # The biggest indoor room each market has, so we can ask what is NORMAL for
    # a market of this size. This is the discriminator that separates a real
    # candidate from a small town: Messina's 7,000 against peers at 11,000 is a
    # gap; a festival village's 800 against peers at 900 is simply a small town.
    ceiling = pd.to_numeric(markets["largest_indoor_capacity"], errors="coerce").values

    med = np.full(len(markets), np.nan)
    med_ceiling = np.full(len(markets), np.nan)
    n_peers = np.zeros(len(markets), dtype=int)
    for i, p in enumerate(pop):
        if not np.isfinite(p) or p <= 0:
            continue
        band = (pop >= p * (1 - tolerance)) & (pop <= p * (1 + tolerance))
        band[i] = False                       # never compare a market to itself
        vals = spm[band & np.isfinite(spm)]
        caps = ceiling[band & np.isfinite(ceiling)]
        if len(vals):
            med[i] = np.median(vals)
            n_peers[i] = len(vals)
        if len(caps):
            med_ceiling[i] = np.median(caps)
    return pd.DataFrame({"shows_per_million": np.round(spm, 1),
                         "peer_median_spm": np.round(med, 1),
                         "peer_median_ceiling": np.round(med_ceiling, 0),
                         "peers_compared": n_peers}, index=markets.index)


def ceiling_caveat(ladder, indoor_ceiling):
    """
    The largest non-indoor room that might really be indoor, if any.

    Looks for a venue that is (a) not labelled inside, (b) bigger than the
    indoor ceiling the verdict rests on, and (c) named like an indoor building.
    Returns its name and capacity, or None. See CONVERTIBLE_HINTS for why this
    is a flag and not a correction.
    """
    if ladder is None or ladder.empty:
        return None
    cap = pd.to_numeric(ladder["capacity"], errors="coerce")
    other = ladder[(ladder["io"] != "inside") & cap.notna()]
    if other.empty:
        return None
    if indoor_ceiling is not None and pd.notna(indoor_ceiling):
        other = other[pd.to_numeric(other["capacity"], errors="coerce") > indoor_ceiling]
    if other.empty:
        return None

    other = other.copy()
    other["_cap"] = pd.to_numeric(other["capacity"], errors="coerce")
    name = other["venue"].astype(str).str.lower()
    strong = name.apply(lambda n: any(h in n for h in STRONG_INDOOR_HINTS))
    weak = name.apply(lambda n: any(h in n for h in WEAK_INDOOR_HINTS))

    # Strong hits first regardless of size: a mislabelled sports hall matters
    # more than a bigger but ambiguous room.
    for flag, label in ((strong, "LIKELY MISLABELLED"), (weak, "check")):
        hit = other[flag]
        if not hit.empty:
            top = hit.sort_values("_cap", ascending=False).iloc[0]
            return (f"{label}: {top['venue']} ({int(top['_cap']):,} seats, "
                    f"labelled {top['io']} by {top['io_source']})")
    return None


def screen_market(ex, row, min_dates):
    """
    Run the layer-1 test for one market and reduce it to a row.

    Returns None where the market cannot support the test at all -- no tours
    skipped it, or it has no capacity information anywhere. Returning None
    rather than zeros matters: "we cannot tell" and "there is no gap" are
    different findings and must not be averaged together later.
    """
    ladder = capacity_ladder(ex, row)
    ceilings = ceilings_from_ladder(ladder)
    skipped = tours_that_skipped(ex, row, min_dates)
    if skipped.empty:
        return None
    tested = capacity_test(skipped, row, ceilings)

    # STRICT gap only. The separate "no venue of this kind" verdict is counted
    # apart, because a market with no indoor room and one with a room that is
    # merely too small are different propositions: the first may have no indoor
    # audience at all, the second has a demonstrated one it cannot serve.
    gap = tested[tested["verdict"] == "capacity gap"]
    no_venue = tested[tested["verdict"] == "capacity gap (none of this kind)"]
    borderline = tested[tested["verdict"] == "borderline"]
    # The size the evidence points at: what the constrained tours actually
    # played elsewhere. Median, not mean, because a single stadium act would
    # otherwise drag the recommendation into fantasy.
    suggested = pd.to_numeric(gap["room_it_needs"], errors="coerce").median()

    return {
        "market": row["city"], "country": row["country"], "countryCode": row["countryCode"],
        "events": int(row["events"]),
        "venues": int(row["venues"]),
        f"population_{PEER_RADIUS_KM}km": row.get(f"population_{PEER_RADIUS_KM}km"),
        CATCHMENT_COL: row.get(CATCHMENT_COL),
        "exclusive_mean_km": row.get("exclusive_mean_km"),
        "events_indoor": int(row.get("events_indoor") or 0),
        "ceiling_caveat": ceiling_caveat(ladder, ceilings.get("indoor")),
        "skipped_tours": len(tested),
        "gap_tours": len(gap),
        "gap_dates": int(pd.to_numeric(gap["country_dates"], errors="coerce").sum()),
        "gap_share": round(len(gap) / len(tested), 3) if len(tested) else None,
        "no_venue_tours": len(no_venue),
        "borderline_tours": len(borderline),
        "suggested_capacity": None if pd.isna(suggested) else int(round(suggested, -2)),
        "indoor_ceiling": ceilings["indoor"],
        "outdoor_ceiling": ceilings["outdoor"],
        "unlabelled_venues": ceilings["unlabelled_venues"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", default=None)
    ap.add_argument("--country", nargs="+", default=None, help="ISO-2 codes to restrict to")
    ap.add_argument("--min-dates", type=int, default=3,
                    help="country dates a tour needs before skipping a market counts")
    ap.add_argument("--min-events", type=int, default=20,
                    help="skip markets quieter than this; their counts are too noisy")
    ap.add_argument("--sort", default="ceiling_below_peers",
                    help="column to rank by. The default asks how far this market's "
                         "biggest indoor room sits below what markets of its size "
                         "typically have")
    ap.add_argument("--all-markets", action="store_true",
                    help="include markets that cannot be judged on indoor capacity "
                         "(festival fields and the like), which are hidden by default")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(HERE, "reports"))
    a = ap.parse_args()

    folder = a.extract or latest_extract()
    ex = load_extract(folder)
    markets = ex["markets"]
    if a.country:
        markets = markets[markets["countryCode"].isin([c.upper() for c in a.country])]
    markets = markets[pd.to_numeric(markets["events"], errors="coerce") >= a.min_events]
    log(f"extract {ex['manifest']['tag']}: screening {len(markets):,} markets "
        f"with >= {a.min_events} shows")

    peers = peer_table(ex["markets"])
    rows = []
    for i, (_, row) in enumerate(markets.iterrows(), start=1):
        r = screen_market(ex, row, a.min_dates)
        if r:
            rows.append(r)
        if i % 100 == 0:
            log(f"   {i:,}/{len(markets):,}")
    out = pd.DataFrame(rows)
    log(f"   {len(out):,} markets produced a result")

    # attach the peer context
    out = out.merge(
        ex["markets"][["city", "countryCode"]].join(peers).rename(columns={"city": "market"}),
        how="left", on=["market", "countryCode"])
    out["vs_peers"] = (out["shows_per_million"] / out["peer_median_spm"]).round(2)
    # How far short the current ceiling is of what the constrained tours used.
    out["headroom"] = (pd.to_numeric(out["suggested_capacity"], errors="coerce")
                       - pd.to_numeric(out["indoor_ceiling"], errors="coerce")).round(0)

    # HOW FAR BELOW COMPARABLE MARKETS this one's biggest indoor room sits.
    #
    # This is the headline measure, and it is the one that separates a genuine
    # candidate from a small town. It asks: given how many people live here,
    # what size room would a typical market of this size have -- and how much
    # smaller is ours? A positive number means under-built relative to peers.
    #
    # It is a comparison, not a forecast. It says nothing about whether the
    # gap is worth filling, only that it exists.
    out["ceiling_below_peers"] = (
        pd.to_numeric(out["peer_median_ceiling"], errors="coerce")
        - pd.to_numeric(out["indoor_ceiling"], errors="coerce")).round(0)
    out["ceiling_vs_peers"] = (pd.to_numeric(out["indoor_ceiling"], errors="coerce")
                               / pd.to_numeric(out["peer_median_ceiling"],
                                               errors="coerce")).round(2)

    # Can this market be judged on indoor capacity at all?
    # Share of the raw radius this market actually keeps once neighbours are
    # taken into account. A low value is the signal that a market is a satellite
    # rather than a place in its own right: Guildford keeps 6%, Hanley 13%,
    # against London's 51% and Bari's 100%.
    out["catchment_kept"] = (pd.to_numeric(out[CATCHMENT_COL], errors="coerce")
                             / pd.to_numeric(out[f"population_{PEER_RADIUS_KM}km"],
                                             errors="coerce")).round(2)

    pop60 = pd.to_numeric(out[CATCHMENT_COL], errors="coerce")
    out["assessable"] = (out["indoor_ceiling"].notna()
                         & (out["events_indoor"] >= MIN_INDOOR_EVENTS)
                         & (pop60 >= MIN_CATCHMENT_60KM)
                         & ~out["market"].str.strip().str.lower().isin(JUNK_MARKETS))

    # Absolute gap_dates mostly measures how many tours the COUNTRY gets: the
    # UK has 1,134 tours with 3+ dates against Italy's 198, so UK markets would
    # top any absolute ranking regardless of merit. Expressing the gap as a
    # share of the tours that toured that country makes markets comparable
    # across borders, and a within-country rank is kept for reading nationally.
    by_country = out.groupby("countryCode")["skipped_tours"].transform("max")
    out["gap_dates_per_1k_country_tours"] = (
        out["gap_dates"] / by_country.replace(0, np.nan) * 1000).round(0)
    out["rank_in_country"] = (out.groupby("countryCode")["gap_dates"]
                              .rank(ascending=False, method="min").astype("Int64"))

    cols = ["market", "country", "assessable", "events", "events_indoor",
            "ceiling_caveat",
            f"population_{PEER_RADIUS_KM}km", CATCHMENT_COL, "catchment_kept",
            "exclusive_mean_km", "indoor_ceiling", "peer_median_ceiling", "ceiling_below_peers",
            "ceiling_vs_peers", "suggested_capacity", "headroom",
            "skipped_tours", "gap_tours", "gap_dates", "gap_dates_per_1k_country_tours",
            "gap_share", "no_venue_tours", "borderline_tours", "outdoor_ceiling",
            "shows_per_million", "peer_median_spm", "vs_peers", "peers_compared",
            "rank_in_country", "venues", "unlabelled_venues", "countryCode"]
    out = out[cols].sort_values(a.sort, ascending=False)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"screen_{ex['manifest']['tag']}.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name="Screen", index=False)
        pd.DataFrame({"column": [
            "gap_tours", "gap_dates", "suggested_capacity", "headroom",
            "shows_per_million", "peer_median_spm", "vs_peers", "unlabelled_venues"],
            "meaning": [
            "Tours that skipped and always played rooms bigger than anything of that kind here.",
            "Show-dates those tours represent -- the volume of displaced demand.",
            "Median room those tours used elsewhere; the size the evidence points to.",
            "suggested_capacity minus the indoor ceiling: how far short the market is.",
            "Current shows per million residents within 60 km.",
            "The same figure for markets of similar catchment (+/-35%).",
            "shows_per_million / peer_median_spm. ABOVE 1 MEANS ALREADY OVER-PERFORMING, "
            "which is evidence against a capacity constraint.",
            "Venues with no indoor/outdoor label even after name inference; a large one "
            "here means the ceilings may be understated."],
        }).to_excel(xw, sheet_name="Columns", index=False)
        pd.DataFrame({"note": ex["notes"]["note"].tolist() + [
            f"Extract {ex['manifest']['tag']}, built {ex['manifest']['built_at']}.",
            f"A tour counts as skipping a market if it played >= {a.min_dates} dates in that "
            f"country and none in that market.",
            "Everything here is descriptive counting. No forecast, nothing causal.",
            "ceiling_caveat names a large non-indoor venue whose name suggests it may be "
            "playable indoors (Lille's Decathlon Arena, a retractable-roof stadium labelled "
            "Outside). Where it is populated, treat the indoor ceiling -- and therefore the "
            "whole verdict -- as unverified until the venue is checked by hand.",
            "There is deliberately no composite score: sort by whichever column matches the "
            "question you are asking.",
            f"Only markets with >= {MIN_INDOOR_EVENTS} indoor shows AND a known indoor "
            f"ceiling are marked assessable. Without that filter the ranking fills with "
            f"festival sites (Glastonbury, Leeds Festival, Silverstone), which have no indoor "
            f"venue at all and so register every skipping tour as blocked by capacity.",
            "gap_dates is an absolute count and therefore tracks how many tours the country "
            "gets; gap_dates_per_1k_country_tours normalises it so markets compare across "
            "borders.",
            f"Peer comparison and the catchment floor both use {CATCHMENT_COL}: residents for "
            f"whom this is the NEAREST market hosting shows. The plain radius counted a town "
            f"beside a big city as owning that city's population -- Guildford scored 13.8m "
            f"because London is 40 km away -- which made satellites look badly under-served. "
            f"catchment_kept shows how much of the raw radius survives; below about 0.3 the "
            f"market is a satellite, not a market.",
            f"Assessable also requires a catchment of at least {MIN_CATCHMENT_60KM:,} within "
            f"{PEER_RADIUS_KM} km. Without it the ranking fills with festival villages whose "
            f"biggest indoor room seats a few hundred, so every touring act registers as "
            f"needing something larger.",
            "ceiling_below_peers is a comparison, not a forecast: it says a market is "
            "under-built relative to others of its size, not that filling the gap would pay.",
        ]}).to_excel(xw, sheet_name="Methodology", index=False)

    log(f"wrote {path}")
    shown = out if a.all_markets else out[out["assessable"]]
    log(f"   {int(out['assessable'].sum()):,} of {len(out):,} markets are assessable "
        f"(>= {MIN_INDOOR_EVENTS} indoor shows and a known indoor ceiling)")
    show = ["market", "country", CATCHMENT_COL, "catchment_kept", "events",
            "indoor_ceiling", "peer_median_ceiling", "ceiling_below_peers",
            "suggested_capacity", "gap_tours", "gap_dates", "vs_peers"]
    print(f"\nTop {a.limit} by {a.sort}.")
    print("ceiling_below_peers: how many seats smaller this market's biggest indoor room is "
          "than the median\n  for markets of similar catchment. vs_peers above 1.0 means it "
          "already does more shows per head\n  than those comparators, which argues against a "
          "capacity constraint.\n")
    print(shown.head(a.limit)[show].to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
