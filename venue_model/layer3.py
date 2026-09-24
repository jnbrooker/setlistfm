#!/usr/bin/env python3
"""
Layer 3 — what actually happened when a room opened.

THE QUESTION THIS FINALLY ASKS PROPERLY

Layer 2's answer to "would a room bring more shows" is an interval spanning a
factor of nearly four, because capacity and unobserved demand cannot be told
apart in a cross-section. Cities build big rooms where promoters already
expected demand.

The way out is to stop comparing cities with each other and start comparing a
city with its own past. A room opens in 2018; what happened to that city
relative to cities where nothing opened? The selection story has no purchase on
that comparison, because whatever made the city attractive was already there in
2017.

    python layer3.py --min-capacity 8000
    python layer3.py --min-capacity 8000 --log --bootstrap 200

WHAT THE TREATMENT IS, AND WHY IT IS NOT WIKIDATA

The obvious treatment is a venue's opening year. Wikidata supplies one for 137
of 886 enriched venues, and within any usable window that is **one opening in
2022, one in 2019, two in 2016**. Four treated cities is not a research design.

So treatment is inferred instead: a market is treated in the first year a room
of at least `min_capacity` appears in it, having had none that size before. The
137 Wikidata dates are then spent on VALIDATING that inference rather than
driving it -- the same pattern as the right-sizing rule on the Venue tab, where
a rule is measured against what is independently known rather than asserted.

THE ASSUMPTION THAT INFERENCE MAKES, STATED PLAINLY

"First appears in the data" is not "opened". A room can exist for years without
hosting an act this database tracks. Three guards reduce that, and none removes
it:

  * the market must already be ACTIVE before the opening year, so a market
    simply entering the dataset is not mistaken for one gaining a room;
  * a market whose first big room predates the panel is dropped, not treated,
    because it has no clean pre-period;
  * the first `BUFFER_YEARS` of the panel cannot be treatment years, since
    anything appearing then is indistinguishable from the window opening.

WHY 2020 AND 2021 ARE DROPPED

Every venue on earth shut. A shock that hits treated and control cities
identically and totally does not identify anything, and leaving it in would put
a crater in the middle of every event study that has nothing to do with rooms.
Dropping them costs two years of a fifteen-year panel and removes a confound
larger than the effect being measured.
"""

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import did                                                       # noqa: E402
from gap import latest_extract, load_extract                     # noqa: E402

# The pandemic. Not a data gap -- a real collapse that hit every city at once.
DROP_YEARS = (2020, 2021)

# A market cannot be treated in the first years of the panel: a room appearing
# then is indistinguishable from the observation window opening.
BUFFER_YEARS = 2

# A market must have hosted at least this many tour-visits in a year to count
# as active. Below it, "gained a room" cannot be separated from "entered the
# dataset".
ACTIVE_VISITS = 3

# Below this many treated markets there is no design, only arithmetic. The
# figure is a judgement rather than a rule: with a dozen treated cities spread
# over several cohorts an event study has something to average, with five it is
# reporting one or two cities and calling it causal. Printing an estimate
# anyway would be the easiest way for this project to mislead, so the threshold
# is enforced rather than noted.
MIN_TREATED_FOR_A_DESIGN = 12


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

def build_panel(ex, drop_years=DROP_YEARS):
    """
    Tour-visits per market per year, balanced with explicit zeros.

    A market-year with no visits is a zero, not a missing value, and the
    difference is not pedantic: a DiD differences Y_t against Y_{g-1}, so a
    dropped zero silently removes the observation instead of contributing the
    fall it represents.
    """
    tc = ex["tour_city"].copy()
    city_to_market = ex["cities"].set_index(["countryCode", "city"])["market"]
    tc["market"] = pd.MultiIndex.from_arrays(
        [tc["countryCode"], tc["city"]]).map(city_to_market)
    tc = tc.dropna(subset=["market"])
    tc["year"] = pd.to_datetime(tc["first_event"], errors="coerce").dt.year
    tc = tc.dropna(subset=["year"])
    tc["year"] = tc["year"].astype(int)
    tc["unit"] = tc["countryCode"] + "|" + tc["market"]

    visits = (tc.groupby(["unit", "year"], as_index=False)
              .agg(visits=("tour", "nunique"), dates=("events", "sum")))

    years = [y for y in range(int(visits["year"].min()),
                              int(visits["year"].max()) + 1)
             if y not in set(drop_years)]
    units = sorted(visits["unit"].unique())
    grid = pd.MultiIndex.from_product([units, years],
                                      names=["unit", "year"]).to_frame(index=False)
    panel = grid.merge(visits, on=["unit", "year"], how="left")
    panel[["visits", "dates"]] = panel[["visits", "dates"]].fillna(0)
    panel["countryCode"] = panel["unit"].str.split("|").str[0]
    panel["market"] = panel["unit"].str.split("|").str[1]
    return panel


# ---------------------------------------------------------------------------
# Who was treated, and when
# ---------------------------------------------------------------------------

def find_openings(ex, panel, min_capacity=8000, buffer_years=BUFFER_YEARS,
                  active_visits=ACTIVE_VISITS):
    """
    The first year each market gained a room of at least `min_capacity`.

    Returns one row per market with its cohort year (0 for never treated) and,
    where it was excluded, the reason. The reasons are kept rather than
    filtered away because the count of usable treated markets IS the finding
    when it is small -- a design with four treated units should say so loudly
    rather than quietly produce an estimate.
    """
    v = ex["venues"].copy()
    v["capacity"] = pd.to_numeric(v["capacity"], errors="coerce")
    v["year"] = pd.to_datetime(v["first_event"], errors="coerce").dt.year
    v = v.dropna(subset=["capacity", "year"])
    v["year"] = v["year"].astype(int)

    city_to_market = ex["cities"].set_index(["countryCode", "city"])["market"]
    v["market"] = pd.MultiIndex.from_arrays(
        [v["countryCode"], v["city"]]).map(city_to_market)
    v = v.dropna(subset=["market"])
    v["unit"] = v["countryCode"] + "|" + v["market"]

    big = v[v["capacity"] >= min_capacity]
    first_big = big.groupby("unit")["year"].min()

    years = sorted(panel["year"].unique())
    panel_start, earliest_treatable = years[0], years[0] + buffer_years

    # when each market first looked active at all
    act = panel[panel["visits"] >= active_visits]
    first_active = act.groupby("unit")["year"].min()

    rows = []
    for unit in sorted(panel["unit"].unique()):
        g = int(first_big[unit]) if unit in first_big.index else 0
        reason = ""
        if g == 0:
            reason = "never gained a room this size"
        elif g < earliest_treatable:
            # Already had one when the window opened, or gained it so early
            # that "appeared" cannot be separated from "window started".
            reason = f"had one by {g}, before the panel can see a change"
            g = -1
        elif unit not in first_active.index or first_active[unit] >= g:
            # The market was not already active, so this is a market entering
            # the dataset rather than a market gaining a room.
            reason = "not active before the room appeared"
            g = -1
        rows.append({"unit": unit, "cohort": g, "reason": reason,
                     "first_big_room": int(first_big[unit])
                     if unit in first_big.index else None,
                     "first_active": int(first_active[unit])
                     if unit in first_active.index else None})
    out = pd.DataFrame(rows)
    out["countryCode"] = out["unit"].str.split("|").str[0]
    out["market"] = out["unit"].str.split("|").str[1]
    out.attrs["panel_start"] = panel_start
    out.attrs["earliest_treatable"] = earliest_treatable
    return out


def validate_openings(ex, openings, min_capacity=8000):
    """
    Check the inferred opening years against the ones Wikidata actually knows.

    The inference is the weakest link in the design, so it is measured rather
    than trusted. For venues carrying a `opened_year` from enrichment, compare
    it with the year the venue first appears in the data.

    A venue appearing in the same year it opened, or shortly after, is the
    inference working. A venue that opened decades before it first appears is
    the inference failing -- and for a room in a market this project cares
    about, failing in the direction that invents a treatment.
    """
    v = ex["venues"].copy()
    v["capacity"] = pd.to_numeric(v["capacity"], errors="coerce")
    v["opened_year"] = pd.to_numeric(v.get("opened_year"), errors="coerce")
    v["first_seen"] = pd.to_datetime(v["first_event"], errors="coerce").dt.year
    d = v.dropna(subset=["opened_year", "first_seen", "capacity"])
    d = d[(d["capacity"] >= min_capacity)
          & d["opened_year"].between(1900, dt.date.today().year)]
    if d.empty:
        return {"checked": 0, "detail": pd.DataFrame()}
    d = d.copy()
    d["gap"] = d["first_seen"] - d["opened_year"]
    return {
        "checked": int(len(d)),
        "median_gap": float(d["gap"].median()),
        "within_1yr": float((d["gap"].abs() <= 1).mean()),
        "within_3yr": float((d["gap"].abs() <= 3).mean()),
        "opened_before_panel": float((d["gap"] > 5).mean()),
        "detail": d[["venue", "city", "capacity", "opened_year",
                     "first_seen", "gap"]].sort_values("gap"),
    }


def assemble(ex, min_capacity=8000, outcome="visits", use_log=False,
             drop_years=DROP_YEARS):
    """Panel plus cohorts, ready for the estimator, with the excluded removed."""
    panel = build_panel(ex, drop_years)
    op = find_openings(ex, panel, min_capacity)
    panel = panel.merge(op[["unit", "cohort"]], on="unit", how="left")
    # cohort -1 means "cannot be identified"; those units are dropped entirely
    # rather than folded into the controls, where they would contaminate them.
    dropped = panel[panel["cohort"] == -1]["unit"].nunique()
    panel = panel[panel["cohort"] != -1].copy()
    panel["y"] = (np.log1p(panel[outcome]) if use_log else panel[outcome])
    panel.attrs["dropped_units"] = dropped
    return panel, op


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", default=None)
    ap.add_argument("--min-capacity", type=int, default=8000,
                    help="a room this size or larger counts as an opening")
    ap.add_argument("--outcome", default="visits", choices=["visits", "dates"])
    ap.add_argument("--log", action="store_true",
                    help="log1p the outcome, so the effect reads as a "
                         "proportional change rather than a count. Levels are "
                         "the default because 'extra tour-visits' matches the "
                         "rest of the project, but they let big cities "
                         "dominate the average.")
    ap.add_argument("--control", default="notyet", choices=["notyet", "never"])
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="resample cities this many times for an interval")
    a = ap.parse_args()

    ex = load_extract(a.extract or latest_extract())
    lo, hi = ex["manifest"].get("event_date_range", ("?", "?"))
    log(f"extract covers {lo} to {hi}")

    panel, op = assemble(ex, a.min_capacity, a.outcome, a.log)
    years = sorted(panel["year"].unique())
    treated = op[op["cohort"] > 0]

    print(f"\nPANEL   {panel['unit'].nunique():,} markets x "
          f"{len(years)} years ({years[0]}-{years[-1]}, "
          f"{', '.join(str(y) for y in DROP_YEARS)} dropped)")
    print(f"        {panel.attrs['dropped_units']:,} markets dropped as "
          f"unidentifiable")
    print(f"\nTREATMENT  a room of {a.min_capacity:,}+ appearing where there "
          f"was none")
    print(f"        {len(treated):,} treated markets, "
          f"{int((op['cohort'] == 0).sum()):,} never-treated controls")
    if len(treated):
        print("        cohorts: " + ", ".join(
            f"{int(y)}:{n}" for y, n in
            treated["cohort"].value_counts().sort_index().items()))
    for reason, n in op[op["cohort"] == -1]["reason"].value_counts().items():
        print(f"        excluded, {reason}: {n}")

    val = validate_openings(ex, op, a.min_capacity)
    if val["checked"]:
        print(f"\nHOW GOOD IS THE INFERENCE?  checked against "
              f"{val['checked']} venues with a known opening year")
        print(f"        first seen within 1 year of opening: "
              f"{val['within_1yr']:.0%}   within 3: {val['within_3yr']:.0%}")
        print(f"        median gap {val['median_gap']:+.0f} years; "
              f"{val['opened_before_panel']:.0%} opened more than 5 years "
              f"before they first appear")
    else:
        print("\nHOW GOOD IS THE INFERENCE?  no venue of this size has a known "
              "opening year, so it cannot be checked at all.")

    if len(treated) < MIN_TREATED_FOR_A_DESIGN:
        print(f"\n!! {len(treated)} treated markets is not a design "
              f"({MIN_TREATED_FOR_A_DESIGN} is the minimum this will estimate "
              f"on). Almost every market with a room this size already had one "
              f"when the window opened, so there is no before to compare "
              f"against.")
        print(f"   The panel spans {years[0]}-{years[-1]}. Rebuild it long "
              f"enough to contain some openings:")
        print("       python extract.py --since 2012 --tag since2012")
        print("       python layer2.py --save --extract extracts/since2012")
        print(f"       python layer3.py --extract extracts/since2012 "
              f"--min-capacity {a.min_capacity}")
        print("   No estimate is printed, deliberately: a number from five "
              "cities would be read as a finding.")
        return

    gt = did.att_gt(panel, "unit", "year", "y", "cohort", a.control)
    es = did.event_study(gt)
    pre = did.pretrend_test(gt)
    unit = "log visits" if a.log else f"{a.outcome} a year"

    print(f"\nEVENT STUDY  effect in {unit}")
    print(es.to_string(index=False))
    print(f"\nOVERALL post-opening effect  {did.overall_att(gt):+.2f} {unit}")
    print(f"PRE-TREND over {pre['leads_tested']} lead years  "
          f"{pre['mean_pre_effect']:+.2f} (size {pre['size']:.2f})")
    if pre["size"] > 0.5 * abs(did.overall_att(gt) or 1):
        print("        !! the pre-trend is a large fraction of the effect. "
              "Treated cities were already moving before their room opened, "
              "which is the selection story this design exists to rule out. "
              "Do not read the effect as causal.")

    dt = did.detrend(gt)
    if dt.get("ok"):
        print("\nPRE-TREND SENSITIVITY")
        print(f"        treated cities were already gaining "
              f"{dt['slope_per_year']:+.2f} {unit} before their room opened")
        print(f"        raw post-opening effect        {dt['raw_att']:+.2f}")
        print(f"        with that trend removed        {dt['detrended_att']:+.2f}")
        print("        The second figure extrapolates the pre-trend across "
              "the post period and subtracts it. That is a sensitivity, not "
              "a correction: it assumes the trend would have continued, "
              "which is as unverifiable as the parallel-trends assumption it "
              "patches. Read the two as a range.")

    if a.bootstrap:
        log(f"bootstrapping {a.bootstrap} resamples of the city list ...")
        b = did.bootstrap(panel, a.bootstrap, unit="unit", time="year",
                          outcome="y", cohort="cohort", control=a.control)
        print(f"\n90% interval on the overall effect  "
              f"{b['overall_ci'][0]:+.2f} to {b['overall_ci'][1]:+.2f}")


if __name__ == "__main__":
    main()
