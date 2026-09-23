#!/usr/bin/env python3
"""
What a show grosses, and how well a room sells — built from Pollstar.

WHAT THIS IS FOR

Layer 2 says how many more tour-visits a room would get. That is a count, and
nobody commissions a building on a count. This turns counts into money, using
825,000 real events rather than an assumption about ticket prices.

    python boxoffice.py --build     # read the main database once, cache a
                                    # small reference file
    python boxoffice.py             # show the curve and the spread

WHY IT IS A CACHED REFERENCE RATHER THAN A LIVE QUERY

setlistfm.db is 7.5 GB and is usually being written to by the scraper or the
venue enrichment. The app must never wait on it, and must never hold a
transaction open against it. So this reads it once, reduces it to a few
kilobytes of coefficients and quantiles, and writes that to models/. Everything
downstream uses the cache.

THE TWO THINGS IT MEASURES

1. GROSS PER EVENT AGAINST CAPACITY. A log-log fit on 825,326 events, which is
   tight: r = 0.89, so capacity alone explains about 80% of the variance in
   what a show takes. Doubling a room's capacity multiplies gross by roughly
   2^beta, and beta is estimated here rather than assumed.

2. HOW WELL A PARTICULAR ROOM SELLS FOR ITS SIZE. Each venue's average
   sell-through minus what its capacity band typically does. This is a real
   and persistent property of a room, not noise: splitting each venue's events
   into two halves and correlating the two residuals gives 0.861 across 3,545
   venues. A room that undersells its size one year undersells it the next.

WHAT THE SECOND NUMBER IS NOT

It is tempting to call it venue quality. It is not. A room's sell-through
reflects the building AND which acts get booked into it AND how well the local
promoter matches act to room. A hall that takes the right acts at the right
size sells out; the same hall taking acts too big for the town does not. The
measure cannot separate those, so it is named `sell-through performance`
throughout and never `quality`.

THE LIMIT THAT MATTERS MOST

This produces GROSS BOX OFFICE -- what the audience pays. It is not what the
venue earns. A venue takes a hire fee plus its share of ancillaries, which is a
fraction of the gross and varies by deal. `ref_hospitality` in the main
database has 32 rows, which is nowhere near enough to model that, so this file
does not try. Every figure it returns is labelled gross, and anyone converting
it to venue revenue has to supply their own rate card.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
REFERENCE = os.path.join(MODEL_DIR, "boxoffice.json")
MAIN_DB = os.path.join(os.path.dirname(HERE), "setlistfm.db")

# Rooms below this are not part of the touring economics this project is about,
# and they dominate the row count badly enough to bend the fit if left in.
MIN_CAPACITY = 500

# A venue needs this many events before its sell-through residual is a
# property of the room rather than of a couple of nights.
MIN_EVENTS_FOR_PERFORMANCE = 20

# Capacity bands for the readable summary. The fit itself is continuous; these
# exist so a reader can check the model against a median they can see.
BANDS = [500, 1500, 3500, 8000, 15000, 30000, 200000]
BAND_LABELS = ["500-1.5k", "1.5-3.5k", "3.5-8k", "8-15k", "15-30k", "30k+"]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Building the reference
# ---------------------------------------------------------------------------

def _load_events(db=MAIN_DB):
    """Every Pollstar event with a usable capacity and gross, read-only."""
    import sqlite3
    if not os.path.exists(db):
        raise SystemExit(f"main database not found at {db}")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=120)
    try:
        return pd.read_sql(
            """SELECT venue_norm, avg_capacity AS capacity,
                      avg_gross_usd AS gross, avg_tickets AS sold,
                      price_avg, avg_capacity_sold AS pct
               FROM pollstar_events
               WHERE avg_capacity >= ? AND avg_gross_usd > 0""",
            con, params=[MIN_CAPACITY])
    finally:
        con.close()


def _fit_gross(d):
    """
    log(gross) = a + b * log(capacity), by least squares.

    Written out rather than imported for the same reason the logit is: three
    lines of NumPy that a reader can check beats a library call they cannot.
    `sigma` is the residual standard deviation in logs, which is what makes the
    spread around the line reportable rather than merely acknowledged -- a
    single show in a 10,000 room does not take the median, it takes something
    within a factor of exp(1.96 * sigma) of it.
    """
    x = np.log(d["capacity"].values)
    y = np.log(d["gross"].values)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return {"intercept": float(coef[0]), "slope": float(coef[1]),
            "sigma": float(resid.std(ddof=2)),
            "r": float(np.corrcoef(x, y)[0, 1]), "n": int(len(d))}


def _sellthrough_residuals(d):
    """
    Each venue's sell-through minus what rooms of its size typically manage.

    Expected sell-through comes from twenty capacity quantiles rather than a
    fitted curve, because the relationship is not monotonic -- very large rooms
    sell through better than mid-sized ones, since only acts that can fill them
    ever book them. That is selection, and a smooth fit would launder it into
    an apparent law that bigger rooms sell better.
    """
    d = d[d["pct"].between(0, 100, inclusive="right")].copy()
    if d.empty:
        return pd.DataFrame(), {}
    bins = pd.qcut(np.log(d["capacity"]), 20, duplicates="drop")
    d["expected_pct"] = d.groupby(bins, observed=True)["pct"].transform("median")
    d["resid"] = d["pct"] - d["expected_pct"]

    v = (d.groupby("venue_norm")
         .agg(events=("resid", "size"), resid=("resid", "mean"),
              pct=("pct", "mean"), capacity=("capacity", "median")))
    v = v[v["events"] >= MIN_EVENTS_FOR_PERFORMANCE]

    # Split-half reliability: is the residual a property of the room, or noise?
    # Reported with the reference so nobody has to take persistence on trust.
    d["_half"] = d.groupby("venue_norm").cumcount() % 2
    h = (d[d["venue_norm"].isin(v.index)]
         .pivot_table(index="venue_norm", columns="_half", values="resid",
                      aggfunc="mean").dropna())
    reliability = float(h[0].corr(h[1])) if len(h) > 30 else float("nan")

    q = v["resid"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2)
    return v, {"reliability": reliability, "venues": int(len(v)),
               "percentiles": {str(k): float(val) for k, val in q.items()}}


def build(db=MAIN_DB, out=REFERENCE):
    log("reading pollstar events (read-only; safe while the scraper runs) ...")
    d = _load_events(db)
    log(f"   {len(d):,} events at {MIN_CAPACITY:,}+ capacity")

    fit = _fit_gross(d)
    log(f"   log(gross) ~ log(capacity): slope {fit['slope']:.3f}, "
        f"r {fit['r']:.3f}")

    v, perf = _sellthrough_residuals(d)
    log(f"   sell-through performance for {perf.get('venues', 0):,} venues, "
        f"split-half reliability {perf.get('reliability', float('nan')):.3f}")

    d["band"] = pd.cut(d["capacity"], BANDS, labels=BAND_LABELS)
    bands = (d.groupby("band", observed=True)
             .agg(events=("gross", "size"),
                  median_gross=("gross", "median"),
                  p25_gross=("gross", lambda s: s.quantile(0.25)),
                  p75_gross=("gross", lambda s: s.quantile(0.75)),
                  median_price=("price_avg", "median"),
                  median_sold=("sold", "median"),
                  median_pct=("pct", "median"))
             .round(1).reset_index())
    bands["band"] = bands["band"].astype(str)

    os.makedirs(MODEL_DIR, exist_ok=True)
    blob = {
        "built_from": os.path.basename(db),
        "min_capacity": MIN_CAPACITY,
        "gross_fit": fit,
        "performance": perf,
        "bands": bands.to_dict("records"),
        # Only venues with enough events to be worth carrying; a few thousand
        # rows keeps the file small enough to load instantly.
        "venue_performance": {
            str(k): round(float(val), 2) for k, val in v["resid"].items()},
        "venue_events": {str(k): int(val) for k, val in v["events"].items()},
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1)
    log(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")
    return blob


def load(path=REFERENCE):
    """The cached reference, or (None, why not)."""
    if not os.path.exists(path):
        return None, ("No box-office reference yet. Run "
                      "`python boxoffice.py --build` once; it reads the main "
                      "database read-only and takes a couple of minutes.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh), None


# ---------------------------------------------------------------------------
# Using it
# ---------------------------------------------------------------------------

def gross_per_event(ref, capacity, performance_pp=0.0):
    """
    Expected gross for one show in a room of this size, with a spread.

    `performance_pp` shifts the answer by a room's sell-through residual in
    percentage points. The conversion is deliberately simple and deliberately
    conservative: gross is assumed proportional to tickets sold, so a room
    selling 10 points above par for its size takes proportionally more. It does
    NOT also assume such a room charges more, although better-selling rooms
    plainly do, because that would be compounding one advantage into two on no
    evidence.

    Returns median and the 25th/75th percentiles implied by the residual
    spread. The interval is wide on purpose: a single show in a 10,000-seat
    room really does take anywhere from a quarter to three times the median,
    depending entirely on who is playing.
    """
    f = ref["gross_fit"]
    base = np.exp(f["intercept"] + f["slope"] * np.log(max(float(capacity), 1)))
    # a sell-through residual of +10pp against a typical ~80% is +12.5% of gross
    uplift = 1.0 + (performance_pp / 100.0) / 0.80
    med = base * max(uplift, 0.1)
    s = f["sigma"]
    return {"median": float(med),
            "p25": float(med * np.exp(-0.6745 * s)),
            "p75": float(med * np.exp(0.6745 * s)),
            "sigma_log": s}


# ---------------------------------------------------------------------------
# A room's own sell-through, taken from the extract rather than by name
# ---------------------------------------------------------------------------
#
# WHY THIS DOES NOT JOIN TO POLLSTAR BY NAME.
#
# It was written that way first and it was quietly broken. Pollstar's own
# `venue_norm` maps "The O2 - London" to `o2 london`, while the extract calls
# the same building "The O2 Arena". Lower-casing one to match the other finds
# nothing, and the failure is silent: every lookup returns None and the screen
# reports "too few events to judge" for venues with hundreds of shows.
#
# The extract already carries `avg_capacity_pct` per venue, matched to Pollstar
# by the main pipeline's own fuzzy matcher. That link is better than anything
# re-derived here and it is keyed by venue_uid, so it cannot mismatch. The
# pooled gross curve above still comes straight from Pollstar, because a curve
# over all events needs no venue join at all.

def performance_table(ex, min_events=5):
    """
    Each venue's sell-through against what rooms of its size manage.

    `min_events` is lower than the Pollstar-side threshold because the
    extract's figure is already an average over a venue's shows rather than one
    night, so it is steadier per row.
    """
    v = ex["venues"].copy()
    v["capacity"] = pd.to_numeric(v["capacity"], errors="coerce")
    v["pct"] = pd.to_numeric(v["avg_capacity_pct"], errors="coerce")
    v = v[(v["capacity"] >= MIN_CAPACITY) & v["pct"].between(0, 100)
          & (pd.to_numeric(v["events"], errors="coerce") >= min_events)].copy()
    if v.empty:
        return v.assign(expected_pct=[], residual_pp=[], percentile=[])

    # Expected sell-through from capacity quantiles, not a fitted curve: the
    # relationship is not monotonic, because very large rooms sell through
    # better than mid-sized ones purely by selection.
    n_bins = min(10, max(2, len(v) // 40))
    bins = pd.qcut(np.log(v["capacity"]), n_bins, duplicates="drop")
    v["expected_pct"] = v.groupby(bins, observed=True)["pct"].transform("median")
    v["residual_pp"] = (v["pct"] - v["expected_pct"]).round(2)
    v["percentile"] = v["residual_pp"].rank(pct=True).round(3)
    return v


BAND_NAMES = ["bottom 10%", "bottom quarter", "below average",
              "above average", "top quarter", "top 10%"]


def venue_performance(perf_table, venue_name):
    """
    A room's sell-through residual in percentage points, and where it ranks.

    Returns None where the venue is not in the table -- no capacity, no
    sell-through, or too few shows. Most rooms fall out, and saying nothing is
    the right answer for them.
    """
    if perf_table is None or perf_table.empty:
        return None
    hit = perf_table[perf_table["venue"].astype(str).str.casefold()
                     == str(venue_name).strip().casefold()]
    if hit.empty:
        return None
    r = hit.iloc[0]
    q = float(r["percentile"])
    place = sum(q >= t for t in (0.10, 0.25, 0.50, 0.75, 0.90))
    return {"residual_pp": float(r["residual_pp"]),
            "band": BAND_NAMES[min(place, 5)],
            "percentile": q,
            "sell_through": float(r["pct"]),
            "typical_for_size": float(r["expected_pct"]),
            "events": int(r["events"])}


def performance_targets(perf_table):
    """The residual at each percentile, so a refit has something to aim at."""
    if perf_table is None or perf_table.empty:
        return {}
    q = perf_table["residual_pp"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return {str(k): round(float(val), 2) for k, val in q.items()}


def project(ref, visits_low, visits_high, capacity, performance_pp=0.0,
            years=None):
    """
    Gross box office for a range of extra tour-visits.

    Takes the tour-count interval from Layer 2 rather than a point, because
    that interval is the dominant source of uncertainty and collapsing it here
    would hide the fact. See `decompose()`.
    """
    g = gross_per_event(ref, capacity, performance_pp)
    out = {
        "capacity": float(capacity),
        "gross_per_event": g,
        "low": visits_low * g["p25"],
        "central_low": visits_low * g["median"],
        "central_high": visits_high * g["median"],
        "high": visits_high * g["p75"],
        "visits_low": visits_low, "visits_high": visits_high,
    }
    if years:
        out["per_year_low"] = out["central_low"] / years
        out["per_year_high"] = out["central_high"] / years
    return out


def decompose(ref, visits_low, visits_high, capacity, performance_pp=0.0):
    """
    WHICH INPUT MAKES THE ANSWER UNCERTAIN -- the point of the whole screen.

    A revenue range that spans a factor of four is not useful on its own. What
    is useful is knowing which ingredient produced the four, because that says
    where to spend the next month of work.

    Both sources are expressed the same way, as the ratio between the high and
    low ends of what each contributes on its own, so they can be compared:

      TOUR COUNT      how many extra visits the room gets. This is the
                      specification A to specification B interval -- the
                      identification problem, the thing Layer 3 would narrow.
      GROSS PER SHOW  how much a single show in a room of this size takes.
                      Measured from 825,000 events, so this one is not going to
                      get better with more data; it is genuine variation
                      between acts, not ignorance.

    Almost always the first dominates, and that is the finding: better ticket
    price data would not help, and a natural experiment would.
    """
    g = gross_per_event(ref, capacity, performance_pp)
    span_visits = (visits_high / visits_low) if visits_low > 0 else np.inf
    span_gross = g["p75"] / g["p25"] if g["p25"] > 0 else np.inf

    # Share of the total log-width each contributes.
    lv = np.log(span_visits) if np.isfinite(span_visits) else np.nan
    lg = np.log(span_gross)
    total = lv + lg
    rows = [
        {"source": "how many tours the room gets",
         "range": f"{visits_low:.1f} to {visits_high:.1f} visits",
         "spans a factor of": round(float(span_visits), 2),
         "share of the uncertainty": (round(float(lv / total), 3)
                                      if total else np.nan),
         "can more data fix it?":
             "No. This is specification A against B -- whether capacity causes "
             "shows or merely accompanies them. Only a natural experiment "
             "(Layer 3) narrows it."},
        {"source": "what one show grosses",
         "range": f"${g['p25']:,.0f} to ${g['p75']:,.0f} per show",
         "spans a factor of": round(float(span_gross), 2),
         "share of the uncertainty": (round(float(lg / total), 3)
                                      if total else np.nan),
         "can more data fix it?":
             "No, but it does not need fixing. It is real variation between "
             "acts measured on 825,000 events, not ignorance about prices."},
    ]
    return pd.DataFrame(rows)


def band_table(ref):
    """The readable check: medians by capacity band, straight from the data."""
    return pd.DataFrame(ref["bands"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true",
                    help="read the main database and rebuild the reference")
    ap.add_argument("--db", default=MAIN_DB)
    a = ap.parse_args()

    if a.build:
        ref = build(a.db)
    else:
        ref, err = load()
        if err:
            raise SystemExit(err)

    f = ref["gross_fit"]
    print(f"\nGross per event, fitted on {f['n']:,} events")
    print(f"  log(gross) = {f['intercept']:.3f} + {f['slope']:.3f} x log(capacity)")
    print(f"  r = {f['r']:.3f}   residual sd (logs) = {f['sigma']:.3f}")
    print(f"  so doubling capacity multiplies gross by "
          f"{2 ** f['slope']:.2f}, and a single show lands within a factor of "
          f"{np.exp(1.96 * f['sigma']):.1f} of the median 95% of the time.")

    p = ref["performance"]
    print(f"\nSell-through performance, {p['venues']:,} venues with "
          f"{MIN_EVENTS_FOR_PERFORMANCE}+ events")
    print(f"  split-half reliability {p['reliability']:.3f} -- a real, "
          f"persistent property of a room, not noise")
    print(f"  percentiles (points above/below par for the size): "
          f"{p['percentiles']}")

    print("\nMedians by capacity band, straight from the data:")
    print(band_table(ref).to_string(index=False))

    print("\nWorked example: a 13,500-seat room gaining 7.1 to 26.5 visits")
    pr = project(ref, 7.1, 26.5, 13_500, years=3.7)
    print(f"  gross box office ${pr['central_low']:,.0f} to "
          f"${pr['central_high']:,.0f} over 3.7 years")
    print(f"  (${pr['per_year_low']:,.0f} to ${pr['per_year_high']:,.0f} a year)")
    print("\nWhere that uncertainty comes from:")
    print(decompose(ref, 7.1, 26.5, 13_500).to_string(index=False))
    print("\nNOTE: gross box office, not venue revenue. The venue takes a hire "
          "fee plus ancillaries,\nwhich this database cannot support "
          "(ref_hospitality has 32 rows).")


if __name__ == "__main__":
    main()
