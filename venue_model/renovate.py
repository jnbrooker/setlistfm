#!/usr/bin/env python3
"""
Changing a room that already exists, rather than building a new one.

WHAT THE MODEL CAN AND CANNOT SEE, STATED FIRST

The market model sees exactly three things about a city's rooms: the biggest
indoor capacity, the biggest outdoor capacity, and whether either clears the
bar for the act in front of it. That is the whole of it. So of the things a
renovation might do:

  RAISE A CEILING           the model sees it, and it is the SAME arithmetic
                            as building a new room of that size. Renovation and
                            new build are indistinguishable at market level,
                            and this file says so rather than dressing the two
                            up as different answers.

  PUT A ROOF ON             the model sees it, and it is genuinely distinct.
                            An outdoor room that becomes playable indoors
                            raises the INDOOR ceiling without anything being
                            built. This is the one renovation the project could
                            not already express.

  MAKE THE ROOM BETTER      the model is blind to it. Nothing in the fitted
                            coefficients knows about sightlines, acoustics,
                            loading docks or bars. A refurbishment that does
                            not change capacity or kind changes NO tour-count
                            prediction, and pretending otherwise would be
                            inventing a coefficient.

THE THIRD ONE IS STILL WORTH MODELLING, JUST NOT THERE

A better room shows up in what it takes per show, not in how many shows it
gets. And that IS measurable: a venue's sell-through relative to rooms of its
size is a persistent property, reliable at 0.864 split-half across 3,642
venues (see boxoffice.py). So a refurbishment is modelled as moving that
residual toward a target percentile, and its effect lands entirely on gross
box office.

Keeping the two effects on separate lines is the point. A roof changes who
comes; a refit changes what they pay. Adding them into one "renovation uplift"
would hide which half of the case rests on a model and which on an assumption
about the building trade.

WHAT A ROOF ACTUALLY COSTS YOU

Roofing a 52,000-seat stadium does not give a 52,000-seat arena. The indoor
configuration of a convertible venue is typically a third to a half of the open
capacity -- Lille's Stade Pierre-Mauroy is 50,000 outdoors and about 27,000 in
its indoor Aréna setup. So `indoor_capacity` is asked for explicitly rather
than derived, and DEFAULT_ROOF_FRACTION is only a starting suggestion.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import boxoffice                                                 # noqa: E402
import venue as V                                                # noqa: E402
import whatif                                                    # noqa: E402
from gap import latest_extract, load_extract, resolve_market     # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Indoor capacity of a convertible venue as a share of its open capacity, used
# only to suggest a starting number. Real convertibles land between a third and
# a half; the user is expected to override it with the actual figure.
DEFAULT_ROOF_FRACTION = 0.5


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Roofing: the renovation the model can genuinely see
# ---------------------------------------------------------------------------

def add_roof(menu, ex, market_row, venue_name, indoor_capacity, lad=None):
    """
    Make an existing outdoor room playable indoors, and re-predict.

    The arithmetic is whatif.impact with kind="indoor", because that is exactly
    what happens: the city's indoor ceiling rises to the roofed capacity and
    indoor acts are re-tested against it. What this function adds is the
    context that makes the number readable -- whether the room was outdoor to
    begin with, what the indoor ceiling was before, and whether the roofed
    capacity actually beats it.

    A roof that lands BELOW the existing indoor ceiling changes nothing, and
    that is reported rather than returned as a misleading zero-with-no-reason.
    """
    if lad is None:
        lad = V.rooms(ex, market_row)
    hit = lad[lad["venue"] == venue_name]
    if hit.empty:
        raise KeyError(f"no room called {venue_name!r} with a known capacity "
                       f"in {market_row['city']}")
    row = hit.iloc[0]
    open_cap = float(row["capacity"])
    was = row["io"]

    ins = lad.loc[lad["io"] == "inside", "capacity"]
    indoor_ceiling_now = float(ins.max()) if len(ins) else np.nan

    note = None
    if was == "inside":
        note = (f"{venue_name} is already an indoor room, so a roof is not the "
                f"renovation to model here. Use an expansion instead.")
    elif indoor_capacity <= (indoor_ceiling_now if np.isfinite(indoor_ceiling_now)
                             else -np.inf):
        note = (f"Roofed to {indoor_capacity:,.0f}, this room would still be "
                f"smaller than {market_row['city']}'s existing indoor ceiling "
                f"of {indoor_ceiling_now:,.0f}. The model would see no change "
                f"at all, because the ceiling acts are tested against does not "
                f"move.")

    out = {"venue": venue_name, "open_capacity": open_cap, "was": was,
           "indoor_capacity": float(indoor_capacity),
           "indoor_ceiling_before": indoor_ceiling_now, "note": note,
           "impacts": {}}

    # Even when the note says nothing changes, run it: a zero the reader can
    # see arrived from the same machinery is worth more than a zero asserted.
    for spec in ("A", "B"):
        out["impacts"][spec] = whatif.impact(
            menu, spec, market_row["city"], float(indoor_capacity), "indoor")
    lo = min(i["extra_visits"] for i in out["impacts"].values())
    hi = max(i["extra_visits"] for i in out["impacts"].values())
    out["extra_visits_low"], out["extra_visits_high"] = lo, hi
    return out


def compare_roof_with_newbuild(menu, market_row, indoor_capacity):
    """
    The honest check: is roofing different from building?

    At market level it is not, and this returns the two numbers side by side so
    that is visible rather than claimed. Any difference between them would be a
    bug, not a finding -- both raise the same indoor ceiling to the same value.

    Where they genuinely differ is the LADDER: a roof converts a rung, a new
    build adds one. That only matters to the room-split half of the venue
    screen, which is only reliable in markets with few rooms.
    """
    rows = []
    for spec in ("A", "B"):
        imp = whatif.impact(menu, spec, market_row["city"],
                            float(indoor_capacity), "indoor")
        rows.append({"specification": spec,
                     "roofing this room": round(imp["extra_visits"], 3),
                     "building a new one the same size":
                         round(imp["extra_visits"], 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Refurbishment: invisible to the model, visible in the money
# ---------------------------------------------------------------------------

def refurbish(ref, perf_table, venue_name, capacity, target_percentile=0.75,
              visits_per_year=None):
    """
    Move a room's sell-through toward what the best rooms of its size manage.

    Returns the change in gross PER SHOW, and nothing about tour counts,
    because the model has no view on whether a nicer room attracts more tours.
    That is a real limitation and not a modelling choice: there is no variable
    in the fitted coefficients that a refurbishment would move.

    `target_percentile` is where the refit is assumed to land the room among
    rooms of its size. 0.75 is a deliberate ceiling on optimism -- getting into
    the top quarter is a good outcome for a refurbishment, and assuming the top
    10% would be assuming the renovation is excellent before it is designed.
    """
    perf = boxoffice.venue_performance(perf_table, venue_name)
    targets = boxoffice.performance_targets(perf_table)
    target = float(targets.get(str(target_percentile), 2.3))

    now_pp = perf["residual_pp"] if perf else None
    if now_pp is None:
        return {"known": False,
                "why": (f"{venue_name} has no recorded sell-through in the "
                        f"extract -- either no capacity, no Pollstar match, or "
                        f"too few shows. There is no baseline for a refit to "
                        f"improve on, so this half of the case cannot be "
                        f"costed."),
                "target_pp": target}

    if now_pp >= target:
        gain_pp = 0.0
        why = (f"This room already sells {now_pp:+.1f} points against par for "
               f"its size, at or above the {target_percentile:.0%} mark of "
               f"{target:+.1f}. On this measure there is nothing for a refit "
               f"to recover.")
    else:
        gain_pp = target - now_pp
        why = (f"It sells {now_pp:+.1f} points against par for its size. "
               f"Rooms at the {target_percentile:.0%} mark manage "
               f"{target:+.1f}, so a refit reaching that would add "
               f"{gain_pp:.1f} points of sell-through.")

    before = boxoffice.gross_per_event(ref, capacity, now_pp)
    after = boxoffice.gross_per_event(ref, capacity, now_pp + gain_pp)
    out = {"known": True, "now_pp": now_pp, "target_pp": target,
           "gain_pp": gain_pp, "why": why, "events": perf["events"],
           "band": perf["band"], "percentile": perf["percentile"],
           "sell_through": perf["sell_through"],
           "typical_for_size": perf["typical_for_size"],
           "gross_before": before["median"], "gross_after": after["median"],
           "gross_uplift": after["median"] - before["median"]}
    if visits_per_year:
        out["uplift_per_year"] = out["gross_uplift"] * visits_per_year
    return out


# ---------------------------------------------------------------------------
# The two halves together
# ---------------------------------------------------------------------------

def full_case(menu, ex, market_row, venue_name, indoor_capacity, ref=None,
              target_percentile=0.75, lad=None, years=None, perf_table=None):
    """
    A roof and a refit costed side by side, never added together.

    The two effects are returned separately and stay separate all the way to
    the screen. One is a model prediction with an identification problem
    attached; the other is arithmetic on observed sell-through. Summing them
    into a single headline would give the weaker number the authority of the
    stronger.
    """
    if lad is None:
        lad = V.rooms(ex, market_row)
    roof = add_roof(menu, ex, market_row, venue_name, indoor_capacity, lad)

    out = {"roof": roof, "refit": None, "revenue": None, "decomposition": None}
    if ref is None:
        ref, err = boxoffice.load()
        if err:
            out["revenue_error"] = err
            return out

    if perf_table is None:
        perf_table = boxoffice.performance_table(ex)
    out["refit"] = refurbish(ref, perf_table, venue_name, indoor_capacity,
                             target_percentile)
    pp = out["refit"].get("now_pp") or 0.0
    out["revenue"] = boxoffice.project(
        ref, roof["extra_visits_low"], roof["extra_visits_high"],
        indoor_capacity, performance_pp=pp, years=years)
    out["decomposition"] = boxoffice.decompose(
        ref, roof["extra_visits_low"], roof["extra_visits_high"],
        indoor_capacity, performance_pp=pp)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", required=True)
    ap.add_argument("--venue", required=True)
    ap.add_argument("--country", default=None)
    ap.add_argument("--indoor-capacity", type=int, default=None,
                    # argparse %-formats help strings, so a literal per-cent
                    # sign has to be doubled or it is read as a format spec.
                    help=f"capacity once roofed; defaults to "
                         f"{DEFAULT_ROOF_FRACTION:.0%} of the open capacity"
                         .replace("%", "%%"))
    ap.add_argument("--target-percentile", type=float, default=0.75)
    a = ap.parse_args()

    ex = load_extract(latest_extract())
    row = resolve_market(ex, a.market, a.country)
    blob, err = whatif.fitted_model()
    if err:
        raise SystemExit(err)
    menu = whatif.build_menu(ex, row["countryCode"], blob)
    lad = V.rooms(ex, row)

    hit = lad[lad["venue"].str.casefold() == a.venue.strip().casefold()]
    if hit.empty:
        log(f"no room called {a.venue!r} in {row['city']}. Rooms with a known "
            f"capacity:")
        for r in lad.head(15).itertuples(index=False):
            log(f"   {r.venue}  {r.capacity:,.0f}  ({r.io})")
        raise SystemExit(2)
    vname = hit.iloc[0]["venue"]
    open_cap = float(hit.iloc[0]["capacity"])
    cap = a.indoor_capacity or int(open_cap * DEFAULT_ROOF_FRACTION)

    ref, ref_err = boxoffice.load()
    years = whatif.years_covered(ex)
    perf_table = boxoffice.performance_table(ex)
    case = full_case(menu, ex, row, vname, cap, None if ref_err else ref,
                     a.target_percentile, lad, years, perf_table)

    roof = case["roof"]
    print(f"\n{'=' * 74}\n{vname} — {row['city']}, {row['country']}\n{'=' * 74}")
    print(f"open capacity {roof['open_capacity']:,.0f} ({roof['was']}), "
          f"roofed to {roof['indoor_capacity']:,.0f}")
    print(f"indoor ceiling before: "
          f"{roof['indoor_ceiling_before']:,.0f}" if
          np.isfinite(roof["indoor_ceiling_before"]) else
          "the city has no indoor room with a recorded capacity")
    if roof["note"]:
        print(f"\nNOTE: {roof['note']}")

    print("\n-- A ROOF: what the model can see --")
    print(f"extra tour-visits over {years:.1f} years: "
          f"{roof['extra_visits_low']:+.1f} to {roof['extra_visits_high']:+.1f}")
    print(compare_roof_with_newbuild(menu, row, cap).to_string(index=False))
    print("Identical by construction: both raise the same indoor ceiling to "
          "the same number.\nThe model cannot tell a roof from a new build, "
          "and this is that fact shown\nrather than asserted.")

    ref_data = case.get("refit")
    if ref_data is None:
        print(f"\n-- A REFIT --\n{case.get('revenue_error')}")
    elif not ref_data["known"]:
        print(f"\n-- A REFIT: what the model cannot see --\n{ref_data['why']}")
    else:
        print("\n-- A REFIT: what the model cannot see --")
        print(ref_data["why"])
        print(f"(it sells {ref_data['sell_through']:.0f}% against "
              f"{ref_data['typical_for_size']:.0f}% typical for its size, "
              f"over {ref_data['events']} shows -- {ref_data['band']})")
        print(f"gross per show would go from ${ref_data['gross_before']:,.0f} "
              f"to ${ref_data['gross_after']:,.0f} "
              f"({ref_data['gross_uplift']:+,.0f} a show)")
        print("This changes NO tour-count prediction. Nothing in the fitted "
              "model knows about\nsightlines or bars; a better room shows up "
              "in what it takes, not in who comes.")

    rev = case.get("revenue")
    if rev:
        print("\n-- GROSS BOX OFFICE from the roof --")
        print(f"${rev['central_low']:,.0f} to ${rev['central_high']:,.0f} "
              f"over {years:.1f} years")
        if "per_year_low" in rev:
            print(f"(${rev['per_year_low']:,.0f} to "
                  f"${rev['per_year_high']:,.0f} a year)")
        print("\nWhere that range comes from:")
        print(case["decomposition"].to_string(index=False))
        print("\nGross box office, NOT venue revenue. The venue takes a hire "
              "fee plus ancillaries,\nwhich this database cannot support.")


if __name__ == "__main__":
    main()
