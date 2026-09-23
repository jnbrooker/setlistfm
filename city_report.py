#!/usr/bin/env python3
"""
Everything the database knows about one city, as an Excel workbook.

    python city_report.py Bari
    python city_report.py "Newcastle upon Tyne" --country "United Kingdom"
    python city_report.py Naples --since 2015-01-01 --categories A B C D E
    python city_report.py Bari --out reports

One sheet per question, all of it filterable in Excel rather than pre-digested,
because the next question is never quite the one you planned for.

    Summary            the headline numbers, and how the city ranks nationally
    Venues             every venue, with capacity, type and where each came from
    By year            events per year, by category and indoor/outdoor
    By month           the city against its country and continent, as shares so
                       a small city and a whole continent are comparable
    By day of year     the same, day 1-366
    By weekday         which nights shows actually happen
    Indoor vs outdoor  city / country / continent
    Capacity bands     what size of room the demand actually asks for
    Artists            who plays here, with their tier and how often
    Tours here         tours with a date in the city
    Tours that skipped THE OPPORTUNITY LIST: tours that played the country
                       several times and never came
    Artists that skipped  the same, act by act rather than tour by tour
    Where skippers went   which cities caught the dates this one missed
    Season by type     month by month, indoor against outdoor - how long the
                       indoor season is, and how short the outdoor window
    Legs in and out    the city's routing partners, and a matrix of the top ones
    Funnel             all tours -> play the country -> play here
    Peer cities        the rest of the country ranked, for benchmarking
    Box office         Pollstar rows where they exist: tickets, %% sold, gross
    Methodology        what every number above counts, and what it does not

Reads the same `events` table as since2023.py and shares its loading, category
filtering and indoor/outdoor labelling, so the numbers agree with it.
"""

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

import paths
from since2023 import (CAP_BANDS, CAP_LABELS, DEFAULT_CATEGORIES, MONTHS,
                       autosize, build_legs, load_events, log, methodology,
                       parse_categories)

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# events has no continent column, so ISO-2 -> continent lives here. Only used to
# give the city a wider backdrop to be compared against.
_CONTINENT_CODES = {
    "Europe": """AD AL AT BA BE BG BY CH CY CZ DE DK EE ES FI FO FR GB GG GI GR HR HU IE IM IS IT JE LI LT
                 LU LV MC MD ME MK MT NL NO PL PT RO RS RU SE SI SK SM UA VA XK""",
    "North America": """AG AI AW BB BM BS BZ CA CR CU CW DM DO GD GL GP GT HN HT JM KN KY LC MQ MS MX NI
                        PA PR SV TC TT US VC VG VI""",
    "South America": "AR BO BR CL CO EC FK GF GY PE PY SR UY VE",
    "Asia": """AE AF AM AZ BD BH BN BT CN GE HK ID IL IN IQ IR JO JP KG KH KP KR KW KZ LA LB LK MM MN MO
               MV MY NP OM PH PK PS QA SA SG SY TH TJ TL TM TR TW UZ VN YE""",
    "Africa": """AO BF BI BJ BW CD CF CG CI CM CV DJ DZ EG EH ER ET GA GH GM GN GQ GW KE KM LR LS LY MA
                 MG ML MR MU MW MZ NA NE NG RE RW SC SD SL SN SO SS ST SZ TD TG TN TZ UG ZA ZM ZW""",
    "Oceania": "AS AU CK FJ FM GU KI MH NC NF NR NU NZ PF PG PW SB TO TV VU WF WS",
}
CONTINENT = {code: name for name, codes in _CONTINENT_CODES.items() for code in codes.split()}


def share(counts):
    """Counts as a share of their own total, so different-sized places compare."""
    total = counts.sum()
    return (counts / total).round(5) if total else counts * 0.0


# --------------------------------------------------------------- the city ---

def resolve_city(df, city, country=None):
    """
    Find the city, and refuse to guess when the name is ambiguous.

    There is a London in Canada and a Naples in Florida, and silently picking
    the busier one would be a quiet way to produce a confidently wrong report.
    """
    m = df["city"].astype(str).str.strip().str.casefold() == city.strip().casefold()
    if country:
        m &= df["country"].astype(str).str.strip().str.casefold() == country.strip().casefold()
    hit = df[m]
    if hit.empty:
        near = (df[df["city"].astype(str).str.contains(city.strip(), case=False, na=False)]
                .groupby(["city", "country"]).size().sort_values(ascending=False).head(8))
        log(f"no events for city {city!r}" + (f" in {country}" if country else ""))
        if len(near):
            log("   did you mean:")
            for (c, k), n in near.items():
                log(f"      {c}, {k}  ({n:,} events)")
        raise SystemExit(2)

    countries = hit["country"].dropna().unique()
    if len(countries) > 1:
        log(f"{city!r} exists in more than one country - pass --country to choose:")
        for k, n in hit.groupby("country").size().sort_values(ascending=False).items():
            log(f"      {k}  ({n:,} events)")
        raise SystemExit(2)
    return hit, str(countries[0])


# ------------------------------------------------------------- the sheets ---

def sheet_summary(city, country, cdf, ndf, kdf, a, categories):
    def top(frame, col):
        s = frame[col].dropna()
        return f"{s.mode().iloc[0]} ({(s == s.mode().iloc[0]).sum():,})" if len(s) else "-"

    io = cdf["io"].value_counts()
    cap = pd.to_numeric(cdf["venue_capacity"], errors="coerce")
    by_city = ndf.groupby("city").size().sort_values(ascending=False)
    rank = int(list(by_city.index).index(city)) + 1 if city in by_city.index else None
    rows = [
        ("City", city), ("Country", country),
        ("Continent", CONTINENT.get(str(cdf["countryCode"].dropna().iloc[0]) if len(cdf["countryCode"].dropna()) else "", "-")),
        ("Period", f"{a.since} to {cdf['date_iso'].max()}"),
        ("Artist categories", ", ".join(categories)),
        ("", ""),
        ("Events", len(cdf)),
        ("Rank in country", f"{rank} of {len(by_city):,} cities" if rank else "-"),
        ("Share of country's events", round(len(cdf) / len(ndf), 4) if len(ndf) else None),
        ("Venues used", cdf["venue_uid"].nunique()),
        ("Distinct headliners", cdf["headliner"].nunique()),
        ("Distinct tours", cdf.loc[cdf["tour"] != "", "tour"].nunique()),
        ("", ""),
        ("Indoor events", int(io.get("inside", 0))),
        ("Outdoor events", int(io.get("outside", 0))),
        ("Unknown indoor/outdoor", int(io.get("unknown", 0))),
        ("", ""),
        ("Venues with a known capacity",
         f"{cdf.loc[cap.notna(), 'venue_uid'].nunique()} of {cdf['venue_uid'].nunique()}"),
        ("Events at a venue of known capacity", int(cap.notna().sum())),
        ("Largest venue capacity", int(cap.max()) if cap.notna().any() else None),
        ("Median capacity of the room, per event", int(cap.median()) if cap.notna().any() else None),
        ("", ""),
        ("Busiest year", top(cdf.assign(y=cdf["event_dt"].dt.year), "y")),
        ("Busiest month", top(cdf.assign(m=cdf["event_dt"].dt.month_name()), "m")),
        ("Busiest weekday", top(cdf.assign(d=cdf["event_dt"].dt.day_name()), "d")),
        ("Most frequent headliner", top(cdf, "headliner")),
        ("Busiest venue", top(cdf, "venue")),
    ]
    return pd.DataFrame(rows, columns=["Measure", "Value"])


def sheet_venues(cdf):
    def mode(s):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) else None

    g = (cdf.groupby("venue_uid")
         .agg(venue=("venue", mode), events=("event_id", "size"),
              capacity=("venue_capacity", "max"), capacity_source=("venue_capacity_source", mode),
              venue_type=("venue_type", mode), inside_outside=("io", mode),
              io_source=("io_source", mode), arena=("arena_name", mode),
              headliners=("headliner", "nunique"),
              tours=("tour", lambda s: s[s != ""].nunique()),
              first_event=("event_dt", "min"), last_event=("event_dt", "max"),
              tickets_sold_avg=("pollstar_tickets_sold", "mean"),
              capacity_pct_avg=("pollstar_capacity_pct", "mean"))
         .sort_values("events", ascending=False).reset_index(drop=True))
    g["years_active"] = ((g["last_event"] - g["first_event"]).dt.days / 365.25).round(1)
    g["events_per_year"] = (g["events"] / g["years_active"].replace(0, np.nan)).round(1)
    for c in ("tickets_sold_avg", "capacity_pct_avg"):
        g[c] = g[c].round(1)
    g.insert(0, "rank", range(1, len(g) + 1))
    return g


def sheet_by_year(cdf, ndf):
    y = cdf.assign(year=cdf["event_dt"].dt.year)
    nat = ndf.assign(year=ndf["event_dt"].dt.year).groupby("year").size()
    out = y.pivot_table(index="year", columns="category", values="event_id",
                        aggfunc="count", fill_value=0)
    io = y.pivot_table(index="year", columns="io", values="event_id",
                       aggfunc="count", fill_value=0)
    out["total"] = out.sum(axis=1)
    for c in ("inside", "outside", "unknown"):
        out[c] = io[c] if c in io.columns else 0
    out["venues"] = y.groupby("year")["venue_uid"].nunique()
    out["headliners"] = y.groupby("year")["headliner"].nunique()
    # the country alongside, so a rise here can be told apart from a rise
    # everywhere -- a city can grow 20% and still be losing ground
    out["country total"] = nat.reindex(out.index)
    out["city share of country"] = (out["total"] / out["country total"]).round(5)
    out["city yoy"] = out["total"].pct_change().round(4)
    out["country yoy"] = out["country total"].pct_change().round(4)
    # The last year in the data is almost always part-finished, and a
    # year-on-year change against a part-year reads as a collapse (or, for the
    # part-year itself, as a boom). Say so rather than let the column mislead.
    last_date = cdf["event_dt"].max()
    out["year complete"] = [
        "" if y < last_date.year else f"partial - data ends {last_date:%d %b}"
        for y in out.index]
    return out


def _seasonal(frames, index, extract, labels):
    out = pd.DataFrame(index=index)
    for name, frame in frames.items():
        counts = extract(frame).reindex(index, fill_value=0)
        out[f"{name} events"] = counts.astype(int)
        out[f"{name} share"] = share(counts)
    out.index.name = labels
    return out


def sheet_month(frames):
    return _seasonal(frames, MONTHS,
                     lambda f: f["event_dt"].dt.month_name().value_counts(), "month")


def sheet_day_of_year(frames):
    t = _seasonal(frames, list(range(1, 367)),
                  lambda f: f["event_dt"].dt.dayofyear.value_counts(), "day_of_year")
    t.insert(0, "date (non-leap)",
             [(dt.date(2023, 1, 1) + dt.timedelta(days=i - 1)).strftime("%d %b")
              if i <= 365 else "29 Feb" for i in t.index])
    return t


def sheet_weekday(frames):
    return _seasonal(frames, WEEKDAYS,
                     lambda f: f["event_dt"].dt.day_name().value_counts(), "weekday")


def sheet_season_by_type(cdf):
    """
    Month by month, split indoor/outdoor.

    The question behind it: how long is the sellable indoor season, and how
    short is the outdoor window it has to share the year with. A city whose
    activity is 70% outdoor in three summer months is a different proposition
    from one with a year-round indoor calendar.
    """
    d = cdf.assign(month=cdf["event_dt"].dt.month_name())
    t = (d.pivot_table(index="month", columns="io", values="event_id",
                       aggfunc="count", fill_value=0).reindex(MONTHS, fill_value=0))
    for c in ("inside", "outside", "unknown"):
        if c not in t.columns:
            t[c] = 0
    t = t[["inside", "outside", "unknown"]]
    t["total"] = t.sum(axis=1)
    known = t["inside"] + t["outside"]
    t["% outside (of known)"] = (t["outside"] / known.replace(0, np.nan)).round(4)
    t["% of the year's events"] = share(t["total"])
    t.index.name = "month"
    return t


def sheet_indoor_outdoor(frames):
    rows = []
    for name, frame in frames.items():
        io = frame["io"].value_counts()
        known = int(io.get("inside", 0) + io.get("outside", 0))
        rows.append({
            "scope": name, "events": len(frame),
            "inside": int(io.get("inside", 0)), "outside": int(io.get("outside", 0)),
            "unknown": int(io.get("unknown", 0)),
            "known": known,
            "% outside (of known)": round(io.get("outside", 0) / known, 4) if known else None,
            "% labelled": round(known / len(frame), 4) if len(frame) else None,
        })
    return pd.DataFrame(rows)


def sheet_capacity(cdf):
    d = cdf.copy()
    d["capacity"] = pd.to_numeric(d["venue_capacity"], errors="coerce")
    d["band"] = pd.cut(d["capacity"], CAP_BANDS, labels=CAP_LABELS, right=False).astype(object)
    d["band"] = d["band"].where(d["capacity"].notna(), "capacity unknown")
    order = CAP_LABELS + ["capacity unknown"]
    g = (d.groupby("band").agg(events=("event_id", "size"), venues=("venue_uid", "nunique"),
                               headliners=("headliner", "nunique"))
         .reindex(order, fill_value=0))
    for lab in ("inside", "outside"):
        g[lab] = d[d["io"] == lab].groupby("band").size().reindex(order, fill_value=0)
    known = g.loc[CAP_LABELS, "events"].sum()
    g["% of events with known capacity"] = [
        round(v / known, 4) if (known and b != "capacity unknown") else None
        for b, v in zip(g.index, g["events"])]
    g.index.name = "capacity band"
    return g


def sheet_artists(cdf):
    g = (cdf.groupby("headliner")
         .agg(events=("event_id", "size"), category=("category", "first"),
              venues=("venue_uid", "nunique"),
              tours=("tour", lambda s: s[s != ""].nunique()),
              first=("event_dt", "min"), last=("event_dt", "max"),
              tickets_avg=("pollstar_tickets_sold", "mean"))
         .sort_values("events", ascending=False).reset_index())
    g["tickets_avg"] = g["tickets_avg"].round(0)
    return g


def _tour_frames(ndf, city):
    """Per-tour view of the country, flagging which ones reached the city."""
    t = ndf[ndf["tour"] != ""].copy()
    t["here"] = t["city"] == city
    g = t.groupby("tour").agg(
        headliner=("headliner", "first"), category=("category", "first"),
        country_events=("event_id", "size"), city_events=("here", "sum"),
        cities=("city", "nunique"),
        first=("event_dt", "min"), last=("event_dt", "max"),
        where=("city", lambda s: ", ".join(sorted(set(s)))))
    g["plays_city"] = g["city_events"] > 0
    return g.sort_values(["plays_city", "country_events"], ascending=[False, False])


def sheet_funnel(df, ndf, city, country, min_dates):
    allt = df[df["tour"] != ""]["tour"].nunique()
    g = _tour_frames(ndf, city)
    big = g[g["country_events"] >= min_dates]
    def row(label, frame, base):
        n = len(frame)
        played = int(frame["plays_city"].sum())
        return {"segment": label, "tours": n, f"play {city}": played,
                "% of this segment": round(played / n, 4) if n else None,
                "% of all tours in the data": round(n / base, 4) if base else None}
    return pd.DataFrame([
        {"segment": "every tour in the data", "tours": allt,
         f"play {city}": int(g["plays_city"].sum()),
         "% of this segment": round(g["plays_city"].sum() / allt, 4) if allt else None,
         "% of all tours in the data": 1.0},
        row(f"tours that play {country}", g, allt),
        row(f"... with >= {min_dates} {country} dates", big, allt),
        row(f"... with >= {min_dates} dates, more than one city",
            big[big["cities"] > 1], allt),
    ])


def sheet_missed(ndf, city, min_dates):
    g = _tour_frames(ndf, city)
    missed = g[(~g["plays_city"]) & (g["country_events"] >= min_dates)]
    return missed.drop(columns=["plays_city", "city_events"]).reset_index()


def sheet_where_skippers_went(ndf, city, min_dates):
    """
    For every tour that played the country but not this city, where it went.

    'Tours that skipped' says how much demand passed by; this says who caught
    it. Cities at the top are the ones this city is losing dates to.
    """
    g = _tour_frames(ndf, city)
    skipped = set(g[(~g["plays_city"]) & (g["country_events"] >= min_dates)].index)
    if not skipped:
        return pd.DataFrame(columns=["city", "tours that skipped and played here", "events"])
    e = ndf[ndf["tour"].isin(skipped)]
    out = (e.groupby("city")
           .agg(**{"tours that skipped and played here": ("tour", "nunique"),
                   "events": ("event_id", "size"),
                   "largest venue": ("venue_capacity", "max")})
           .sort_values("tours that skipped and played here", ascending=False)
           .reset_index())
    out["% of the skipping tours"] = (
        out["tours that skipped and played here"] / len(skipped)).round(4)
    return out


def sheet_artists_that_skipped(ndf, city, min_dates):
    """Artist-level counterpart: acts touring the country that never called."""
    a = ndf.copy()
    a["here"] = a["city"] == city
    g = a.groupby("headliner").agg(
        category=("category", "first"), country_events=("event_id", "size"),
        city_events=("here", "sum"), cities=("city", "nunique"),
        first=("event_dt", "min"), last=("event_dt", "max"),
        where=("city", lambda s: ", ".join(sorted(set(s))[:12])))
    missed = g[(g["city_events"] == 0) & (g["country_events"] >= min_dates)]
    return (missed.drop(columns=["city_events"])
            .sort_values("country_events", ascending=False).reset_index())


def sheet_legs(df, city):
    legs = build_legs(df)
    legs = legs[legs["origin_city"] != legs["dest_city"]]
    into = legs[legs["dest_city"] == city]
    out = legs[legs["origin_city"] == city]
    partners = pd.concat([
        into.groupby("origin_city").size().rename("arrived from"),
        out.groupby("dest_city").size().rename("departed to"),
    ], axis=1).fillna(0).astype(int)
    partners["total"] = partners.sum(axis=1)
    partners = partners.sort_values("total", ascending=False)
    partners.index.name = "partner city"
    cols = ["tour", "headliner", "origin_date", "origin_city", "origin_country",
            "origin_venue", "dest_date", "dest_city", "dest_country", "dest_venue",
            "days_between"]
    detail = pd.concat([into, out]).sort_values("origin_date")[cols]
    return partners.reset_index(), detail


def sheet_matrix(df, city, partners, top):
    """Origin x destination among the city and the places it trades with most."""
    legs = build_legs(df)
    legs = legs[legs["origin_city"] != legs["dest_city"]]
    keep = [city] + [c for c in partners["partner city"].head(top).tolist() if c != city]
    sub = legs[legs["origin_city"].isin(keep) & legs["dest_city"].isin(keep)]
    m = (sub.groupby(["origin_city", "dest_city"]).size().rename("legs").reset_index()
         .pivot(index="origin_city", columns="dest_city", values="legs")
         .reindex(index=keep, columns=keep))
    m.index.name = "origin ↓ / destination →"
    return m


def sheet_peers(ndf, city):
    g = (ndf.groupby("city")
         .agg(events=("event_id", "size"), venues=("venue_uid", "nunique"),
              headliners=("headliner", "nunique"),
              tours=("tour", lambda s: s[s != ""].nunique()),
              largest_venue=("venue_capacity", "max"))
         .sort_values("events", ascending=False).reset_index())
    io = ndf.pivot_table(index="city", columns="io", values="event_id",
                         aggfunc="count", fill_value=0)
    for lab in ("inside", "outside"):
        g[lab] = g["city"].map(io[lab]) if lab in io.columns else 0
    known = g["inside"] + g["outside"]
    g["% outside"] = (g["outside"] / known.replace(0, np.nan)).round(4)
    g.insert(0, "rank", range(1, len(g) + 1))
    g["this city"] = np.where(g["city"] == city, "<<<", "")
    return g


def sheet_box_office(cdf):
    b = cdf[cdf["pollstar_id"].notna()] if "pollstar_id" in cdf.columns else cdf.iloc[0:0]
    cols = [c for c in ["date_iso", "headliner", "venue", "category",
                        "pollstar_run_shows", "pollstar_tickets_sold", "pollstar_capacity",
                        "pollstar_capacity_pct", "pollstar_gross_usd", "pollstar_price_min",
                        "pollstar_price_max", "pollstar_price_avg", "pollstar_promoter"]
            if c in b.columns]
    return b[cols].sort_values("date_iso", ascending=False) if len(b) else pd.DataFrame(columns=cols)


# ------------------------------------------------------------------- main ---

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("city")
    p.add_argument("--country", default=None, help="needed when the city name is ambiguous")
    p.add_argument("--db", default=paths.DB)
    p.add_argument("--since", default="2023-01-01")
    p.add_argument("--categories", nargs="+", default=None,
                   help=f"letters A-E, OTHER, TENANT, NONTENANT, or 'all' "
                        f"(default: {' '.join(DEFAULT_CATEGORIES)})")
    p.add_argument("--out", default=paths.here("reports"))
    p.add_argument("--min-tour-dates", type=int, default=3,
                   help="country dates a tour needs before skipping the city counts as a miss")
    p.add_argument("--top-partners", type=int, default=14, help="cities on the leg matrix")
    p.add_argument("--no-infer", action="store_true",
                   help="indoor/outdoor from database labels only")
    a = p.parse_args()

    categories = parse_categories(a.categories)
    log(f"loading events since {a.since} in {categories}")
    df = load_events(a.db, a.since, categories, infer=not a.no_infer,
                     with_pollstar=True)
    log(f"{len(df):,} events loaded")

    cdf, country = resolve_city(df, a.city, a.country)
    city = str(cdf["city"].iloc[0])
    ndf = df[df["country"] == country]
    code = str(cdf["countryCode"].dropna().iloc[0]) if len(cdf["countryCode"].dropna()) else ""
    continent = CONTINENT.get(code, "")
    kdf = df[df["countryCode"].map(CONTINENT) == continent] if continent else df.iloc[0:0]
    log(f"{city}, {country}: {len(cdf):,} events | {country}: {len(ndf):,} | "
        f"{continent or 'continent unknown'}: {len(kdf):,}")

    frames = {city: cdf, country: ndf}
    if continent and len(kdf):
        frames[continent] = kdf

    partners, legs_detail = sheet_legs(df, city)
    os.makedirs(a.out, exist_ok=True)
    safe = "".join(ch for ch in city if ch.isalnum() or ch in " -_").strip().replace(" ", "_")
    out = os.path.join(a.out, f"{safe}_{a.since[:4]}.xlsx")

    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        sheet_summary(city, country, cdf, ndf, kdf, a, categories).to_excel(
            xw, sheet_name="Summary", index=False)
        sheet_venues(cdf).to_excel(xw, sheet_name="Venues", index=False)
        sheet_by_year(cdf, ndf).to_excel(xw, sheet_name="By year")
        sheet_month(frames).to_excel(xw, sheet_name="By month")
        sheet_day_of_year(frames).to_excel(xw, sheet_name="By day of year")
        sheet_weekday(frames).to_excel(xw, sheet_name="By weekday")
        sheet_indoor_outdoor(frames).to_excel(xw, sheet_name="Indoor vs outdoor", index=False)
        sheet_season_by_type(cdf).to_excel(xw, sheet_name="Season by type")
        sheet_capacity(cdf).to_excel(xw, sheet_name="Capacity bands")
        sheet_artists(cdf).to_excel(xw, sheet_name="Artists", index=False)
        tours = _tour_frames(ndf, city)
        tours[tours["plays_city"]].reset_index().to_excel(
            xw, sheet_name="Tours here", index=False)
        sheet_missed(ndf, city, a.min_tour_dates).to_excel(
            xw, sheet_name="Tours that skipped", index=False)
        sheet_artists_that_skipped(ndf, city, a.min_tour_dates).to_excel(
            xw, sheet_name="Artists that skipped", index=False)
        sheet_where_skippers_went(ndf, city, a.min_tour_dates).to_excel(
            xw, sheet_name="Where skippers went", index=False)
        sheet_funnel(df, ndf, city, country, a.min_tour_dates).to_excel(
            xw, sheet_name="Funnel", index=False)
        partners.to_excel(xw, sheet_name="Leg partners", index=False)
        sheet_matrix(df, city, partners, a.top_partners).to_excel(xw, sheet_name="Leg matrix")
        legs_detail.to_excel(xw, sheet_name="Leg detail", index=False)
        sheet_peers(ndf, city).to_excel(xw, sheet_name="Peer cities", index=False)
        sheet_box_office(cdf).to_excel(xw, sheet_name="Box office", index=False)
        methodology(xw, f"{city} - what these numbers count", [
            f"Source: {a.db} (events table), built {dt.datetime.now():%Y-%m-%d %H:%M}",
            f"City: {city}, {country}" + (f" ({continent})" if continent else ""),
            f"Period: events on or after {a.since}",
            f"Artist categories: {', '.join(categories)}. Change with --categories; "
            f"'all' adds family/comedy and sporting events.",
            "",
            "One row of `events` is one show. Rows sourced from Pollstar rather than a "
            "setlist cover a whole run at one venue, so a six-performance family show is "
            "one row with pollstar_run_shows = 6.",
            "Indoor/outdoor: the database's venue label, then its arena label, then a guess "
            "from the venue name; 'unknown' where none applied. Shares headed '% of known' "
            "exclude the unknowns, so read them next to '% labelled'.",
            "Capacity is the venue's, not the attendance: a 4,000 crowd in a 60,000 stadium "
            "counts in the 60,000+ band. It says where shows happen, not how many tickets sold.",
            "",
            "'By month', 'By day of year' and 'By weekday' give each scope its own share of "
            "its own total, so a city and a continent can be compared on one chart.",
            f"'Tours that skipped' lists tours with at least {a.min_tour_dates} {country} "
            f"dates and none in {city} - the closest thing here to an opportunity list.",
            "A leg is two consecutive shows on the same tour; 'Leg partners' counts the "
            "cities immediately before and after a date in this city.",
            "",
            "Box office exists only where a Pollstar row matched, which is a minority of "
            "events - absence there means unreported, not unsold.",
        ])
        autosize(xw)

    log(f"wrote {out}")
    log(f"   {len(cdf):,} events | {cdf['venue_uid'].nunique():,} venues | "
        f"{cdf['headliner'].nunique():,} headliners | "
        f"{len(sheet_missed(ndf, city, a.min_tour_dates)):,} tours skipped the city")


if __name__ == "__main__":
    sys.exit(main())
