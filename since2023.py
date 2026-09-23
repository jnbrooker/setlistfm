#!/usr/bin/env python3
"""
Build every "since 2023" analysis workbook from setlistfm.db in one go.

Reads `events` (plus `venues` for capacity), filters to the artist categories
you ask for, and writes to since2023/ :

  italy_top_venues.xlsx            every Italian venue ranked by events, with
                                   renamed venues merged (arena_id, else the
                                   venues table's venue_uid)
  italy_indoor_outdoor_by_city.xlsx  events per Italian city split inside /
                                   outside / unknown            (+ .png)
  southern_italy_tour_legs.xlsx    origin -> destination legs touching
                                   Campania, Sicily, Calabria, Apulia,
                                   Basilicate or Molise         (+ .png)
  tour_funnel.xlsx                 all tours -> play Italy -> play the South
                                   -> play Bari / Naples, with indoor/outdoor
  seasonality.xlsx                 shows by month and by day-of-year for
                                   Europe and Italy, all / inside / outside
                                   (+ .png)
  outdoor_by_capacity.xlsx         outdoor events in Italy and Europe bucketed
                                   by venue capacity, with the venues behind
                                   each bucket                  (+ .png)

Indoor/outdoor comes from the database's venue label where it has one,
otherwise from the venue name (venue_io.py), otherwise 'unknown'.

USAGE
    python since2023.py                              # Category A, B, C
    python since2023.py --categories A B C D E       # everyone
    python since2023.py --categories all             # incl. non-artist + sporting events
    python since2023.py --categories A B C TENANT NONTENANT
    python since2023.py --since 2020-01-01 --out since2020
    python since2023.py --no-infer                   # DB labels only
    python since2023.py --only capacity              # one output, others untouched
"""

import argparse
import datetime as dt
import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths
from venue_io import infer_from_name

SOUTH = ["Campania", "Sicily", "Calabria", "Apulia", "Basilicate", "Molise"]

# category letters -> the label stored in events.category
CATEGORY_LABELS = {
    "A": "Category A", "B": "Category B", "C": "Category C",
    "D": "Category D", "E": "Category E",
    "OTHER": "Family, Entertainment, Comedy & Other",
    "TENANT": "Tenant Sporting Event",
    "NONTENANT": "Non-Tenant Sporting Event",
}
DEFAULT_CATEGORIES = ["A", "B", "C"]

# ISO-2 codes we treat as Europe (events has no continent column)
EUROPE = set("""
AD AL AT BA BE BG BY CH CY CZ DE DK EE ES FI FO FR GB GG GI GR HR HU IE IM IS IT JE LI LT LU LV MC MD ME MK
MT NL NO PL PT RO RS RU SE SI SK SM UA VA XK
""".split())

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
C_INSIDE, C_OUTSIDE, C_ALL = "#0072B2", "#E69F00", "#2A9D8F"   # Okabe-Ito blue / orange, teal for totals


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ----------------------------------------------------------------- loading

def parse_categories(values):
    if values is None:
        values = DEFAULT_CATEGORIES
    if any(v.lower() == "all" for v in values):
        return list(CATEGORY_LABELS.values())
    out = []
    for v in values:
        key = v.strip().upper().replace("CATEGORY ", "")
        if key not in CATEGORY_LABELS:
            raise SystemExit(f"unknown category {v!r}; choose from {', '.join(CATEGORY_LABELS)} or 'all'")
        out.append(CATEGORY_LABELS[key])
    return out


# The box-office columns roughly double the width of the frame, so they are
# fetched only when something asks for them (city_report.py does; the Italy
# workbooks do not).
POLLSTAR_EXTRA = """, e.source, e.duplicate_risk, e.pollstar_id, e.pollstar_run_shows,
       e.pollstar_tickets_sold, e.pollstar_capacity, e.pollstar_capacity_pct,
       e.pollstar_gross_usd, e.pollstar_price_min, e.pollstar_price_max,
       e.pollstar_price_avg, e.pollstar_promoter, e.pollstar_genre"""


def load_events(db, since, categories, infer=True, with_pollstar=False):
    """Events on/after `since` in the chosen categories, with io / io_source added."""
    marks = ",".join("?" * len(categories))
    sql = f"""
        select e.event_id, e.date_iso, e.venue, e.venue_uid, e.city, e.state, e.country, e.countryCode,
               e.latitude, e.longitude, e.headliner, e.artists, e.tour, e.category, e.event_type,
               e.arena_id, e.arena_name, e.arena_capacity, e.arena_type, e.arena_outside_inside,
               e.venue_capacity, e.venue_capacity_source, e.venue_type, e.venue_outside_inside
               {POLLSTAR_EXTRA if with_pollstar else ""}
        from events e
        where e.date_iso >= ? and e.category in ({marks})
    """
    with sqlite3.connect(db) as con:
        df = pd.read_sql(sql, con, params=[since, *categories])
    df["event_dt"] = pd.to_datetime(df["date_iso"], errors="coerce")
    df = df[df["event_dt"].notna()].copy()
    df["tour"] = df["tour"].fillna("").astype(str).str.strip()

    # indoor / outdoor: venue label -> arena label -> name inference
    venue_lab = df["venue_outside_inside"].astype(str).str.strip().str.lower()
    arena_lab = df["arena_outside_inside"].astype(str).str.strip().str.lower()
    io = pd.Series("unknown", index=df.index)
    src = pd.Series(None, index=df.index, dtype=object)
    m = venue_lab.isin(["inside", "outside"])
    io[m], src[m] = venue_lab[m], "db venue"
    m = ~m & arena_lab.isin(["inside", "outside"])
    io[m], src[m] = arena_lab[m], "db arena"
    if infer:
        need = io == "unknown"
        guess = df.loc[need, "venue"].map(infer_from_name)
        hit = guess[guess.notna()]
        io[hit.index], src[hit.index] = hit, "venue name"
    df["io"], df["io_source"] = io, src
    return df


# ---------------------------------------------------------------- helpers

def methodology(xw, title, lines):
    pd.DataFrame({title: lines}).to_excel(xw, sheet_name="Methodology", index=False)


def autosize(xw):
    for ws in xw.sheets.values():
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col[:300] if c.value is not None), default=8)
            ws.column_dimensions[col[0].column_letter].width = min(max(8, width + 2), 60)
        ws.freeze_panes = "A2"


def common_notes(a, categories, n_events):
    return [
        f"Source: {a.db} (events table), built {dt.datetime.now():%Y-%m-%d %H:%M}",
        f"Period: events on/after {a.since}",
        f"Artist categories included: {', '.join(categories)}  ({n_events:,} events after filtering)",
        "Indoor/outdoor: database venue label, else database arena label, else "
        + ("inferred from the venue name (venue_io.py), else " if not a.no_infer else "")
        + "'unknown'.",
    ]


# ------------------------------------------------------------- 1. venues

def italy_top_venues(df, a, categories, out):
    it = df[df["country"] == "Italy"].copy()
    it["_key"] = np.where(it["arena_id"].notna(), "arena:" + it["arena_id"].astype(str), "venue:" + it["venue_uid"].astype(str))

    def most_common(s):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) else None

    g = (it.groupby("_key")
         .agg(venue=("venue", most_common), city=("city", most_common), state=("state", most_common),
              events=("event_id", "size"), names_combined=("venue", "nunique"),
              combined_from=("venue", lambda s: "; ".join(sorted(set(s), key=str.casefold))),
              capacity=("venue_capacity", "max"), capacity_source=("venue_capacity_source", most_common),
              arena_capacity=("arena_capacity", "max"), venue_type=("venue_type", most_common),
              inside_outside=("io", most_common), io_source=("io_source", most_common),
              first_event=("event_dt", "min"), last_event=("event_dt", "max"),
              headliners=("headliner", "nunique"), latitude=("latitude", most_common), longitude=("longitude", most_common))
         .sort_values(["events", "venue"], ascending=[False, True]).reset_index(drop=True))
    g.insert(0, "rank", range(1, len(g) + 1))
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        g.to_excel(xw, sheet_name=f"Italy since {a.since[:4]}", index=False)
        methodology(xw, "Italian venues by events hosted", common_notes(a, categories, len(df)) + [
            "One row per building: events at a curated arena are grouped by arena_id (which folds every alias of its name), "
            "everything else by the venues table's venue_uid. names_combined / combined_from show the spellings merged.",
            "capacity = venue_capacity from the events table (best available: curated sheet, Pollstar, or inferred); "
            "arena_capacity is the curated figure where the venue is a curated arena.",
        ])
        autosize(xw)
    log(f"venues: {len(it):,} Italian events -> {len(g):,} venues")


# ------------------------------------------------- 2. indoor/outdoor by city

def italy_indoor_outdoor(df, a, categories, out, top=30, charts=True):
    d = df[df["country"] == "Italy"]
    pivot = d.pivot_table(index="city", columns="io", values="event_id", aggfunc="count", fill_value=0)
    for c in ["inside", "outside", "unknown"]:
        if c not in pivot.columns:
            pivot[c] = 0
    pivot["total"] = pivot["inside"] + pivot["outside"]
    pivot["pct_outside"] = np.where(pivot["total"] > 0, pivot["outside"] / pivot["total"], np.nan)
    res = pivot[["inside", "outside", "total", "pct_outside", "unknown"]].sort_values("total", ascending=False)
    res.index.name = "city"
    labels = (d.groupby(["venue", "city"]).agg(events=("event_id", "size"), label=("io", "first"), source=("io_source", "first"))
              .sort_values("events", ascending=False).reset_index())
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        res.to_excel(xw, sheet_name="By city")
        labels.to_excel(xw, sheet_name="Venue labels", index=False)
        methodology(xw, "Italian events by city, indoor vs outdoor", common_notes(a, categories, len(df)) + [
            "inside / outside = events at venues labelled as such; total = inside + outside; "
            "unknown = venues with no label (excluded from total and pct_outside).",
            "'Venue labels' lists every venue with its label and where it came from - correct venue_io.py or the DB if one is wrong.",
        ])
        autosize(xw)
    if charts:
        r = res.head(top)
        y = np.arange(len(r)); h = 0.4
        plt.figure(figsize=(11, max(6, 0.42 * len(r))))
        plt.barh(y - h / 2, r["inside"], height=h, color=C_INSIDE, edgecolor="black", linewidth=0.3, label="Inside")
        plt.barh(y + h / 2, r["outside"], height=h, color=C_OUTSIDE, edgecolor="black", linewidth=0.3, label="Outside")
        plt.yticks(y, r.index); plt.gca().invert_yaxis()
        plt.xlabel("Number of events")
        plt.title(f"Events by City in Italy: Indoor vs Outdoor (since {a.since}, top {len(r)} cities)")
        plt.legend(title="Venue type"); plt.grid(axis="x", linestyle="--", alpha=0.6); plt.tight_layout()
        plt.savefig(out.replace(".xlsx", ".png"), dpi=150); plt.close()
    log(f"indoor/outdoor: {int(res['total'].sum()):,} labelled Italian events, {int(res['unknown'].sum()):,} unknown")


# --------------------------------------------------------- 3. tour legs

def build_legs(df):
    t = df[df["tour"] != ""].sort_values(["tour", "event_dt"], kind="stable").copy()
    for c in ["event_dt", "city", "state", "country", "venue", "io"]:
        t["dest_" + c] = t.groupby("tour")[c].shift(-1)
    legs = t[t["dest_city"].notna()].rename(columns={
        "event_dt": "origin_date", "city": "origin_city", "state": "origin_state", "country": "origin_country",
        "venue": "origin_venue", "io": "origin_io", "dest_event_dt": "dest_date"})
    legs["days_between"] = (legs["dest_date"] - legs["origin_date"]).dt.days
    return legs


def southern_tour_legs(df, a, categories, out, top=15, charts=True):
    legs = build_legs(df)
    o_s = (legs["origin_country"] == "Italy") & legs["origin_state"].isin(SOUTH)
    d_s = (legs["dest_country"] == "Italy") & legs["dest_state"].isin(SOUTH)
    L = legs[o_s | d_s].copy()
    L["direction"] = np.select([o_s[L.index] & d_s[L.index], o_s[L.index]], ["within South", "leaving South"], "arriving South")
    L = L[L["origin_city"] != L["dest_city"]]
    lab = lambda c, k: np.where(k == "Italy", c, c + " (" + k.astype(str) + ")")
    L["origin"] = lab(L["origin_city"], L["origin_country"])
    L["dest"] = lab(L["dest_city"], L["dest_country"])

    pairs = L.groupby(["origin", "dest"]).size().rename("legs").reset_index().sort_values("legs", ascending=False)
    cities = pd.concat([L.groupby("origin").size().rename("legs_out"), L.groupby("dest").size().rename("legs_in")], axis=1).fillna(0).astype(int)
    cities["total_legs"] = cities["legs_out"] + cities["legs_in"]
    meta = pd.concat([L[["origin", "origin_state", "origin_country"]].set_axis(["city", "state", "country"], axis=1),
                      L[["dest", "dest_state", "dest_country"]].set_axis(["city", "state", "country"], axis=1)]).drop_duplicates("city").set_index("city")
    cities = cities.join(meta)
    cities["southern"] = (cities["country"] == "Italy") & cities["state"].isin(SOUTH)
    cities = cities.sort_values("total_legs", ascending=False).rename_axis("city").reset_index()
    topc = cities["city"].head(top).tolist()
    matrix = pairs[pairs["origin"].isin(topc) & pairs["dest"].isin(topc)].pivot(index="origin", columns="dest", values="legs").reindex(index=topc, columns=topc)
    matrix.index.name = "origin ↓ / destination →"

    cols = ["tour", "headliner", "direction", "origin_date", "origin", "origin_state", "origin_country", "origin_venue", "origin_io",
            "dest_date", "dest", "dest_state", "dest_country", "dest_venue", "dest_io", "days_between", "category"]
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        matrix.to_excel(xw, sheet_name="Matrix (heatmap)")
        pairs.to_excel(xw, sheet_name="All pairs", index=False)
        cities.to_excel(xw, sheet_name="City totals", index=False)
        L[cols].sort_values(["origin_date", "tour"]).to_excel(xw, sheet_name="Leg detail", index=False)
        methodology(xw, "Tour legs to and from Southern Italy", common_notes(a, categories, len(df)) + [
            "A leg = two consecutive shows on the same tour (sorted by date), origin -> destination, within the filtered events.",
            f"Kept when origin OR destination is in {', '.join(SOUTH)}. Same-city legs (multi-night stands) dropped.",
            f"Matrix shows the top {top} cities by total legs; 'All pairs' has every pair. Foreign cities are 'City (Country)'.",
        ])
        autosize(xw)
    if charts:
        vals = matrix.to_numpy(dtype=float); n = len(topc)
        fig, ax = plt.subplots(figsize=(max(9, 0.75 * n + 3), max(7, 0.6 * n + 2)))
        cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#FAFAFA")
        im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, aspect="auto")
        ax.set_xticks(range(n), topc, rotation=45, ha="right"); ax.set_yticks(range(n), topc)
        ax.set_xticks(np.arange(-0.5, n), minor=True); ax.set_yticks(np.arange(-0.5, n), minor=True)
        ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        vmax = np.nanmax(vals) if np.isfinite(vals).any() else 1
        for i in range(n):
            for j in range(n):
                if np.isfinite(vals[i, j]):
                    ax.text(j, i, int(vals[i, j]), ha="center", va="center", fontsize=8, color="white" if vals[i, j] > 0.6 * vmax else "#1a1a1a")
        ax.set_xlabel("DESTINATION  (goes to →)", labelpad=10); ax.set_ylabel("← ORIGIN  (comes from)", labelpad=10)
        ax.set_title(f"Tour legs to and from Southern Italy (since {a.since}) — top {n} cities, {len(L):,} legs", pad=12)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("Tour legs"); cb.outline.set_visible(False)
        fig.tight_layout(); fig.savefig(out.replace(".xlsx", ".png"), dpi=150); plt.close(fig)
    log(f"tour legs: {len(legs):,} legs -> {len(L):,} touching the South")


# ------------------------------------------------------------ 4. funnel

def tour_funnel(df, a, categories, out, min_shows=5):
    t = df[df["tour"] != ""].copy()
    t["it"] = t["country"] == "Italy"
    t["south"] = t["it"] & t["state"].isin(SOUTH)
    t["apulia"] = t["it"] & (t["state"] == "Apulia")
    t["bari"] = t["it"] & (t["city"] == "Bari")
    t["naples"] = t["it"] & (t["city"] == "Naples")
    areas = ["it", "south", "apulia", "bari", "naples"]
    for ar in areas:
        for io in ["inside", "outside", "unknown"]:
            t[f"{ar}_{io}"] = t[ar] & (t["io"] == io)
    aggs = {f"{ar}_{io}_shows": (f"{ar}_{io}", "sum") for ar in areas for io in ["inside", "outside", "unknown"]}
    g = t.groupby("tour").agg(
        **aggs, headliner=("headliner", "first"), category=("category", "first"),
        first_show=("event_dt", "min"), last_show=("event_dt", "max"), shows=("event_id", "size"),
        italy_shows=("it", "sum"), south_shows=("south", "sum"), apulia_shows=("apulia", "sum"),
        bari_shows=("bari", "sum"), naples_shows=("naples", "sum"), countries=("country", "nunique"),
        italy_cities=("city", lambda s: ", ".join(sorted(set(s[t.loc[s.index, "it"]])))))
    for ar, col in [("italy", "italy_shows"), ("south", "south_shows"), ("apulia", "apulia_shows"), ("bari", "bari_shows"), ("naples", "naples_shows")]:
        g[f"plays_{ar}"] = g[col] > 0
    g["scope"] = np.where(g["shows"] == g["italy_shows"], "Italy-only", "international")

    def pct(x, y):
        return round(x / y, 4) if y else None

    def row(gg, label):
        n = len(gg); i = gg["plays_italy"].sum(); s = gg["plays_south"].sum()
        r = {"segment": label, "tours": n, "play Italy": i, "% of tours": pct(i, n),
             "play South": s, "% of Italy tours": pct(s, i),
             "play Apulia": gg["plays_apulia"].sum(), "% of Italy tours (Apulia)": pct(gg["plays_apulia"].sum(), i),
             "play Bari": gg["plays_bari"].sum(), "% of Italy tours (Bari)": pct(gg["plays_bari"].sum(), i),
             "% of South tours (Bari)": pct(gg["plays_bari"].sum(), s),
             "play Naples": gg["plays_naples"].sum(), "% of Italy tours (Naples)": pct(gg["plays_naples"].sum(), i),
             "% of South tours (Naples)": pct(gg["plays_naples"].sum(), s)}
        for area, key, base in [("Italy", "it", i), ("South", "south", s), ("Apulia", "apulia", gg["plays_apulia"].sum()),
                                ("Bari", "bari", gg["plays_bari"].sum()), ("Naples", "naples", gg["plays_naples"].sum())]:
            for lab in ["inside", "outside", "unknown"]:
                k = (gg[f"{key}_{lab}_shows"] > 0).sum()
                r[f"{area}: tours with an {lab} show"] = k
                r[f"{area}: % {lab} (of tours playing {area})"] = pct(k, base)
            r[f"{area}: shows inside / outside / unknown"] = " / ".join(str(int(gg[f"{key}_{lab}_shows"].sum())) for lab in ["inside", "outside", "unknown"])
        return r

    big = g[g["shows"] >= min_shows]
    funnel = pd.DataFrame([
        row(g, "all tours (>=1 show)"),
        row(big, f"tours with >={min_shows} shows"),
        row(big[big["scope"] == "international"], f"  international, >={min_shows} shows"),
        row(big[big["scope"] == "Italy-only"], f"  Italy-only, >={min_shows} shows"),
        row(big[(big["scope"] == "international") & (big["italy_shows"] >= 3)], f"  international, >={min_shows} shows, >=3 Italian dates"),
    ])
    core = [c for c in funnel.columns if ":" not in c]
    io_cols = ["segment"] + [c for c in funnel.columns if ":" in c]
    it = g[g["plays_italy"]].sort_values(["plays_bari", "plays_south", "italy_shows"], ascending=False)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        funnel[core].to_excel(xw, sheet_name="Funnel", index=False)
        funnel[io_cols].T.to_excel(xw, sheet_name="Indoor-outdoor split", header=False)
        it.to_excel(xw, sheet_name="Tours playing Italy")
        it[it["plays_south"] & ~it["plays_bari"] & (it["italy_shows"] >= 3)].to_excel(xw, sheet_name="South but not Bari")
        methodology(xw, "Tour funnel: all tours -> Italy -> South -> Bari", common_notes(a, categories, len(df)) + [
            "Tour = the setlist.fm tour name on the event; events with no tour name are excluded. A tour is in the period if any of its shows is.",
            "'international' = has at least one show outside Italy; 'Italy-only' = every show in Italy.",
            "'plays X' = at least one show in X. '% of tours' is over the segment; the other percentages are over tours that play Italy (or the South).",
            "Indoor-outdoor split: a tour counts under 'inside' if any of its shows in that area was at an indoor venue (so inside + outside can exceed 100%).",
        ])
        autosize(xw)
    log(f"funnel: {len(g):,} tours, {int(g['plays_italy'].sum()):,} play Italy, {int(g['plays_south'].sum()):,} the South, {int(g['plays_bari'].sum()):,} Bari")


# -------------------------------------------------------- 5. seasonality

def seasonality(df, a, categories, out, charts=True):
    eu = df[df["countryCode"].isin(EUROPE)]
    it = df[df["country"] == "Italy"]
    sets = {"Europe": eu, "Italy": it}
    month_tables, day_tables = {}, {}
    for region, d in sets.items():
        mt = pd.DataFrame(index=MONTHS)
        dyt = pd.DataFrame(index=range(1, 367))
        for lab, sub in [("all", d), ("inside", d[d["io"] == "inside"]), ("outside", d[d["io"] == "outside"]), ("unknown", d[d["io"] == "unknown"])]:
            mt[lab] = sub["event_dt"].dt.month_name().value_counts().reindex(MONTHS).fillna(0).astype(int)
            dyt[lab] = sub["event_dt"].dt.dayofyear.value_counts().reindex(range(1, 367), fill_value=0).astype(int)
        mt.index.name, dyt.index.name = "month", "day_of_year"
        # a readable calendar date for the day-of-year rows (non-leap year)
        dyt.insert(0, "date (non-leap)", [(dt.date(2023, 1, 1) + dt.timedelta(days=i - 1)).strftime("%d %b") if i <= 365 else "29 Feb" for i in dyt.index])
        month_tables[region], day_tables[region] = mt, dyt
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for region in sets:
            month_tables[region].to_excel(xw, sheet_name=f"{region} by month")
            day_tables[region].to_excel(xw, sheet_name=f"{region} by day")
        methodology(xw, "Shows by month and by day of year", common_notes(a, categories, len(df)) + [
            f"Europe = events whose countryCode is in: {' '.join(sorted(EUROPE))}.",
            "Months are aggregated across all years in the period (Jan-Dec totals), as in month_graph() in data_exploration.ipynb; "
            "day of year 1-366 as in day_graph(). all = every event; inside / outside / unknown by venue label.",
        ])
        autosize(xw)
    if charts:
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        for col, region in enumerate(sets):
            mt, dyt = month_tables[region], day_tables[region]
            ax = axes[0, col]; x = np.arange(12); w = 0.4
            ax.bar(x - w / 2, mt["inside"], w, color=C_INSIDE, label="Inside")
            ax.bar(x + w / 2, mt["outside"], w, color=C_OUTSIDE, label="Outside")
            ax.set_xticks(x, [m[:3] for m in MONTHS]); ax.set_title(f"{region}: shows by month (since {a.since})")
            ax.set_ylabel("Number of shows"); ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax = axes[1, col]
            ax.bar(dyt.index, dyt["all"], width=1.0, color=C_ALL)
            ax.set_xlim(1, 366); ax.set_title(f"{region}: shows by day of year (all venues)")
            ax.set_xlabel("Day of the year (1 to 366)"); ax.set_ylabel("Number of shows")
        fig.tight_layout(); fig.savefig(out.replace(".xlsx", ".png"), dpi=150); plt.close(fig)
    log(f"seasonality: Europe {len(eu):,} events, Italy {len(it):,}")


# ------------------------------------------------- 6. outdoor by capacity

CAP_BANDS = [0, 1_000, 2_500, 5_000, 10_000, 20_000, 40_000, 60_000, 10**9]
CAP_LABELS = ["under 1,000", "1,000-2,499", "2,500-4,999", "5,000-9,999", "10,000-19,999",
              "20,000-39,999", "40,000-59,999", "60,000+"]


def outdoor_by_capacity(df, a, categories, out, charts=True):
    """Outdoor events bucketed by the venue's best-known capacity, Italy and Europe."""
    od = df[df["io"] == "outside"].copy()
    od["capacity"] = pd.to_numeric(od["venue_capacity"], errors="coerce")
    od["band"] = pd.cut(od["capacity"], CAP_BANDS, labels=CAP_LABELS, right=False).astype(object)
    od["band"] = od["band"].where(od["capacity"].notna(), "capacity unknown")
    order = CAP_LABELS + ["capacity unknown"]

    regions = {"Italy": od[od["country"] == "Italy"], "Europe": od[od["countryCode"].isin(EUROPE)]}
    tables = {}
    for region, d in regions.items():
        g = (d.groupby("band").agg(events=("event_id", "size"), venues=("venue_uid", "nunique"),
                                   headliners=("headliner", "nunique"))
             .reindex(order, fill_value=0))
        known = g.loc[CAP_LABELS, "events"].sum()
        g["% of events with known capacity"] = [round(v / known, 4) if (known and b != "capacity unknown") else None
                                                for b, v in zip(g.index, g["events"])]
        g["% of all outdoor events"] = (g["events"] / len(d)).round(4) if len(d) else None
        g.index.name = "capacity band"
        tables[region] = g

    def most_common(s):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) else None

    venues = (od[od["countryCode"].isin(EUROPE)]
              .groupby("venue_uid")
              .agg(venue=("venue", most_common), city=("city", most_common), country=("country", most_common),
                   capacity=("capacity", "max"), capacity_source=("venue_capacity_source", most_common),
                   band=("band", most_common), venue_type=("venue_type", most_common),
                   io_source=("io_source", most_common), events=("event_id", "size"),
                   headliners=("headliner", "nunique"), first_event=("event_dt", "min"), last_event=("event_dt", "max"))
              .sort_values(["capacity", "events"], ascending=[False, False]).reset_index(drop=True))

    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for region, g in tables.items():
            g.to_excel(xw, sheet_name=f"{region} by capacity")
        venues[venues["country"] == "Italy"].to_excel(xw, sheet_name="Italy outdoor venues", index=False)
        venues.to_excel(xw, sheet_name="Europe outdoor venues", index=False)
        methodology(xw, "Outdoor events by venue capacity", common_notes(a, categories, len(df)) + [
            "Outdoor = events whose venue is labelled 'outside' (see above). Capacity = venue_capacity on the event: "
            "the dashboard's curated figure where the venue is a curated arena, otherwise the largest Pollstar-reported "
            "capacity seen at that venue. Venues with no figure fall in 'capacity unknown'.",
            "Bands are by the venue's capacity, so a 4,000-capacity gig at a 60,000 stadium counts as 60,000+ - this is "
            "about where outdoor shows happen, not how many tickets were sold.",
            f"Europe = countryCode in {' '.join(sorted(EUROPE))}.",
            "'% of events with known capacity' excludes the unknown row; '% of all outdoor events' includes it.",
            "The venue sheets list every outdoor venue behind the counts with its capacity and where that came from.",
        ])
        autosize(xw)

    if charts:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for ax, (region, g) in zip(axes, tables.items()):
            y = np.arange(len(order))
            ax.barh(y, g["events"], color=[C_OUTSIDE] * len(CAP_LABELS) + ["#BBBBBB"], edgecolor="black", linewidth=0.3)
            ax.set_yticks(y, order); ax.invert_yaxis()
            for i, (n, v) in enumerate(zip(g["events"], g["venues"])):
                if n:
                    ax.text(n, i, f"  {n:,} events / {v:,} venues", va="center", fontsize=8)
            ax.set_xlim(0, g["events"].max() * 1.35 if g["events"].max() else 1)
            ax.set_xlabel("Outdoor events"); ax.set_title(f"{region}: outdoor events by venue capacity (since {a.since})")
            ax.grid(axis="x", linestyle="--", alpha=0.5)
        fig.tight_layout(); fig.savefig(out.replace(".xlsx", ".png"), dpi=150); plt.close(fig)
    it, eu = tables["Italy"], tables["Europe"]
    log(f"outdoor by capacity: Italy {int(it['events'].sum()):,} outdoor events "
        f"({int(it.loc['capacity unknown', 'events']):,} unknown capacity); Europe {int(eu['events'].sum()):,} "
        f"({int(eu.loc['capacity unknown', 'events']):,} unknown)")


# ------------------------------------------------------------------ main

OUTPUTS = ["venues", "indoor", "legs", "funnel", "seasonality", "capacity"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=paths.DB)
    ap.add_argument("--since", default="2023-01-01", help="ISO date; events on/after this are included")
    ap.add_argument("--categories", nargs="+", default=None,
                    help=f"letters A-E and/or OTHER, or 'all' (default: {' '.join(DEFAULT_CATEGORIES)})")
    ap.add_argument("--out", default=paths.here("since2023"), help="output folder")
    ap.add_argument("--top-cities", type=int, default=15, help="cities on the tour-legs heatmap")
    ap.add_argument("--min-shows", type=int, default=5, help="'substantial tour' threshold in the funnel")
    ap.add_argument("--no-infer", action="store_true", help="indoor/outdoor from database labels only")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, choices=list(OUTPUTS),
                    help="build only these outputs (others are left untouched): " + " ".join(OUTPUTS))
    a = ap.parse_args()

    categories = parse_categories(a.categories)
    os.makedirs(a.out, exist_ok=True)
    log(f"loading events since {a.since} in {categories} from {a.db}")
    df = load_events(a.db, a.since, categories, infer=not a.no_infer)
    log(f"{len(df):,} events loaded ({df['io'].value_counts().to_dict()})")

    charts = not a.no_charts
    p = lambda name: os.path.join(a.out, name)
    builders = {
        "venues":      lambda: italy_top_venues(df, a, categories, p("italy_top_venues.xlsx")),
        "indoor":      lambda: italy_indoor_outdoor(df, a, categories, p("italy_indoor_outdoor_by_city.xlsx"), charts=charts),
        "legs":        lambda: southern_tour_legs(df, a, categories, p("southern_italy_tour_legs.xlsx"), top=a.top_cities, charts=charts),
        "funnel":      lambda: tour_funnel(df, a, categories, p("tour_funnel.xlsx"), min_shows=a.min_shows),
        "seasonality": lambda: seasonality(df, a, categories, p("seasonality.xlsx"), charts=charts),
        "capacity":    lambda: outdoor_by_capacity(df, a, categories, p("outdoor_by_capacity.xlsx"), charts=charts),
    }
    for name in (a.only or OUTPUTS):
        builders[name]()
    log(f"done -> {a.out}")


if __name__ == "__main__":
    main()
