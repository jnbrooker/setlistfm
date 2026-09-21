#!/usr/bin/env python3
"""
Arena event dashboard -- a Streamlit rebuild of arenas-dashboard.xlsx, reading
from setlistfm.db instead of from pasted sheets.

    pip install streamlit pandas
    streamlit run dashboard.py

TWO PAGES, mirroring the workbook's two front-end tabs:

  Arena profile   the single-arena view: location and ownership, capacity,
                  category mix for a year and all-time, observed crowd sizes,
                  a month calendar shaded by the strongest act on each night,
                  and hospitality packages.

  Venue search    the Stadium Search view: filter on geography, venue traits,
                  capacity and activity, then sort.

THE SCOPE TOGGLE
The workbook is fixed to the arenas someone curated by hand. The database holds
175,515 buildings, so the sidebar switches between:

  Dashboard arenas   the curated set, every field human-verified
  All venues         everything we have seen an event at, where capacity and
                     type may be observed from Pollstar or inferred

Numbers will not match the spreadsheet, and that is expected rather than a bug.
The workbook counts events from a Pollstar extract of 153,214 rows; this reads
`events`, which is setlist.fm-derived with Pollstar joined on -- 1.7M rows. The
O2 Arena shows 1,230 events in the sheet and 1,566 here. Ticket figures run the
other way: they exist only where a Pollstar row matched, so many nights show
attendance blank.
"""

import calendar
import datetime as dt
import html
import os
import sqlite3

import pandas as pd
import streamlit as st

import paths
from artist_categories import norm_key

DB_PATH = paths.DB

# (strong, tint) -- the strong shade edges the cell, the tint fills it
CATEGORY_COLOURS = {
    # one distinct hue per tier -- A and B were both purple and read as the
    # same colour at bar and calendar-cell size
    "Category A": ("#4B3FA8", "#EDEBF9"),          # purple
    "Category B": ("#C1502A", "#FAECE7"),          # coral
    "Category C": ("#0F7A5A", "#E6F4EF"),          # teal
    "Category D": ("#2D6FB8", "#E7F0FA"),          # blue
    "Category E": ("#8C8B84", "#F2F2EF"),          # gray
    "Family, Entertainment, Comedy & Other": ("#B0700F", "#FBF1E1"),   # amber
    "Tenant Sporting Event": ("#993556", "#FBEAF0"),                   # dark pink
    "Non-Tenant Sporting Event": ("#D4537E", "#FCF0F4"),               # pink
}
CATEGORY_ORDER = ["Category A", "Category B", "Category C", "Category D",
                  "Category E", "Family, Entertainment, Comedy & Other",
                  "Tenant Sporting Event", "Non-Tenant Sporting Event"]

CSS = """
<style>
:root{--dg-ink:#16323F;--dg-line:#D3DAE0;--dg-bg:#EEF1F3;--dg-muted:#5F6B73;}
[data-testid="stAppViewContainer"]{background:var(--dg-bg);}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.4rem;padding-bottom:2rem;max-width:1500px;}
[data-testid="stSidebar"]{background:#FFFFFF;border-right:1px solid var(--dg-line);}
h1,h2,h3{color:var(--dg-ink);letter-spacing:-.01em;}
h1{font-size:25px!important;font-weight:700!important;margin-bottom:.1rem;}
.dg-bar{background:var(--dg-ink);color:#fff;font-size:12px;font-weight:700;
  letter-spacing:.06em;text-transform:uppercase;padding:6px 10px;border-radius:3px 3px 0 0;
  display:flex;justify-content:space-between;align-items:center;margin-top:14px;}
.dg-bar span{font-weight:400;text-transform:none;letter-spacing:0;opacity:.75;font-size:11px;}
.dg-card{background:#fff;border:1px solid var(--dg-line);border-top:none;
  border-radius:0 0 3px 3px;padding:10px 12px;margin-bottom:4px;}
.dg-kv{font-size:12.5px;line-height:1.75;color:#24323A;}
.dg-kv b{color:var(--dg-muted);font-weight:500;}
.dg-kv .v{float:right;font-weight:600;font-variant-numeric:tabular-nums;}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--dg-line);
  border-radius:3px;padding:8px 12px;}
[data-testid="stMetricValue"]{font-size:21px!important;font-weight:700;color:var(--dg-ink);
  font-variant-numeric:tabular-nums;}
[data-testid="stMetricLabel"] p{font-size:11px!important;text-transform:uppercase;
  letter-spacing:.05em;color:var(--dg-muted)!important;}
[data-testid="stDataFrame"]{border:1px solid var(--dg-line);border-radius:3px;}
[data-testid="stDataFrame"] *{font-size:12px!important;}
table.cal{width:100%;table-layout:fixed;border-collapse:separate;border-spacing:4px;}
table.cal th{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;
  color:var(--dg-muted);font-weight:700;padding:2px 0 4px;text-align:left;}
table.cal td{height:104px;width:14.28%;vertical-align:top;background:#fff;
  border:1px solid var(--dg-line);border-radius:3px;padding:5px 7px;overflow:hidden;}
table.cal td.cal-out{background:#F6F8F9;border-color:#E7EBEE;}
table.cal td.cal-has{border-top-width:3px;}
.cal-num{font-size:11px;font-weight:700;color:var(--dg-ink);
  font-variant-numeric:tabular-nums;display:block;margin-bottom:3px;}
.cal-out .cal-num{color:#AEB7BD;}
.cal-act{font-size:11px;line-height:1.3;color:#22303A;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;}
.cal-more{font-size:10px;color:var(--dg-muted);margin-top:2px;}
.dg-legend{font-size:11px;color:var(--dg-muted);margin-top:6px;}
.dg-bars{padding:10px 12px 6px;}
.dg-row{display:flex;align-items:center;gap:8px;margin-bottom:7px;}
.dg-rl{width:104px;font-size:11.5px;color:#24323A;text-align:right;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.dg-rt{flex:1;background:#EDF0F2;height:15px;border-radius:2px;overflow:hidden;}
.dg-rf{height:100%;border-radius:2px;}
.dg-rv{width:74px;font-size:11.5px;font-weight:700;color:var(--dg-ink);
  font-variant-numeric:tabular-nums;text-align:right;}
.dg-rv span{font-weight:400;color:var(--dg-muted);margin-left:5px;}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div{
  border:1px solid var(--dg-line);background:#fff;border-radius:3px;min-height:38px;}
.dg-pick div[data-baseweb="select"] > div{border:1px solid var(--dg-ink)!important;
  box-shadow:inset 0 0 0 1px rgba(22,50,63,.06);}
.dg-pick [data-testid="stSelectbox"] label{font-size:11px;text-transform:uppercase;
  letter-spacing:.05em;color:var(--dg-muted);}
[data-testid="stSidebar"] [data-testid="stRadio"] label p{font-size:13px;}
.dg-chip{display:inline-block;padding:1px 7px;border-radius:2px;margin-right:6px;
  font-size:10.5px;font-weight:600;}
</style>
"""


# ---------------------------------------------------------------- data access

@st.cache_resource
def get_conn():
    if not os.path.exists(DB_PATH):
        st.error(f"{DB_PATH} not found.")
        st.stop()
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True,
                           check_same_thread=False)


@st.cache_data(show_spinner=False)
def q(sql, params=()):
    return pd.read_sql_query(sql, get_conn(), params=params)


@st.cache_data(show_spinner="Loading venues ...")
def venue_index():
    """One row per building, with the dashboard's extra columns where linked."""
    return q("""
        SELECT v.venue_uid, v.venue, v.city, v.country, v.countryCode,
               v.events, v.first_event, v.last_event, v.capacity,
               v.capacity_source, v.capacity_observed_max,
               v.capacity_observed_typical, v.venue_type, v.outside_inside,
               v.latitude, v.longitude, v.arena_id,
               a.name AS arena_name, a.continent, a.region, a.state,
               a.opened_year, a.renovated_year, a.owner_operator,
               a.also_known_as, a.website, a.address, a.postcode,
               a.arena_type, a.verification_status,
               a.concert_capacity AS verified_capacity,
               a.observed_sport_capacity_max, a.hospitality_capacity
        FROM venues v
        LEFT JOIN arenas a ON a.arena_id = v.arena_id
        ORDER BY v.events DESC""")


@st.cache_data(show_spinner="Summarising activity ...")
def category_mix():
    return q("""
        SELECT venue_uid, COALESCE(NULLIF(category,''),'Uncategorised') AS category,
               COUNT(*) AS events,
               SUM(COALESCE(pollstar_tickets_sold,0)) AS tickets
        FROM events WHERE venue_uid IS NOT NULL
        GROUP BY venue_uid, category""")


@st.cache_data(show_spinner=False)
def venue_events(venue_uid):
    return q("""
        SELECT date_iso, headliner, support, artists, n_artists, category,
               tour, num_songs, pollstar_tickets_sold, pollstar_capacity,
               pollstar_capacity_pct, pollstar_gross_usd, pollstar_promoter
        FROM events WHERE venue_uid = ? ORDER BY date_iso""", (venue_uid,))


@st.cache_data(show_spinner=False)
def hospitality_all():
    return q("""SELECT venue, package, total_capacity, room_capacity, currency,
                       price, vat_incl, price_basis, contact, info, price_quoted
                FROM ref_hospitality ORDER BY venue, package""")


def _label_hits(label, keys):
    """
    Does a workbook hospitality label belong to any of these venues?

    Some rows are labelled with the building AND its tenant -- "Uber Arena /
    ALBA Berlin", "Eisbaeren Berlin / Uber Arena" -- so each side of the slash
    is tested. Matching the whole label alone would drop them.
    """
    text = str(label or "")
    if norm_key(text) in keys:
        return True
    return any(norm_key(part) in keys for part in text.split("/"))


def hospitality_for_set(venue_names):
    """Packages for a whole set of venues, and nothing outside it."""
    df = hospitality_all()
    if df.empty:
        return df
    keys = {norm_key(v) for v in venue_names if v}
    return df[df["venue"].apply(lambda lab: _label_hits(lab, keys))]


def hospitality_for(venue_name):
    """
    Packages for one venue.

    The workbook labels some rows with the building AND its tenant -- "Uber
    Arena / ALBA Berlin", "Eisbaeren Berlin / Uber Arena" -- so a straight name
    match drops them. Each side of the slash is tried instead.

    A caveat the data cannot resolve: the hospitality tab records no city, so
    two buildings sharing a name (3Arena in Dublin and in Stockholm) cannot be
    told apart. Both will show the same packages; the venue label is shown in
    the detail so the mismatch is at least visible.
    """
    return hospitality_for_set([venue_name])


@st.cache_data(show_spinner=False)
def hospitality_venues():
    return q("SELECT DISTINCT venue FROM ref_hospitality ORDER BY venue")


@st.cache_data(show_spinner=False)
def fx_rates():
    return q("SELECT iso, units_per_gbp, basis FROM ref_fx_rates ORDER BY iso")


# -------------------------------------------------------------------- helpers

def fmt_year(v, dash="-"):
    """Years must not get a thousands separator - 2007, never 2,007."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or v in ("", 0):
        return dash
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v)


def fmt(v, dash="-"):
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "" or v == 0:
        return dash
    if isinstance(v, (int, float)) and float(v).is_integer():
        return f"{int(v):,}"
    return str(v)


def bar(title, note=None):
    st.markdown(f'<div class="dg-bar">{html.escape(title)}'
                + (f'<span>{html.escape(str(note))}</span>' if note else "")
                + '</div>', unsafe_allow_html=True)


def kv_card(pairs):
    rows = "".join(
        f'<div class="dg-kv"><b>{html.escape(k)}</b>'
        f'<span class="v">{html.escape(str(v))}</span></div>' for k, v in pairs)
    st.markdown(f'<div class="dg-card">{rows}</div>', unsafe_allow_html=True)


def strongest(acts):
    return min(acts, key=lambda r: CATEGORY_ORDER.index(r["category"])
               if r["category"] in CATEGORY_ORDER else 99)


def render_calendar(year, month_no, by_day):
    """
    A real calendar grid: every cell the same size, shaded by the strongest
    category playing that night, days outside the month greyed rather than
    dropped so the weeks keep their shape.
    """
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    head = "".join(f"<th>{d}</th>" for d in days)
    weeks = []
    for week in calendar.Calendar(firstweekday=0).monthdayscalendar(year, month_no):
        cells = []
        for day in week:
            if day == 0:
                cells.append('<td class="cal-out"></td>')
                continue
            acts = by_day.get(day, [])
            if not acts:
                cells.append(f'<td><span class="cal-num">{day}</span></td>')
                continue
            strong, tint = CATEGORY_COLOURS.get(
                strongest(acts)["category"], ("#8C8B84", "#F2F2EF"))
            names = "".join(
                f'<div class="cal-act">{html.escape(str(a["headliner"]))}</div>'
                for a in acts[:3])
            more = (f'<div class="cal-more">+{len(acts) - 3} more</div>'
                    if len(acts) > 3 else "")
            cells.append(
                f'<td class="cal-has" style="background:{tint};'
                f'border-top-color:{strong}">'
                f'<span class="cal-num">{day}</span>{names}{more}</td>')
        weeks.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<table class="cal"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(weeks)}</tbody></table>')


def category_bars(mix, total):
    """
    A share-of-events bar per category, in that category's own colour.

    Streamlit's bar_chart picks its own colours and cannot key them to the
    category palette, which is the one thing this chart needs to do -- the
    calendar and the legend already use those colours to mean something.
    """
    rows = []
    biggest = max(int(mix["events"].max()), 1)
    for _, r in mix.iterrows():
        strong, tint = CATEGORY_COLOURS.get(r["category"], ("#8C8B84", "#F2F2EF"))
        width = 100.0 * r["events"] / biggest
        label = str(r["category"]).replace(
            "Family, Entertainment, Comedy & Other", "Family / other")
        tickets = (f'{int(r["tickets"]):,} tickets' if r["tickets"]
                   else "no tickets reported")
        rows.append(
            f'<div class="dg-row">'
            f'<div class="dg-rl">{html.escape(label)}</div>'
            f'<div class="dg-rt" title="{tickets}">'
            f'<div class="dg-rf" style="width:{width:.1f}%;background:{strong}"></div>'
            f'</div>'
            f'<div class="dg-rv">{int(r["events"]):,}'
            f'<span>{100.0 * r["events"] / total:.0f}%</span></div></div>')
    return f'<div class="dg-card dg-bars">{"".join(rows)}</div>'


def simple_bars(pairs, colour="#4B3FA8"):
    """
    Horizontal bars for an ordered label/value list.

    Rolled by hand rather than st.bar_chart because the month series has to stay
    in calendar order -- Vega would sort "April" before "January".
    """
    values = [v for _, v in pairs]
    biggest = max(values) if values and max(values) else 1
    rows = []
    for label, value in pairs:
        rows.append(
            f'<div class="dg-row">'
            f'<div class="dg-rl">{html.escape(str(label))}</div>'
            f'<div class="dg-rt"><div class="dg-rf" '
            f'style="width:{100.0 * value / biggest:.1f}%;background:{colour}">'
            f'</div></div>'
            f'<div class="dg-rv">{int(value):,}</div></div>')
    return f'<div class="dg-card dg-bars">{"".join(rows)}</div>'


MONTHS = list(calendar.month_abbr)[1:]


def year_month_panels(year_counts, month_counts, note=""):
    """Events per year and events per month, side by side."""
    c1, c2 = st.columns(2)
    with c1:
        bar("Events per year", note)
        st.markdown(simple_bars(year_counts), unsafe_allow_html=True)
    with c2:
        bar("Events per month", "all years combined")
        st.markdown(simple_bars(month_counts, "#0F7A5A"), unsafe_allow_html=True)


def category_legend(present):
    chips = "".join(
        f'<span class="dg-chip" style="background:{CATEGORY_COLOURS[c][1]};'
        f'color:{CATEGORY_COLOURS[c][0]};border-left:3px solid {CATEGORY_COLOURS[c][0]}">'
        f'{html.escape(c.replace("Family, Entertainment, Comedy & Other", "Family / other"))}'
        f'</span>' for c in CATEGORY_ORDER if c in present)
    st.markdown(f'<div class="dg-legend">{chips}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ the pages

DEFAULT_FROM = "2015-01-01"


def _select_uids(uids):
    """
    Park the chosen venue ids in a temp table so the aggregates can be done in
    SQL. Passing tens of thousands of ids as bound parameters would blow past
    SQLite's variable limit, and pulling the matching events into pandas would
    mean moving a million rows to count them.
    """
    con = get_conn()
    con.execute("CREATE TEMP TABLE IF NOT EXISTS sel (uid TEXT PRIMARY KEY)")
    con.execute("DELETE FROM sel")
    con.executemany("INSERT OR IGNORE INTO sel VALUES (?)",
                    [(u,) for u in uids])
    con.commit()


@st.cache_data(show_spinner="Aggregating ...")
def agg_set(uid_key, d1, d2):
    """
    Every roll-up the aggregated view needs, for one set and date range.

    These deliberately do NOT go through the cached q(): the chosen venues live
    in a temp table, so two different sets produce byte-identical SQL and
    params. q() would key on those and hand back the previous set's numbers --
    which it did, reporting 734 venues for a 12-venue London search. agg_set is
    itself cached on the venue set, so nothing is recomputed needlessly.
    """
    _select_uids(uid_key)
    con = get_conn()

    def qq(sql, params):
        return pd.read_sql_query(sql, con, params=params)
    where = ("FROM events e JOIN sel ON sel.uid = e.venue_uid "
             "WHERE e.date_iso BETWEEN ? AND ?")
    p = (d1, d2)
    head = qq(f"""SELECT COUNT(*) AS events,
                        SUM(COALESCE(e.pollstar_tickets_sold,0)) AS tickets,
                        COUNT(DISTINCT e.venue_uid) AS venues,
                        MIN(e.date_iso) AS first_event,
                        MAX(e.date_iso) AS last_event {where}""", p)
    cats = qq(f"""SELECT COALESCE(NULLIF(e.category,''),'Uncategorised') AS category,
                        COUNT(*) AS events,
                        SUM(COALESCE(e.pollstar_tickets_sold,0)) AS tickets
                 {where} GROUP BY category""", p)
    years = qq(f"""SELECT substr(e.date_iso,1,4) AS year, COUNT(*) AS events
                  {where} GROUP BY year ORDER BY year""", p)
    months = qq(f"""SELECT substr(e.date_iso,6,2) AS month, COUNT(*) AS events
                   {where} GROUP BY month ORDER BY month""", p)
    venues = qq(f"""SELECT e.venue, e.city, e.country, COUNT(*) AS events,
                          SUM(COALESCE(e.pollstar_tickets_sold,0)) AS tickets,
                          MAX(e.venue_capacity) AS capacity
                   {where} GROUP BY e.venue_uid
                   ORDER BY events DESC LIMIT 200""", p)
    artists = qq(f"""SELECT e.headliner, COUNT(*) AS shows,
                           COUNT(DISTINCT e.venue_uid) AS venues,
                           MAX(e.category) AS category
                    {where} GROUP BY e.headliner_norm
                    ORDER BY shows DESC LIMIT 100""", p)
    return head, cats, years, months, venues, artists


def period_controls(ev_min="2000-01-01", ev_max=None):
    """From/to dates, defaulting to the start of 2015."""
    today = dt.date.today()
    hi = dt.date.fromisoformat(ev_max) if ev_max else today
    lo = dt.date.fromisoformat(DEFAULT_FROM)
    c1, c2, _ = st.columns([1, 1, 3])
    d1 = c1.date_input("From", value=lo, key="pfrom",
                       min_value=dt.date(1900, 1, 1), max_value=hi)
    d2 = c2.date_input("To", value=hi, key="pto",
                       min_value=dt.date(1900, 1, 1),
                       max_value=max(hi, today))
    return d1.isoformat(), d2.isoformat()


def hospitality_panel(hosp, scope_note, venue_names=None):
    """
    The package picker and its detail card.

    Only ever offers packages belonging to the venue or venue set it was handed
    -- there is no way to reach another arena's packages from here.
    """
    bar("Hospitality packages", scope_note)
    if hosp.empty:
        known = ", ".join(hospitality_venues()["venue"].head(14))
        st.markdown(
            '<div class="dg-card">No hospitality packages recorded for this '
            f'selection.<br><span style="color:var(--dg-muted);font-size:12px">'
            f'The workbook records packages for: {html.escape(known)}</span>'
            '</div>', unsafe_allow_html=True)
        return

    labels = [f'{r["package"]}  -  {r["venue"]}' for _, r in hosp.iterrows()]
    pick = st.selectbox("Read package", labels, key="hosppick")
    pkg = hosp.iloc[labels.index(pick)]

    price = ""
    if pd.notna(pkg["price"]) and pkg["price"]:
        price = f'{pkg["currency"] or ""} {float(pkg["price"]):,.0f}'.strip()
    if pkg["price_quoted"]:
        price = f'{price}   ({pkg["price_quoted"]})' if price else pkg["price_quoted"]

    d1, d2 = st.columns([2, 3])
    with d1:
        kv_card([("Venue", fmt(pkg["venue"])),
                 ("Package", fmt(pkg["package"])),
                 ("Price", fmt(price)),
                 ("Basis", fmt(pkg["price_basis"])),
                 ("VAT included", fmt(pkg["vat_incl"])),
                 ("Total capacity", fmt(pkg["total_capacity"])),
                 ("Room capacity", fmt(pkg["room_capacity"])),
                 ("Contact", fmt(pkg["contact"]))])
    with d2:
        info = html.escape(str(pkg["info"] or ""))
        st.markdown(
            '<div class="dg-card" style="border-top:1px solid var(--dg-line);'
            'min-height:100%"><div style="font-size:11px;text-transform:'
            'uppercase;letter-spacing:.05em;color:var(--dg-muted);'
            'margin-bottom:6px">Package information</div>'
            f'<div style="font-size:12.5px;line-height:1.6;color:#24323A">'
            f'{info or "Nothing recorded."}</div></div>',
            unsafe_allow_html=True)

    # the workbook sometimes files a package under the resident team rather than
    # the building, so say so rather than leave it looking like a stray arena
    if venue_names and norm_key(pkg["venue"]) not in {norm_key(v) for v in venue_names}:
        st.caption(f'Filed in the workbook as "{pkg["venue"]}" - the same '
                   f'building under a tenant or sponsor name.')

    with st.expander(f"All {len(hosp)} packages in this selection"):
        st.dataframe(hosp.drop(columns=["info"]), use_container_width=True,
                     hide_index=True)


def page_aggregate(vi, scope_label):
    bar("Aggregated summary", f"{len(vi):,} venues in the set")
    d1, d2 = period_controls()
    head, cats, years, months, venues, artists = agg_set(
        tuple(sorted(vi["venue_uid"])), d1, d2)
    h = head.iloc[0]
    if not h["events"]:
        st.markdown('<div class="dg-card">No events in that period for this '
                    'set.</div>', unsafe_allow_html=True)
        return

    m = st.columns(4)
    m[0].metric("Venues with events", f"{int(h['venues']):,}")
    m[1].metric("Events", f"{int(h['events']):,}")
    m[2].metric("Tickets reported", f"{int(h['tickets']):,}")
    m[3].metric("Events per venue", f"{h['events'] / max(h['venues'], 1):,.1f}")

    total = int(cats["events"].sum())
    cats = cats.set_index("category").reindex(
        [c for c in CATEGORY_ORDER if c in set(cats["category"])]
    ).dropna(how="all").reset_index()
    bar("Category mix", f"{d1} to {d2}")
    left, right = st.columns([5, 4])
    with left:
        show = cats.copy()
        show["share"] = 100.0 * show["events"] / total
        st.dataframe(
            show.rename(columns={"category": "Category", "events": "Events",
                                 "tickets": "Tickets", "share": "Share"}),
            use_container_width=True, hide_index=True,
            column_config={
                "Events": st.column_config.NumberColumn(format="%d"),
                "Tickets": st.column_config.NumberColumn(format="%d"),
                "Share": st.column_config.ProgressColumn(
                    format="%.0f%%", min_value=0.0, max_value=100.0)})
    with right:
        st.markdown(category_bars(cats, total), unsafe_allow_html=True)

    mlookup = dict(zip(months["month"], months["events"]))
    year_month_panels(
        list(zip(years["year"], years["events"])),
        [(MONTHS[i - 1], int(mlookup.get(f"{i:02d}", 0))) for i in range(1, 13)],
        f"{d1} to {d2}")

    bar("Busiest venues", "top 200 by events")
    st.dataframe(venues, use_container_width=True, hide_index=True,
                 column_config={
                     "events": st.column_config.NumberColumn(format="%d"),
                     "tickets": st.column_config.NumberColumn(format="%d"),
                     "capacity": st.column_config.NumberColumn(format="%d")})

    bar("Most frequent headliners", "top 100 across the set")
    st.dataframe(artists, use_container_width=True, hide_index=True,
                 column_config={
                     "shows": st.column_config.NumberColumn(format="%d"),
                     "venues": st.column_config.NumberColumn(format="%d")})

    venue_names = list(vi["venue"].dropna())
    hospitality_panel(hospitality_for_set(venue_names),
                      f"across the {len(vi):,} venues in the set", venue_names)


def page_profile(vi, scope_label, mode):
    if mode == "Filtered set":
        page_aggregate(vi, scope_label)
        return

    names = vi.assign(label=vi["venue"] + " - " + vi["city"].fillna(""))
    default = 0
    o2 = names.index[names["venue"].str.lower() == "the o2 arena"]
    if len(o2):
        default = int(names.index.get_loc(o2[0]))
    # ?venue=<name> deep-links straight to a venue, so a profile can be shared
    wanted = st.query_params.get("venue")
    if wanted:
        hit = names.index[names["venue"].str.lower() == str(wanted).lower()]
        if len(hit):
            default = int(names.index.get_loc(hit[0]))
    bar("Venue", f"{len(names):,} in {scope_label.lower()}")
    st.markdown('<div class="dg-card dg-pick">', unsafe_allow_html=True)
    choice = st.selectbox("Choose a venue", names["label"], index=default,
                          key="venuepick")
    st.markdown("</div>", unsafe_allow_html=True)
    row = names[names["label"] == choice].iloc[0]

    ev = venue_events(row["venue_uid"])
    if ev.empty:
        st.info("No events on record for this venue.")
        return
    full = ev
    d1, d2 = period_controls(ev_max=ev["date_iso"].max())
    ev = ev[(ev["date_iso"] >= d1) & (ev["date_iso"] <= d2)]
    if ev.empty:
        st.markdown('<div class="dg-card">No events in that period. Widen the '
                    f'dates - this venue has {len(full):,} events from '
                    f'{full["date_iso"].min()}.</div>', unsafe_allow_html=True)
        return
    ev["year"] = ev["date_iso"].str.slice(0, 4)

    m = st.columns(4)
    m[0].metric("Events in period", f"{len(ev):,}")
    m[1].metric("Event history", f"{ev['year'].min()}-{ev['year'].max()}")
    m[2].metric("Tickets reported",
                f"{int(ev['pollstar_tickets_sold'].fillna(0).sum()):,}")
    cap = row.get("capacity")
    sold = ev["pollstar_tickets_sold"].dropna()
    m[3].metric("Capacity used",
                f"{sold.median() / cap:.0%}" if cap and len(sold) else "-")

    c1, c2, c3 = st.columns(3)
    with c1:
        bar("Location")
        kv_card([("City", fmt(row.get("city"))),
                 ("Region / state", fmt(row.get("region"))),
                 ("Country", fmt(row.get("country"))),
                 ("Continent", fmt(row.get("continent"))),
                 ("Address", fmt(row.get("address"))),
                 ("Postcode", fmt(row.get("postcode")))])
    with c2:
        bar("Venue & ownership")
        kv_card([("Venue type", fmt(row.get("venue_type"))),
                 ("Indoor / outdoor", fmt(row.get("outside_inside"))),
                 ("Opened", fmt_year(row.get("opened_year"))),
                 ("Renovated", fmt_year(row.get("renovated_year"))),
                 ("Owner / operator", fmt(row.get("owner_operator"))),
                 ("Also known as", fmt(row.get("also_known_as")))])
    with c3:
        bar("Capacity", row.get("capacity_source") or "not recorded")
        kv_card([("Concert capacity", fmt(row.get("capacity"))),
                 ("Verified figure", fmt(row.get("verified_capacity"))),
                 ("Observed biggest", fmt(row.get("capacity_observed_max"))),
                 ("Observed typical", fmt(row.get("capacity_observed_typical"))),
                 ("Sport capacity", fmt(row.get("observed_sport_capacity_max"))),
                 ("Hospitality", fmt(row.get("hospitality_capacity")))])

    years = ["All-time"] + sorted(ev["year"].unique(), reverse=True)
    bar("Category mix", choice)
    year = st.selectbox("Period", years, index=0, key="mixyear",
                        label_visibility="collapsed")
    scope_ev = ev if year == "All-time" else ev[ev["year"] == year]
    mix = (scope_ev.groupby("category")
           .agg(events=("date_iso", "size"),
                tickets=("pollstar_tickets_sold",
                         lambda s: int(s.fillna(0).sum())))
           .reindex([c for c in CATEGORY_ORDER if c in set(scope_ev["category"])])
           .dropna(how="all").reset_index())
    if not mix.empty:
        total = mix["events"].sum()
        # ProgressColumn printf-formats the raw value, so 0.33 would render as
        # "0%" -- give it 0-100 and format that
        mix["share"] = 100.0 * mix["events"] / total
        left, right = st.columns([5, 4])
        with left:
            st.dataframe(
                mix.rename(columns={"category": "Category", "events": "Events",
                                    "tickets": "Tickets", "share": "Share"}),
                use_container_width=True, hide_index=True,
                column_config={
                    "Events": st.column_config.NumberColumn(format="%d"),
                    "Tickets": st.column_config.NumberColumn(format="%d"),
                    "Share": st.column_config.ProgressColumn(
                        format="%.0f%%", min_value=0.0, max_value=100.0),
                })
        with right:
            st.markdown(category_bars(mix, total), unsafe_allow_html=True)

    yc = ev["year"].value_counts().sort_index()
    mc = ev["date_iso"].str.slice(5, 7).value_counts()
    year_month_panels(
        [(y, n) for y, n in yc.items()],
        [(MONTHS[i - 1], int(mc.get(f"{i:02d}", 0))) for i in range(1, 13)],
        f"{d1} to {d2}")

    cal_years = sorted(ev["year"].unique(), reverse=True)
    bar("Month calendar")
    p1, p2, _ = st.columns([1, 1, 3])
    cy = p1.selectbox("Year", cal_years, key="caly")
    cm = p2.selectbox("Month", list(calendar.month_name)[1:],
                      index=dt.date.today().month - 1, key="calm")
    month_no = list(calendar.month_name).index(cm)
    month = ev[ev["date_iso"].str.startswith(f"{cy}-{month_no:02d}")]
    if month.empty:
        st.markdown('<div class="dg-card">No events that month.</div>',
                    unsafe_allow_html=True)
    else:
        a, b, c = st.columns(3)
        a.metric("Events this month", len(month))
        b.metric("Tickets sold",
                 f"{int(month['pollstar_tickets_sold'].fillna(0).sum()):,}")
        busiest = dt.date.fromisoformat(month.groupby("date_iso").size().idxmax())
        c.metric("Busiest day", busiest.strftime("%a %d %b"))
        by_day = {}
        for _, r in month.iterrows():
            by_day.setdefault(int(r["date_iso"][8:10]), []).append(r)
        st.markdown(render_calendar(int(cy), month_no, by_day),
                    unsafe_allow_html=True)
        category_legend(set(month["category"]))

    hospitality_panel(hospitality_for(row["venue"]),
                      row["venue"], [row["venue"]])

    with st.expander(f"All {len(ev):,} events"):
        st.dataframe(ev.drop(columns=["year"]), use_container_width=True,
                     hide_index=True)


def page_search(vi, mix, scope_label):
    df = vi.copy()
    bar("Filters", "blank or Any is ignored; filters combine with AND")
    g1, g2, g3 = st.columns(3)
    with g1:
        # each level narrows the options below it, so the city list stays short
        # enough to pick from instead of having to be typed blind
        cont = st.selectbox("Continent", ["Any"] + sorted(
            x for x in df["continent"].dropna().unique() if x))
        pool = df if cont == "Any" else df[df["continent"] == cont]
        country = st.selectbox("Country", ["Any"] + sorted(
            x for x in pool["country"].dropna().unique() if x))
        if country != "Any":
            pool = pool[pool["country"] == country]
        cities = sorted(x for x in pool["city"].dropna().unique() if x)
        city = st.selectbox(f"City ({len(cities):,})", ["Any"] + cities)
        name = st.text_input("Venue name contains")
    with g2:
        vtype = st.selectbox("Venue type", ["Any"] + sorted(
            x for x in df["venue_type"].dropna().unique() if x))
        io = st.selectbox("Indoor / outdoor", ["Any"] + sorted(
            x for x in df["outside_inside"].dropna().unique() if x))
        opened_from = st.number_input("Opened from (year)", 0, 2100, 0, step=1)
        opened_to = st.number_input("Opened to (year)", 0, 2100, 0, step=1)
    with g3:
        cap_min = st.number_input("Concert capacity min", 0, 500000, 0, step=1000)
        cap_max = st.number_input("Concert capacity max", 0, 500000, 0, step=1000)
        ev_min = st.number_input("Events on record min", 0, 100000, 0, step=10)
        dom = st.selectbox("Dominant category", ["Any"] + CATEGORY_ORDER)

    if cont != "Any":
        df = df[df["continent"] == cont]
    if country != "Any":
        df = df[df["country"] == country]
    if city != "Any":
        df = df[df["city"] == city]
    if name:
        df = df[df["venue"].fillna("").str.contains(name, case=False)]
    if vtype != "Any":
        df = df[df["venue_type"] == vtype]
    if io != "Any":
        df = df[df["outside_inside"] == io]
    if opened_from:
        df = df[df["opened_year"].fillna(0) >= opened_from]
    if opened_to:
        df = df[(df["opened_year"].fillna(0) <= opened_to)
                & (df["opened_year"].fillna(0) > 0)]
    if cap_min:
        df = df[df["capacity"].fillna(0) >= cap_min]
    if cap_max:
        df = df[(df["capacity"].fillna(0) <= cap_max)
                & (df["capacity"].fillna(0) > 0)]
    if ev_min:
        df = df[df["events"] >= ev_min]
    if dom != "Any":
        top = (mix.sort_values("events", ascending=False)
               .drop_duplicates("venue_uid").set_index("venue_uid")["category"])
        df = df[df["venue_uid"].map(top) == dom]

    tick = mix.groupby("venue_uid")["tickets"].sum()
    df = df.assign(tickets=df["venue_uid"].map(tick).fillna(0).astype(int))

    o1, o2, o3 = st.columns(3)
    sort_by = o1.selectbox("Sort by", ["Concert capacity", "Events on record",
                                       "Tickets sold", "Venue name"])
    direction = o2.selectbox("Direction", ["Largest first", "Smallest first"])
    top_n = o3.number_input("Show top", 1, 5000, 200, step=50)
    col = {"Concert capacity": "capacity", "Events on record": "events",
           "Tickets sold": "tickets", "Venue name": "venue"}[sort_by]
    df = df.sort_values(col, ascending=(direction == "Smallest first"),
                        na_position="last")

    # the whole match, before "Show top" trims what is displayed -- otherwise an
    # aggregate built from this set would silently cover only the visible rows
    st.session_state["search_uids"] = set(df["venue_uid"])
    matched = len(df)
    df = df.head(int(top_n))
    bar("Results", f"{matched:,} venues match"
        + (f", showing {len(df):,}" if matched > len(df) else "")
        + f" - from {scope_label.lower()}")
    st.caption(f"All {matched:,} matches become the working set for the Arena "
               "profile page. 'Show top' only limits this table.")
    st.dataframe(
        df[["venue", "city", "country", "capacity", "capacity_source",
            "venue_type", "outside_inside", "opened_year", "events",
            "tickets", "first_event", "last_event"]],
        use_container_width=True, hide_index=True,
        column_config={
            "opened_year": st.column_config.NumberColumn("opened", format="%d"),
            "capacity": st.column_config.NumberColumn(format="%d"),
            "events": st.column_config.NumberColumn(format="%d"),
            "tickets": st.column_config.NumberColumn(format="%d"),
        })

    mapped = df.dropna(subset=["latitude", "longitude"]).copy()
    if not mapped.empty:
        mapped["lat"] = pd.to_numeric(mapped["latitude"], errors="coerce")
        mapped["lon"] = pd.to_numeric(mapped["longitude"], errors="coerce")
        mapped = mapped.dropna(subset=["lat", "lon"])
        if not mapped.empty:
            bar("Map", f"{len(mapped):,} with coordinates")
            st.map(mapped[["lat", "lon"]], size=20, color="#4B3FA8")


# ------------------------------------------------------------------------ main

def main():
    st.set_page_config(page_title="Arena event dashboard", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    vi = venue_index()
    mix = category_mix()

    mode = "Single arena"
    with st.sidebar:
        st.markdown("### Arena events")
        scope = st.radio(
            "Scope", ["Dashboard arenas", "All venues"],
            help="The workbook covers the curated arenas. The database holds "
                 "every building we have seen an event at.")
        if scope == "Dashboard arenas":
            vi = vi[vi["arena_id"].notna()]
        page = st.radio("Page", ["Arena profile", "Venue search"])
        if page == "Arena profile":
            mode = st.radio(
                "Mode", ["Single arena", "Filtered set"], key="profile_mode",
                help="Single arena profiles one venue. Filtered set summarises "
                     "every venue the last search matched.")

        # whatever the last search matched is the working set from then on --
        # no separate opt-in toggle, since choosing Filtered set already says so
        picked = st.session_state.get("search_uids")
        if picked:
            hit = vi[vi["venue_uid"].isin(picked)]
            if not hit.empty and len(hit) < len(vi):
                vi = hit
                scope = "search results"
                st.caption(f"Working set: {len(vi):,} venues from your search")
                if st.button("Clear search", use_container_width=True):
                    st.session_state.pop("search_uids", None)
                    st.rerun()
            else:
                st.caption(f"{len(vi):,} venues in scope")
        else:
            st.caption(f"{len(vi):,} venues in scope")
        st.divider()
        st.caption(
            "Counts come from `events` (setlist.fm derived, Pollstar joined "
            "on), not the workbook's Pollstar extract, so they will not match "
            "the spreadsheet. Tickets appear only where a Pollstar row matched.")
        with st.expander("FX rates"):
            st.dataframe(fx_rates(), use_container_width=True, hide_index=True)

    st.title("Arena event dashboard")
    if page == "Arena profile":
        page_profile(vi, scope, mode)
    else:
        page_search(vi, mix, scope)


if __name__ == "__main__":
    main()
