#!/usr/bin/env python3
"""
Phase 3: the derived events table.

Collapses `setlists` (one row per artist per night) into `events` (one row per
show), attaches the artist categorisation, and adds the tour routing columns.
Rebuilt from scratch on every run, so it is always a pure function of the raw
data. `setlists` is never touched.

WHAT IT DOES

1. Groups by date + venue, so a headliner and its support acts become one event.
   Venue identity is `venue_id` where we have it and `venue|city` where we don't
   -- 1.1M of the rows came from the old CSVs and carry no venue_id at all.

2. Picks the headliner as the strongest act on the bill: best category first,
   then most songs played, then the longer career, then name for determinism.
   The event then takes that act's category. This is deliberate -- it is the
   dashboard's own rule for mixed bills, which re-tiers an event by the acts
   actually on it rather than by the billing string.

3. Adds previous_city / next_city per (headliner, tour), ordered by date, with
   "First Show" / "Final Show" at the ends and both blank when there is no tour
   name -- byte-identical to the with_cities_setlistfm.csv convention.

4. Attaches Pollstar box-office data (tickets sold, capacity, gross, promoter,
   ticket prices) where a Pollstar row can be matched to the event. Most events
   will have none -- Pollstar covers reported shows only -- which is expected.

USAGE
    python build_events.py load-pollstar   # pollstar-data.xlsx -> pollstar_events
    python build_events.py build           # rebuild events, incl. the Pollstar join
    python build_events.py match-pollstar  # redo just the join
    python build_events.py status          # what's in it
    python build_events.py export          # -> events.csv

Run `python artist_categories.py refresh` first: the category columns come from
artist_categories, and events built before it is populated will be uncategorised.
"""

import argparse
import csv
import datetime as dt
import os
import re
import sqlite3
import sys

# the same name normalisation the categoriser uses, so both sides of every
# join agree on what "Motley Crue" and "Gov't Mule" reduce to
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from artist_categories import norm_key  # noqa: E402

# resolved against the project folder, not the working directory -- see paths.py
DB_PATH = paths.DB
DEFAULT_EXPORT = paths.EVENTS_CSV
POLLSTAR_WORKBOOK = paths.POLLSTAR_WORKBOOK
ARENA_WORKBOOK = paths.ARENA_WORKBOOK
ARENA_SHEET = "Arena Data"
ARENA_ALIASES_MANUAL = paths.ARENA_ALIASES_MANUAL

# Arena Data columns that should be stored as numbers rather than text.
ARENA_NUMERIC = {
    "concert_capacity", "concert_capacity_seated", "concert_capacity_standing",
    "observed_concert_capacity_max", "observed_concert_capacity_typical",
    "observed_concert_bookings", "sporting_capacity", "seating_capacity",
    "observed_sport_capacity_max", "observed_sport_capacity_typical",
    "observed_sport_fixtures", "hospitality_capacity", "max_capacity",
    "opened_year", "renovated_year", "latitude", "longitude",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id          TEXT PRIMARY KEY,   -- date_iso | venue_key
    date_iso          TEXT,
    eventDate         TEXT,
    venue_key         TEXT,
    venue             TEXT,
    venue_id          TEXT,
    city              TEXT,
    state             TEXT,
    stateCode         TEXT,
    country           TEXT,
    countryCode       TEXT,
    latitude          TEXT,
    longitude         TEXT,
    headliner         TEXT,
    headliner_key     TEXT,
    support           TEXT,      -- ' + ' joined, billing order after the headliner
    artists           TEXT,      -- every act, headliner first
    n_artists         INTEGER,
    tour              TEXT,
    num_songs         INTEGER,   -- summed across the bill
    event_type        TEXT,
    category          TEXT,      -- the headliner's, i.e. the best on the bill
    monthly_listeners INTEGER,
    pollstar_rank     INTEGER,
    previous_city     TEXT,
    next_city         TEXT,
    setlist_ids       TEXT,
    setlist_urls      TEXT,
    built_at          TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_date     ON events(date_iso);
CREATE INDEX IF NOT EXISTS ix_events_venue    ON events(venue_key);
CREATE INDEX IF NOT EXISTS ix_events_headline ON events(headliner_key, tour, date_iso);
CREATE INDEX IF NOT EXISTS ix_events_category ON events(category);

-- Pollstar box-office rows. One row can cover a RUN of shows (start..end with
-- "Number of Shows" > 1), so totals describe the run and the averages describe
-- a single night. Matching is therefore a date-range test, not an equality.
CREATE TABLE IF NOT EXISTS pollstar_events (
    id                INTEGER PRIMARY KEY,
    start_iso         TEXT,
    end_iso           TEXT,
    n_shows           INTEGER,
    headliner         TEXT,
    headliner_norm    TEXT,
    support           TEXT,
    venue_id          TEXT,
    venue             TEXT,
    venue_norm        TEXT,
    venue_type        TEXT,
    city              TEXT,
    city_norm         TEXT,
    state             TEXT,
    zip               TEXT,
    country           TEXT,
    currency          TEXT,
    promoter          TEXT,
    genre             TEXT,
    market            TEXT,
    total_tickets     INTEGER,
    avg_tickets       INTEGER,
    total_gross_usd   REAL,
    avg_gross_usd     REAL,
    avg_capacity      INTEGER,
    total_capacity    INTEGER,
    avg_capacity_sold REAL,
    price_min         REAL,
    price_max         REAL,
    price_avg         REAL,
    arena_id          TEXT,
    loaded_at         TEXT
);
CREATE INDEX IF NOT EXISTS ix_ps_headliner ON pollstar_events(headliner_norm, start_iso);
CREATE INDEX IF NOT EXISTS ix_ps_venue     ON pollstar_events(venue_norm);

-- Every spelling of a venue we know, mapped to one arena. Arenas get renamed
-- constantly (Manchester Arena -> MEN Arena -> AO Arena), so matching venues on
-- their name alone silently loses a decade of a building's history.
CREATE TABLE IF NOT EXISTS arena_aliases (
    alias_norm  TEXT NOT NULL,
    arena_id    TEXT NOT NULL,
    alias       TEXT,
    source      TEXT,          -- name | also_known_as | matched_venue | manual
    city_norm   TEXT,
    country     TEXT,
    PRIMARY KEY (alias_norm, arena_id)
);
CREATE INDEX IF NOT EXISTS ix_arena_alias ON arena_aliases(alias_norm);
CREATE INDEX IF NOT EXISTS ix_arena_alias_id ON arena_aliases(arena_id);

-- Every venue we have ever seen an event at, with the best capacity / type /
-- indoor-outdoor we can assemble for it, and WHERE each of those came from.
--
-- This is deliberately NOT the `arenas` table: `load-arenas` drops and rebuilds
-- that one from the dashboard sheet, so anything derived or scraped into it
-- would be destroyed on the next load. The dashboard's 676 hand-verified rows
-- stay authoritative here -- derived values only ever fill a gap, never
-- overwrite one -- and `needs_review` marks what a future scrape should target.
CREATE TABLE IF NOT EXISTS venues (
    venue_uid            TEXT PRIMARY KEY,   -- venue_norm|city_norm|countryCode
    venue                TEXT,
    venue_norm           TEXT,
    city                 TEXT,
    city_norm            TEXT,
    country              TEXT,
    countryCode          TEXT,
    latitude             TEXT,
    longitude            TEXT,
    venue_keys           TEXT,               -- the raw event keys that merged here
    n_keys               INTEGER,            -- how many identities this building had
    events               INTEGER,
    first_event          TEXT,
    last_event           TEXT,
    arena_id             TEXT,               -- dashboard arena, when known
    capacity             INTEGER,            -- best available
    capacity_source      TEXT,               -- dashboard | pollstar | web | manual
    capacity_observed_max     INTEGER,       -- from Pollstar box office
    capacity_observed_typical INTEGER,
    capacity_samples     INTEGER,
    venue_type           TEXT,
    venue_type_source    TEXT,
    outside_inside       TEXT,
    outside_inside_source TEXT,
    pollstar_venue_id    TEXT,
    needs_review         INTEGER DEFAULT 0,  -- worth enriching from the web
    updated_at           TEXT
);
CREATE INDEX IF NOT EXISTS ix_venues_events ON venues(events DESC);
CREATE INDEX IF NOT EXISTS ix_venues_review ON venues(needs_review, events DESC);
CREATE INDEX IF NOT EXISTS ix_venues_norm   ON venues(venue_norm);

-- Hospitality packages, straight from the dashboard workbook. Small and
-- hand-maintained, so it is simply mirrored rather than derived.
CREATE TABLE IF NOT EXISTS ref_hospitality (
    id             INTEGER PRIMARY KEY,
    type           TEXT,
    venue          TEXT,
    venue_norm     TEXT,
    package        TEXT,
    total_capacity TEXT,
    room_capacity  TEXT,
    currency       TEXT,
    price          REAL,
    vat_incl       TEXT,
    price_basis    TEXT,
    contact        TEXT,
    info           TEXT,
    price_quoted   TEXT,
    loaded_at      TEXT
);
CREATE INDEX IF NOT EXISTS ix_hosp_venue ON ref_hospitality(venue_norm);

-- One spot snapshot of GBP conversion rates, for comparing prices.
CREATE TABLE IF NOT EXISTS ref_fx_rates (
    iso            TEXT PRIMARY KEY,
    units_per_gbp  REAL,
    basis          TEXT,
    loaded_at      TEXT
);
"""

# columns appended to `events` by the Pollstar match. Kept separate from the
# core schema so a rebuild can add them to an events table that predates them.
POLLSTAR_COLUMNS = [
    ("headliner_norm", "TEXT"), ("venue_norm", "TEXT"), ("city_norm", "TEXT"),
    ("arena_id", "TEXT"), ("arena_name", "TEXT"), ("arena_capacity", "INTEGER"),
    ("arena_type", "TEXT"), ("arena_outside_inside", "TEXT"),
    ("pollstar_id", "INTEGER"), ("pollstar_match", "TEXT"),
    ("pollstar_date_offset", "INTEGER"),
    ("pollstar_headliner", "TEXT"), ("pollstar_support", "TEXT"),
    ("pollstar_venue", "TEXT"), ("pollstar_venue_id", "TEXT"),
    ("pollstar_venue_type", "TEXT"), ("pollstar_market", "TEXT"),
    ("pollstar_genre", "TEXT"), ("pollstar_promoter", "TEXT"),
    ("pollstar_run_shows", "INTEGER"),
    ("pollstar_tickets_sold", "INTEGER"),    # per show (Pollstar's average)
    ("pollstar_capacity", "INTEGER"),        # per show
    ("pollstar_capacity_pct", "REAL"),
    ("pollstar_gross_usd", "REAL"),          # per show
    ("pollstar_run_tickets", "INTEGER"),     # whole run, for reference
    ("pollstar_run_gross_usd", "REAL"),
    # best-available venue facts, which may be derived; the arena_* columns
    # above stay reserved for the dashboard's verified rows
    ("venue_uid", "TEXT"),
    ("venue_capacity", "INTEGER"), ("venue_capacity_source", "TEXT"),
    ("venue_type", "TEXT"), ("venue_outside_inside", "TEXT"),
    ("pollstar_price_min", "REAL"), ("pollstar_price_max", "REAL"),
    ("pollstar_price_avg", "REAL"),
    ("pollstar_start", "TEXT"), ("pollstar_end", "TEXT"),
]

# How the raw rows are keyed into events and ranked within one.
VENUE_KEY = "COALESCE(NULLIF(sl.venue_id,''), sl.venue || '|' || sl.city)"

CATEGORY_RANK = """
    CASE c.category
        WHEN 'Category A' THEN 1
        WHEN 'Category B' THEN 2
        WHEN 'Category C' THEN 3
        WHEN 'Category D' THEN 4
        WHEN 'Category E' THEN 5
        ELSE 6
    END"""

BUILD = f"""
INSERT INTO events (
    event_id, date_iso, eventDate, venue_key, venue, venue_id, city, state,
    stateCode, country, countryCode, latitude, longitude, headliner,
    headliner_key, support, artists, n_artists, tour, num_songs, event_type,
    category, monthly_listeners, pollstar_rank, setlist_ids, setlist_urls,
    built_at)
WITH ranked AS (
    SELECT
        sl.date_iso || '|' || {VENUE_KEY}        AS event_id,
        {VENUE_KEY}                              AS venue_key,
        sl.date_iso, sl.eventDate, sl.venue, sl.venue_id, sl.city, sl.state,
        sl.stateCode, sl.country, sl.countryCode, sl.latitude, sl.longitude,
        sl.artist, sl.tour, sl.num_songs, sl.setlist_id, sl.setlist_url,
        LOWER(TRIM(sl.artist))                   AS artist_key,
        c.event_type, c.category, c.monthly_listeners, c.pollstar_rank,
        ROW_NUMBER() OVER (
            PARTITION BY sl.date_iso || '|' || {VENUE_KEY}
            ORDER BY {CATEGORY_RANK},
                     COALESCE(sl.num_songs, 0) DESC,
                     COALESCE(c.shows, 0) DESC,
                     sl.artist
        ) AS rn
    FROM setlists sl
    LEFT JOIN artist_categories c ON c.artist_key = LOWER(TRIM(sl.artist))
    WHERE TRIM(COALESCE(sl.date_iso, '')) <> ''
      AND TRIM(COALESCE(sl.artist, ''))   <> ''
)
SELECT
    event_id,
    MIN(date_iso), MIN(eventDate), MIN(venue_key),
    MAX(CASE WHEN rn = 1 THEN venue END),
    MAX(CASE WHEN rn = 1 THEN venue_id END),
    MAX(CASE WHEN rn = 1 THEN city END),
    MAX(CASE WHEN rn = 1 THEN state END),
    MAX(CASE WHEN rn = 1 THEN stateCode END),
    MAX(CASE WHEN rn = 1 THEN country END),
    MAX(CASE WHEN rn = 1 THEN countryCode END),
    MAX(CASE WHEN rn = 1 THEN latitude END),
    MAX(CASE WHEN rn = 1 THEN longitude END),
    MAX(CASE WHEN rn = 1 THEN artist END),
    MAX(CASE WHEN rn = 1 THEN artist_key END),
    group_concat(CASE WHEN rn > 1 THEN artist END, ' + ' ORDER BY rn),
    group_concat(artist, ' + ' ORDER BY rn),
    COUNT(*),
    -- the headliner's tour if it has one, otherwise any tour named on the bill
    COALESCE(NULLIF(MAX(CASE WHEN rn = 1 THEN tour END), ''),
             MAX(NULLIF(tour, '')), ''),
    SUM(COALESCE(num_songs, 0)),
    MAX(CASE WHEN rn = 1 THEN event_type END),
    MAX(CASE WHEN rn = 1 THEN category END),
    MAX(CASE WHEN rn = 1 THEN monthly_listeners END),
    MAX(CASE WHEN rn = 1 THEN pollstar_rank END),
    group_concat(setlist_id, ' | ' ORDER BY rn),
    group_concat(setlist_url, ' | ' ORDER BY rn),
    ?
FROM ranked
GROUP BY event_id
"""

# previous/next city, per (headliner, tour), ordered by date.
ROUTES = """
WITH r AS (
    SELECT event_id,
           LAG(city)  OVER w AS prev_city,
           LEAD(city) OVER w AS next_city
    FROM events
    WHERE TRIM(COALESCE(tour, '')) <> ''
    WINDOW w AS (PARTITION BY headliner_key, tour ORDER BY date_iso, event_id)
)
UPDATE events
   SET previous_city = COALESCE(r.prev_city, 'First Show'),
       next_city     = COALESCE(r.next_city, 'Final Show')
  FROM r
 WHERE r.event_id = events.event_id
"""

EXPORT_COLUMNS = [
    "date_iso", "eventDate", "headliner", "support", "artists", "n_artists",
    "tour", "venue", "city", "state", "stateCode", "country", "countryCode",
    "latitude", "longitude", "num_songs", "event_type", "category",
    "monthly_listeners", "pollstar_rank", "previous_city", "next_city",
    "pollstar_match", "pollstar_headliner", "pollstar_support", "pollstar_venue",
    "pollstar_venue_type", "pollstar_market", "pollstar_genre",
    "pollstar_promoter", "pollstar_run_shows", "pollstar_tickets_sold",
    "pollstar_capacity", "pollstar_capacity_pct", "pollstar_gross_usd",
    "pollstar_price_min", "pollstar_price_max", "pollstar_price_avg",
    "event_id", "setlist_urls",
]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


try:
    from tqdm import tqdm as _tqdm
except ImportError:            # tqdm is nice-to-have, never required
    _tqdm = None


def progress(iterable, desc, total=None, unit="rows"):
    """tqdm when it is installed, a quiet periodic log line when it is not."""
    if _tqdm is not None:
        return _tqdm(iterable, desc=desc, total=total, unit=unit,
                     unit_scale=True, file=sys.stderr, leave=False,
                     dynamic_ncols=True, mininterval=0.5)

    def fallback():
        n = 0
        for item in iterable:
            yield item
            n += 1
            if n % 50000 == 0:
                log(f"   {desc}: {n:,}"
                    + (f" / {total:,}" if total else ""))
    return fallback()


def sanitise(name):
    """Excel header -> safe SQLite column name."""
    col = re.sub(r"[^0-9a-zA-Z]+", "_", str(name or "").strip().lower()).strip("_")
    return col or "col"


def utcnow():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - run this from the jambase folder.")
    conn = sqlite3.connect(path, timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.create_function("norm_key", 1, norm_key, deterministic=True)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def scalar(conn, sql, params=()):
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def _load_hospitality(wb, conn, now):
    """Mirror the Hospitality Information tab."""
    if "Hospitality Information" not in wb.sheetnames:
        log("   no 'Hospitality Information' tab - skipping")
        return
    ws = wb["Hospitality Information"]
    it = ws.iter_rows(values_only=True)
    hm = {sanitise(c): i for i, c in enumerate(next(it)) if c}

    def g(r, *names):
        for n in names:
            if n in hm and hm[n] < len(r):
                return r[hm[n]]
        return None

    rows = []
    for r in it:
        venue = _txt(g(r, "team_arena_venue", "venue"))
        if not venue:
            continue
        rows.append((_txt(g(r, "type")), venue, norm_key(venue),
                     _txt(g(r, "package_name", "package")),
                     _txt(g(r, "total_capacity")), _txt(g(r, "room_capacity")),
                     _txt(g(r, "currency")), _num(g(r, "price")),
                     _txt(g(r, "vat_incl")), _txt(g(r, "price_basis")),
                     _txt(g(r, "contact")), _txt(g(r, "package_information")),
                     _txt(g(r, "price_as_quoted")), now))
    conn.execute("DELETE FROM ref_hospitality")
    conn.executemany("""INSERT INTO ref_hospitality (type, venue, venue_norm, package,
        total_capacity, room_capacity, currency, price, vat_incl, price_basis,
        contact, info, price_quoted, loaded_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    conn.commit()
    n_v = scalar(conn, "SELECT COUNT(DISTINCT venue_norm) FROM ref_hospitality") or 0
    log(f"   ref_hospitality: {len(rows):,} packages across {n_v:,} venues")


def _load_fx(wb, conn, now):
    """
    Mirror the FX Rates tab.

    The header sits a few rows down under some prose, so the ISO/rate columns
    are found by scanning for the header row rather than assumed at row 1.
    """
    if "FX Rates" not in wb.sheetnames:
        log("   no 'FX Rates' tab - skipping")
        return
    ws = wb["FX Rates"]
    rows, header = [], None
    for r in ws.iter_rows(values_only=True):
        cells = [_txt(c) for c in r[:4]]
        if header is None:
            if cells and cells[0].lower() == "iso":
                header = True
            continue
        iso = cells[0].strip().upper()
        rate = _num(r[1] if len(r) > 1 else None)
        if len(iso) == 3 and iso.isalpha() and rate:
            rows.append((iso, rate, cells[2] if len(cells) > 2 else "", now))
    if not rows:
        log("   FX Rates: no ISO/rate rows found - skipping")
        return
    conn.execute("DELETE FROM ref_fx_rates")
    conn.executemany("INSERT OR REPLACE INTO ref_fx_rates "
                     "(iso, units_per_gbp, basis, loaded_at) VALUES (?,?,?,?)", rows)
    conn.commit()
    log(f"   ref_fx_rates: {len(rows):,} currencies")


def cmd_load_arenas(args, conn):
    """
    Load the dashboard's Arena Data tab into `arenas`, then build the venue
    name conversion table from it.

    The table is dropped and recreated from the sheet's own headers, so adding a
    column in the workbook needs no code change here.
    """
    try:
        import openpyxl
    except ImportError:
        raise SystemExit("pip install openpyxl")
    if not os.path.exists(args.workbook):
        raise SystemExit(f"{args.workbook} not found.")

    log(f"== loading arenas from {args.workbook} [{args.sheet}] ==")
    wb = openpyxl.load_workbook(args.workbook, read_only=True, data_only=True)
    if args.sheet not in wb.sheetnames:
        raise SystemExit(f"no '{args.sheet}' sheet - found {wb.sheetnames}")
    ws = wb[args.sheet]
    it = ws.iter_rows(values_only=True)
    header = [sanitise(c) for c in next(it)]
    if "name" not in header:
        raise SystemExit(f"'{args.sheet}' has no 'name' column")

    cols = []
    seen = set()
    for h in header:                      # de-duplicate repeated headers
        base, i = h, 2
        while h in seen:
            h = f"{base}_{i}"
            i += 1
        seen.add(h)
        cols.append(h)
    decls = ", ".join(f'"{c}" {"REAL" if c in ARENA_NUMERIC else "TEXT"}'
                      for c in cols)
    conn.execute("DROP TABLE IF EXISTS arenas")
    conn.execute(f"CREATE TABLE arenas ({decls})")

    rows = []
    for r in progress(it, "arena rows", total=max(0, (ws.max_row or 0) - 1) or None):
        if not r or not r[0]:
            continue
        out = []
        for i, c in enumerate(cols):
            v = r[i] if i < len(r) else None
            out.append(_num(v) if c in ARENA_NUMERIC else _txt(v))
        rows.append(out)
    conn.executemany(f"INSERT INTO arenas VALUES ({','.join('?' * len(cols))})", rows)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_arenas_id ON arenas(arena_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_arenas_psid "
                 "ON arenas(pollstar_venue_id)")
    conn.commit()

    # concert_capacity is the populated one; max_capacity is empty in this export
    withcap = scalar(conn, "SELECT COUNT(*) FROM arenas "
                           "WHERE concert_capacity > 0") or 0
    withps = scalar(conn, "SELECT COUNT(*) FROM arenas "
                          "WHERE TRIM(COALESCE(pollstar_venue_id,''))<>''") or 0
    log(f"   arenas: {len(rows):,} rows, {len(cols)} columns "
        f"({withcap:,} with a capacity, {withps:,} with a Pollstar venue id)")
    now = utcnow()
    _load_hospitality(wb, conn, now)
    _load_fx(wb, conn, now)
    wb.close()
    build_arena_aliases(conn)


def build_arena_aliases(conn):
    """
    One row per known spelling of a venue -> arena_id.

    Sources, in order of trust: the canonical `name`, each comma-separated
    entry in `also_known_as`, whatever `matched_venue` recorded from a previous
    reconciliation, and finally a hand-maintained CSV for anything those miss.
    """
    log("== building arena_aliases ==")
    have = {r[1] for r in conn.execute("PRAGMA table_info(arenas)")}
    conn.execute("DELETE FROM arena_aliases")
    rows = []

    def add(alias, arena_id, source, city, country):
        alias = (alias or "").strip()
        if not alias or not arena_id:
            return
        rows.append((norm_key(alias), str(arena_id), alias, source,
                     norm_key(city or ""), country or ""))

    cols = ["name", "arena_id", "city", "country"]
    extra = [c for c in ("also_known_as", "matched_venue") if c in have]
    cur = conn.execute(f"SELECT {','.join(cols + extra)} FROM arenas")
    for r in cur:
        name, arena_id, city, country = r[0], r[1], r[2], r[3]
        add(name, arena_id, "name", city, country)
        for j, c in enumerate(extra, start=4):
            for part in str(r[j] or "").split(","):
                if part.strip() and norm_key(part) != norm_key(name):
                    add(part, arena_id, c, city, country)

    manual = 0
    if os.path.exists(ARENA_ALIASES_MANUAL):
        with open(ARENA_ALIASES_MANUAL, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if (row.get("alias") or "").strip() and (row.get("arena_id") or "").strip():
                    add(row["alias"], row["arena_id"].strip(), "manual",
                        row.get("city", ""), row.get("country", ""))
                    manual += 1

    conn.executemany("INSERT OR IGNORE INTO arena_aliases "
                     "(alias_norm, arena_id, alias, source, city_norm, country) "
                     "VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    total = scalar(conn, "SELECT COUNT(*) FROM arena_aliases") or 0
    arenas = scalar(conn, "SELECT COUNT(DISTINCT arena_id) FROM arena_aliases") or 0
    for src, n in conn.execute("SELECT source, COUNT(*) FROM arena_aliases "
                               "GROUP BY 1 ORDER BY 2 DESC"):
        log(f"      {src:16s} {n:6,}")
    log(f"   {total:,} aliases for {arenas:,} arenas"
        + (f" ({manual:,} manual)" if manual else ""))
    log(f"   add anything missing to {ARENA_ALIASES_MANUAL} "
        f"(alias,arena_id[,city,country]) and re-run `load-arenas`")


def _num(v, cast=float):
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return cast(v)
    txt = str(v).strip().replace(",", "").replace("$", "").replace("%", "")
    if not txt or txt in ("-", "N/A"):
        return None
    try:
        return cast(float(txt))
    except ValueError:
        return None


_DATE_PATTERNS = ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%y")


def _date_iso(v):
    """Pollstar writes US M/D/YYYY strings; real dates arrive as datetimes."""
    if v in (None, ""):
        return ""
    if isinstance(v, dt.datetime):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, dt.date):
        return v.strftime("%Y-%m-%d")
    txt = str(v).strip()
    for pat in _DATE_PATTERNS:
        try:
            return dt.datetime.strptime(txt, pat).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def _txt(v):
    if v is None:
        return ""
    if isinstance(v, (dt.datetime, dt.date)):
        return _date_iso(v)
    return str(v).strip()


def cmd_load_pollstar(args, conn):
    """Load the Pollstar box-office export into pollstar_events."""
    try:
        import openpyxl
    except ImportError:
        raise SystemExit("pip install openpyxl")
    if not os.path.exists(args.workbook):
        raise SystemExit(f"{args.workbook} not found.")

    log(f"== loading {args.workbook} ==")
    wb = openpyxl.load_workbook(args.workbook, read_only=True, data_only=True)
    ws = wb[args.sheet] if args.sheet else wb[wb.sheetnames[0]]
    rows_iter = ws.iter_rows(values_only=True)
    header = next(rows_iter)
    hm = {str(c).strip().lower(): i for i, c in enumerate(header) if c}

    def col(*names):
        for n in names:
            if n in hm:
                return hm[n]
        return None

    idx = {
        "start": col("start date"), "end": col("end date"),
        "n_shows": col("number of shows"),
        "headliner": col("headliner"), "support": col("support"),
        "venue_id": col("venue id"), "venue": col("venue"),
        "venue_type": col("venue type"), "city": col("city"),
        "state": col("state"), "zip": col("zip"), "country": col("country"),
        "currency": col("currency"), "promoter": col("promoter"),
        "genre": col("genre"), "market": col("market"),
        "total_tickets": col("total tickets sold"),
        "avg_tickets": col("average tickets sold"),
        "total_gross": col("total gross usd"),
        "avg_gross": col("average gross usd"),
        "avg_capacity": col("average show capacity"),
        "total_capacity": col("total show capacity"),
        "avg_cap_sold": col("average capacity sold"),
        "price_min": col("ticket price min"), "price_max": col("ticket price max"),
        "price_avg": col("ticket price average"),
    }
    missing = [k for k in ("start", "headliner", "venue") if idx[k] is None]
    if missing:
        raise SystemExit(f"{args.workbook} is missing columns: {missing}")

    def get(r, key):
        i = idx[key]
        return r[i] if i is not None and i < len(r) else None

    now = utcnow()
    conn.execute("DELETE FROM pollstar_events")
    batch, total, skipped = [], 0, 0
    INSERT = """INSERT INTO pollstar_events (start_iso, end_iso, n_shows,
        headliner, headliner_norm, support, venue_id, venue, venue_norm,
        venue_type, city, city_norm, state, zip, country, currency, promoter,
        genre, market, total_tickets, avg_tickets, total_gross_usd,
        avg_gross_usd, avg_capacity, total_capacity, avg_capacity_sold,
        price_min, price_max, price_avg, loaded_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""

    n_rows = max(0, (ws.max_row or 0) - 1) or None
    for r in progress(rows_iter, "pollstar rows", total=n_rows):
        headliner = _txt(get(r, "headliner"))
        start = _date_iso(get(r, "start"))
        if not headliner or not start:
            skipped += 1
            continue
        venue = _txt(get(r, "venue"))
        city = _txt(get(r, "city"))
        batch.append((
            start, _date_iso(get(r, "end")) or start,
            _num(get(r, "n_shows"), int) or 1,
            headliner, norm_key(headliner), _txt(get(r, "support")),
            _txt(get(r, "venue_id")), venue, norm_key(venue),
            _txt(get(r, "venue_type")), city, norm_key(city),
            _txt(get(r, "state")), _txt(get(r, "zip")), _txt(get(r, "country")),
            _txt(get(r, "currency")), _txt(get(r, "promoter")),
            _txt(get(r, "genre")), _txt(get(r, "market")),
            _num(get(r, "total_tickets"), int), _num(get(r, "avg_tickets"), int),
            _num(get(r, "total_gross")), _num(get(r, "avg_gross")),
            _num(get(r, "avg_capacity"), int), _num(get(r, "total_capacity"), int),
            _num(get(r, "avg_cap_sold")),
            _num(get(r, "price_min")), _num(get(r, "price_max")),
            _num(get(r, "price_avg")), now))
        if len(batch) >= 20000:
            conn.executemany(INSERT, batch)
            conn.commit()
            total += len(batch)
            batch = []
    if batch:
        conn.executemany(INSERT, batch)
        conn.commit()
        total += len(batch)
    wb.close()

    runs = scalar(conn, "SELECT COUNT(*) FROM pollstar_events WHERE n_shows > 1") or 0
    lo = scalar(conn, "SELECT MIN(start_iso) FROM pollstar_events")
    hi = scalar(conn, "SELECT MAX(end_iso) FROM pollstar_events")
    log(f"   pollstar_events: {total:,} rows ({skipped:,} skipped: no headliner "
        f"or no date)")
    log(f"   dates {lo} .. {hi}; {runs:,} rows cover a multi-night run")
    log("   next: `build` (or `match-pollstar` to redo just the join)")


def match_sql(slack):
    """
    Build the match statement. `slack` widens the Pollstar date window by N days
    each way, for the cases where the two sources disagree about which calendar
    day a late-night show belongs to.

    Tiers, best first:
        arena  same building, by arena_id -- survives renaming
        venue  identical normalised venue name
        city   same town, for venues the two sources spell differently
    Within a tier, an exact-date hit beats a slack one, and a single-night
    Pollstar row beats a multi-night run.
    """
    lo = f"date(p.start_iso, '-{int(slack)} day')" if slack else "p.start_iso"
    hi = f"date(p.end_iso, '+{int(slack)} day')" if slack else "p.end_iso"
    return f"""
WITH cand AS (
    SELECT e.event_id, p.id AS pid,
           CASE WHEN p.arena_id IS NOT NULL AND p.arena_id = e.arena_id THEN 1
                WHEN p.venue_norm = e.venue_norm                        THEN 2
                ELSE 3 END AS tier_n,
           CASE WHEN e.date_iso BETWEEN p.start_iso AND p.end_iso THEN 0
                WHEN e.date_iso < p.start_iso
                     THEN CAST(julianday(p.start_iso) - julianday(e.date_iso) AS INT)
                ELSE CAST(julianday(e.date_iso) - julianday(p.end_iso) AS INT)
           END AS day_off,
           p.n_shows
    FROM events e
    JOIN pollstar_events p
      ON p.headliner_norm = e.headliner_norm
     AND e.date_iso BETWEEN {lo} AND {hi}
    WHERE e.headliner_norm <> ''
      AND ((p.arena_id IS NOT NULL AND p.arena_id = e.arena_id)
           OR p.venue_norm = e.venue_norm
           OR p.city_norm  = e.city_norm)
),
best AS (
    SELECT event_id, pid, tier_n, day_off,
           ROW_NUMBER() OVER (PARTITION BY event_id
                              ORDER BY tier_n, ABS(day_off), n_shows, pid) AS rn
    FROM cand
)
UPDATE events SET
    pollstar_id            = p.id,
    pollstar_match         = CASE b.tier_n WHEN 1 THEN 'arena'
                                           WHEN 2 THEN 'venue' ELSE 'city' END,
    pollstar_date_offset   = b.day_off,
    pollstar_headliner     = p.headliner,
    pollstar_support       = p.support,
    pollstar_venue         = p.venue,
    pollstar_venue_id      = p.venue_id,
    pollstar_venue_type    = p.venue_type,
    pollstar_market        = p.market,
    pollstar_genre         = p.genre,
    pollstar_promoter      = p.promoter,
    pollstar_run_shows     = p.n_shows,
    pollstar_tickets_sold  = p.avg_tickets,
    pollstar_capacity      = p.avg_capacity,
    pollstar_capacity_pct  = p.avg_capacity_sold,
    pollstar_gross_usd     = p.avg_gross_usd,
    pollstar_run_tickets   = p.total_tickets,
    pollstar_run_gross_usd = p.total_gross_usd,
    pollstar_price_min     = p.price_min,
    pollstar_price_max     = p.price_max,
    pollstar_price_avg     = p.price_avg,
    pollstar_start         = p.start_iso,
    pollstar_end           = p.end_iso
FROM best b JOIN pollstar_events p ON p.id = b.pid
WHERE b.event_id = events.event_id AND b.rn = 1
"""


def ensure_columns(conn):
    """
    Add any column the schema has gained since a table was created.

    CREATE TABLE IF NOT EXISTS leaves an older table untouched, so without this
    a rebuild against an existing database fails on the first new column.
    """
    wanted = {
        "events": POLLSTAR_COLUMNS,
        "pollstar_events": [("arena_id", "TEXT")],
    }
    for table, cols in wanted.items():
        have = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
        if not have:
            continue
        for name, decl in cols:
            if name not in have:
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN {name} {decl}')
                log(f"   migrated: added {table}.{name}")
    conn.commit()


RESOLVE_PS_ARENA = """
UPDATE pollstar_events SET arena_id = (
    SELECT a.arena_id FROM arenas a
    WHERE TRIM(COALESCE(a.pollstar_venue_id,'')) <> ''
      AND a.pollstar_venue_id = pollstar_events.venue_id)
WHERE TRIM(COALESCE(venue_id,'')) <> ''
"""

RESOLVE_PS_ARENA_BY_NAME = """
UPDATE pollstar_events SET arena_id = (
    SELECT al.arena_id FROM arena_aliases al
    WHERE al.alias_norm = pollstar_events.venue_norm)
WHERE arena_id IS NULL AND venue_norm <> ''
"""

RESOLVE_EVENT_ARENA = """
UPDATE events SET arena_id = (
    SELECT al.arena_id FROM arena_aliases al
    WHERE al.alias_norm = events.venue_norm)
WHERE venue_norm <> ''
"""

ATTACH_ARENA = """
UPDATE events SET
    arena_name           = a.name,
    arena_capacity       = a.concert_capacity,
    arena_type           = a.arena_type,
    arena_outside_inside = a.outside_inside
FROM arenas a
WHERE a.arena_id = events.arena_id
"""


def resolve_arenas(conn):
    """
    Put an arena_id on both sides of the join.

    Pollstar rows resolve by their own venue id against the 573 arenas that
    carry one -- an exact id match, immune to renaming. Anything left, and all
    setlist.fm venues, resolve by name through arena_aliases, which knows that
    Manchester Arena, MEN Arena and AO Arena are one building.
    """
    have_arenas = scalar(conn, "SELECT COUNT(*) FROM arenas") or 0
    if not have_arenas:
        log("   !! arenas is empty - run `load-arenas` first. Falling back to "
            "venue-name matching only.")
        return False
    conn.execute("UPDATE pollstar_events SET arena_id = NULL")
    conn.execute(RESOLVE_PS_ARENA)
    by_id = scalar(conn, "SELECT COUNT(*) FROM pollstar_events "
                         "WHERE arena_id IS NOT NULL") or 0
    conn.execute(RESOLVE_PS_ARENA_BY_NAME)
    ps_total = scalar(conn, "SELECT COUNT(*) FROM pollstar_events "
                            "WHERE arena_id IS NOT NULL") or 0
    conn.execute("CREATE INDEX IF NOT EXISTS ix_ps_arena "
                 "ON pollstar_events(arena_id, start_iso)")
    conn.execute(RESOLVE_EVENT_ARENA)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_events_arena ON events(arena_id)")
    conn.execute(ATTACH_ARENA)
    conn.commit()
    ev = scalar(conn, "SELECT COUNT(*) FROM events WHERE arena_id IS NOT NULL") or 0
    n_ev = scalar(conn, "SELECT COUNT(*) FROM events") or 1
    n_ps = scalar(conn, "SELECT COUNT(*) FROM pollstar_events") or 1
    log(f"   arenas resolved: {ps_total:,}/{n_ps:,} Pollstar rows "
        f"({by_id:,} by Pollstar venue id, {ps_total - by_id:,} by name), "
        f"{ev:,}/{n_ev:,} events ({100.0*ev/n_ev:.1f}%)")
    return True


def cmd_match_pollstar(args, conn):
    """
    Attach Pollstar box-office rows to events.

    Matched on normalised headliner + the event date falling inside the
    Pollstar row's start..end range, then narrowed by venue, falling back to
    city when the two sources spell the venue differently (they often do:
    "Maine Savings Amphitheater" vs "Maine Savings Amphitheatre"). Where an
    artist has several candidate rows for one night, a single-night row wins
    over a multi-night run.

    Name differences are absorbed by norm_key -- the same normalisation the
    categoriser uses -- so accents, punctuation, "&"/"and" and a leading "The"
    do not break the join.
    """
    ensure_columns(conn)
    n_ps = scalar(conn, "SELECT COUNT(*) FROM pollstar_events") or 0
    if not n_ps:
        log("!! pollstar_events is empty - run `load-pollstar` first. "
            "Leaving the Pollstar columns blank.")
        return
    n_ev = scalar(conn, "SELECT COUNT(*) FROM events") or 0
    log(f"== matching {n_ev:,} events against {n_ps:,} Pollstar rows ==")

    # the join keys live on `events`; fill them in if this is being run against
    # an events table built before they existed, or after a partial rebuild
    unkeyed = scalar(conn, "SELECT COUNT(*) FROM events "
                           "WHERE headliner_norm IS NULL OR headliner_norm=''") or 0
    if unkeyed:
        log(f"   normalising {unkeyed:,} events that have no join key yet ...")
        conn.execute("""UPDATE events SET headliner_norm = norm_key(headliner),
                                          venue_norm     = norm_key(venue),
                                          city_norm      = norm_key(city)
                        WHERE headliner_norm IS NULL OR headliner_norm=''""")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_events_hnorm "
                     "ON events(headliner_norm, date_iso)")
        conn.commit()

    ensure_columns(conn)
    resolve_arenas(conn)
    conn.execute("""UPDATE events SET pollstar_id=NULL, pollstar_match=NULL,
                    pollstar_date_offset=NULL, pollstar_headliner=NULL,
                    pollstar_tickets_sold=NULL, pollstar_gross_usd=NULL,
                    pollstar_capacity=NULL""")
    log(f"   matching (date slack +/-{args.date_slack} days) ...")
    conn.execute(match_sql(args.date_slack))
    conn.commit()
    _pollstar_summary(conn)


def _pollstar_summary(conn):
    n_ev = scalar(conn, "SELECT COUNT(*) FROM events") or 0
    matched = scalar(conn, "SELECT COUNT(*) FROM events "
                           "WHERE pollstar_id IS NOT NULL") or 0
    log(f"   {matched:,} of {n_ev:,} events matched "
        f"({100.0*matched/n_ev:.1f}%)" if n_ev else "   no events")
    for tier, n in conn.execute("""SELECT pollstar_match, COUNT(*) FROM events
                                   WHERE pollstar_id IS NOT NULL
                                   GROUP BY 1 ORDER BY 2 DESC"""):
        log(f"      matched on {tier:6s} {n:9,}")
    for off, n in conn.execute("""SELECT pollstar_date_offset, COUNT(*) FROM events
                                  WHERE pollstar_id IS NOT NULL
                                  GROUP BY 1 ORDER BY 1"""):
        label = "exact date" if off == 0 else f"{off} day(s) out"
        log(f"      {label:16s} {n:9,}")
    # how much of the Pollstar file found a home
    used = scalar(conn, "SELECT COUNT(DISTINCT pollstar_id) FROM events "
                        "WHERE pollstar_id IS NOT NULL") or 0
    n_ps = scalar(conn, "SELECT COUNT(*) FROM pollstar_events") or 0
    log(f"   {used:,} of {n_ps:,} Pollstar rows used ({100.0*used/n_ps:.1f}%)"
        if n_ps else "")
    for cat, n, m in conn.execute("""SELECT COALESCE(NULLIF(category,''),'(none)'),
                                            COUNT(*),
                                            SUM(pollstar_id IS NOT NULL)
                                     FROM events GROUP BY 1 ORDER BY 2 DESC"""):
        log(f"      {cat:40s} {m or 0:8,} of {n:8,} matched")


ATTACH_VENUE = """
UPDATE events SET
    venue_capacity        = v.capacity,
    venue_capacity_source = v.capacity_source,
    venue_type            = v.venue_type,
    venue_outside_inside  = v.outside_inside
FROM venues v
WHERE v.venue_uid = events.venue_uid
"""


def cmd_build_venues(args, conn):
    """
    Assemble one row per BUILDING from everything we already hold.

    The grain is deliberately not events.venue_key. That key is venue_id where
    setlist.fm gave us one and "venue|city" where it did not, so the 1.1M rows
    imported from the old CSVs gave the same building a second identity --
    27,937 venues were split this way, across roughly a million events, each
    half carrying its own conflicting capacity. Here the key is the normalised
    venue name plus city plus country, so the halves merge and pool their
    Pollstar observations.

    Priority for every fact: the dashboard's verified arena row, then what
    Pollstar reported at the door, then a conservative inference from the venue
    type. Every field records which of those it came from, so a later scrape can
    fill only the gaps and never overwrite something verified.
    """
    now = utcnow()
    ensure_columns(conn)
    log("== building venues ==")

    # the canonical building id, written back onto events so nothing needs to
    # recompute it later
    log("   assigning canonical venue ids ...")
    conn.execute("""
        UPDATE events SET venue_uid =
            CASE WHEN COALESCE(venue_norm,'') = '' THEN venue_key
                 ELSE venue_norm || '|' || COALESCE(city_norm,'')
                                 || '|' || COALESCE(countryCode,'')
            END""")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_events_vuid ON events(venue_uid)")
    conn.commit()

    conn.execute("DROP TABLE IF EXISTS tmp_venues")
    conn.execute("""
        CREATE TEMP TABLE tmp_venues AS
        SELECT venue_uid,
               MAX(venue)       AS venue,
               MAX(venue_norm)  AS venue_norm,
               MAX(city)        AS city,
               MAX(city_norm)   AS city_norm,
               MAX(country)     AS country,
               MAX(countryCode) AS countryCode,
               MAX(latitude)    AS latitude,
               MAX(longitude)   AS longitude,
               COUNT(DISTINCT venue_key) AS n_keys,
               COUNT(*)         AS events,
               MIN(date_iso)    AS first_event,
               MAX(date_iso)    AS last_event,
               MAX(arena_id)    AS arena_id,
               MAX(pollstar_venue_id) AS pollstar_venue_id,
               MAX(pollstar_capacity) AS cap_max,
               CAST(AVG(pollstar_capacity) AS INT) AS cap_typical,
               SUM(pollstar_capacity IS NOT NULL) AS cap_samples,
               MAX(pollstar_venue_type) AS ps_type
        FROM events
        WHERE TRIM(COALESCE(venue_uid,'')) <> ''
        GROUP BY venue_uid""")
    conn.commit()
    n = scalar(conn, "SELECT COUNT(*) FROM tmp_venues") or 0
    keys = scalar(conn, "SELECT COUNT(DISTINCT venue_key) FROM events") or 0
    merged = scalar(conn, "SELECT COUNT(*) FROM tmp_venues WHERE n_keys > 1") or 0
    log(f"   {keys:,} raw venue keys -> {n:,} buildings "
        f"({merged:,} had more than one identity)")

    conn.execute("DROP TABLE IF EXISTS venues")
    conn.executescript(SCHEMA)
    conn.execute("""
        INSERT INTO venues (venue_uid, venue, venue_norm, city, city_norm, country,
            countryCode, latitude, longitude, n_keys, events, first_event,
            last_event, arena_id, pollstar_venue_id, capacity, capacity_source,
            capacity_observed_max, capacity_observed_typical, capacity_samples,
            venue_type, venue_type_source, outside_inside, outside_inside_source,
            needs_review, updated_at)
        SELECT t.venue_uid, t.venue, t.venue_norm, t.city, t.city_norm, t.country,
               t.countryCode, t.latitude, t.longitude, t.n_keys, t.events,
               t.first_event, t.last_event, t.arena_id, t.pollstar_venue_id,
               COALESCE(NULLIF(a.concert_capacity, 0), t.cap_max),
               CASE WHEN NULLIF(a.concert_capacity, 0) IS NOT NULL THEN 'dashboard'
                    WHEN t.cap_max IS NOT NULL THEN 'pollstar'
                    ELSE NULL END,
               t.cap_max, t.cap_typical, COALESCE(t.cap_samples, 0),
               COALESCE(NULLIF(a.arena_type, ''), NULLIF(t.ps_type, '')),
               CASE WHEN NULLIF(a.arena_type, '') IS NOT NULL THEN 'dashboard'
                    WHEN NULLIF(t.ps_type, '') IS NOT NULL THEN 'pollstar'
                    ELSE NULL END,
               CASE WHEN NULLIF(a.outside_inside, '') IS NOT NULL THEN a.outside_inside
                    WHEN LOWER(COALESCE(t.ps_type,'')) IN ('amphitheater','amphitheatre','fairground','festival site','outdoor venues','race track','racetrack','stadium') THEN 'Outside'
                    WHEN LOWER(COALESCE(t.ps_type,'')) IN ('arena','auditorium / theatre','ballroom','casino','club','convention center','theater','theatre')  THEN 'Inside'
                    ELSE NULL END,
               CASE WHEN NULLIF(a.outside_inside, '') IS NOT NULL THEN 'dashboard'
                    WHEN LOWER(COALESCE(t.ps_type,'')) IN ('amphitheater','amphitheatre','arena','auditorium / theatre','ballroom','casino','club','convention center','fairground','festival site','outdoor venues','race track','racetrack','stadium','theater','theatre')
                         THEN 'inferred from venue type'
                    ELSE NULL END,
               0, ?
        FROM tmp_venues t
        LEFT JOIN arenas a ON a.arena_id = t.arena_id""", (now,))
    conn.commit()

    # keep a note of which raw keys merged into each building
    conn.execute("""
        UPDATE venues SET venue_keys = (
            SELECT group_concat(DISTINCT e.venue_key)
            FROM events e WHERE e.venue_uid = venues.venue_uid)
        WHERE n_keys > 1""")
    conn.commit()

    conn.execute("""UPDATE venues SET needs_review = 1
                    WHERE events >= ?
                      AND (capacity IS NULL OR outside_inside IS NULL)""",
                 (args.review_min_events,))
    conn.commit()

    # push the assembled facts back onto events so queries need no join
    log("   attaching venue facts to events ...")
    conn.execute("UPDATE events SET venue_capacity=NULL, venue_capacity_source=NULL, "
                 "venue_type=NULL, venue_outside_inside=NULL")
    conn.execute(ATTACH_VENUE)
    conn.commit()
    _venue_summary(conn, args.review_min_events)


def _venue_summary(conn, review_min=None):
    n = scalar(conn, "SELECT COUNT(*) FROM venues") or 1
    for label, where in (("capacity known", "capacity IS NOT NULL"),
                         ("venue type known", "venue_type IS NOT NULL"),
                         ("indoor/outdoor known", "outside_inside IS NOT NULL"),
                         ("linked to a dashboard arena", "arena_id IS NOT NULL")):
        c = scalar(conn, f"SELECT COUNT(*) FROM venues WHERE {where}") or 0
        ev = scalar(conn, f"SELECT SUM(events) FROM venues WHERE {where}") or 0
        log(f"   {label:30s} {c:7,} venues ({100.0*c/n:4.1f}%)  {ev:9,} events")
    m = scalar(conn, "SELECT COUNT(*) FROM venues WHERE n_keys > 1") or 0
    if m:
        log(f"   {m:,} buildings were merged from more than one raw venue key")
    log("   capacity provenance:")
    for src, c in conn.execute("""SELECT COALESCE(capacity_source,'(none)'), COUNT(*)
                                  FROM venues GROUP BY 1 ORDER BY 2 DESC"""):
        log(f"      {src:22s} {c:7,}")
    if review_min is not None:
        r = scalar(conn, "SELECT COUNT(*) FROM venues WHERE needs_review=1") or 0
        rev = scalar(conn, "SELECT SUM(events) FROM venues WHERE needs_review=1") or 0
        log(f"   flagged for enrichment (>= {review_min} events, missing "
            f"capacity or indoor/outdoor): {r:,} venues covering {rev:,} events")


def cmd_missing_venues(args, conn):
    """Export the enrichment worklist, biggest venues first."""
    cols = ["venue", "city", "country", "events", "first_event", "last_event",
            "capacity", "capacity_source", "venue_type", "outside_inside",
            "latitude", "longitude", "venue_uid"]
    rows = conn.execute(f"""
        SELECT {','.join(cols)} FROM venues
        WHERE events >= ? AND (capacity IS NULL OR outside_inside IS NULL)
        ORDER BY events DESC LIMIT ?""",
        (args.min_events, args.limit)).fetchall()
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])
    log(f"== {len(rows):,} venues need enriching -> {args.out} ==")
    log(f"   {'events':>7}  {'cap':>7}  venue")
    for r in rows[:20]:
        log(f"   {r[3]:7,}  {(r[6] or 0):7,}  {(r[0] or '')[:40]:40s} {(r[1] or '')[:18]}")


def stage(n, total, label):
    log(f"   [{n}/{total}] {label} ...")


def cmd_build(args, conn):
    have_cats = scalar(conn, "SELECT COUNT(*) FROM sqlite_master "
                             "WHERE type='table' AND name='artist_categories'")
    if not have_cats:
        raise SystemExit("artist_categories is missing - run "
                         "`python artist_categories.py refresh` first.")
    n_cats = scalar(conn, "SELECT COUNT(*) FROM artist_categories") or 0
    if n_cats == 0:
        log("!! artist_categories is empty - events will have no categories. "
            "Run `python artist_categories.py refresh` first.")

    src = scalar(conn, "SELECT COUNT(*) FROM setlists "
                       "WHERE TRIM(COALESCE(date_iso,''))<>'' "
                       "AND TRIM(COALESCE(artist,''))<>''") or 0
    skipped = (scalar(conn, "SELECT COUNT(*) FROM setlists") or 0) - src
    log(f"== building events from {src:,} setlist rows "
        f"({n_cats:,} categorised artists) ==")
    if skipped:
        log(f"   {skipped:,} rows skipped: no date or no artist")

    steps = 4 if args.no_pollstar else 5
    stage(1, steps, "collapsing date + venue into events")
    ensure_columns(conn)
    conn.execute("DELETE FROM events")
    conn.execute(BUILD, (utcnow(),))
    conn.commit()
    n = scalar(conn, "SELECT COUNT(*) FROM events") or 0
    multi = scalar(conn, "SELECT COUNT(*) FROM events WHERE n_artists > 1") or 0
    log(f"   {n:,} events ({multi:,} with more than one act on the bill)")

    stage(2, steps, "normalising names for joining")
    conn.execute("""UPDATE events SET headliner_norm = norm_key(headliner),
                                      venue_norm     = norm_key(venue),
                                      city_norm      = norm_key(city)""")
    conn.commit()
    conn.execute("CREATE INDEX IF NOT EXISTS ix_events_hnorm "
                 "ON events(headliner_norm, date_iso)")
    conn.commit()

    stage(3, steps, "routing previous/next city")
    conn.execute("UPDATE events SET previous_city='', next_city=''")
    conn.execute(ROUTES)
    conn.commit()
    routed = scalar(conn, "SELECT COUNT(*) FROM events "
                          "WHERE TRIM(COALESCE(previous_city,''))<>''") or 0
    log(f"   {routed:,} events routed "
        f"({n - routed:,} have no tour name, so no route)")

    if args.no_pollstar:
        log("   skipping the Pollstar match (--no-pollstar)")
    else:
        stage(4, steps, "matching Pollstar")
        cmd_match_pollstar(args, conn)

    stage(steps, steps, "assembling venues")
    cmd_build_venues(args, conn)

    _integrity(conn)
    _summary(conn)


def _integrity(conn):
    ok = True
    src = scalar(conn, "SELECT COUNT(*) FROM setlists "
                       "WHERE TRIM(COALESCE(date_iso,''))<>'' "
                       "AND TRIM(COALESCE(artist,''))<>''") or 0
    got = scalar(conn, "SELECT SUM(n_artists) FROM events") or 0
    if src == got:
        log(f"== integrity: {got:,} bill slots across events = {src:,} setlist "
            f"rows. OK ==")
    else:
        ok = False
        log(f"!! integrity: events account for {got:,} setlist rows but there "
            f"are {src:,} ({src - got:+,})")

    dupes = scalar(conn, """SELECT COUNT(*) FROM (
                                SELECT headliner_key, date_iso, venue_key
                                FROM events GROUP BY 1,2,3 HAVING COUNT(*)>1)""") or 0
    if dupes:
        ok = False
        log(f"!! {dupes:,} duplicate headliner/date/venue combinations")

    orphan = scalar(conn, "SELECT COUNT(*) FROM events "
                          "WHERE TRIM(COALESCE(category,''))=''") or 0
    if orphan:
        log(f"   note: {orphan:,} events have no category "
            f"(headliner missing from artist_categories)")
    return ok


def _summary(conn):
    log("   events by category:")
    for cat, n, acts in conn.execute("""
            SELECT COALESCE(NULLIF(category,''),'(uncategorised)'),
                   COUNT(*), SUM(n_artists)
            FROM events GROUP BY 1 ORDER BY COUNT(*) DESC"""):
        log(f"      {cat:40s} {n:8,} events  {acts:9,} bill slots")


def cmd_status(args, conn):
    n = scalar(conn, "SELECT COUNT(*) FROM events") or 0
    if not n:
        log("events is empty - run `python build_events.py build`")
        return
    built = scalar(conn, "SELECT MAX(built_at) FROM events")
    lo = scalar(conn, "SELECT MIN(date_iso) FROM events")
    hi = scalar(conn, "SELECT MAX(date_iso) FROM events")
    multi = scalar(conn, "SELECT COUNT(*) FROM events WHERE n_artists>1") or 0
    biggest = scalar(conn, "SELECT MAX(n_artists) FROM events") or 0
    venues = scalar(conn, "SELECT COUNT(DISTINCT venue_key) FROM events") or 0
    log(f"== events ({n:,} rows, built {built}) ==")
    log(f"  date range      : {lo} .. {hi}")
    log(f"  distinct venues : {venues:,}")
    log(f"  multi-act bills : {multi:,} ({100.0*multi/n:.1f}%), "
        f"biggest bill {biggest} acts")
    _summary(conn)
    if scalar(conn, "SELECT COUNT(*) FROM pollstar_events"):
        _pollstar_summary(conn)
    _integrity(conn)
    log("  a few multi-act bills:")
    for d, v, h, s in conn.execute("""SELECT date_iso, venue, headliner, support
                                      FROM events WHERE n_artists BETWEEN 3 AND 5
                                      ORDER BY date_iso DESC LIMIT 5"""):
        log(f"      {d}  {(v or '')[:28]:28s}  {h} + [{s}]")


def cmd_export(args, conn):
    log(f"== exporting -> {args.to} ==")
    cur = conn.execute(f"SELECT {','.join(EXPORT_COLUMNS)} FROM events "
                       f"ORDER BY date_iso, venue_key")
    n = 0
    with open(args.to, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(EXPORT_COLUMNS)
        for row in cur:
            w.writerow(["" if v is None else v for v in row])
            n += 1
    log(f"   {n:,} events -> {args.to}")


def main():
    p = argparse.ArgumentParser(
        description="Build the derived events table (combined bills, categories, "
                    "tour routing). Never alters the setlists table.")
    p.add_argument("--db", default=DB_PATH)
    sub = p.add_subparsers(dest="cmd", required=True)

    la = sub.add_parser("load-arenas",
                        help="Load the dashboard's Arena Data tab + build the "
                             "venue name conversion table.")
    la.add_argument("--workbook", default=ARENA_WORKBOOK)
    la.add_argument("--sheet", default=ARENA_SHEET)
    la.set_defaults(func=cmd_load_arenas)

    lp = sub.add_parser("load-pollstar",
                        help="Load pollstar-data.xlsx into the pollstar_events table.")
    lp.add_argument("--workbook", default=POLLSTAR_WORKBOOK)
    lp.add_argument("--sheet", default=None)
    lp.set_defaults(func=cmd_load_pollstar)

    b = sub.add_parser("build",
                       help="Rebuild events from setlists + artist_categories, "
                            "then attach Pollstar.")
    b.add_argument("--no-pollstar", action="store_true",
                   help="Skip the Pollstar match.")
    b.add_argument("--review-min-events", type=int, default=20,
                   help="Flag venues with at least this many events for enrichment.")
    b.add_argument("--date-slack", type=int, default=1,
                   help="Allow the event date to fall this many days outside "
                        "the Pollstar range (default 1).")
    b.set_defaults(func=cmd_build)

    bv = sub.add_parser("build-venues",
                        help="Assemble the persistent venues table from events, "
                             "arenas and Pollstar.")
    bv.add_argument("--review-min-events", type=int, default=20,
                    help="Flag venues with at least this many events for enrichment.")
    bv.set_defaults(func=cmd_build_venues)

    mv = sub.add_parser("missing-venues",
                        help="Export the venues still missing capacity or "
                             "indoor/outdoor, biggest first.")
    mv.add_argument("--out", default="venues_to_enrich.csv")
    mv.add_argument("--min-events", type=int, default=20)
    mv.add_argument("--limit", type=int, default=5000)
    mv.set_defaults(func=cmd_missing_venues)

    mp = sub.add_parser("match-pollstar",
                        help="Redo just the Pollstar join on the existing events.")
    mp.add_argument("--date-slack", type=int, default=1,
                    help="Allow the event date to fall this many days outside "
                         "the Pollstar range (default 1).")
    mp.set_defaults(func=cmd_match_pollstar)

    s = sub.add_parser("status", help="What's in the events table.")
    s.set_defaults(func=cmd_status)

    e = sub.add_parser("export", help="Write events to CSV.")
    e.add_argument("--to", default=DEFAULT_EXPORT)
    e.set_defaults(func=cmd_export)

    args = p.parse_args()
    conn = connect(args.db)
    try:
        args.func(args, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
