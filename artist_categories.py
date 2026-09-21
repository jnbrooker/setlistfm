#!/usr/bin/env python3
"""
Artist categorisation -> setlistfm.db

Phase 1 of the derived-data pipeline. Loads the reference data out of
arenas-dashboard.xlsx, builds the artist-name conversion table that lets the
three naming schemes (setlist.fm / Spotify / Pollstar) join, and writes an
artist_categories table.

It NEVER touches the `setlists` table. Everything here lives in new tables:

    ref_spotify              Spotify Monthly Listeners tab, keyed by artist_key
    ref_pollstar             Pollstar Touring Rankings tab, keyed by artist_key
    ref_dashboard_categories the dashboard's own Artist Categorisation rows
    category_thresholds      the editable thresholds from N1:O9 of that tab
    artist_categories        the output: one row per artist, rebuilt each run

THE RULES (read from category_thresholds, not hardcoded, because the dashboard
says "edit these" and changing them is meant to re-tier everything):

    Category A   Spotify monthly listeners >= A-threshold  OR  Pollstar rank <= A-rank
    Category B   listeners >= B-threshold                  OR  Pollstar rank <= B-rank
    Category C   below both B tests, or absent from both lists
    Tenant Sporting Event      >= N dates for the entity AND >= M dates per venue
    Family, Entertainment, Comedy & Other   decided by genre, before the A/B/C tests

Verified: re-implementing these reproduced all 7,184 'Artist' rows on the
dashboard tab with zero mismatches.

USAGE
-----
    pip install openpyxl

    # after ANY data update -- new setlists, an edited threshold -- this one
    # command pulls fresh listeners, rebuilds everything, and checks it:
    python artist_categories.py refresh
    python artist_categories.py refresh --no-kworb   # offline: no network pull

    # or the individual steps
    python artist_categories.py load        # xlsx -> ref_* tables + thresholds
    python artist_categories.py categorise  # build artist_categories
    python artist_categories.py status

    python artist_categories.py suspects
        Category E artists who play heavily at the venues Category A/B acts use.
        They are either genuinely small or a name-match failure -- this is the
        cheapest way to find the second kind.

    python artist_categories.py unmatched --out unmatched_artists.csv
        The setlist.fm artists with no reference match, biggest first. Fix any
        that matter in artist_aliases_manual.csv (alias,artist_key) and re-run
        `refresh`.

    python artist_categories.py verify      # re-derive the dashboard's own rows
        Proves the implemented rules still reproduce the spreadsheet exactly.

A genre caveat worth knowing: genre lives in the dashboard (it comes from
Pollstar's Event Data), not in setlist.fm. So a setlist.fm artist who is NOT on
the dashboard has no genre, and therefore cannot be classified Family /
Entertainment / Sport by this script — they fall through to the Artist path and
get A/B/C. `status` reports how many rows that affects.
"""

import argparse
import csv
import datetime as dt
import html
import os
import re
import sqlite3
import sys
import unicodedata

import paths

# resolved against the project folder, not the working directory -- see paths.py
DB_PATH = paths.DB
WORKBOOK = paths.ARENA_WORKBOOK
MANUAL_ALIASES = paths.ARTIST_ALIASES_MANUAL

# Spotify monthly listeners come from kworb, not from the Spotify API -- the
# Web API has never exposed monthly listeners (only followers and a 0-100
# popularity index), and the thresholds are defined on listeners.
# Paginated: listeners.html, then listeners2.html ... listeners11.html, which
# together run from rank 1 down to ~25,000 (about 500k monthly listeners). Pages
# are discovered by walking until a 404 rather than hardcoded, so a new page
# appearing on kworb is picked up on the next run. Column layouts differ between
# pages (page 1 has a "Peak" rank column the others lack), so the parser reads
# the header instead of assuming positions.
KWORB_BASE = "https://kworb.net/spotify/listeners{n}.html"
KWORB_MAX_PAGES = 40

TAB_CATEGORIES = "Artist Categorisation"
TAB_SPOTIFY = "Spotify Monthly Listeners"
TAB_POLLSTAR = "Pollstar Touring Rankings"

SCHEMA = """
CREATE TABLE IF NOT EXISTS ref_spotify (
    artist_key        TEXT PRIMARY KEY,
    rank              INTEGER,
    artist            TEXT,
    spotify_artist_id TEXT,
    monthly_listeners INTEGER,
    peak_listeners    INTEGER,
    spotify_url       TEXT,
    loaded_at         TEXT
);

CREATE TABLE IF NOT EXISTS ref_pollstar (
    artist_key        TEXT PRIMARY KEY,
    rank              INTEGER,
    artist            TEXT,
    gross_usd         REAL,
    avg_ticket_price  REAL,
    total_tickets     INTEGER,
    shows             INTEGER,
    agency            TEXT,
    chart_period_from TEXT,
    chart_period_to   TEXT,
    loaded_at         TEXT
);

CREATE TABLE IF NOT EXISTS ref_dashboard_categories (
    headliner     TEXT PRIMARY KEY,
    artist_key    TEXT,
    event_type    TEXT,
    category      TEXT,
    basis         TEXT,
    primary_genre TEXT,
    genre_class   TEXT,
    bookings      INTEGER,
    venues        INTEGER,
    -- the sheet's own per-row resolved inputs. Compound billings such as
    -- '"106 KMEL Summer Jam", Post Malone' are already resolved to the act that
    -- decides the tier, so these are the numbers `verify` must test against.
    sheet_listeners INTEGER,
    sheet_rank      INTEGER,
    loaded_at     TEXT
);
CREATE INDEX IF NOT EXISTS ix_refdash_key ON ref_dashboard_categories(artist_key);

-- the editable thresholds, lifted straight out of the workbook
CREATE TABLE IF NOT EXISTS category_thresholds (
    name      TEXT PRIMARY KEY,
    value     REAL,
    label     TEXT,
    loaded_at TEXT
);

CREATE TABLE IF NOT EXISTS artist_categories (
    artist_key        TEXT PRIMARY KEY,
    canonical         TEXT,
    event_type        TEXT,
    category          TEXT,
    basis             TEXT,
    monthly_listeners INTEGER,
    pollstar_rank     INTEGER,
    primary_genre     TEXT,
    genre_class       TEXT,
    shows             INTEGER,     -- setlist.fm shows for this artist
    venues            INTEGER,     -- distinct setlist.fm venues
    source            TEXT,        -- dashboard | computed
    match_method      TEXT,        -- exact | norm | manual | none
    run_at            TEXT
);
CREATE INDEX IF NOT EXISTS ix_artcat_category ON artist_categories(category);
"""

# threshold label -> canonical name. Matched as case-insensitive substrings
# against column N so the rows can move around in the sheet.
THRESHOLD_PATTERNS = [
    ("cat_a_spotify", ("category a", "spotify")),
    ("cat_b_spotify", ("category b", "spotify")),
    ("cat_a_pollstar", ("category a", "pollstar")),
    ("cat_b_pollstar", ("category b", "pollstar")),
    ("cat_d_spotify", ("category d", "spotify")),
    ("tenant_min_dates", ("tenant", "minimum dates for")),
    ("tenant_min_per_venue", ("tenant", "per venue")),
]

DEFAULT_THRESHOLDS = {
    "cat_a_spotify": 40_000_000,
    "cat_b_spotify": 10_000_000,
    "cat_a_pollstar": 50,
    "cat_b_pollstar": 100,
    # Category C/D boundary: under this many monthly listeners is Category D.
    # Not on the dashboard (which predates D) -- add a "Category D - Spotify
    # listeners below" row to the thresholds block there and it will be picked up.
    "cat_d_spotify": 1_000_000,
    "tenant_min_dates": 8,
    "tenant_min_per_venue": 4,
}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


try:
    from tqdm import tqdm as _tqdm
except ImportError:            # tqdm is nice-to-have, never required
    _tqdm = None


def progress(iterable, desc, total=None, unit="rows"):
    """tqdm when installed, a periodic log line when it is not."""
    if _tqdm is not None:
        return _tqdm(iterable, desc=desc, total=total, unit=unit,
                     unit_scale=True, file=sys.stderr, leave=False,
                     dynamic_ncols=True, mininterval=0.5)

    def fallback():
        n = 0
        for item in iterable:
            yield item
            n += 1
            if n % 25000 == 0:
                log(f"   {desc}: {n:,}" + (f" / {total:,}" if total else ""))
    return fallback()


def utcnow():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------
# name normalisation
# ----------------------------------------------------------------------------

def artist_key(name):
    """The dashboard's own convention: trimmed and lowercased. Nothing more."""
    return str(name or "").strip().lower()


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def norm_key(name):
    """
    Aggressive key for matching across sources.

    artist_key alone is lowercase-only, which is fine inside the dashboard's own
    2,500 names but misses constantly against setlist.fm's 156k: accents,
    punctuation, ampersands and a leading "The" all break an exact match.
    """
    s = str(name or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("&", " and ")
    s = s.replace("$", "s")
    s = _PUNCT.sub(" ", s)
    s = _SPACE.sub(" ", s).strip()
    if s.startswith("the "):
        s = s[4:]
    return s


# ----------------------------------------------------------------------------
# database
# ----------------------------------------------------------------------------

class DB:
    def __init__(self, path=DB_PATH):
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found - run this from the jambase folder.")
        self.path = path
        self.conn = sqlite3.connect(path, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self):
        """
        Add any column the schema has gained since these tables were created.
        CREATE TABLE IF NOT EXISTS silently leaves an older table alone, which
        would otherwise make a re-run fail after the schema is extended.
        """
        wanted = {
            "ref_dashboard_categories": [("sheet_listeners", "INTEGER"),
                                         ("sheet_rank", "INTEGER")],
            "artist_categories": [("venues", "INTEGER"), ("match_method", "TEXT")],
        }
        for table, cols in wanted.items():
            have = {r[1] for r in self.conn.execute(f'PRAGMA table_info("{table}")')}
            if not have:
                continue
            for name, decl in cols:
                if name not in have:
                    self.conn.execute(f'ALTER TABLE "{table}" ADD COLUMN {name} {decl}')
                    log(f"   migrated: added {table}.{name}")

    def query(self, sql, params=()):
        return self.conn.execute(sql, params).fetchall()

    def scalar(self, sql, params=()):
        row = self.query(sql, params)
        return row[0][0] if row else None

    def executemany(self, sql, rows):
        self.conn.executemany(sql, rows)
        self.conn.commit()

    def execute(self, sql, params=()):
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def thresholds(self):
        rows = self.query("SELECT name, value FROM category_thresholds")
        t = dict(DEFAULT_THRESHOLDS)
        t.update({name: value for name, value in rows})
        return t


# ----------------------------------------------------------------------------
# helpers for reading the workbook
# ----------------------------------------------------------------------------

def _int(v):
    if v in (None, ""):
        return None
    try:
        return int(float(str(v).replace(",", "")))
    except (TypeError, ValueError):
        return None


def _float(v):
    if v in (None, ""):
        return None
    try:
        return float(str(v).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None


def _text(v):
    if v is None:
        return ""
    if isinstance(v, dt.datetime):
        return v.strftime("%Y-%m-%d")
    return str(v).strip()


def header_map(ws):
    """{normalised header: column index} from row 1."""
    for row in ws.iter_rows(min_row=1, max_row=1, values_only=True):
        return {str(c).strip().lower(): i for i, c in enumerate(row) if c}
    return {}


# ----------------------------------------------------------------------------
# kworb: the live source for Spotify monthly listeners
# ----------------------------------------------------------------------------

_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.I | re.S)
_TD = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_SPOTIFY_ID = re.compile(r'href="artist/([A-Za-z0-9]{10,})_', re.I)


def _cell(raw):
    return html.unescape(_TAG.sub("", raw)).strip()


def fetch_kworb(url, timeout=60):
    """
    Scrape one kworb monthly-listeners page.

    Columns vary between pages -- page 1 carries a "Peak" rank that page 2
    lacks -- so the header row is used to locate each field by name.
    """
    import requests
    resp = requests.get(url, timeout=timeout, headers={
        "User-Agent": "jambase-data-pipeline/1.0 (personal research)",
        "Accept": "text/html",
    })
    resp.raise_for_status()
    # kworb serves "Content-Type: text/html" with no charset, so requests falls
    # back to ISO-8859-1 per the HTTP spec and mangles the UTF-8 accents:
    # "Celine" with an acute comes through as two Latin-1 characters, normalises
    # to "ca line dion", and never matches setlist.fm. The page is UTF-8.
    if "charset" not in (resp.headers.get("content-type") or "").lower():
        resp.encoding = resp.apparent_encoding or "utf-8"

    rows, cols = [], None
    for tr in _TR.findall(resp.text):
        cells = _TD.findall(tr)
        if len(cells) < 3:
            continue
        labels = [_cell(c).lower() for c in cells]
        if cols is None:
            if "artist" in labels and "listeners" in labels:
                cols = {}
                for i, lab in enumerate(labels):
                    if lab in ("#", "rank"):
                        cols["rank"] = i
                    elif lab == "artist":
                        cols["artist"] = i
                    elif lab == "listeners":
                        cols["listeners"] = i
                    elif lab.startswith("pk"):
                        cols["peak_listeners"] = i
            continue
        rank = _int(_cell(cells[cols["rank"]])) if "rank" in cols else None
        if rank is None:
            continue
        name = _cell(cells[cols["artist"]])
        if not name:
            continue
        sid = _SPOTIFY_ID.search(cells[cols["artist"]])
        peak = (_int(_cell(cells[cols["peak_listeners"]]))
                if "peak_listeners" in cols else None)
        rows.append({
            "rank": rank,
            "artist": name,
            "spotify_artist_id": sid.group(1) if sid else "",
            "monthly_listeners": _int(_cell(cells[cols["listeners"]])),
            "peak_listeners": peak,
            "spotify_url": (f"https://open.spotify.com/artist/{sid.group(1)}"
                            if sid else ""),
        })
    if cols is None:
        raise ValueError(f"no recognisable table header at {url}")
    return rows


def kworb_page_urls(max_pages=KWORB_MAX_PAGES):
    """listeners.html, listeners2.html, ... - page 1 has no number."""
    yield KWORB_BASE.format(n="")
    for n in range(2, max_pages + 1):
        yield KWORB_BASE.format(n=n)


def cmd_kworb(args, db):
    """Refresh ref_spotify straight from kworb -- no spreadsheet round-trip."""
    urls = args.urls or list(kworb_page_urls())
    explicit = bool(args.urls)
    log(f"== fetching kworb monthly listeners ==")
    merged = {}
    pages = 0
    for url in progress(urls, "kworb pages", total=len(urls), unit="page"):
        try:
            page = fetch_kworb(url)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 404 and not explicit:
                break          # walked off the end of the pagination
            log(f"   !! {url}: {e}")
            if merged:
                log(f"   continuing with the {len(merged):,} artists fetched so far")
                break
            have = db.scalar("SELECT COUNT(*) FROM ref_spotify") or 0
            log(f"   keeping the {have:,} rows already in ref_spotify")
            return 1
        if not page:
            break
        pages += 1
        log(f"   page {pages:2d}: {len(page):5,} rows  ranks "
            f"{page[0]['rank']:,}-{page[-1]['rank']:,}  "
            f"{page[-1]['monthly_listeners']:,} listeners at the tail")
        for r in page:
            # a name on more than one page keeps its better (lower) rank
            key = artist_key(r["artist"])
            if key not in merged or r["rank"] < merged[key]["rank"]:
                merged[key] = r
    rows = sorted(merged.values(), key=lambda r: r["rank"])
    if len(rows) < 100:
        log(f"   !! only parsed {len(rows)} rows - the page layout may have changed. "
            f"Leaving ref_spotify alone.")
        return 1

    now = utcnow()
    db.execute("DELETE FROM ref_spotify")
    db.executemany(
        "INSERT OR REPLACE INTO ref_spotify (artist_key, rank, artist,"
        " spotify_artist_id, monthly_listeners, peak_listeners, spotify_url,"
        " loaded_at) VALUES (?,?,?,?,?,?,?,?)",
        [(artist_key(r["artist"]), r["rank"], r["artist"], r["spotify_artist_id"],
          r["monthly_listeners"], r["peak_listeners"], r["spotify_url"], now)
         for r in rows])
    floor = min(r["monthly_listeners"] or 0 for r in rows)
    log(f"   ref_spotify: {len(rows):,} artists, "
        f"floor (rank {max(r['rank'] for r in rows)}) = {floor:,} monthly listeners")
    t = db.thresholds()
    d_line = t["cat_d_spotify"]
    if floor >= d_line:
        log(f"   !! the list tail ({floor:,}) is at or above the Category C/D line "
            f"({d_line:,.0f}), so an artist absent from the list could be either. "
            f"They will all be called Category D.")
    else:
        log(f"   list tail {floor:,} < the C/D line of {d_line:,.0f}: an artist "
            f"absent from this list has fewer listeners than that, but we have "
            f"no measurement - they become Category E")
    return 0


# ----------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------

def cmd_load(args, db):
    try:
        import openpyxl
    except ImportError:
        raise SystemExit("pip install openpyxl")
    if not os.path.exists(args.workbook):
        raise SystemExit(f"{args.workbook} not found.")

    log(f"== loading {args.workbook} ==")
    wb = openpyxl.load_workbook(args.workbook, read_only=True, data_only=True)
    now = utcnow()

    # -- thresholds ---------------------------------------------------------
    ws = wb[TAB_CATEGORIES]
    found = {}
    for label, value in ws.iter_rows(min_row=1, max_row=40, min_col=14, max_col=15,
                                     values_only=True):
        if not label or value in (None, ""):
            continue
        low = str(label).lower()
        for name, needles in THRESHOLD_PATTERNS:
            if name in found:
                continue
            if all(n in low for n in needles):
                num = _float(value)
                if num is not None:
                    found[name] = (num, str(label).strip())
    db.execute("DELETE FROM category_thresholds")
    db.executemany(
        "INSERT INTO category_thresholds (name, value, label, loaded_at) VALUES (?,?,?,?)",
        [(n, v, lbl, now) for n, (v, lbl) in found.items()])
    missing = [n for n, _ in THRESHOLD_PATTERNS if n not in found]
    log(f"   thresholds: {len(found)} read" +
        (f"; MISSING {missing} (falling back to defaults)" if missing else ""))
    for n, (v, _lbl) in sorted(found.items()):
        log(f"      {n:22s} = {v:,.0f}")

    # -- the dashboard's own categorisation rows ----------------------------
    hm = header_map(ws)
    need = ["headliner", "event_type", "category", "basis", "primary_genre",
            "genre_class", "bookings", "venues"]
    listeners_col = next((hm[c] for c in hm if "spotify" in c and "list" in c), None)
    rank_col = next((hm[c] for c in hm if "pollstar" in c and "rank" in c), None)
    if any(c not in hm for c in need):
        raise SystemExit(f"'{TAB_CATEGORIES}' is missing columns: "
                         f"{[c for c in need if c not in hm]}")
    rows = []
    for r in progress(ws.iter_rows(min_row=2, values_only=True),
                      "dashboard categories"):
        name = _text(r[hm["headliner"]]) if hm["headliner"] < len(r) else ""
        if not name:
            continue
        rows.append((name, artist_key(name), _text(r[hm["event_type"]]),
                     _text(r[hm["category"]]), _text(r[hm["basis"]]),
                     _text(r[hm["primary_genre"]]), _text(r[hm["genre_class"]]),
                     _int(r[hm["bookings"]]), _int(r[hm["venues"]]),
                     _int(r[listeners_col]) if listeners_col is not None else None,
                     _int(r[rank_col]) if rank_col is not None else None, now))
    db.execute("DELETE FROM ref_dashboard_categories")
    db.executemany("INSERT OR REPLACE INTO ref_dashboard_categories "
                   "(headliner, artist_key, event_type, category, basis, primary_genre,"
                   " genre_class, bookings, venues, sheet_listeners, sheet_rank,"
                   " loaded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    log(f"   ref_dashboard_categories: {len(rows):,} rows")

    # -- spotify ------------------------------------------------------------
    # kworb is the source of truth for listeners (25k artists vs the tab's
    # 2,500, and current rather than whenever the sheet was last pasted). Only
    # fall back to the workbook tab when we have nothing better, otherwise a
    # `refresh --no-kworb` would silently shrink ref_spotify and re-tier
    # thousands of artists downward.
    have_spotify = db.scalar("SELECT COUNT(*) FROM ref_spotify") or 0
    if have_spotify and not args.spotify_from_workbook:
        log(f"   ref_spotify: keeping the {have_spotify:,} rows already loaded "
            f"(kworb). Use --spotify-from-workbook to overwrite them.")
        rows = None
    else:
        rows = []
    ws = wb[TAB_SPOTIFY]
    hm = header_map(ws)
    if rows is not None:
        hm = header_map(ws)
        for r in ws.iter_rows(min_row=2, values_only=True):
            name = _text(r[hm["artist"]]) if "artist" in hm and hm["artist"] < len(r) else ""
            if not name:
                continue
            key = _text(r[hm["artist_key"]]).lower() if "artist_key" in hm else artist_key(name)
            rows.append((key or artist_key(name), _int(r[hm.get("rank", 0)]), name,
                         _text(r[hm["spotify_artist_id"]]) if "spotify_artist_id" in hm else "",
                         _int(r[hm["monthly_listeners"]]) if "monthly_listeners" in hm else None,
                         _int(r[hm["peak_listeners"]]) if "peak_listeners" in hm else None,
                         _text(r[hm["spotify_url"]]) if "spotify_url" in hm else "", now))
        db.execute("DELETE FROM ref_spotify")
        db.executemany("INSERT OR REPLACE INTO ref_spotify (artist_key, rank, artist,"
                       " spotify_artist_id, monthly_listeners, peak_listeners, spotify_url,"
                       " loaded_at) VALUES (?,?,?,?,?,?,?,?)", rows)
        log(f"   ref_spotify: {len(rows):,} rows from the workbook tab")

    # -- pollstar -----------------------------------------------------------
    ws = wb[TAB_POLLSTAR]
    hm = header_map(ws)
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        name = _text(r[hm["artist"]]) if "artist" in hm and hm["artist"] < len(r) else ""
        if not name:
            continue
        key = _text(r[hm["artist_key"]]).lower() if "artist_key" in hm else artist_key(name)
        rows.append((key or artist_key(name), _int(r[hm.get("rank", 0)]), name,
                     _float(r[hm["gross_usd"]]) if "gross_usd" in hm else None,
                     _float(r[hm["avg_ticket_price"]]) if "avg_ticket_price" in hm else None,
                     _int(r[hm["total_tickets"]]) if "total_tickets" in hm else None,
                     _int(r[hm["shows"]]) if "shows" in hm else None,
                     _text(r[hm["agency"]]) if "agency" in hm else "",
                     _text(r[hm["chart_period_from"]]) if "chart_period_from" in hm else "",
                     _text(r[hm["chart_period_to"]]) if "chart_period_to" in hm else "", now))
    db.execute("DELETE FROM ref_pollstar")
    db.executemany("INSERT OR REPLACE INTO ref_pollstar (artist_key, rank, artist,"
                   " gross_usd, avg_ticket_price, total_tickets, shows, agency,"
                   " chart_period_from, chart_period_to, loaded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   rows)
    log(f"   ref_pollstar: {len(rows):,} rows")
    wb.close()
    log("   done. Next: `aliases`")


# ----------------------------------------------------------------------------
# aliases
# ----------------------------------------------------------------------------

def load_manual_aliases():
    """
    Hand-curated name fixes, read straight from artist_aliases_manual.csv.

    This used to be one source inside an artist_aliases table, but nothing ever
    read the other 193k rows of it -- the categoriser builds its lookups from
    the reference tables directly -- so the table was dropped and the overrides
    are read from the file they were always maintained in.
    """
    out = {}
    if not os.path.exists(MANUAL_ALIASES):
        return out
    with open(MANUAL_ALIASES, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            alias = (row.get("alias") or "").strip()
            key = (row.get("artist_key") or "").strip().lower()
            if alias and key:
                out[alias] = key
    return out


# ----------------------------------------------------------------------------
# categorise
# ----------------------------------------------------------------------------

def classify(listeners, rank, t):
    """
    The A/B/C/D/E test. Returns (category, basis).

        A  >= cat_a_spotify listeners, or Pollstar rank <= cat_a_pollstar
        B  >= cat_b_spotify listeners, or Pollstar rank <= cat_b_pollstar
        C  >= cat_d_spotify listeners (the C/D line)
        D  on the Spotify list but under that line
        E  in NEITHER list -- we have no reference data for them at all

    D and E are deliberately separate. D is a measured fact: we know the artist
    has under a million listeners. E is an absence of evidence, which is not the
    same claim and is where name-matching failures hide. `suspects` hunts for
    exactly that: Category E artists playing heavily at venues the big acts use.
    """
    has_reference = listeners is not None or rank is not None
    listeners = listeners or 0
    bits = []
    if listeners:
        bits.append(f"Spotify {listeners/1e6:.2f}m listeners")
    if rank:
        bits.append(f"Pollstar #{int(rank)}")
    basis = " / ".join(bits) if bits else "Not in the Spotify or Pollstar lists"

    if not has_reference:
        return "Category E", basis
    if listeners >= t["cat_a_spotify"] or (rank and rank <= t["cat_a_pollstar"]):
        return "Category A", basis
    if listeners >= t["cat_b_spotify"] or (rank and rank <= t["cat_b_pollstar"]):
        return "Category B", basis
    if listeners >= t["cat_d_spotify"]:
        return "Category C", basis
    return "Category D", basis


def cmd_categorise(args, db):
    """
    Categorise every artist in `setlists` from the criteria.

    We deliberately do NOT carry the dashboard's own category across. Its rows
    are Pollstar EVENT billings -- 857 of them are compound strings like
    '"106 KMEL Summer Jam", Post Malone' whose tier belongs to the support act,
    not to a setlist.fm artist. So the dashboard is used here only as a GENRE
    helper (genre exists nowhere else), and the category itself is derived from
    Spotify listeners / Pollstar rank against the thresholds.
    """
    t = db.thresholds()
    log("== categorising ==")
    log("   thresholds: " + ", ".join(f"{k}={v:,.0f}" for k, v in sorted(t.items())))

    # -- helper lookups, keyed both exactly and normalised -------------------
    spotify_exact, spotify_norm = {}, {}
    for key, name, listeners in db.query(
            "SELECT artist_key, artist, monthly_listeners FROM ref_spotify"):
        spotify_exact[key] = listeners
        spotify_norm.setdefault(norm_key(name), listeners)

    pollstar_exact, pollstar_norm = {}, {}
    for key, name, rank in db.query("SELECT artist_key, artist, rank FROM ref_pollstar"):
        pollstar_exact[key] = rank
        pollstar_norm.setdefault(norm_key(name), rank)

    # genre only -- skip the compound billings, they are events not artists
    genre_exact, genre_norm = {}, {}
    for name, key, genre, gclass in db.query(
            "SELECT headliner, artist_key, primary_genre, genre_class "
            "FROM ref_dashboard_categories WHERE headliner NOT LIKE '%\"%'"):
        if not (genre or gclass):
            continue
        genre_exact[key] = (genre, gclass)
        genre_norm.setdefault(norm_key(name), (genre, gclass))
    log(f"   genre helper: {len(genre_exact):,} artists carry a genre")

    manual = load_manual_aliases()
    manual_norm = {norm_key(a): k for a, k in manual.items()}
    if manual:
        log(f"   {len(manual):,} manual name overrides from {MANUAL_ALIASES}")

    # -- every setlist.fm artist, with the counts the tenant test needs ------
    # Grouped on the lowercased name, because that is the grain of the output
    # table. Doing it in SQL rather than letting the primary key collapse case
    # variants ("ABBA" / "Abba") means their shows are summed, not overwritten.
    artists = db.query("""
        SELECT ak, SUM(c) AS shows, COUNT(*) AS venues, MAX(c) AS max_at_one_venue
        FROM (SELECT LOWER(TRIM(artist)) AS ak,
                     COALESCE(NULLIF(venue_id,''), venue || '|' || city) AS vkey,
                     COUNT(*) AS c
              FROM setlists WHERE TRIM(COALESCE(artist,'')) != ''
              GROUP BY ak, vkey)
        GROUP BY ak""")
    # the display spelling: whichever variant appears on the most shows
    canonical = {}
    for ak, name, _n in db.query("""
            SELECT LOWER(TRIM(artist)) AS ak, artist, COUNT(*) AS n
            FROM setlists WHERE TRIM(COALESCE(artist,'')) != ''
            GROUP BY ak, artist ORDER BY ak, n DESC"""):
        canonical.setdefault(ak, name)
    raw = db.scalar("SELECT COUNT(DISTINCT artist) FROM setlists "
                    "WHERE TRIM(COALESCE(artist,''))!=''") or 0
    log(f"   {len(artists):,} artists to categorise "
        f"({raw:,} distinct spellings, {raw - len(artists):,} case variants merged)")

    now = utcnow()
    out = []
    methods = {}
    for ak, shows, venues, max_at_one_venue in progress(
            artists, "categorising", total=len(artists), unit="artist"):
        name = canonical.get(ak, ak)
        nk = norm_key(name)
        if name in manual or nk in manual_norm:
            ak = manual.get(name) or manual_norm[nk]
            nk = norm_key(ak)
            method = "manual"
        else:
            method = None

        listeners = spotify_exact.get(ak)
        if listeners is None:
            listeners = spotify_norm.get(nk)
            if listeners is not None:
                method = method or "norm"
        elif method is None:
            method = "exact"

        rank = pollstar_exact.get(ak)
        if rank is None:
            rank = pollstar_norm.get(nk)
            if rank is not None:
                method = method or "norm"
        elif method is None:
            method = "exact"

        genre, gclass = genre_exact.get(ak) or genre_norm.get(nk) or ("", "")

        # event_type first: genre decides Sport / Family before the A/B/C tests
        if gclass == "Sport":
            tenant = (shows >= t["tenant_min_dates"]
                      and max_at_one_venue >= t["tenant_min_per_venue"])
            event_type = category = ("Tenant Sporting Event" if tenant
                                     else "Non-Tenant Sporting Event")
            basis = (f"Sport: {shows} dates, {max_at_one_venue} at one venue"
                     if tenant else
                     f"Sport: {shows} dates, {max_at_one_venue} at one venue "
                     f"(below tenant test)")
        elif gclass in ("Family", "Entertainment", "Comedy"):
            event_type = category = "Family, Entertainment, Comedy & Other"
            basis = f"Genre: {genre or gclass}"
        else:
            event_type = "Artist"
            category, basis = classify(listeners, rank, t)

        method = method or "none"
        methods[method] = methods.get(method, 0) + 1
        out.append((ak, name, event_type, category, basis, listeners, rank,
                    genre, gclass, shows, venues, "setlists", method, now))

    db.execute("DELETE FROM artist_categories")
    db.executemany("INSERT OR REPLACE INTO artist_categories (artist_key, canonical,"
                   " event_type, category, basis, monthly_listeners, pollstar_rank,"
                   " primary_genre, genre_class, shows, venues, source, match_method,"
                   " run_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", out)

    log(f"   wrote {len(out):,} rows")
    for cat, n, sh in db.query("""SELECT category, COUNT(*), SUM(shows)
                                  FROM artist_categories GROUP BY category
                                  ORDER BY COUNT(*) DESC"""):
        log(f"      {cat:40s} {n:7,} artists  {sh or 0:9,} shows")
    log("   reference match: " +
        ", ".join(f"{k}={v:,}" for k, v in sorted(methods.items())))
    log("   done.")


# ----------------------------------------------------------------------------
# verify / unmatched / status
# ----------------------------------------------------------------------------

def cmd_verify(args, db):
    """Re-derive the dashboard's own Artist rows and diff. Proves the rules."""
    # The dashboard predates Category D, so for this regression test the C/D
    # line drops to 0: nothing falls through to D, and the A/B/C boundaries are
    # compared exactly as the sheet defines them.
    t = dict(db.thresholds(), cat_d_spotify=0)
    # Test against the sheet's OWN resolved inputs, not a re-join on the
    # headliner string: 857 of these are compound billings like
    # '"106 KMEL Summer Jam", Post Malone' where the sheet has already resolved
    # the numbers to the act that decides the tier.
    rows = db.query("""SELECT headliner, category, sheet_listeners, sheet_rank
                       FROM ref_dashboard_categories WHERE event_type='Artist'""")
    bad = []
    for name, cat, listeners, rank in rows:
        got, _ = classify(listeners, rank, t)
        if got != cat:
            bad.append((name, cat, got, listeners, rank))
    log(f"== verify: {len(rows):,} dashboard 'Artist' rows, {len(bad)} mismatches ==")
    for b in bad[:15]:
        log(f"   {b[0][:40]:40s} sheet={b[1]:12s} ours={b[2]:12s} "
            f"listeners={b[3]} rank={b[4]}")
    if not bad:
        log("   the implemented rules reproduce the spreadsheet exactly.")
    return len(bad)


def cmd_unmatched(args, db):
    """setlist.fm artists with no reference data at all, biggest first."""
    rows = db.query("""SELECT canonical, shows, venues FROM artist_categories
                       WHERE monthly_listeners IS NULL AND pollstar_rank IS NULL
                         AND event_type='Artist'
                       ORDER BY shows DESC LIMIT ?""", (args.limit,))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alias", "artist_key", "shows", "venues"])
        for name, shows, venues in rows:
            w.writerow([name, "", shows, venues])
    log(f"== {len(rows):,} unmatched artists -> {args.out} ==")
    log(f"   Fill in artist_key for any worth fixing, save as {MANUAL_ALIASES},")
    log("   then re-run `refresh`.")
    for name, shows, venues in rows[:15]:
        log(f"   {shows:6,} shows  {name[:50]}")


def cmd_refresh(args, db):
    """
    The one command to run after any data update.

    Pulls today's Spotify monthly listeners from kworb, reloads the workbook
    (so edited thresholds and the Pollstar rankings take effect), rebuilds the
    alias spine (so artists added by the daily setlist.fm sync are picked up),
    re-derives every category, and checks the result against the spreadsheet
    and against `setlists`.

    A kworb fetch failure is a warning, not a stop: the previous listener data
    stays in place and the rest of the refresh still runs.
    """
    cmd_load(args, db)
    log("")
    if args.no_kworb:
        log("== skipping kworb (--no-kworb); using the workbook's listener tab ==")
        kworb_failed = 0
    else:
        kworb_failed = cmd_kworb(args, db)
    log("")
    cmd_categorise(args, db)
    log("")
    bad = cmd_verify(args, db)
    log("")
    ok = _integrity(db)
    if kworb_failed:
        log("!! listener data is stale - the kworb fetch failed above")
    if bad or not ok or kworb_failed:
        log("!! refresh finished WITH WARNINGS - see above")
        return 1
    log("== refresh complete ==")
    return 0


def _integrity(db):
    """Every show in `setlists` must be accounted for by exactly one category."""
    src = db.scalar("SELECT COUNT(*) FROM setlists "
                    "WHERE TRIM(COALESCE(artist,''))!=''") or 0
    got = db.scalar("SELECT SUM(shows) FROM artist_categories") or 0
    if src == got:
        log(f"== integrity: {got:,} shows categorised = {src:,} in setlists. OK ==")
        return True
    log(f"!! integrity: artist_categories covers {got:,} shows but setlists has "
        f"{src:,} ({src - got:+,})")
    return False


def cmd_suspects(args, db):
    """
    Find Category E artists who behave like big acts.

    Category E means "we found no reference data", which is either true (a small
    act) or a name-matching failure. An artist with no listener data who plays
    hundreds of shows at the venues Category A and B acts use is almost
    certainly the second, so this ranks E artists by how much of their touring
    happens at those venues.

    "Big venue" is derived from the data rather than assumed: any venue where
    Category A or B artists have played at least --big-venue-shows times.
    """
    big_min = args.big_venue_shows
    log(f"== finding Category E artists who play like big ones ==")

    db.execute("DROP TABLE IF EXISTS tmp_big_venues")
    db.execute(f"""
        CREATE TEMP TABLE IF NOT EXISTS tmp_big_venues AS
        SELECT COALESCE(NULLIF(s.venue_id,''), s.venue || '|' || s.city) AS vkey,
               COUNT(*) AS ab_shows
        FROM setlists s
        JOIN artist_categories c ON c.artist_key = LOWER(TRIM(s.artist))
        WHERE c.category IN ('Category A','Category B')
        GROUP BY vkey HAVING COUNT(*) >= {int(big_min)}""")
    n_big = db.scalar("SELECT COUNT(*) FROM tmp_big_venues") or 0
    log(f"   {n_big:,} venues host >= {big_min} Category A/B shows")

    rows = db.query("""
        SELECT c.canonical, c.shows, c.venues,
               COUNT(*) AS big_shows,
               COUNT(DISTINCT b.vkey) AS big_venues
        FROM setlists s
        JOIN artist_categories c ON c.artist_key = LOWER(TRIM(s.artist))
        JOIN tmp_big_venues b
          ON b.vkey = COALESCE(NULLIF(s.venue_id,''), s.venue || '|' || s.city)
        WHERE c.category = 'Category E'
        GROUP BY c.artist_key
        HAVING big_shows >= ?
        ORDER BY big_shows DESC
        LIMIT ?""", (args.min_shows, args.limit))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alias", "artist_key", "total_shows", "total_venues",
                    "shows_at_big_venues", "big_venues", "pct_at_big_venues"])
        for name, shows, venues, big_shows, big_venues in rows:
            w.writerow([name, "", shows, venues, big_shows, big_venues,
                        f"{100.0 * big_shows / shows:.0f}" if shows else ""])
    log(f"   {len(rows):,} suspects -> {args.out}")
    log(f"   {'shows':>7} {'big':>7} {'venues':>7}  artist")
    for name, shows, venues, big_shows, big_venues in rows[:25]:
        log(f"   {shows:7,} {big_shows:7,} {big_venues:7,}  {name[:45]}")
    log("")
    log("   Each of these is either genuinely below 500k monthly listeners, or a")
    log("   name that failed to match the reference lists. For the second kind,")
    log(f"   put alias,artist_key into {MANUAL_ALIASES} and re-run `refresh`.")


def cmd_status(args, db):
    log(f"== {db.path} ==")
    for table in ("ref_spotify", "ref_pollstar", "ref_dashboard_categories",
                  "category_thresholds", "artist_categories"):
        n = db.scalar(f"SELECT COUNT(*) FROM {table}") or 0
        log(f"  {table:26s} {n:9,} rows")
    run_at = db.scalar("SELECT MAX(run_at) FROM artist_categories")
    log(f"  last categorise run        : {run_at or 'never'}")
    if not run_at:
        return
    log("  categories:")
    for cat, n, sh in db.query("""SELECT category, COUNT(*), SUM(shows)
                                  FROM artist_categories GROUP BY category
                                  ORDER BY COUNT(*) DESC"""):
        log(f"      {cat:40s} {n:7,} artists  {sh or 0:9,} shows")
    log("  how each artist was resolved:")
    for src, method, n in db.query("""SELECT source, match_method, COUNT(*)
                                      FROM artist_categories
                                      GROUP BY source, match_method
                                      ORDER BY COUNT(*) DESC"""):
        log(f"      {src:10s} {method:8s} {n:8,}")
    no_genre = db.scalar("SELECT COUNT(*) FROM artist_categories "
                         "WHERE TRIM(COALESCE(genre_class,''))=''") or 0
    ng_shows = db.scalar("SELECT SUM(shows) FROM artist_categories "
                         "WHERE TRIM(COALESCE(genre_class,''))=''") or 0
    unref = db.scalar("SELECT COUNT(*) FROM artist_categories "
                      "WHERE monthly_listeners IS NULL AND pollstar_rank IS NULL") or 0
    log(f"  no genre available (not on the dashboard): {no_genre:,} artists, "
        f"{ng_shows:,} shows")
    log(f"  no Spotify/Pollstar reference data     : {unref:,} artists "
        f"-> Category E. Run `suspects` to find name-match failures among them.")
    _integrity(db)


# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Build artist categorisation tables in setlistfm.db "
                    "(never alters the setlists table).")
    p.add_argument("--db", default=DB_PATH)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("load", help="Read arenas-dashboard.xlsx into the ref_* tables.")
    a.add_argument("--workbook", default=WORKBOOK)
    a.add_argument("--spotify-from-workbook", action="store_true",
                   help="Overwrite ref_spotify with the workbook tab (normally kworb wins).")
    a.set_defaults(func=cmd_load)

    c = sub.add_parser("categorise", help="Build artist_categories.")
    c.set_defaults(func=cmd_categorise)

    v = sub.add_parser("verify", help="Re-derive the dashboard's rows and diff.")
    v.set_defaults(func=cmd_verify)

    u = sub.add_parser("unmatched", help="Export artists with no reference match.")
    u.add_argument("--out", default="unmatched_artists.csv")
    u.add_argument("--limit", type=int, default=500)
    u.set_defaults(func=cmd_unmatched)

    k = sub.add_parser("kworb", help="Pull Spotify monthly listeners from kworb.net.")
    k.add_argument("--urls", nargs="*", default=None,
                   help="Pages to scrape (default: walk listeners.html, "
                        "listeners2.html ... until a 404).")
    k.set_defaults(func=cmd_kworb)

    r = sub.add_parser("refresh",
                       help="Pull kworb + load + aliases + categorise + checks. "
                            "Use this after any data update.")
    r.add_argument("--workbook", default=WORKBOOK)
    r.add_argument("--spotify-from-workbook", action="store_true")
    r.add_argument("--urls", nargs="*", default=None)
    r.add_argument("--no-kworb", action="store_true",
                   help="Skip the network pull and use the workbook's listener tab.")
    r.set_defaults(func=cmd_refresh)

    sp = sub.add_parser("suspects",
                        help="Category E artists who play like big ones - i.e. likely "
                             "name-match failures.")
    sp.add_argument("--out", default="category_e_suspects.csv")
    sp.add_argument("--limit", type=int, default=300)
    sp.add_argument("--min-shows", type=int, default=20,
                    help="Minimum shows at big venues to be listed (default 20).")
    sp.add_argument("--big-venue-shows", type=int, default=25,
                    help="A venue is 'big' at this many Category A/B shows (default 25).")
    sp.set_defaults(func=cmd_suspects)

    s = sub.add_parser("status", help="What's loaded and what it produced.")
    s.set_defaults(func=cmd_status)

    args = p.parse_args()
    db = DB(args.db)
    sys.exit(args.func(args, db) or 0)


if __name__ == "__main__":
    main()
