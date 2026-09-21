#!/usr/bin/env python3
"""
setlist.fm -> SQLite pipeline: full backfill + daily auto-sync.

Companion to setlistfm_harvest.py (which is artist-driven and CSV-based). This
one is *date-driven* and keeps a real database, so it can hold every setlist on
setlist.fm and stay current without ever knowing an artist name up front.

How it gets everything
----------------------
* /search/setlists?date=dd-MM-yyyy returns every setlist for that calendar day,
  any artist, any venue. Walking every date from a year ahead (setlist.fm holds
  upcoming shows) back to the 1960s therefore covers the whole site:
  ~10.5M setlists / 20 per page ~= 525k requests ~= 11 days at 50k/day.
* /search/setlists?lastUpdated=yyyyMMddHHmmss returns every setlist created OR
  EDITED since a timestamp. ~4-5k/day, ~250 requests. That's the daily sync;
  it also picks up corrections to shows we already have.

Everything is an upsert keyed on setlist_id, so import/backfill/sync can be run
in any order, any number of times, and interrupted at any point.

QUICK START
-----------
    pip install requests

    python setlistfm_db.py import                 # load your existing CSVs (once)
    python setlistfm_db.py backfill --workers 12  # everything not in the DB; Ctrl-C any time
    python setlistfm_db.py sync                   # daily: new shows + edits
    python setlistfm_db.py export                 # -> with_cities_setlistfm_db.csv
    python setlistfm_db.py status

Schedule `sync` daily (Windows Task Scheduler; run once from any terminal):
    schtasks /Create /SC DAILY /ST 03:00 /TN "setlistfm sync" /TR "cmd /c cd /d C:\\Coding\\jambase && python setlistfm_db.py sync && python setlistfm_db.py export"

The API key is resolved the same way as setlistfm_harvest.py (setlistfm_key.txt).
"""

import argparse
import csv
import datetime as dt
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse the harvester's client (rate limiter, 429 handling, daily-cap plumbing)
# and its CSV column layout so exports stay byte-compatible.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from setlistfm_harvest import (  # noqa: E402
    BudgetExhausted, CSV_COLUMNS, DAILY_CAP, DEFAULT_RATE, MAX_RATE, STATE_DIR,
    SetlistFM, flatten, log, mask, resolve_api_key,
)

# resolved against the project folder, not the working directory -- see paths.py
DB_PATH = paths.DB
DEFAULT_EXPORT = paths.WITH_CITIES_CSV
LEGACY_CSVS = [paths.LEGACY_TOURS_CSV, paths.TOURS_CSV]
HARVESTER_USAGE = os.path.join(STATE_DIR, "daily_usage.json")

SEARCH_CAP = 10_000          # the search endpoint never returns more than this
EARLIEST_DATE = "1960-01-01"
FUTURE_DAYS = 400            # setlist.fm holds upcoming shows; look this far ahead

SCHEMA = """
CREATE TABLE IF NOT EXISTS setlists (
    setlist_id   TEXT PRIMARY KEY,
    artist       TEXT,
    artist_mbid  TEXT,
    eventDate    TEXT,
    date_iso     TEXT,
    tour         TEXT,
    venue        TEXT,
    venue_id     TEXT,
    city         TEXT,
    state        TEXT,
    stateCode    TEXT,
    country      TEXT,
    countryCode  TEXT,
    latitude     TEXT,
    longitude    TEXT,
    num_songs    INTEGER,
    setlist_url  TEXT,
    last_updated TEXT,          -- setlist.fm's lastUpdated; NULL for CSV imports
    source       TEXT,          -- csv | backfill | sync
    fetched_at   TEXT
);
CREATE INDEX IF NOT EXISTS ix_setlists_date   ON setlists(date_iso);
CREATE INDEX IF NOT EXISTS ix_setlists_artist ON setlists(artist, tour, date_iso);
CREATE INDEX IF NOT EXISTS ix_setlists_venue  ON setlists(venue_id);

-- optional: per-song rows (--songs on backfill/sync)
CREATE TABLE IF NOT EXISTS songs (
    setlist_id  TEXT,
    position    INTEGER,
    set_no      INTEGER,
    encore      INTEGER,
    name        TEXT,
    cover_of    TEXT,
    with_artist TEXT,
    tape        INTEGER,
    info        TEXT,
    PRIMARY KEY (setlist_id, position)
);

-- one row per calendar date the backfill has looked at
CREATE TABLE IF NOT EXISTS backfill_dates (
    date_iso    TEXT PRIMARY KEY,
    total       INTEGER,        -- what the API said was on that day
    pages_done  INTEGER DEFAULT 0,
    done        INTEGER DEFAULT 0,
    capped      INTEGER DEFAULT 0,  -- total exceeded the 10k search cap
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS sync_log (
    started_at  TEXT,
    since       TEXT,           -- lastUpdated cutoff sent to the API
    total       INTEGER,
    upserted    INTEGER,
    requests    INTEGER,
    completed   INTEGER
);

CREATE TABLE IF NOT EXISTS quota (
    day  TEXT PRIMARY KEY,
    used INTEGER
);
"""

UPSERT = """
INSERT INTO setlists (setlist_id, artist, artist_mbid, eventDate, date_iso, tour,
    venue, venue_id, city, state, stateCode, country, countryCode, latitude,
    longitude, num_songs, setlist_url, last_updated, source, fetched_at)
VALUES (:setlist_id, :artist, :artist_mbid, :eventDate, :date_iso, :tour,
    :venue, :venue_id, :city, :state, :stateCode, :country, :countryCode, :latitude,
    :longitude, :num_songs, :setlist_url, :last_updated, :source, :fetched_at)
ON CONFLICT(setlist_id) DO UPDATE SET
    artist=excluded.artist, artist_mbid=excluded.artist_mbid,
    eventDate=excluded.eventDate, date_iso=excluded.date_iso, tour=excluded.tour,
    venue=excluded.venue, venue_id=excluded.venue_id, city=excluded.city,
    state=excluded.state, stateCode=excluded.stateCode, country=excluded.country,
    countryCode=excluded.countryCode, latitude=excluded.latitude,
    longitude=excluded.longitude, num_songs=excluded.num_songs,
    setlist_url=excluded.setlist_url, last_updated=excluded.last_updated,
    source=excluded.source, fetched_at=excluded.fetched_at
"""


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def today_utc():
    return utcnow().strftime("%Y-%m-%d")


# ----------------------------------------------------------------------------
# database
# ----------------------------------------------------------------------------

class DB:
    """One connection, one lock. Workers fetch in parallel, writes serialise."""

    def __init__(self, path=DB_PATH):
        self.path = path
        if not os.path.exists(path):
            log(f"!! {path} does not exist - creating a new, empty database. "
                f"If that is not what you meant, check JAMBASE_DB or --db.")
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()

    def upsert_setlists(self, rows, songs=None):
        if not rows:
            return 0
        with self.lock:
            self.conn.executemany(UPSERT, rows)
            if songs:
                self.conn.executemany(
                    "DELETE FROM songs WHERE setlist_id=?",
                    [(r["setlist_id"],) for r in rows])
                self.conn.executemany(
                    "INSERT OR REPLACE INTO songs VALUES (?,?,?,?,?,?,?,?,?)", songs)
            self.conn.commit()
        return len(rows)

    def execute(self, sql, params=()):
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def query(self, sql, params=()):
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def scalar(self, sql, params=()):
        row = self.query(sql, params)
        return row[0][0] if row else None

    # -- daily quota, shared-ish with the harvester --------------------------

    def harvester_used_today(self):
        """Requests setlistfm_harvest.py has spent today, so the two tools
        don't jointly blow the 50k/day limit."""
        try:
            with open(HARVESTER_USAGE, encoding="utf-8") as f:
                return int(json.load(f).get(today_utc(), 0))
        except (OSError, ValueError):
            return 0

    def used_today(self):
        return (self.scalar("SELECT used FROM quota WHERE day=?", (today_utc(),)) or 0)

    def spend_daily(self, n=1, cap=DAILY_CAP):
        """Same interface SetlistFM expects from the harvester's State."""
        day = today_utc()
        with self.lock:
            used = self.conn.execute("SELECT used FROM quota WHERE day=?", (day,)).fetchone()
            used = used[0] if used else 0
            if cap and used + n > cap:
                return False
            self.conn.execute("INSERT INTO quota(day, used) VALUES (?, ?) "
                              "ON CONFLICT(day) DO UPDATE SET used=?", (day, used + n, used + n))
            self.conn.execute("DELETE FROM quota WHERE day < ?",
                              ((utcnow() - dt.timedelta(days=30)).strftime("%Y-%m-%d"),))
            self.conn.commit()
            return True


# ----------------------------------------------------------------------------
# row shaping
# ----------------------------------------------------------------------------

def to_row(setlist, source):
    a = setlist.get("artist") or {}
    row = flatten(setlist, a.get("name", ""))
    row["artist_mbid"] = a.get("mbid", "")
    row["venue_id"] = (setlist.get("venue") or {}).get("id", "")
    row["last_updated"] = setlist.get("lastUpdated", "")
    row["source"] = source
    row["fetched_at"] = utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return row


def to_songs(setlist):
    out = []
    pos = 0
    sid = setlist.get("id", "")
    for set_no, st in enumerate((setlist.get("sets") or {}).get("set") or [], 1):
        encore = int(st.get("encore", 0) or 0)
        for song in st.get("song") or []:
            pos += 1
            out.append((sid, pos, set_no, encore, song.get("name", ""),
                        (song.get("cover") or {}).get("name", ""),
                        (song.get("with") or {}).get("name", ""),
                        1 if song.get("tape") else 0, song.get("info", "")))
    return out


def ingest(db, setlists, source, keep_songs):
    rows = [to_row(s, source) for s in setlists if s.get("id")]
    songs = [t for s in setlists for t in to_songs(s)] if keep_songs else None
    return db.upsert_setlists(rows, songs)


# ----------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------

def cmd_import(args, db, client):
    """Load existing CSVs so the DB knows what we already have."""
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    files = args.files or [p for p in LEGACY_CSVS if os.path.exists(p)]
    if not files:
        log("Nothing to import.")
        return
    now = utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for path in files:
        n = 0
        batch = []
        log(f"== importing {path} ==")
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                if not r.get("setlist_id"):
                    continue
                row = {c: r.get(c, "") for c in CSV_COLUMNS}
                try:
                    row["num_songs"] = int(row["num_songs"] or 0)
                except ValueError:
                    row["num_songs"] = 0
                row.update(artist_mbid="", venue_id="", last_updated=None,
                           source="csv", fetched_at=now)
                batch.append(row)
                if len(batch) >= 5000:
                    # never downgrade an API-sourced row to a CSV one
                    n += _upsert_csv_batch(db, batch)
                    batch = []
        n += _upsert_csv_batch(db, batch)
        log(f"   {n:,} rows processed")
    _print_status(db)


def _upsert_csv_batch(db, batch):
    if not batch:
        return 0
    with db.lock:
        # only overwrite rows that themselves came from a CSV
        db.conn.executemany(UPSERT + " WHERE setlists.source='csv'", batch)
        db.conn.commit()
    return len(batch)


def _date_range(args):
    """Newest -> oldest by default (recent shows are the valuable ones)."""
    end = dt.date.fromisoformat(args.to) if args.to else dt.date.today() + dt.timedelta(days=FUTURE_DAYS)
    start = dt.date.fromisoformat(getattr(args, "from_")) if getattr(args, "from_") else dt.date.fromisoformat(EARLIEST_DATE)
    d = end
    while d >= start:
        yield d
        d -= dt.timedelta(days=1)


def _backfill_one_date(db, client, day, keep_songs, stop=None):
    """Fetch every page for one calendar date. Resumes at pages_done."""
    iso = day.isoformat()
    api_date = day.strftime("%d-%m-%Y")
    rec = db.query("SELECT pages_done, done FROM backfill_dates WHERE date_iso=?", (iso,))
    pages_done = rec[0][0] if rec else 0
    if rec and rec[0][1]:
        return iso, 0, 0, True
    page = pages_done + 1
    total = None
    upserted = 0
    while True:
        if stop is not None and stop.is_set():
            return iso, total or 0, upserted, True   # partial; resumes at pages_done
        data = client._get("/search/setlists", params={"date": api_date, "p": page})
        if not data or not data.get("setlist"):
            break
        total = data.get("total", 0)
        per = data.get("itemsPerPage", 20) or 20
        last_page = max(1, (min(total, SEARCH_CAP) + per - 1) // per)
        upserted += ingest(db, data["setlist"], "backfill", keep_songs)
        db.execute("INSERT INTO backfill_dates(date_iso,total,pages_done,done,capped,updated_at) "
                   "VALUES (?,?,?,0,?,?) ON CONFLICT(date_iso) DO UPDATE SET "
                   "total=excluded.total, pages_done=excluded.pages_done, "
                   "capped=excluded.capped, updated_at=excluded.updated_at",
                   (iso, total, page, 1 if total > SEARCH_CAP else 0,
                    utcnow().isoformat(timespec="seconds")))
        if page >= last_page:
            break
        page += 1
    db.execute("INSERT INTO backfill_dates(date_iso,total,pages_done,done,capped,updated_at) "
               "VALUES (?,?,?,1,?,?) ON CONFLICT(date_iso) DO UPDATE SET "
               "total=excluded.total, done=1, updated_at=excluded.updated_at",
               (iso, total or 0, page, 1 if (total or 0) > SEARCH_CAP else 0,
                utcnow().isoformat(timespec="seconds")))
    return iso, total or 0, upserted, False


def cmd_backfill(args, db, client):
    """Walk calendar dates and pull every setlist on each one."""
    done = {r[0] for r in db.query("SELECT date_iso FROM backfill_dates WHERE done=1")}
    todo = [d for d in _date_range(args) if d.isoformat() not in done]
    if args.limit:
        todo = todo[:args.limit]
    log(f"== backfill: {len(todo):,} dates to fetch ({len(done):,} already done), "
        f"{args.workers} workers ==")
    if not todo:
        return
    total_up = 0
    dates_done = 0
    t0 = time.time()
    stop = threading.Event()

    def work(day):
        if stop.is_set():
            return None
        return _backfill_one_date(db, client, day, args.songs, stop)

    # explicit pool (not `with`) so Ctrl-C / quota-exhausted can cancel the
    # thousands of queued dates instead of waiting for them all to run
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = [pool.submit(work, d) for d in todo]
    try:
        for fut in as_completed(futures):
            res = fut.result()          # BudgetExhausted propagates from here
            if res is None:
                continue
            iso, total, up, skipped = res
            if skipped:
                continue
            dates_done += 1
            total_up += up
            if dates_done % 10 == 0 or total >= 500:
                rate = client.requests_used / max(1, time.time() - t0)
                log(f"   {iso}: {total:>5,} setlists  | {dates_done:,}/{len(todo):,} dates, "
                    f"{total_up:,} rows, {client.requests_used:,} req ({rate:.1f}/s)")
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        log(f"\n   backfill: {dates_done:,} dates, {total_up:,} rows upserted, "
            f"{client.requests_used:,} requests this run")
        capped = db.scalar("SELECT COUNT(*) FROM backfill_dates WHERE capped=1")
        if capped:
            log(f"   !! {capped} date(s) had >10,000 setlists (API cap); "
                f"see: SELECT * FROM backfill_dates WHERE capped=1")


def cmd_sync(args, db, client):
    """Pull everything created or edited since the last completed sync."""
    if args.since:
        since_dt = dt.datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    else:
        last_start = db.scalar("SELECT MAX(started_at) FROM sync_log WHERE completed=1")
        if last_start:
            # overlap the previous run by an hour so nothing slips between them
            since_dt = dt.datetime.fromisoformat(last_start) - dt.timedelta(hours=1)
        else:
            newest = db.scalar("SELECT MAX(last_updated) FROM setlists")
            since_dt = (dt.datetime.fromisoformat(newest.replace("+0000", "+00:00"))
                        if newest else utcnow() - dt.timedelta(days=args.first_days))
    since = since_dt.astimezone(dt.timezone.utc).strftime("%Y%m%d%H%M%S")
    started = utcnow().isoformat(timespec="seconds")
    log(f"== sync: setlists updated since {since_dt:%Y-%m-%d %H:%M} UTC ==")
    db.execute("INSERT INTO sync_log VALUES (?,?,0,0,0,0)", (started, since))

    page, total, up = 1, 0, 0
    while True:
        data = client._get("/search/setlists", params={"lastUpdated": since, "p": page})
        if not data or not data.get("setlist"):
            break
        total = data.get("total", 0)
        per = data.get("itemsPerPage", 20) or 20
        last_page = max(1, (min(total, SEARCH_CAP) + per - 1) // per)
        up += ingest(db, data["setlist"], "sync", args.songs)
        if page == 1:
            log(f"   {total:,} setlists changed -> {last_page} pages")
        if page % 25 == 0:
            log(f"   page {page}/{last_page}, {up:,} upserted")
        if page >= last_page:
            break
        page += 1
    db.execute("UPDATE sync_log SET total=?, upserted=?, requests=?, completed=1 "
               "WHERE started_at=?", (total, up, client.requests_used, started))
    log(f"   done: {up:,} setlists upserted, {client.requests_used} requests")
    if total > SEARCH_CAP:
        log(f"   !! {total:,} changes exceeded the {SEARCH_CAP:,} search cap: run sync more "
            f"often (it's cheap), or backfill the gap with --from/--to.")


def cmd_export(args, db, client):
    """Write the with_cities CSV (same previous/next rules as the harvester's enrich)."""
    out = args.to
    log(f"== exporting -> {out} ==")
    cur = db.conn.execute(
        "SELECT " + ",".join(CSV_COLUMNS) + " FROM setlists "
        "ORDER BY artist, tour, date_iso, setlist_id")
    written = 0
    with open(out, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(CSV_COLUMNS + ["previous_city", "next_city"])
        group_key = None
        group = []

        def flush():
            nonlocal written
            for i, r in enumerate(group):
                if (r[3] or "").strip():
                    prev_c = group[i - 1][5] if i > 0 else "First Show"
                    next_c = group[i + 1][5] if i < len(group) - 1 else "Final Show"
                else:
                    prev_c = next_c = ""
                w.writerow(list(r) + [prev_c, next_c])
                written += 1

        for r in cur:
            key = (r[0], (r[3] or "").strip())
            if key != group_key:
                flush()
                group, group_key = [], key
            group.append(r)
        flush()
    log(f"   {written:,} rows written")


def _print_status(db):
    n = db.scalar("SELECT COUNT(*) FROM setlists") or 0
    by_src = dict(db.query("SELECT source, COUNT(*) FROM setlists GROUP BY source"))
    artists = db.scalar("SELECT COUNT(DISTINCT artist) FROM setlists") or 0
    venues = db.scalar("SELECT COUNT(DISTINCT venue_id) FROM setlists WHERE venue_id!=''") or 0
    dates_done = db.scalar("SELECT COUNT(*) FROM backfill_dates WHERE done=1") or 0
    dates_rows = db.scalar("SELECT COALESCE(SUM(total),0) FROM backfill_dates WHERE done=1") or 0
    oldest = db.scalar("SELECT MIN(date_iso) FROM backfill_dates WHERE done=1")
    last_sync = db.query("SELECT started_at, total, upserted FROM sync_log "
                         "WHERE completed=1 ORDER BY started_at DESC LIMIT 1")
    songs = db.scalar("SELECT COUNT(*) FROM songs") or 0
    used = db.used_today()
    hv = db.harvester_used_today()
    size_mb = os.path.getsize(db.path) / 1e6 if os.path.exists(db.path) else 0
    log(f"== {db.path} ({size_mb:,.0f} MB) ==")
    log(f"  setlists          : {n:,}  ({', '.join(f'{k}: {v:,}' for k, v in sorted(by_src.items()))})")
    log(f"  distinct artists  : {artists:,}")
    log(f"  distinct venues   : {venues:,} (with a venue id)")
    log(f"  songs stored      : {songs:,}")
    log(f"  backfill          : {dates_done:,} dates complete ({dates_rows:,} setlists), "
        f"oldest done: {oldest or '-'}")
    if last_sync:
        log(f"  last sync         : {last_sync[0][0]}  ({last_sync[0][1]:,} changed, "
            f"{last_sync[0][2]:,} upserted)")
    else:
        log(f"  last sync         : never")
    log(f"  API used today    : {used:,} here + {hv:,} by setlistfm_harvest.py "
        f"= {used + hv:,} / {DAILY_CAP:,}")


def cmd_status(args, db, client):
    _print_status(db)


# ----------------------------------------------------------------------------

def main():
    # Shared options use SUPPRESS so a value given before the subcommand isn't
    # clobbered by the subparser's default; real defaults live in set_defaults.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=argparse.SUPPRESS, help=f"SQLite file (default {DB_PATH}).")
    common.add_argument("--api-key", default=argparse.SUPPRESS)
    common.add_argument("--rate", type=float, default=argparse.SUPPRESS,
                        help=f"Requests/second across all workers (default {DEFAULT_RATE}, cap {MAX_RATE}).")
    common.add_argument("--max-requests", type=int, default=argparse.SUPPRESS,
                        help="Stop cleanly after this many API requests this run.")
    common.add_argument("--daily-cap", type=int, default=argparse.SUPPRESS,
                        help=f"Requests per UTC day incl. the harvester's usage (default {DAILY_CAP}).")
    common.add_argument("--songs", action="store_true", default=argparse.SUPPRESS,
                        help="Also store every song into the songs table (bigger DB).")

    p = argparse.ArgumentParser(parents=[common],
                                description="setlist.fm -> SQLite: full backfill + daily sync.")
    key, key_source = resolve_api_key()
    # NOT p.set_defaults(): that rewrites the shared actions' defaults too, and
    # the subparser would then re-apply them over a value given before the
    # subcommand. Fill in whatever wasn't given, after parsing.
    shared_defaults = dict(db=DB_PATH, api_key=key, rate=DEFAULT_RATE, max_requests=None,
                           daily_cap=DAILY_CAP, songs=False)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("import", parents=[common], help="Load existing CSVs into the DB.")
    s.add_argument("files", nargs="*", help=f"CSVs (default: {' + '.join(LEGACY_CSVS)})")
    s.set_defaults(func=cmd_import, needs_client=False)

    b = sub.add_parser("backfill", parents=[common],
                       help="Fetch every setlist for every calendar date not yet done.")
    b.add_argument("--from", dest="from_", metavar="YYYY-MM-DD", help=f"Oldest date (default {EARLIEST_DATE}).")
    b.add_argument("--to", metavar="YYYY-MM-DD", help=f"Newest date (default today+{FUTURE_DAYS}d).")
    b.add_argument("--workers", type=int, default=8, help="Parallel dates (default 8).")
    b.add_argument("--limit", type=int, help="Only do the next N dates.")
    b.set_defaults(func=cmd_backfill, needs_client=True)

    y = sub.add_parser("sync", parents=[common], help="Pull everything created/edited since last sync.")
    y.add_argument("--since", metavar="YYYY-MM-DD", help="Override the cutoff.")
    y.add_argument("--first-days", type=int, default=7,
                   help="On the very first sync with an empty DB, look back this many days.")
    y.set_defaults(func=cmd_sync, needs_client=True)

    e = sub.add_parser("export", parents=[common], help="Write the with_cities-format CSV from the DB.")
    e.add_argument("--to", default=DEFAULT_EXPORT)
    e.set_defaults(func=cmd_export, needs_client=False)

    st = sub.add_parser("status", parents=[common])
    st.set_defaults(func=cmd_status, needs_client=False)

    args = p.parse_args()
    for k, v in shared_defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    if args.rate > MAX_RATE:
        p.error(f"--rate must be <= {MAX_RATE}")

    db = DB(args.db)
    client = None
    if args.needs_client:
        if not args.api_key:
            p.error("No API key (put it in setlistfm_key.txt).")
        # leave room for whatever the harvester has already spent today
        cap = max(0, args.daily_cap - db.harvester_used_today()) if args.daily_cap else 0
        log(f"   api key {mask(args.api_key)} (from {key_source})")
        log(f"   daily quota: {db.used_today():,} used here, "
            f"{db.harvester_used_today():,} by harvester -> {max(0, cap - db.used_today()):,} left")
        client = SetlistFM(args.api_key, rate_per_sec=args.rate, budget=args.max_requests,
                           quota=db, daily_cap=cap)
    try:
        args.func(args, db, client)
    except BudgetExhausted as e:
        log(f"\n!! {e} -- progress is saved, re-run to continue.")
    except KeyboardInterrupt:
        log("\n!! interrupted -- progress is saved.")


if __name__ == "__main__":
    main()
