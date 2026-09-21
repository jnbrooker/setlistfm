#!/usr/bin/env python3
"""
setlist.fm incremental harvester -> CSV (successor to setlistfm_tours.py)

What's different from setlistfm_tours.py
----------------------------------------
* Persistent state (setlistfm_state/): every artist we've ever seen, scraped or
  queued, is remembered by MusicBrainz id. Re-running never re-scrapes an artist
  that's already done, and an artist that was cut off mid-way resumes at the page
  it stopped on instead of starting from page 1.
* The existing setlistfm_tours.csv is never written to. It's read once (`seed`)
  to mark those artists as already done, and the result is cached in the state
  so the 140MB file is only parsed a single time.
* Self-expanding discovery: every venue we scrape gets recorded, and `discover
  --snowball` goes back to the venues we know about to find artists who played
  there. That's the venue -> artists -> their whole tour history loop you were
  doing by hand, automated and de-duplicated.
* Parallel workers sharing one global rate limiter, so throughput actually sits
  at the API cap instead of well under it. A 429 pauses every worker at once.
* A per-run request budget, so you can burn exactly as much of the daily API
  quota as you want and stop cleanly with the state intact.

Output columns are byte-identical to setlistfm_tours.csv, so the new file can be
concatenated onto the old one (pd.concat / COPY) with no reshaping.

QUICK START
-----------
    pip install requests

    # 1. one-off: tell the state what's already in your big CSV (~2-3 min)
    python setlistfm_harvest.py seed

    # 2. find artists to scrape (any combination, run as often as you like)
    python setlistfm_harvest.py discover --city Melbourne --pages 30
    python setlistfm_harvest.py discover --country GB --year 2024 --pages 50
    python setlistfm_harvest.py discover --venue "Melbourne Cricket Ground"

    # 3. scrape the queue (Ctrl-C any time; state is saved continuously)
    python setlistfm_harvest.py scrape --workers 3 --max-requests 4000

    # 4. keep growing: harvest artists from venues we discovered while scraping
    python setlistfm_harvest.py discover --snowball 25
    python setlistfm_harvest.py scrape

    python setlistfm_harvest.py status          # where am I up to

    # 5. when you want one file for analysis (originals left untouched)
    python setlistfm_harvest.py export          # -> setlistfm_tours_combined.csv

Note: setlist.fm still answers the odd request with a 429 even under 2 req/s.
That's handled (every worker pauses together and retries), but if you see a lot
of them, drop to --rate 1.2.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import paths

# ============================================================
#  API KEY -- never hardcode one here; this file is in version control.
#  Resolved in this order, first one wins:
#     1. --api-key "..."     (command line)
#     2. $SETLISTFM_API_KEY  (environment variable)
#     3. setlistfm_key.txt   (git-ignored; copy setlistfm_key.txt.example)
#  Get a key at https://api.setlist.fm/docs/1.0/index.html
# ============================================================
API_KEY = ""
KEY_FILE = paths.KEY_FILE
# ============================================================

API_ROOT = "https://api.setlist.fm/rest/1.0"

# Your setlist.fm tier's limits. Standard tier is 2.0/sec and ~1440/day;
# the upgraded tier is 16.0/sec and 50000/day. DEFAULT_RATE stays a little under
# the ceiling because the API still 429s on bursts right at the line.
MAX_RATE = 16.0
DEFAULT_RATE = 12.0
DAILY_CAP = 50000

STATE_DIR = paths.STATE_DIR

DEFAULT_OUTPUT = paths.TOURS_CSV
LEGACY_CSV = paths.LEGACY_TOURS_CSV

# identical to setlistfm_tours.py so the two CSVs concatenate cleanly
CSV_COLUMNS = [
    "artist",
    "eventDate",
    "date_iso",
    "tour",
    "venue",
    "city",
    "state",
    "stateCode",
    "country",
    "countryCode",
    "latitude",
    "longitude",
    "num_songs",
    "setlist_url",
    "setlist_id",
]


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def norm_name(name):
    """Loose artist-name key: lowercase, collapse punctuation/whitespace."""
    s = (name or "").strip().lower()
    s = re.sub(r"[‘’“”]", "'", s)
    s = re.sub(r"\s+", " ", s)
    return s


def iso_date(event_date):
    """API returns dd-mm-yyyy; convert to yyyy-mm-dd. Return '' if malformed."""
    if not event_date:
        return ""
    parts = event_date.split("-")
    if len(parts) == 3:
        d, m, y = parts
        return f"{y}-{m}-{d}"
    return ""


def flatten(setlist, artist_label):
    venue = setlist.get("venue", {}) or {}
    city = venue.get("city", {}) or {}
    country = city.get("country", {}) or {}
    coords = city.get("coords", {}) or {}
    tour = setlist.get("tour", {}) or {}

    num_songs = 0
    for st in (setlist.get("sets", {}) or {}).get("set", []) or []:
        num_songs += len(st.get("song", []) or [])

    event_date = setlist.get("eventDate", "")
    return {
        "artist": artist_label,
        "eventDate": event_date,
        "date_iso": iso_date(event_date),
        "tour": tour.get("name", ""),
        "venue": venue.get("name", ""),
        "city": city.get("name", ""),
        "state": city.get("state", ""),
        "stateCode": city.get("stateCode", ""),
        "country": country.get("name", ""),
        "countryCode": country.get("code", ""),
        "latitude": coords.get("lat", ""),
        "longitude": coords.get("long", ""),
        "num_songs": num_songs,
        "setlist_url": setlist.get("url", ""),
        "setlist_id": setlist.get("id", ""),
    }


def resolve_api_key():
    """Env var, then key file, then the constant. Never printed in full."""
    env = os.environ.get("SETLISTFM_API_KEY")
    if env and env.strip():
        return env.strip(), "SETLISTFM_API_KEY env var"
    # KEY_FILE is already absolute (paths.py pins it to the project folder), so
    # the old "try cwd, then the script directory" dance is no longer needed
    for path in (KEY_FILE,):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        return line, path
    return API_KEY, "API_KEY constant in this file"  # empty unless someone sets it


def mask(key):
    return f"{key[:4]}...{key[-4:]}" if key and len(key) > 8 else "????"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------
# state
# ----------------------------------------------------------------------------

class State:
    """
    Everything we know, on disk, so no work is ever repeated.

    artists.json : {mbid: {name, status, shows, next_page, seen}}
                   status = pending | done | partial | missing
                   next_page = page to resume from for a partial artist
      plus a "_names" map {normalised name: mbid|""} so artists known only by
      name (from the legacy CSV) are still skipped.
    venues.json  : {venue_id: {name, city, country, explored, artists_found}}
    """

    def __init__(self, directory=STATE_DIR):
        self.dir = directory
        os.makedirs(self.dir, exist_ok=True)
        self.artists_path = os.path.join(directory, "artists.json")
        self.venues_path = os.path.join(directory, "venues.json")
        self.seeds_path = os.path.join(directory, "seeded_files.json")
        self.usage_path = os.path.join(directory, "daily_usage.json")
        self.lock = threading.Lock()
        self.artists = self._load(self.artists_path, {"_names": {}})
        self.artists.setdefault("_names", {})
        self.venues = self._load(self.venues_path, {})
        self.seeded = self._load(self.seeds_path, {})
        self.usage = self._load(self.usage_path, {})
        self._dirty = False

    @staticmethod
    def _load(path, default):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return json.loads(json.dumps(default))

    @staticmethod
    def _atomic_write(path, obj):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        # On Windows, os.replace fails with PermissionError if the target is
        # momentarily held open (antivirus scan, editor, indexer). Retry briefly.
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.25 * (attempt + 1))

    def save(self):
        with self.lock:
            self._atomic_write(self.artists_path, self.artists)
            self._atomic_write(self.venues_path, self.venues)
            self._atomic_write(self.seeds_path, self.seeded)
            self._atomic_write(self.usage_path, self.usage)
            self._dirty = False

    # -- daily quota ---------------------------------------------------------

    @staticmethod
    def today():
        return time.strftime("%Y-%m-%d", time.gmtime())  # UTC, like the API's day

    def used_today(self):
        return self.usage.get(self.today(), 0)

    def spend_daily(self, n=1, cap=DAILY_CAP):
        """Count one API request against today's quota. False = cap reached."""
        day = self.today()
        used = self.usage.get(day, 0)
        if cap and used + n > cap:
            return False
        self.usage[day] = used + n
        # keep the file small
        if len(self.usage) > 30:
            for old in sorted(self.usage)[:-30]:
                self.usage.pop(old, None)
        return True

    # -- artists ------------------------------------------------------------

    def known_name(self, name):
        return norm_name(name) in self.artists["_names"]

    def add_artist(self, mbid, name):
        """Queue an artist if we've never seen them (by mbid *or* by name)."""
        with self.lock:
            key = norm_name(name)
            if mbid and mbid in self.artists:
                self.artists[mbid]["seen"] = self.artists[mbid].get("seen", 0) + 1
                self.artists["_names"].setdefault(key, mbid)
                return False
            if key in self.artists["_names"]:
                # known by name only (legacy CSV) -> attach the mbid, stay done
                existing = self.artists["_names"][key]
                if not existing and mbid:
                    self.artists["_names"][key] = mbid
                    self.artists[mbid] = {"name": name, "status": "done",
                                          "shows": 0, "next_page": 1, "seen": 1,
                                          "legacy": True}
                return False
            if not mbid:
                return False
            self.artists[mbid] = {"name": name, "status": "pending",
                                  "shows": 0, "next_page": 1, "seen": 1}
            self.artists["_names"][key] = mbid
            self._dirty = True
            return True

    def mark_name_done(self, name):
        """Record a legacy artist known only by name."""
        key = norm_name(name)
        if key and key not in self.artists["_names"]:
            self.artists["_names"][key] = ""
            return True
        return False

    def pending(self):
        out = [(mbid, rec) for mbid, rec in self.artists.items()
               if mbid != "_names" and rec.get("status") in ("pending", "partial")]
        # most-frequently-seen artists first: they're the ones that connect the graph
        out.sort(key=lambda kv: (-kv[1].get("seen", 0), kv[1].get("name", "").lower()))
        return out

    def update_artist(self, mbid, **fields):
        with self.lock:
            self.artists.setdefault(mbid, {}).update(fields)
            self._dirty = True

    # -- venues -------------------------------------------------------------

    def add_venue(self, venue):
        vid = (venue or {}).get("id")
        if not vid:
            return
        with self.lock:
            if vid in self.venues:
                return
            city = (venue.get("city") or {})
            self.venues[vid] = {
                "name": venue.get("name", ""),
                "city": city.get("name", ""),
                "country": (city.get("country") or {}).get("code", ""),
                "explored": False,
                "artists_found": 0,
            }
            self._dirty = True

    def unexplored_venues(self):
        return [(vid, rec) for vid, rec in self.venues.items() if not rec.get("explored")]

    # -- stats --------------------------------------------------------------

    def counts(self):
        c = {"pending": 0, "partial": 0, "done": 0, "missing": 0}
        for mbid, rec in self.artists.items():
            if mbid == "_names":
                continue
            c[rec.get("status", "pending")] = c.get(rec.get("status", "pending"), 0) + 1
        return c


# ----------------------------------------------------------------------------
# API client
# ----------------------------------------------------------------------------

class BudgetExhausted(Exception):
    pass


class RateLimiter:
    """
    Thread-safe request spacing with a global pause every worker respects,
    plus AIMD self-tuning: each 429 halves the effective rate, and a run of
    clean responses walks it back up towards the configured ceiling.

    That means you can just ask for --rate 16; if the key turns out to be on the
    standard 2/sec tier, the scraper finds its own ceiling in a few seconds
    instead of spending the whole run bouncing off 429s.
    """

    FLOOR_RATE = 0.8          # never crawl slower than this
    RECOVER_AFTER = 40        # clean responses before nudging the rate back up

    def __init__(self, rate_per_sec, adaptive=True):
        self.max_rate = rate_per_sec
        self.rate = rate_per_sec
        self.adaptive = adaptive
        self._next_slot = 0.0
        self._pause_until = 0.0
        self._clean = 0
        self._lock = threading.Lock()

    @property
    def min_interval(self):
        return 1.0 / self.rate

    def wait(self):
        while True:
            with self._lock:
                now = time.monotonic()
                if now < self._pause_until:
                    sleep_for = self._pause_until - now
                else:
                    slot = max(now, self._next_slot)
                    self._next_slot = slot + self.min_interval
                    sleep_for = slot - now
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    return
            time.sleep(min(sleep_for, 5.0))

    def pause(self, seconds):
        with self._lock:
            self._pause_until = max(self._pause_until, time.monotonic() + seconds)

    def throttle(self):
        """Called on a 429: multiplicative decrease."""
        if not self.adaptive:
            return None
        with self._lock:
            self._clean = 0
            new_rate = max(self.FLOOR_RATE, self.rate / 2.0)
            if abs(new_rate - self.rate) < 1e-9:
                return None
            self.rate = new_rate
            return new_rate

    def reward(self):
        """Called on a 200: additive increase after a clean streak."""
        if not self.adaptive:
            return None
        with self._lock:
            self._clean += 1
            if self._clean < self.RECOVER_AFTER or self.rate >= self.max_rate:
                return None
            self._clean = 0
            self.rate = min(self.max_rate, self.rate + max(0.25, self.max_rate * 0.1))
            return self.rate


class SetlistFM:
    def __init__(self, api_key, rate_per_sec=DEFAULT_RATE, max_retries=6, budget=None,
                 quota=None, daily_cap=DAILY_CAP, adaptive=True):
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": api_key,
            "Accept": "application/json",
            "User-Agent": "tour-route-scraper/2.0",
        })
        self.limiter = RateLimiter(rate_per_sec, adaptive=adaptive)
        self.max_retries = max_retries
        self._budget = budget
        self._used = 0
        self._quota = quota          # State, for the cross-run daily counter
        self._daily_cap = daily_cap
        self._lock = threading.Lock()

    @property
    def requests_used(self):
        return self._used

    def _spend(self):
        with self._lock:
            if self._budget is not None and self._used >= self._budget:
                raise BudgetExhausted(f"request budget of {self._budget} used up")
            if self._quota is not None and not self._quota.spend_daily(1, self._daily_cap):
                raise BudgetExhausted(
                    f"daily API quota of {self._daily_cap} requests is used up "
                    f"(resets at UTC midnight)")
            self._used += 1

    def _get(self, path, params=None):
        url = f"{API_ROOT}{path}"
        backoff = 2.0
        for attempt in range(1, self.max_retries + 1):
            self._spend()
            self.limiter.wait()
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise
                log(f"    ! network error ({e}); retry in {backoff:.0f}s")
                time.sleep(backoff)
                backoff *= 2
                continue

            if resp.status_code == 200:
                recovered = self.limiter.reward()
                if recovered:
                    log(f"    ~ rate recovering to {recovered:.1f} req/s")
                try:
                    return resp.json()
                except ValueError:
                    return None
            if resp.status_code == 404:
                return None
            if resp.status_code in (429, 403) or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else backoff
                wait += random.uniform(0, 1.0)  # de-sync the workers
                slowed = self.limiter.throttle()
                note = f", rate -> {slowed:.1f} req/s" if slowed else ""
                log(f"    ! HTTP {resp.status_code}; all workers pausing {wait:.0f}s{note} "
                    f"(attempt {attempt}/{self.max_retries})")
                self.limiter.pause(wait)   # stop every thread, not just this one
                time.sleep(wait)
                backoff = min(backoff * 2, 300)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"Gave up on {url} after {self.max_retries} attempts")

    # -- endpoints ----------------------------------------------------------

    def find_artist_mbid(self, name):
        data = self._get("/search/artists", params={"artistName": name,
                                                    "sort": "relevance", "p": 1})
        if not data or not data.get("artist"):
            return None, None
        top = data["artist"][0]
        return top.get("mbid"), top.get("name")

    def search_setlists(self, params, max_pages=50, start_page=1):
        """Yield (setlist, page, last_page) across /search/setlists pages."""
        page = start_page
        while page < start_page + max_pages:
            data = self._get("/search/setlists", params=dict(params, p=page))
            if not data:
                return
            setlists = data.get("setlist", []) or []
            if not setlists:
                return
            total = data.get("total", 0)
            per_page = data.get("itemsPerPage", 20) or 20
            last_page = max(1, (total + per_page - 1) // per_page)
            for s in setlists:
                yield s, page, last_page
            if page >= last_page:
                return
            page += 1

    def artist_setlists(self, mbid, start_page=1):
        """Yield (setlist, page, last_page) for one artist, resumable."""
        page = start_page
        while True:
            data = self._get(f"/artist/{mbid}/setlists", params={"p": page})
            if not data:
                return
            setlists = data.get("setlist", []) or []
            if not setlists:
                return
            total = data.get("total", 0)
            per_page = data.get("itemsPerPage", 20) or 20
            last_page = max(1, (total + per_page - 1) // per_page)
            for s in setlists:
                yield s, page, last_page
            if page >= last_page:
                return
            page += 1

    def venue_setlists(self, venue_id, max_pages=10):
        page = 1
        while page <= max_pages:
            data = self._get(f"/venue/{venue_id}/setlists", params={"p": page})
            if not data:
                return
            setlists = data.get("setlist", []) or []
            if not setlists:
                return
            total = data.get("total", 0)
            per_page = data.get("itemsPerPage", 20) or 20
            last_page = max(1, (total + per_page - 1) // per_page)
            for s in setlists:
                yield s
            if page >= last_page:
                return
            page += 1

    def search_venues(self, city=None, name=None, country=None, max_pages=5):
        page = 1
        while page <= max_pages:
            params = {"p": page}
            if city:
                params["cityName"] = city
            if name:
                params["name"] = name
            if country:
                params["country"] = country
            data = self._get("/search/venues", params=params)
            if not data:
                return
            venues = data.get("venue", []) or []
            if not venues:
                return
            total = data.get("total", 0)
            per_page = data.get("itemsPerPage", 20) or 20
            last_page = max(1, (total + per_page - 1) // per_page)
            for v in venues:
                yield v
            if page >= last_page:
                return
            page += 1


# ----------------------------------------------------------------------------
# output writer
# ----------------------------------------------------------------------------

class CsvSink:
    """Append-only writer, de-duplicating on setlist_id within the new file."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.seen_ids = set()
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        if exists:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    sid = row.get("setlist_id")
                    if sid:
                        self.seen_ids.add(sid)
            log(f"   output already holds {len(self.seen_ids)} shows; appending")
        self._f = open(path, "a" if exists else "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=CSV_COLUMNS)
        if not exists:
            self._w.writeheader()
            self._f.flush()

    def write(self, rows):
        """rows: list of dicts. Returns how many were actually new."""
        written = 0
        with self.lock:
            for row in rows:
                sid = row.get("setlist_id")
                if sid and sid in self.seen_ids:
                    continue
                if sid:
                    self.seen_ids.add(sid)
                self._w.writerow(row)
                written += 1
            self._f.flush()
        return written

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------

def cmd_seed(args, state, client=None):
    """Read existing CSVs and mark every artist in them as already done."""
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    paths = args.files or [LEGACY_CSV, args.output]
    for path in paths:
        if not os.path.exists(path):
            log(f"   skip {path} (not found)")
            continue
        sig = f"{os.path.getsize(path)}:{int(os.path.getmtime(path))}"
        if state.seeded.get(path) == sig and not args.force:
            log(f"   skip {path} (already seeded, unchanged)")
            continue
        log(f"== Seeding from {path} ==")
        added = rows = 0
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if "artist" not in (reader.fieldnames or []):
                log(f"   !! no 'artist' column in {path}, skipping")
                continue
            for row in reader:
                rows += 1
                if state.mark_name_done(row.get("artist", "")):
                    added += 1
                if rows % 500000 == 0:
                    log(f"   ...{rows:,} rows, {added:,} artists")
        state.seeded[path] = sig
        state.save()
        log(f"   {rows:,} rows -> {added:,} artists marked done")
    counts = state.counts()
    log(f"\nState now knows {len(state.artists['_names']):,} artist names "
        f"({counts['done']:,} done, {counts['pending']:,} pending).")


def _queue_from_setlists(state, setlists_iter, label, pages_note=True):
    """Consume an iterator of setlists, queueing new artists + recording venues."""
    new = seen = 0
    last_page = None
    for item in setlists_iter:
        s, page, last = item if isinstance(item, tuple) else (item, None, None)
        last_page = last
        seen += 1
        a = s.get("artist", {}) or {}
        if state.add_artist(a.get("mbid"), a.get("name", "")):
            new += 1
        state.add_venue(s.get("venue"))
        if pages_note and page and seen % 100 == 0:
            log(f"    {label}: page {page}/{last} -> {new} new artists")
    return new, seen


def cmd_discover(args, state, client):
    """Queue up artists from venue / city / country / year searches, or snowball."""
    total_new = 0

    if args.artist:
        for name in args.artist:
            if state.known_name(name) and not args.force:
                log(f"   '{name}' already known, skipping lookup")
                continue
            mbid, resolved = client.find_artist_mbid(name)
            if not mbid:
                log(f"   !! no match for '{name}'")
                continue
            if state.add_artist(mbid, resolved):
                total_new += 1
                log(f"   queued {resolved}")
            else:
                log(f"   {resolved} already known")

    search_params = {}
    if args.venue:
        search_params["venueName"] = args.venue
    if args.city:
        search_params["cityName"] = args.city
    if args.country:
        search_params["countryCode"] = args.country
    if args.year:
        search_params["year"] = args.year
    if args.tour:
        search_params["tourName"] = args.tour
    if search_params:
        label = " ".join(f"{k}={v}" for k, v in search_params.items())
        log(f"== Discovering artists: {label} ==")
        new, seen = _queue_from_setlists(
            state, client.search_setlists(search_params, max_pages=args.pages), label)
        log(f"   {seen} setlists scanned -> {new} new artists")
        total_new += new
        state.save()

    if args.venue_ids:
        for vid in args.venue_ids:
            log(f"== Venue {vid} ==")
            new, seen = _queue_from_setlists(
                state, client.venue_setlists(vid, max_pages=args.pages), vid, False)
            rec = state.venues.setdefault(vid, {"name": "", "city": "", "country": ""})
            rec["explored"] = True
            rec["artists_found"] = new
            log(f"   {seen} setlists -> {new} new artists")
            total_new += new
            state.save()

    if args.venues_in_city:
        log(f"== Venues in {args.venues_in_city} ==")
        found = 0
        for v in client.search_venues(city=args.venues_in_city, country=args.country,
                                      max_pages=args.pages):
            state.add_venue(v)
            found += 1
        log(f"   recorded {found} venues (now run --snowball to mine them)")
        state.save()

    if args.snowball:
        todo = state.unexplored_venues()
        # biggest/most central venues first is impossible to know up front, so
        # just take them in discovery order but let the user cap the count
        todo = todo[:args.snowball]
        log(f"== Snowball: mining {len(todo)} unexplored venues "
            f"({len(state.unexplored_venues())} known unexplored) ==")
        for i, (vid, rec) in enumerate(todo, 1):
            new, seen = _queue_from_setlists(
                state, client.venue_setlists(vid, max_pages=args.venue_pages), vid, False)
            rec["explored"] = True
            rec["artists_found"] = new
            total_new += new
            log(f"   [{i}/{len(todo)}] {rec.get('name','?')}, {rec.get('city','?')}: "
                f"{seen} setlists -> {new} new artists")
            state.save()

    state.save()
    counts = state.counts()
    log(f"\n{total_new} new artists queued. "
        f"Queue is now {counts['pending'] + counts['partial']:,} artists "
        f"({client.requests_used} API requests used).")


def cmd_scrape(args, state, client):
    """Work through the pending queue, writing rows to the new CSV."""
    todo = state.pending()
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        log("Nothing pending. Run `discover` to queue more artists.")
        return

    sink = CsvSink(args.output)
    log(f"== Scraping {len(todo):,} artists -> {args.output} "
        f"({args.workers} workers, {args.rate} req/s cap) ==")

    done_count = [0]
    rows_total = [0]
    stop = threading.Event()
    progress_lock = threading.Lock()

    def bump(n):
        with progress_lock:
            rows_total[0] += n

    def work(item):
        mbid, rec = item
        if stop.is_set():
            return
        name = rec.get("name", mbid)
        start_page = rec.get("next_page", 1) if rec.get("status") == "partial" else 1
        buffer = []
        page_done = start_page - 1
        count = 0
        try:
            for s, page, last in client.artist_setlists(mbid, start_page=start_page):
                if page != page_done and buffer:
                    # flush per page so an interrupt loses at most one page
                    bump(sink.write(buffer))
                    buffer = []
                    state.update_artist(mbid, status="partial", next_page=page,
                                        shows=count)
                page_done = page
                buffer.append(flatten(s, name))
                state.add_venue(s.get("venue"))
                count += 1
        except BudgetExhausted:
            if buffer:
                bump(sink.write(buffer))
            state.update_artist(mbid, status="partial", next_page=page_done + 1)
            stop.set()
            return
        except Exception as e:
            if buffer:
                bump(sink.write(buffer))
            log(f"   !! {name}: {e}")
            state.update_artist(mbid, status="partial", next_page=page_done + 1)
            return

        if buffer:
            bump(sink.write(buffer))
        state.update_artist(mbid, status="done" if count else "missing",
                            shows=count, next_page=1)
        with progress_lock:
            done_count[0] += 1
            n = done_count[0]
            log(f"   [{n}/{len(todo)}] {name}: {count} shows "
                f"({rows_total[0]:,} rows, {client.requests_used} reqs)")
            if n % 25 == 0:
                state.save()

    try:
        if args.workers <= 1:
            for item in todo:
                if stop.is_set():
                    break
                work(item)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(work, todo))
    except KeyboardInterrupt:
        log("\n!! interrupted -- saving state")
        stop.set()
    finally:
        state.save()
        sink.close()

    counts = state.counts()
    log(f"\nDone. {rows_total[0]:,} new rows -> {args.output}. "
        f"{client.requests_used} API requests at a settled "
        f"{client.limiter.rate:.1f} req/s. "
        f"{counts['done']:,} artists done, "
        f"{counts['pending'] + counts['partial']:,} still queued.")
    if stop.is_set():
        log("Stopped early (budget or interrupt). Re-run `scrape` to continue.")


def cmd_requeue(args, state, client):
    """
    Force artists back into the queue.

    Use this for an artist who IS in the legacy CSV but whose history you think
    is incomplete (e.g. a run that died half way), or to top an artist up with
    shows played since you last scraped them. Rows are de-duplicated on
    setlist_id, so a re-scrape only ever adds what you're missing.
    """
    n = 0
    for name in args.artist:
        mbid, resolved = client.find_artist_mbid(name)
        if not mbid:
            log(f"   !! no match for '{name}'")
            continue
        state.artists[mbid] = {"name": resolved, "status": "pending", "shows": 0,
                               "next_page": 1, "seen": 99}
        state.artists["_names"][norm_name(resolved)] = mbid
        n += 1
        log(f"   requeued {resolved}")
    if args.missing:
        for mbid, rec in state.artists.items():
            if mbid != "_names" and rec.get("status") == "missing":
                rec["status"] = "pending"
                rec["next_page"] = 1
                n += 1
        log(f"   requeued every 'missing' artist")
    state.save()
    log("")
    log(f"{n} artists queued. Run `scrape` next.")


def cmd_status(args, state, client=None):
    counts = state.counts()
    names = len(state.artists["_names"])
    venues = len(state.venues)
    unexplored = len(state.unexplored_venues())
    log("== setlist.fm harvester state ==")
    log(f"  artist names known : {names:,}")
    log(f"  done               : {counts['done']:,}")
    log(f"  pending            : {counts['pending']:,}")
    log(f"  partial (resumable): {counts['partial']:,}")
    log(f"  venues recorded    : {venues:,} ({unexplored:,} not yet mined)")
    log(f"  API used today     : {state.used_today():,} / {args.daily_cap:,} (UTC day)")
    for path in (LEGACY_CSV, args.output):
        if os.path.exists(path):
            mb = os.path.getsize(path) / 1e6
            tag = "seeded" if path in state.seeded else "NOT seeded"
            log(f"  {path}: {mb:,.1f} MB ({tag})")
    pending = state.pending()[:10]
    if pending:
        log("  next up: " + ", ".join(r.get("name", m) for m, r in pending))


def cmd_export(args, state, client=None):
    """Concatenate the legacy CSV and the new one into a single file."""
    out = args.combined
    sources = [p for p in (LEGACY_CSV, args.output) if os.path.exists(p)]
    if not sources:
        log("Nothing to combine.")
        return
    log(f"== Combining {', '.join(sources)} -> {out} ==")
    written = 0
    with open(out, "w", newline="", encoding="utf-8") as fo:
        writer = csv.DictWriter(fo, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        seen = set()
        csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
        for path in sources:
            with open(path, newline="", encoding="utf-8", errors="replace") as fi:
                for row in csv.DictReader(fi):
                    sid = row.get("setlist_id")
                    if sid:
                        if sid in seen:
                            continue
                        seen.add(sid)
                    writer.writerow({c: row.get(c, "") for c in CSV_COLUMNS})
                    written += 1
    log(f"   {written:,} unique shows -> {out}")


ROUTE_COLUMNS = CSV_COLUMNS + ["previous_city", "next_city"]


def cmd_enrich(args, state=None, client=None):
    """
    Add previous_city / next_city, matching with_cities_setlistfm.csv exactly.

    Semantics reverse-engineered from that file and verified against it:
      * rows are grouped by (artist, tour) and ordered by date within the group
      * previous_city / next_city are the neighbouring show's city in that tour
      * the first show of a tour gets "First Show", the last gets "Final Show"
      * a row with no tour name gets both columns blank

    Two passes over the input so a multi-GB file never has to fit in memory.
    """
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    sources = args.sources or [p for p in (LEGACY_CSV, args.output) if os.path.exists(p)]
    sources = [p for p in sources if os.path.exists(p)]
    if not sources:
        log("Nothing to enrich.")
        return
    out = args.to
    log(f"== Enriching {', '.join(sources)} -> {out} ==")

    # pass 1: collect (artist, tour) -> ordered list of (date, row_index, city)
    groups = {}
    index = 0
    seen_ids = set()
    keep = []            # row_index -> True if we write it (dedup on setlist_id)
    for path in sources:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                sid = row.get("setlist_id") or ""
                if sid and sid in seen_ids:
                    keep.append(False)
                    index += 1
                    continue
                if sid:
                    seen_ids.add(sid)
                keep.append(True)
                tour = (row.get("tour") or "").strip()
                if tour:
                    key = (row.get("artist", ""), tour)
                    groups.setdefault(key, []).append(
                        (row.get("date_iso", ""), index, row.get("city", "")))
                index += 1
    log(f"   {index:,} rows, {len(groups):,} artist/tour groups")

    # work out each row's neighbours
    route = {}
    for key, items in groups.items():
        items.sort(key=lambda t: (t[0], t[1]))
        for i, (_, idx, _city) in enumerate(items):
            prev_city = items[i - 1][2] if i > 0 else "First Show"
            next_city = items[i + 1][2] if i < len(items) - 1 else "Final Show"
            route[idx] = (prev_city, next_city)
    del groups

    # pass 2: stream the rows back out with the two extra columns
    index = 0
    written = 0
    with open(out, "w", newline="", encoding="utf-8") as fo:
        writer = csv.DictWriter(fo, fieldnames=ROUTE_COLUMNS)
        writer.writeheader()
        for path in sources:
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    if not keep[index]:
                        index += 1
                        continue
                    prev_city, next_city = route.get(index, ("", ""))
                    out_row = {c: row.get(c, "") for c in CSV_COLUMNS}
                    out_row["previous_city"] = prev_city
                    out_row["next_city"] = next_city
                    writer.writerow(out_row)
                    written += 1
                    index += 1
    log(f"   {written:,} unique shows -> {out}")


# ----------------------------------------------------------------------------

def main():
    # Global options live on a shared parent, so they work either before or
    # after the subcommand ("... --max-requests 5000 scrape" and
    # "... scrape --max-requests 5000" both do the same thing).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--api-key", default=argparse.SUPPRESS)
    common.add_argument("--rate", type=float, default=argparse.SUPPRESS,
                        help=f"Max requests/second across all workers (default {DEFAULT_RATE}, cap {MAX_RATE}).")
    common.add_argument("-o", "--output", default=argparse.SUPPRESS,
                        help=f"New-rows CSV (default {DEFAULT_OUTPUT}). "
                             f"{LEGACY_CSV} is never written to.")
    common.add_argument("--max-requests", type=int, default=argparse.SUPPRESS,
                        help="Stop cleanly after this many API requests (daily-quota guard).")
    common.add_argument("--state-dir", default=argparse.SUPPRESS)
    common.add_argument("--no-adaptive", action="store_true", default=argparse.SUPPRESS,
                        help="Don't auto-reduce the rate on 429s; hold --rate exactly.")
    common.add_argument("--daily-cap", type=int, default=argparse.SUPPRESS,
                        help=f"Requests allowed per UTC day, counted across runs "
                             f"(default {DAILY_CAP}; 0 = no cap).")

    p = argparse.ArgumentParser(
        parents=[common],
        description="Incremental setlist.fm harvester (never re-scrapes what you have).")
    resolved_key, key_source = resolve_api_key()
    p.set_defaults(api_key=resolved_key,
                   rate=DEFAULT_RATE, output=DEFAULT_OUTPUT, max_requests=None,
                   state_dir=STATE_DIR, daily_cap=DAILY_CAP, no_adaptive=False)

    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seed", parents=[common], help="Index existing CSVs so their artists are never re-scraped.")
    s.add_argument("files", nargs="*", help=f"CSVs to index (default: {LEGACY_CSV} + output).")
    s.add_argument("--force", action="store_true", help="Re-read even if unchanged since last seed.")
    s.set_defaults(func=cmd_seed, needs_client=False)

    d = sub.add_parser("discover", parents=[common], help="Queue new artists from venue/city/country searches.")
    d.add_argument("--artist", nargs="*", default=[], help="Explicit artist name(s).")
    d.add_argument("--venue", help="Venue name, e.g. 'Melbourne Cricket Ground'.")
    d.add_argument("--venue-ids", nargs="*", default=[], help="setlist.fm venue id(s).")
    d.add_argument("--venues-in-city", help="Record every venue in this city for later snowballing.")
    d.add_argument("--city", help="City name.")
    d.add_argument("--country", help="ISO country code, e.g. GB, AU.")
    d.add_argument("--year", type=int, help="Year filter.")
    d.add_argument("--tour", help="Tour name filter.")
    d.add_argument("--pages", type=int, default=20, help="Max search pages (20 shows/page).")
    d.add_argument("--snowball", type=int, metavar="N",
                   help="Mine N venues we recorded while scraping for artists we don't have.")
    d.add_argument("--venue-pages", type=int, default=5,
                   help="Pages per venue when snowballing (default 5 = 100 shows).")
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_discover, needs_client=True)

    c = sub.add_parser("scrape", parents=[common], help="Scrape every pending artist's full history.")
    c.add_argument("--workers", type=int, default=8, help="Parallel artists (default 8).")
    c.add_argument("--limit", type=int, help="Only do the first N queued artists.")
    c.set_defaults(func=cmd_scrape, needs_client=True)

    rq = sub.add_parser("requeue", parents=[common],
                        help="Force artists back into the queue (top-up / fix a half-done artist).")
    rq.add_argument("--artist", nargs="*", default=[], help="Artist name(s) to re-scrape.")
    rq.add_argument("--missing", action="store_true",
                    help="Also retry every artist that came back with zero shows.")
    rq.set_defaults(func=cmd_requeue, needs_client=True)

    st = sub.add_parser("status", parents=[common], help="Show what's done and what's queued.")
    st.set_defaults(func=cmd_status, needs_client=False)

    en = sub.add_parser("enrich", parents=[common],
                        help="Add previous_city/next_city -> with_cities_setlistfm format.")
    en.add_argument("sources", nargs="*",
                    help=f"Input CSV(s) (default: {LEGACY_CSV} + the new-rows CSV).")
    en.add_argument("--to", default="with_cities_setlistfm_new.csv", help="Output path.")
    en.set_defaults(func=cmd_enrich, needs_client=False)

    e = sub.add_parser("export", parents=[common], help="Merge legacy + new CSV into one de-duplicated file.")
    e.add_argument("--combined", default="setlistfm_tours_combined.csv")
    e.set_defaults(func=cmd_export, needs_client=False)

    args = p.parse_args()

    if args.rate > MAX_RATE:
        p.error(f"Rate must be <= {MAX_RATE} req/s (your setlist.fm tier's limit).")

    state = State(args.state_dir)
    client = None
    if getattr(args, "needs_client", False):
        if not args.api_key:
            p.error(f"No API key. Set SETLISTFM_API_KEY, pass --api-key, or put "
                    f"one line in {KEY_FILE} (see setlistfm_key.txt.example).")
        log(f"   api key {mask(args.api_key)} (from {key_source})")
        client = SetlistFM(args.api_key, rate_per_sec=args.rate, budget=args.max_requests,
                           quota=state, daily_cap=args.daily_cap,
                           adaptive=not args.no_adaptive)
        left = args.daily_cap - state.used_today() if args.daily_cap else None
        if left is not None:
            log(f"   daily quota: {state.used_today():,}/{args.daily_cap:,} used, "
                f"{left:,} left today")

    try:
        args.func(args, state, client)
    except BudgetExhausted as e:
        log(f"\n!! {e} -- state saved, re-run to continue.")
        state.save()
    except KeyboardInterrupt:
        log("\n!! interrupted -- state saved.")
        state.save()


if __name__ == "__main__":
    main()
