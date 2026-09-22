#!/usr/bin/env python3
"""
Venue enrichment from Wikidata -> setlistfm.db

Only 7.8% of venues have a capacity and 6.1% an indoor/outdoor label, because
both come from the curated arena sheet (676 rows) or from Pollstar reporting a
box office. This fills gaps for everything else by looking venues up on
Wikidata, and writes what it finds to `ref_venue_enrichment`.

WHY A TABLE OF ITS OWN

`venues` is dropped and rebuilt by every `build`, and `arenas` is dropped by
`load-arenas`, so anything written into either would not survive. This table is
raw, like `pollstar_events` and `fixtures`: only this script writes it, and
`build-venues` reads it. So the lookup runs when YOU ask, and every ordinary
pipeline run picks the results up without going near the network.

The curated sheet still wins on any venue it covers -- a lookup only ever fills
a gap, and every column records where it came from.

WHICH WIKIDATA INTERFACE

The MediaWiki entity API (www.wikidata.org/w/api.php), not the SPARQL endpoint:
one search plus one batched fetch per venue, and it stays up when the query
service is rate-limiting. Results are cached in the table, misses included, so
a second run does not ask about the same venue twice.

USAGE
    python enrich_venues.py targets --min-events 20     # what needs looking up
    python enrich_venues.py pull --limit 500            # look up the biggest gaps
    python enrich_venues.py pull --limit 5000 --min-events 5
    python enrich_venues.py status                      # coverage so far
    python enrich_venues.py load-manual                 # apply the override CSV

Then any `pipeline.py run` folds the results into `venues`.
"""

import argparse
import csv
import datetime as dt
import json
import math
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import paths
from artist_categories import norm_key

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

API = "https://www.wikidata.org/w/api.php"
USER_AGENT = ("setlistfm-venue-enrichment/1.0 "
              "(live-music venue research; contact: repository owner)")
MANUAL_CSV = paths.here("venue_enrichment_manual.csv")

# Be a good citizen: the API allows far more, but nothing here is urgent.
REQS_PER_SEC = 3.0
MAX_RETRIES = 5

# Wikidata properties we read
P_CAPACITY, P_COORD, P_ADMIN, P_COUNTRY, P_INSTANCE, P_INCEPTION = (
    "P1083", "P625", "P131", "P17", "P31", "P571")

# How far apart our coordinate and Wikidata's may be and still be the same place.
# Generous, because most of our coordinates are city centroids.
COORD_TOLERANCE_KM = 25

# Wikidata class label -> (our venue_type, indoor/outdoor). Matched as keywords
# against the class label, so "multi-purpose stadium" lands on stadium.
TYPE_KEYWORDS = [
    ("stadium", "Stadium", "Outside"),
    ("racetrack", "Race Track", "Outside"),
    ("racecourse", "Race Track", "Outside"),
    ("speedway", "Race Track", "Outside"),
    ("amphitheat", "Amphitheatre", "Outside"),
    ("fairground", "Fairground", "Outside"),
    ("festival", "Festival Site", "Outside"),
    ("park", "Outdoor Venues", "Outside"),
    ("garden", "Outdoor Venues", "Outside"),
    ("arena", "Arena", "Inside"),
    ("concert hall", "Auditorium / Theatre", "Inside"),
    ("opera house", "Auditorium / Theatre", "Inside"),
    ("theatre", "Auditorium / Theatre", "Inside"),
    ("theater", "Auditorium / Theatre", "Inside"),
    ("auditorium", "Auditorium / Theatre", "Inside"),
    ("music venue", "Club", "Inside"),
    ("nightclub", "Club", "Inside"),
    ("indoor", "Arena", "Inside"),
    ("convention", "Convention Center", "Inside"),
    ("casino", "Casino", "Inside"),
    ("church", "Auditorium / Theatre", "Inside"),
    ("cathedral", "Auditorium / Theatre", "Inside"),
]


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def utcnow():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- the API ---

class Wikidata:
    """Thin MediaWiki client: paced, retried, and it counts what it used."""

    def __init__(self, reqs_per_sec=REQS_PER_SEC):
        self.gap = 1.0 / reqs_per_sec
        self.last = 0.0
        self.calls = 0
        self.label_cache = {}

    def _get(self, params):
        params.setdefault("format", "json")
        params.setdefault("formatversion", "2")
        url = API + "?" + urllib.parse.urlencode(params)
        for attempt in range(MAX_RETRIES):
            wait = self.gap - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=45) as r:
                    self.calls += 1
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code in (429, 503) and attempt < MAX_RETRIES - 1:
                    back = 2 ** attempt * 5
                    log(f"   HTTP {e.code}; backing off {back}s")
                    time.sleep(back)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("unreachable")

    def search(self, text, limit=8):
        try:
            r = self._get({"action": "wbsearchentities", "search": text[:300],
                           "language": "en", "uselang": "en", "type": "item",
                           "limit": limit})
        except Exception as e:
            log(f"   search failed for {text!r}: {type(e).__name__}")
            return []
        return r.get("search", [])

    def entities(self, qids, props="claims|labels|descriptions|sitelinks"):
        out = {}
        qids = [q for q in qids if q]
        for i in range(0, len(qids), 50):
            chunk = qids[i:i + 50]
            try:
                r = self._get({"action": "wbgetentities", "ids": "|".join(chunk),
                               "props": props, "languages": "en"})
            except Exception as e:
                log(f"   fetch failed for {len(chunk)} ids: {type(e).__name__}")
                continue
            out.update(r.get("entities", {}) or {})
        return out

    def labels(self, qids):
        """Labels for referenced items (cities, countries, classes), cached."""
        need = [q for q in set(qids) if q and q not in self.label_cache]
        if need:
            for qid, ent in self.entities(need, props="labels").items():
                self.label_cache[qid] = (ent.get("labels", {})
                                         .get("en", {}).get("value", ""))
            for q in need:
                self.label_cache.setdefault(q, "")
        return {q: self.label_cache.get(q, "") for q in qids if q}


# ------------------------------------------------------------- extraction ---

def _claim_values(ent, prop):
    for c in ent.get("claims", {}).get(prop, []):
        dv = c.get("mainsnak", {}).get("datavalue", {})
        if dv:
            yield dv.get("value")


def first_claim(ent, prop):
    for v in _claim_values(ent, prop):
        return v
    return None


def parse_entity(ent):
    """The handful of facts we want out of a Wikidata item."""
    cap = first_claim(ent, P_CAPACITY)
    if isinstance(cap, dict):
        try:
            cap = int(float(str(cap.get("amount", "")).lstrip("+")))
        except (TypeError, ValueError):
            cap = None
    coord = first_claim(ent, P_COORD) or {}
    inception = first_claim(ent, P_INCEPTION) or {}
    year = None
    if isinstance(inception, dict) and inception.get("time"):
        try:
            year = int(str(inception["time"])[1:5])
        except ValueError:
            year = None
    ref = lambda v: v.get("id") if isinstance(v, dict) else None
    return {
        "capacity": cap if (cap and 0 < cap < 400000) else None,
        "latitude": coord.get("latitude"),
        "longitude": coord.get("longitude"),
        "admin_qid": ref(first_claim(ent, P_ADMIN)),
        "country_qid": ref(first_claim(ent, P_COUNTRY)),
        "class_qids": [ref(v) for v in _claim_values(ent, P_INSTANCE) if ref(v)],
        "opened_year": year,
        "label": ent.get("labels", {}).get("en", {}).get("value", ""),
        "description": ent.get("descriptions", {}).get("en", {}).get("value", ""),
        "wikipedia": ent.get("sitelinks", {}).get("enwiki", {}).get("title"),
    }


def classify(class_labels, description):
    """venue_type and indoor/outdoor from the item's classes, then its blurb."""
    haystacks = [l.lower() for l in class_labels if l] + [(description or "").lower()]
    # keywords outer, text inner: TYPE_KEYWORDS is ordered most-specific first,
    # and an item carries several classes. Looping the other way let a vaguer
    # class win just because it came first -- the Arena di Verona is classed as
    # both an amphitheatre and a theatre, and came out "Inside".
    for keyword, vtype, io in TYPE_KEYWORDS:
        for text in haystacks:
            if keyword in text:
                return vtype, io
    return None, None


def km_apart(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    except (TypeError, ValueError):
        return None
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ----------------------------------------------------------------- scoring --

def score(target, cand, city_label, country_label):
    """
    Decide how much to believe a candidate.

    High     the name matches exactly once normalised, and the place agrees
             (same city, or a coordinate close enough to ours)
    Medium   the name matches exactly but only the country agrees, or the name
             is a near-match and the city agrees
    Low      anything else plausible
    None     not a match

    Only High is used by the build; Medium and Low are kept so they can be
    reviewed or promoted by hand.
    """
    name_exact = norm_key(cand["label"]) == target["venue_norm"]
    ratio = 0
    if not name_exact and fuzz is not None:
        ratio = fuzz.ratio(norm_key(cand["label"]), target["venue_norm"])
    if not name_exact and ratio < 88:
        return None, None

    city_norm = target["city_norm"] or ""
    blurb = f'{city_label} {cand["description"]}'.lower()
    city_ok = bool(city_norm) and (norm_key(city_label) == city_norm
                                   or city_norm in blurb)
    country_ok = bool(target["country"]) and (
        (country_label or "").strip().lower() == target["country"].strip().lower())

    dist = None
    if not target["centroid"] and cand["latitude"] is not None:
        dist = km_apart(target["latitude"], target["longitude"],
                        cand["latitude"], cand["longitude"])
    coord_ok = dist is not None and dist <= COORD_TOLERANCE_KM

    if name_exact and (city_ok or coord_ok):
        return "High", ("name + city" if city_ok else f"name + coords ({dist:.0f}km)")
    if name_exact and country_ok:
        return "Medium", "name + country"
    if ratio >= 88 and (city_ok or coord_ok):
        return "Medium", f"fuzzy name {ratio:.0f} + place"
    if name_exact:
        return "Low", "name only"
    return "Low", f"fuzzy name {ratio:.0f}"


# -------------------------------------------------------------------- db ----

def connect(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found")
    con = sqlite3.connect(path, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.row_factory = sqlite3.Row
    return con


def ensure_table(con):
    con.execute("""CREATE TABLE IF NOT EXISTS ref_venue_enrichment (
        venue_uid TEXT PRIMARY KEY, venue TEXT, city TEXT, country TEXT,
        status TEXT, capacity INTEGER, venue_type TEXT, outside_inside TEXT,
        latitude TEXT, longitude TEXT, opened_year INTEGER, wikidata_id TEXT,
        wikipedia_page TEXT, source TEXT, confidence TEXT, match_method TEXT,
        checked_at TEXT)""")
    con.execute("CREATE INDEX IF NOT EXISTS ix_enrich_status "
                "ON ref_venue_enrichment(status, confidence)")
    con.commit()


TARGETS = """
WITH centroids AS (
    SELECT latitude, longitude FROM venues
    WHERE latitude IS NOT NULL GROUP BY 1, 2 HAVING COUNT(*) > 20
)
SELECT v.venue_uid, v.venue, v.venue_norm, v.city, v.city_norm, v.country,
       v.latitude, v.longitude, v.events,
       (c.latitude IS NOT NULL) AS centroid
FROM venues v
LEFT JOIN centroids c ON c.latitude = v.latitude AND c.longitude = v.longitude
WHERE v.events >= ?
  AND ({country_filter})
  AND (v.capacity IS NULL OR v.outside_inside IS NULL)
  AND NOT EXISTS (SELECT 1 FROM ref_venue_enrichment e
                  WHERE e.venue_uid = v.venue_uid {already})
ORDER BY v.events DESC
LIMIT ?
"""


def targets(con, min_events, limit, redo="none", countries=None):
    """
    Venues still worth asking about, biggest first.

    `redo` says what counts as "already done" and so gets skipped:
        none    anything previously looked up, hit or miss  (the default --
                this is what makes a run resumable)
        misses  only previous hits are skipped, so misses are asked again --
                for when Wikidata has since gained the item
        all     nothing is skipped -- for when OUR logic changed and the stored
                answers need re-deriving
    """
    already = {"none": "", "misses": "AND e.status = 'matched'", "all": "AND 0"}[redo]
    marks = ",".join("?" * len(countries)) if countries else ""
    sql = TARGETS.format(
        already=already,
        country_filter=f"v.country IN ({marks})" if countries else "1=1")
    params = [min_events, *(countries or []), limit]
    return [dict(r) for r in con.execute(sql, params)]


def cmd_targets(args, con):
    ensure_table(con)
    rows = targets(con, args.min_events, args.limit, countries=args.country)
    marks = ",".join("?" * len(args.country)) if args.country else ""
    total = con.execute(f"""SELECT COUNT(*), SUM(events) FROM venues
                           WHERE events >= ?
                             AND ({f"country IN ({marks})" if args.country else "1=1"})
                             AND (capacity IS NULL OR outside_inside IS NULL)
                             AND venue_uid NOT IN (SELECT venue_uid FROM ref_venue_enrichment)""",
                        (args.min_events, *(args.country or []))).fetchone()
    where = f" in {', '.join(args.country)}" if args.country else ""
    log(f"{total[0]:,} venues{where} with >= {args.min_events} events still need a "
        f"lookup ({total[1] or 0:,} events between them)")
    for r in rows[:args.show]:
        log(f"   {r['events']:>6,}  {r['venue']} - {r['city']}, {r['country']}")
    return 0


def cmd_pull(args, con):
    ensure_table(con)
    rows = targets(con, args.min_events, args.limit, redo=args.redo,
                   countries=args.country)
    if not rows:
        log("nothing to look up - every venue above the threshold has been checked")
        return 0
    wd = Wikidata(reqs_per_sec=args.rate)
    log(f"looking up {len(rows):,} venues on Wikidata at up to {args.rate}/s "
        f"(Wikimedia throttling decides the real pace, so the ETA below is "
        f"measured as it goes, not guessed)")

    # A bar at a terminal; periodic lines with a measured ETA when the output is
    # piped or redirected, because a long job that prints nothing for minutes at
    # a time is indistinguishable from a hung one.
    use_bar = _tqdm is not None and sys.stderr.isatty()
    it = (_tqdm(rows, desc="wikidata", unit="venue", file=sys.stderr,
                dynamic_ncols=True, leave=False) if use_bar else rows)
    started = time.monotonic()

    INSERT = """INSERT OR REPLACE INTO ref_venue_enrichment
        (venue_uid, venue, city, country, status, capacity, venue_type,
         outside_inside, latitude, longitude, opened_year, wikidata_id,
         wikipedia_page, source, confidence, match_method, checked_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    now = utcnow()
    tally = {"High": 0, "Medium": 0, "Low": 0, "no match": 0}

    for i, t in enumerate(it, start=1):
        if use_bar:
            it.set_description(f"wikidata | {t['venue'][:28]}", refresh=False)
        hits = wd.search(t["venue"])
        best = best_conf = best_how = None
        if hits:
            ents = wd.entities([h["id"] for h in hits])
            parsed = {q: parse_entity(e) for q, e in ents.items()}
            refs = set()
            for p in parsed.values():
                refs.update([p["admin_qid"], p["country_qid"], *p["class_qids"]])
            names = wd.labels([r for r in refs if r])
            for qid, p in parsed.items():
                conf, how = score(t, p, names.get(p["admin_qid"], ""),
                                  names.get(p["country_qid"], ""))
                if conf is None:
                    continue
                rank = {"High": 3, "Medium": 2, "Low": 1}
                if best is None or rank[conf] > rank[best_conf]:
                    p["qid"] = qid
                    p["class_labels"] = [names.get(c, "") for c in p["class_qids"]]
                    best, best_conf, best_how = p, conf, how
        if best is None:
            con.execute(INSERT, (t["venue_uid"], t["venue"], t["city"], t["country"],
                                 "no match", None, None, None, None, None, None,
                                 None, None, "wikidata", None, None, now))
            tally["no match"] += 1
        else:
            vtype, io = classify(best["class_labels"], best["description"])
            con.execute(INSERT, (
                t["venue_uid"], t["venue"], t["city"], t["country"], "matched",
                best["capacity"], vtype, io,
                str(best["latitude"]) if best["latitude"] is not None else None,
                str(best["longitude"]) if best["longitude"] is not None else None,
                best["opened_year"], best["qid"], best["wikipedia"],
                "wikidata", best_conf, best_how, now))
            tally[best_conf] += 1
        if i % 25 == 0 or i == len(rows):
            con.commit()
        if use_bar:
            it.set_postfix(high=tally["High"], med=tally["Medium"],
                           miss=tally["no match"], refresh=False)
        elif i % args.progress_every == 0 or i == len(rows):
            elapsed = time.monotonic() - started
            per = elapsed / i
            eta = (len(rows) - i) * per
            log(f"   {i:,}/{len(rows):,} ({i / len(rows):.0%})  "
                f"high={tally['High']:,} med={tally['Medium']:,} "
                f"low={tally['Low']:,} miss={tally['no match']:,}  "
                f"{per:.1f}s/venue, ~{eta / 60:.0f} min left  "
                f"({wd.calls:,} API calls)")
    if use_bar:
        it.close()
    con.commit()
    log(f"done: {tally['High']:,} high-confidence matches now available to the build")
    log("   run `python pipeline.py run` (or `build_events.py build-venues`) to use them")
    return 0


def cmd_load_manual(args, con):
    """Hand corrections, which outrank anything looked up."""
    ensure_table(con)
    if not os.path.exists(MANUAL_CSV):
        log(f"no {MANUAL_CSV}; create it with columns: "
            f"venue_uid,capacity,venue_type,outside_inside,latitude,longitude,note")
        return 0
    now, n = utcnow(), 0
    with open(MANUAL_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            uid = (row.get("venue_uid") or "").strip()
            if not uid:
                continue
            con.execute("""INSERT OR REPLACE INTO ref_venue_enrichment
                (venue_uid, venue, city, country, status, capacity, venue_type,
                 outside_inside, latitude, longitude, source, confidence,
                 match_method, checked_at)
                VALUES (?,?,?,?,'matched',?,?,?,?,?,'manual','High','manual override',?)""",
                (uid, row.get("venue"), row.get("city"), row.get("country"),
                 row.get("capacity") or None, row.get("venue_type") or None,
                 row.get("outside_inside") or None, row.get("latitude") or None,
                 row.get("longitude") or None, now))
            n += 1
    con.commit()
    log(f"applied {n:,} manual overrides from {MANUAL_CSV}")
    return 0


def cmd_status(args, con):
    ensure_table(con)
    log("== ref_venue_enrichment ==")
    for r in con.execute("""SELECT status, COALESCE(confidence,'-') conf, COUNT(*) n,
                                   SUM(capacity IS NOT NULL) with_cap,
                                   SUM(outside_inside IS NOT NULL) with_io,
                                   SUM(latitude IS NOT NULL) with_coords
                            FROM ref_venue_enrichment GROUP BY 1,2 ORDER BY n DESC"""):
        log(f"   {r['status']:10s} {r['conf']:7s} {r['n']:7,}  "
            f"capacity {r['with_cap']:6,}  in/out {r['with_io']:6,}  coords {r['with_coords']:6,}")
    row = con.execute("""SELECT COUNT(*) n, SUM(capacity IS NOT NULL) cap,
                                SUM(outside_inside IS NOT NULL) io FROM venues""").fetchone()
    log(f"   venues table: {row['n']:,} venues, {row['cap']:,} with capacity, "
        f"{row['io']:,} with indoor/outdoor")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=paths.DB)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("targets", help="What still needs a lookup.")
    t.add_argument("--min-events", type=int, default=20)
    t.add_argument("--limit", type=int, default=100000)
    t.add_argument("--show", type=int, default=20)
    t.add_argument("--country", nargs="+", default=None,
                   help="only these countries, e.g. --country Italy France")
    t.set_defaults(func=cmd_targets)

    pl = sub.add_parser("pull", help="Look venues up on Wikidata.")
    pl.add_argument("--limit", type=int, default=500, help="venues per run")
    pl.add_argument("--min-events", type=int, default=20)
    pl.add_argument("--rate", type=float, default=REQS_PER_SEC, help="requests/sec")
    pl.add_argument("--progress-every", type=int, default=10,
                    help="venues between progress lines when not at a terminal")
    pl.add_argument("--country", nargs="+", default=None,
                    help="only these countries, e.g. --country Italy")
    pl.add_argument("--redo", choices=("none", "misses", "all"), default="none",
                    help="what to ask about again: none (default, resumable -- skips "
                         "everything already looked up), misses (retry previous "
                         "no-matches), all (re-derive everything, e.g. after the "
                         "classification rules change)")
    pl.set_defaults(func=cmd_pull)

    lm = sub.add_parser("load-manual", help=f"Apply {os.path.basename(MANUAL_CSV)}.")
    lm.set_defaults(func=cmd_load_manual)

    s = sub.add_parser("status", help="Coverage so far.")
    s.set_defaults(func=cmd_status)

    args = p.parse_args()
    con = connect(args.db)
    try:
        return args.func(args, con)
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
