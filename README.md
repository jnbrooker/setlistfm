# setlistfm

A live-music dataset and dashboard. It scrapes every setlist on setlist.fm into
SQLite, tiers the artists by audience size, folds support acts into the shows
they played, attaches Pollstar box office where it matches, and serves the whole
thing through a Streamlit dashboard.

The database itself is not in this repo — it is ~4 GB, and everything below the
raw `setlists` layer rebuilds from scratch in about twenty minutes.

## Setup

```bash
pip install -r requirements.txt
cp setlistfm_key.txt.example setlistfm_key.txt   # then paste your key in
```

A setlist.fm API key comes from https://api.setlist.fm/docs/1.0/index.html. The
standard tier allows 2 requests/second; the upgraded tier allows 16/sec and
50,000/day, which is what the scrapers are tuned for. `$SETLISTFM_API_KEY`
overrides the file.

Two spreadsheets are read but not committed, because one is over GitHub's 100 MB
file limit: `arenas-dashboard.xlsx` (curated arena data, artist categorisation
thresholds, Spotify and Pollstar reference tabs) and `pollstar-data.xlsx`
(box-office rows).

## Daily use

```bash
python setlistfm_db.py sync          # new and edited setlists from the API
python pipeline.py run --skip sync   # rebuild everything derived
streamlit run dashboard.py
```

`pipeline.py run` on its own does all three stages. Add `--pollstar` when the
Pollstar workbook changes and `--export` to write `events.csv`. It logs to
`logs/`, takes a lock so two runs cannot overlap, and stops at the first failure.

## The scripts

| | |
|---|---|
| `setlistfm_db.py` | the scraper. `backfill` walks calendar dates, `sync` pulls anything created or edited since the last run. Both upsert on `setlist_id`, so they can run in any order and be interrupted freely. |
| `artist_categories.py` | tiers every artist A–E from Spotify monthly listeners (scraped from kworb) and Pollstar touring rank, against thresholds read from the workbook. |
| `build_events.py` | collapses setlists into events, picks the headliner, routes previous/next city, matches Pollstar, and assembles the venue table. |
| `pipeline.py` | runs the above in order, with logging and locking. |
| `dashboard.py` | the Streamlit front end. |
| `paths.py` | every default path, resolved against this folder rather than the working directory. |

## The data model

Raw, never rewritten by the derived steps:

- `setlists` — one row per artist per night, straight from the API

Reference, loaded from the workbooks and kworb:

- `arenas`, `arena_aliases` — curated venues and every spelling of their names
- `ref_spotify`, `ref_pollstar`, `ref_dashboard_categories`, `category_thresholds`
- `ref_hospitality`, `ref_fx_rates`
- `pollstar_events` — box-office rows, matched to events by date, artist and venue

Derived, dropped and rebuilt on every run:

- `artist_categories` — one row per artist with its tier and the evidence for it
- `venues` — one row per building, with the best capacity available and a
  `capacity_source` saying whether that came from the curated sheet, from
  Pollstar's reported figures, or was inferred
- `events` — one row per show: the bill combined, category attached, tour
  routing, and Pollstar box office where it matched

Because the derived tables are a pure function of the raw layer plus the
reference data, a wrong answer is never repaired in place. Fix the rule and
re-run.

## Categorisation

Thresholds live in the workbook and are read at run time, not hardcoded:

- **A** — 40m+ Spotify monthly listeners, or Pollstar rank 50 or better
- **B** — 10m+ listeners, or rank 100 or better
- **C** — 1m+ listeners
- **D** — on the Spotify list but under 1m
- **E** — in neither list, so no measurement either way

`artist_categories.py verify` re-derives the workbook's own rows and diffs them,
which is the regression test that the implemented rules still match the sheet.

## Things worth knowing

- **Numbers will not match the workbook.** It counts events from a Pollstar
  extract of ~153k rows; this counts from `events`, which is setlist.fm-derived.
  The O2 Arena shows 1,230 events there and 1,566 here. Neither is wrong.
- **Tickets are sparse.** They exist only where a Pollstar row matched, which is
  about 10% of events overall and ~40% of Category A shows in the US since 1995.
- **Some capacities are observed, not verified.** `venues.capacity_source` says
  which, and observed figures have outliers.
- **`backfill_dates.capped = 1`** marks dates with more than 10,000 setlists,
  which the search endpoint cannot page past. Those dates are incomplete and
  re-running does not fix them.
