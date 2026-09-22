#!/usr/bin/env python3
"""
Build a refreshed copy of arenas-dashboard.xlsx from setlistfm.db.

Nothing here touches the pipeline, and nothing here writes to the original
workbook -- the template is opened read-only and a brand new .xlsx is written.

WHAT IT CHANGES

  Event Data   replaced with events from the database: everything at a curated
               arena since --from, in Category A/B/C plus the non-Artist types
               (family, entertainment, comedy, sport). Sporting rows come
               from the fixtures table and Pollstar's Sports genre, with
               competition/result filled where known.
  Currency     every Pollstar money column is already USD, so the sheet's
               currency, currency_iso, fx_units_per_gbp, gross_gbp and
               ticket_price_avg_gbp are filled in here using the USD rate from
               ref_fx_rates. They used to be left blank, which left the
               dashboard's sterling figures empty until someone patched them.
  Arena Data   replaced with every venue appearing in that Event Data. Curated
               columns (owner, opened year, verification and so on) are carried
               across from the template for arenas it already knew; venues new
               to the sheet get the columns we can fill and blanks elsewhere.

Everything else -- Dashboard, Stadium Search, the calc sheets, Artist
Categorisation, FX Rates, Hospitality, both charts, all formatting and
conditional formatting -- is copied through untouched.

WHY IT IS BUILT THIS WAY

  * The file is rebuilt at the zip level, copying every part byte-for-byte
    except the sheets that change. Round-tripping through openpyxl drops 27
    internal parts, including both charts' styling and relationships.
  * The workbook's formula layers hard-code row bounds. Event Data is
    referenced up to row 153,214 and Artist Categorisation to row 14,000, so
    writing fewer rows than that is safe. Arena Data, though, is bounded at row
    677 in 83 places, which would hide any venue past it -- so those bounds are
    widened to ARENA_BOUND and the sheet is padded with blank rows to match.
  * Cells are written as static values, with one exception. `in_scope` (column
    AV) stays a formula: it is what tells the Dashboard which rows belong to the
    venue you have selected, so freezing it would pin the dashboard to whatever
    venue happened to be chosen when this ran.

USAGE
    python export_workbook.py
    python export_workbook.py --from 2018-01-01 --out arenas-2018.xlsx
    python export_workbook.py --all-venues     # ignore the curated-arena filter
"""

import argparse
import datetime as dt
import os
import re
import shutil
import sqlite3
import zipfile

import paths

TEMPLATE = paths.ARENA_WORKBOOK
DEFAULT_OUT = paths.here("arenas-dashboard-refreshed.xlsx")
DEFAULT_FROM = "2015-01-01"

# Arena Data is referenced as $2:$677 in 83 places across Search Calc and Dash
# Calc. Widening those to this bound, and padding the sheet to match, lets new
# venues be seen. A lookup over trailing blank rows is harmless.
ARENA_OLD_BOUND = 677
ARENA_BOUND = 2000

SHEET_EVENTS = "Event Data"
SHEET_ARENAS = "Arena Data"
PATCH_SHEETS = ("Search Calc", "Dash Calc")

EXCEL_EPOCH = dt.date(1899, 12, 30)


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------- xlsx parts

def sheet_parts(z):
    """{sheet name: 'xl/worksheets/sheetN.xml'} from the workbook relationships."""
    wbx = z.read("xl/workbook.xml").decode("utf8")
    rels = z.read("xl/_rels/workbook.xml.rels").decode("utf8")
    rid = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="(worksheets/[^"]+)"', rels))
    out = {}
    for m in re.finditer(r'<sheet name="([^"]+)"[^>]*r:id="(rId\d+)"', wbx):
        target = rid.get(m.group(2))
        if target:
            out[m.group(1)] = "xl/" + target
    return out


def split_sheet(xml):
    """(everything before sheetData, the sheetData body, everything after)."""
    m = re.search(r"<sheetData\s*/>", xml)
    if m:
        return xml[:m.start()], "", xml[m.end():]
    a = xml.index("<sheetData>")
    b = xml.index("</sheetData>")
    return xml[:a + len("<sheetData>")], xml[a + len("<sheetData>"):b], xml[b:]


def first_rows(body, count=3):
    """The first `count` <row> elements, as raw XML."""
    return re.findall(r"<row[^>]*>.*?</row>", body, re.S)[:count]


def column_styles(row_xml):
    """{column letter: style index} taken from a template data row."""
    styles = {}
    for m in re.finditer(r'<c r="([A-Z]+)\d+"([^>]*)>', row_xml):
        s = re.search(r' s="(\d+)"', m.group(2))
        styles[m.group(1)] = s.group(1) if s else None
    return styles


def col_letter(i):
    """0 -> A, 25 -> Z, 26 -> AA."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


_ESCAPE = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
_BAD_XML_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def esc(text):
    out = _BAD_XML_CHARS.sub("", str(text))
    return "".join(_ESCAPE.get(ch, ch) for ch in out)


def cell(ref, style, value, formula=None):
    """One <c> element. Values are written as numbers or inline strings."""
    s = f' s="{style}"' if style else ""
    if formula is not None:
        return f'<c r="{ref}"{s}><f>{esc(formula)}</f></c>'
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return f'<c r="{ref}"{s}><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)):
        return f'<c r="{ref}"{s}><v>{value}</v></c>'
    if isinstance(value, dt.date):
        return f'<c r="{ref}"{s}><v>{(value - EXCEL_EPOCH).days}</v></c>'
    # everything else is text. Inline strings avoid touching sharedStrings.xml,
    # which keeps that part byte-identical to the template.
    return f'<c r="{ref}"{s} t="inlineStr"><is><t xml:space="preserve">{esc(value)}</t></is></c>'


def build_rows(records, styles, n_cols, start_row=2, formulas=None):
    """Rows of XML. `formulas` maps a column letter to a template formula."""
    formulas = formulas or {}
    out = []
    for n, rec in enumerate(records):
        r = start_row + n
        cells = []
        for i in range(n_cols):
            letter = col_letter(i)
            ref = f"{letter}{r}"
            if letter in formulas:
                cells.append(cell(ref, styles.get(letter),
                                  None, formula=formulas[letter].replace("\x00", str(r))))
            else:
                cells.append(cell(ref, styles.get(letter),
                                  rec[i] if i < len(rec) else None))
        out.append(f'<row r="{r}" spans="1:{n_cols}">' + "".join(cells) + "</row>")
    return out


def blank_rows(first, last, n_cols):
    return [f'<row r="{r}" spans="1:{n_cols}"/>' for r in range(first, last + 1)]


def retarget(formula, row):
    """
    Point a template formula at a different row: $F2 -> $F57, AK2 -> AK57.

    Only RELATIVE row references move. An absolute one ($B$2, or a spill such
    as $DM$2#) is pinned on purpose and has to stay: in_scope compares each
    row's slug against the one cell holding the venue the dashboard is showing,
    so walking that reference down the sheet puts every row out of scope and
    every panel reads zero. The old pattern ended `\\$?)2`, which swallowed the
    `$` of an absolute row and retargeted it along with the rest.
    """
    return re.sub(r"(\$?[A-Z]{1,2})2\b", lambda m: m.group(1) + "\x00", formula)


# ------------------------------------------------------------------- the data

EVENT_SQL = """
SELECT e.venue, e.city, e.country, e.date_iso, e.headliner, e.support,
       e.pollstar_genre, e.pollstar_promoter, e.pollstar_market,
       e.pollstar_run_shows, e.pollstar_run_tickets, e.pollstar_tickets_sold,
       e.pollstar_capacity, e.pollstar_capacity_pct, e.pollstar_gross_usd,
       e.pollstar_price_min, e.pollstar_price_max, e.pollstar_price_avg,
       e.pollstar_venue_id, e.arena_id, e.n_artists, e.category, e.tour,
       e.setlist_ids, e.setlist_urls, e.num_songs, e.latitude, e.longitude,
       e.previous_city, e.next_city, e.venue_uid, e.pollstar_id,
       COALESCE(e.source, 'setlistfm'), e.competition, e.result
FROM events e
WHERE e.date_iso >= ?
  AND (e.category IN ('Category A','Category B','Category C')
       OR e.event_type <> 'Artist')
  {arena_filter}
ORDER BY e.date_iso, e.venue
"""


def slug(text):
    s = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    return s


def iso_to_date(s):
    try:
        return dt.date.fromisoformat(str(s)[:10])
    except (TypeError, ValueError):
        return None


def gbp_rate(con):
    """
    USD per GBP, from ref_fx_rates (loaded with the arena workbook).

    Every money column Pollstar gives us is already in USD -- its `currency`
    field records what was taken at the box office, but the figures themselves
    are converted, which you can see in the data: average gross divided by
    tickets equals the stated average price to three decimal places in USD,
    Euro, Sterling, Canadian, Australian and Swiss rows alike. So one rate
    converts the lot.
    """
    row = con.execute("SELECT units_per_gbp, basis FROM ref_fx_rates "
                      "WHERE iso = 'USD'").fetchone()
    if not row or not row[0]:
        log("!! no USD rate in ref_fx_rates - leaving the GBP columns blank "
            "(run `build_events.py load-arenas` to load them)")
        return None, None
    return float(row[0]), row[1]


def event_records(con, since, curated_only, headers):
    sql = EVENT_SQL.format(
        arena_filter="AND e.arena_id IS NOT NULL" if curated_only else "")
    rows = con.execute(sql, (since,)).fetchall()
    n = len(headers)
    rate, basis = gbp_rate(con)
    if rate:
        log(f"   money columns: USD, converted at {rate} USD/GBP ({basis})")
    # find the currency columns by name, so a reordered sheet cannot silently
    # write them into the wrong place
    hx = {str(h).strip().lower(): i for i, h in enumerate(headers) if h}
    i_cur, i_iso = hx.get("currency"), hx.get("currency_iso")
    i_fx, i_ggbp = hx.get("fx_units_per_gbp"), hx.get("gross_gbp")
    i_pgbp = hx.get("ticket_price_avg_gbp")
    out = []
    for r in rows:
        (venue, city, country, date_iso, headliner, support, genre, promoter,
         market, run_shows, run_tickets, avg_tickets, capacity, cap_pct,
         gross, pmin, pmax, pavg, ps_venue, arena_id, n_artists, category,
         tour, sl_ids, sl_urls, songs, lat, lon, prev_city, next_city,
         venue_uid, ps_id, source, competition, result) = r
        rec = [None] * n
        rec[0] = venue
        rec[1] = city
        rec[2] = country
        rec[3] = slug(venue)
        rec[4] = iso_to_date(date_iso)
        rec[5] = headliner
        rec[6] = support
        rec[7] = genre
        rec[8] = promoter
        rec[9] = market
        rec[10] = competition
        rec[11] = result
        rec[12] = run_shows
        rec[13] = run_tickets
        rec[14] = avg_tickets
        # P tickets_available: not held
        rec[16] = capacity
        rec[17] = cap_pct
        rec[18] = gross
        rec[19] = pmin
        rec[20] = pmax
        rec[21] = pavg
        # W currency: not carried on events
        rec[23] = source if source != "setlistfm" else ("pollstar+setlistfm" if ps_id else "setlistfm")
        # Y pollstar_event_id: the export carries no Pollstar event id
        rec[25] = ps_venue
        rec[26] = arena_id
        rec[27] = n_artists
        rec[28] = category
        # currency / FX. Filled only where there is actually money on the row,
        # so a setlist-only event is not labelled with a currency it never had.
        if rate and any(v is not None for v in (gross, pmin, pmax, pavg)):
            if i_cur is not None:
                rec[i_cur] = "USD"
            if i_iso is not None:
                rec[i_iso] = "USD"
            if i_fx is not None:
                rec[i_fx] = rate
            if i_ggbp is not None and gross is not None:
                rec[i_ggbp] = round(gross / rate, 2)
            if i_pgbp is not None and pavg is not None:
                rec[i_pgbp] = round(pavg / rate, 2)
        # AH..AK support_*: left blank, recomputed by the sheet if wanted
        rec[37] = tour
        rec[38] = sl_ids
        rec[39] = sl_urls
        rec[40] = songs
        rec[41] = float(lat) if lat not in (None, "") else None
        rec[42] = float(lon) if lon not in (None, "") else None
        rec[43] = source if source != "setlistfm" else ("pollstar+setlistfm" if ps_id else "setlistfm")
        rec[44] = prev_city
        rec[45] = next_city
        rec[46] = 1
        # AV in_scope stays a formula
        out.append(rec)
    return out


def arena_records(con, since, curated_only, headers, template_rows):
    """Every venue in the exported events, curated columns carried across."""
    sql = f"""
        SELECT v.venue, v.city, v.country, v.countryCode, v.capacity,
               v.venue_type, v.outside_inside, v.latitude, v.longitude,
               v.arena_id, v.venue_uid, COUNT(*) AS events
        FROM events e JOIN venues v ON v.venue_uid = e.venue_uid
        WHERE e.date_iso >= ?
          AND (e.category IN ('Category A','Category B','Category C')
               OR e.event_type <> 'Artist')
          {"AND e.arena_id IS NOT NULL" if curated_only else ""}
        GROUP BY v.venue_uid
        ORDER BY events DESC
    """
    idx = {h: i for i, h in enumerate(headers)}
    n = len(headers)
    out = []
    for (venue, city, country, cc, capacity, vtype, io, lat, lon,
         arena_id, venue_uid, events) in con.execute(sql, (since,)):
        # start from the template's own row for this arena, so the curated
        # columns (owner, opened year, verification, address ...) survive
        rec = list(template_rows.get(str(arena_id or ""), [None] * n))
        rec += [None] * (n - len(rec))
        rec = rec[:n]

        def put(col, value):
            if col in idx and value not in (None, ""):
                rec[idx[col]] = value

        put("name", venue)
        put("city", city)
        put("country", country)
        put("concert_capacity", capacity)
        put("venue_type", vtype)
        put("outside_inside", io)
        put("latitude", float(lat) if lat not in (None, "") else None)
        put("longitude", float(lon) if lon not in (None, "") else None)
        put("arena_id", arena_id or venue_uid)
        put("observed_concert_bookings", events)
        out.append(rec)
    return out


def template_arena_rows(z, parts, headers):
    """Existing Arena Data rows, keyed by arena_id, read with openpyxl."""
    import openpyxl
    import warnings
    warnings.filterwarnings("ignore")
    wb = openpyxl.load_workbook(TEMPLATE, read_only=True, data_only=True)
    ws = wb[SHEET_ARENAS]
    it = ws.iter_rows(values_only=True)
    next(it)
    idx = {h: i for i, h in enumerate(headers)}
    key = idx.get("arena_id")
    out = {}
    for r in it:
        if not r or not r[0]:
            continue
        row = list(r) + [None] * (len(headers) - len(r))
        if key is not None and row[key]:
            out[str(row[key])] = row[:len(headers)]
    wb.close()
    return out


def shared_strings(z):
    """xl/sharedStrings.xml as a list. Header cells reference it by index."""
    try:
        xml = z.read("xl/sharedStrings.xml").decode("utf8")
    except KeyError:
        return []
    out = []
    for si in re.findall(r"<si>(.*?)</si>", xml, re.S):
        out.append("".join(re.findall(r"<t[^>]*>(.*?)</t>", si, re.S)))
    return out


def headers_of(z, part, shared):
    """
    Column names from row 1.

    They are usually shared strings rather than inline ones, so the index has to
    be resolved -- reading only inline strings returns a row of blanks, and any
    name-keyed write downstream then silently does nothing.
    """
    xml = z.read(part).decode("utf8")
    _, body, _ = split_sheet(xml)
    row1 = first_rows(body, 1)
    if not row1:
        return []
    names = []
    for m in re.finditer(r'<c ([^>]*)>(.*?)</c>', row1[0], re.S):
        attrs, inner = m.group(1), m.group(2)
        t = re.search(r' t="([^"]+)"', attrs)
        kind = t.group(1) if t else "n"
        if kind == "s":
            v = re.search(r"<v>(\d+)</v>", inner)
            names.append(shared[int(v.group(1))] if v and int(v.group(1)) < len(shared) else "")
        elif kind == "inlineStr":
            tt = re.search(r"<t[^>]*>(.*?)</t>", inner, re.S)
            names.append(tt.group(1) if tt else "")
        else:
            v = re.search(r"<v>(.*?)</v>", inner, re.S)
            names.append(v.group(1) if v else "")
    return names


# ------------------------------------------------------------------- the build

def main():
    p = argparse.ArgumentParser(
        description="Write a refreshed copy of the arenas dashboard from the database.")
    p.add_argument("--db", default=paths.DB)
    p.add_argument("--template", default=TEMPLATE)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--from", dest="since", default=DEFAULT_FROM,
                   metavar="YYYY-MM-DD")
    p.add_argument("--all-venues", action="store_true",
                   help="Include venues with no curated arena. Far more rows; "
                        "the workbook's formulas will struggle.")
    args = p.parse_args()

    if not os.path.exists(args.template):
        raise SystemExit(f"template not found: {args.template}")
    if os.path.abspath(args.out) == os.path.abspath(args.template):
        raise SystemExit("--out must differ from the template; it is never overwritten.")

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    z = zipfile.ZipFile(args.template)
    parts = sheet_parts(z)
    for needed in (SHEET_EVENTS, SHEET_ARENAS):
        if needed not in parts:
            raise SystemExit(f"template has no '{needed}' sheet")

    log(f"template : {args.template}")
    log(f"database : {args.db}")
    log(f"scope    : from {args.since}, "
        + ("all venues" if args.all_venues else "curated arenas only"))

    shared = shared_strings(z)
    ev_headers = headers_of(z, parts[SHEET_EVENTS], shared)
    ar_headers = headers_of(z, parts[SHEET_ARENAS], shared)
    if not any(ar_headers):
        raise SystemExit("could not read Arena Data column names from the template")
    log(f"columns  : Event Data {len(ev_headers)}, Arena Data {len(ar_headers)}")

    curated = not args.all_venues
    log("querying events ...")
    events = event_records(con, args.since, curated, ev_headers)
    log(f"   {len(events):,} events")

    log("querying venues ...")
    tmpl_rows = template_arena_rows(z, parts, ar_headers)
    arenas = arena_records(con, args.since, curated, ar_headers, tmpl_rows)
    log(f"   {len(arenas):,} venues ({len(tmpl_rows):,} known to the template)")

    if len(events) > 153213:
        log(f"!! {len(events):,} events exceeds the 153,214 the sheet's formulas "
            f"cover; rows past that will be invisible to the dashboard")
    if len(arenas) > ARENA_BOUND - 1:
        log(f"!! {len(arenas):,} venues exceeds ARENA_BOUND {ARENA_BOUND}")

    # ---- Event Data --------------------------------------------------------
    ev_xml = z.read(parts[SHEET_EVENTS]).decode("utf8")
    head, body, tail = split_sheet(ev_xml)
    rows = first_rows(body, 2)
    header_row, sample = rows[0], (rows[1] if len(rows) > 1 else rows[0])
    styles = column_styles(sample)
    in_scope_col = col_letter(len(ev_headers) - 1)
    m = re.search(rf'<c r="{in_scope_col}2"[^>]*>\s*<f>(.*?)</f>', sample, re.S)
    formulas = {}
    if m:
        formulas[in_scope_col] = retarget(m.group(1), 2)
        log(f"   keeping {in_scope_col} (in_scope) as a formula")
    else:
        log(f"!! no in_scope formula found in {in_scope_col}2 - writing values")
    log("building Event Data ...")
    ev_body = header_row + "".join(
        build_rows(events, styles, len(ev_headers), 2, formulas))
    new_ev = head + ev_body + tail
    new_ev = re.sub(r'<dimension ref="[^"]*"/>',
                    f'<dimension ref="A1:{col_letter(len(ev_headers)-1)}{len(events)+1}"/>',
                    new_ev, count=1)

    # ---- Arena Data --------------------------------------------------------
    ar_xml = z.read(parts[SHEET_ARENAS]).decode("utf8")
    head2, body2, tail2 = split_sheet(ar_xml)
    rows2 = first_rows(body2, 2)
    header_row2, sample2 = rows2[0], (rows2[1] if len(rows2) > 1 else rows2[0])
    styles2 = column_styles(sample2)
    log("building Arena Data ...")
    ar_rows = build_rows(arenas, styles2, len(ar_headers), 2)
    ar_rows += blank_rows(len(arenas) + 2, ARENA_BOUND, len(ar_headers))
    new_ar = head2 + header_row2 + "".join(ar_rows) + tail2
    new_ar = re.sub(r'<dimension ref="[^"]*"/>',
                    f'<dimension ref="A1:{col_letter(len(ar_headers)-1)}{ARENA_BOUND}"/>',
                    new_ar, count=1)

    # ---- widen the Arena Data bound in the calc sheets ---------------------
    patched = {parts[SHEET_EVENTS]: new_ev, parts[SHEET_ARENAS]: new_ar}
    widened = 0
    for name in PATCH_SHEETS:
        part = parts.get(name)
        if not part:
            continue
        xml = z.read(part).decode("utf8")
        xml, n = re.subn(
            r"(Arena Data'!\$?[A-Z]{1,2}\$?\d+:\$?[A-Z]{1,2}\$?)" + str(ARENA_OLD_BOUND),
            r"\g<1>" + str(ARENA_BOUND), xml)
        if n:
            patched[part] = xml
            widened += n
    log(f"   widened {widened} Arena Data range bounds "
        f"{ARENA_OLD_BOUND} -> {ARENA_BOUND}")

    # ---- write the new workbook -------------------------------------------
    # every other part is copied byte-for-byte, which is what keeps the charts,
    # formatting and conditional formatting intact
    tmp = args.out + ".tmp"
    log(f"writing {args.out} ...")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for item in z.infolist():
            if item.filename in patched:
                out.writestr(item, patched[item.filename].encode("utf8"))
            elif item.filename == "xl/calcChain.xml":
                continue          # stale after rewriting cells; Excel rebuilds it
            else:
                out.writestr(item, z.read(item.filename))
    z.close()
    con.close()
    shutil.move(tmp, args.out)
    size = os.path.getsize(args.out) / 1e6
    log(f"done. {len(events):,} events, {len(arenas):,} venues, {size:,.1f} MB")
    log("Open it and let Excel recalculate; the original is untouched.")


if __name__ == "__main__":
    main()
