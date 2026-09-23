# venue_model — would a new venue here actually change anything?

A transparent model of where touring artists choose to play, built to answer one
question: **if this city gained a room of size X, how many more shows would it
get, and why?**

Deliberately not machine learning. Every number this produces must be traceable
to a variable you can name and a coefficient you can read, because the output is
meant to survive questioning by people who will disagree with it.

---

## The four layers, and what each one is allowed to claim

Each layer is more ambitious and less certain than the one below. **They are
kept separate on purpose** so a reader can see exactly how far out on the limb
any given number sits.

| Layer | What it does | Language it licenses |
|---|---|---|
| **1. Descriptive** | capacity ladder, tours that skipped, where they went | "here is what happens today" |
| **2. Statistical** | conditional logit: which cities a tour picks, and why | "this *associates with* that" |
| **3. Causal** | difference-in-differences around real venue openings | "this *caused* that" |
| **4. Simulation** | Monte Carlo propagating layers 2–3 into shows and revenue | "here is the *distribution* of outcomes" |

Layer 1 is pure counting and needs no model at all. It is also, probably, most
of the value — if the descriptive picture shows no gap, nothing above it will
manufacture one.

---

## The central difficulty, stated plainly

The question is **causal** and the data is **observational**.

Cities with big arenas get big shows. But cities that *build* big arenas are
cities where promoters already expected demand. The arena did not cause the
shows; anticipated demand caused both. Regress shows on capacity across cities
and you get a large, clean, highly significant coefficient that is **mostly
selection**.

Any layer-2 number is therefore an *association*, and is labelled as one
everywhere it appears. Only layer 3 — comparing cities that built against
matched cities that did not — licenses the word "caused", and only with wide
error bars because there are a few dozen usable openings, not thousands.

**Consequence for the UI:** the app never shows a bare point estimate. Every
figure carries an interval and a note saying which layer produced it.

---

## Transparency, concretely

Three rules the code holds to:

1. **Every feature is named, unit-ed and sourced.** No derived variable enters a
   model without an entry in the feature dictionary saying what it is, where it
   came from, and what it does not capture.
2. **Every output decomposes.** An expected-shows number can be broken into the
   contribution of each variable — capacity, catchment, position on the route,
   cannibalisation — so "why" is answerable without reading code.
3. **Every extract is versioned.** A figure quoted in a report names the extract
   that produced it. The live database changes hourly while scrapers run; a
   feasibility number that cannot be reproduced is not a finding.

---

## Data sources

| Source | Used for | Known limits |
|---|---|---|
| `setlistfm.db` → `events` | shows, venues, capacity, indoor/outdoor, tour routing | Pollstar box office on a minority of rows; 61% of acts are Category E; capacity is the *venue's*, not the configuration on the night |
| `demographicdata.db` → `mbi_master` | catchment population, age structure, purchasing power | 7 countries only (DE GB FR IT ES PT CH); postcode-level |
| `demographicdata.db` → `postcode_locations` | postcode centroids, for radius catchments | — |

**Scope is therefore Western Europe**, set by the demographic coverage rather
than chosen. 3,684 cities in those countries have usable coordinates and at
least five shows since 2023.

### Why the two databases are joined on geography, not names

City names do not match across sources — Milan/Milano, Munich/München — and
fuzzy matching across languages introduces errors that are invisible
downstream. Instead each city's coordinate is matched against postcode
centroids within a radius. No name is ever compared.

Sanity check on the result: Bari 831k within 30km and 1.67M within 60km; Naples
3.5m; Milan 4.5m; Manchester 9.9m at 60km. These match published metro figures.

---

## Layout

```
venue_model/
  README.md          this
  catchment.py       demographics within a radius of a point
  extract.py         builds a versioned, self-contained extract
  gap.py             layer 1 — the descriptive picture
  app.py             the front end
  extracts/          output, gitignored
    2026-09-23/
      extract.db     one small SQLite file
      MANIFEST.json  row counts, date range, source versions, caveats
```

The app reads an **extract**, never the live database. That keeps it immune to
pipeline runs, fast, portable without shipping licensed data, and — most
importantly — reproducible.

---

## Running it

```bash
python extract.py                 # build a dated snapshot in extracts/
python screen.py                  # rank every market, writes reports/screen_<date>.xlsx
python gap.py --market Naples     # one market in depth, writes an Excel workbook
streamlit run app.py              # the Layer 1 explorer
```

The app reads only the extract, never the live database, so it is safe to run
while the pipeline or the venue enrichment is working.

### What the app is for

It exists to make the thresholds arguable. Every cut-off that decides a verdict
— the minimum tour length, the capacity-gap multiple, the borderline band — is
a sidebar control rather than a constant, and the table behind each verdict
shows the room the act plays elsewhere, the ceiling it was judged against and
the ratio between them. Anyone who thinks a finding is an artefact of a cut-off
can move the cut-off and watch it move.

### Catchment: the one thing to understand before reading any ranking

A 60 km radius around Guildford contains 13.8 million people, because London is
40 km away. Ranked on that, Guildford is the most under-served market in
Europe. It is not; it is a suburb of the best-served one.

So the ranking uses `exclusive_population_60km`: residents for whom this market
is the *nearest* one that actually hosts shows. Guildford keeps 855,000 of its
13.8 million, Hanley 916,000 of its 7.2 million, London 8.0 million of its 15.6,
and Bari all 1.67 million of its own — it has no competitor within 60 km. The
`catchment_kept` column shows the ratio, and below roughly 0.3 a market is a
satellite rather than a market.

This is a Voronoi allocation. It understates big cities slightly and overstates
small ones, because people do travel past a small market to reach a large one.
Both measures are kept side by side so either reading is available.

---

## Reading one venue rather than one market

`venue.py` and the app's **Venue** tab answer a different question: not "should
this city build a room" but "what is *this* room doing".

It has to start from an admission. **Layer 2 does not model venues.** Its
choice set is cities, and the only thing it knows about the rooms in one is the
capacity of the biggest of the kind an act plays. Nothing in the fitted model
distinguishes the Palapartenope from any other 6,541-seat hall. So the screen is
built in two halves that rest on very different foundations, and it keeps them
apart rather than averaging them into one confident-looking number.

### The solid half: market pull

Delete the venue, let the city's ceiling fall to whatever room is left, and ask
the model again. The difference is the venue's pull, and it needs no assumption
beyond the model itself.

It is zero for most venues, and that is not a criticism of them. A room only
has pull if it is the biggest of its kind in the market, because only then does
removing it change the ceiling any act is tested against. Milan's Lucid Club
pulls nothing; the tours it hosts were coming to Milan regardless.

Like every other model output here it is reported as the A-to-B interval.
Naples' Stadio Maradona comes out at **+0.9 to +18.0 tour-visits** — a very
wide bracket, which is the identification problem being honest rather than a
bug.

### The shaky half: which room an act lands in

This needs an extra assumption, stated rather than buried:

> **The nearest-capacity rule.** A tour plays the room whose capacity is
> closest, in log terms, to the room it uses elsewhere.

The obvious alternative — the smallest room that will *hold* the act — was
tried first and is much worse, because it assumes acts never scale down. They
do constantly: a tour averaging 9,881 elsewhere played Naples' 6,541-seat
Palapartenope. The floor rule calls that "nothing here fits".

`validate_rightsizing()` measures the rule per market against a deliberately
stupid baseline — always guess the busiest room:

| Market | rooms | nearest-capacity | baseline | verdict |
|---|---|---|---|---|
| Bari | 7 | 76% | 24% | good |
| Naples | 13 | 51% | 29% | usable |
| Rome | 27 | 22% | 22% | no better than guessing |
| Milan | 37 | 11% | 18% | **worse than guessing** |

The pattern is not subtle. Where a market has a handful of rooms the rule
works; where it has thirty-seven at overlapping sizes, nothing in this data
says which one an act picks. So the accuracy and the baseline are printed on
the screen beside the split, and in a market where the rule loses, the app says
so in as many words and tells you to read only the pull.

### What it can still say when the rule fails

Two judgements never depend on the assignment rule, because they are a
comparison of two numbers and a label: whether the act's usual room **fits**
inside this one, and whether the **kind matches**. Those stay reliable
everywhere, which is why the "tours it is not getting, and why" breakdown
separates *too big* and *wrong kind* from *another room here suits it better*.

---

## Renovating, and what it is worth in money

### Renovation is mostly the same question as building

The market model sees three things about a city's rooms: the biggest indoor
capacity, the biggest outdoor capacity, and whether either clears the bar for
the act in front of it. So of the things a renovation might do:

- **Raising a ceiling** — visible, but *identical arithmetic to a new build of
  that size*. `renovate.py` prints the two side by side and says so, rather
  than dressing one number up as two.
- **Putting a roof on** — visible and genuinely distinct. An outdoor room
  becomes playable indoors and the indoor ceiling rises without anything being
  built. This is the Lille Stade Pierre-Mauroy case, and the only renovation
  the project could not previously express.
- **Making the room better** — invisible. No coefficient knows about
  sightlines, acoustics or bars, and inventing one would be inventing a result.

Where a roof and a new build *do* differ is the ladder: a roof converts a rung,
a build adds one. That only surfaces in the Venue tab's room split, which is
reliable only in markets with few rooms.

### A refit shows up in the money, not the tour count

A venue's sell-through relative to rooms of its size is a persistent property —
**split-half reliability 0.864 across 3,642 venues**. A room that undersells its
size one year undersells it the next. So a refurbishment is modelled as moving
that residual toward a target percentile, and its effect lands entirely on gross
per show. It is kept on its own line and never added to the roof's effect: one
is a model output with an identification problem attached, the other is
arithmetic on observed sell-through.

It is called **sell-through performance**, never quality. A room's sell-through
reflects the building *and* which acts get booked into it *and* how well the
promoter matches act to room, and this cannot separate those.

### Gross box office, from 691,000 real events

`boxoffice.py --build` reads the main database once (read-only, safe during a
scrape) and caches a few kilobytes of coefficients:

```
log(gross) = 0.223 + 1.389 x log(capacity)      r = 0.872, n = 691,148
```

Two things in that worth noticing. The slope is **1.389, not 1.0** — doubling a
room's capacity multiplies gross by **2.62**, because bigger rooms also charge
more (median ticket USD 25.60 in a 1,500-seat room, USD 95.90 in a 30,000-seat
one). And the residual spread is wide: a single show lands within a factor of
4.9 of the median only 95% of the time, which is why nothing here is ever a
point estimate.

**This is gross box office, not venue revenue.** It is what the audience pays.
The venue takes a hire fee plus ancillaries, which needs a rate card this
database does not have — `ref_hospitality` has 32 rows. Converting the figure
is left to the reader, explicitly.

### The decomposition is the point

A revenue range spanning a factor of four is useless on its own. What is useful
is knowing *which ingredient produced the four*, because that says where the
next month of work should go. `boxoffice.decompose()` splits it:

| source | spans a factor of | share | can more data fix it? |
|---|---|---|---|
| how many tours the room gets | 3.9 | 56% | No — only a natural experiment (Layer 3) |
| what one show grosses | 3.0 | 44% | No, and it needs no fixing — real variation between acts |

The two are **comparable in size**, which was not what I expected: better
ticket-price data would barely narrow the answer, and neither would more
events. Only Layer 3 would.
