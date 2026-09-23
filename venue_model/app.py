#!/usr/bin/env python3
"""
venue_model — the front end.

WHAT THIS IS

The first screen answers the question the rest of the project exists to serve:
given a market's demographics and the rooms it has, is it getting more or fewer
touring acts than it should? Everything else follows from that.

  Chances             the probability a tour picks this city, what moves it,
                      and what a new room would do to it
  Tours vs expected   every market, actual against predicted, and why
  Market              one place in descriptive detail
  Venue               one room: what it pulls, and which tours it should and
                      should not expect
  Renovate            put a roof on an existing room, or refit it, and cost
                      both in gross box office
  Design a venue      put a room that does not exist into a market
  Build case          if a market is short of tours, what size room the
                      blocked acts actually use
  Method & limits     thresholds, the fitted model, what the data cannot say

FOUR RULES GOVERN THIS FILE

  1. NO NUMBER WITHOUT ITS ARITHMETIC. Wherever a figure is derived, the inputs
     are on the same screen, in a table, and a reader can recompute it by hand.

  2. THE LEVERS ARE EXPOSED. Every threshold that decides a verdict is a
     sidebar control. A threshold you cannot move is one you have to trust.

  3. WHAT THE DATA CANNOT SAY IS ON SCREEN, NOT IN A FOOTNOTE.

  4. COUNTING AND MODELLING ARE NEVER MIXED IN ONE FIGURE. Model output is
     labelled and reported as a range; counts are labelled as counts.

RUN IT

    streamlit run app.py
"""

import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import boxoffice as BO                                           # noqa: E402
import probability as PR                                         # noqa: E402
import recommend as R                                            # noqa: E402
import renovate as RN                                            # noqa: E402
import venue as V                                                # noqa: E402
import whatif                                                    # noqa: E402
from gap import (capacity_ladder, capacity_test,                 # noqa: E402
                 capacity_test_summary, ceilings_from_ladder, ladder_gaps,
                 load_extract, tours_that_skipped, where_skippers_went)
from screen import CATCHMENT_COL, PEER_RADIUS_KM, ceiling_caveat  # noqa: E402

st.set_page_config(page_title="venue_model", layout="wide",
                   initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="reading extract ...")
def get_extract(folder):
    return load_extract(folder)


@st.cache_data(show_spinner=False)
def get_screen(folder):
    """The whole-Europe screening results, if screen.py has been run."""
    reports = os.path.join(HERE, "reports")
    if not os.path.isdir(reports):
        return None, None
    stamp = os.path.basename(folder)
    path = os.path.join(reports, f"screen_{stamp}.xlsx")
    if not os.path.exists(path):
        hits = sorted(f for f in os.listdir(reports) if f.startswith("screen_"))
        if not hits:
            return None, None
        path = os.path.join(reports, hits[-1])
    return pd.read_excel(path), os.path.basename(path)


@st.cache_data(show_spinner="working out the case ...")
def get_recommendation(folder, city, code, min_dates, gap_mult, borderline_mult,
                       min_blocked):
    """Keyed on every threshold, so moving a slider really re-runs the test."""
    ex_ = get_extract(folder)
    mk = ex_["markets"]
    row = mk[(mk["city"] == city) & (mk["countryCode"] == code)].iloc[0]
    rec = R.recommend(ex_, row, min_dates, gap_mult, borderline_mult, min_blocked)
    rec["caveat"] = ceiling_caveat(rec["ladder"], rec["ceilings"].get("indoor"))
    raw = row.get(f"population_{PEER_RADIUS_KM}km")
    excl = row.get(CATCHMENT_COL)
    rec["kept"] = (excl / raw) if raw and excl and raw > 0 else None
    rec["row"] = row
    return rec


# The choice menu is the expensive object in the app: Great Britain is twenty
# seconds and a hundred megabytes. cache_resource returns the same object
# rather than a copy; max_entries keeps at most two countries resident.
@st.cache_resource(max_entries=2, show_spinner=False)
def get_menu(folder, code):
    ex_ = get_extract(folder)
    blob, err = whatif.fitted_model(os.path.basename(folder))
    if err:
        return None, err
    return whatif.build_menu(ex_, code, blob), None


@st.cache_data(show_spinner=False)
def get_model_card(folder):
    blob, err = whatif.fitted_model(os.path.basename(folder))
    return blob, err


# What a proposed room is allowed to be, by kind.
#
# One range cannot serve both. The largest indoor room in the data is Paris La
# Defense Arena at 45,000 and the 99th percentile is 20,000; outdoors the 95th
# percentile is 77,280 and Wembley is 92,034. A slider capped at 30,000 —
# which is what this had — simply could not express a stadium, so every
# outdoor proposal was silently truncated to an arena.
#
# The outdoor ceiling stops at 100,000 rather than the 262,737 of Rome's Tor
# Vergata papal site, because that is a field a million people once stood in
# and not a venue anyone is proposing.
CAPACITY_RANGE = {
    "indoor":  {"min": 1_000, "max": 45_000, "step": 500,  "default": 12_000},
    "outdoor": {"min": 2_000, "max": 100_000, "step": 1_000, "default": 30_000},
}


def capacity_slider(label, kind, key, suggested=None, help=None):
    """
    A capacity slider scaled to the kind of room being proposed.

    The kind is part of the widget key on purpose. Streamlit keeps a slider's
    value against its key, so switching from outdoor to indoor while holding a
    60,000 value would otherwise push it outside the new range. A fresh key per
    kind resets it to something sensible instead of erroring.
    """
    r = CAPACITY_RANGE.get(kind, CAPACITY_RANGE["indoor"])
    start = suggested if suggested and np.isfinite(suggested) else r["default"]
    start = int(min(max(start, r["min"]), r["max"]))
    # round to the step so the slider does not start on a value it cannot return
    start = int(round(start / r["step"]) * r["step"])
    return st.slider(label, r["min"], r["max"], start, r["step"],
                     key=f"{key}_{kind}", help=help)


def money(v, unit=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:,.0f}{unit}"


def explain(text):
    st.caption(text)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

extracts_dir = os.path.join(HERE, "extracts")
folders = sorted((os.path.join(extracts_dir, d) for d in os.listdir(extracts_dir)
                  if os.path.isdir(os.path.join(extracts_dir, d))), reverse=True) \
    if os.path.isdir(extracts_dir) else []
if not folders:
    st.error("No extract found. Run `python extract.py` first.")
    st.stop()

with st.sidebar:
    st.markdown("### Data")
    folder = st.selectbox("Extract", folders, format_func=os.path.basename)
    ex = get_extract(folder)
    markets = ex["markets"]
    st.caption(f"{len(markets):,} markets · snapshot {os.path.basename(folder)}")

    st.markdown("### Is it a market at all?")
    min_kept = st.slider(
        "Minimum share of the catchment it keeps", 0.0, 1.0, 0.3, 0.05,
        help="How much of the 60 km radius survives once neighbouring markets "
             "take their residents. Guildford keeps 6% — it is a satellite of "
             "London. Set to 0 to include them.")

    st.markdown("### Capacity thresholds")
    st.caption("Used by the Market and Build case screens.")
    min_dates = st.slider(
        "Minimum dates in the country to count as 'toured here'", 1, 10, 3,
        help="An act that played one date in the capital was never choosing "
             "between cities.")
    gap_mult = st.slider(
        "A tour needs a room this many times our ceiling to count as blocked",
        1.0, 2.0, 1.2, 0.05,
        help="At 1.2 an act must typically play rooms 20% larger than anything "
             "we have of that kind. A case that does not survive 1.4 is thin.")
    borderline_mult = st.slider(
        "Below this multiple, the evidence is borderline", 0.5, 1.0, 0.8, 0.05)
    min_blocked = st.slider(
        "Minimum blocked tours before a build case is worth making", 1, 40,
        R.MIN_BLOCKED_TOURS)

    st.markdown("### Simulation")
    draws = st.select_slider(
        "Coefficient draws", [100, 200, 500, 1000], value=200,
        help="How many times to redraw the coefficients from their covariance. "
             "More is smoother and slower.")

    st.markdown("### Display")
    show_arithmetic = st.checkbox("Show the arithmetic behind each verdict", True)


CASES = ["build", "too few", "no case", "satellite", "unknown"]

VERDICT_MEANING = {
    "capacity gap": "Plays rooms clearly bigger than anything we have of that "
                    "kind.",
    "capacity gap (none of this kind)": "We have no room of that kind at all.",
    "borderline": "Close to our ceiling. The evidence does not settle it.",
    "not capacity": "Plays rooms we can already match. Something else kept "
                    "them away.",
    "unknown": "No capacity recorded for the rooms this act used.",
}
VERDICT_COLOUR = {"capacity gap": "#c0392b",
                  "capacity gap (none of this kind)": "#e67e22",
                  "borderline": "#f1c40f",
                  "not capacity": "#7f8c8d",
                  "unknown": "#bdc3c7"}

PERF_COLOUR = {"far fewer than expected": "#c0392b",
               "fewer than expected": "#e67e22",
               "about as expected": "#95a5a6",
               "more than expected": "#3498db",
               "far more than expected": "#2471a3"}


def pick_market(key, default_city=None):
    names = (markets.assign(label=markets["city"] + ", " + markets["country"])
             .sort_values("events", ascending=False))
    labels = names["label"].tolist()
    idx = 0
    if default_city and default_city in set(names["city"]):
        idx = labels.index(names[names["city"] == default_city]["label"].iloc[0])
    label = st.selectbox("Market", labels, index=idx, key=key)
    return names[names["label"] == label].iloc[0]


def pick_country(key):
    """Country selector for the screens that need a fitted choice menu."""
    codes = sorted(set(ex["markets"]["countryCode"]) & set(whatif.choice.COVERED))
    name_of = (ex["markets"].drop_duplicates("countryCode")
               .set_index("countryCode")["country"].to_dict())
    default = codes.index("IT") if "IT" in codes else 0
    return st.selectbox("Country", codes, index=default, key=key,
                        format_func=lambda c: name_of.get(c, c))


# ---------------------------------------------------------------------------
# The counterfactual, said in words and then in tables
#
# The raw output is "+5.9 to +20.8", which is not an answer anyone can use: no
# time frame, no baseline, and "tour-visit" is a term this project invented. So
# the panel leads with a sentence, then the two tables that make it checkable.
# ---------------------------------------------------------------------------

def counterfactual_panel(menu, a, b, city, kind, capacity, compact=False):
    years = whatif.years_covered(ex)
    d = whatif.describe(a, b, years)

    st.markdown(f"### {d['headline']}")
    if d.get("sub"):
        st.markdown(d["sub"])
    st.caption(f"One touring act coming to {city} once. Two nights counts "
               f"once; only acts that played two or more cities in the country "
               f"are counted at all.")

    st.dataframe(pd.DataFrame([
        {"specification": "A — upper bound",
         "extra acts": round(a["extra_visits"], 1),
         "what it assumes": "Capacity keeps credit for every unobserved reason "
                            "a city is attractive"},
        {"specification": "B — lower bound",
         "extra acts": round(b["extra_visits"], 1),
         "what it assumes": "Shows already hosted take that credit instead, "
                            "including some the room really earns"},
    ]), width="stretch", hide_index=True,
        column_config={"what it assumes": st.column_config.TextColumn(
            "what it assumes", width="large")})
    explain("The truth is inside that range. Never the midpoint: the width is "
            "the measure of how unsettled the question is.")

    # --- the model's homework ---------------------------------------------
    ratio = (a["expected_before"] / a["actual_visits"]) if a["actual_visits"] else None
    if ratio and (ratio > 1.3 or ratio < 0.77):
        direction = "over-rates" if ratio > 1 else "under-rates"
        st.warning(
            f"**Check this first.** The model says {city} should already be "
            f"getting {a['expected_before']:.0f} touring acts. It got "
            f"{a['actual_visits']:,}. It {direction} {city} by a factor of "
            f"{max(ratio, 1 / ratio):.1f} on a question we can check, so the "
            f"range above is likely wrong in the same direction.")
    else:
        st.success(
            f"**Sanity check passed.** The model expects "
            f"{a['expected_before']:.0f} acts for {city} as it stands and "
            f"{a['actual_visits']:,} came, so it describes this market "
            f"reasonably before being asked to imagine anything.")

    mp = whatif.moving_parts(menu, "A", a)
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**What the building changes**")
        st.dataframe(mp["physical"], width="stretch", hide_index=True)
    with right:
        st.markdown("**What the model does with that**")
        st.dataframe(
            mp["response"], width="stretch", hide_index=True,
            column_config={
                "odds multiplier": st.column_config.NumberColumn(
                    "odds ×", format="%.2f"),
                "typical change": st.column_config.NumberColumn(
                    "typical change", format="%.2f"),
                "what it is": st.column_config.TextColumn(
                    "what it is", width="large")})
    explain(f"Nothing else moves: {', '.join(mp['unchanged'])} are identical "
            f"before and after. That is what makes the change attributable to "
            f"the building.")

    with st.expander("How the occasions break down"):
        st.dataframe(pd.DataFrame([
            {"occasions": menu["n_occasions"],
             "which ones": f"Every city a tour picked in this country "
                           f"({menu['n_tours']:,} tours)"},
            {"occasions": a["occasions"],
             "which ones": f"…on which {city} was still an option (a menu "
                           f"excludes cities the tour already played)"},
            {"occasions": a["occasions_helped"],
             "which ones": f"…and the act plays {kind} rooms"},
            {"occasions": int(round(a["occasions_helped"]
                                    * (a["cleared_after"] - a["cleared_before"]))),
             "which ones": "…and the new room clears a bar the old one did not"},
        ]), width="stretch", hide_index=True,
            column_config={"which ones": st.column_config.TextColumn(
                "which ones", width="large")})

    if compact:
        return

    # --- the size curve ----------------------------------------------------
    st.markdown("##### How big should it be?")
    sweeps = []
    for spec, lab in (("A", "A — upper bound"), ("B", "B — lower bound")):
        sw = whatif.sweep(menu, spec, city, kind)
        sw["specification"] = lab
        sweeps.append(sw)
    curve = pd.concat(sweeps, ignore_index=True)

    line = alt.Chart(curve).mark_line(point=True).encode(
        x=alt.X("capacity:Q", title="room size"),
        y=alt.Y("expected extra visits:Q", title="extra touring acts"),
        color=alt.Color("specification:N", title=None,
                        scale=alt.Scale(domain=["A — upper bound",
                                                "B — lower bound"],
                                        range=["#c0392b", "#2980b9"])),
        tooltip=["specification", "capacity", "expected extra visits",
                 "tours materially helped"],
    ).properties(height=320)
    here = alt.Chart(pd.DataFrame({"x": [capacity]})).mark_rule(
        color="#2c3e50", strokeDash=[5, 5]).encode(x="x:Q")
    st.altair_chart(line + here, width="stretch")
    explain(f"Dashed line: your {capacity:,} seats. The steps are the finding "
            f"— gains arrive as the room passes sizes touring acts actually "
            f"use, and the flat stretches are seats that buy nothing.")

    # --- simulation --------------------------------------------------------
    st.markdown("##### How much of that is estimation noise?")
    sim = whatif.simulate_impact(menu, city, capacity, kind, draws=draws)
    lo5, hi95 = np.percentile(sim, [5, 95])

    hist = alt.Chart(pd.DataFrame({"extra acts": sim})).mark_bar(
        opacity=0.85, color="#c0392b").encode(
        x=alt.X("extra acts:Q", bin=alt.Bin(maxbins=40),
                title="extra touring acts (specification A)"),
        y=alt.Y("count():Q", title="draws"))
    marks = alt.Chart(pd.DataFrame({
        "x": [a["extra_visits"], b["extra_visits"]],
        "estimate": ["A", "B"]})).mark_rule(
        strokeDash=[5, 5], size=2).encode(
        x="x:Q", color=alt.Color("estimate:N", title="point estimate",
                                 scale=alt.Scale(domain=["A", "B"],
                                                 range=["#2c3e50", "#2980b9"])))
    st.altair_chart((hist + marks).properties(height=260), width="stretch")

    st.dataframe(pd.DataFrame([
        {"source of uncertainty": "Coefficient estimation (simulated)",
         "range": f"{lo5:.1f} to {hi95:.1f}",
         "width": round(hi95 - lo5, 1),
         "what it covers": "How much the answer would move if the same model "
                           "were fitted to another sample of tours"},
        {"source of uncertainty": "Specification A against B",
         "range": f"{min(a['extra_visits'], b['extra_visits']):.1f} to "
                  f"{max(a['extra_visits'], b['extra_visits']):.1f}",
         "width": round(abs(a["extra_visits"] - b["extra_visits"]), 1),
         "what it covers": "Whether capacity causes shows or merely accompanies "
                           "them — the selection problem"},
    ]), width="stretch", hide_index=True,
        column_config={"what it covers": st.column_config.TextColumn(
            "what it covers", width="large")})
    explain(
        f"{draws:,} draws from the coefficients' covariance. **Compare the two "
        f"widths.** The simulated band is almost always the narrower, and that "
        f"is the point: a tight band means the sample is large, not that the "
        f"answer is settled. Nothing here covers the model being the wrong "
        f"shape.")

    with st.expander("The curve as a table"):
        st.dataframe(curve.round(2), width="stretch", hide_index=True)
    with st.expander("The tours whose odds move most"):
        mv = whatif.why(menu, "A", a)["movers"]
        mcols = [c for c in ["headliner", "category", "act_plays",
                             "room_needed", "ceiling", "p_before", "p_after",
                             "gain"] if c in mv.columns]
        st.dataframe(
            mv[mcols], width="stretch", hide_index=True,
            column_config={
                "room_needed": st.column_config.NumberColumn(
                    "room it uses", format="%d"),
                "ceiling": st.column_config.NumberColumn(
                    "ceiling before", format="%d"),
                "p_before": st.column_config.NumberColumn(
                    "chance before", format="%.3f"),
                "p_after": st.column_config.NumberColumn(
                    "chance after", format="%.3f"),
                "gain": st.column_config.NumberColumn("change", format="%.3f")})


# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

(tab_prob, tab_how, tab_perf, tab_market, tab_venue, tab_renovate, tab_design,
 tab_build, tab_method) = st.tabs(
    ["Chances", "How it works", "Tours vs expected", "Market", "Venue",
     "Renovate", "Design a venue", "Build case", "Method & limits"])

# One palette for the whole probability screen, so a colour means the same
# thing on every chart: blue is how things are, green is what a room adds,
# grey is context.
NOW, AFTER, MUTED = "#2980b9", "#27ae60", "#95a5a6"


# The box-office reference and the sell-through table, both cheap once cached.
@st.cache_data(show_spinner=False)
def get_boxoffice():
    return BO.load()


@st.cache_data(show_spinner=False)
def get_performance(folder):
    return BO.performance_table(get_extract(folder))



# ======================================================== CHANCES ========

with tab_prob:
    st.header("What are the chances a tour picks this city?")
    st.markdown(
        "Every other screen reports a **count**. This one reports the thing "
        "the model actually produces — a probability, for each tour, of "
        "choosing this city over every other city it could have played. "
        "Counts are those probabilities added up, and the addition hides the "
        "shape: twelve expected visits from fifteen near-certain acts is a "
        "different market from twelve out of four hundred long shots.")

    pc1, pc2, pc3 = st.columns([1, 2, 1])
    with pc1:
        p_code = pick_country("prob_country")
    p_markets = (ex["markets"][ex["markets"]["countryCode"] == p_code]
                 .sort_values("events", ascending=False))
    with pc2:
        p_city = st.selectbox("City", p_markets["city"].tolist(),
                              key="prob_market")
    with pc3:
        p_spec = st.radio(
            "Specification", ["A", "B"], horizontal=True, key="prob_spec",
            help="A lets capacity keep the credit for every unobserved reason "
                 "a city is attractive (upper bound). B hands as much of that "
                 "credit as possible to past activity (lower bound). The "
                 "truth is between them.")
    p_row = p_markets[p_markets["city"] == p_city].iloc[0]

    p_menu, p_err = get_menu(folder, p_code)
    if p_err:
        st.error(p_err)
    else:
        if p_menu.get("warning"):
            st.warning(p_menu["warning"])
        tours = PR.tours_for_city(p_menu, p_spec, p_city)
        years = whatif.years_covered(ex)
        # ACTUAL TOUR-VISITS, not the market's event count. p_row["events"]
        # counts every show, and a tour playing three nights is three of them
        # but only one visit -- so putting it beside an expected-visits figure
        # invites a comparison that is wrong by a factor of the average run
        # length. Milan read 1,381 against 64.9 before this was fixed.
        actual = int(p_menu["rows"].loc[
            whatif._target_mask(p_menu, p_city), "chosen"].sum())

        # ---- 1. the headline ------------------------------------------
        total_visits = float(tours["expected visits"].sum())
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tours in play", f"{int(tours['in play'].sum()):,}",
                  delta=f"of {len(tours):,} that toured {p_row['country']}",
                  delta_color="off")
        # Annualised, because 3.7 years is an artefact of the data window and
        # nobody plans a building against it. The window is complete and
        # roughly uniform -- 8.2k events in 2023, 8.3k in 2024, 9.3k in 2025
        # and 2026 running at a comparable monthly rate -- so the division is
        # sound rather than a guess.
        k2.metric("Expected tour-visits a year", f"{total_visits / years:.1f}",
                  delta=f"{total_visits:.0f} over {years:.1f} years",
                  delta_color="off")
        k3.metric("Best single chance", f"{tours['best occasion'].max():.0%}",
                  delta=tours.iloc[0]["headliner"], delta_color="off")
        # Headline per year, total as the subtitle -- the mirror of the
        # expected metric beside it, so the two are read off directly against
        # each other rather than one being a rate and the other a total.
        k4.metric("Tour-visits it actually got", f"{actual / years:.1f}",
                  delta=f"{actual:,} over {years:.1f} years",
                  delta_color="off")
        explain(
            f"A **tour-visit** is one tour choosing this city, however many "
            f"nights it then plays — so it is not the same as the "
            f"{int(p_row['events']):,} shows the city hosted. Expected and "
            f"actual above are both visits, and both annualised.")
        explain(
            "**Per-act figures below are not annualised** — they "
            "stay as a chance rather than a rate, because a stadium act tours "
            "every three years, so \"0.3 visits a year\" would read as a third "
            "of a show annually when it means one show every three years.")

        st.markdown("#### How those chances are made up")
        shape = PR.probability_shape(tours)
        # Per year, to match the headline metric. The share column is a ratio
        # so it is unaffected by the division.
        shape["a year"] = (shape["expected visits"] / years).round(2)
        order = shape["likelihood"].tolist()
        cshape = alt.Chart(shape).mark_bar().encode(
            x=alt.X("a year:Q", title="expected tour-visits a year"),
            y=alt.Y("likelihood:N", sort=order, title=None),
            color=alt.Color("likelihood:N", sort=order, legend=None,
                            scale=alt.Scale(range=["#1a5276", "#2980b9",
                                                   "#7fb3d5", "#bdc3c7",
                                                   "#ecf0f1"])),
            tooltip=[alt.Tooltip("likelihood:N"),
                     alt.Tooltip("tours:Q", title="tours"),
                     alt.Tooltip("a year:Q", title="visits a year",
                                 format=".2f"),
                     alt.Tooltip("expected visits:Q",
                                 title="over the whole period", format=".2f"),
                     alt.Tooltip("share of expected visits:Q", format=".1%")],
        ).properties(height=190)
        st.altair_chart(cshape, use_container_width=True)
        explain(
            "Read the top band first. A city whose expected visits come mostly "
            "from acts with a better-than-even chance has a settled touring "
            "market; one whose visits come from thousands of long shots is "
            "relying on luck, and a new room changes a lottery rather than a "
            "schedule.")

        with st.expander("Every tour, with its chance of coming here"):
            st.dataframe(
                tours[["headliner", "category", "act_plays", "room_needed",
                       "best occasion", "expected visits", "occasions"]].head(300),
                width="stretch", hide_index=True,
                column_config={
                    "room_needed": st.column_config.NumberColumn(
                        "room it plays elsewhere", format="%d"),
                    "best occasion": st.column_config.ProgressColumn(
                        "chance of this city", format="%.3f",
                        min_value=0.0, max_value=1.0),
                    "expected visits": st.column_config.NumberColumn(
                        "expected visits", format="%.3f",
                        help="Summed over the tour's choice occasions, so an "
                             "act touring the country twice can exceed 1.")})

        # ---- 2. what drives it ----------------------------------------
        st.divider()
        st.markdown("#### What is driving those chances")
        drv = PR.drivers(p_menu, p_spec, p_city)
        drv = drv[drv["utility"].abs() > 1e-9]
        cdrv = alt.Chart(drv).mark_bar().encode(
            x=alt.X("utility:Q", title="effect on log-odds against a typical rival city"),
            y=alt.Y("variable:N", sort="-x", title=None),
            color=alt.condition(alt.datum.utility > 0, alt.value(NOW),
                                alt.value("#c0392b")),
            tooltip=[alt.Tooltip("variable:N"),
                     alt.Tooltip("this market:Q", format=".3f"),
                     alt.Tooltip("average rival:Q", format=".3f"),
                     alt.Tooltip("coefficient:Q", format=".3f"),
                     alt.Tooltip("utility:Q", format=".3f"),
                     alt.Tooltip("odds x:Q", title="multiplies the odds by",
                                 format=".2f"),
                     alt.Tooltip("what it is:N")],
        ).properties(height=max(200, 34 * len(drv)))
        st.altair_chart(cdrv, use_container_width=True)
        explain(
            "Each bar is **(this city's value minus the average rival's) x the "
            "coefficient**. Hover for both values and the arithmetic. The bars "
            "add up in log-odds, which is why they do not add up in "
            "probability: a softmax has to take from somewhere, so one city "
            "can only gain what the others lose.")
        with st.expander("The same thing as a table"):
            st.dataframe(drv, width="stretch", hide_index=True,
                         column_config={"what it is":
                                        st.column_config.TextColumn(
                                            "what it is", width="large")})

        # ---- 3. what the existing rooms contribute --------------------
        st.divider()
        st.markdown("#### How much of that comes from the rooms it has")
        p_lad = V.rooms(ex, p_row)
        if p_lad.empty:
            st.info("No room here has a recorded capacity.")
        else:
            pulls = []
            for vn in p_lad["venue"].head(12):
                try:
                    pu = V.market_pull(p_menu, p_spec, ex, p_row, vn, p_lad)
                except (KeyError, ValueError):
                    continue
                cap = float(p_lad.loc[p_lad["venue"] == vn, "capacity"].iloc[0])
                pulls.append({"venue": vn, "capacity": cap,
                              "pull": round(pu["pull"], 3),
                              "binding": pu["binding"]})
            pull_df = pd.DataFrame(pulls)
            cpull = alt.Chart(pull_df).mark_bar().encode(
                x=alt.X("pull:Q", title="expected visits the city would lose "
                                        "without this room"),
                y=alt.Y("venue:N", sort="-x", title=None),
                color=alt.condition(alt.datum.pull > 0, alt.value(NOW),
                                    alt.value(MUTED)),
                tooltip=["venue", alt.Tooltip("capacity:Q", format=","),
                         alt.Tooltip("pull:Q", format=".3f"),
                         alt.Tooltip("binding:N",
                                     title="is it the ceiling?")],
            ).properties(height=max(180, 26 * len(pull_df)))
            st.altair_chart(cpull, use_container_width=True)
            explain(
                "Grey rooms pull nothing, and that is not a criticism of them. "
                "The model works through the **ceiling** — the biggest room of "
                "each kind — so removing anything smaller changes no verdict. "
                "The tours those rooms host were coming anyway and would have "
                "played elsewhere in the city.")

        # ---- 4. add a room --------------------------------------------
        st.divider()
        st.markdown("#### What a new room would do to those chances")
        a1, a2 = st.columns([3, 1])
        with a2:
            p_kind = st.radio("Kind", ["indoor", "outdoor"], key="prob_kind")
            cats = sorted(tours["category"].dropna().unique())
            p_cats = st.multiselect(
                "Artist tiers", cats, default=cats, key="prob_cats",
                help="Filters every figure in this section. Category A acts "
                     "are the ones a large room is built for, and looking at "
                     "them alone is usually the honest test of an arena case.")
            p_draws = st.slider("Simulation draws", 60, 400, 150, 20,
                                key="prob_draws",
                                help="Coefficient draws from the fitted "
                                     "covariance. More draws is a smoother "
                                     "band, not a better answer.")
        with a1:
            ceil_now = (p_lad.loc[p_lad["io"] == ("inside" if p_kind == "indoor"
                                                  else "outside"), "capacity"].max()
                        if not p_lad.empty else np.nan)
            p_cap = capacity_slider(
                "Capacity of the proposed room", p_kind, "prob_cap",
                suggested=(ceil_now * 1.5
                           if np.isfinite(ceil_now) else None),
                help="Indoor tops out at 45,000 (Paris La Defense Arena is the "
                     "largest in the data); outdoor reaches 100,000, which "
                     "covers Wembley at 92,034.")
            if np.isfinite(ceil_now):
                st.caption(f"The biggest {p_kind} room here today holds "
                           f"**{ceil_now:,.0f}**. A proposal below that moves "
                           f"nothing, because the ceiling is what acts are "
                           f"tested against.")

        move_all = PR.with_new_room(p_menu, p_spec, p_city, p_cap, p_kind)
        move = (move_all[move_all["category"].isin(p_cats)]
                if p_cats else move_all.head(0))
        gained = float(move["change"].sum())
        cleared = int(move["room clears its bar"].sum())
        if len(p_cats) < len(cats):
            st.caption(f"Showing **{', '.join(p_cats) or 'nothing'}** only — "
                       f"{len(move):,} of {len(move_all):,} tours.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected visits a year now",
                  f"{move['before'].sum() / years:.1f}")
        m2.metric("With the new room", f"{move['after'].sum() / years:.1f}",
                  delta=f"{gained / years:+.1f} a year")
        m3.metric("Over the whole period", f"{gained:+.1f}",
                  delta=f"{years:.1f} years", delta_color="off")
        m4.metric("Acts the room is big enough for", f"{cleared:,}",
                  delta=f"of {len(move):,}", delta_color="off")

        # A proposal at or below the existing ceiling moves nothing, and a
        # dumbbell chart of eighteen motionless rows is worse than saying so:
        # it invites a reader to hunt for a difference that is not there.
        if gained < 0.05:
            st.info(
                f"**A {p_cap:,}-capacity {p_kind} room changes nothing here.** "
                + (f"The city already has a {ceil_now:,.0f}-capacity {p_kind} "
                   f"room, and the model tests acts against the ceiling, so a "
                   f"smaller proposal offers them nothing they cannot already "
                   f"have. Raise the slider above {ceil_now:,.0f} to see "
                   f"movement."
                   if np.isfinite(ceil_now) and p_cap <= ceil_now else
                   "No act that toured the country is held back by room size "
                   "here."))
            movers = move.head(0).copy()
        else:
            movers = move[move["change"] > 1e-6].head(18).copy()
        movers["label"] = (movers["headliner"].astype(str).str.slice(0, 28)
                           + "  (" + movers["room_needed"].fillna(0)
                           .astype(int).astype(str) + ")")
        base = alt.Chart(movers if len(movers) else
                         pd.DataFrame({"label": [], "before": [], "after": [],
                                       "change": [], "headliner": [],
                                       "category": [], "room_needed": []})).encode(
            y=alt.Y("label:N", sort="-x", title=None))
        seg = base.mark_rule(strokeWidth=3, color=MUTED).encode(
            x=alt.X("before:Q",
                    title=f"expected visits from this act, over {years:.1f} years"),
            x2="after:Q")
        dot_b = base.mark_point(size=90, filled=True, color=NOW).encode(
            x=alt.X("before:Q"),
            tooltip=["headliner", "category",
                     alt.Tooltip("room_needed:Q", title="room it plays",
                                 format=","),
                     alt.Tooltip("before:Q", format=".3f"),
                     alt.Tooltip("after:Q", format=".3f"),
                     alt.Tooltip("change:Q", format="+.3f")])
        dot_a = base.mark_point(size=110, filled=True, color=AFTER).encode(
            x="after:Q",
            tooltip=["headliner", alt.Tooltip("after:Q", format=".3f"),
                     alt.Tooltip("change:Q", format="+.3f")])
        if len(movers):
            st.altair_chart((seg + dot_b + dot_a).properties(
                height=max(240, 26 * len(movers))), use_container_width=True)
        explain(
            f"**Per act this is a total over {years:.1f} years, not a rate** — "
            f"deliberately. A stadium act tours every three years, so an "
            f"annualised 0.3 would read as a third of a show a year when it "
            f"means one show every three. Blue is today, green is with a "
            f"{p_cap:,}-capacity {p_kind} room. "
            f"The acts that move are the ones whose usual room is bigger than "
            f"anything here now but fits inside the proposal — everyone else "
            f"sits still, because the model has nothing new to offer them.")

        # ---- the simulation -------------------------------------------
        st.markdown("##### How firm is that, given the coefficients are estimates?")
        try:
            before_s, after_s = PR.simulate_city(
                p_menu, p_city, p_spec, p_cap, p_kind, draws=p_draws)
            sims = pd.concat([
                pd.DataFrame({"a year": before_s / years, "scenario": "today"}),
                pd.DataFrame({"a year": after_s / years,
                              "scenario": f"with a {p_cap:,} {p_kind} room"})])
            hist = alt.Chart(sims).mark_area(opacity=0.55, interpolate="step").encode(
                x=alt.X("a year:Q", bin=alt.Bin(maxbins=40),
                        title="expected tour-visits a year"),
                y=alt.Y("count():Q", stack=None, title="draws"),
                color=alt.Color("scenario:N",
                                scale=alt.Scale(range=[NOW, AFTER]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=["scenario", alt.Tooltip("count():Q", title="draws")],
            ).properties(height=260).interactive()
            st.altair_chart(hist, use_container_width=True)

            d = (after_s - before_s) / years
            s1, s2, s3 = st.columns(3)
            s1.metric("Median gain", f"{np.median(d):+.1f} visits a year")
            s2.metric("5th to 95th percentile",
                      f"{np.percentile(d, 5):+.1f} to "
                      f"{np.percentile(d, 95):+.1f} a year")
            s3.metric("Draws where it gains nothing",
                      f"{(d <= 0).mean():.0%}")
            st.warning(
                "**This band is the narrowest uncertainty in the project, and "
                "the least important.** It covers only how far the answer "
                "would move if the same model were fitted to another sample of "
                "tours. It does **not** cover whether capacity causes shows or "
                "merely accompanies them — that is the gap between "
                "specifications A and B, and it is far wider. Switch the "
                "specification at the top and watch the whole chart move by "
                "more than this band's width.")
        except ValueError as e:
            st.info(str(e))

        # ---- what that is worth --------------------------------------
        st.divider()
        st.markdown("#### What those extra shows would gross")
        ref_bo2, bo_err2 = get_boxoffice()
        if bo_err2:
            st.info(bo_err2)
        elif gained < 0.05:
            st.info("Nothing to value: the proposed room adds no shows.")
        else:
            perf_tbl = get_performance(folder)
            # Use the room's own sell-through where the market has one that is
            # measurable, so a city whose rooms historically undersell is not
            # credited with average takings.
            pp = 0.0
            if not p_lad.empty:
                top_room = p_lad.iloc[0]["venue"]
                vp = BO.venue_performance(perf_tbl, top_room)
                if vp:
                    pp = vp["residual_pp"]
                    st.caption(
                        f"Using **{top_room}**'s sell-through as the local "
                        f"benchmark: {vp['residual_pp']:+.1f} points against "
                        f"par for its size ({vp['band']}).")

            # The A-to-B interval, not a point: the same bracket every other
            # model number in this app carries.
            other = "B" if p_spec == "A" else "A"
            move_other = PR.with_new_room(p_menu, other, p_city, p_cap, p_kind)
            move_other = (move_other[move_other["category"].isin(p_cats)]
                          if p_cats else move_other.head(0))
            g_other = float(move_other["change"].sum())
            lo_v, hi_v = min(gained, g_other), max(gained, g_other)

            rev = BO.project(ref_bo2, lo_v, hi_v, p_cap,
                             performance_pp=pp, years=years)
            r1, r2 = st.columns(2)
            r1.metric("Gross box office a year",
                      f"USD {rev['per_year_low']:,.0f} to "
                      f"{rev['per_year_high']:,.0f}")
            r2.metric(f"Over {years:.1f} years",
                      f"USD {rev['central_low']:,.0f} to "
                      f"{rev['central_high']:,.0f}")
            st.caption(
                f"Specification A gives {max(gained, g_other) if p_spec == 'A' else min(gained, g_other):+.1f} "
                f"extra visits and B gives the other end; the range above "
                f"spans both, at a median gross of "
                f"USD {rev['gross_per_event']['median']:,.0f} a show for a "
                f"{p_cap:,}-capacity room.")
            st.error(
                "**Gross box office, not venue revenue.** This is what the "
                "audience pays. The venue takes a hire fee plus a share of "
                "ancillaries, which needs a rate card this database does not "
                "have — `ref_hospitality` holds 32 rows. Converting this to "
                "what a building earns is a step you have to take yourself.")

            with st.expander("Where that range comes from"):
                st.dataframe(
                    BO.decompose(ref_bo2, lo_v, hi_v, p_cap,
                                 performance_pp=pp),
                    width="stretch", hide_index=True,
                    column_config={
                        "can more data fix it?": st.column_config.TextColumn(
                            "can more data fix it?", width="large"),
                        "share of the uncertainty":
                            st.column_config.ProgressColumn(
                                "share of the uncertainty", format="%.2f",
                                min_value=0.0, max_value=1.0)})
                explain(
                    "The two sources are usually comparable in size, which is "
                    "the useful finding: better ticket-price data would not "
                    "narrow this much, and neither would more events. Only "
                    "Layer 3 — a natural experiment around real venue "
                    "openings — would.")

        # ---- 5. one tour ----------------------------------------------
        st.divider()
        st.markdown("#### One act at a time")
        st.markdown(
            "The same arithmetic for a single tour: its chance of this city, "
            "where it is more likely to go instead, and what the proposed room "
            "would change.")

        in_play = tours[tours["in play"]].copy()
        if p_cats:
            in_play = in_play[in_play["category"].isin(p_cats)]
        in_play["label"] = (in_play["headliner"].astype(str) + "  —  "
                            + in_play["category"].astype(str) + "  —  "
                            + in_play["best occasion"].map("{:.1%}".format))
        if in_play.empty:
            st.info("No tour of the selected tiers is in play here.")
        else:
            pick = st.selectbox("Tour", in_play["label"].tolist(),
                                key="prob_tour")
            t_row = in_play[in_play["label"] == pick].iloc[0]
            t_name = t_row["tour"]
            det = PR.tour_detail(p_menu, t_name)
            chg = PR.tour_with_new_room(p_menu, p_spec, p_city, t_name,
                                        p_cap, p_kind)

            st.markdown(f"##### {det['headliner']} — *{t_name}*")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Chance of this city now",
                      f"{t_row['best occasion']:.1%}")
            t2.metric("Room it plays elsewhere", money(det["room_needed"]),
                      delta=f"{det['plays']} act", delta_color="off")
            if chg:
                t3.metric(f"With a {p_cap:,} {p_kind} room",
                          f"{chg['best_after']:.1%}",
                          delta=f"{chg['best_after'] - chg['best_before']:+.1%}")
                t4.metric("Does the proposal fit it?",
                          "yes" if chg["clears"] else "no",
                          delta=(f"{p_cap:,} clears its {det['room_needed']:,.0f}"
                                 if chg["clears"] and det["room_needed"]
                                 else "still too small for this act"),
                          delta_color="off")

            # ---- where the city now sits in this act's running order ----
            ranked = PR.cities_for_tour_change(
                p_menu, p_spec, p_city, t_name, p_cap, p_kind, limit=16)
            sentence = PR.rank_sentence(ranked, p_city)
            if sentence:
                st.markdown(sentence)

            rl = ranked.melt(
                id_vars=["market", "played", "rank before", "rank after"],
                value_vars=["chance_before", "chance_after"],
                var_name="when", value_name="chance")
            rl["when"] = rl["when"].map(
                {"chance_before": "now",
                 "chance_after": f"with a {p_cap:,} {p_kind} room"})
            rl["this city"] = rl["market"] == p_city
            # Whether the act actually PLAYED a city is the most important
            # thing on this chart and it was previously only in the tooltip.
            # It gets three encodings now -- a black outline, a label, and the
            # sort order -- because a reader should not have to hover to find
            # out which bars are history and which are counterfactual.
            order = ranked["market"].tolist()
            rl["marker"] = np.where(rl["played"], "played", "")
            bars = alt.Chart(rl).mark_bar(cornerRadiusEnd=2).encode(
                x=alt.X("chance:Q", title="chance this act plays the city",
                        axis=alt.Axis(format="%")),
                y=alt.Y("market:N", sort=order, title=None),
                yOffset="when:N",
                color=alt.Color("when:N", scale=alt.Scale(range=[NOW, AFTER]),
                                legend=alt.Legend(title=None, orient="top")),
                stroke=alt.condition(alt.datum.played, alt.value("#2c3e50"),
                                     alt.value(None)),
                strokeWidth=alt.condition(alt.datum.played, alt.value(1.4),
                                          alt.value(0)),
                opacity=alt.condition(alt.datum["this city"], alt.value(1.0),
                                      alt.value(0.5)),
                tooltip=["market", "when",
                         alt.Tooltip("chance:Q", format=".2%"),
                         alt.Tooltip("rank before:Q", title="rank now"),
                         alt.Tooltip("rank after:Q", title="rank after"),
                         alt.Tooltip("played:N", title="act played here")],
            )
            labels = alt.Chart(
                rl[rl["when"] == "now"]).mark_text(
                    align="left", dx=5, fontSize=11, color="#2c3e50",
                    fontWeight="bold").encode(
                x=alt.X("chance:Q"),
                y=alt.Y("market:N", sort=order),
                text="marker:N")
            crank = (bars + labels).properties(
                height=max(260, 34 * len(ranked)))
            st.altair_chart(crank, use_container_width=True)
            st.caption(
                "Bars with a dark outline and a **played** label are cities "
                "this act really visited. Everything without one is a city it "
                "passed over — those are the counterfactual numbers, and the "
                "ones worth arguing about.")
            explain(
                f"Solid bars are **{p_city}**. Every other city dips slightly "
                f"when it gains, because the alternatives share one "
                f"denominator — the probability has to come from somewhere, "
                f"and that is the competitive effect made visible. Cities "
                f"marked `played` were measured on the single occasion they "
                f"won, so their figure is conditioned on winning and reads "
                f"high; the directly comparable numbers are the ones the act "
                f"passed over.")

            # ---- what the act actually did ------------------------------
            st.markdown("##### What this tour actually played")
            itin = PR.tour_itinerary(ex, t_name, p_code)
            if itin.empty:
                st.info("No itinerary recorded for this tour in this country.")
            else:
                been = set(itin["market"].dropna())
                st.markdown(
                    f"**{len(itin)} stops** in {p_row['country']}"
                    + (f", and **{p_city} was not one of them**."
                       if p_city not in been else
                       f", including **{p_city}**."))
                st.dataframe(
                    itin, width="stretch", hide_index=True,
                    column_config={
                        "first_event": st.column_config.TextColumn("date"),
                        "largest_capacity_played":
                            st.column_config.NumberColumn(
                                "room used", format="%d"),
                        "events": st.column_config.NumberColumn(
                            "dates", format="%d")})
                explain(
                    "Fact, not model output — this is where the act went. Read "
                    "it against the chart above: a city with a high chance and "
                    "no stop is the case worth explaining, and the model "
                    "cannot tell you which of routing, guarantees or a "
                    "promoter relationship accounts for it.")


# ====================================================== HOW IT WORKS ======

with tab_how:
    st.header("What this model is actually doing")
    st.markdown("""
The whole project answers one question — **would a new room bring more shows to
this city** — and it answers it in layers, because the question is causal and
the data is observational. Each layer is more ambitious and less certain than
the one below, and they are kept apart so you can see how far out on the limb
any given number sits.
""")

    st.dataframe(pd.DataFrame([
        {"layer": "1. Counting",
         "what it does": "Capacity ladder, tours that skipped, whether room "
                         "size was the obstacle, peer comparison.",
         "what it licenses": "\"Here is what happens today.\"",
         "where it is": "Market, Build case, Venue"},
        {"layer": "2. Choice model",
         "what it does": "A conditional logit over which cities tours picked.",
         "what it licenses": "\"This associates with that, holding the act "
                             "constant.\"",
         "where it is": "Chances, Tours vs expected, Design a venue"},
        {"layer": "3. Causal",
         "what it does": "Difference-in-differences around real venue "
                         "openings. NOT BUILT.",
         "what it licenses": "\"This caused that.\"",
         "where it is": "nowhere yet"},
        {"layer": "4. Simulation",
         "what it does": "Redrawing the coefficients to get a spread.",
         "what it licenses": "\"Here is the distribution.\"",
         "where it is": "the band on Chances"},
    ]), width="stretch", hide_index=True,
        column_config={"what it does": st.column_config.TextColumn(
            "what it does", width="large")})

    st.divider()
    st.markdown("""
#### The model in one line

For each tour, and each city it could have played:

$$P(\\text{this city}) = \\frac{e^{\\,x'\\beta}}{\\sum_{\\text{all cities on the menu}} e^{\\,x'\\beta}}$$

That is McFadden's conditional logit. `x` is the city's attributes, `β` the
fitted coefficients. The denominator runs over every city that tour could have
chosen, which is what makes this a **choice** model rather than a forecast of
each city separately: the probabilities on one menu must add to one, so a city
can only gain by taking from the others.

#### Why that form, and what it buys for free

**Everything about the act cancels.** An artist's popularity, budget, genre and
fanbase are the same for every city on their menu, so they vanish from the
ratio. That sounds like a loss and is the model's main virtue: the coefficients
are identified purely from **within-tour** variation. The question is not "which
tours play big cities" but *given that this tour played four cities in Italy,
why those four* — and every confounder attached to the artist has been
differenced away without having to measure it.

To let an artist trait matter it has to be **interacted** with a city trait.
That is why `room big enough × log room needed` is in the model: it asks whether
capacity binds harder for bigger acts. It does.

#### What goes in

""")
    card, card_err = get_model_card(folder)
    if card_err:
        st.info(card_err)
    elif card:
        notes = card.get("variable_notes", {})
        specA = card["specs"]["A"]
        specB = card["specs"]["B"]
        tbl = pd.DataFrame({
            "variable": specA["names"],
            "A": np.round(specA["beta"], 3),
            "B": [dict(zip(specB["names"], np.round(specB["beta"], 3)))
                  .get(n, np.nan) for n in specA["names"]],
            "what it means": [notes.get(n, "") for n in specA["names"]],
        })
        st.dataframe(tbl, width="stretch", hide_index=True,
                     column_config={"what it means":
                                    st.column_config.TextColumn(
                                        "what it means", width="large")})
        st.caption(
            f"Fitted on {card['n_rows']:,} rows — {card['n_occasions']:,} "
            f"choice occasions from {card['n_tours']:,} tours across "
            f"{', '.join(card['countries'])}. Standard errors clustered by "
            f"tour, because one tour contributes many occasions and they are "
            f"obviously not independent.")

    st.divider()
    st.markdown("""
#### The problem at the centre of it, stated plainly

**The question is causal and the data is observational.**

Cities with big arenas get big shows. But cities that *build* big arenas are
cities where promoters already expected demand. The arena did not cause the
shows; anticipated demand caused both. Regress shows on capacity across cities
and you get a large, clean, highly significant coefficient that is **mostly
selection**.

Nothing in this data fixes that. So instead of pretending otherwise, two
specifications are fitted and **the answer is the interval between them**:

- **A** — catchment, income, capacity, spread. Capacity keeps the credit for
  every unobserved reason a city is attractive. Its coefficient is an
  **upper bound**.
- **B** — the same, plus how many shows the city already hosts. Past activity
  is the best available proxy for those unobserved reasons, but it is also
  partly *the thing capacity delivers*, so it strips out some of the real
  effect too. A **lower bound**.

The truth is inside. This data cannot say where. If the two bounds agree, a
finding is robust to the worry; if they are far apart, no further modelling of
the same data will settle it.

**The one result that survives the harshest test:** `has a room big enough`
holds up under specification B almost intact, while plain `log largest room`
collapses to nothing. General size is explained away by past activity — which
is what you would expect if it were mostly selection. Whether the room clears
*this particular act's* bar is not.

#### What would actually narrow it

Layer 3: difference-in-differences around real venue openings. A venue that
opened in 2018 gives a before and an after for the same city, which is the only
thing in reach that separates the room causing shows from the room accompanying
them. It does not exist yet, and it is the single most valuable thing left to
build.

#### Things the model cannot see

- Everything about a building except its size and whether it has a roof.
  Sightlines, loading bays, rail links, the promoter and the rent are invisible.
  A better room of the same capacity scores identically.
- A second room the same size as the existing one. The model works through the
  **ceiling**, so it registers as nothing. Date congestion is real and unseen.
- Which room inside a city an act picks. That needs the extra assumption on the
  Venue tab, and it is only reliable where a market has few rooms.
- Money, until Pollstar is brought in — and even then it is **gross box
  office**, not what a venue earns.
""")


# ================================================== TOURS VS EXPECTED =====

with tab_perf:
    st.header("Is this place getting the tours its size and rooms suggest?")
    st.markdown(
        "The model predicts how often each city gets picked from its "
        "**catchment**, its **spending power**, the **size of its biggest "
        "room**, whether that room is **big enough for the act**, and where it "
        "sits relative to the tour's other stops. Compare that with what "
        "actually happened.")

    blob, model_err = get_model_card(folder)
    if model_err:
        st.warning(model_err)
    else:
        pc1, _ = st.columns([1, 3])
        with pc1:
            code = pick_country("perf_country")
        with st.spinner("building this country's choice menu — once per "
                        "country, then everything is instant"):
            menu, err = get_menu(folder, code)
        if err:
            st.warning(err)
        else:
            perf = whatif.performance(menu)
            attrs = ex["markets"]
            attrs = attrs[attrs["countryCode"] == code].assign(
                kept=lambda d: d[CATCHMENT_COL]
                / d[f"population_{PEER_RADIUS_KM}km"])
            perf = perf.merge(attrs[["city", "kept"]], left_on="market",
                              right_on="city", how="left").drop(columns="city")

            view = perf.copy()
            if min_kept > 0:
                view = view[view["kept"].fillna(1) >= min_kept]

            counts = view["verdict"].value_counts()
            cols = st.columns(5)
            for col, v in zip(cols, whatif.VERDICT_ORDER):
                col.metric(v.replace(" than expected", ""),
                           f"{int(counts.get(v, 0)):,}")
            explain(
                f"{len(view):,} markets on this country's menu"
                + (f", after dropping {len(perf) - len(view)} that keep less "
                   f"than {min_kept:.0%} of their catchment and are better "
                   f"understood as satellites of a bigger market."
                   if len(view) < len(perf) else "."))

            st.dataframe(
                view[["market", "verdict", "actual", "expected", "gap",
                      "actual vs expected", "catchment", "indoor_ceiling",
                      "kept"]].round({"expected": 1, "gap": 1,
                                      "actual vs expected": 2}),
                width="stretch", hide_index=True,
                column_config={
                    "actual": st.column_config.NumberColumn(
                        "acts it got", format="%d"),
                    "expected": st.column_config.NumberColumn(
                        "acts predicted", format="%.1f"),
                    "gap": st.column_config.NumberColumn(
                        "shortfall / surplus", format="%.1f"),
                    "actual vs expected": st.column_config.NumberColumn(
                        "got ÷ predicted", format="%.2f",
                        help="Below 1 means fewer tours than its demographics "
                             "and rooms predict."),
                    "catchment": st.column_config.NumberColumn(
                        "catchment", format="%d"),
                    "indoor_ceiling": st.column_config.NumberColumn(
                        "biggest indoor room", format="%d"),
                    "kept": st.column_config.ProgressColumn(
                        "kept", format="%.2f", min_value=0.0, max_value=1.0)})

            # --- the picture -----------------------------------------------
            st.markdown("#### Predicted against actual")
            plot = view[(view["expected"] > 0) & (view["actual"] > 0)].copy()
            if not plot.empty:
                pts = alt.Chart(plot).mark_circle(opacity=0.8).encode(
                    x=alt.X("expected:Q", scale=alt.Scale(type="log"),
                            title="acts predicted from demographics and rooms"),
                    y=alt.Y("actual:Q", scale=alt.Scale(type="log"),
                            title="acts it actually got"),
                    color=alt.Color(
                        "verdict:N", title=None,
                        scale=alt.Scale(domain=whatif.VERDICT_ORDER,
                                        range=[PERF_COLOUR[v] for v in
                                               whatif.VERDICT_ORDER])),
                    size=alt.Size("catchment:Q", title="catchment",
                                  scale=alt.Scale(range=[40, 600])),
                    tooltip=["market", "actual", "expected",
                             "actual vs expected", "catchment",
                             "indoor_ceiling"],
                ).properties(height=420)
                lim = [float(min(plot["expected"].min(), plot["actual"].min())),
                       float(max(plot["expected"].max(), plot["actual"].max()))]
                parity = alt.Chart(pd.DataFrame({"x": lim, "y": lim})).mark_line(
                    color="#7f8c8d", strokeDash=[5, 5]).encode(x="x:Q", y="y:Q")
                st.altair_chart(pts + parity, width="stretch")
                explain("Dashed line is parity. **Below it: fewer tours than "
                        "predicted.** Above it: more. Both axes are log, so "
                        "equal vertical distances are equal ratios.")

            # --- one market --------------------------------------------------
            st.divider()
            listing = view["market"].tolist()
            chosen = st.selectbox("Look at one market", listing, index=0,
                                  key="perf_one")
            row_p = perf[perf["market"] == chosen].iloc[0]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Touring acts it got", f"{int(row_p['actual']):,}")
            m2.metric("Predicted", f"{row_p['expected']:.0f}")
            m3.metric("Got ÷ predicted", f"{row_p['actual vs expected']:.2f}",
                      delta=row_p["verdict"], delta_color="off")
            m4.metric("Biggest indoor room", money(row_p["indoor_ceiling"]))

            if row_p["actual vs expected"] < 0.87:
                st.warning(
                    f"**{chosen} gets {row_p['actual vs expected']:.0%} of the "
                    f"acts its fundamentals predict.** That means one of two "
                    f"things and this data cannot separate them: it is "
                    f"genuinely under-served, or the model is missing "
                    f"something about it — no local promoter, a hostile "
                    f"calendar, a border, an audience that does not buy this "
                    f"kind of ticket. A residual is a question for someone who "
                    f"knows the market, not a finding.")
            elif row_p["actual vs expected"] > 1.15:
                st.info(
                    f"**{chosen} gets {row_p['actual vs expected']:.1f}× the "
                    f"acts its fundamentals predict.** Usually something the "
                    f"model cannot see is pulling tours in — most often a "
                    f"festival, which books a field rather than a room and "
                    f"does not need the catchment to be local.")

            st.markdown(f"#### Why the model expects {row_p['expected']:.0f}")
            st.dataframe(
                whatif.why_expected(menu, chosen), width="stretch",
                hide_index=True,
                column_config={
                    "this market": st.column_config.NumberColumn(
                        "this market", format="%.2f"),
                    "average rival": st.column_config.NumberColumn(
                        "average rival", format="%.2f"),
                    "odds x": st.column_config.NumberColumn(
                        "odds ×", format="%.2f",
                        help="How much this one variable multiplies this "
                             "market's odds of being picked, against a typical "
                             "rival city on the same menus."),
                    "what it is": st.column_config.TextColumn(
                        "what it is", width="large")})
            explain(
                "Each row compares this market against the average alternative "
                "on the menus it appeared on. Variables are on their modelled "
                "scale (logs, centred), which is why the raw values look odd — "
                "the readable column is **odds ×**. Utility adds up exactly; "
                "odds do not, because probability is a softmax.")

            # --- is the gap bigger than the noise? -------------------------
            st.markdown("#### Is that gap bigger than the model's own "
                        "uncertainty?")
            if st.button("Simulate", key="perf_sim", type="primary"):
                st.session_state["perf_sim_on"] = True
            if st.session_state.get("perf_sim_on"):
                with st.spinner(f"drawing {draws:,} sets of coefficients"):
                    band = whatif.expected_bands(menu, draws=draws)
                band = band.merge(perf[["market", "kept"]], on="market",
                                  how="left")
                if min_kept > 0:
                    band = band[band["kept"].fillna(1) >= min_kept]

                outside = int((band["outside the band"] != "inside").sum())
                st.markdown(
                    f"**{outside:,} of {len(band):,} markets fall outside the "
                    f"band.** That is the expected result and it is the "
                    f"important one: the coefficients come from 1.15 million "
                    f"rows, so their own wobble is tiny next to how far real "
                    f"markets sit from the model.")
                st.dataframe(
                    band[["market", "actual", "expected", "p5", "p95",
                          "actual vs expected", "outside the band"]]
                    .round({"actual vs expected": 2}),
                    width="stretch", hide_index=True,
                    column_config={"outside the band":
                                   st.column_config.TextColumn(
                                       "outside the band", width="medium")})
                explain(
                    "p5 and p95 are the 5th and 95th percentiles of the "
                    "predicted count across the draws. **Estimation** "
                    "uncertainty only — not the selection problem, and not the "
                    "model being the wrong shape. A narrow band is not "
                    "confidence.")

            # --- the counting cross-check -----------------------------------
            screen, _sf = get_screen(folder)
            if screen is not None:
                with st.expander("The same question answered by counting, with "
                                 "no model"):
                    cols_sc = [c for c in ["market", "shows_per_million",
                                           "peer_median_spm", "vs_peers",
                                           CATCHMENT_COL, "indoor_ceiling",
                                           "peer_median_ceiling",
                                           "ceiling_below_peers"]
                               if c in screen.columns]
                    sc = screen[screen["countryCode"] == code][cols_sc]
                    st.dataframe(sc.sort_values("vs_peers"), width="stretch",
                                 hide_index=True,
                                 column_config={
                                     "vs_peers": st.column_config.NumberColumn(
                                         "shows vs peers", format="%.2f")})
                    explain(
                        "Shows per million people here, divided by the median "
                        "for markets of similar catchment. Below 1 means fewer "
                        "shows per head than comparable places. It knows "
                        "nothing about routing or room size, so where it and "
                        "the model disagree, the disagreement is informative.")


# =========================================================== MARKET =======

with tab_market:
    row = pick_market("detail_market")

    ladder = capacity_ladder(ex, row)
    ceilings = ceilings_from_ladder(ladder)
    caveat = ceiling_caveat(ladder, ceilings.get("indoor"))

    st.subheader(f"{row['city']}, {row['country']}")
    c1, c2, c3, c4, c5 = st.columns(5)
    raw = row.get(f"population_{PEER_RADIUS_KM}km")
    exclusive = row.get(CATCHMENT_COL)
    kept = (exclusive / raw) if raw and exclusive and raw > 0 else None
    c1.metric("Shows since 2023", money(row.get("events")))
    c2.metric(f"People within {PEER_RADIUS_KM} km", money(raw))
    c3.metric("…for whom this is the nearest market", money(exclusive),
              delta=f"{kept:.0%} kept" if kept is not None else None,
              delta_color="off")
    c4.metric("Biggest indoor room", money(ceilings.get("indoor")))
    c5.metric("Biggest outdoor room", money(ceilings.get("outdoor")))
    explain("The third figure is the one the analysis uses: a plain radius "
            "credits a town beside a big city with that city's population.")
    if kept is not None and kept < 0.3:
        st.warning(f"**Keeps only {kept:.0%} of its radius.** Most people "
                   f"within {PEER_RADIUS_KM} km are closer to a bigger market, "
                   f"so treat any under-served finding here with suspicion.")
    if caveat:
        st.warning(f"**Ceiling caveat** — {caveat}. If that venue is playable "
                   f"indoors, the indoor ceiling above is wrong and every "
                   f"verdict below inherits the error.")

    st.divider()
    st.markdown("#### What rooms exist here")
    lad = ladder.head(25)
    if not lad.empty:
        st.altair_chart(alt.Chart(lad).mark_bar().encode(
            x=alt.X("capacity:Q", title="capacity"),
            y=alt.Y("venue:N", sort="-x", title=None),
            color=alt.Color("io:N", title="indoor / outdoor",
                            scale=alt.Scale(domain=["inside", "outside"],
                                            range=["#2980b9", "#27ae60"])),
            tooltip=["venue", "capacity", "io", "io_source", "events",
                     "headliners"],
        ).properties(height=min(520, 26 * len(lad))), width="stretch")
    unl = int(ceilings.get("unlabelled_venues") or 0)
    if unl:
        biggest = ceilings.get("largest_unlabelled")
        if biggest is not None and not pd.isna(biggest):
            explain(f"{unl} venues have no indoor/outdoor label, the largest "
                    f"holding {money(biggest)}. If that room is indoors the "
                    f"ceiling above is too low.")
        else:
            explain(f"{unl} venues have no indoor/outdoor label, but none has "
                    f"a recorded capacity, so none could have set a ceiling.")
    with st.expander("The ladder as a table, and the gaps in it"):
        st.dataframe(ladder, width="stretch", hide_index=True)
        st.markdown("**Missing rungs** — capacity bands with no venue:")
        st.dataframe(ladder_gaps(ladder), width="stretch", hide_index=True)

    st.divider()
    st.markdown("#### Was room size the obstacle for the tours that skipped?")
    skipped = tours_that_skipped(ex, row, min_dates)
    if skipped.empty:
        st.info(f"No tour with {min_dates}+ dates in {row['country']} skipped "
                f"this market.")
    else:
        tested = capacity_test(skipped, row, ceilings, gap_mult, borderline_mult)
        summary = capacity_test_summary(tested)
        st.markdown(f"**{len(tested):,} tours** played {min_dates}+ dates in "
                    f"{row['country']} without coming here. Only the first bar "
                    f"is evidence a building would have changed that.")

        order = ["capacity gap", "capacity gap (none of this kind)",
                 "borderline", "not capacity", "unknown"]
        summary["verdict"] = pd.Categorical(summary["verdict"], order,
                                            ordered=True)
        summary = summary.sort_values("verdict")
        st.altair_chart(alt.Chart(summary).mark_bar().encode(
            x=alt.X("tours:Q", title="tours that skipped"),
            y=alt.Y("verdict:N", sort=order, title=None),
            color=alt.Color("verdict:N", legend=None,
                            scale=alt.Scale(domain=list(VERDICT_COLOUR),
                                            range=list(VERDICT_COLOUR.values()))),
            tooltip=["verdict", "tours", "country_dates", "median_room_needed",
                     "our_ceiling"],
        ).properties(height=170), width="stretch")

        st.dataframe(
            summary.assign(meaning=summary["verdict"].astype(str).map(
                VERDICT_MEANING))[["verdict", "tours",
                                   "share of skipping tours", "meaning"]],
            width="stretch", hide_index=True,
            column_config={
                "share of skipping tours": st.column_config.NumberColumn(
                    "share", format="%.2f"),
                "meaning": st.column_config.TextColumn("meaning",
                                                       width="large")})

        if show_arithmetic:
            st.markdown("##### Every verdict, and the sum behind it")
            explain(f"`room_it_needs` ÷ `our_ceiling_of_that_kind` = `ratio`. "
                    f"Above {gap_mult:g} is a gap, above {borderline_mult:g} "
                    f"is borderline, below is not capacity.")
            cols = [c for c in ["headliner", "category", "plays",
                                "country_dates", "room_it_needs",
                                "our_ceiling_of_that_kind", "ratio", "verdict",
                                "where"] if c in tested.columns]
            st.dataframe(
                tested[cols].sort_values("ratio", ascending=False),
                width="stretch", hide_index=True,
                column_config={
                    "ratio": st.column_config.NumberColumn("ratio",
                                                           format="%.2f"),
                    "where": st.column_config.TextColumn("played instead",
                                                         width="medium")})

        st.markdown("#### Where those tours went instead")
        st.dataframe(where_skippers_went(ex, row, skipped).head(20),
                     width="stretch", hide_index=True,
                     column_config={"share of the skipping tours":
                                    st.column_config.ProgressColumn(
                                        "share", format="%.2f",
                                        min_value=0.0, max_value=1.0)})


# ============================================================ VENUE =======

with tab_venue:
    st.header("One room: what it pulls, and what it should expect")
    st.markdown(
        "The model chooses **cities**, not rooms. So this screen is in two "
        "halves, and they are not equally trustworthy. **What the room pulls** "
        "is pure model arithmetic: delete the venue, let the city's ceiling "
        "fall to whatever is left, and ask again. **Which tours land in this "
        "room rather than a neighbouring one** needs an extra assumption, and "
        "how well that assumption holds is measured and shown below rather "
        "than asserted.")

    vcol1, vcol2 = st.columns([1, 2])
    with vcol1:
        v_code = pick_country("venue_country")
    v_markets = (ex["markets"][ex["markets"]["countryCode"] == v_code]
                 .sort_values("events", ascending=False))
    with vcol2:
        v_city = st.selectbox("Market", v_markets["city"].tolist(),
                              key="venue_market")
    v_row = v_markets[v_markets["city"] == v_city].iloc[0]

    v_lad = V.rooms(ex, v_row)
    if v_lad.empty:
        st.info(f"No venue in {v_city} has a recorded capacity, so there is "
                f"nothing to assess here.")
    else:
        v_menu, v_err = get_menu(folder, v_code)
        if v_err:
            st.error(v_err)
        else:
            labels = [f"{r.venue}  —  {r.capacity:,.0f} ({r.io or 'unlabelled'})"
                      for r in v_lad.itertuples(index=False)]
            pick = st.selectbox("Venue", labels, key="venue_pick")
            v_name = v_lad.iloc[labels.index(pick)]["venue"]

            v_val = V.validate_rightsizing(ex, v_row, v_lad)
            # Pull is model output, so it is reported as the interval between
            # the two specifications rather than as a point -- the same rule
            # every other model number in this app follows. A is the upper
            # bound (capacity keeps credit for every unobserved reason the city
            # is attractive); B is the lower (past activity takes as much of
            # that credit as it can).
            v_pull = {sp: V.market_pull(v_menu, sp, ex, v_row, v_name, v_lad)
                      for sp in ("A", "B")}
            v_lo = min(v_pull["A"]["pull"], v_pull["B"]["pull"])
            v_hi = max(v_pull["A"]["pull"], v_pull["B"]["pull"])
            v_per, v_cap, v_kind = V.expectations(
                v_menu, "A", ex, v_row, v_name, v_lad)
            v_sum = V.summarise(v_per, v_cap, v_name, v_row, v_pull["A"], v_val)

            # ---- the headline numbers -------------------------------------
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Capacity", money(v_cap),
                      delta=v_kind or "unlabelled", delta_color="off")
            m2.metric("Tours it pulls to the city",
                      "0" if not v_sum["binding"]
                      else f"{v_lo:+.1f} to {v_hi:+.1f}",
                      delta="binding ceiling" if v_sum["binding"]
                            else "not the ceiling", delta_color="off")
            m3.metric("Tours in play for the city", f"{v_sum['in_play']:,}")
            m4.metric("Of those, it could host", f"{v_sum['fits']:,}")

            # ---- half one: pull, which needs no assumption ----------------
            kind_word = {"inside": "indoor room",
                         "outside": "outdoor space"}.get(v_kind, "room")
            st.markdown("#### What this room pulls to the city")
            if v_sum["binding"]:
                now_in, now_out, _ = v_pull["A"]["ceilings_now"]
                w_in, w_out, _ = v_pull["A"]["ceilings_without"]
                st.success(
                    f"**This is the biggest {kind_word} in {v_city}.** "
                    f"Without it the city's "
                    f"{'indoor' if v_kind == 'inside' else 'outdoor'} ceiling "
                    f"falls from "
                    f"{money(now_in if v_kind == 'inside' else now_out)} to "
                    f"{money(w_in if v_kind == 'inside' else w_out)}. The "
                    f"model expects the city to lose between "
                    f"**{v_lo:.1f} and {v_hi:.1f} tour-visits** over "
                    f"{whatif.years_covered(ex):.1f} years without it.")
                explain(
                    "This is the one figure here that does not depend on any "
                    "assumption about which room an act picks: it is the "
                    "market model run twice, with the room and without it. It "
                    "is a range because the two specifications bracket how "
                    "much credit capacity should get — see Method & limits.")
                st.caption(
                    f"Specification A (capacity keeps the credit): "
                    f"{v_pull['A']['expected_with']:.1f} → "
                    f"{v_pull['A']['expected_without']:.1f} tour-visits.  "
                    f"Specification B (past activity takes it): "
                    f"{v_pull['B']['expected_with']:.1f} → "
                    f"{v_pull['B']['expected_without']:.1f}.")
                if len(v_pull["A"]["per_tour"]):
                    with st.expander("The tours behind that number "
                                     "(specification A)"):
                        top = v_pull["A"]["per_tour"].head(25)
                        st.dataframe(
                            top[["headliner", "category", "act_plays",
                                 "room_needed", "p_with", "p_without", "pull"]],
                            use_container_width=True, hide_index=True,
                            column_config={
                                "room_needed": st.column_config.NumberColumn(
                                    "room it plays elsewhere", format="%d"),
                                "p_with": st.column_config.NumberColumn(
                                    "chance of the city, with this room",
                                    format="%.3f"),
                                "p_without": st.column_config.NumberColumn(
                                    "without it", format="%.3f"),
                                "pull": st.column_config.NumberColumn(
                                    "difference", format="%.3f")})
                        explain(
                            "Each row is one tour's probability of choosing "
                            "this city, with the room and without it. They sum "
                            "to the pull above.")
            else:
                st.info(
                    f"**This room pulls nothing to {v_city}, and that is not a "
                    f"criticism of it.** It is not the largest "
                    f"{kind_word} in the market, so removing it would "
                    f"not change the ceiling any act is tested against. The "
                    f"tours it hosts were coming to {v_city} anyway and would "
                    f"have played another room here.")
                explain(
                    "Market pull and usefulness are different things. A busy "
                    "mid-sized hall can have a pull of zero and still be where "
                    "most of the city's shows actually happen.")

            # ---- half two: the room split, with its reliability up front --
            st.divider()
            st.markdown("#### Which tours should land in this room")
            if v_sum["rule_trustworthy"]:
                st.caption(V.reliability(v_val))
            else:
                st.warning(V.reliability(v_val))

            in_play = v_per[v_per["likely to come"]].copy()
            if in_play.empty:
                st.info("The model gives this city little chance of any tour, "
                        "so there is nothing to split between its rooms.")
            else:
                counts = (in_play["can_host"].value_counts()
                          .rename_axis("verdict").reset_index(name="tours"))
                bar = alt.Chart(counts).mark_bar().encode(
                    x=alt.X("tours:Q", title="tours the city might get"),
                    y=alt.Y("verdict:N", sort="-x", title=None),
                    color=alt.Color(
                        "verdict:N", legend=None,
                        scale=alt.Scale(
                            domain=["fits", "too big for this room",
                                    "far too small for the act",
                                    "wrong kind of room", "size unknown"],
                            range=["#27ae60", "#c0392b", "#e67e22",
                                   "#7f8c8d", "#bdc3c7"])),
                    tooltip=["verdict", "tours"]).properties(height=170)
                st.altair_chart(bar, use_container_width=True)
                explain(
                    "Only tours with at least a 2% chance of the city are "
                    "counted, because below that the act is not realistically "
                    "in play and a long tail of near-zero rows buries the ones "
                    "that matter.")
                for _, c in counts.iterrows():
                    st.markdown(f"- **{c['verdict']}** — {int(c['tours'])} "
                                f"tours. {V.CAN_HOST_MEANING[c['verdict']]}")

                st.markdown("##### Tours it should expect")
                want = in_play[in_play["would_host"]].sort_values(
                    "p_city", ascending=False)
                if want.empty:
                    st.info("The rule puts none of the city's likely tours in "
                            "this room — they all suit another room here better.")
                else:
                    st.dataframe(
                        want[["headliner", "category", "act_plays",
                              "room_needed", "ratio to this room", "p_city"]]
                        .head(40), use_container_width=True, hide_index=True,
                        column_config={
                            "room_needed": st.column_config.NumberColumn(
                                "room it plays elsewhere", format="%d"),
                            "ratio to this room": st.column_config.NumberColumn(
                                "vs this room", format="%.2f",
                                help="What it plays elsewhere divided by this "
                                     "room's capacity. Near 1.0 is a good fit."),
                            "p_city": st.column_config.NumberColumn(
                                "chance of the city", format="%.3f")})
                    explain(
                        f"Expected visits landing here: "
                        f"**{v_sum['expected_here']:.1f}** of the "
                        f"{v_sum['expected_city']:.1f} the city is expected to "
                        f"get. That split inherits the rule's reliability "
                        f"above; the city total does not.")

                st.markdown("##### Tours it is not getting, and why")
                miss = in_play[~in_play["would_host"]].copy()
                for reason in ("too big for this room",
                               "far too small for the act",
                               "wrong kind of room"):
                    sub = miss[miss["can_host"] == reason]
                    if sub.empty:
                        continue
                    with st.expander(f"{reason} — {len(sub)} tours"):
                        st.caption(V.CAN_HOST_MEANING[reason])
                        st.dataframe(
                            sub.sort_values("p_city", ascending=False)[
                                ["headliner", "category", "act_plays",
                                 "room_needed", "ratio to this room",
                                 "room the rule picks", "p_city"]].head(30),
                            use_container_width=True, hide_index=True,
                            column_config={
                                "room_needed": st.column_config.NumberColumn(
                                    "room it plays elsewhere", format="%d"),
                                "ratio to this room":
                                    st.column_config.NumberColumn(
                                        "vs this room", format="%.2f"),
                                "room the rule picks":
                                    st.column_config.TextColumn(
                                        "where it would go instead",
                                        width="medium"),
                                "p_city": st.column_config.NumberColumn(
                                    "chance of the city", format="%.3f")})

                others = miss[miss["can_host"] == "fits"]
                if len(others):
                    with st.expander(
                            f"could host, but another room here suits it "
                            f"better — {len(others)} tours"):
                        st.caption(
                            "This room could physically take these acts. The "
                            "assignment rule sends them elsewhere in the same "
                            "city, which is the part of this screen that "
                            "depends on the rule being right.")
                        st.dataframe(
                            others.sort_values("p_city", ascending=False)[
                                ["headliner", "room_needed",
                                 "ratio to this room", "room the rule picks",
                                 "that room holds", "p_city"]].head(30),
                            use_container_width=True, hide_index=True)

            # ---- the check against fact -----------------------------------
            st.divider()
            st.markdown("#### What actually played here")
            actual = V.actual_at_venue(ex, v_row, v_name)
            if actual.empty:
                st.info("The extract records no tour at this venue. Where a "
                        "tour used several rooms in one city only one is "
                        "kept, so a room can be under-recorded.")
            else:
                st.dataframe(actual.head(40), use_container_width=True,
                             hide_index=True)
                explain(
                    f"{len(actual)} tour-visits recorded. Compare with the "
                    f"expected figure above: the period covered is "
                    f"{whatif.years_covered(ex):.1f} years, so a room expecting "
                    f"far more or far fewer than it got is the signal worth "
                    f"chasing.")

# ========================================================== RENOVATE =====

with tab_renovate:
    st.header("Change a room that already exists")
    st.markdown(
        "A renovation can do three things, and the model can see only two of "
        "them. **Raising a ceiling** it sees — but as exactly the same "
        "arithmetic as building a new room that size, so there is no separate "
        "answer to give. **Putting a roof on** it sees, and that one is "
        "genuinely distinct: an outdoor room becomes playable indoors and the "
        "city's indoor ceiling rises without anything being built. **Making "
        "the room better** it is blind to — nothing in the fitted coefficients "
        "knows about sightlines or bars.")

    ref_bo, bo_err = get_boxoffice()
    if bo_err:
        st.warning(bo_err)

    rcol1, rcol2 = st.columns([1, 2])
    with rcol1:
        r_code = pick_country("renovate_country")
    r_markets = (ex["markets"][ex["markets"]["countryCode"] == r_code]
                 .sort_values("events", ascending=False))
    with rcol2:
        r_city = st.selectbox("Market", r_markets["city"].tolist(),
                              key="renovate_market")
    r_row = r_markets[r_markets["city"] == r_city].iloc[0]
    r_lad = V.rooms(ex, r_row)

    if r_lad.empty:
        st.info(f"No room in {r_city} has a recorded capacity.")
    else:
        # Roofing only makes sense for a room that is currently outdoors.
        outdoor = r_lad[r_lad["io"] == "outside"]
        if outdoor.empty:
            st.info(
                f"{r_city} has no outdoor room with a recorded capacity, so "
                f"there is nothing here a roof would change. Use **Design a "
                f"venue** for an expansion, which is the same arithmetic.")
        else:
            r_menu, r_err = get_menu(folder, r_code)
            if r_err:
                st.error(r_err)
            else:
                labels = [f"{r.venue}  —  {r.capacity:,.0f} open"
                          for r in outdoor.itertuples(index=False)]
                pick = st.selectbox("Outdoor room to roof", labels,
                                    key="renovate_pick")
                r_name = outdoor.iloc[labels.index(pick)]["venue"]
                open_cap = float(outdoor.iloc[labels.index(pick)]["capacity"])

                suggested = int(open_cap * RN.DEFAULT_ROOF_FRACTION)
                r_cap = st.slider(
                    "Capacity once roofed", 1_000,
                    max(2_000, int(open_cap)), min(suggested, int(open_cap)),
                    500, key="renovate_cap",
                    help="Roofing a stadium does not give an arena of the same "
                         "size. Convertible venues land between a third and a "
                         "half of their open capacity — Lille's Stade "
                         "Pierre-Mauroy is 50,000 open and about 27,000 in its "
                         "indoor configuration.")

                years = whatif.years_covered(ex)
                roof = RN.add_roof(r_menu, ex, r_row, r_name, r_cap, r_lad)

                if roof["note"]:
                    st.warning(roof["note"])

                c1, c2, c3 = st.columns(3)
                c1.metric("Indoor ceiling now",
                          money(roof["indoor_ceiling_before"]))
                c2.metric("Indoor ceiling once roofed", money(r_cap))
                c3.metric(f"Extra tour-visits over {years:.1f} years",
                          f"{roof['extra_visits_low']:+.1f} to "
                          f"{roof['extra_visits_high']:+.1f}")

                st.markdown("#### A roof: what the model can see")
                st.dataframe(
                    RN.compare_roof_with_newbuild(r_menu, r_row, r_cap),
                    width="stretch", hide_index=True)
                explain(
                    "The two columns are identical, and that is the finding "
                    "rather than a bug. Both raise the same indoor ceiling to "
                    "the same number, and the ceiling is the only thing the "
                    "model sees. Where a roof and a new build genuinely "
                    "differ is the ladder — a roof converts a rung, a build "
                    "adds one — and that only shows up in the room split on "
                    "the Venue tab, which is reliable only in markets with "
                    "few rooms.")

                # ---- the refit, which the model cannot see -----------------
                st.divider()
                st.markdown("#### A refit: what the model cannot see")
                perf_table = get_performance(folder)
                target = st.select_slider(
                    "Where a refit would land this room among rooms of its size",
                    options=[0.25, 0.5, 0.75, 0.9], value=0.75,
                    format_func=lambda v: f"{v:.0%}", key="renovate_target",
                    help="0.75 is a deliberate ceiling on optimism: reaching "
                         "the top quarter is a good outcome for a "
                         "refurbishment. Assuming the top 10% assumes the "
                         "renovation is excellent before it is designed.")

                if bo_err:
                    st.info("Needs the box-office reference. Run "
                            "`python boxoffice.py --build` once.")
                else:
                    refit = RN.refurbish(ref_bo, perf_table, r_name, r_cap,
                                         target)
                    if not refit["known"]:
                        st.info(refit["why"])
                    else:
                        f1, f2, f3 = st.columns(3)
                        f1.metric("Sells now", f"{refit['sell_through']:.0f}%",
                                  delta=f"{refit['now_pp']:+.1f} pts vs its size",
                                  delta_color="off")
                        f2.metric("Typical for its size",
                                  f"{refit['typical_for_size']:.0f}%",
                                  delta=refit["band"], delta_color="off")
                        f3.metric("Gross per show after a refit",
                                  f"USD {refit['gross_after']:,.0f}",
                                  delta=f"{refit['gross_uplift']:+,.0f}",
                                  delta_color="off")
                        st.markdown(refit["why"])
                        explain(
                            "This changes no tour-count prediction at all. A "
                            "better room shows up in what it takes per show, "
                            "not in who comes — so it is kept on its own line "
                            "and never added to the roof's effect. One is a "
                            "model output with an identification problem "
                            "attached; the other is arithmetic on observed "
                            "sell-through.")

                    # ---- the money ----------------------------------------
                    st.divider()
                    st.markdown("#### Gross box office from the roof")
                    pp = refit.get("now_pp") or 0.0
                    rev = BO.project(ref_bo, roof["extra_visits_low"],
                                     roof["extra_visits_high"], r_cap,
                                     performance_pp=pp, years=years)
                    g1, g2 = st.columns(2)
                    # Streamlit reads $...$ as LaTeX and would eat both
                    # symbols, leaving a bare pair of numbers.
                    # Labelled USD rather than "$": Streamlit reads $...$ as
                    # LaTeX and silently swallows both symbols, and every
                    # figure in this project is USD by construction anyway.
                    g1.metric(f"Over {years:.1f} years",
                              f"USD {rev['central_low']:,.0f} to "
                              f"{rev['central_high']:,.0f}")
                    g2.metric("A year",
                              f"USD {rev['per_year_low']:,.0f} to "
                              f"{rev['per_year_high']:,.0f}")
                    st.error(
                        "**This is gross box office, not venue revenue.** It "
                        "is what the audience pays. The venue takes a hire fee "
                        "plus a share of ancillaries, which needs a rate card "
                        "this database does not have — `ref_hospitality` has "
                        "32 rows. Converting this to what the building earns "
                        "is a step you have to take yourself.")

                    st.markdown("##### Where that range comes from")
                    dec = BO.decompose(ref_bo, roof["extra_visits_low"],
                                       roof["extra_visits_high"], r_cap,
                                       performance_pp=pp)
                    st.dataframe(
                        dec, width="stretch", hide_index=True,
                        column_config={
                            "can more data fix it?":
                                st.column_config.TextColumn(
                                    "can more data fix it?", width="large"),
                            "share of the uncertainty":
                                st.column_config.ProgressColumn(
                                    "share of the uncertainty",
                                    format="%.2f", min_value=0.0,
                                    max_value=1.0)})
                    explain(
                        "This is the most useful row on the screen. The two "
                        "sources are comparable in size, so better ticket "
                        "price data would not narrow the answer much — and "
                        "neither would more events. What would narrow it is "
                        "Layer 3: a natural experiment around real venue "
                        "openings, which is the only thing that separates "
                        "capacity causing shows from capacity accompanying "
                        "them.")

                    with st.expander("The box-office curve this rests on"):
                        f = ref_bo["gross_fit"]
                        st.markdown(
                            f"`log(gross) = {f['intercept']:.3f} + "
                            f"{f['slope']:.3f} x log(capacity)`, fitted on "
                            f"**{f['n']:,} events** with r = {f['r']:.3f}.")
                        st.markdown(
                            f"So doubling a room's capacity multiplies gross "
                            f"by **{2 ** f['slope']:.2f}**, not by two — "
                            f"bigger rooms also charge more. A single show "
                            f"lands within a factor of "
                            f"{np.exp(1.96 * f['sigma']):.1f} of the median "
                            f"95% of the time, which is why nothing here is a "
                            f"point estimate.")
                        st.dataframe(BO.band_table(ref_bo), width="stretch",
                                     hide_index=True)
                        explain(
                            "Medians straight from the data, so the fit can "
                            "be checked against something visible. Note "
                            "sell-through RISING with capacity: that is "
                            "selection, not a law. Only acts that can fill a "
                            "30,000 room ever book one.")

# ===================================================== DESIGN A VENUE =====

with tab_design:
    st.header("Put a room that does not exist into a market")
    st.markdown("Pick a market, a size and a kind. Two answers come back: how "
                "many blocked tours the room **physically fits**, which is "
                "counting, and how many more the model expects it to **win**, "
                "which is not.")

    blob, model_err = get_model_card(folder)

    d1, d2 = st.columns([2, 3])
    with d1:
        drow = pick_market("design_market", None)
        name = st.text_input("Call it", "New Arena",
                             help="Cosmetic. The model sees a capacity and a "
                                  "kind, and nothing else about a building.")
        kind = st.radio("Kind", ["indoor", "outdoor"], horizontal=True,
                        help="An indoor arena does nothing for a festival act.")
    with d2:
        drec = get_recommendation(folder, drow["city"], drow["countryCode"],
                                  min_dates, gap_mult, borderline_mult,
                                  min_blocked)
        ceil_now = drec["ceilings"].get(kind)
        suggested = drec.get("size_median") or (
            int(ceil_now * 2) if ceil_now and np.isfinite(ceil_now) else 10_000)
        capacity = capacity_slider("Capacity", kind, "design_cap",
                                   suggested=suggested)
        st.caption(
            f"The blocked tours in {drow['city']} typically play "
            + (f"**{drec['size_median']:,}** seats ({drec['kind']})."
               if drec["case"] == "build"
               else "no consistent size — there is no capacity case here, so "
                    "any size below is your proposal, not a recommendation."))

    st.divider()
    st.markdown(f"#### Where {R.article(capacity)} {capacity:,}-seat {kind} "
                f"room sits among the rooms {drow['city']} already has")

    ladder = drec["ladder"]
    io_want = "inside" if kind == "indoor" else "outside"
    lad = ladder[ladder["io"] == io_want].head(14)[["venue", "capacity"]].copy()
    lad["what"] = "exists today"
    proposal = pd.DataFrame([{"venue": f"{name} (proposed)",
                              "capacity": float(capacity),
                              "what": "your proposal"}])
    both = pd.concat([lad, proposal], ignore_index=True).dropna(
        subset=["capacity"])
    if not both.empty:
        st.altair_chart(alt.Chart(both).mark_bar().encode(
            x=alt.X("capacity:Q", title="capacity"),
            y=alt.Y("venue:N", sort="-x", title=None),
            color=alt.Color("what:N", title=None,
                            scale=alt.Scale(
                                domain=["exists today", "your proposal"],
                                range=["#95a5a6", "#c0392b"])),
            tooltip=["venue", "capacity", "what"],
        ).properties(height=min(460, 28 * len(both))), width="stretch")

    if ceil_now and np.isfinite(ceil_now) and capacity <= ceil_now:
        st.warning(f"**{drow['city']} already has a bigger {kind} room "
                   f"({int(ceil_now):,} seats),** so this changes nothing in "
                   f"the model, which only sees the ceiling. A second room of "
                   f"the same size would genuinely relieve date congestion — "
                   f"this data cannot see that.")
    elif ceil_now and np.isfinite(ceil_now):
        st.caption(f"Raises the {kind} ceiling in {drow['city']} from "
                   f"{int(ceil_now):,} to {capacity:,} — "
                   f"{capacity - int(ceil_now):,} more seats.")
    else:
        st.caption(f"{drow['city']} has no {kind} room with a recorded "
                   f"capacity, so this sets the ceiling outright.")

    st.divider()
    st.markdown("#### Counting: blocked tours this room is big enough for")
    fits_tbl, n_fit, n_of_kind = whatif.clears(drec.get("blocked"), capacity,
                                               kind)
    if n_of_kind == 0:
        st.info(f"No tour that skipped {drow['city']} was blocked by the size "
                f"of its {kind} rooms at the current thresholds.")
    else:
        f1, f2, f3 = st.columns(3)
        f1.metric("Blocked tours of this kind", f"{n_of_kind:,}")
        f2.metric("Big enough for", f"{n_fit:,}",
                  delta=f"{n_fit / n_of_kind:.0%}", delta_color="off")
        f3.metric("Still too small for", f"{n_of_kind - n_fit:,}")
        explain("Counting, no model. 'Big enough' means the act has worked in "
                "a room this size elsewhere. It does **not** mean it would come.")
        cols = [c for c in ["headliner", "category", "plays", "country_dates",
                            "room_it_needs", "fits in the proposed room",
                            "seats short", "where"] if c in fits_tbl.columns]
        st.dataframe(fits_tbl[cols], width="stretch", hide_index=True,
                     column_config={"seats short":
                                    st.column_config.NumberColumn(
                                        "seats short", format="%d")})

    st.divider()
    st.markdown("#### Modelling: how many more tours the model expects")
    if model_err:
        st.warning(model_err)
    else:
        if st.button(f"Rebuild the {drow['country']} menu with {name} on it",
                     key="design_run", type="primary"):
            st.session_state["design_on"] = True
        if st.session_state.get("design_on"):
            with st.spinner(f"building the {drow['country']} choice menu"):
                menu_d, err = get_menu(folder, drow["countryCode"])
            if err:
                st.warning(err)
            elif drow["city"] not in set(menu_d["markets"]):
                st.warning(
                    f"{drow['city']} is not on the {drow['countryCode']} "
                    f"choice menu — a market needs at least "
                    f"{whatif.choice.MIN_MARKET_EVENTS} shows and a measured "
                    f"catchment to be somewhere a touring act was realistically "
                    f"weighing. The counting above still stands.")
            else:
                a = whatif.impact(menu_d, "A", drow["city"], capacity, kind)
                b = whatif.impact(menu_d, "B", drow["city"], capacity, kind)
                counterfactual_panel(menu_d, a, b, drow["city"], kind, capacity)

    st.divider()
    st.markdown("#### What this cannot tell you")
    st.dataframe(pd.DataFrame([
        {"blind spot": "Everything about the building except size and roof",
         "consequence": "Sightlines, loading bay, rail links, the promoter and "
                        "the rent are invisible. A better room of the same "
                        "capacity scores identically."},
        {"blind spot": "A second room the same size as the existing one",
         "consequence": "Registers as nothing: the model works through the "
                        "ceiling. Date congestion is real and unseen here."},
        {"blind spot": "Cause against selection",
         "consequence": "Cities have big arenas because promoters expected "
                        "demand. The capacity coefficient carries both, which "
                        "is why every answer is a range."},
        {"blind spot": "What the venue earns",
         "consequence": "The Renovate tab turns tour counts into GROSS BOX "
                        "OFFICE from 691,000 real events. That is what the "
                        "audience pays, not what the venue keeps: the hire "
                        "fee and ancillaries need a rate card this database "
                        "does not have (ref_hospitality has 32 rows)."},
    ]), width="stretch", hide_index=True,
        column_config={"consequence": st.column_config.TextColumn(
            "consequence", width="large")})


# ======================================================== BUILD CASE ======

with tab_build:
    st.header("If a market is short of tours, what size room would help?")
    st.caption("Counting only. This says what the acts that skipped actually "
               "play elsewhere — not that they would come.")

    screen, screen_file = get_screen(folder)
    ranked = None
    if screen is None:
        st.info("No screening results yet. Run `python screen.py`.")
    else:
        ranked = R.rank_all(screen, min_blocked=min_blocked, min_kept=min_kept,
                            require_below_peers=True)
        b1, b2 = st.columns(2)
        with b1:
            cases = st.multiselect("Show", CASES, default=["build"])
        with b2:
            pick_c = st.multiselect(
                "Countries", sorted(ranked["country"].dropna().unique()),
                default=[])
        view = ranked
        if cases:
            view = view[view["case"].isin(cases)]
        if pick_c:
            view = view[view["country"].isin(pick_c)]

        show = [c for c in ["case", "market", "country", "why",
                            "indoor_ceiling", "peer_median_ceiling",
                            "suggested_capacity", "seats to add", "gap_tours",
                            "skipped_tours", "catchment_kept",
                            "ceiling_caveat"] if c in view.columns]
        st.dataframe(
            view[show].head(80), width="stretch", hide_index=True,
            column_config={
                "why": st.column_config.TextColumn("why", width="large"),
                "suggested_capacity": st.column_config.NumberColumn(
                    "size to build", format="%d"),
                "seats to add": st.column_config.NumberColumn(
                    "seats more than today", format="%d"),
                "catchment_kept": st.column_config.ProgressColumn(
                    "kept", format="%.2f", min_value=0.0, max_value=1.0)})
        if screen_file:
            explain(f"From `reports/{screen_file}`. The per-market case below "
                    f"is re-derived from the extract every time.")

    st.divider()
    default_city = None
    if ranked is not None:
        builds = ranked[ranked["case"] == "build"]
        if not builds.empty:
            default_city = builds["market"].iloc[0]
    brow = pick_market("rec_market", default_city)
    rec = get_recommendation(folder, brow["city"], brow["countryCode"],
                             min_dates, gap_mult, borderline_mult, min_blocked)

    if rec["case"] == "build":
        st.success(f"### {R.headline(rec)}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clears half the blocked tours", f"{rec['size_median']:,}")
        m2.metric("…three in four", f"{rec['size_upper']:,}")
        m3.metric("Biggest room now", money(rec["ceilings"].get(rec["kind"])),
                  delta=rec["kind"], delta_color="off")
        m4.metric("Tours blocked", f"{rec['blocked_tours']:,}",
                  delta=f"of {rec['skipped_tours']:,} that skipped",
                  delta_color="off")
    else:
        st.info(f"### {R.headline(rec)}")
        st.markdown(rec.get("reason", ""))

    st.markdown("#### How that conclusion was reached")
    st.dataframe(
        pd.DataFrame(R.narrative(rec, rec["row"], rec["kept"], rec["caveat"]))
        [["step", "question", "answer", "because"]],
        width="stretch", hide_index=True,
        column_config={
            "step": st.column_config.NumberColumn("#", width="small"),
            "question": st.column_config.TextColumn("question", width="medium"),
            "answer": st.column_config.TextColumn("answer", width="medium"),
            "because": st.column_config.TextColumn("because", width="large")})
    explain("All counting, no model.")

    falsifiers = R.what_would_change_it(rec, rec["row"], rec["kept"],
                                        rec["caveat"])
    if falsifiers:
        st.markdown("#### What would overturn this")
        for f in falsifiers:
            st.markdown(f"- {f}")

    blocked = rec.get("blocked")
    if blocked is not None and not blocked.empty:
        with st.expander(f"The {len(blocked)} blocked tours"):
            cols = [c for c in ["headliner", "category", "plays",
                                "country_dates", "room_it_needs",
                                "our_ceiling_of_that_kind", "ratio", "verdict",
                                "where"] if c in blocked.columns]
            st.dataframe(blocked[cols].sort_values("room_it_needs",
                                                   ascending=False),
                         width="stretch", hide_index=True)
        went = R.where_blocked_tours_went(ex, rec["row"], blocked)
        if not went.empty:
            with st.expander("Where those blocked tours played instead"):
                st.dataframe(went, width="stretch", hide_index=True,
                             column_config={"share of the blocked tours":
                                            st.column_config.ProgressColumn(
                                                "share", format="%.2f",
                                                min_value=0.0, max_value=1.0)})


# =========================================================== METHOD =======

with tab_method:
    st.header("What this can and cannot tell you")
    st.dataframe(pd.DataFrame([
        {"layer": "1 — counting",
         "what it produces": "Capacity ladders, who skipped, what rooms they "
                             "play elsewhere, peer comparisons",
         "language it licenses": "here is what happens today"},
        {"layer": "2 — statistical",
         "what it produces": "The conditional logit: expected visits, the "
                             "counterfactual, the size curve",
         "language it licenses": "this associates with that"},
        {"layer": "3 — causal",
         "what it produces": "Not built. Difference-in-differences around real "
                             "venue openings",
         "language it licenses": "this caused that"},
        {"layer": "4 — simulation",
         "what it produces": "Coefficient draws propagated through the "
                             "counterfactual",
         "language it licenses": "here is the spread from estimation alone"},
    ]), width="stretch", hide_index=True,
        column_config={"what it produces": st.column_config.TextColumn(
            "what it produces", width="large")})

    st.markdown("""
**The direction of causation is the whole difficulty.** Cities with big arenas
get big shows — but cities that *build* big arenas are cities where promoters
already expected demand. Anything comparing markets on capacity and show count
measures both at once. That is why every model output here is a range between
**A**, which lets capacity keep credit for everything unobserved, and **B**,
which hands as much of that credit as possible to past activity.

**The simulation does not fix this.** It covers estimation uncertainty only —
how much the answer would move on another sample of tours. It is almost always
the narrower of the two ranges, and a narrow band is not confidence.

**Layer 3 is what would settle it**, because a venue that opened in 2018 gives a
before and an after for the same city. It does not exist yet.
""")

    st.markdown("#### The thresholds currently in force")
    st.dataframe(pd.DataFrame([
        {"threshold": "minimum catchment kept", "value": min_kept,
         "what it changes": "which markets are treated as satellites of a "
                            "larger one rather than markets in their own right"},
        {"threshold": "minimum country dates", "value": min_dates,
         "what it changes": "which tours count as having genuinely skipped a "
                            "market rather than never having considered it"},
        {"threshold": "capacity-gap multiple", "value": gap_mult,
         "what it changes": "how much larger a tour's usual room must be than "
                            "our ceiling before room size is blamed"},
        {"threshold": "borderline multiple", "value": borderline_mult,
         "what it changes": "where the evidence is reported as inconclusive"},
        {"threshold": "minimum blocked tours", "value": min_blocked,
         "what it changes": "how many blocked tours before a build case reads "
                            "'build' rather than 'too few'"},
        {"threshold": "coefficient draws", "value": draws,
         "what it changes": "how smooth the simulated bands are"},
        {"threshold": "peer population band", "value": "±35%",
         "what it changes": "which markets count as comparable"},
        {"threshold": "market merge radius", "value": "15 km",
         "what it changes": "which neighbouring cities fold into one market"},
    ]).astype({"value": str}), width="stretch", hide_index=True)

    st.markdown("#### The fitted model")
    blob, model_err = get_model_card(folder)
    if model_err:
        st.warning(model_err)
    else:
        st.caption(f"Fitted {blob['fitted']} on extract `{blob['extract']}` — "
                   f"{blob['n_rows']:,} rows, {blob['n_occasions']:,} choice "
                   f"occasions, {blob['n_tours']:,} tours across "
                   f"{', '.join(blob['countries'])}.")
        for spec, label in (
                ("A", "A — capacity keeps credit for every unobserved reason a "
                      "city is attractive (UPPER BOUND)"),
                ("B", "B — also controls for shows already hosted, which "
                      "absorbs part of the room's real effect (LOWER BOUND)")):
            s = blob["specs"][spec]
            st.markdown(f"**Specification {label}**")
            beta, se = np.array(s["beta"]), np.array(s["se"])
            st.dataframe(pd.DataFrame({
                "variable": s["names"],
                "coefficient": beta.round(4),
                "std error": se.round(4),
                "odds ratio": np.exp(beta).round(3),
                "2.5%": (beta - 1.96 * se).round(4),
                "97.5%": (beta + 1.96 * se).round(4),
                "distinguishable from zero": np.abs(beta) > 1.96 * se,
                "what it is": [blob["variable_notes"].get(n, "")
                               for n in s["names"]],
            }), width="stretch", hide_index=True,
                column_config={"what it is": st.column_config.TextColumn(
                    "what it is", width="large")})
            st.caption(f"pseudo-R² {s['pseudo_r2']:.3f} · standard errors "
                       f"{s['se_kind']} · {s['iterations']} iterations")

        st.markdown("**The constants the columns were centred and filled on**")
        st.caption("Saved with the coefficients because a coefficient on a "
                   "centred column only means anything against the centring it "
                   "was fitted with. The app applies these exactly rather than "
                   "re-deriving them from whichever country is loaded.")
        st.dataframe(pd.DataFrame([blob["design_constants"]]).T.rename(
            columns={0: "value"}), width="stretch")

    st.markdown("#### Known limitations in the source data")
    st.dataframe(ex["notes"], width="stretch", hide_index=True)

    st.markdown("#### What every column means")
    st.dataframe(ex["features"], width="stretch", hide_index=True)
