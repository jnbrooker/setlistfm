#!/usr/bin/env python3
"""
Where everything lives.

Every path the project reads or writes by default is resolved against THIS
file's folder, not the current working directory. That makes the whole thing
location-proof: move the jambase folder anywhere, run a script from anywhere,
and it still finds the same database, workbooks and state.

Python puts a script's own directory on sys.path when you run it, so
`import paths` resolves no matter where you are standing.

Each default can be overridden with an environment variable, which is how you
point a script at a second database or a workbook kept elsewhere:

    JAMBASE_DB                 setlistfm.db
    JAMBASE_ARENA_WORKBOOK     arenas-dashboard.xlsx
    JAMBASE_POLLSTAR_WORKBOOK  pollstar-data.xlsx
    JAMBASE_STATE              setlistfm_state/
    JAMBASE_KEY                setlistfm_key.txt

A relative path passed on the command line still resolves against the current
directory, as a command-line argument should -- only the defaults are pinned.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))


def here(name):
    """Absolute path to a file in the project folder."""
    return os.path.join(HERE, name)


def env_path(var, name):
    """An environment override if set, otherwise the file in the project folder."""
    value = os.environ.get(var)
    return os.path.abspath(os.path.expanduser(value)) if value else here(name)


# the database and the two source workbooks
DB = env_path("JAMBASE_DB", "setlistfm.db")
ARENA_WORKBOOK = env_path("JAMBASE_ARENA_WORKBOOK", "arenas-dashboard.xlsx")
POLLSTAR_WORKBOOK = env_path("JAMBASE_POLLSTAR_WORKBOOK", "pollstar-data.xlsx")

# scraper state and credentials
STATE_DIR = env_path("JAMBASE_STATE", "setlistfm_state")
KEY_FILE = env_path("JAMBASE_KEY", "setlistfm_key.txt")

# hand-maintained override lists
ARTIST_ALIASES_MANUAL = here("artist_aliases_manual.csv")
ARENA_ALIASES_MANUAL = here("arena_aliases_manual.csv")
# Hand-made venue identity decisions -- "these names are one building" -- that
# seed_identity.py folds into venue_identity. A file rather than rows typed into
# the table, because venue_identity is rebuilt on every seed and a file is what
# survives that, and what shows up in a git diff when someone changes it.
VENUE_IDENTITY_MANUAL = here("venue_identity_manual.csv")

# default outputs
EVENTS_CSV = here("events.csv")
TOURS_CSV = here("setlistfm_tours_new.csv")
WITH_CITIES_CSV = here("with_cities_setlistfm_db.csv")
LEGACY_TOURS_CSV = here("setlistfm_tours.csv")
