"""
Indoor / outdoor labelling for venues.

1. Use the hand-checked label from venues_df_clauded.csv where there is one.
2. Otherwise infer from the venue name (Italian-centric keyword rules).
3. Otherwise 'unknown'.

Every labelled row also gets io_source = 'clauded' | 'keyword' so the two
can be separated again.
"""
import re

import pandas as pd

# Venues whose name would be misread by the keyword rules.
KNOWN_INSIDE = {
    "unipol arena", "inalpi arena", "pala alpitour", "kioene arena", "virtus arena", "allianz cloud arena",
    "arena spettacoli padova fiere", "unipol forum", "mediolanum forum di assago", "forum di assago",
    "nelson mandela forum", "alcatraz", "fabrique", "estragon", "atlantico live", "magazzini generali",
    "viper", "viper theatre", "santeria toscana 31", "hiroshima mon amour", "new age", "casa della musica federico i",
    "hall", "teatro palapartenope", "palapartenope", "gran teatro geox", "teatro degli arcimboldi",
    "teatro dal verme", "auditorium parco della musica", "sala petrassi, auditorium parco della musica ennio morricone",
    "sala santa cecilia, auditorium parco della musica ennio morricone", "sala sinopoli, auditorium parco della musica ennio morricone",
    "teatro cartiere carrara", "teatro europauditorium", "teatro lirico giorgio gaber", "teatro out off",
    "demodé club", "eremo club", "teatroteam", "teatro team", "teatro petruzzelli", "duel:beat", "teatro augusteo",
    "che tempo che fa", "largo venue", "monk", "orion", "hall of rock", "live club", "legend club", "circolo magnolia",
    "the cage theatre - teatro mascagni di villa corridi", "ogr (officine grandi riparazioni)", "arci bellezza",
    "mamamia", "politeama rossetti", "hacienda", "urban", "fuori orario", "casa del jazz",
}
KNOWN_OUTSIDE = {
    "carroponte", "circo massimo", "terme di caracalla", "fiera del levante", "arena della vittoria",
    "reggia di caserta", "trentino music arena", "beky bay", "porto turistico", "isola del castello",
    "cavea luciano berio, auditorium parco della musica ennio morricone", "arena dei pini", "arena flegrea",
    "parco gondar", "praja gallipoli", "praja", "masseria ferragnano", "foro boario", "fossato del castello",
    "ippodromo delle capannelle", "ippodromo del galoppo di san siro", "ippodromo snai san siro", "rcf arena campovolo",
    "ex-base nato", "ex base nato", "villa manin", "villa bertelli", "villa ca' cornaro", "castello sforzesco",
    "castello carrarese", "castello pasquini", "giardino bellini", "giardini del frontone", "campo sportivo",
    "stadio comunale", "velodromo comunale paolo borsellino", "piazzale del castello", "piazza castello",
    "anfiteatro romano", "anfiteatro ivan graziani", "teatro romano", "teatro antico", "teatro greco",
    "sferisterio", "arena di verona", "arena santa giuliana", "parco della musica di milano", "idroscalo",
    "autodromo nazionale monza", "visarno arena", "ippodromo del visarno", "piazza del plebiscito", "piazza duomo",
    "piazza del popolo", "piazza roma", "piazza", "palazzo farnese", "assago summer arena", "rocca maggiore",
    "cava del sole", "arco della pace", "real sito di carditello", "cantieri culturali della zisa",
}

OUTSIDE_RE = re.compile(
    r"\b(piazza|piazzale|piazzetta|parco|park|giardin\w*|villa|castello|stadio|stadium|anfiteatro|amphitheat\w*|"
    r"cavea|terme|circo|campo sportivo|ippodromo|autodromo|velodromo|lungomare|porto|molo|spiaggia|lido|beach|bay|"
    r"isola|masseria|fiera|festival|open air|reggia|cortile|chiostro|fossato|cave|foro|rotonda|arena|"
    r"teatro (romano|antico|greco|all'aperto|di verdura|del silenzio)|summer|estate|sagrato|belvedere|"
    r"marina|darsena|prato|campo|piazzale|largo)\b", re.I)
INSIDE_RE = re.compile(
    r"\b(pala\w*|palasport|palazzo dello sport|palazzetto|forum|teatro|theatre|theater|auditorium|sala|club|"
    r"live|hall|discoteca|disco|circolo|cinema|casa della musica|magazzin\w*|fabbrica|spazio|centro congressi|"
    r"opera|conservatorio|chiesa|basilica|cattedrale|duomo|dome)\b", re.I)


def infer_from_name(name):
    if not isinstance(name, str):
        return None
    n = name.strip().casefold()
    if n in KNOWN_INSIDE:
        return "inside"
    if n in KNOWN_OUTSIDE:
        return "outside"
    if n.startswith("pala") and not n.startswith("palazzo"):
        return "inside"
    if OUTSIDE_RE.search(n) and not re.search(r"\b(pala\w*|palasport|palazzo dello sport)\b", n):
        return "outside"
    if INSIDE_RE.search(n):
        return "inside"
    return None


def label_venues(df, venues_csv=None, infer=True):
    """Add io ('inside'|'outside'|'unknown') and io_source columns to an events frame."""
    df = df.copy()
    df["io"] = "unknown"
    df["io_source"] = None
    if venues_csv:
        v = (pd.read_csv(venues_csv)[["venue", "city", "country", "outside-inside"]]
             .dropna(subset=["outside-inside"]).drop_duplicates(["venue", "city", "country"]))
        df = df.merge(v, how="left", on=["venue", "city", "country"])
        lab = df["outside-inside"].astype(str).str.strip().str.lower()
        ok = lab.isin(["inside", "outside"])
        df.loc[ok, "io"] = lab[ok]
        df.loc[ok, "io_source"] = "clauded"
        df = df.drop(columns=["outside-inside"])
    if infer:
        need = df["io"] == "unknown"
        guess = df.loc[need, "venue"].map(infer_from_name)
        hit = guess.notna()
        df.loc[guess[hit].index, "io"] = guess[hit]
        df.loc[guess[hit].index, "io_source"] = "keyword"
    return df
