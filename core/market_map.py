"""LFN firm list: cleaning, keyword classification, geocoding and a prospect score.

The public demo does not call Zefix. The federal API requires HTTP Basic
credentials, and a search cannot list a canton without a name. This module
reads ``data/lfn_firms.csv`` when that file has rows, and otherwise a built-in
example whose names all start with "Exemple".

Geocoding uses the swisstopo Search API (geo.admin.ch), commune centroids
only, and stores the result in ``data/commune_geocodes.csv``.
"""

from __future__ import annotations

import hashlib
import json
import re
import ssl
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIRM_CSV_PATH = ROOT / "data" / "lfn_firms.csv"
GEOCODE_CACHE_PATH = ROOT / "data" / "commune_geocodes.csv"

GEOADMIN_SEARCH = "https://api3.geo.admin.ch/rest/services/api/SearchServer"
CANTONS = ("GE", "VD")
FIRM_TYPES = ("lawyer", "fiduciary", "notary")
SENIORITY_CAP_YEARS = 30

# Point scales are the criterion before the weight. 1 is the top of that scale.
FORM_POINTS: dict[str, float] = {
    "SA": 1.0,
    "Sàrl": 0.8,
    "SNC": 0.5,
    "Raison individuelle": 0.3,
    "Other": 0.3,
}
TYPE_POINTS: dict[str, float] = {
    "fiduciary": 1.0,
    "notary": 0.85,
    "lawyer": 0.7,
}
DEFAULT_WEIGHTS: dict[str, float] = {
    "legal_form": 1.0,
    "seniority": 1.0,
    "firm_type": 1.0,
    "local_density": 1.0,
}

# Order is the tie-break: a name that matches two professions keeps the earlier one.
_KEYWORD_GROUPS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("notary", re.compile(r"\bnotaires?\b|\bnotariats?\b|\bnotariales?\b|\bnotarial\b|\bnotar\b", re.I)),
    (
        "fiduciary",
        re.compile(
            r"\bfiduciaires?\b|\btreuhand\w*\b|\brevisions?\b|\brévisions?\b|"
            r"\breviseurs?\b|\bréviseurs?\b|\bexpert[- ]comptables?\b|\bexpertise comptable\b",
            re.I,
        ),
    ),
    ("lawyer", re.compile(r"\bavocats?\b|\bavocates?\b|\brechtsanwalt\w*\b|\banwalt\w*\b|\banwälte\b", re.I)),
)

_HEADER_MAP = {
    "nom": "name",
    "name": "name",
    "type": "firm_type",
    "forme_juridique": "legal_form",
    "forme juridique": "legal_form",
    "legal_form": "legal_form",
    "commune": "commune",
    "date_inscription": "registered_on",
    "date d'inscription": "registered_on",
    "registered_on": "registered_on",
    "statut": "status",
    "status": "status",
    "canton": "canton",
}

_FORM_ALIASES = {
    "sa": "SA",
    "societe anonyme": "SA",
    "aktiengesellschaft": "SA",
    "ag": "SA",
    "sarl": "Sàrl",
    "sàrl": "Sàrl",
    "societe a responsabilite limitee": "Sàrl",
    "gmbh": "Sàrl",
    "snc": "SNC",
    "societe en nom collectif": "SNC",
    "raison individuelle": "Raison individuelle",
    "entreprise individuelle": "Raison individuelle",
    "einzelunternehmen": "Raison individuelle",
    "ri": "Raison individuelle",
}

_STATUS_ALIASES = {
    "active": "active",
    "actif": "active",
    "activee": "active",
    "en liquidation": "in liquidation",
    "liquidation": "in liquidation",
    "being_cancelled": "in liquidation",
    "being cancelled": "in liquidation",
    "in liquidation": "in liquidation",
    "deleted": "deleted",
    "cancelled": "deleted",
    "radie": "deleted",
    "radiee": "deleted",
    "annule": "deleted",
}

_CANTON_ALIASES = {
    "ge": "GE",
    "geneve": "GE",
    "genève": "GE",
    "geneva": "GE",
    "vd": "VD",
    "vaud": "VD",
}


class FirmFileError(ValueError):
    """The local firm file has rows, but they cannot be used."""


@dataclass(frozen=True)
class FirmBook:
    """Firms ready to score, and where they came from."""

    firms: pd.DataFrame
    source: str
    fictional: bool


def normalize_text(value: object) -> str:
    """Lower case, no accents, punctuation turned into spaces."""
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def classify_name(name: object) -> str | None:
    """Map a firm name to lawyer, fiduciary or notary. None if no keyword matches.

    "Étude" alone is not enough: in Geneva it can be a law office or a notary office.
    A name that matches two professions keeps the first group in ``_KEYWORD_GROUPS``.
    """
    text = normalize_text(name)
    if not text:
        return None
    for firm_type, pattern in _KEYWORD_GROUPS:
        if pattern.search(text):
            return firm_type
    return None


def normalize_legal_form(value: object) -> str:
    key = normalize_text(value)
    if not key:
        return "Other"
    return _FORM_ALIASES.get(key, "Other")


def normalize_status(value: object) -> str:
    key = normalize_text(value)
    return _STATUS_ALIASES.get(key, "active" if not key else "other")


def normalize_canton(value: object) -> str:
    key = normalize_text(value)
    return _CANTON_ALIASES.get(key, "")


def _rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: _HEADER_MAP[str(column).strip().lower()] for column in frame.columns if str(column).strip().lower() in _HEADER_MAP}
    out = frame.rename(columns=renamed)
    missing = {"name", "commune", "registered_on"} - set(out.columns)
    if missing:
        raise FirmFileError(
            "The firm file needs columns nom, commune and date_inscription. Missing: " + ", ".join(sorted(missing))
        )
    for column in ("firm_type", "legal_form", "status", "canton"):
        if column not in out.columns:
            out[column] = ""
    return out


def prepare_firms(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename, classify, keep Geneva and Vaud, and drop duplicate firms.

    A duplicate is the same normalised name in the same commune. The earliest
    registration date is kept. Rows without a recognised profession are dropped.
    """
    renamed = _rename_columns(frame)
    rows: list[dict[str, object]] = []
    for record in renamed.to_dict(orient="records"):
        name = str(record.get("name") or "").strip()
        commune = str(record.get("commune") or "").strip()
        if not name or not commune or name.lower() == "nan":
            continue
        declared = normalize_text(record.get("firm_type"))
        if declared in FIRM_TYPES:
            firm_type = declared
        else:
            firm_type = classify_name(name)
        if firm_type not in FIRM_TYPES:
            continue
        registered = pd.to_datetime(record.get("registered_on"), errors="coerce")
        if pd.isna(registered):
            continue
        canton = normalize_canton(record.get("canton"))
        rows.append(
            {
                "name": name,
                "firm_type": firm_type,
                "legal_form": normalize_legal_form(record.get("legal_form")),
                "commune": commune,
                "canton": canton,
                "registered_on": registered.date(),
                "status": normalize_status(record.get("status")),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["name", "firm_type", "legal_form", "commune", "canton", "registered_on", "status", "name_key"]
        )
    clean = pd.DataFrame(rows)
    clean["name_key"] = clean["name"].map(normalize_text)
    clean = clean.sort_values("registered_on").drop_duplicates(["name_key", "commune"], keep="first")
    return clean.reset_index(drop=True)


def restrict_cantons(firms: pd.DataFrame, cache_path: Path | None = None) -> pd.DataFrame:
    """Keep Geneva and Vaud. A blank canton is filled when the commune cache has exactly one match."""
    if firms.empty:
        return firms
    path = GEOCODE_CACHE_PATH if cache_path is None else cache_path
    cache = _read_geocode_cache(path)
    by_commune: dict[str, set[str]] = {}
    for record in cache.itertuples(index=False):
        canton = str(record.canton).upper()
        if canton in CANTONS:
            by_commune.setdefault(normalize_text(record.commune), set()).add(canton)

    def resolve(row: pd.Series) -> str:
        if row.canton in CANTONS:
            return str(row.canton)
        matches = by_commune.get(normalize_text(row.commune), set())
        if len(matches) == 1:
            return next(iter(matches))
        return ""

    located = firms.copy()
    located["canton"] = located.apply(resolve, axis=1)
    return located[located["canton"].isin(CANTONS)].reset_index(drop=True)


def fictional_firms() -> pd.DataFrame:
    """Example book. Every name starts with 'Exemple' so it cannot be read as a register extract."""
    rows = [
        ("Exemple Avocats du Rhône SA", "lawyer", "SA", "Genève", "GE", "1998-06-01", "active"),
        ("Exemple Étude de la Treille Sàrl", "notary", "Sàrl", "Genève", "GE", "2001-02-14", "active"),
        ("Exemple Fiduciaire du Léman SA", "fiduciary", "SA", "Genève", "GE", "1995-11-03", "active"),
        ("Exemple Avocats des Eaux-Vives Sàrl", "lawyer", "Sàrl", "Genève", "GE", "2012-09-20", "active"),
        ("Exemple Notaires de Plainpalais Sàrl", "notary", "Sàrl", "Genève", "GE", "2008-04-07", "active"),
        ("Exemple Révision du Rhône SA", "fiduciary", "SA", "Genève", "GE", "2003-01-15", "active"),
        ("Exemple Avocate des Pâquis", "lawyer", "Raison individuelle", "Genève", "GE", "2019-05-22", "active"),
        ("Exemple Treuhand Genève Sàrl", "fiduciary", "Sàrl", "Genève", "GE", "2016-08-30", "active"),
        ("Exemple Fiduciaire de Carouge Sàrl", "fiduciary", "Sàrl", "Carouge", "GE", "2011-03-11", "active"),
        ("Exemple Avocats de Carouge SA", "lawyer", "SA", "Carouge", "GE", "2006-12-01", "active"),
        ("Exemple Notaire de Lancy", "notary", "Raison individuelle", "Lancy", "GE", "2018-07-19", "active"),
        ("Exemple Fiduciaire de Vernier SNC", "fiduciary", "SNC", "Vernier", "GE", "2009-10-05", "active"),
        ("Exemple Avocats de Meyrin Sàrl", "lawyer", "Sàrl", "Meyrin", "GE", "2014-02-28", "in liquidation"),
        ("Exemple Étude Notariale d'Onex Sàrl", "notary", "Sàrl", "Onex", "GE", "2000-05-17", "active"),
        ("Exemple Fiduciaire de Thônex SA", "fiduciary", "SA", "Thônex", "GE", "2021-01-08", "active"),
        ("Exemple Avocats de Versoix Sàrl", "lawyer", "Sàrl", "Versoix", "GE", "2017-06-14", "active"),
        ("Exemple Avocats du Flon SA", "lawyer", "SA", "Lausanne", "VD", "1992-04-02", "active"),
        ("Exemple Fiduciaire Lausannoise SA", "fiduciary", "SA", "Lausanne", "VD", "1988-09-21", "active"),
        ("Exemple Notaires de la Palud Sàrl", "notary", "Sàrl", "Lausanne", "VD", "2005-11-09", "active"),
        ("Exemple Révision du Léman SA", "fiduciary", "SA", "Lausanne", "VD", "2002-03-26", "active"),
        ("Exemple Avocats de la Cité Sàrl", "lawyer", "Sàrl", "Lausanne", "VD", "2015-08-13", "active"),
        ("Exemple Treuhand Lausanne Sàrl", "fiduciary", "Sàrl", "Lausanne", "VD", "2010-12-02", "active"),
        ("Exemple Étude Notariale d'Ouchy SNC", "notary", "SNC", "Lausanne", "VD", "1999-01-29", "active"),
        ("Exemple Avocate de Bellevaux", "lawyer", "Raison individuelle", "Lausanne", "VD", "2022-04-18", "active"),
        ("Exemple Avocats de Nyon SA", "lawyer", "SA", "Nyon", "VD", "2007-07-07", "active"),
        ("Exemple Fiduciaire de la Côte Sàrl", "fiduciary", "Sàrl", "Nyon", "VD", "2013-05-03", "active"),
        ("Exemple Notaires de Nyon Sàrl", "notary", "Sàrl", "Nyon", "VD", "2004-10-25", "active"),
        ("Exemple Fiduciaire de Morges SA", "fiduciary", "SA", "Morges", "VD", "1997-02-11", "active"),
        ("Exemple Avocats de Morges Sàrl", "lawyer", "Sàrl", "Morges", "VD", "2018-09-01", "active"),
        ("Exemple Notaire de Vevey", "notary", "Raison individuelle", "Vevey", "VD", "2011-06-16", "active"),
        ("Exemple Fiduciaire de Vevey Sàrl", "fiduciary", "Sàrl", "Vevey", "VD", "2008-01-23", "active"),
        ("Exemple Avocats de Montreux SA", "lawyer", "SA", "Montreux", "VD", "2000-08-08", "active"),
        ("Exemple Treuhand Riviera Sàrl", "fiduciary", "Sàrl", "Montreux", "VD", "2016-03-19", "in liquidation"),
        ("Exemple Fiduciaire d'Yverdon SA", "fiduciary", "SA", "Yverdon-les-Bains", "VD", "1994-05-30", "active"),
        ("Exemple Avocats d'Yverdon Sàrl", "lawyer", "Sàrl", "Yverdon-les-Bains", "VD", "2012-11-27", "active"),
        ("Exemple Révision de Renens Sàrl", "fiduciary", "Sàrl", "Renens", "VD", "2019-09-09", "active"),
        ("Exemple Notaires de Pully Sàrl", "notary", "Sàrl", "Pully", "VD", "2003-04-04", "active"),
        ("Exemple Avocats de Rolle Sàrl", "lawyer", "Sàrl", "Rolle", "VD", "2020-02-02", "active"),
        ("Exemple Fiduciaire du Chablais SA", "fiduciary", "SA", "Aigle", "VD", "2006-06-06", "active"),
        ("Exemple Notaire d'Aigle", "notary", "Raison individuelle", "Aigle", "VD", "2014-12-12", "active"),
    ]
    return pd.DataFrame(rows, columns=["name", "firm_type", "legal_form", "commune", "canton", "registered_on", "status"])


def load_firm_book(csv_path: Path | None = None) -> FirmBook:
    """Use the CSV when it has data rows. Otherwise return the fictional example."""
    path = FIRM_CSV_PATH if csv_path is None else csv_path
    if path.exists():
        raw = pd.read_csv(path, comment="#", dtype=str).dropna(how="all")
        if len(raw):
            firms = restrict_cantons(prepare_firms(raw), path.parent / "commune_geocodes.csv")
            if firms.empty:
                raise FirmFileError(
                    "data/lfn_firms.csv has rows, but none could be kept. "
                    "Each row needs a name, a Geneva or Vaud canton, a commune, a registration date, "
                    "and a type (lawyer, fiduciary, notary) or a profession keyword in the name."
                )
            return FirmBook(firms, "local file", False)
    firms = prepare_firms(fictional_firms())
    return FirmBook(firms, "fictional example", True)


def score_firms(
    firms: pd.DataFrame,
    *,
    weights: Mapping[str, float] | None = None,
    type_points: Mapping[str, float] | None = None,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Rank firms from 0 to 100 as a weighted average of four criteria.

    Definition
        score = 100 × (w_form·p_form + w_age·p_age + w_type·p_type + w_density·p_density) / Σw
        If the weights sum to 0, the score is 0.

    Points, each on [0, 1]
        p_form: SA 1, Sàrl 0.8, SNC 0.5, raison individuelle and any other form 0.3.
        p_age: years since registration / 30, capped at 1. A future date scores 0.
        p_type: fiduciary 1, notary 0.85, lawyer 0.7, unless ``type_points`` overrides them.
        p_density: number of firms in the same commune divided by the busiest commune.

    Assumptions
        The advisor should call structured, established firms in the communes where
        the profession already clusters. Status (active, in liquidation) is not in the score.
        Density is computed on the frame passed in, before the page filters it.

    Limits
        The points are a prospecting order, not a measure of wealth or credit quality.
        Keyword classification can mis-label a firm. Sole practitioners who are not
        in the commercial register never appear. Seniority uses the registration date
        the file provides, which for Zefix is not a dedicated field.
    """
    if firms.empty:
        out = firms.copy()
        for column in ("form_points", "seniority_points", "type_points", "density_points", "score"):
            out[column] = []
        return out

    used_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        for key in DEFAULT_WEIGHTS:
            used_weights[key] = max(0.0, float(weights.get(key, DEFAULT_WEIGHTS[key])))
    points = dict(TYPE_POINTS)
    if type_points:
        for key in TYPE_POINTS:
            points[key] = min(1.0, max(0.0, float(type_points.get(key, TYPE_POINTS[key]))))
    today = date.today() if as_of is None else as_of

    scored = firms.copy()
    scored["form_points"] = scored["legal_form"].map(lambda form: FORM_POINTS.get(str(form), FORM_POINTS["Other"]))
    scored["seniority_points"] = scored["registered_on"].map(lambda registered: _seniority(registered, today))
    scored["type_points"] = scored["firm_type"].map(lambda kind: points.get(str(kind), 0.0))
    counts = scored.groupby("commune")["name"].transform("size")
    busiest = float(counts.max()) if len(counts) else 1.0
    scored["density_points"] = counts / busiest if busiest else 0.0

    weight_sum = sum(used_weights.values())
    if weight_sum == 0:
        scored["score"] = 0.0
    else:
        weighted = (
            used_weights["legal_form"] * scored["form_points"]
            + used_weights["seniority"] * scored["seniority_points"]
            + used_weights["firm_type"] * scored["type_points"]
            + used_weights["local_density"] * scored["density_points"]
        )
        scored["score"] = (100 * weighted / weight_sum).round(1)
    return scored.sort_values(["score", "name"], ascending=[False, True]).reset_index(drop=True)


def _seniority(registered: object, as_of: date) -> float:
    if not isinstance(registered, date):
        return 0.0
    years = max(0.0, (as_of - registered).days / 365.25)
    return min(1.0, years / SENIORITY_CAP_YEARS)


def top_prospects(firms: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Highest scores first. ``firms`` must already be scored."""
    if "score" not in firms.columns:
        raise ValueError("score the firms before ranking them")
    return firms.sort_values(["score", "name"], ascending=[False, True]).head(limit).reset_index(drop=True)


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def fetch_commune(name: str, canton: str) -> tuple[float, float] | None:
    """Commune centroid from geo.admin.ch (swisstopo). None if the commune is not found.

    The request asks only the gg25 municipality layer, in WGS84. No street, no person.
    """
    query = urllib.parse.urlencode(
        {"searchText": name, "type": "locations", "origins": "gg25", "sr": "4326", "limit": "8"}
    )
    request = urllib.request.Request(
        f"{GEOADMIN_SEARCH}?{query}",
        headers={"Accept": "application/json", "User-Agent": "lfn-advisor-copilot-student/0.1"},
    )
    with urllib.request.urlopen(request, timeout=20, context=_ssl_context()) as response:
        payload = json.loads(response.read().decode("utf-8"))
    target = normalize_text(name)
    canton_key = canton.lower()
    for item in payload.get("results") or []:
        attrs = item.get("attrs") or {}
        if attrs.get("origin") != "gg25":
            continue
        detail = str(attrs.get("detail") or "").lower()
        label = re.sub(r"<[^>]+>", "", str(attrs.get("label") or ""))
        label_name = normalize_text(label.split("(")[0])
        if label_name != target:
            continue
        if canton_key and not detail.endswith(" " + canton_key):
            continue
        lat, lon = attrs.get("lat"), attrs.get("lon")
        if lat is None or lon is None:
            continue
        return float(lat), float(lon)
    return None


def _read_geocode_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["commune", "canton", "lat", "lon"])
    cache = pd.read_csv(path)
    for column in ("commune", "canton", "lat", "lon"):
        if column not in cache.columns:
            cache[column] = pd.NA
    return cache


def attach_coordinates(
    firms: pd.DataFrame,
    cache_path: Path | None = None,
    fetcher: Callable[[str, str], tuple[float, float] | None] | None = None,
) -> pd.DataFrame:
    """Add commune centroids. Missing communes are fetched once and appended to the cache.

    ``lat`` and ``lon`` are the commune centre. ``lat_display`` and ``lon_display``
    move each firm by a few hundred metres so firms in the same commune stay visible.
    That offset is not an address.
    """
    if firms.empty:
        out = firms.copy()
        out["lat"] = []
        out["lon"] = []
        out["lat_display"] = []
        out["lon_display"] = []
        return out

    path = GEOCODE_CACHE_PATH if cache_path is None else cache_path
    lookup = fetcher or fetch_commune
    cache = _read_geocode_cache(path)
    cache["key"] = cache["commune"].map(normalize_text) + "|" + cache["canton"].map(lambda value: str(value).upper())
    known = {row.key: (float(row.lat), float(row.lon)) for row in cache.itertuples(index=False) if pd.notna(row.lat)}
    added: list[dict[str, object]] = []

    latitudes: list[float | None] = []
    longitudes: list[float | None] = []
    for record in firms.itertuples(index=False):
        key = f"{normalize_text(record.commune)}|{record.canton}"
        point = known.get(key)
        if point is None:
            try:
                point = lookup(str(record.commune), str(record.canton))
            except (OSError, ValueError, json.JSONDecodeError):
                point = None
            if point is not None:
                known[key] = point
                added.append({"commune": record.commune, "canton": record.canton, "lat": point[0], "lon": point[1]})
        latitudes.append(None if point is None else point[0])
        longitudes.append(None if point is None else point[1])

    if added:
        updated = pd.concat([cache.drop(columns=["key"]), pd.DataFrame(added)], ignore_index=True)
        updated = updated.drop_duplicates(["commune", "canton"], keep="last")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            updated.to_csv(path, index=False)
        except OSError:
            pass

    located = firms.copy()
    located["lat"] = latitudes
    located["lon"] = longitudes
    display_lat: list[float | None] = []
    display_lon: list[float | None] = []
    for record in located.itertuples(index=False):
        if pd.isna(record.lat) or pd.isna(record.lon):
            display_lat.append(None)
            display_lon.append(None)
            continue
        digest = hashlib.sha256(str(record.name).encode("utf-8")).digest()
        display_lat.append(float(record.lat) + (digest[0] / 255 - 0.5) * 0.012)
        display_lon.append(float(record.lon) + (digest[1] / 255 - 0.5) * 0.008)
    located["lat_display"] = display_lat
    located["lon_display"] = display_lon
    return located


def counts_by_type(firms: pd.DataFrame) -> pd.DataFrame:
    if firms.empty:
        return pd.DataFrame(columns=["firm_type", "firms"])
    return firms.groupby("firm_type", as_index=False).size().rename(columns={"size": "firms"})


def counts_by_commune(firms: pd.DataFrame) -> pd.DataFrame:
    if firms.empty:
        return pd.DataFrame(columns=["commune", "canton", "firms"])
    return (
        firms.groupby(["commune", "canton"], as_index=False)
        .size()
        .rename(columns={"size": "firms"})
        .sort_values(["firms", "commune"], ascending=[False, True])
        .reset_index(drop=True)
    )


def creations_by_year(firms: pd.DataFrame) -> pd.DataFrame:
    if firms.empty:
        return pd.DataFrame(columns=["year", "firms"])
    years = pd.DataFrame({"year": firms["registered_on"].map(lambda value: value.year if isinstance(value, date) else pd.NA)})
    return years.dropna().groupby("year", as_index=False).size().rename(columns={"size": "firms"})
