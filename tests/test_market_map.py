"""Market map: classification, the prospect score, and the fictional fallback. No network."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from core.market_map import (
    attach_coordinates,
    classify_name,
    load_firm_book,
    prepare_firms,
    restrict_cantons,
    score_firms,
    top_prospects,
)

AS_OF = date(2026, 10, 6)


def _firms() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Ancien Fiduciaire SA", "fiduciary", "SA", "Genève", "GE", date(1990, 1, 1), "active"),
            ("Jeune Avocate", "lawyer", "Raison individuelle", "Genève", "GE", date(2024, 1, 1), "active"),
            ("Notaire Seul", "notary", "Sàrl", "Aigle", "VD", date(2010, 6, 1), "active"),
            ("Deuxième de Genève SA", "lawyer", "SA", "Genève", "GE", date(2000, 1, 1), "active"),
        ],
        columns=["name", "firm_type", "legal_form", "commune", "canton", "registered_on", "status"],
    )


def test_keywords_cover_the_three_professions_and_ignore_etude_alone() -> None:
    assert classify_name("Étude Dupont, notaires") == "notary"
    assert classify_name("Treuhand Müller AG") == "fiduciary"
    assert classify_name("Cabinet de révision du Léman") == "fiduciary"
    assert classify_name("Me Anne Avocate") == "lawyer"
    assert classify_name("Étude de la Treille") is None
    assert classify_name("Notaire et avocat associés") == "notary"


def test_prepare_drops_other_cantons_duplicates_and_unclassified_names() -> None:
    raw = pd.DataFrame(
        [
            ["Exemple Avocats SA", "lawyer", "SA", "Genève", "GE", "2010-01-01", "active"],
            ["exemple avocats sa", "lawyer", "SA", "Genève", "GE", "2001-01-01", "active"],
            ["Holding sans profession SA", "", "SA", "Lausanne", "VD", "2010-01-01", "active"],
            ["Avocats de Zurich SA", "lawyer", "SA", "Zürich", "ZH", "2010-01-01", "active"],
        ],
        columns=["nom", "type", "forme_juridique", "commune", "canton", "date_inscription", "statut"],
    )
    clean = restrict_cantons(prepare_firms(raw))
    assert list(clean["name"]) == ["exemple avocats sa"] or list(clean["name"]) == ["Exemple Avocats SA"]
    assert len(clean) == 1
    assert clean.iloc[0]["registered_on"] == date(2001, 1, 1)
    assert set(clean["canton"]) == {"GE"}


def test_blank_canton_is_filled_from_the_commune_cache(tmp_path: Path) -> None:
    cache = tmp_path / "commune_geocodes.csv"
    pd.DataFrame([{"commune": "Nyon", "canton": "VD", "lat": 46.38, "lon": 6.24}]).to_csv(cache, index=False)
    raw = pd.DataFrame(
        [["Exemple Fiduciaire de Nyon Sàrl", "fiduciary", "Sàrl", "Nyon", "", "2015-05-05", "active"]],
        columns=["nom", "type", "forme_juridique", "commune", "canton", "date_inscription", "statut"],
    )
    clean = restrict_cantons(prepare_firms(raw), cache)
    assert list(clean["canton"]) == ["VD"]


def test_score_follows_each_weight() -> None:
    firms = _firms()
    by_form = score_firms(firms, weights={"legal_form": 1, "seniority": 0, "firm_type": 0, "local_density": 0}, as_of=AS_OF)
    form_order = by_form.set_index("name")["score"]
    assert form_order["Ancien Fiduciaire SA"] > form_order["Jeune Avocate"]

    by_age = score_firms(firms, weights={"legal_form": 0, "seniority": 1, "firm_type": 0, "local_density": 0}, as_of=AS_OF)
    age_order = by_age.set_index("name")["score"]
    assert age_order["Ancien Fiduciaire SA"] > age_order["Jeune Avocate"]
    assert age_order["Ancien Fiduciaire SA"] == 100.0

    by_type = score_firms(firms, weights={"legal_form": 0, "seniority": 0, "firm_type": 1, "local_density": 0}, as_of=AS_OF)
    type_order = by_type.set_index("name")["score"]
    assert type_order["Ancien Fiduciaire SA"] > type_order["Notaire Seul"] > type_order["Jeune Avocate"]

    by_density = score_firms(firms, weights={"legal_form": 0, "seniority": 0, "firm_type": 0, "local_density": 1}, as_of=AS_OF)
    density = by_density.set_index("name")["score"]
    assert density["Deuxième de Genève SA"] > density["Notaire Seul"]

    zero = score_firms(firms, weights={"legal_form": 0, "seniority": 0, "firm_type": 0, "local_density": 0}, as_of=AS_OF)
    assert set(zero["score"]) == {0.0}
    assert zero["score"].between(0, 100).all()


def test_type_points_can_flatten_the_profession_ranking() -> None:
    firms = _firms()
    flat = score_firms(
        firms,
        weights={"legal_form": 0, "seniority": 0, "firm_type": 1, "local_density": 0},
        type_points={"lawyer": 1, "fiduciary": 1, "notary": 1},
        as_of=AS_OF,
    )
    assert set(flat["score"]) == {100.0}


def test_top_prospects_returns_at_most_twenty() -> None:
    firms = _firms()
    ranked = top_prospects(score_firms(firms, as_of=AS_OF), 20)
    assert len(ranked) == 4
    assert ranked["score"].is_monotonic_decreasing


def test_empty_csv_is_the_fictional_example_and_a_filled_csv_is_not(tmp_path: Path) -> None:
    empty = tmp_path / "lfn_firms.csv"
    empty.write_text("nom,type,forme_juridique,commune,canton,date_inscription,statut\n", encoding="utf-8")
    book = load_firm_book(empty)
    assert book.fictional
    assert book.firms["name"].str.startswith("Exemple").all()
    assert set(book.firms["canton"]) <= {"GE", "VD"}
    assert {"street", "email", "phone", "person"} .isdisjoint(book.firms.columns)

    filled = tmp_path / "filled.csv"
    filled.write_text(
        "nom,type,forme_juridique,commune,canton,date_inscription,statut\n"
        "Cabinet Exemple Avocats SA,lawyer,SA,Lausanne,VD,2011-04-04,active\n",
        encoding="utf-8",
    )
    local = load_firm_book(filled)
    assert not local.fictional
    assert list(local.firms["name"]) == ["Cabinet Exemple Avocats SA"]


def test_geocoding_reads_the_cache_and_does_not_call_the_network(tmp_path: Path) -> None:
    cache = tmp_path / "commune_geocodes.csv"
    pd.DataFrame([{"commune": "Lausanne", "canton": "VD", "lat": 46.52, "lon": 6.63}]).to_csv(cache, index=False)
    firms = pd.DataFrame(
        [["Exemple Avocats SA", "lawyer", "SA", "Lausanne", "VD", date(2010, 1, 1), "active"]],
        columns=["name", "firm_type", "legal_form", "commune", "canton", "registered_on", "status"],
    )

    def fail(name: str, canton: str) -> tuple[float, float]:
        raise AssertionError(f"network call for {name} {canton}")

    located = attach_coordinates(firms, cache, fetcher=fail)
    assert located.iloc[0]["lat"] == pytest.approx(46.52)
    assert located.iloc[0]["lat_display"] != pytest.approx(46.52)


def test_a_cache_miss_is_stored(tmp_path: Path) -> None:
    cache = tmp_path / "commune_geocodes.csv"
    firms = pd.DataFrame(
        [
            ["Exemple A SA", "lawyer", "SA", "Morges", "VD", date(2010, 1, 1), "active"],
            ["Exemple B Sàrl", "notary", "Sàrl", "Morges", "VD", date(2012, 1, 1), "active"],
        ],
        columns=["name", "firm_type", "legal_form", "commune", "canton", "registered_on", "status"],
    )
    calls: list[str] = []

    def once(name: str, canton: str) -> tuple[float, float]:
        calls.append(name)
        return 46.51, 6.50

    attach_coordinates(firms, cache, fetcher=once)
    assert calls == ["Morges"]
    stored = pd.read_csv(cache)
    assert list(stored["commune"]) == ["Morges"]
