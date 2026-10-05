"""Sanity checks on the fictional client book."""

from __future__ import annotations

from core.portfolios import INSTRUMENTS, REFERENCE_FX, build_clients


def test_ten_clients_with_valid_attributes() -> None:
    clients = build_clients()
    assert len(clients) == 10
    assert {c.client_type for c in clients} == {"lawyer", "fiduciary", "notary"}
    assert {c.account_type for c in clients} == {"private", "firm"}
    assert {c.risk_profile for c in clients} == {"conservative", "balanced", "dynamic"}
    assert all(c.reference_currency == "CHF" for c in clients)
    assert any(c.lombard for c in clients) and not all(c.lombard for c in clients)


def test_generation_is_deterministic() -> None:
    assert build_clients() == build_clients()
    assert build_clients(seed=1) != build_clients()


def test_reference_value_close_to_target_aum() -> None:
    # At reference prices, each portfolio should be worth roughly its target AUM.
    rochat = next(c for c in build_clients() if c.client_id == "C03")
    value = sum(
        h.quantity * INSTRUMENTS[h.ticker].reference_price * REFERENCE_FX[INSTRUMENTS[h.ticker].currency]
        for h in rochat.holdings
    )
    assert abs(value / 2_800_000 - 1) < 0.01
