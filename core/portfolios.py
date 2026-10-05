"""Fictional LFN client book: clients, portfolios and Lombard loans.

Every name, amount and position below is invented. Any resemblance to a real
firm or person is coincidental. Portfolios are generated deterministically
(fixed seed) so the demo is reproducible.

How a portfolio is built
------------------------
1. Each client has a target AUM in CHF and a target weight per asset class.
2. Some clients carry deliberate "overweights" (e.g. an inherited Nestlé
   position) so that the Risk Monitor has realistic issues to detect.
3. The remaining weight of each asset class is split across randomly chosen
   instruments of that class (seeded Dirichlet draw).
4. Target CHF amounts are converted into share quantities using fixed
   *reference prices*. From then on, quantities are frozen: the portfolio is
   valued with real market prices, so actual weights drift away from targets
   exactly as they would in a real account.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

SEED = 42

ClientType = Literal["lawyer", "fiduciary", "notary"]
AccountType = Literal["private", "firm"]
RiskProfile = Literal["conservative", "balanced", "dynamic"]
AssetClass = Literal["swiss_equity", "intl_equity", "bonds", "gold", "cash"]

REFERENCE_CURRENCY = "CHF"
EQUITY_CLASSES: tuple[AssetClass, ...] = ("swiss_equity", "intl_equity")

ASSET_CLASS_LABELS: dict[str, str] = {
    "swiss_equity": "Swiss equities",
    "intl_equity": "International equities",
    "bonds": "Bond ETFs",
    "gold": "Gold",
    "cash": "Cash",
}


@dataclass(frozen=True)
class Instrument:
    """A tradable instrument (or a cash account) in the investment universe.

    Attributes:
        ticker: Yahoo Finance symbol, or ``CASH_<CCY>`` for cash accounts.
        name: Human-readable name.
        asset_class: One of :data:`AssetClass`.
        currency: Currency the instrument is quoted in.
        reference_price: Indicative local-currency price, used only once to
            turn target CHF amounts into quantities. Not used for valuation.
        modified_duration: Approximate modified duration in years (bond ETFs
            only), used by the interest-rate stress test. Static assumption.
    """

    ticker: str
    name: str
    asset_class: AssetClass
    currency: str
    reference_price: float
    modified_duration: float = 0.0


# Investment universe. Durations are rounded, indicative figures.
INSTRUMENTS: dict[str, Instrument] = {
    i.ticker: i
    for i in [
        # Swiss equities (CHF)
        Instrument("NESN.SW", "Nestlé", "swiss_equity", "CHF", 75.0),
        Instrument("NOVN.SW", "Novartis", "swiss_equity", "CHF", 115.0),
        Instrument("RO.SW", "Roche (bearer)", "swiss_equity", "CHF", 355.0),
        Instrument("UBSG.SW", "UBS Group", "swiss_equity", "CHF", 40.0),
        Instrument("ZURN.SW", "Zurich Insurance", "swiss_equity", "CHF", 570.0),
        Instrument("ABBN.SW", "ABB", "swiss_equity", "CHF", 82.0),
        Instrument("CFR.SW", "Richemont", "swiss_equity", "CHF", 175.0),
        Instrument("LONN.SW", "Lonza", "swiss_equity", "CHF", 575.0),
        Instrument("SREN.SW", "Swiss Re", "swiss_equity", "CHF", 140.0),
        Instrument("GIVN.SW", "Givaudan", "swiss_equity", "CHF", 3370.0),
        # International equities
        Instrument("AAPL", "Apple", "intl_equity", "USD", 330.0),
        Instrument("MSFT", "Microsoft", "intl_equity", "USD", 525.0),
        Instrument("JNJ", "Johnson & Johnson", "intl_equity", "USD", 250.0),
        Instrument("ASML.AS", "ASML", "intl_equity", "EUR", 1650.0),
        Instrument("SAP.DE", "SAP", "intl_equity", "EUR", 185.0),
        Instrument("MC.PA", "LVMH", "intl_equity", "EUR", 380.0),
        # Bond ETFs
        Instrument("CSBGC0.SW", "iShares Swiss Govt Bond 0-3", "bonds", "CHF", 102.5, 1.6),
        Instrument("CSBGC3.SW", "iShares Swiss Govt Bond 3-7", "bonds", "CHF", 62.0, 4.9),
        Instrument("IEAG.AS", "iShares Euro Aggregate Bond", "bonds", "EUR", 103.0, 6.4),
        Instrument("IEF", "iShares 7-10Y US Treasury", "bonds", "USD", 89.0, 7.1),
        Instrument("AGG", "iShares Core US Aggregate Bond", "bonds", "USD", 94.0, 6.0),
        # Gold
        Instrument("GLD", "SPDR Gold Shares", "gold", "USD", 380.0),
        # Cash accounts (price = 1 unit of local currency)
        Instrument("CASH_CHF", "Cash CHF", "cash", "CHF", 1.0),
        Instrument("CASH_EUR", "Cash EUR", "cash", "EUR", 1.0),
        Instrument("CASH_USD", "Cash USD", "cash", "USD", 1.0),
    ]
}

# Indicative FX rates (CHF per unit) used only to size quantities.
REFERENCE_FX: dict[str, float] = {"CHF": 1.0, "EUR": 0.93, "USD": 0.83}

# Standard Lombard advance rates ("valeurs d'avance") by asset class: the share
# of market value the bank accepts to lend against. Indicative market practice
# for liquid, diversified collateral.
DEFAULT_ADVANCE_RATES: dict[str, float] = {
    "cash": 0.95,
    "bonds": 0.85,
    "swiss_equity": 0.70,
    "intl_equity": 0.60,
    "gold": 0.65,
}


@dataclass(frozen=True)
class Holding:
    """A position: an instrument ticker and a quantity (shares or cash units)."""

    ticker: str
    quantity: float


@dataclass(frozen=True)
class LombardLoan:
    """A Lombard credit secured by the client's portfolio.

    Attributes:
        amount_chf: Drawn amount in CHF. Assumed to be used outside the
            portfolio (e.g. property purchase), so it is not held as cash.
        advance_rates: Advance rate per asset class (0-1).
    """

    amount_chf: float
    advance_rates: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ADVANCE_RATES))


@dataclass(frozen=True)
class Client:
    """A fictional LFN client and their portfolio."""

    client_id: str
    name: str
    client_type: ClientType
    account_type: AccountType
    risk_profile: RiskProfile
    reference_currency: str
    holdings: tuple[Holding, ...]
    lombard: LombardLoan | None = None

    @property
    def tickers(self) -> list[str]:
        return [h.ticker for h in self.holdings]

    @property
    def quantities(self) -> dict[str, float]:
        return {h.ticker: h.quantity for h in self.holdings}


@dataclass(frozen=True)
class _ClientSpec:
    """Blueprint used to generate a client's portfolio."""

    client_id: str
    name: str
    client_type: ClientType
    account_type: AccountType
    risk_profile: RiskProfile
    aum_chf: float
    class_weights: dict[str, float]
    cash_split: dict[str, float]
    picks_per_class: dict[str, int]
    overweights: dict[str, float] = field(default_factory=dict)
    universe: dict[str, tuple[str, ...]] = field(default_factory=dict)
    lombard_chf: float | None = None
    advance_rate_overrides: dict[str, float] = field(default_factory=dict)


# Home bias: conservative CHF investors typically hold CHF bonds only.
_CHF_BONDS = ("CSBGC0.SW", "CSBGC3.SW")
_CORE_BONDS = ("CSBGC0.SW", "CSBGC3.SW", "IEAG.AS")


_SPECS: list[_ClientSpec] = [
    _ClientSpec(
        "C01", "Étude Dubois & Associés", "lawyer", "firm", "conservative", 4_200_000,
        {"swiss_equity": 0.15, "intl_equity": 0.05, "bonds": 0.45, "gold": 0.05, "cash": 0.30},
        {"CHF": 0.85, "EUR": 0.15}, {"swiss_equity": 4, "intl_equity": 2, "bonds": 2},
        universe={"bonds": _CHF_BONDS},
    ),
    _ClientSpec(
        "C02", "Fiduciaire Lémanique SA", "fiduciary", "firm", "balanced", 6_500_000,
        {"swiss_equity": 0.30, "intl_equity": 0.20, "bonds": 0.35, "gold": 0.05, "cash": 0.10},
        {"CHF": 0.7, "EUR": 0.2, "USD": 0.1}, {"swiss_equity": 6, "intl_equity": 4, "bonds": 3},
        universe={"bonds": _CORE_BONDS}, lombard_chf=1_500_000,
    ),
    _ClientSpec(
        # Inherited Nestlé block pushes equities and single-line concentration
        # above the conservative limits.
        "C03", "Me Rochat, notaire", "notary", "private", "conservative", 2_800_000,
        {"swiss_equity": 0.34, "intl_equity": 0.08, "bonds": 0.40, "gold": 0.03, "cash": 0.15},
        {"CHF": 1.0}, {"swiss_equity": 3, "intl_equity": 2, "bonds": 2},
        overweights={"NESN.SW": 0.16}, universe={"bonds": _CHF_BONDS},
    ),
    _ClientSpec(
        # Highly leveraged equity portfolio: Lombard usage close to the limit.
        "C04", "Me Claire Fontaine, avocate", "lawyer", "private", "dynamic", 3_100_000,
        {"swiss_equity": 0.48, "intl_equity": 0.35, "bonds": 0.10, "gold": 0.0, "cash": 0.07},
        {"CHF": 0.6, "USD": 0.4}, {"swiss_equity": 5, "intl_equity": 4, "bonds": 1},
        lombard_chf=1_950_000,
    ),
    _ClientSpec(
        "C05", "Fiduciaire du Rhône Sàrl", "fiduciary", "firm", "conservative", 1_900_000,
        {"swiss_equity": 0.20, "intl_equity": 0.08, "bonds": 0.50, "gold": 0.04, "cash": 0.18},
        {"CHF": 0.9, "EUR": 0.1}, {"swiss_equity": 4, "intl_equity": 2, "bonds": 2},
        universe={"bonds": _CHF_BONDS},
    ),
    _ClientSpec(
        "C06", "Étude Morel Notaires", "notary", "firm", "balanced", 5_400_000,
        {"swiss_equity": 0.32, "intl_equity": 0.20, "bonds": 0.33, "gold": 0.05, "cash": 0.10},
        {"CHF": 0.8, "EUR": 0.2}, {"swiss_equity": 6, "intl_equity": 3, "bonds": 3},
        universe={"bonds": _CORE_BONDS},
    ),
    _ClientSpec(
        # Former banker: large UBS position, single-line limit breached.
        "C07", "Me Laurent Vuilleumier, avocat", "lawyer", "private", "balanced", 2_200_000,
        {"swiss_equity": 0.35, "intl_equity": 0.17, "bonds": 0.33, "gold": 0.05, "cash": 0.10},
        {"CHF": 0.8, "USD": 0.2}, {"swiss_equity": 4, "intl_equity": 3, "bonds": 2},
        overweights={"UBSG.SW": 0.17}, universe={"bonds": _CHF_BONDS},
    ),
    _ClientSpec(
        "C08", "Cabinet Favre Fiscalité SA", "fiduciary", "firm", "dynamic", 3_700_000,
        {"swiss_equity": 0.40, "intl_equity": 0.35, "bonds": 0.12, "gold": 0.08, "cash": 0.05},
        {"CHF": 0.5, "EUR": 0.25, "USD": 0.25}, {"swiss_equity": 6, "intl_equity": 5, "bonds": 2},
        lombard_chf=1_900_000,
    ),
    _ClientSpec(
        # Mostly USD assets: foreign-currency exposure above the profile limit.
        "C09", "Me Sophie Duc, notaire", "notary", "private", "dynamic", 1_600_000,
        {"swiss_equity": 0.12, "intl_equity": 0.55, "bonds": 0.18, "gold": 0.08, "cash": 0.07},
        {"USD": 0.7, "CHF": 0.3}, {"swiss_equity": 2, "intl_equity": 5, "bonds": 2},
    ),
    _ClientSpec(
        "C10", "Étude Bonvin & Perrin", "lawyer", "firm", "balanced", 7_800_000,
        {"swiss_equity": 0.30, "intl_equity": 0.22, "bonds": 0.35, "gold": 0.05, "cash": 0.08},
        {"CHF": 0.75, "EUR": 0.15, "USD": 0.10}, {"swiss_equity": 7, "intl_equity": 4, "bonds": 3},
        universe={"bonds": _CORE_BONDS}, lombard_chf=2_000_000,
        # Concentrated collateral pool: the bank applies a haircut on equities.
        advance_rate_overrides={"swiss_equity": 0.65, "intl_equity": 0.55},
    ),
]


def _instruments_of(asset_class: str) -> list[str]:
    return [t for t, i in INSTRUMENTS.items() if i.asset_class == asset_class]


def _target_amounts(spec: _ClientSpec, rng: np.random.Generator) -> dict[str, float]:
    """Split a client's AUM into target CHF amounts per ticker (pure, seeded)."""
    amounts: dict[str, float] = {}

    for asset_class, class_weight in spec.class_weights.items():
        if class_weight <= 0:
            continue
        if asset_class == "cash":
            for ccy, share in spec.cash_split.items():
                amounts[f"CASH_{ccy}"] = spec.aum_chf * class_weight * share
            continue

        # Overweights are carved out of their asset class budget.
        fixed = {t: w for t, w in spec.overweights.items() if INSTRUMENTS[t].asset_class == asset_class}
        remainder = class_weight - sum(fixed.values())
        if remainder < 0:
            raise ValueError(f"{spec.client_id}: overweights exceed the {asset_class} weight.")
        for ticker, weight in fixed.items():
            amounts[ticker] = spec.aum_chf * weight

        universe = spec.universe.get(asset_class) or _instruments_of(asset_class)
        candidates = [t for t in universe if t not in fixed]
        n_picks = min(spec.picks_per_class.get(asset_class, 1), len(candidates))
        if remainder == 0 or n_picks == 0:
            continue
        picks = rng.choice(candidates, size=n_picks, replace=False)
        # Dirichlet(alpha=4) gives uneven but not extreme splits.
        splits = rng.dirichlet(np.full(n_picks, 4.0))
        for ticker, share in zip(picks, splits):
            amounts[str(ticker)] = spec.aum_chf * remainder * share

    return amounts


def _to_holdings(amounts: dict[str, float]) -> tuple[Holding, ...]:
    """Convert CHF target amounts into quantities using reference prices.

    Securities are rounded to whole shares; cash is rounded to the cent.
    """
    holdings: list[Holding] = []
    for ticker, amount_chf in amounts.items():
        inst = INSTRUMENTS[ticker]
        local_amount = amount_chf / REFERENCE_FX[inst.currency]
        if inst.asset_class == "cash":
            quantity = round(local_amount, 2)
        else:
            quantity = float(max(1, round(local_amount / inst.reference_price)))
        holdings.append(Holding(ticker, quantity))
    return tuple(sorted(holdings, key=lambda h: (INSTRUMENTS[h.ticker].asset_class, h.ticker)))


def build_clients(seed: int = SEED) -> list[Client]:
    """Generate the 10 fictional LFN clients (deterministic for a given seed).

    Each client gets its own child RNG (``seed + index``) so that editing one
    client does not reshuffle the others.
    """
    clients: list[Client] = []
    for idx, spec in enumerate(_SPECS):
        rng = np.random.default_rng(seed + idx)
        holdings = _to_holdings(_target_amounts(spec, rng))
        lombard = None
        if spec.lombard_chf:
            rates = {**DEFAULT_ADVANCE_RATES, **spec.advance_rate_overrides}
            lombard = LombardLoan(amount_chf=spec.lombard_chf, advance_rates=rates)
        clients.append(
            Client(
                client_id=spec.client_id,
                name=spec.name,
                client_type=spec.client_type,
                account_type=spec.account_type,
                risk_profile=spec.risk_profile,
                reference_currency=REFERENCE_CURRENCY,
                holdings=holdings,
                lombard=lombard,
            )
        )
    return clients


def all_tickers(clients: list[Client]) -> list[str]:
    """Market tickers needed to value the book (cash accounts excluded)."""
    return sorted({h.ticker for c in clients for h in c.holdings if not h.ticker.startswith("CASH_")})
