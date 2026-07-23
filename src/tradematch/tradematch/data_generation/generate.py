"""Synthetic equity trade data generator for TradeMatch.

Produces three files:
  - ledger.csv           the firm's internal record of trades
  - custodian.csv        the external record, with deliberately planted breaks
  - break_manifest.json  the ground-truth answer key of every planted break

The manifest is what lets us evaluate the agent later: we can only measure
"did it catch the breaks?" because we know exactly which breaks exist.
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42            # fixed seed => same data every run => reproducible tests
N_TRADES = 200       # number of clean trades before breaks are injected
BREAK_RATE = 0.15    # ~15% of trades will have a break planted

# A small universe of real tickers with their real CUSIPs. Matching on CUSIP
# (a permanent 9-char security id) is what real reconciliation systems do.
SECURITIES = [
    ("AAPL", "037833100"),
    ("MSFT", "594918104"),
    ("AMZN", "023135106"),
    ("GOOGL", "02079K305"),
    ("NVDA", "67066G104"),
    ("TSLA", "88160R101"),
    ("JPM", "46625H100"),
    ("V", "92826C839"),
]

COUNTERPARTIES = ["GOLDMAN", "MORGAN", "CITADEL", "JANE_ST", "VIRTU"]

BREAK_TYPES = ["timing", "price", "quantity", "missing", "duplicate"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """One equity trade. Fields split into identity vs. economic."""
    trade_id: str
    trade_date: str        # ISO date string, e.g. "2026-07-14"
    settlement_date: str   # ISO date string; T+1 => trade_date + 1 business day
    ticker: str
    cusip: str
    side: str              # "BUY" or "SELL"
    quantity: int
    price: float
    gross_amount: float    # quantity * price, rounded to cents
    currency: str
    counterparty: str


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _add_business_days(start: date, n: int) -> date:
    """Add n business days to a date, skipping Saturdays and Sundays."""
    current = start
    added = 0
    while added < n:
        current += timedelta(days=1)
        if current.weekday() < 5:      # Mon=0 .. Fri=4 are business days
            added += 1
    return current


def _business_days(start: date, n: int) -> list[date]:
    """Return the first n business days on or after `start` (skips weekends)."""
    days: list[date] = []
    current = start
    while len(days) < n:
        if current.weekday() < 5:      # Mon=0 .. Fri=4
            days.append(current)
        current += timedelta(days=1)
    return days


def _make_base_trades(rng: random.Random) -> list[Trade]:
    """Create N_TRADES clean, internally-consistent trades (the ground truth)."""
    trades: list[Trade] = []
    start = date(2026, 7, 1)
    trading_days = _business_days(start, 10)   # a 2-week window of valid days

    for i in range(N_TRADES):
        ticker, cusip = rng.choice(SECURITIES)
        trade_date = rng.choice(trading_days)
        # T+1 settlement: one business day after the trade date
        settlement_date = _add_business_days(trade_date, 1)
        side = rng.choice(["BUY", "SELL"])
        quantity = rng.randint(1, 50) * 100          # round lots: 100..5000
        price = round(rng.uniform(50, 500), 2)
        gross = round(quantity * price, 2)

        trades.append(Trade(
            trade_id=f"TRD-{i:05d}",
            trade_date=trade_date.isoformat(),
            settlement_date=settlement_date.isoformat(),
            ticker=ticker,
            cusip=cusip,
            side=side,
            quantity=quantity,
            price=price,
            gross_amount=gross,
            currency="USD",
            counterparty=rng.choice(COUNTERPARTIES),
        ))
    return trades


# ---------------------------------------------------------------------------
# Break injection
# ---------------------------------------------------------------------------

def _inject_breaks(trades: list[Trade], rng: random.Random):
    """Return (ledger, custodian, manifest).

    The base `trades` are TRUTH. For each break we pick a side -- ledger or
    custodian -- and corrupt that side away from truth. Either side can be
    wrong, because in reality either side can be wrong.

    The manifest records which side was corrupted. That is EVALUATION-ONLY
    ground truth: the agent must never see it. At runtime the agent sees two
    disagreeing records and no oracle telling it which one is correct.
    """
    # Both sides start as independent copies of truth, so either can be mutated.
    ledger = [Trade(**asdict(t)) for t in trades]
    custodian = [Trade(**asdict(t)) for t in trades]
    manifest: list[dict] = []

    n_breaks = int(len(trades) * BREAK_RATE)
    break_indices = rng.sample(range(len(trades)), n_breaks)

    # Length-changing edits are collected per side and applied after the loop.
    remove: dict[str, set[int]] = {"ledger": set(), "custodian": set()}
    add: dict[str, list[Trade]] = {"ledger": [], "custodian": []}

    for idx in break_indices:
        break_type = rng.choice(BREAK_TYPES)
        corrupted_side = rng.choice(["ledger", "custodian"])

        # `target` is the record we corrupt; `truth` is the untouched original.
        target = (ledger if corrupted_side == "ledger" else custodian)[idx]
        truth = trades[idx]

        if break_type == "timing":
            # This side booked settlement 1-2 business days later.
            new_settle = _add_business_days(
                date.fromisoformat(target.settlement_date), rng.randint(1, 2)
            )
            target.settlement_date = new_settle.isoformat()

        elif break_type == "price":
            # Price off by a small percentage; gross recomputed to stay consistent.
            target.price = round(target.price * rng.uniform(1.01, 1.05), 2)
            target.gross_amount = round(target.quantity * target.price, 2)

        elif break_type == "quantity":
            target.quantity = target.quantity + rng.choice([-100, 100, 200])
            target.gross_amount = round(target.quantity * target.price, 2)

        elif break_type == "missing":
            # The trade never made it into this side's records.
            remove[corrupted_side].add(idx)

        elif break_type == "duplicate":
            # This side booked the same trade twice.
            add[corrupted_side].append(Trade(**asdict(target)))

        manifest.append({
            "trade_id": truth.trade_id,
            "break_type": break_type,
            "corrupted_side": corrupted_side,
            "true_value": _field_for(truth, break_type),
            "corrupted_value": (
                None if break_type == "missing" else _field_for(target, break_type)
            ),
        })

    ledger = [t for i, t in enumerate(ledger) if i not in remove["ledger"]]
    custodian = [t for i, t in enumerate(custodian) if i not in remove["custodian"]]
    ledger.extend(add["ledger"])
    custodian.extend(add["custodian"])

    return ledger, custodian, manifest


def _field_for(trade: Trade, break_type: str):
    """Return the field value relevant to a given break type (for the manifest)."""
    return {
        "timing": trade.settlement_date,
        "price": trade.price,
        "quantity": trade.quantity,
        "missing": trade.trade_id,
        "duplicate": trade.trade_id,
    }[break_type]


# ---------------------------------------------------------------------------
# Writing output
# ---------------------------------------------------------------------------

def _write_csv(path: Path, trades: list[Trade]):
    fieldnames = list(Trade.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            writer.writerow(asdict(t))


def main():
    rng = random.Random(SEED)
    out_dir = Path(__file__).resolve().parents[3] / "data"
    out_dir.mkdir(exist_ok=True)

    base = _make_base_trades(rng)
    ledger, custodian, manifest = _inject_breaks(base, rng)

    _write_csv(out_dir / "ledger.csv", ledger)
    _write_csv(out_dir / "custodian.csv", custodian)
    (out_dir / "break_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Summary so we can read the result together.
    counts: dict[str, int] = {}
    sides: dict[str, int] = {}
    for m in manifest:
        counts[m["break_type"]] = counts.get(m["break_type"], 0) + 1
        sides[m["corrupted_side"]] = sides.get(m["corrupted_side"], 0) + 1

    print(f"ledger rows     : {len(ledger)}")
    print(f"custodian rows  : {len(custodian)}")
    print(f"breaks planted  : {len(manifest)}")
    print(f"by type         : {counts}")
    print(f"corrupted side  : {sides}")


if __name__ == "__main__":
    main()