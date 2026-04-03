"""
prob_engine.py — Probability Estimation Module
==============================================

This module estimates a probability-like signal for binary outcome markets
by combining multiple microstructure-derived inputs:

- Trade-based signals (recent VWAP)
- Price dynamics (EMA of observed prints)
- Order book information (depth imbalance)
- Short-term flow indicators

The goal is to produce a stable estimate that can be used for
market-making or directional strategies.
"""

import math
import time
import aiohttp
from dataclasses import dataclass

DATA_API = "https://data-api.polymarket.com"

# ── Parameters ──────────────────────────────────────────────────────────────
TRADE_HALFLIFE_S = 300.0
MIN_TRADES_CONFIDENT = 5
MIN_PRINTS_CONFIDENT = 3

DEPTH_INFLUENCE = 0.40
FLOW_INFLUENCE = 0.25

MIN_CONFIDENCE = 0.20

EDGE_SKEW_THRESHOLD = 3.0
EDGE_DIRECTIONAL_THRESHOLD = 15.0


@dataclass
class ProbEstimate:
    p_est: float = 0.5
    confidence: float = 0.0
    edge_cents: float = 0.0
    alpha_mode: str = "PASSIVE_MM"
    trade_vwap: float = 0.0
    n_trades: int = 0
    signal_breakdown: str = ""


async def fetch_recent_trades(
    session: aiohttp.ClientSession,
    condition_id: str,
    n: int = 20,
) -> list[dict]:
    """Fetch recent trades for a given market."""
    if not condition_id or not condition_id.startswith("0x"):
        return []

    try:
        async with session.get(
            f"{DATA_API}/trades",
            params={"market": condition_id, "limit": n},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as r:
            if r.status != 200:
                return []

            data = await r.json()
            if not isinstance(data, list):
                return []

            return [
                {
                    "price": float(t.get("price", 0.5)),
                    "size": float(t.get("size", 0)),
                    "ts": float(t.get("timestamp", 0)),
                    "side": t.get("side", ""),
                }
                for t in data
                if t.get("size") and float(t.get("size", 0)) > 0
            ]
    except Exception:
        return []


def _exp_weight(ts: float, now: float) -> float:
    """Exponential decay weighting for recent trades."""
    age = max(0.0, now - ts)
    return math.exp(-age / TRADE_HALFLIFE_S)


def estimate_probability(
    trades: list[dict],
    print_ema: float,
    n_prints: int,
    trade_imbalance: float,
    bid_l1_size: float,
    ask_l1_size: float,
    current_bid: float,
    current_ask: float,
) -> ProbEstimate:
    """
    Combine multiple signals into a probability-like estimate.
    """

    mid = (current_bid + current_ask) / 2
    half_spread = (current_ask - current_bid) / 2
    now = time.time()

    signals = []
    details = []

    # ── Trade VWAP ──────────────────────────────────────────────────────────
    n_tr = 0
    vwap = 0.0

    if trades:
        w_sum = sum(t["size"] * _exp_weight(t["ts"], now) for t in trades)

        if w_sum > 0:
            vwap = sum(
                t["price"] * t["size"] * _exp_weight(t["ts"], now)
                for t in trades
            ) / w_sum

            n_tr = len(trades)
            w_vwap = 2.0 * min(1.0, n_tr / MIN_TRADES_CONFIDENT)

            signals.append((vwap, w_vwap))
            details.append(f"vwap={vwap:.3f}")

    # ── Print EMA ───────────────────────────────────────────────────────────
    if n_prints >= 2 and 0.02 < print_ema < 0.98:
        w_print = 1.5 * min(1.0, n_prints / MIN_PRINTS_CONFIDENT)
        signals.append((print_ema, w_print))
        details.append(f"print={print_ema:.3f}")

    # ── Book depth imbalance ────────────────────────────────────────────────
    total_depth = bid_l1_size + ask_l1_size

    if total_depth > 0 and half_spread > 0:
        depth_ratio = (bid_l1_size - ask_l1_size) / total_depth
        book_price = mid + depth_ratio * half_spread * DEPTH_INFLUENCE

        signals.append((book_price, 0.6))
        details.append(f"depth={depth_ratio:+.2f}")

    # ── Anchor ──────────────────────────────────────────────────────────────
    signals.append((mid, 0.4))

    # ── Combine signals ─────────────────────────────────────────────────────
    total_w = sum(w for _, w in signals)
    p_raw = sum(p * w for p, w in signals) / total_w

    # ── Flow adjustment ─────────────────────────────────────────────────────
    flow_adj = trade_imbalance * half_spread * FLOW_INFLUENCE
    p_est = max(0.03, min(0.97, p_raw + flow_adj))

    if abs(flow_adj) > 1e-4:
        details.append(f"flow={flow_adj*100:+.2f}c")

    # ── Confidence ──────────────────────────────────────────────────────────
    non_anchor_w = total_w - 0.4
    confidence = round(min(1.0, non_anchor_w / 3.5), 3)

    # ── Edge & mode ─────────────────────────────────────────────────────────
    edge_cents = round((p_est - mid) * 100, 2)

    if confidence < MIN_CONFIDENCE or abs(edge_cents) < EDGE_SKEW_THRESHOLD:
        mode = "PASSIVE_MM"
    elif abs(edge_cents) < EDGE_DIRECTIONAL_THRESHOLD:
        mode = "SKEWED_MM"
    else:
        mode = "DIRECTIONAL"

    return ProbEstimate(
        p_est=round(p_est, 4),
        confidence=confidence,
        edge_cents=edge_cents,
        alpha_mode=mode,
        trade_vwap=round(vwap, 4),
        n_trades=n_tr,
        signal_breakdown=" | ".join(details) if details else "anchor_only",
    )
