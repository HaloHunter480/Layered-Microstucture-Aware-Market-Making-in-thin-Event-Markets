"""
vpin.py — Volume-synchronized Probability of Informed Trading (VPIN)
====================================================================

VPIN estimates order flow toxicity using volume-time instead of clock-time.

Key idea:
- Markets move in volume bursts, not uniform time.
- VPIN captures whether recent flow is one-sided (informed) or balanced (noise).

Interpretation:
    VPIN < 0.25 → CLEAN (uninformed flow, high opportunity)
    0.25–0.45   → NORMAL
    0.45–0.65   → ELEVATED (reduce size)
    > 0.65      → TOXIC (avoid trading)

References:
- Easley, López de Prado, O’Hara (2012)
"""

import numpy as np
from collections import deque


class VPIN:
    """
    Volume-synchronized Probability of Informed Trading.

    Steps:
    1. Split traded volume into equal-sized buckets
    2. Classify buy vs sell volume using bulk classification
    3. Compute imbalance per bucket
    4. Apply exponential decay weighting
    """

    def __init__(
        self,
        bucket_size: float = 0.2,
        window: int = 18,
        min_buckets: int = 3,
        decay_lambda: float = 0.1,
    ):
        self.bucket_size = float(bucket_size)
        self.window = max(2, int(window))
        self.min_buckets = max(1, int(min_buckets))
        self.decay_lambda = max(0.0, float(decay_lambda))

        # Current bucket state
        self.current_buy = 0.0
        self.current_sell = 0.0
        self.current_total = 0.0

        # Completed buckets
        self.bucket_imbalances = deque(maxlen=self.window)

        # For volatility estimation
        self.price_history = deque(maxlen=100)

    # ───────────────────────────────────────────────────────────────
    # Bulk Volume Classification
    # ───────────────────────────────────────────────────────────────
    def bulk_classify(self, open_p: float, close_p: float, volume: float):
        """
        Estimate buy/sell volume using price movement.

        Uses a smooth sigmoid approximation instead of hard classification.
        """
        if len(self.price_history) < 2:
            buy_frac = 0.5
        else:
            returns = np.diff(np.array(self.price_history))
            sigma = np.std(returns)

            if sigma > 1e-8:
                z = (close_p - open_p) / sigma
                buy_frac = 1 / (1 + np.exp(-z))  # logistic approximation
            else:
                buy_frac = 0.5 if close_p >= open_p else 0.5

        return volume * buy_frac, volume * (1 - buy_frac)

    # ───────────────────────────────────────────────────────────────
    # Update
    # ───────────────────────────────────────────────────────────────
    def update(self, open_p: float, close_p: float, volume: float):
        """Update VPIN with a new price bar."""
        self.price_history.append(close_p)

        buy_vol, sell_vol = self.bulk_classify(open_p, close_p, volume)

        remaining_vol = volume
        remaining_buy = buy_vol
        remaining_sell = sell_vol

        while remaining_vol > 0:
            space = self.bucket_size - self.current_total
            fill = min(space, remaining_vol)

            frac = fill / max(volume, 1e-12)

            self.current_buy += remaining_buy * frac
            self.current_sell += remaining_sell * frac
            self.current_total += fill

            remaining_vol -= fill
            remaining_buy *= (1 - frac)
            remaining_sell *= (1 - frac)

            # Bucket completed
            if self.current_total >= self.bucket_size:
                imbalance = abs(self.current_buy - self.current_sell) / self.bucket_size
                self.bucket_imbalances.append(imbalance)

                self.current_buy = 0.0
                self.current_sell = 0.0
                self.current_total = 0.0

    # ───────────────────────────────────────────────────────────────
    # Core Calculation
    # ───────────────────────────────────────────────────────────────
    def _weighted_mean(self, values):
        """Exponential decay weighting toward recent buckets."""
        arr = np.asarray(values, dtype=np.float64)

        if len(arr) == 0:
            return 0.5

        if self.decay_lambda == 0:
            return float(np.mean(arr))

        n = len(arr)
        ages = (n - 1) - np.arange(n)
        weights = np.exp(-self.decay_lambda * ages)

        return float(np.sum(arr * weights) / np.sum(weights))

    # ───────────────────────────────────────────────────────────────
    # Properties
    # ───────────────────────────────────────────────────────────────
    @property
    def is_ready(self) -> bool:
        return len(self.bucket_imbalances) >= self.min_buckets

    @property
    def vpin(self) -> float:
        """Current VPIN value (0–1)."""
        if not self.is_ready:
            return 0.5
        return self._weighted_mean(list(self.bucket_imbalances))

    @property
    def regime(self) -> str:
        """Simple regime classification."""
        v = self.vpin
        if v < 0.25:
            return "CLEAN"
        elif v < 0.45:
            return "NORMAL"
        elif v < 0.65:
            return "ELEVATED"
        else:
            return "TOXIC"

    @property
    def action(self) -> str:
        """Suggested trading behavior."""
        r = self.regime
        if r == "CLEAN":
            return "Full aggression"
        elif r == "NORMAL":
            return "Standard trading"
        elif r == "ELEVATED":
            return "Reduce size"
        else:
            return "Stop trading"

    def market_state(self) -> dict:
        """Return full structured output."""
        return {
            "vpin": round(self.vpin, 4),
            "regime": self.regime,
            "action": self.action,
            "buckets": len(self.bucket_imbalances),
        }
