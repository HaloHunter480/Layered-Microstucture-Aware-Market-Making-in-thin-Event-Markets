# orderbook.py
# Microstructure-aware order book representation for L2 data.
# Provides signals derived from limit order book state and trade flow.

import time
import numpy as np
from collections import deque


class OrderBook:
    """
    Maintains a real-time L2 order book and computes
    microstructure signals from liquidity and trade flow.
    """

    def __init__(self, depth=20):
        self.bids = {}  # price -> size
        self.asks = {}  # price -> size
        self.depth = depth
        self.last_update_id = 0

        # Cumulative Volume Delta (net aggressive flow)
        self.cvd = 0.0
        self.cvd_history = deque(maxlen=500)

        # Recent trades for short-term analysis
        self.recent_trades = deque(maxlen=1000)

    def update(self, data: dict):
        """Process depth update (partial book)."""
        for bid in data.get('bids', []):
            price, size = float(bid[0]), float(bid[1])
            if size == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = size

        for ask in data.get('asks', []):
            price, size = float(ask[0]), float(ask[1])
            if size == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = size

    def add_trade(self, price: float, size: float, is_buyer_maker: bool):
        """
        Track aggressive trade flow.

        is_buyer_maker = True  → sell aggressor
        is_buyer_maker = False → buy aggressor
        """
        direction = -1 if is_buyer_maker else 1
        self.cvd += direction * size
        self.cvd_history.append(self.cvd)

        self.recent_trades.append({
            'price': price,
            'size': size,
            'direction': direction,
            'ts': time.time()
        })

    @property
    def best_bid(self) -> float:
        return max(self.bids.keys()) if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return min(self.asks.keys()) if self.asks else float('inf')

    @property
    def mid_price(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return self.best_ask - self.best_bid

    @property
    def spread_ratio(self) -> float:
        """Spread as fraction of mid price."""
        mid = self.mid_price
        return self.spread / mid if mid > 0 else 1.0

    def order_book_imbalance(self, levels=5) -> float:
        """
        Measures relative bid vs ask liquidity at top levels.

        OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        """
        if not self.bids or not self.asks:
            return 0.0

        # NOTE: Sorting each call is inefficient for production systems.
        sorted_bids = sorted(self.bids.keys(), reverse=True)[:levels]
        sorted_asks = sorted(self.asks.keys())[:levels]

        bid_vol = sum(self.bids.get(p, 0) for p in sorted_bids)
        ask_vol = sum(self.asks.get(p, 0) for p in sorted_asks)

        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    def depth_ratio(self, levels=10) -> float:
        """
        Ratio of bid depth to ask depth across N levels.
        """
        if not self.bids or not self.asks:
            return 1.0

        sorted_bids = sorted(self.bids.keys(), reverse=True)[:levels]
        sorted_asks = sorted(self.asks.keys())[:levels]

        bid_depth = sum(self.bids.get(p, 0) for p in sorted_bids)
        ask_depth = sum(self.asks.get(p, 0) for p in sorted_asks)

        return bid_depth / ask_depth if ask_depth > 0 else 1.0

    def buy_sell_ratio_recent(self, seconds=30) -> float:
        """
        Fraction of recent traded volume that was buy-side.
        """
        now = time.time()
        recent = [t for t in self.recent_trades if now - t['ts'] < seconds]

        if not recent:
            return 0.5

        buy_vol = sum(t['size'] for t in recent if t['direction'] == 1)
        total_vol = sum(t['size'] for t in recent)

        return buy_vol / total_vol if total_vol > 0 else 0.5

    def vwap_recent(self, seconds=60) -> float:
        """
        Volume-weighted average price of recent trades.
        """
        now = time.time()
        recent = [t for t in self.recent_trades if now - t['ts'] < seconds]

        if not recent:
            return self.mid_price

        total_pv = sum(t['price'] * t['size'] for t in recent)
        total_v = sum(t['size'] for t in recent)

        return total_pv / total_v if total_v > 0 else self.mid_price

    def cvd_momentum(self, window=50) -> float:
        """
        Normalized rate of change of cumulative volume delta.
        """
        if len(self.cvd_history) < window:
            return 0.0

        recent = list(self.cvd_history)[-window:]
        x = np.arange(len(recent))

        slope = np.polyfit(x, recent, 1)[0]
        std = np.std(recent)

        return slope / std if std > 0 else 0.0

    def large_order_detection(self, threshold_multiplier=3.0) -> dict:
        """
        Detect unusually large resting orders relative to average depth.
        """
        if not self.bids or not self.asks:
            return {'bid_wall': None, 'ask_wall': None}

        all_bid_sizes = list(self.bids.values())
        all_ask_sizes = list(self.asks.values())

        mean_bid = np.mean(all_bid_sizes)
        mean_ask = np.mean(all_ask_sizes)

        bid_wall = None
        ask_wall = None

        for price, size in self.bids.items():
            if size > threshold_multiplier * mean_bid:
                if bid_wall is None or size > bid_wall['size']:
                    bid_wall = {'price': price, 'size': size}

        for price, size in self.asks.items():
            if size > threshold_multiplier * mean_ask:
                if ask_wall is None or size > ask_wall['size']:
                    ask_wall = {'price': price, 'size': size}

        return {'bid_wall': bid_wall, 'ask_wall': ask_wall}
