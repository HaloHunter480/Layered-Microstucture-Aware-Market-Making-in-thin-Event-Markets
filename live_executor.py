"""
Execution Engine for Polymarket CLOB
====================================

Provides a structured interface for:
- Authenticated order submission via py-clob-client
- Basic risk management constraints
- Order tracking and latency measurement

This module is designed as a simplified, research-oriented
execution layer rather than a full production deployment.
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from collections import deque
from datetime import datetime

from dotenv import load_dotenv
from hexbytes import HexBytes
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    OrderArgs,
    MarketOrderArgs,
    OrderType,
    PartialCreateOrderOptions,
    BalanceAllowanceParams,
    AssetType,
)

load_dotenv()

# ─── Logging ────────────────────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("executor")
logger.setLevel(logging.INFO)

handler = logging.FileHandler(
    f"logs/executor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    success: bool
    order_id: str = ""
    token_id: str = ""
    side: str = ""
    price: float = 0.0
    size: float = 0.0
    cost: float = 0.0
    latency_ms: float = 0.0
    error: str = ""
    timestamp: float = 0.0


@dataclass
class Position:
    token_id: str
    side: str
    size: float
    avg_price: float
    cost: float
    timestamp: float


# ─── Risk Manager ───────────────────────────────────────────────────────────

class RiskManager:
    """
    Enforces simple trading constraints such as:
    - Maximum loss per session
    - Maximum exposure
    - Order frequency limits
    """

    def __init__(
        self,
        max_loss_per_session: float = 50.0,
        max_position: float = 200.0,
        max_orders_per_window: int = 3,
        max_order_size: float = 100.0,
        min_order_size: float = 1.0,
    ):
        self.max_loss = max_loss_per_session
        self.max_position = max_position
        self.max_orders_per_window = max_orders_per_window
        self.max_order_size = max_order_size
        self.min_order_size = min_order_size

        self.session_pnl = 0.0
        self.total_exposure = 0.0
        self.orders_this_window = 0
        self.current_window_id = 0

    def allow_trade(self, size: float, price: float) -> Tuple[bool, str]:
        window_id = int(time.time() // 300)
        if window_id != self.current_window_id:
            self.current_window_id = window_id
            self.orders_this_window = 0

        cost = size * price

        if cost < self.min_order_size:
            return False, "order too small"

        if cost > self.max_order_size:
            return False, "order too large"

        if self.total_exposure + cost > self.max_position:
            return False, "position limit exceeded"

        if self.session_pnl < -self.max_loss:
            return False, "max loss exceeded"

        if self.orders_this_window >= self.max_orders_per_window:
            return False, "too many orders in window"

        return True, "ok"

    def record_trade(self, result: TradeResult):
        if result.success:
            self.orders_this_window += 1
            self.total_exposure += result.cost


# ─── Executor ───────────────────────────────────────────────────────────────

class LiveExecutor:
    """
    Simplified execution interface for Polymarket.
    """

    def __init__(self, tick_size: str = "0.01"):
        self.tick_size = tick_size

        self.private_key = os.getenv("POLY_PRIVATE_KEY", "")
        self.api_key = os.getenv("POLY_API_KEY", "")
        self.api_secret = os.getenv("POLY_API_SECRET", "")
        self.api_passphrase = os.getenv("POLY_API_PASSPHRASE", "")

        self.client: Optional[ClobClient] = None
        self.connected = False

        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.latencies_ms = deque(maxlen=100)

    def connect(self) -> bool:
        if not self.private_key:
            logger.error("Missing private key")
            return False

        try:
            self.client = ClobClient(
                host="https://clob.polymarket.com",
                key=HexBytes(self.private_key),
                chain_id=137,
            )

            if self.api_key:
                creds = ApiCreds(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    api_passphrase=self.api_passphrase,
                )
                self.client.set_api_creds(creds)

            self.connected = True
            logger.info("Connected to Polymarket")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> TradeResult:

        if not self.client or not self.connected:
            return TradeResult(success=False, error="not connected")

        t0 = time.perf_counter_ns()

        try:
            price = max(0.01, min(0.99, round(price, 2)))
            size = round(size, 2)

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side,
                fee_rate_bps=0,
            )

            options = PartialCreateOrderOptions(tick_size=self.tick_size)

            response = self.client.create_and_post_order(order_args, options)

            t1 = time.perf_counter_ns()
            latency = (t1 - t0) / 1e6

            self.total_orders += 1
            self.latencies_ms.append(latency)

            order_id = ""
            success = False

            if isinstance(response, dict):
                order_id = response.get("orderID", "")
                success = bool(order_id)

            result = TradeResult(
                success=success,
                order_id=order_id,
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                cost=price * size,
                latency_ms=latency,
                timestamp=time.time(),
            )

            if success:
                self.successful_orders += 1
            else:
                self.failed_orders += 1
                result.error = str(response)

            return result

        except Exception as e:
            self.failed_orders += 1
            return TradeResult(
                success=False,
                error=str(e),
                latency_ms=(time.perf_counter_ns() - t0) / 1e6,
                timestamp=time.time(),
            )

    def metrics(self) -> str:
        avg_lat = sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0
        return f"orders={self.total_orders}, success={self.successful_orders}, avg_latency={avg_lat:.1f}ms"
