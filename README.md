**Microstucture Aware Market Making in Thin Event Markets**
## Summary

This project develops a microstructure-aware trading system for binary options markets (Polymarket), focusing on:

- Empirical probability estimation (no Black-Scholes assumptions)
- Order flow toxicity detection (VPIN, Hawkes processes)
- Risk-adjusted position sizing (Kelly with uncertainty)
- Realistic execution modeling

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper Trading](https://img.shields.io/badge/status-paper%20trading-green)](https://github.com)

## 📊 Project Overview

This system trades 5-minute Bitcoin binary options on Polymarket using:
- **Empirical probability surfaces** with walk-forward validation & regime stratification
- **Kelly Criterion** (Monte Carlo uncertainty-adjusted) for position sizing
- **Hold-to-expiry precision mode** — enter only when model is confident (prob ∉ [0.38, 0.62])
- **Regime classification** (EWMA volatility + momentum)
- **Coinbase price feed** — matches Polymarket settlement source for minimal basis risk

---

## 🎓 Academic Documentation
Detailed mathematical framework and derivations are available in:

**[MATHEMATICAL_MODELS.md](MATHEMATICAL_MODELS.md)** 

This includes:
- Empirical probability modeling
- Kelly criterion and position sizing
- Hawkes processes for trade clustering
- VPIN for order flow toxicity
- Market microstructure modeling
- Execution and risk considerations
---

## 🏗️ Architecture
```
The system follows a modular pipeline for signal generation and execution:

1. Data Layer
   - Real-time order book and trade data
   - External price feed (Coinbase)

2. Signal Layer
   - Empirical probability estimation
   - Order flow signals (imbalance, momentum)
   - Toxic flow detection (VPIN, Hawkes intensity)

3. Decision Layer
   - Alpha classification (passive / skewed / directional)
   - Confidence filtering
   - Position sizing (Kelly with constraints)

4. Execution Layer
   - Limit and market order placement
   - Slippage and fill modeling
   - Latency-aware execution

5. Risk Layer
   - Position limits
   - Drawdown protection
   - Trade filtering in high-toxicity regimes
```

---

## 🔬 Mathematical Models

### 1. Empirical Probability Surface

Instead of Black-Scholes assumptions, we build probability surfaces from historical data:

```
P(UP | Δ, T) = historical_frequency(UP | price_diff=Δ, time_remaining=T)

where:
Δ = (BTC_current - Strike) / Strike (percentage difference)
T = Time remaining in 5-minute window (seconds)
```

**Training Data**: 45,000 1-minute observations (70% train / 30% OOS validation)

**v6 Enhancements**:
- Walk-forward: surface built from train only, OOS win rate on confident predictions (target ≥60%)
- Regime-stratified: bins by (pct_diff, time_remaining, regime); regime used only if OOS-validated
- Mud zone: no trade when prob ∈ [0.38, 0.62]

### 2. Kelly Criterion Position Sizing

Optimal bet sizing for maximizing logarithmic utility:

```python
f* = (p·b - q) / b  # Full Kelly
f_actual = 0.08 · f* · confidence  # Fractional Kelly (8%)

where:
p = Estimated win probability
b = Odds (payout/stake - 1)
q = 1 - p
confidence = weighted combination of sample size, edge magnitude, spread, toxic risk
```

### 3. Toxic Flow Detection

Multi-faceted detection system:

**Hawkes Process** (Volume Clustering):
```
λ(t) = μ + α·∑ exp(-β(t - t_i))
```

**VPIN** (Order Flow Toxicity):
```
VPIN = |Buy_Volume - Sell_Volume| / Total_Volume
Threshold: 0.80 (80% one-sided flow over 10s window)
```

**Action**: Cancel all resting orders immediately if toxic flow detected.

### 4. Asymmetric Market Making

Dynamic quote skewing based on order flow:

```python
If buying UP and flow is bullish (+0.8):
    Place bid LOWER (defensive - avoid momentum)
If buying UP and flow is bearish (-0.8):
    Place bid HIGHER (aggressive - flow in our favor)

skew = -sign(side) × pressure × 0.04  # Max 4% adjustment
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.11+
pip install -r requirements.txt
```

### Installation

```bash
git clone https://github.com/yourusername/polymarket-arbitrage.git
cd polymarket-arbitrage
pip install -r requirements.txt
```

### Configuration

Edit the configuration constants in `professional_strategy.py`:

```python
# Position sizing
BANKROLL = 500.0              # Starting bankroll
KELLY_FRACTION = 0.08         # 8% of full Kelly
MIN_BET_SIZE = 4.0           
MAX_BET_SIZE = 30.0

# Execution thresholds
MIN_MAKER_EDGE = 0.008        # 0.8% for limit orders
MIN_TAKER_EDGE = 0.045        # 4.5% for market orders

# Risk management
MAX_TRADES_PER_WINDOW = 2
SIGNAL_COOLDOWN = 25.0        # Seconds between signals

# Reality checks
WINDOW_INIT_DELAY = 15.0      # Wait 15s after window opens
MIN_TOB_VOLUME = 50.0         # Minimum $50 at best bid/ask
MAX_REALISTIC_EDGE = 0.15     # Cap edge at 15%
```

### Run Paper Trading

```bash
# Professional strategy (full research stack)
python3 professional_strategy.py

# Empirical strategy (simpler version)
python3 btc_empirical_bot.py

# Backtest (300-trade simulation)
python3 backtest_300.py
```

### Live Trading

```bash
# Live trading (v6 — hold-to-expiry, Coinbase feed)
# Requires: .env with POLY_PRIVATE_KEY, POLY_PROXY_ADDRESS
python3 live_test_2usd.py
```

**Requirements**:
1. `.env` file with `POLY_PRIVATE_KEY`, `POLY_PROXY_ADDRESS`
2. USDC + POL on Polygon (for gas)
3. Polymarket proxy wallet approved

See `REALITY_CHECK.md` for live trading challenges.

---

## 📊 Features

### ✅ Core Features
- [x] Real-time multi-exchange data aggregation
- [x] Empirical probability calibration from 45 days of BTC data
- [x] Kelly Criterion position sizing with confidence adjustments
- [x] Hawkes process volume clustering detection
- [x] VPIN toxic flow measurement
- [x] Kyle's Lambda market impact estimation
- [x] Asymmetric quote skewing based on order flow
- [x] Pre-emptive order cancellation on toxic flow
- [x] Realistic fill simulation (40% maker, 1.5% taker slippage)
- [x] Order book reality checks (15s delay, $50 min volume, 15% edge cap)

### v6 Live System Additions
- [x] Live order execution (`live_test_2usd.py`)
- [x] Regime classification (EWMA volatility + momentum)
- [x] Walk-forward empirical validation (OOS win rate on confident preds)
- [x] Empirical calibration tracking (Brier, reliability diagram)
- [x] Coinbase price feed (matches Polymarket settlement)
- [x] Hold-to-expiry precision mode (no mid-window sells)

### 🚧 Limitations
- [ ] Portfolio optimization across multiple markets
- [ ] Deep learning for pattern recognition
- [ ] Cross-market arbitrage

---

## 📈 Performance Analysis

### Paper Trading Results (75 minutes, 15 windows)

| Metric | Value |
|--------|-------|
| **Starting Bankroll** | $500 |
| **Ending Bankroll** | $1,238 |
| **Total Return** | +148% |
| **Total Trades** | 28 |
| **Win Rate** | 71.4% (20W / 8L) |
| **Maker/Taker Split** | 11 maker / 17 taker |
| **Maker Fill Rate** | ~40% (as simulated) |
| **Avg Trade Size** | $4-30 (Kelly-weighted) |

### Reality Check Effectiveness

1. **Window Initialization Delay**: ✅ No trades in first 15s of new windows
2. **Ghost Town Filter**: ✅ Blocked thin order books
3. **Edge Capping**: ✅ Limited position sizing on anomalous edges (prevented exponential growth)
4. **Toxic Flow Detection**: ✅ 2,700+ toxic events detected, orders cancelled appropriately
5. **Fill Simulation**: ✅ Realistic maker fill rates applied

### Expected Live Performance

Based on market microstructure theory and adverse selection:

| Metric | Paper | Expected Live |
|--------|-------|---------------|
| **Daily Return** | 200%+ | 0.5-2% |
| **Win Rate** | 71% | 48-52% |
| **Maker Fill Rate** | 40% | 20-40% |
| **Max Drawdown** | 8% | 15-30% |
| **Sharpe Ratio** | 2.5 | 0.1-0.7 |

**Why the gap?**
- Adverse selection: Informed traders pick off maker orders
- Fill degradation: Best prices get hit by HFT before us
- Slippage: Real markets have wider spreads than paper
- Latency: 50-200ms latency vs. instantaneous paper execution

---

## 🔬 Research & References

### Academic Papers

1. **Kelly Criterion**
   - Kelly, J.L. (1956). "A New Interpretation of Information Rate"
   
2. **Market Microstructure**
   - Kyle, A.S. (1985). "Continuous Auctions and Insider Trading"
   - O'Hara, M. (1995). "Market Microstructure Theory"
   
3. **Point Processes**
   - Hawkes, A.G. (1971). "Spectra of some self-exciting point processes"
   
4. **High-Frequency Trading**
   - Easley, D., López de Prado, M.M., O'Hara, M. (2011). "The Microstructure of the Flash Crash: Flow Toxicity, Liquidity Crashes and the Probability of Informed Trading"
   - Aldridge, I. (2013). "High-Frequency Trading: A Practical Guide to Algorithmic Strategies and Trading Systems"

### Books

- "Advances in Financial Machine Learning" by Marcos López de Prado
- "Algorithmic Trading: Winning Strategies and Their Rationale" by Ernie Chan
- "Market Microstructure in Practice" by Lehalle & Laruelle

---

## ⚠️ Risk Disclaimer

**THIS SOFTWARE IS FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.**

- **Not Financial Advice**: This project is a research implementation and should not be used for live trading without extensive testing and risk management.
- **No Guarantees**: Past performance (paper trading) does not guarantee future results.
- **High Risk**: Trading binary options and cryptocurrency involves substantial risk of loss.
- **Adverse Selection**: Real markets exhibit significant adverse selection that paper trading cannot capture.
- **Regulatory**: Ensure compliance with local regulations before trading.

**By using this software, you acknowledge that:**
- You understand the risks of algorithmic trading
- You will not hold the authors liable for any financial losses
- You will conduct thorough due diligence before any live trading
- You understand paper trading results are not indicative of live performance

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Live Trading Integration**
   - EIP-712 order signing
   - Real CLOB order submission
   - Position reconciliation

2. **Advanced Models**
   - Regime detection (Hidden Markov Models)
   - Deep learning for pattern recognition
   - Multi-asset portfolio optimization

3. **Infrastructure**
   - Lower latency execution (Rust/C++)
   - AWS deployment scripts
   - Monitoring & alerting

4. **Testing**
   - Unit tests for mathematical models
   - Backtesting framework
   - Monte Carlo simulation

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/polymarket-arbitrage.git
cd polymarket-arbitrage

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests (if implemented)
pytest tests/
```

---

## 📄 License

MIT License

Copyright (c) 2026 Harjot

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 📧 Contact

- **GitHub**: [@HaloHunter480](https://github.com/HaloHunter480)
- **Email**:harjot.quant@gmail.com

For academic inquiries or collaboration opportunities, please reach out via email.

---

## 🙏 Acknowledgments

- **Polymarket** for providing prediction market infrastructure
- **Binance** for real-time BTC price data
- **Academic Community** for foundational research in market microstructure
- **Open Source Community** for Python libraries (asyncio, numpy, scipy)

---

## 📚 Additional Resources

- [Polymarket API Documentation](https://docs.polymarket.com/)
- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [Kelly Criterion Calculator](https://www.albionresearch.com/kelly/)
- [Market Microstructure Blog](https://mechanicalmarkets.wordpress.com/)

---

**Built with ❤️ for quantitative finance research**

*Last Updated: February 2026*
