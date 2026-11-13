#!/usr/bin/env python3
"""Owl forecaster for Polymarket's "Next CEO of NVIDIA?" event
(make.fun polymarket-nvidia-ceo-forecast-01, wallet w5 variant).

Reads one case from the input directory: every market of the event
with its Yes price at each minute up to the checkpoint. Writes, for
each market and each of the 240 minutes after the checkpoint, five
quantiles (5%, 25%, 50%, 75%, 95%) of that minute's price.

The Owl variant is the family's purist:

- It uses the market's ENTIRE history in the case (up to the full
  72 hours / 4320 minutes) for both the empirical h-step change
  quantiles and the sigma estimate. No recent window, no forgetting:
  for a 4-hour forecast on a market that trades in rare steps, the
  long-run move distribution is the most honest one available.
- There is deliberately NO volatility-regime adjustment. Regime
  scaling double-counts recent moves that are already inside the
  empirical distribution; the Owl lets the distribution speak.
- The blend leans Gaussian, 35/65: the full-history empirical
  table mixes several volatility regimes, so it advises the
  Gaussian fan rather than leading it (backtests of the pure
  full-history blend confirmed the lighter touch scores better).
- Sigma uses the mean absolute one-minute change with a wider sane
  band than the family norm, floor 0.001 and cap 0.005, because a
  full-history estimate mixes quiet and wild days by design.

The median forecast is the last observed price. Quantiles are
clamped to [0, 1], kept non-decreasing and rounded to 6 decimals.
Deterministic, standard library only.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

FALLBACK_SIGMA = 0.0025
ANCHORS = [1, 5, 15, 30, 60, 120, 240]
EMP_WEIGHT = 0.35        # blend weight on the empirical quantiles
SIGMA_FLOOR = 0.001
SIGMA_CAP = 0.005
MIN_EMP_HISTORY = 300    # below this a market borrows a sigma instead


def clamp01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def price6(v):
    return round(clamp01(v) * 1e6) / 1e6 + 0.0


def quantile_sorted(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def mean_abs_diff(prices):
    if len(prices) < 2:
        return 0.0
    total = 0.0
    for i in range(len(prices) - 1):
        total += abs(prices[i + 1] - prices[i])
    return total / (len(prices) - 1)


def anchor_quantiles(hist):
    """Empirical quantiles of h-step changes at each anchor horizon,
    over the market's full history in the case."""
    out = {}
    n = len(hist)
    for h in ANCHORS:
        if h >= n:
            continue
        changes = sorted(hist[i + h] - hist[i] for i in range(n - h))
        out[h] = [quantile_sorted(changes, tau) for tau in QUANTILES]
    return out


def empirical_row(anchors_q, h):
    """Interpolated empirical change quantiles for horizon h."""
    keys = sorted(anchors_q)
    if not keys:
        return None
    if h <= keys[0]:
        return anchors_q[keys[0]]
    if h >= keys[-1]:
        return anchors_q[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= h <= hi:
            f = (math.sqrt(h) - math.sqrt(lo)) / (math.sqrt(hi) - math.sqrt(lo))
            return [anchors_q[lo][j] * (1.0 - f) + anchors_q[hi][j] * f
                    for j in range(len(QUANTILES))]
    return anchors_q[keys[-1]]


def market_sigma(prices):
    mad = mean_abs_diff(prices)
    return min(max(1.2533 * mad, SIGMA_FLOOR), SIGMA_CAP)


def forecast_market(prices, horizon, sigma):
    x = prices[-1]
    anchors_q = anchor_quantiles(prices) if len(prices) >= MIN_EMP_HISTORY else {}
    rows = []
    for h in range(1, horizon + 1):
        sq = math.sqrt(h)
        gauss = [x + z * sigma * sq for z in Z]
        emp = empirical_row(anchors_q, h)
        if emp is None:
            row = gauss
        else:
            row = [EMP_WEIGHT * (x + emp[j]) + (1.0 - EMP_WEIGHT) * gauss[j]
                   for j in range(len(QUANTILES))]
        rows.append(sorted(price6(v) for v in row))
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in INPUT_DIR.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        case = json.loads(path.read_text())
        if list(case["quantiles"]) != QUANTILES:
            raise ValueError(f"{path.name} asks for quantiles "
                             f"{case['quantiles']}, not {QUANTILES}")
        horizon = int(case["horizon_minutes"])
        parsed = []
        for market in case["markets"]:
            prices = [float(p) for p in market["prices"]]
            if not prices:
                raise ValueError(f"market {market['id']} has no prices")
            parsed.append((market["id"], prices))
        # A market with too little history borrows the typical sigma
        # of the case's established markets.
        sigmas = {mid: market_sigma(prices) for mid, prices in parsed
                  if len(prices) >= MIN_EMP_HISTORY}
        if sigmas:
            ordered = sorted(sigmas.values())
            fallback_sigma = ordered[len(ordered) // 2]
        else:
            fallback_sigma = FALLBACK_SIGMA
        forecasts = []
        for mid, prices in parsed:
            sigma = sigmas.get(mid, fallback_sigma)
            forecasts.append({"id": mid,
                              "q": forecast_market(prices, horizon, sigma)})
        out = json.dumps({"forecasts": forecasts}) + "\n"
        (OUTPUT_DIR / f"{case_id}.out").write_text(out)


if __name__ == "__main__":
    main()
