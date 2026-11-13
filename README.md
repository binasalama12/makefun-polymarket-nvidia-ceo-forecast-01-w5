# polymarket-nvidia-ceo-forecast-01 entry — Owl (w5)

A forecaster for Polymarket's "Next CEO of NVIDIA?" event. `main.py`
(Python 3.13, standard library only) reads one case from the input
directory and writes, for each market and each of the 240 minutes
after the checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%) of
that minute's price to the output directory.

## Model

The purist variant:

- **Median = persistence.** The last observed price.
- **Full memory:** the market's entire history in the case (up to
  72 hours) feeds the empirical change table (anchors 1, 5, 15,
  30, 60, 120, 240, interpolated in sqrt(h)) and the sigma
  estimate (mean absolute one-minute change, floor 0.001, cap
  0.005).
- **Spread = a 35/65 blend** leaning Gaussian: the full-history
  empirical table mixes several volatility regimes, so it advises
  the Gaussian fan rather than leading it.
- **No regime adjustment**, deliberately: regime scaling would
  double-count recent moves already inside the empirical
  distribution.
- A market with too little history borrows the median sigma of the
  case's established markets.

Quantiles are clamped to [0, 1], kept non-decreasing and rounded to
6 decimals. Deterministic; well under a second per case.

Backtested on the kit's practice inputs (later cases as ground
truth), this scores a mean pinball loss of about 0.388 against the
house baseline's 0.399 on the same folds.

## License

MIT — see `LICENSE`.
