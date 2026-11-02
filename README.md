# AKF / HJB-PINN Statistical-Arbitrage Research

This repository contains a **synthetic research prototype** for pair-based
statistical arbitrage. It combines causal adaptive Kalman filtering, a
pair-level HJB/PINN inventory controller, and simplified
Avellaneda–Stoikov-style quote execution.

## Run the abstract-metric experiment

From the repository root, run:

```bash
python3 run_stress_ablation.py
```

By default this runs 250 matched simulations in each of three synthetic
conditions and writes:

- `results/stress_ablation_results.csv` — metrics for each path and controller
- `results/stress_ablation_summary.csv` — compact scenario/controller summary
- `paper/abstract_metrics.tex` — generated LaTeX table

Include the generated table in a LaTeX document located at the repository root
with `\input{paper/abstract_metrics.tex}`. To run a smaller smoke test:

```bash
python3 run_stress_ablation.py --simulations-per-scenario 20
```

To run tests:

```bash
python3 -m unittest discover -s tests -v
```

Install dependencies from `requirements.txt` if needed.

## Repository layout

- `pinn_research/` — model components and experiment implementations
- `tests/` — unit and integration tests
- `results/` — generated simulation CSVs
- `paper/` — generated LaTeX table fragment
- `docs/research-model.md` — model, equations, experiment methods, and caveats
- Root `run_*.py` files — small command-line entry points

## Research limitations

These are synthetic stress tests, not historical or live-market validation.
The current Monte Carlo studies one selected pair among three assets, not a
fully coupled multi-pair portfolio. The quote-opportunity/fill metric is a
simulator proxy, not a venue order-to-trade ratio. PINN residual measures
approximation of the modeled HJB, not empirical market accuracy. Nothing in
this repository establishes profitability or suitability for live trading.
