# Research Model: HJB-Embedded Multi-Asset Statistical Arbitrage

**Working title:** *Stochastic Risk Analysis for Multi-Asset Statistical Arbitrage with HJB-Embedded PINNs, Adaptive State Estimation, and AS Baselines*
**Status:** Staged prototype. The market-making HJB and the statistical-arbitrage strategy are separate components; a unified portfolio controller has not yet been derived.

## Step 1 — Specify the mathematical control model

This step defines what the trading model observes, controls, and optimizes. It does not yet implement the simulator, Kalman filter, or PINN.

### 1.1 Research question and scope

**Research direction:** Evaluate a model that selects and trades statistical relationships across assets, accounting for execution, portfolio risk, and heavy-tail/regime-change stress. The draft's concrete selection concept is to evaluate candidate opportunities and allocate among them across a multi-asset portfolio.

Statistical arbitrage and market making are distinct strategies. A pair strategy takes offsetting long and short positions based on a relative-value signal; the AS model posts bids and asks to earn spread while controlling inventory. They may share execution/risk infrastructure, but their signals and objectives must be evaluated separately. For the first explicit stat-arb signal, this prototype uses training-window cointegration and held-out spread mean reversion. A true integration with HJB/AKF remains future work; the current single-asset HJB does not select pairs.

The components are separated into:

1. Cointegrated-pairs selector and relative-value portfolio: the initial statistical-arbitrage implementation.
2. AS quoting policy: market-making baseline; not a statistical-arbitrage selector.
3. HJB policy and PINN: a separate market-making control prototype, not yet integrated with pair selection.
4. Adaptive Kalman filter (AKF): state estimator; its impact on statistical-arbitrage decisions still needs an ablation.

### 1.2 State, controls, and quotes

For \(n\) assets, define the state

\[
x=(c,q,X), \qquad q=(q_1,\ldots,q_n)\in\mathbb Z^n,
\]

where \(c\) is cash, \(q_i\) is inventory of asset \(i\), and \(X_i=\log S_i\) is its log mid-price, so \(S_i=e^{X_i}>0\). Let \(T\) be the end of the trading horizon.

For each asset \(i\), the control is a pair of quote offsets

\[
u_i=(\delta_i^b,\delta_i^a), \qquad \delta_i^b,\delta_i^a\ge 0,
\]

which gives bid and ask prices

\[
p_i^b=S_i-\delta_i^b,\qquad p_i^a=S_i+\delta_i^a.
\]

Require \(p_i^b\ge 0\), and impose documented inventory bounds \(q_i^{\min}\le q_i\le q_i^{\max}\). A bid is disabled at \(q_i=q_i^{\max}\); an ask is disabled at \(q_i=q_i^{\min}\). If short selling is not allowed, set \(q_i^{\min}=0\). Cash, order-size, and quote-distance limits can also be added, but must be fixed and applied identically to both policies.

### 1.3 Price dynamics and stress scenarios

The baseline continuous-price dynamics are correlated log-price diffusions:

\[
dX_t=\mu\,dt+L\,dW_t,\qquad LL^\top=\Sigma,
\]

where \(W_t\) is an \(n\)-dimensional Brownian motion, \(\mu\) is the log-price drift vector, and \(\Sigma\) is the covariance matrix per unit time. These parameters must be estimated from training data or set explicitly for simulation; they are not specified by the research proposal.

To represent jump stress, add a compound Poisson process:

\[
dX_t=\mu\,dt+L\,dW_t+dJ_t,\qquad
J_t=\sum_{m=1}^{N_t}Y_m.
\]

Here \(N_t\) has declared intensity \(\rho\), and the jump vector \(Y_m\) has a declared distribution. A heavy-tailed choice, such as a Pareto-tailed jump magnitude with a specified tail index and cutoff, must be documented along with jump direction and cross-asset dependence. The resulting HJB includes a nonlocal jump term. Without that term, the model is diffusion-only and cannot claim to solve a jump-risk HJB.

For a regime-change stress test, let \(\mu,\Sigma,\rho\), or execution parameters change according to a declared deterministic schedule or simulated latent regime. This creates a regime-change test; it does not mean the AKF has detected the regime. Detection requires a separately defined and evaluated regime estimator.

### 1.4 Order fills and cash/inventory transitions

As a starting execution model, let bid and ask fills for each asset be conditionally independent Poisson arrivals with intensities

\[
\lambda_i^b(\delta_i^b)=A_i^b e^{-k_i^b\delta_i^b},\qquad
\lambda_i^a(\delta_i^a)=A_i^a e^{-k_i^a\delta_i^a},
\]

where \(A_i^{b/a}>0\) and \(k_i^{b/a}>0\). These parameters describe fill frequency and sensitivity to quote distance. They must be calibrated on training data or explicitly designated simulation parameters. The exponential form is a modeling assumption, not a universal order-book law.

For unit-size fills and per-share fees \(f_i^b,f_i^a\), a bid fill changes

\[
(c,q)\longmapsto(c-p_i^b-f_i^b,\ q+e_i),
\]

and an ask fill changes

\[
(c,q)\longmapsto(c+p_i^a-f_i^a,\ q-e_i),
\]

where \(e_i\) is the \(i\)-th unit vector. For order size \(m_i\), multiply the quote cash flow and fee by \(m_i\), and change inventory by \(m_ie_i\). This model must state how partial fills, cancellations, and insufficient cash are handled; the simplest first simulator can use unit fills, bounded inventory, and no leverage.

If fees are already applied in these cash transitions, do not subtract them again as a running cost in the HJB.

### 1.5 Objective

At the horizon, mark remaining inventory to the mid-price:

\[
W_T=c_T+q_T^\top S_T.
\]

Use exponential utility as the initial risk-sensitive objective:

\[
U(W_T)=-e^{-\eta W_T},\qquad \eta>0,
\]

and maximize expected terminal utility. The parameter \(\eta\) controls risk aversion and must be fixed before comparing strategies. Because the scale of wealth affects numerical behavior, a practical implementation may instead apply the utility to wealth relative to initial capital; this normalization must be declared and used consistently.

This objective does not guarantee positive profit. Report P&L and risk metrics separately from expected utility. If a terminal inventory penalty or running risk penalty is later added, it changes the objective and must be identified as a model extension.

### 1.6 Multi-asset HJB equation

Let \(V(t,c,q,X)\) be the maximum expected terminal utility from state \((c,q,X)\). For unit fills, define the bid and ask continuation increments:

\[
\begin{aligned}
\Delta_i^b V
&=V(t,c-p_i^b-f_i^b,q+e_i,X)-V(t,c,q,X),\\
\Delta_i^a V
&=V(t,c+p_i^a-f_i^a,q-e_i,X)-V(t,c,q,X).
\end{aligned}
\]

The diffusion generator acting on \(V\) is

\[
\mathcal D V
=\mu^\top\nabla_X V+
\frac12\operatorname{Tr}\!\left(\Sigma\nabla^2_{XX}V\right).
\]

For the compound Poisson log-price jumps above, its jump contribution is

\[
\mathcal J V
=\rho\,\mathbb E_Y\!\left[
V(t,c,q,X+Y)-V(t,c,q,X)
\right].
\]

The proposed backward HJB is

\[
\boxed{
\partial_t V+\mathcal D V+\mathcal J V+
\sup_{u\in\mathcal U(q,X)}
\sum_{i=1}^{n}
\left[
\lambda_i^b(\delta_i^b)\Delta_i^bV+
\lambda_i^a(\delta_i^a)\Delta_i^aV
\right]=0
}
\]

with terminal condition

\[
V(T,c,q,X)=-\exp\!\left(
-\eta\left[c+q^\top e^X\right]\right).
\]

Here \(e^X\) is taken componentwise, and \(\mathcal U(q,X)\) is the feasible set of quotes respecting nonnegative bid prices, inventory limits, and any additional operational limits. At an inventory boundary, the corresponding order that would breach the bound is excluded (equivalently, its intensity is zero). With no leverage, also exclude a bid fill if its cash transition would violate the cash constraint. The jump expectation is evaluated over the declared jump distribution.

This equation assumes fills arrive independently conditional on the current state and chosen quote offsets. If order arrivals are dependent across assets, have delayed execution, or depend on queue position, the transition model and HJB generator must be changed to represent those mechanics.

### 1.7 What this model does and does not claim

- The HJB optimizes the stated expected terminal utility under the assumed dynamics; it does not establish that the assumptions match a real exchange.
- The heavy-tail case is modeled through a specified jump distribution, not by labeling ordinary Gaussian diffusion as heavy-tailed.
- The AKF may estimate specified latent inputs, but does not inherently remove outliers or identify bull/bear regimes.
- A PINN can approximate this HJB solution. Low PDE residual does not prove a good trading policy; held-out financial evaluation is still required.
- Multi-asset state and correlated price risk do not, on their own, constitute statistical arbitrage.

### 1.8 Inputs still required before numerical results

Do not invent empirical values. Record the asset universe, price-data source and interval, calibration dates, \(\mu,\Sigma\), jump distribution and intensity, fill parameters \(A_i^{b/a},k_i^{b/a}\), fees, order sizes, inventory/cash limits, and \(\eta\). If those inputs are unavailable, use a clearly labeled synthetic simulator and publish its parameters and random seeds.

## Step 2 — Simulator/data interface and AS baseline

**Implemented in** [`initial.py`](./initial.py), with focused modules and tests:

- Data validation and simulation: [`market_data.py`](./market_data.py), tested in [`test_market_data.py`](./test_market_data.py).
- AS model and policy: [`baselines.py`](./baselines.py), tested in [`test_baselines.py`](./test_baselines.py).
- Execution and portfolio bookkeeping: [`backtest.py`](./backtest.py), tested in [`test_backtest.py`](./test_backtest.py).
- Adaptive state estimation: [`filters.py`](./filters.py), tested in [`test_filters.py`](./test_filters.py).
- Shared array aliases live in [`model_types.py`](./model_types.py). [`initial.py`](./initial.py) remains as a compatibility re-export for the original imports; new code should import from the focused modules directly.

- `MarketData` accepts aligned mid-price arrays or loads long-form CSV data with `timestamp`, `symbol`, and `mid_price` columns. It rejects duplicate or missing timestamp/asset rows, invalid prices, and non-increasing timestamps.
- `simulate_gbm` generates a reproducible correlated log-price diffusion from explicit drift, covariance, time step, and seed inputs. This is a baseline simulator; it does not yet include jumps or regime changes.
- `AvellanedaStoikov` computes inventory-aware, non-crossing bid/ask quotes with configured inventory bounds. Per-asset parameters are tied to an explicit symbol order.
- `run_backtest` applies one unit-order fill opportunity per side and asset per interval, samples fill/no-fill from the exponential-intensity model, applies per-unit fees, enforces inventory bounds, and prevents purchases that exceed available cash. Results include cash, inventory, equity, quotes, submitted-side flags, and fills.

The initial execution simulator is intentionally discrete and simplified: at most one unit can fill per side and asset per interval; there are no queue positions, partial fills, cancellations, leverage, or cross-asset-dependent fill events. Its outputs are simulation results under declared assumptions, not evidence of live-market performance. `ASParameters.volatility` is the per-square-root-time volatility used by the quoting policy; `dt` and the input price interval must use consistent time units.

## Step 3 — Adaptive Kalman state estimation

**Implemented in** [`filters.py`](./filters.py), with validation in [`test_filters.py`](./test_filters.py).

- `simulate_log_price_observations` creates reproducible noisy log-price observations from simulated market truth using a declared measurement-noise variance and seed.
- `AdaptiveKalmanFilter` estimates each asset's latent log price online using independent scalar filters. It uses a random-walk state transition, fixed measurement variance, and adapts process variance using innovation covariance matching:

\[
Q_k=(1-d)Q_{k-1}
+d\left(K_k^2 e_k^2+P_{k|k}-P_{k-1|k-1}\right),
\]

where \(e_k\) is the one-step innovation, \(K_k\) the Kalman gain, \(P\) the state-estimate variance, and \(d\) the adaptation rate. The implementation floors \(Q_k\) at a positive configured minimum to prevent invalid negative process variance. It emits filtered log prices, their exponentiated prices, innovations, and the process-variance history; each estimate uses only current and earlier observations.
- Validation compares filtered log-price RMSE against the unfiltered noisy observations on a seeded synthetic path with known latent prices. This measures state-estimation error only; it is not evidence that the filter improves trading returns.

This is a simple adaptive random-walk filter, not an EKF/SKF comparison, a heavy-tail-robust filter, or a bull/bear regime detector. The measurement variance is fixed, assets are filtered independently, and adaptation parameters need calibration. The filter has not yet been wired into the AS backtest as a separately isolated policy input; doing so requires keeping observed execution prices distinct from estimated prices used for quotes.

## Step 4 — Reduced HJB reference solution

**Implemented in** [`hjb.py`](./hjb.py), with numerical checks in [`test_hjb.py`](./test_hjb.py).

Before attempting the full multi-asset PINN, this step solves a deliberately reduced one-asset problem. Assume an arithmetic Brownian mid-price \(dS_t=\sigma\,dW_t\), finite inventory \(q\in\{q_{\min},\ldots,q_{\max}\}\), unit fills, exponential arrival intensity \(\lambda(\delta)=A e^{-k\delta}\), per-unit fee \(f\), and exponential utility \(U(w)=-e^{-\eta w}\). Quote offsets are constrained to \([0,\delta_{\max}]\). There are no price jumps, drift, log-price dynamics, or cross-asset effects in this reference problem.

With wealth \(w=c+qS\), write

\[
V(t,c,q,S)=-e^{-\eta(c+qS)}v_q(t).
\]

A fill at either quote changes marked wealth by \(\delta-f\). The diffusion generator contributes \(\frac12\eta^2\sigma^2q^2v_q\). Define the transformed fill operator

\[
\mathcal F_q(v)=
\inf_{\delta^b,\delta^a\in[0,\delta_{\max}]}
\left[
\lambda(\delta^b)\left(e^{-\eta(\delta^b-f)}v_{q-1}-v_q\right)
+\lambda(\delta^a)\left(e^{-\eta(\delta^a-f)}v_{q+1}-v_q\right)
\right],
\]

omitting the bid term at \(q_{\min}\) and the ask term at \(q_{\max}\), so fills cannot leave the finite inventory state space. The reduced backward HJB is

\[
\partial_t v_q+\frac12\eta^2\sigma^2q^2v_q+\mathcal F_q(v)=0,
\qquad v_q(T)=1.
\]

The optimal offset for either available side, with its adjacent inventory value \(v_{\mathrm{next}}\), is the clipped analytical minimizer

\[
\delta^*=
\operatorname{clip}_{[0,\delta_{\max}]}
\left[
f+\frac{1}{\eta}\log\!\left(\frac{(k+\eta)v_{\mathrm{next}}}{k v_q}\right)
\right].
\]

The solver integrates the transformed inventory ODE backward from the terminal condition using SciPy's DOP853 integrator, then evaluates the HJB residual on the returned time/inventory grid. It reconstructs the original value function through `InventoryHJBResult.value(cash, mid_price)`. Inventory edges use a no-outward-fill truncation, not a reflecting or extrapolated value boundary.

The residual is a finite-difference diagnostic of this numerical reference, not a trading-performance metric and not a PINN result. On the tested synthetic configuration (11 inventory states, 201 time nodes), the maximum absolute residual at interior times is below \(10^{-6}\). This verifies the implementation on that configuration, not the empirical validity of its assumptions.

## Step 5 — PINN approximation of the reduced HJB

**Implemented in** [`pinn.py`](./pinn.py), tested in [`test_pinn.py`](./test_pinn.py).

The PINN approximates the positive transformed factor \(v(t,q)\) from Step 4, not the full multi-asset value function. Its inputs are normalized time and discrete inventory; a softplus output enforces \(v>0\). Training minimizes the reduced HJB residual at randomly sampled times and all inventory states, plus a weighted terminal-condition loss \(v(T,q)=1\). Quote offsets inside the residual use the same clipped analytical optimizer and inventory-edge fill exclusions as the reference solver.

`train_inventory_pinn` accepts an explicit training configuration and seed. `InventoryPINN.compare` reports value-factor RMSE/max error against the numerical reference, PINN residual summaries, and terminal-condition RMSE. In the seeded smoke-test configuration (500 iterations, 32 collocation times, width 32, two hidden layers, five inventory states, 51 reference time nodes), measured factor RMSE was \(0.0199\), maximum factor error \(0.0459\), mean absolute PINN residual \(0.0114\), and terminal RMSE \(0.00109\). This is a working approximation, not a high-accuracy solve; more training/tuning may be needed. The numerical reference is not used as a training target; it is used only for evaluation. This is a physics-only PINN with a soft terminal boundary penalty, not a financial-performance test or proof of generalization.

PyTorch is now declared alongside NumPy and SciPy in [`requirements.txt`](./requirements.txt). The current implementation is intentionally limited to the reduced one-asset diffusion model; it does not add jumps, multi-asset state, regimes, AKF integration, or a portfolio policy.

## Exploratory market-making scenario comparison (not statistical arbitrage)

**Implemented in** [`evaluation.py`](./evaluation.py), runnable through [`run_evaluation.py`](./run_evaluation.py), with tests in [`test_evaluation.py`](./test_evaluation.py). The learned PINN offsets are connected to the existing execution simulator through `HJBQuotingPolicy` in [`pinn.py`](./pinn.py).

The evaluation currently supports one synthetic asset only. It pairs the AS and HJB-PINN policies on identical seeded price paths and fill random seeds. Scenarios are:

- **Ordinary:** arithmetic Brownian price increments with \(\sigma=0.2\).
- **Heavy-tail stress:** the same diffusion plus compound Poisson jumps (\(\rho=2\) per horizon), with symmetric Pareto magnitudes, shape \(3\), scale \(0.02\), truncated at log magnitude \(0.2\). This is a finite truncated jump stress, not a claim about asymptotic tail behavior.
- **Regime-change stress:** volatility doubles halfway through the horizon; the policies do not receive a regime signal.

The reproducible run command is `python3 run_evaluation.py`. It uses 252 intervals over a unit horizon, initial mid-price \(100\), initial cash \(10{,}000\), fill rate \(A=5\), distance sensitivity \(k=1.5\), risk aversion \(\eta=0.1\), inventory limits \([-5,5]\), per-unit fee \(0.001\), PINN seed \(19\), and scenario seeds \((11,23,37,51,79)\). The PINN uses 2,000 training iterations and its HJB assumptions match the ordinary scenario; the jump and volatility-shift scenarios are out-of-model stresses.

Observed terminal P&L over these five synthetic seeds:

| Scenario | AS mean (sample SD) | HJB-PINN mean (sample SD) | Paired HJB-PINN minus AS mean (sample SD) |
|---|---:|---:|---:|
| Ordinary | 2.018 (1.574) | 1.970 (1.489) | -0.048 (0.087) |
| Heavy-tail stress | 5.123 (9.243) | 5.484 (9.398) | 0.361 (0.829) |
| Regime-change stress | 2.062 (1.690) | 2.047 (1.675) | -0.015 (0.016) |

These are exploratory results, not evidence that HJB-PINN beats AS: the sample is tiny, synthetic, and includes only one asset. The paired differences are small relative to their run-to-run variation, and the heavy-tail case is not represented in the policy's diffusion-only HJB. The trained PINN on this larger \([-5,5]\) inventory/horizon configuration also has substantially higher approximation error than the earlier five-state smoke test; the runner prints its reference-comparison diagnostics before the policy results. Improve and validate PINN accuracy before drawing policy conclusions.

## Step 6 — Multi-asset statistical arbitrage via cointegrated pairs

**Implemented in** [`stat_arb.py`](./stat_arb.py), with a reproducible synthetic run in [`run_stat_arb.py`](./run_stat_arb.py) and tests in [`test_stat_arb.py`](./test_stat_arb.py).

The selector examines all candidate asset pairs using the training window only. For each oriented pair it estimates

\[
\log P_y = a + \beta \log P_x + \epsilon,
\]

tests the residual relation with the Engle–Granger two-step cointegration test, adjusts the candidate p-values with Benjamini–Hochberg, estimates spread half-life, and ranks eligible pairs. It greedily selects up to a configured number of **non-overlapping** pairs, so one asset is not assigned to multiple independent pairs in this initial portfolio implementation. If no pair meets the declared filters, it raises an explicit error rather than silently trading.

On held-out observations, the calibrated spread is standardized using training mean and standard deviation. The strategy enters a long-spread position below \(-z_{\rm entry}\), or a short-spread position above \(+z_{\rm entry}\), and exits after reversion through the configured exit band. It uses a fixed gross-notional budget per selected pair, trades both legs, charges transaction costs on turnover, and liquidates remaining holdings at the test horizon. Parameters, pair identities, hedge ratios, and spread statistics are never refit on test data.

This is a simple pairs-trading baseline, not the requested final HJB-embedded multi-asset agent. Cointegration tests can be sensitive to window choice and multiple testing; the FDR adjustment does not remove all model-selection bias. The mean-reversion half-life and entry/exit thresholds must be selected without using the final test window. Short availability, borrow costs, market impact, and live execution are not modeled.

`python3 run_stat_arb.py` generates two synthetic cointegrated pairs and one unrelated asset, fits on the first 750 observations, and evaluates on the next 500 observations for five seeds. It selected both intended pairs in each run. Capital-relative test P&L averaged \(20{,}358.75\) with sample standard deviation \(5{,}517.94\) from initial capital \(100{,}000\), after average transaction costs of about \(203.56\). Because the synthetic generator explicitly creates mean-reverting pairs, this verifies the pipeline against known structure only; it is not evidence of real-market profitability. It also lacks a realistic borrow/short-sale model and a full set of benchmark strategies.

## Step 7 — Performance and risk report (not yet implemented)

Report held-out net P&L, maximum drawdown and duration, exposure/inventory variability, turnover and transaction costs, pair selection stability, and HJB approximation errors. Separate market-making metrics (such as spread capture and order-to-trade ratio) from statistical-arbitrage metrics. Keep numerical residuals separate from financial performance.
