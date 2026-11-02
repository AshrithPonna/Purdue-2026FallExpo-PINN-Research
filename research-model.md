# Research Model: HJB-Embedded Multi-Asset Market Making

**Working title:** *Multi-Asset Market Making under Heavy-Tailed Risk: An HJB Control Framework with Adaptive State Estimation*
**Status:** Step 1 of a staged model build. Equations define a proposed starting model; calibration values and empirical conclusions are not yet available.

## Step 1 — Specify the mathematical control model

This step defines what the trading model observes, controls, and optimizes. It does not yet implement the simulator, Kalman filter, or PINN.

### 1.1 Research question and scope

**Question:** Under the same market data, execution assumptions, costs, and position limits, does an HJB-based multi-asset quoting policy outperform a multi-asset Avellaneda–Stoikov (AS) baseline on risk-adjusted performance in ordinary and stressed markets?

The initial task is **market making**, not statistical arbitrage: the agent posts bids and asks for each asset and manages the inventory created by fills. Selecting assets from a cross-sectional signal or trading a mean-reverting pair would add a separate strategy and is out of scope for this first model.

The comparison is separated into:

1. AS quoting policy: baseline.
2. HJB policy: optimized quoting decisions under the same simulated or observed market and constraints.
3. Adaptive Kalman filter (AKF): optional estimator, tested as an ablation rather than presumed to improve results.
4. PINN: later numerical method for approximating the HJB value function. It is not itself a market model or anomaly detector.

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

## Later steps (not implemented)

5. Train a PINN to approximate the value function and compare it with the numerical reference.
6. Evaluate both policies out of sample in ordinary, heavy-tail, and regime-change scenarios.
7. Report net P&L, maximum drawdown and duration, inventory variance, spread capture, fill rate, order-to-trade ratio, and HJB approximation errors. Keep numerical residuals separate from financial performance.
