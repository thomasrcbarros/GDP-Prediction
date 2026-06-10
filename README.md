# GDP-Prediction — Nowcasting Brazilian GDP

Brazilian GDP is released by IBGE roughly two months after the end of the quarter. This
project estimates economic growth **in real time** (*nowcasting*) using leading monthly
indicators — the approach used by central banks and financial institutions — before the
official release.

It implements **ARIMA**, **SARIMA**, **VAR**, **VARX** (conditional VAR) and a **bridge
equation** with rolling *backtesting* and explicit **look-ahead bias** evaluation
(respecting each series' real publication calendar). Public data from **BCB (SGS)** and
**IBGE**.

### Models

| Model | Type | How it uses the indicators |
|--------|------|--------------------------|
| ARIMA / SARIMA | univariate | only GDP's own past (+ COVID dummies) |
| VAR | multivariate | joint system; **forecasts** the indicators |
| VARX | multivariate | joint VAR, but **conditions** GDP on the already-published contemporaneous indicators |
| bridge | regression | regresses GDP on the already-published contemporaneous indicators |
| umidas | regression | U-MIDAS: months 1-3 of IBC-Br within the quarter as unrestricted regressors (Foroni-Marcellino-Schumacher, 2015) |
| pool | combination | equal-weight average of 3 bridges — IBC-Br bridge, U-MIDAS and sectoral bridge (PIM+PMS+PMC); forecast-combination literature |

The **umidas[A]** and **pool[C]** variants enter the backtest with the goal of **beating
the bridge[A] baseline**.

ARIMA/SARIMA merely extrapolate GDP's past. The VAR forecasts the indicators themselves —
wasting the nowcasting advantage. **VARX** and **bridge** exploit the real edge: the
current quarter's monthly indicators (IBC-Br, PIM, PMS, PMC) that have **already been
published** while GDP is not yet out.

### Baseline: IBC-Br (the official GDP preview)

**IBC-Br** is the Central Bank's economic activity index — the *official preview* of GDP.
That is why the **bridge on IBC-Br (set A)** is adopted as the **project baseline**: it is
the number any more elaborate model must beat to justify itself.

### Variable sets (A/B/C/D/E)

The multivariate models are tested with different indicator sets (activity indicators and
Ibovespa enter as **seasonally adjusted % QoQ change**, the same scale as the target;
confidence indices enter in levels):

| Set | Variables | Use |
|----------|-----------|-----|
| **A** | IBC-Br (BCB seasonal adj.) | baseline (bridge) |
| **A2** | IBC-Br BCB seasonal adj. **+ own seasonal adj. (STL)** | bridge — **beats the baseline** |
| **B** | PIM + PMS + PMC (industry, services, retail) | VARX |
| **C** | IBC-Br + PIM + PMS + PMC | VARX |
| **D** | IBC-Br + IPCA + FX (macro block) | VARX |
| **E** | endog.: IBC-Br + credit spread · exog.: PCA factor(PIM,PMS,PMC) + consumer confidence | VARX (+ COVID dummy) |

**Set E** is a redesigned VARX with **3 endogenous variables** (`pib_growth`, `ibcbr`,
credit `spread`) and **exogenous** ones: a **PCA factor** summarizing the activity
indicators (PIM, PMS, PMC) into a single component, **consumer confidence** and the
**COVID-peak dummy (2020Q1-Q2)** (a true exogenous regressor, not conditioned on). GDP is
conditioned on the contemporaneous values of IBC-Br and spread, both published before GDP.

### Bridge[A2]: dual seasonal adjustment (beats IBC-Br)

The largest source of divergence between IBC-Br and GDP is **seasonal adjustment**
(different models/samples; BCB Special Study 3/2018). Set **A2** exploits this: in addition
to the BCB's seasonally adjusted IBC-Br, it computes an **own seasonal adjustment via STL**
on the raw IBC-Br (SGS 24363) and uses **both** in a bridge. The divergence between the two
adjustment methods carries signal about GDP's seasonality. In the realistic backtest (same
window, 71 quarters): bridge[A] RMSE 0.562 → **bridge[A2] 0.524** (−6.8%; pre-2020 0.603 →
0.541).

The backtest now reports, beyond total RMSE: **pre-2020 RMSE** and **post-2020 RMSE**
separately, and can use a **rolling estimation window** (`--train-window N`, in quarters;
e.g. `20` ≈ 5 years) instead of the full sample.

## Results (`--model all --backtest`, QoQ target)

Realistic *rolling* backtest (expanding window, *one-step-ahead*) on the common evaluation
window (39 quarters, ~2016–2025; `n` reflects each variant's window):

| model | RMSE | pre-2020 RMSE | post-2020 RMSE | ex-COVID RMSE | MAE | bias |
|--------|-----:|-----:|-----:|-----:|-----:|-----:|
| **bridge[A2]** (dual seas. adj.) | **0.435** | **0.335** | 0.488 | **0.418** | **0.361** | 0.037 |
| bridge[A] (official IBC-Br) | 0.448 | 0.405 | 0.474 | 0.435 | 0.390 | 0.035 |
| varx[D] (IBC-Br+IPCA+FX) | 0.503 | 0.355 | 0.576 | 0.414 | 0.378 | 0.003 |
| pool[C] (combination) | 0.935 | 0.433 | 1.141 | 0.554 | 0.590 | −0.122 |
| varx[C] | 1.331 | 1.584 | 1.145 | 1.132 | 0.872 | 0.158 |
| varx[E] | 1.410 | 2.004 | 0.944 | 1.345 | 0.877 | −0.466 |
| umidas[A] | 1.607 | 0.550 | 2.002 | 0.571 | 0.744 | 0.035 |
| varx[B] | 1.694 | 1.241 | 1.924 | 1.604 | 1.128 | 0.180 |
| mean (baseline) | 2.123 | 0.534 | 2.673 | 0.572 | 1.022 | 0.104 |
| ARIMA | 2.129 | 0.619 | 2.669 | 0.646 | 1.053 | 0.036 |
| SARIMA | 2.133 | 0.706 | 2.661 | 0.670 | 1.108 | −0.175 |
| random walk | 3.143 | 0.817 | 3.954 | 0.959 | 1.486 | −0.040 |

On the long window (71 quarters, excluding the series that only start in 2011): bridge[A]
0.562 → **bridge[A2] 0.524** (−6.8%). 2026Q1 nowcast: bridge[A2] **+0.99%**, bridge[A]
+1.20%.

![Backtest comparison](backtest_comparison.png)

### Study conclusions

1. **IBC-Br is an extremely strong benchmark.** It is the official GDP preview (a
   national-accounts-weighted aggregation of the same sectoral proxies), so any model that
   merely recombines PIM/PMS/PMC, confidence, spread, etc. tends to **reproduce it with
   noise** and lose. That is what happened with VAR, VARX, ARIMA/SARIMA, U-MIDAS and the
   bridge *pool*.
2. **What beat IBC-Br was attacking the methodological divergence, not forecasting better.**
   The biggest error source between IBC-Br and GDP is the different **seasonal adjustment**
   (BCB Special Study 3/2018). `bridge[A2]` uses **two** seasonal adjustments of IBC-Br (the
   official BCB one + an own STL one on the raw series) — the *difference* between methods
   carries signal about GDP's seasonality. Result: **best model in every window**, with the
   gain concentrated pre-2020 (0.405 → 0.335, −17%).
3. **Forecast combination and U-MIDAS did not help here.** With equal weights, adding an
   optimal model (IBC-Br bridge) to worse ones **hurts** — U-MIDAS (the monthly IBC-Br
   decomposition) is especially poor in the volatile post-2020 period.
4. **Agricultural correction left out.** Quarterly agro-GDP series are released together
   with GDP (circular/look-ahead), and the LSPA is an annual-harvest forecast with no clean
   mapping to QoQ — high overfitting risk on ~70 observations.
5. **Caveat.** STL is estimated over the full sample (like the BCB's own seasonally adjusted
   series, which is revised) — there is mild *look-ahead* in the seasonality, consistent
   with the rest of the pipeline. Re-estimating STL per window is a future refinement; the
   gain should persist.

## Indicators

| Series | Source | Code | Lag | Transformation |
|-------|-------|--------|-----------|---------------|
| GDP (quarterly seas. adj. index) — *target* | BCB SGS | 22109 | ~60 d | % QoQ change |
| IBC-Br seasonally adjusted | BCB SGS | 24364 | ~45 d | % QoQ change |
| PIM-PF general industry (seas. adj.) | BCB SGS | 21859 | ~35 d | % QoQ change |
| PMS services volume (seas. adj.) | IBGE SIDRA | 8688 / 7168 | ~45 d | % QoQ change |
| PMC retail volume (seas. adj.) | IBGE SIDRA | 8880 / 7170 | ~45 d | % QoQ change |
| IPCA (monthly change) | BCB SGS | 433 | ~10 d | quarterly sum |
| FX BRL/USD (sell, average) | BCB SGS | 3698 | ~1 d | % QoQ change |
| Average credit spread — Total | BCB SGS | 20783 | ~30 d | level (p.p.) |
| Consumer Confidence | BCB SGS | 4393 | ~5 d | quarterly average (level) |

> The activity factor of set E is the 1st principal component (PCA) of standardized PIM,
> PMS and PMC. The data client also supports the `yahoo` source (e.g. `^BVSP`), in case you
> want to reintroduce market indices. Confirm names/values in your local run with
> `python scripts/fetch_data.py --refresh`.

The publication lags (`config.py`) are the core of the *look-ahead bias* evaluation: they
let you simulate exactly which data would have been available at a given reference date.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Fetch and cache the series in data/ (use --refresh to force a new download)
python scripts/fetch_data.py

# 2. Generate the nowcast and run the backtesting
python scripts/run_nowcast.py --model all --backtest
#   --model {arima,sarima,var,varx,bridge,all}  choose the model(s)
#   --set {A,B,C,D,E}                variable set (multivariate models)
#   --target {qoq,yoy}               QoQ or YoY growth
#   --backtest                       run realistic vs look-ahead backtest
#   --train-window N                 rolling estimation window (N quarters; ~20 = 5 years)
#   --plot                           generate comparison chart (implies --backtest)
#   --plot-file PATH                 output PNG (default: backtest_comparison.png)
#   --refresh                        re-fetch the data

# Baseline (bridge with IBC-Br):
python scripts/run_nowcast.py --model bridge --set A --backtest

# Winning bridge with dual seasonal adjustment:
python scripts/run_nowcast.py --model bridge --set A2 --backtest

# 5-year rolling window (20 quarters) vs full sample:
python scripts/run_nowcast.py --model varx --set E --backtest --train-window 20

# With a chart comparing all variants:
python scripts/run_nowcast.py --model all --plot
```

> With `--model all`, the backtest re-trains ARIMA/SARIMA with order search in each window
> — it takes several minutes. To iterate fast, run one model/set at a time
> (e.g. `--model bridge --set A2`).

`--plot` saves a PNG with three panels: (1) each model's *one-step-ahead* forecasts vs.
observed GDP over the backtest period; (2) RMSE per model (realistic regime); (3) next
quarter's nowcast per model.

Typical backtest output: a table comparing, per model, the error in the **realistic**
regime (expanding window, *one-step-ahead*) with the **look-ahead** regime (fit over the
whole sample). The difference quantifies the anticipation bias — the look-ahead regime
shows systematically smaller errors precisely because it leaks future information.

## Structure

```
config.py                      # series codes, lags, parameters
src/gdp_nowcast/
├── data_sources.py            # BCB SGS + IBGE clients (with CSV cache)
├── dataset.py                 # monthly→quarterly alignment, target, anti look-ahead
├── preprocessing.py           # ADF, reversible differencing
├── models/{arima,sarima,var,varx,bridge,pool}.py
├── backtest.py                # rolling-origin, metrics, look-ahead
└── nowcast.py                 # end-to-end pipeline
scripts/{fetch_data,run_nowcast}.py
tests/                         # pytest
```

## Tests

```bash
python -m pytest
```

They cover quarterly alignment, transformation reversibility, the anti look-ahead guard
(asserting that data after the reference date does not leak) and model fitting/forecasting
(including the VARX conditioning).

## Notes

- Data in `data/*.csv` is not versioned (see `.gitignore`); run `fetch_data.py` to
  populate it.
- The nowcast uses already seasonally adjusted/revised series from SGS; a natural extension
  would be to incorporate real *vintages* (data as published at the time) for an even more
  faithful look-ahead test.
