# Phase 2 design: datasets, split, GCN, metrics

Status: implemented (see section 7, As built). It implements CLAUDE.md 1.3–1.4. Items marked **Decision** wait for the user (section 6).

## 0. The rule this design follows: train/serve parity
None of the allowed free sources gives live flight delays (BTS runs about 3 months behind). FAA NAS Status is the one live delay signal, and it is also the scorecard's ground truth. So **every model feature must be built the same way at training time and in the live app.** That rules out "delay in the last hour" and backlog features, even though they would score well offline. Delay propagation therefore enters through the graph: weather and traffic at neighbouring airports feed each node's prediction.

## 1. Datasets (train/real.py, US only)
BTS columns read (pyarrow, only these): FlightDate, Reporting_Airline, Origin, Dest, CRSDepTime, CRSElapsedTime, DepDelay, DepDelayMinutes, ArrDelayMinutes, ArrDel15, Cancelled, Diverted, Distance, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay. Candidate URL: `transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{Y}_{M}.zip`. real.py tries a list of candidate URLs and fails loudly if all of them fail.

Time handling: BTS times are local. The scheduled departure in UTC is FlightDate + CRSDepTime, interpreted in the origin airport's time zone (tz name from OpenFlights airports.dat) and converted to UTC. Scheduled arrival in UTC = scheduled departure + CRSElapsedTime. Never parse DepTime or ArrTime, because of `2400` values and midnight rollover.

**flights.parquet**: one row per flight that was neither cancelled nor diverted. Target `arr_delay` = ArrDelayMinutes. Cause columns are 0-filled when ArrDel15 = 0.
**airport_hour.parquet**: one row per (airport i, UTC hour t, horizon h). The issue time is τ = t − h, with h sampled from {2, 6, 24} for each snapshot. Targets are taken over flights scheduled to depart i in hour t: `y_dep` = mean DepDelayMinutes, `y_late` = mean LateAircraftDelay, `y_nas` = mean NASDelay, and `n` = flight count (rows with n = 0 are masked).

| feature (all standardised with train-only stats) | train source | live source |
|---|---|---|
| typical scheduled deps/arrs at (airport, hour-of-week), mean of the 4 weeks ending ≥ 90 days before t | BTS | latest BTS month (same lag) |
| obs at τ: visibility, ceiling, wind, gust, precip, thunderstorm, flight category | IEM ASOS | METAR |
| forecast for t: precip, gust, low cloud, visibility, CAPE | Open-Meteo historical-forecast API | Open-Meteo forecast |
| hour/dow/month sin-cos (local), US holiday (pandas), h | calendar | calendar |
| tier, log degree (train window), weather-zone one-hot | static | static |

**Forbidden as features:** anything computed from flights scheduled in hour t or later, cause columns, taxi-out, NAS status, and ADS-B counts (there is no free historical ADS-B to train on). En-route weather points (25/50/75 % of the great circle) are snapped to a 2° grid, so each grid cell needs one Open-Meteo call for the whole window, which stays inside the free limits. TAF has no confirmed free archive: use it live for display only.

## 2. Leakage-safe split (time-ordered, never shuffled)
Window: the latest 30 complete BTS months, starting no earlier than Open-Meteo historical-forecast coverage (2022, to verify).

| block | months | used for |
|---|---|---|
| train | 1–24 | fitting; the last 2 months are the inner holdout for early stopping; 4 forward-chaining OOF folds for the Phase 3 stacker |
| gap | 7 days | dropped (≥ max h + 24 h lag) |
| calib | 25–27 | conformal calibration only; never used for fitting or model selection |
| gap | 7 days | dropped |
| test | 28–30 | reported numbers only; evaluated once per release |

Every statistic is fitted on train only: scalers, adjacency weights, degree, target encodings, typical schedules, and the percentile map behind the stress index. Random subsampling of flights for LightGBM is allowed inside a block, never across blocks. Calib and test fall in different seasons, so report coverage per test month.

## 3. GCN (ml/gcn.py, numpy + scipy.sparse)
**Graph:** nodes are US airports averaging ≥ 20 scheduled departures/day in the train window (the count goes in the card). `A[j,i]` = mean daily scheduled flights j→i over train. The operator is directed in-flow: `Â = ½I + ½·D_in⁻¹Aᵀ`, which is row-stochastic, so each node is half itself and half the flight-weighted mean of its upstream airports. Ablation: symmetric Kipf normalisation. There are no node-ID features, so the weights transfer to the world graph (OpenFlights edges) at inference; non-US outputs are labelled low-confidence.

**Model** (per snapshot, X ∈ ℝ^{N×F}; P = ÂX is precomputed because Â and X are fixed):
`Z = P W₁ + b₁,  H = ReLU(Z),  Ŷ = Â H W₂ + b₂`, with W₁ ∈ ℝ^{F×32} and W₂ ∈ ℝ^{32×2} (heads: y_dep, y_late).

**Loss:** `L = (1/S) Σ m·w·huber₁₅(Ŷ − Y) + λ(‖W₁‖² + ‖W₂‖²)`, with w = min(n, 20)/20, S = Σ m·w, λ = 1e-4.

**Gradients:** with G = m·w·huber′(Ŷ − Y)/S:
`∂W₂ = (ÂH)ᵀG + 2λW₂;  ∂b₂ = 1ᵀG;  ∂H = ÂᵀG W₂ᵀ;  ∂Z = ∂H ⊙ 1[Z>0];  ∂W₁ = Pᵀ∂Z + 2λW₁;  ∂b₁ = 1ᵀ∂Z`.

**Training:** Adam (lr 1e-3, β 0.9/0.999, ε 1e-8), 32 snapshots per batch, He/Xavier init, early stopping on inner-holdout MAE with patience 10.

**Gradient-check test:** N = 6, F = 3, hidden 4, random row-stochastic Â, float64, central differences with ε = 1e-6. Pass if relative error < 1e-6 for every parameter. Redraw the inputs if any |Z| < 1e-4 or |Ŷ − Y| is within 1e-4 of δ, so no point sits on a ReLU or Huber kink.

**Export:** `gcn.npz` (weights, scaler, node list) plus `adj.npz` (sparse Â). Inference is two sparse matmuls.

## 4. Base-model targets (so the Phase 3 breakdown can be checked against BTS)
| component | model | grain | target |
|---|---|---|---|
| cascade | GCN head 2 | airport-hour | LateAircraftDelay |
| congestion | LightGBM | airport-hour | NASDelay; stress index for the UI = train-set percentile of the prediction |
| weather | LightGBM | flight | WeatherDelay |
| carrier | **Decision 1** | flight | CarrierDelay + SecurityDelay |
| route | algorithmic, labelled as such | flight | none: max(0, wind-adjusted time − CRSElapsedTime) |
| bird | prior-based estimate | flight | none (see Decision 5) |

GCN head 1 (y_dep, total delay) drives the Congestion page and what-if. The final target is ArrDelayMinutes. BTS limitations to state in the paper:
- Cause minutes exist only when ArrDel15 = 1.
- BTS "NAS" includes non-extreme weather, so "weather" here means extreme weather only.
- Cancelled and diverted flights are excluded, which biases means downward on bad days.

## 5. Metrics (test block only; every number with its n)
- **Airport-hour models** (GCN, congestion): MAE and RMSE in minutes, plus F1 for "mean departure delay > 15 min". Report both unweighted and flight-weighted.
- **Flight-level models** (weather, final ensemble): MAE, RMSE, and F1 for ArrDel15. The 15-minute threshold is fixed and never tuned on test.
- **Baselines on the same rows:** historical mean per airport × hour-of-week × month (train only), and LightGBM on the same features without the graph. Only claim "the graph helps" if the GCN beats the no-graph LightGBM.
- **Reporting:** horizons h = 2 and h = 24; 95 % CI from a block bootstrap over test days (1,000 resamples).
- **Per weather zone:** only zones with ≥ 1,000 test flights get a number; the rest say "insufficient data". BTS has no monsoon-zone airports.
- **model_card.json, per model:** name, kind (ml | algorithmic | prior), trained_on, window and split dates, n per block, metrics, baselines, per_zone, created_at. CI's synthetic run writes its card to a temp dir and never overwrites the committed card.

## 6. Decisions (proposed defaults)
1. **Carrier model:** 1.3 lists a carrier component but no model for it. Default: a lookup table of mean CarrierDelay + SecurityDelay by airline × origin × month, fitted on train only.
2. **NAS status:** it has no free history to train on and it is the scorecard's ground truth (feeding it in makes the scorecard circular). Default: not a model input; show it beside the prediction instead.
3. **ADS-B traffic count:** it can't be trained on. Default: the trained "traffic" feature is the typical BTS schedule; ADS-B is used for live display only.
4. **Departure time:** the Search page has no departure-time input, but flight-level models need one. Default: add a departure-time field that defaults to now + 2 h.
5. **Bird strike data:** the FAA Wildlife Strike Database (free, public) would give real strike labels, but it is not in rule 1's list. Default: leave it out and keep bird prior-only.
6. **Monsoon zone:** BTS has no monsoon data, and tropical and subpolar are thin. Default: in Phase 3, any zone with < 1,000 OOF rows uses pooled stacker weights, and the UI says so.


## 7. As built (2026-10-17)
Decisions 1-6 were adopted as proposed. Where the build differs from the design above:
- **Features.** `log_degree` was dropped (the world graph is derived, so its degree is not real); airport features are scheduled load, obs, forecast, calendar, tier and zone one-hots. The typical-schedule profile is one static table built from the train block only (not a rolling 4-week mean ending 90 days before t): it carries no delay information, so it is leakage-safe. The Open-Meteo forecast request has 10 variables (temperature, precipitation, wind, gusts, visibility, low cloud, CAPE, weather code, 250 hPa wind speed and direction) so one location counts as one call.
- **Graph.** Training nodes are the 42 US seed airports (not every airport above 20 departures/day), because every node needs coordinates, an ICAO code and a time zone. The training operator is the in-flow operator from BTS flight counts. At inference the same weights run on the derived 155-airport world graph with tier-size edge weights.
- **Training.** Every LightGBM fit (components and baselines) stops early on a time-ordered inner holdout (last 15% of the fit rows, 1-day gap). The carrier lookup uses hierarchical shrinkage (k = 100) instead of a hard count threshold. The GCN's standardiser gives constant-in-training features std = inf. The `route` component is the headwind cost only (>= 0) from 250 hPa winds at the 25/50/75% points; the stop-insertion planner feeds the advice, not the delay.
- **Metrics.** Both MAE and RMSE are reported, each with n and a block-bootstrap (by day) 95% CI, next to the historical-mean and no-graph baselines; the stacker is also compared with a direct LightGBM. F1 uses the fixed 15-minute threshold. `graph_helps` is true only if the whole paired RMSE-ratio interval of the GCN against the no-graph LightGBM is below 1 (it was MAE until section 8).
- **Real run.** 4 months (2026-04..07) on a dev machine instead of 30 months: window 2026-04-01..07-31, train to 07-07, calib 07-10..19, test 07-22..31. The first run had no early stopping; its results were close to the committed ones.
- **Not done.** The 30-month run (train.yml), OpenFlights/BTS-based world routes, an airport-hour historical prior feature for the GCN (the congestion model has one since section 8; the GCN's got worse).

## 8. Improvement round 1: congestion, weather delay, carrier (rules fixed 2026-10-17)
Why: on the committed 4-month run the congestion model (RMSE 10.83 vs 10.92) and the weather model (14.25 vs 14.27) beat their naive baselines by under 1% RMSE, and their MAE was worse than the baselines'. These rules were fixed after the first train-only comparison of the congestion and GCN variants and before any test-block number of a changed model was computed.

**Headline metric: RMSE in minutes, unweighted, on the test block**, with MAE beside it (flight-weighted MAE for airport-hours). All three models are conditional-mean models, so RMSE is the consistent score; MAE on a zero-inflated target rewards predicting near 0. Same row sets as the committed card: congestion = test airport-hours with n > 0 (7,544), weather and carrier = test flights (94,968).
**Claim rule:** "better than the baseline" only if the 95% CI of RMSE(model) / RMSE(baseline) lies below 1. The CI is a paired block bootstrap over test days (1,000 resamples). The test block is 10 days, so this is a weak test: a CI that straddles 1 means "not distinguishable", whatever the point estimate.
**Baselines stay as they were** (congestion: train mean by airport x hour-of-week; weather and carrier: train mean). Added, never swapped: weather also gets the train mean by origin airport; congestion also gets the same LightGBM without the new prior features, so the card shows what the prior itself contributed.
**Variants are chosen on train out-of-fold (OOF) predictions only**: forward-chaining, the model for chunk j is fitted on chunks before j minus a 1-day gap, exactly the stacker's folds (chunks 1-4). Never the calib block, never test. Choose the simplest variant whose pooled OOF RMSE is within 0.3% of the best; a variant replaces the control only if it also improves the control's pooled OOF RMSE by at least 0.3% and wins in at least 3 of the 4 folds. Every variant tried is listed in the table below with its OOF RMSE, so the number of looks is visible; the card carries the OOF evidence of the model that shipped (`oof` next to `metrics`).
**Test is read once**, for the chosen variants, after the code is final. (What actually happened is under "Outcome".)

**Airport-hour historical prior (`congestion.HistPrior`).** Flight-weighted mean of the airport-hour targets (NAS, departure and late-aircraft delay) by airport and by airport x hour-of-week, shrunk down global -> tier -> airport -> airport x hour (k = 60 flights, fixed in advance; k = 20 and 200 were run afterwards as a sensitivity check, see the table, and not selected on). Unseen airports use their tier's values. Training rows get the leave-one-week-out value (a row never sees its own week); rows outside the fit window, and the live app, use the table fitted on the whole fit window. For the stacker folds the table is refitted per fold on that fold's fit window, so a fold model never sees a table that contains its own chunk. Live, the table is `hist.json`.

### Variants tried (train OOF, chunks 1-4; pooled RMSE in minutes, lower is better)
The scripts were one-off analysis scripts and are not in the repo (the shared notes folder of the session has them); the shipped pipeline (`train_all.py`) recomputes the OOF numbers of the shipped models and baselines on every run. Per-fold values are chunk 1 / 2 / 3 / 4.

| model | variant | pooled OOF RMSE | vs control | folds won |
|---|---|---|---|---|
| congestion (n = 59,618 airport-hours) | per airport x hour-of-week mean (the baseline) | 9.788 | | |
| | A0 control: LightGBM, no prior | 9.328 (6.650 / 8.148 / 9.567 / 12.060) | | |
| | **A1 + HistPrior (k = 60): shipped** | **9.283** (6.672 / 8.128 / 9.472 / 11.999) | -0.48% | 3 of 4 |
| | A2 = A1 + regularised params (lr 0.03, 15 leaves, min leaf 200, L2 20) | 9.265 | -0.67% | |
| | A3 = A1 + Tweedie 1.3 / 1.6 | 9.272 / 9.289 | -0.60% / -0.41% | |
| | A5 = A1 with k = 20 / 200 | 9.280 / 9.295 | -0.51% / -0.35% | |
| GCN (heads y_dep / y_late) | G0 control | 30.797 / 17.468 | | |
| | G1 + HistPrior | 31.104 / 17.494 | +1.0% / +0.1% | no |
| weather delay (n = 745,197 flights) | constant train mean (the baseline) | 14.728 (10.078 / 15.220 / 15.507 / 17.253) | | |
| | origin-airport mean | 14.723 | | |
| | **W0 control: LightGBM L2: shipped** | 14.715 | | |
| | W0b L2, regularised (lr 0.03, min leaf 1000, L2 100, 15 leaves, feature fraction 0.5) | 14.687 | -0.19% | 4 of 4 |
| | W1 Tweedie 1.5 | 14.713 | -0.01% | |
| | W2 hurdle: P(any weather delay) x LightGBM L2 on the positives | 14.680 (10.058 / 15.182 / 15.457 / 17.176) | -0.24% | 4 of 4 |
| | W2b hurdle, gamma amount / W2c hurdle, constant amount | 14.681 / 14.680 | -0.23% / -0.24% | |
| | W2d hurdle, regularised classifier | 14.677 | -0.25% | |
| | W3 hurdle / Tweedie + origin-airport prior (leave-one-week-out) | 14.690 / 14.717 | worse than without | |
| carrier + security delay (n = 745,197) | global train mean (the baseline) | 34.143 | | |
| | **control: airline > x origin > x month, k = 100: shipped** | 34.129 | | |
| | same, k = 30 / 300 / 1000 / 3000 | 34.141 / 34.119 / 34.113 / 34.113 | at best -0.05% | |
| | airline > x origin, no month level, k = 100 / 300 / 1000 | 34.120 / 34.116 / 34.114 | at best -0.04% | |
| | airline only, k = 100 / 300 / 1000 | 34.125 / 34.124 / 34.124 | | |

Decisions by the rule above: congestion adopts the prior (A1 is within 0.3% of the best, A2, and improves the control by 0.48% in 3 of 4 folds). The GCN keeps its features (the prior made it worse), so `GCN_FEATURES` is `AIRPORT_FEATURES` minus the prior and the no-graph baseline uses the same list. Weather stays on the L2 control: the best variant is a hurdle model that improves it by only 0.24%, below the 0.3% bar, even though it wins all four folds; Tweedie and the airport prior do not help. Carrier stays as it was: no setting improves it by 0.3%, and every one of them is within 0.1% of the global mean.

### Outcome (test block, 4-month real run; `backend/models/model_card.json`)
RMSE in minutes, the model against the baselines, ratio with a paired block-bootstrap 95% CI over the 10 test days:

| model | RMSE | baseline | ratio (95% CI) | before this round |
|---|---|---|---|---|
| congestion (n = 7,544) | 10.64 | hist mean 10.92 | 0.974 (0.956 to 0.998): better | 10.83 vs 10.92 |
| | | same LightGBM without the prior 10.83 | 0.983 (0.975 to 1.004): not distinguishable | |
| weather delay (n = 94,968) | 14.25 | mean 14.27 | 0.998 (0.996 to 1.006): not distinguishable | unchanged model |
| | | origin-airport mean 14.28 | 0.998 (0.996 to 1.004): not distinguishable | |
| carrier (n = 94,968) | 40.95 | global mean 40.97 | 1.000 (0.998 to 1.001): not distinguishable | unchanged model |
| GCN, departure delay | 33.03 | hist mean 34.55 | 0.956 (0.937 to 0.975): better | |
| | | LightGBM without graph 32.83 | 1.006 (0.978 to 1.020): not distinguishable | |
| ensemble | 64.14 | hist mean 72.59 | 0.884 (0.833 to 0.926): better | 64.18 |
| | | direct LightGBM 63.79 | 1.005 (0.998 to 1.014): not distinguishable | |

Read honestly: the congestion model went from 0.9% to 2.6% better than the per-airport hour-of-week mean on RMSE, and the whole interval is just below 1 (upper bound 0.998), so it clears the claim rule narrowly. On the 59,618 OOF airport-hours the same comparison is 0.948 (0.923 to 0.964) against the hist mean and 0.995 (0.992 to 0.9995) against the same model without the prior, so the prior itself is worth about half a percent. MAE is still worse than the hist mean (5.09 vs 4.68; flight-weighted 4.72 vs 4.46): an L2 model predicts the mean of a heavy-tailed target, MAE rewards the median, and RMSE is the pre-registered headline. The weather and carrier components have no measurable skill over a constant on this data (weather OOF ratio 0.999 (0.998 to 1.001)); WeatherDelay is 1.2% of train flights, the positives have a median of 34 min and a 99th percentile of 582 min, and the share of train flights with a weather delay rises from 0.7% in April to 2.6% in the first week of July, so a 4-month window mixes very different regimes. That is a statement about 4 months of data, not about the method; the same OOF protocol should be re-run on the 30-month data before anything is built on a hurdle model.

What happened, in order, so the number of looks is visible: the congestion (A0 to A5) and GCN (G0, G1) variants were compared on OOF first; the rules above were then written down, after the weather L2 control was known and before the Tweedie and hurdle results (so the 0.3% bar was chosen after seeing congestion differences of 0.35% to 0.67%); the remaining weather variants and the carrier and bird checks followed; the test block was read for the retrained bundle. After reading it, one claim rule was tightened: `graph_helps` had been "GCN MAE below the no-graph LightGBM MAE", which compares a Huber-trained model with an L2-trained one on MAE; it is now "whole paired RMSE-ratio interval below 1", which makes it false (ratio 1.006, interval 0.978 to 1.020) for departure delay, and the Models page says so. No model, feature or hyper-parameter was changed after the test block was read.

**Bird weight.** The stacker weight of the bird index is 0, and that is the right answer rather than a scaling bug: the index is a function of local time of day (about 0.23 in the morning and 0.55 in the evening), its raw correlation with arrival delay is 0.005, its correlation with the hour-of-day mean delay is 0.34, and after removing hour x month x origin cell means the partial correlation is -0.013. Delays and the index both peak in the evening, so whatever the index could explain is already carried by the time-of-day features. BTS has no bird cause code, so nothing here can validate it as a delay driver; it stays an advisory strike-risk estimate ("prior-based estimate").

**Known limits of this round.** The test block is 10 days, so every interval above is wide; the OOF samples are about 8 times larger but they selected the variants. The prior's leave-one-week-out values use weeks on both sides of a row, which the live table cannot (it only has the past); the per-fold refit and the OOF-versus-test agreement are the check on that, not a proof. Non-US airports get their tier's US-derived prior values. The runway-closure what-if does not move the congestion stress (the old model did not respond to scheduled load at ORD at all; the new one moves from stress 70 to 64 when the load is cut and not at all when it is raised), only the GCN delay.
