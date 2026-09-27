# 14-Model Catalogue

This catalogue follows the model definitions and workflow of the supplied R/Shiny forecasting platform.

| # | Model | LHS / System | Core drivers | Default lag |
|---|---|---|---|---:|
| 1 | Export Demand — ARDL | goods_exports | global demand, REER, terms of trade, gold, platinum, tobacco, lithium | 2 |
| 2 | Import Demand — ARDL | goods_imports | GDP, REER, inflation, crude oil, world trade | 2 |
| 3 | Current Account — ARDL | current_account_balance | terms of trade, REER, GDP, global demand, FDI, remittances, debt | 2 |
| 4 | Exchange Rate — Monetary / BEER | nominal_fx_index | inflation, REER, reserves, terms of trade, global demand | 1 |
| 5 | REER Equilibrium — BEER | reer_index | net IIP, GDP, terms of trade, world trade, global demand | 1 |
| 6 | Reserve Accumulation — ARDL | gross_reserves_usd_mn | current account, FDI, borrowing, repayment, trade balance, import cover | 2 |
| 7 | FDI Inflows — ARDL | fdi_inflows | GDP, REER, global demand, terms of trade, trade balance, debt | 2 |
| 8 | External Debt Dynamics | external_debt_outstanding | new borrowing, principal repayment, interest, GDP, current account, REER | 1 |
| 9 | IIP / External Sustainability | net_iip | current account, FDI, REER, debt, reserves, GDP | 1 |
| 10 | Terms of Trade / Commodity Prices | terms_of_trade_index | gold, platinum, tobacco, lithium, copper, oil, global demand | 1 |
| 11 | Multivariate VAR | system | REER, exports, imports, CA, reserves, FDI, inflation, GDP | 2 |
| 12 | VECM & Johansen Cointegration | system | REER, exports, imports, CA, reserves, FDI, GDP | 2 |
| 13 | GARCH(1,1) | reer_index volatility | return-like transformation | 1 |
| 14 | Shrinkage BVAR | system | REER, exports, imports, CA, reserves, FDI, inflation, GDP | 2 |

## Modelling workflow

Each model is presented through:

1. Specification and data availability
2. Pre-estimation diagnostics
3. Structural-break screening
4. Estimation
5. Post-estimation diagnostics
6. Forecast / prediction

The application uses transparent labels where a Python implementation is an analytical proxy rather than a literal one-to-one replication of an R package algorithm. In particular, the single-equation ARDL-style models are dynamic regressions and are not presented as a full Pesaran-Shin-Smith bounds-test implementation.
