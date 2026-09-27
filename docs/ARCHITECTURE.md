# Application Architecture

## Navigation

- Data Explorer
- Executive Dashboard
- About the Program/Application
- Unit Roots & Integration
- Modelling & Forecasting
- Forecasting
- Scenario Analysis
- Debt Sustainability & Debt Forecasting
- External Balance Assessment
- Methodology
- Database Information

## Original Shiny alignment

The Streamlit application is deliberately organised around the same major analytical modules, terminology and workflow found in the supplied R/Shiny source. The visual system uses a dark navy sidebar, blue analytical headers, gold accents, KPI cards and boxed analytical sections.

## Public prototype treatment

Institutional logos and unauthorised institutional branding are excluded. The application carries a neutral public-prototype disclaimer.

## Data layer

The `data/` folder contains the simulated external-sector CSV tables supplied with the original package. The application constructs a monthly analytical master dataset from the monthly external indicators, BOP, FDI, debt, IIP, reserves and remittances tables.

## Extension points

The code is designed so that the bundled CSV layer can later be replaced by a controlled database/API connector, while the navigation and analytical modules remain intact.
