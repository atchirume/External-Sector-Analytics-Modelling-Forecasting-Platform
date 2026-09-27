# External Sector Analytics, Modelling & Forecasting — Corrected v6

Public research and demonstration prototype based on the supplied R/Shiny external-sector analytics platform.

## v6 correction

This release specifically hardens the External Balance Assessment (EBA) module and the analytical master-data layer.

### EBA fix
The previous EBA implementation referenced a hard-coded `reer` source column even though the master dataset uses `reer_index`. The corrected implementation constructs a uniquely named analytical working frame (`ca`, `gdp`, `reer`, `iip`, `tot`, `global`, `res`, `infl`) and performs all EBA calculations against those aliases.

The EBA module now includes:
- current-account benchmark regression;
- REER equilibrium regression;
- CA and REER gaps;
- external-sustainability current-account norm;
- NFA/GDP calculation;
- local REER sensitivity to the CA gap;
- fitted CA norm contribution decomposition;
- integrated assessment table;
- model-observation diagnostics;
- explicit distinction from the official IMF EBA methodology.

### Master-data hardening
- duplicate columns are detected and coalesced;
- numeric conversion is safe after duplicate resolution;
- common numeric formatting is cleaned;
- dates are normalised and duplicate dates removed;
- the master dataset is prevented from passing duplicate fields into analytical modules.

## Public prototype disclaimer

This application is intended for research, analytical and demonstration purposes. It is not an officially authorised institutional system and should not be interpreted as representing the policies, decisions, systems or official statistics of any government or other institution.

Author: Chirume Admire Tarisirayi
