# Cross-domain spring wheat NDVI prediction

Code for the study **Transferable spring wheat NDVI prediction across unseen region–year domains using phenological alignment and SAR temporal dynamics**.

The workflow evaluates spring wheat NDVI prediction under strict cross-domain validation using phenological and thermal-time information, Sentinel-1 SAR observations, SAR temporal dynamics, and environmental variables.

## Validation design

Three validation schemes are included:

- Leave-one-region-year-out (LORYO)
- Leave-one-region-out (LORO)
- Leave-one-year-out (LOYO)

Five feature configurations are evaluated under LORYO:

| Configuration | Information |
| --- | --- |
| T0 | Phenology and thermal time |
| S0 | Current SAR |
| U0 | Phenology/thermal time + current SAR |
| U1 | U0 + SAR temporal dynamics |
| U2 | U1 + environmental information |

U0, U1, and U2 are also evaluated under LORO and LOYO.

## Repository structure

```text
.
├── src/
│   ├── cross_domain_modeling.py
│   └── run_cross_domain_analysis.py
├── data/
│   └── README.md
├── results/
│   └── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

`cross_domain_modeling.py` contains data preparation, causal temporal feature construction, model fitting, and source-only validation functions.

`run_cross_domain_analysis.py` is the main entry point for the LORYO, LORO, LOYO, statistical, and diagnostic analyses.

## Data

Place one of the following files in `data/`:

```text
spring_wheat_with_phenology_phases.csv
spring_wheat_2019_2023_all_regions.csv
```

The input data should contain region, year, sample identifier, observation date, orbit information, NDVI, Sentinel-1 SAR variables, thermal-time variables, and environmental variables used by the analysis.

Raw and processed data are not included unless redistribution is permitted by the original data providers.

## Analysis

Region-year combinations are treated as independent domains. Target domains are excluded from model fitting and hyperparameter selection. Temporal SAR predictors are constructed from current and previous observations within each region, year, sample, and orbit sequence.

Performance statistics include R², RMSE, MAE, bias, calibration statistics, domain-level bootstrap confidence intervals, paired Wilcoxon tests, and Holm-adjusted p-values. The workflow also includes held-out permutation importance, domain-space PCA, and Pearson/Mantel analyses.

## Installation

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python src/run_cross_domain_analysis.py
```

Outputs are written to `results/`.

## Citation

Xiao, W., Zhu, C., Zhang, L., Wu, W., and Zhao, L. *Transferable spring wheat NDVI prediction across unseen region–year domains using phenological alignment and SAR temporal dynamics.* Citation details will be updated after publication.

## License

This repository is distributed under the MIT License.
