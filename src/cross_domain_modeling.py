



























from __future__ import annotations

import json
import math
import random
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
)




SEED = 20260731
QUICK_MODE = False          
N_JOBS = 4                  
INNER_CV_SPLITS = 4
BOOTSTRAP_REPS = 2000




PROGRESS_MODE = "source_quantile"  
SOURCE_QUANTILE = 0.95
FIXED_MAX_DAP = 180.0
FIXED_MAX_GDD = 2200.0



PHASE_BOUNDARIES = (0.18, 0.55, 0.82)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT = PROJECT_DIR / "results" / "base"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)





def read_csv_robust(path: Path) -> pd.DataFrame:
    errors: List[str] = []
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as exc:  
            errors.append(f"{enc}: {exc}")
    raise RuntimeError(f"Cannot read {path}. Tried encodings:\n" + "\n".join(errors))


def locate_input_file() -> Path:
    names = [
        "spring_wheat_with_phenology_phases.csv",
        "spring_wheat_2019_2023_all_regions.csv",
    ]
    roots = [
        SCRIPT_DIR,
        SCRIPT_DIR.parent,
        SCRIPT_DIR / "python_results",
        SCRIPT_DIR.parent / "python_results",
        SCRIPT_DIR / "phenology_results",
        SCRIPT_DIR.parent / "phenology_results",
        SCRIPT_DIR.parent / "data",
    ]
    checked: List[Path] = []
    for root in roots:
        for name in names:
            p = root / name
            checked.append(p)
            if p.exists():
                return p
    
    for root in (SCRIPT_DIR, SCRIPT_DIR.parent):
        for name in names:
            hits = list(root.glob(f"**/{name}"))
            if hits:
                return hits[0]
    nearby = sorted(str(p) for p in SCRIPT_DIR.glob("*.csv"))[:30]
    raise FileNotFoundError(
        "Input CSV not found. Checked:\n"
        + "\n".join(map(str, checked))
        + "\nNearby CSV files:\n"
        + "\n".join(nearby)
    )


def first_existing(df: pd.DataFrame, names: Sequence[str], required: bool = True) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    if required:
        raise KeyError(f"None of these columns exists: {list(names)}")
    return None


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def bootstrap_mean_ci(values: Sequence[float], reps: int = BOOTSTRAP_REPS) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(SEED)
    boot = np.empty(reps, dtype=float)
    for i in range(reps):
        boot[i] = np.mean(rng.choice(x, size=len(x), replace=True))
    return tuple(np.quantile(boot, [0.025, 0.975]).astype(float))


def sanitize_sheet_name(name: str) -> str:
    bad = set('[]:*?/\\')
    s = ''.join('_' if ch in bad else ch for ch in name)
    return s[:31] or "Sheet1"





def standardize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    date_col = first_existing(df, ["date", "Date", "datetime", "acquisition_date"])
    region_col = first_existing(df, ["region", "Region", "state", "State"])
    year_col = first_existing(df, ["year", "Year"], required=False)
    target_col = first_existing(df, ["NDVI", "ndvi", "target_ndvi", "Observed_NDVI"])
    sample_col = first_existing(
        df,
        ["sample_id", "Sample_ID", "field_id", "pixel_id", "point_id", "id"],
        required=False,
    )
    orbit_col = first_existing(
        df,
        ["orbit_pass", "orbit", "pass", "Orbit", "orbit_direction"],
        required=False,
    )

    rename = {
        date_col: "date",
        region_col: "region",
        target_col: "NDVI",
    }
    if year_col:
        rename[year_col] = "year"
    if sample_col:
        rename[sample_col] = "sample_id"
    if orbit_col:
        rename[orbit_col] = "orbit_pass"
    df = df.rename(columns=rename)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "year" not in df:
        df["year"] = df["date"].dt.year
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["region"] = df["region"].astype(str).str.strip().str.replace(" ", "_", regex=False)
    df["NDVI"] = safe_numeric(df["NDVI"])

    if "sample_id" not in df:
        
        
        df["sample_id"] = df["region"].astype(str)
    if "orbit_pass" not in df:
        df["orbit_pass"] = "unknown"

    
    aliases = {
        "VV": "VV_dB",
        "VH": "VH_dB",
        "incidence_angle": "angle",
        "ERA5_T2M": "ERA5_T2M_C",
        "ERA5_soil_moisture": "ERA5_SM",
        "Rain_10d": "Rain_10d_mm",
        "GDD": "GDD_cum",
        "AGDD": "GDD_cum",
        "DAP": "DAP_estimated",
    }
    for old, new in aliases.items():
        if new not in df and old in df:
            df[new] = df[old]

    required = ["VV_dB", "VH_dB"]
    missing = [c for c in required if c not in df]
    if missing:
        raise KeyError(f"Required SAR columns missing: {missing}")

    for c in [
        "VV_dB", "VH_dB", "VH_VV_dB", "RVI", "angle", "ERA5_T2M_C",
        "ERA5_SM", "Rain_10d_mm", "GDD_10d", "GDD_cum", "DAP_estimated",
    ]:
        if c in df:
            df[c] = safe_numeric(df[c])

    
    
    
    if "DAP_estimated" not in df:
        first_date = df.groupby(["region", "year"], observed=True)["date"].transform("min")
        df["DAP_estimated"] = (df["date"] - first_date).dt.days.astype(float)

    
    
    if "GDD_cum" not in df:
        if "ERA5_T2M_C" not in df:
            raise KeyError("Need GDD_cum/AGDD or ERA5_T2M_C to construct strict progress.")
        base_temp = 0.0
        temp_gdd = np.maximum(df["ERA5_T2M_C"].fillna(base_temp) - base_temp, 0.0)
        tmp = df.assign(_gdd=temp_gdd).sort_values(["region", "year", "date"])
        tmp["GDD_cum"] = tmp.groupby(["region", "year"], observed=True)["_gdd"].cumsum()
        df = tmp.drop(columns="_gdd").sort_index()

    
    if "GDD_10d" not in df:
        ordered = df.sort_values(["region", "year", "date"]).copy()
        increments = ordered.groupby(["region", "year"], observed=True)["GDD_cum"].diff()
        increments = increments.where(increments >= 0, np.nan).fillna(0.0)
        ordered["GDD_10d"] = (
            increments.groupby([ordered["region"], ordered["year"]], observed=True)
            .rolling(window=2, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
        )
        df = ordered.sort_index()

    return df


def add_static_sar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vv_lin = np.power(10.0, out["VV_dB"] / 10.0)
    vh_lin = np.power(10.0, out["VH_dB"] / 10.0)
    eps = 1e-8
    if "VH_VV_dB" not in out:
        out["VH_VV_dB"] = out["VH_dB"] - out["VV_dB"]
    if "RVI" not in out:
        out["RVI"] = 4.0 * vh_lin / (vv_lin + vh_lin + eps)
    out["Span_linear"] = vv_lin + vh_lin
    out["VH_VV_ratio_linear"] = vh_lin / (vv_lin + eps)
    out["NRPB"] = (vv_lin - vh_lin) / (vv_lin + vh_lin + eps)
    out["VH_minus_VV_dB"] = out["VH_dB"] - out["VV_dB"]
    return out


def add_causal_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["sample_id", "orbit_pass"]
    out = out.sort_values(group_cols + ["date"]).copy()
    for base in ["VV_dB", "VH_dB", "RVI", "VH_VV_dB"]:
        if base not in out:
            continue
        grp = out.groupby(group_cols, observed=True)[base]
        out[f"{base}_lag1"] = grp.shift(1)
        out[f"{base}_delta1"] = out[base] - out[f"{base}_lag1"]
        out[f"{base}_delta2"] = out[base] - grp.shift(2)
        
        rolled = grp.rolling(window=3, min_periods=1)
        out[f"{base}_roll3_mean"] = rolled.mean().reset_index(level=group_cols, drop=True)
        out[f"{base}_roll3_std"] = rolled.std(ddof=0).reset_index(level=group_cols, drop=True)
    return out.sort_index()


class SourceOnlyProgressTransformer(BaseEstimator, TransformerMixin):


    def __init__(
        self,
        mode: str = PROGRESS_MODE,
        quantile: float = SOURCE_QUANTILE,
        fixed_max_dap: float = FIXED_MAX_DAP,
        fixed_max_gdd: float = FIXED_MAX_GDD,
        phase_boundaries: Tuple[float, float, float] = PHASE_BOUNDARIES,
    ) -> None:
        self.mode = mode
        self.quantile = quantile
        self.fixed_max_dap = fixed_max_dap
        self.fixed_max_gdd = fixed_max_gdd
        self.phase_boundaries = phase_boundaries

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        Xdf = pd.DataFrame(X).copy()
        if self.mode not in {"source_quantile", "fixed"}:
            raise ValueError("mode must be 'source_quantile' or 'fixed'")
        if self.mode == "fixed":
            self.max_dap_ = float(self.fixed_max_dap)
            self.max_gdd_ = float(self.fixed_max_gdd)
        else:
            dap = safe_numeric(Xdf["DAP_estimated"]).dropna()
            gdd = safe_numeric(Xdf["GDD_cum"]).dropna()
            self.max_dap_ = float(dap.quantile(self.quantile)) if len(dap) else self.fixed_max_dap
            self.max_gdd_ = float(gdd.quantile(self.quantile)) if len(gdd) else self.fixed_max_gdd
            if not np.isfinite(self.max_dap_) or self.max_dap_ <= 0:
                self.max_dap_ = float(self.fixed_max_dap)
            if not np.isfinite(self.max_gdd_) or self.max_gdd_ <= 0:
                self.max_gdd_ = float(self.fixed_max_gdd)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(X).copy()
        dap = safe_numeric(out["DAP_estimated"])
        gdd = safe_numeric(out["GDD_cum"])
        out["season_progress"] = np.clip(dap / self.max_dap_, 0.0, 1.25)
        out["GDD_progress"] = np.clip(gdd / self.max_gdd_, 0.0, 1.25)

        p = out["GDD_progress"].clip(0.0, 1.0)
        b1, b2, b3 = self.phase_boundaries
        conditions = [p < b1, (p >= b1) & (p < b2), (p >= b2) & (p < b3), p >= b3]
        labels = ["Bare_soil", "Canopy_growth", "Peak_canopy", "Senescence"]
        out["operational_phase"] = np.select(conditions, labels, default="Unknown")

        phase_start = np.select(conditions, [0.0, b1, b2, b3], default=0.0)
        phase_end = np.select(conditions, [b1, b2, b3, 1.0], default=1.0)
        denom = np.maximum(phase_end - phase_start, 1e-8)
        out["phase_progress"] = np.clip((p - phase_start) / denom, 0.0, 1.0)

        doy = pd.to_datetime(out["date"], errors="coerce").dt.dayofyear.astype(float)
        out["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
        out["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
        return out





STATIC_SAR = [
    "VV_dB", "VH_dB", "VH_VV_dB", "RVI", "angle",
    "Span_linear", "VH_VV_ratio_linear", "NRPB", "VH_minus_VV_dB",
]
DYNAMIC_SAR = [
    f"{base}_{suffix}"
    for base in ["VV_dB", "VH_dB", "RVI", "VH_VV_dB"]
    for suffix in ["lag1", "delta1", "delta2", "roll3_mean", "roll3_std"]
]
ENV = ["ERA5_T2M_C", "ERA5_SM", "Rain_10d_mm", "GDD_10d", "GDD_cum"]
TIME_PROGRESS = ["DAP_estimated", "season_progress", "GDD_progress", "phase_progress", "doy_sin", "doy_cos"]
CAT = ["operational_phase", "orbit_pass"]

FEATURE_SETS: Dict[str, List[str]] = {
    "U0_static": STATIC_SAR + TIME_PROGRESS,
    "U1_dynamic": STATIC_SAR + DYNAMIC_SAR + TIME_PROGRESS,
    "U2_dynamic_env": STATIC_SAR + DYNAMIC_SAR + ENV + TIME_PROGRESS,
}


def available_features(df: pd.DataFrame, requested: Sequence[str]) -> List[str]:
    
    
    generated = {"season_progress", "GDD_progress", "phase_progress", "doy_sin", "doy_cos"}
    return [c for c in requested if c in df.columns or c in generated]


def model_candidates() -> Dict[str, Tuple[BaseEstimator, Dict[str, List[object]]]]:
    candidates: Dict[str, Tuple[BaseEstimator, Dict[str, List[object]]]] = {}

    candidates["Ridge"] = (
        Ridge(),
        {"model__alpha": [0.1, 1.0, 10.0] if not QUICK_MODE else [1.0]},
    )
    candidates["ExtraTrees"] = (
        ExtraTreesRegressor(random_state=SEED, n_jobs=N_JOBS),
        {
            "model__n_estimators": [400] if not QUICK_MODE else [150],
            "model__max_depth": [None, 20] if not QUICK_MODE else [20],
            "model__min_samples_leaf": [1, 3] if not QUICK_MODE else [2],
            "model__max_features": [0.7, 1.0] if not QUICK_MODE else [0.8],
        },
    )
    candidates["RandomForest"] = (
        RandomForestRegressor(random_state=SEED, n_jobs=N_JOBS),
        {
            "model__n_estimators": [350] if not QUICK_MODE else [120],
            "model__max_depth": [18, None] if not QUICK_MODE else [18],
            "model__min_samples_leaf": [1, 3] if not QUICK_MODE else [2],
            "model__max_features": [0.7, 1.0] if not QUICK_MODE else [0.8],
        },
    )

    try:
        from lightgbm import LGBMRegressor
        candidates["LightGBM"] = (
            LGBMRegressor(random_state=SEED, n_jobs=N_JOBS, verbosity=-1),
            {
                "model__n_estimators": [400, 700] if not QUICK_MODE else [250],
                "model__learning_rate": [0.03, 0.06] if not QUICK_MODE else [0.05],
                "model__num_leaves": [31, 63] if not QUICK_MODE else [31],
                "model__min_child_samples": [20, 50] if not QUICK_MODE else [30],
            },
        )
    except Exception:
        print("[INFO] LightGBM unavailable; skipped.")

    try:
        from catboost import CatBoostRegressor
        candidates["CatBoost"] = (
            CatBoostRegressor(
                random_seed=SEED,
                verbose=False,
                allow_writing_files=False,
                thread_count=N_JOBS,
                loss_function="RMSE",
            ),
            {
                "model__iterations": [400, 700] if not QUICK_MODE else [250],
                "model__depth": [6, 8] if not QUICK_MODE else [6],
                "model__learning_rate": [0.03, 0.06] if not QUICK_MODE else [0.05],
                "model__l2_leaf_reg": [3, 10] if not QUICK_MODE else [5],
            },
        )
    except Exception:
        print("[INFO] CatBoost unavailable; skipped.")

    return candidates


def build_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    estimator: BaseEstimator,
) -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(numeric_features)),
            ("cat", categorical_pipe, list(categorical_features)),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("strict_progress", SourceOnlyProgressTransformer()),
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


def choose_best_source_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> Tuple[Pipeline, str, Dict[str, object], float]:
    unique_groups = pd.Series(groups).nunique()
    n_splits = min(INNER_CV_SPLITS, int(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two source region-year groups for inner CV.")
    cv = GroupKFold(n_splits=n_splits)

    best_estimator: Optional[Pipeline] = None
    best_name = ""
    best_params: Dict[str, object] = {}
    best_rmse = np.inf

    for name, (model, grid) in model_candidates().items():
        pipe = build_pipeline(numeric_features, categorical_features, model)
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=N_JOBS,
            refit=True,
            verbose=0,
            error_score="raise",
        )
        search.fit(X_train, y_train, groups=groups)
        current_rmse = -float(search.best_score_)
        print(f"      {name:<13} source-CV RMSE={current_rmse:.5f}")
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_estimator = search.best_estimator_
            best_name = name
            best_params = search.best_params_

    if best_estimator is None:
        raise RuntimeError("No model completed successfully.")
    return best_estimator, best_name, best_params, best_rmse





def run_strict_loryo(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    domains = (
        df[["region", "year"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["region", "year"])
        .itertuples(index=False, name=None)
    )
    domains = list(domains)
    if QUICK_MODE:
        domains = domains[:4]

    metric_rows: List[Dict[str, object]] = []
    pred_rows: List[pd.DataFrame] = []
    threshold_rows: List[Dict[str, object]] = []

    for d_idx, (test_region, test_year) in enumerate(domains, start=1):
        test_mask = (df["region"] == test_region) & (df["year"] == test_year)
        train = df.loc[~test_mask].copy()
        test = df.loc[test_mask].copy()
        print(f"\n[{d_idx}/{len(domains)}] Target: {test_region} {test_year}; train={len(train)}, test={len(test)}")

        
        
        common_cols = sorted(set(
            ["date", "DAP_estimated", "GDD_cum", "orbit_pass"]
            + sum(FEATURE_SETS.values(), [])
        ))
        common_cols = [c for c in common_cols if c in df.columns or c in {
            "season_progress", "GDD_progress", "phase_progress", "doy_sin", "doy_cos"
        }]
        
        raw_common_cols = [c for c in common_cols if c in df.columns]

        y_train = train["NDVI"].astype(float)
        y_test = test["NDVI"].astype(float)
        valid_train = y_train.notna()
        valid_test = y_test.notna()
        train = train.loc[valid_train].copy()
        test = test.loc[valid_test].copy()
        y_train = y_train.loc[valid_train]
        y_test = y_test.loc[valid_test]
        groups = train["region"].astype(str) + "__" + train["year"].astype(str)

        for fs_idx, (feature_set, requested) in enumerate(FEATURE_SETS.items(), start=1):
            numeric = available_features(df, requested)
            categorical = [c for c in CAT if c in df.columns or c == "operational_phase"]
            needed_raw = sorted(set(raw_common_cols + [c for c in numeric if c in df.columns] + [
                "date", "DAP_estimated", "GDD_cum", "orbit_pass"
            ]))
            X_train = train[needed_raw].copy()
            X_test = test[needed_raw].copy()

            print(f"   ({fs_idx}/3) {feature_set}: {len(numeric)} numeric + {len(categorical)} categorical")
            model, model_name, params, source_cv_rmse = choose_best_source_only(
                X_train, y_train, groups, numeric, categorical
            )
            pred = np.asarray(model.predict(X_test), dtype=float)

            this_rmse = rmse(y_test.to_numpy(), pred)
            this_mae = float(mean_absolute_error(y_test, pred))
            this_r2 = float(r2_score(y_test, pred))
            this_bias = float(np.mean(pred - y_test.to_numpy()))

            progress_step: SourceOnlyProgressTransformer = model.named_steps["strict_progress"]
            threshold_rows.append({
                "test_region": test_region,
                "test_year": int(test_year),
                "feature_set": feature_set,
                "progress_mode": progress_step.mode,
                "source_only_max_dap": progress_step.max_dap_,
                "source_only_max_gdd": progress_step.max_gdd_,
                "source_quantile": progress_step.quantile,
            })

            metric_rows.append({
                "analysis": "strict_operational_LORYO",
                "test_region": test_region,
                "test_year": int(test_year),
                "feature_set": feature_set,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "selected_model": model_name,
                "selected_params": json.dumps(params, ensure_ascii=False, default=str),
                "source_cv_rmse": source_cv_rmse,
                "rmse": this_rmse,
                "mae": this_mae,
                "r2": this_r2,
                "bias": this_bias,
                "target_sd": float(np.std(y_test, ddof=1)),
            })

            keep_meta = [c for c in ["sample_id", "date", "region", "year", "orbit_pass"] if c in test]
            p = test[keep_meta].copy().reset_index(drop=True)
            p["observed_ndvi"] = y_test.to_numpy()
            p["predicted_ndvi"] = pred
            p["residual_pred_minus_obs"] = pred - y_test.to_numpy()
            p["feature_set"] = feature_set
            p["selected_model"] = model_name
            pred_rows.append(p)
            print(f"      -> RMSE={this_rmse:.5f}, MAE={this_mae:.5f}, R2={this_r2:.5f}")

    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True), pd.DataFrame(threshold_rows)





def summarize_results(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for fs, g in metrics.groupby("feature_set", sort=False):
        p = predictions[predictions["feature_set"] == fs]
        rmse_ci = bootstrap_mean_ci(g["rmse"])
        mae_ci = bootstrap_mean_ci(g["mae"])
        r2_ci = bootstrap_mean_ci(g["r2"])
        rows.append({
            "analysis": "strict_operational_LORYO",
            "feature_set": fs,
            "n_tests": len(g),
            "mean_rmse": g["rmse"].mean(),
            "median_rmse": g["rmse"].median(),
            "rmse_ci_low": rmse_ci[0],
            "rmse_ci_high": rmse_ci[1],
            "mean_mae": g["mae"].mean(),
            "median_mae": g["mae"].median(),
            "mae_ci_low": mae_ci[0],
            "mae_ci_high": mae_ci[1],
            "mean_r2": g["r2"].mean(),
            "median_r2": g["r2"].median(),
            "r2_ci_low": r2_ci[0],
            "r2_ci_high": r2_ci[1],
            "negative_r2_rate": np.mean(g["r2"] < 0),
            "pooled_rmse": rmse(p["observed_ndvi"].to_numpy(), p["predicted_ndvi"].to_numpy()),
            "pooled_mae": mean_absolute_error(p["observed_ndvi"], p["predicted_ndvi"]),
            "pooled_r2": r2_score(p["observed_ndvi"], p["predicted_ndvi"]),
            "pooled_bias": np.mean(p["residual_pred_minus_obs"]),
            "pooled_n": len(p),
        })
    return pd.DataFrame(rows)


def paired_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("U0_static", "U1_dynamic"),
        ("U1_dynamic", "U2_dynamic_env"),
        ("U0_static", "U2_dynamic_env"),
    ]
    rows: List[Dict[str, object]] = []
    key = ["test_region", "test_year"]
    for a, b in comparisons:
        ga = metrics[metrics.feature_set == a].set_index(key)
        gb = metrics[metrics.feature_set == b].set_index(key)
        common = ga.index.intersection(gb.index)
        for metric in ["rmse", "mae", "r2"]:
            delta = gb.loc[common, metric].to_numpy() - ga.loc[common, metric].to_numpy()
            ci = bootstrap_mean_ci(delta)
            try:
                stat, pval = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                stat, pval = np.nan, np.nan
            better = delta < 0 if metric in {"rmse", "mae"} else delta > 0
            rows.append({
                "metric": metric,
                "comparison": f"{b} vs {a}",
                "n_pairs": len(delta),
                "mean_delta_B_minus_A": np.mean(delta),
                "median_delta_B_minus_A": np.median(delta),
                "bootstrap_ci_low": ci[0],
                "bootstrap_ci_high": ci[1],
                "wilcoxon_statistic": stat,
                "wilcoxon_p_value": pval,
                "fraction_B_better": np.mean(better),
            })
    return pd.DataFrame(rows)


def ndvi_bin_bias(pred: pd.DataFrame, feature_set: str = "U2_dynamic_env") -> pd.DataFrame:
    p = pred[pred.feature_set == feature_set].copy()
    bins = np.arange(0.0, 1.0001, 0.05)
    p["ndvi_bin"] = pd.cut(p["observed_ndvi"], bins=bins, include_lowest=True, right=False)
    out = (
        p.groupby("ndvi_bin", observed=True)
        .agg(
            n=("observed_ndvi", "size"),
            mean_observed=("observed_ndvi", "mean"),
            mean_predicted=("predicted_ndvi", "mean"),
            bias=("residual_pred_minus_obs", "mean"),
            rmse=("residual_pred_minus_obs", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            mae=("residual_pred_minus_obs", lambda x: float(np.mean(np.abs(x)))),
        )
        .reset_index()
    )
    out["ndvi_bin"] = out["ndvi_bin"].astype(str)
    return out





def save_hexbin(predictions: pd.DataFrame, summary: pd.DataFrame) -> None:
    fs = "U2_dynamic_env"
    p = predictions[predictions.feature_set == fs]
    s = summary[summary.feature_set == fs].iloc[0]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    hb = ax.hexbin(
        p["observed_ndvi"], p["predicted_ndvi"],
        gridsize=70, mincnt=1, bins="log", cmap="viridis",
    )
    lim_low = float(max(0.0, min(p["observed_ndvi"].min(), p["predicted_ndvi"].min())))
    lim_high = float(min(1.0, max(p["observed_ndvi"].max(), p["predicted_ndvi"].max())))
    ax.plot([lim_low, lim_high], [lim_low, lim_high], "--", linewidth=1.6, color="black", label="1:1 line")
    ax.set_xlim(lim_low, lim_high)
    ax.set_ylim(lim_low, lim_high)
    ax.set_xlabel("Observed NDVI")
    ax.set_ylabel("Predicted NDVI")
    ax.set_title("Strict operational out-of-domain predictions (U2)")
    text = (
        f"Pooled R² = {s.pooled_r2:.3f}\n"
        f"RMSE = {s.pooled_rmse:.3f}\n"
        f"MAE = {s.pooled_mae:.3f}\n"
        f"n = {int(s.pooled_n):,}"
    )
    ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="0.5"))
    ax.legend(loc="lower right", frameon=False)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("log10(count)")
    fig.tight_layout()
    fig.savefig(OUT / "Fig_01_U2_hexbin_observed_vs_predicted.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / "Fig_01_U2_hexbin_observed_vs_predicted.pdf", bbox_inches="tight")
    plt.close(fig)


def save_bias_curve(bin_df: pd.DataFrame) -> None:
    b = bin_df[bin_df.n >= 20].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axhline(0.0, linestyle="--", linewidth=1.2, color="black")
    ax.plot(b["mean_observed"], b["bias"], marker="o", linewidth=1.8)
    ax.set_xlabel("Observed NDVI (bin mean)")
    ax.set_ylabel("Mean bias (predicted − observed)")
    ax.set_title("Conditional prediction bias across NDVI range")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "Fig_02_U2_NDVI_bin_bias_curve.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / "Fig_02_U2_NDVI_bin_bias_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def save_r2_heatmap(metrics: pd.DataFrame) -> None:
    fs = "U2_dynamic_env"
    m = metrics[metrics.feature_set == fs].pivot(index="test_region", columns="test_year", values="r2")
    m = m.sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(8.3, 4.3))
    im = ax.imshow(m.to_numpy(), aspect="auto", vmin=min(0.0, np.nanmin(m.to_numpy())), vmax=0.85, cmap="viridis")
    ax.set_xticks(np.arange(len(m.columns)), labels=[str(int(x)) for x in m.columns])
    ax.set_yticks(np.arange(len(m.index)), labels=[x.replace("_", " ") for x in m.index])
    ax.set_xlabel("Target year")
    ax.set_ylabel("Target region")
    ax.set_title("Strict operational LORYO stability (U2 R²)")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            val = m.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val < 0.45 else "black", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Test-domain R²")
    fig.tight_layout()
    fig.savefig(OUT / "Fig_03_U2_region_year_R2_heatmap.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / "Fig_03_U2_region_year_R2_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def save_paired_domain_plot(metrics: pd.DataFrame) -> None:
    order = ["U0_static", "U1_dynamic", "U2_dynamic_env"]
    wide = metrics.pivot_table(index=["test_region", "test_year"], columns="feature_set", values="r2")
    wide = wide.dropna(subset=order)
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    x = np.arange(len(order))
    for _, row in wide.iterrows():
        ax.plot(x, row[order].to_numpy(), marker="o", linewidth=0.9, alpha=0.45)
    means = wide[order].mean(axis=0).to_numpy()
    ax.plot(x, means, marker="D", linewidth=2.8, color="black", label="Domain mean")
    ax.set_xticks(x, labels=["Static SAR", "Dynamic SAR", "Dynamic SAR + environment"])
    ax.set_ylabel("Test-domain R²")
    ax.set_title("Paired performance across region-year target domains")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "Fig_04_paired_domain_R2.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / "Fig_04_paired_domain_R2.pdf", bbox_inches="tight")
    plt.close(fig)


def save_summary_bar(summary: pd.DataFrame) -> None:
    order = ["U0_static", "U1_dynamic", "U2_dynamic_env"]
    s = summary.set_index("feature_set").loc[order]
    means = s["mean_r2"].to_numpy()
    low = means - s["r2_ci_low"].to_numpy()
    high = s["r2_ci_high"].to_numpy() - means
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(order))
    ax.bar(x, means, yerr=np.vstack([low, high]), capsize=5)
    ax.set_xticks(x, labels=["Static SAR", "Dynamic SAR", "Dynamic SAR + environment"])
    ax.set_ylabel("Mean test-domain R²")
    ax.set_title("Strict operational LORYO performance")
    ax.set_ylim(0, min(1.0, max(means + high) + 0.12))
    for i, v in enumerate(means):
        ax.text(i, v + high[i] + 0.015, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "Fig_05_model_comparison_R2_with_CI.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / "Fig_05_model_comparison_R2_with_CI.pdf", bbox_inches="tight")
    plt.close(fig)





def make_operational_audit(df: pd.DataFrame) -> pd.DataFrame:
    sample_fallback = df["sample_id"].astype(str).equals(df["region"].astype(str))
    rows = [
        {
            "check": "target_excluded",
            "status": "PASS",
            "detail": "NDVI is never included in any predictor set or progress calculation.",
        },
        {
            "check": "outer_target_excluded_from_progress_threshold",
            "status": "PASS",
            "detail": "Progress thresholds are fitted by a pipeline transformer on each training fold only.",
        },
        {
            "check": "inner_validation_excluded_from_progress_threshold",
            "status": "PASS",
            "detail": "Because the transformer is inside GridSearchCV, each inner validation fold is excluded when thresholds are fitted.",
        },
        {
            "check": "dynamic_features_causal",
            "status": "PASS",
            "detail": "Only lag(1/2), backward differences, and non-centered backward rolling windows are used.",
        },
        {
            "check": "fixed_phase_boundaries",
            "status": "PASS",
            "detail": f"Operational phase uses fixed GDD-progress boundaries {PHASE_BOUNDARIES}; no target NDVI is used.",
        },
        {
            "check": "sample_identifier_quality",
            "status": "CHECK" if sample_fallback else "PASS",
            "detail": "sample_id fallback equals region; verify plot/pixel identifiers." if sample_fallback else "A dedicated sample identifier was found.",
        },
        {
            "check": "weather_window_causality",
            "status": "CHECK",
            "detail": "Confirm source Rain_10d_mm, GDD_10d and ERA5 variables contain only information available on or before the SAR prediction date.",
        },
        {
            "check": "claim_scope",
            "status": "PASS",
            "detail": "This version supports an operational/near-real-time claim for progress normalization, subject to weather-window causality confirmation.",
        },
    ]
    return pd.DataFrame(rows)


def export_excel(tables: Dict[str, pd.DataFrame]) -> None:
    final = OUT / "strict_operational_complete_analysis.xlsx"
    temp = OUT / "strict_operational_complete_analysis.tmp.xlsx"
    if temp.exists():
        temp.unlink()
    try:
        with pd.ExcelWriter(temp, engine="openpyxl") as writer:
            wrote = False
            for sheet, df in tables.items():
                if df is None or df.empty:
                    continue
                df.head(1_048_575).to_excel(
                    writer,
                    sheet_name=sanitize_sheet_name(sheet),
                    index=False,
                )
                wrote = True
            if not wrote:
                pd.DataFrame({"message": ["No result tables were produced."]}).to_excel(
                    writer, sheet_name="README", index=False
                )
        if final.exists():
            final.unlink()
        temp.replace(final)
    except Exception as exc:
        if temp.exists():
            temp.unlink()
        print(f"[WARNING] Excel export failed, CSV files remain valid: {exc}")


def main() -> None:
    print("=" * 78)
    print("STRICT OPERATIONAL LORYO + PUBLICATION FIGURES")
    print(f"Script directory : {SCRIPT_DIR}")
    print(f"Output directory : {OUT}")
    print(f"Progress mode    : {PROGRESS_MODE}")
    print("=" * 78)

    input_file = locate_input_file()
    print(f"Input file       : {input_file}")
    raw = read_csv_robust(input_file)
    df = standardize_columns(raw)
    df = add_static_sar_features(df)
    df = add_causal_dynamic_features(df)
    df = df.dropna(subset=["date", "region", "year", "NDVI", "VV_dB", "VH_dB"]).copy()
    df["year"] = df["year"].astype(int)

    audit = make_operational_audit(df)
    audit.to_csv(OUT / "00_strict_operational_audit.csv", index=False, encoding="utf-8-sig")

    metrics, predictions, thresholds = run_strict_loryo(df)
    summary = summarize_results(metrics, predictions)
    significance = paired_tests(metrics)
    bin_bias = ndvi_bin_bias(predictions)

    metrics.to_csv(OUT / "01_strict_operational_domain_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "02_strict_operational_summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / "03_strict_operational_predictions.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(OUT / "04_source_only_progress_thresholds.csv", index=False, encoding="utf-8-sig")
    significance.to_csv(OUT / "05_paired_significance.csv", index=False, encoding="utf-8-sig")
    bin_bias.to_csv(OUT / "06_U2_NDVI_bin_bias.csv", index=False, encoding="utf-8-sig")

    save_hexbin(predictions, summary)
    save_bias_curve(bin_bias)
    save_r2_heatmap(metrics)
    save_paired_domain_plot(metrics)
    save_summary_bar(summary)

    export_excel({
        "operational_audit": audit,
        "domain_results": metrics,
        "overall_summary": summary,
        "predictions": predictions,
        "source_thresholds": thresholds,
        "paired_significance": significance,
        "NDVI_bin_bias": bin_bias,
    })

    print("\nCompleted successfully.")
    print("All new results are isolated in:")
    print(OUT)
    print("\nMain summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
