





























































from __future__ import annotations

import gc
import importlib.util
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, pearsonr
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.colorbar as colorbar
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler




SEED = 20260808
N_JOBS = 4
QUICK_MODE = False
INNER_CV_SPLITS = 4

BOOTSTRAP_REPS = 2000
BIAS_BOOTSTRAP_REPS = 1000
PERMUTATION_REPEATS = 8
PERMUTATION_MAX_ROWS = 5000
MIN_BIN_N = 20


LORYO_FEATURES = [
    "T0_phenology_only",
    "S0_current_SAR_only",
    "U0_static",
    "U1_dynamic",
    "U2_dynamic_env",
]

SENSITIVITY_FEATURES = ["U0_static", "U1_dynamic", "U2_dynamic_env"]

RUN_LORYO = True
RUN_LORO = True
RUN_LOYO = True
RUN_PERMUTATION_IMPORTANCE = True
RUN_DOMAIN_PCA = True



RUN_MANTEL_NETWORK = True



MANTEL_PERMUTATIONS = 199



MANTEL_MAX_ROWS = 180
MANTEL_SHOW_NONSIGNIFICANT = True

RESUME = True

SCRIPT_DIR = Path(__file__).resolve().parent









BASE_SCRIPT_NAME = "cross_domain_modeling.py"

BASE_SCRIPT = SCRIPT_DIR / BASE_SCRIPT_NAME
if not BASE_SCRIPT.exists():
    raise FileNotFoundError(f"Cannot find required module: {BASE_SCRIPT}")

PROJECT_DIR = SCRIPT_DIR.parent
OUT = PROJECT_DIR / "results"

CHECKPOINT_DIR = OUT / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[PATH] Integrated script : {Path(__file__).resolve()}")
print(f"[PATH] Project directory : {PROJECT_DIR}")
print(f"[PATH] Base script       : {BASE_SCRIPT}")
print(f"[PATH] Output directory  : {OUT}")

random.seed(SEED)
np.random.seed(SEED)




def load_base_module(path: Path):
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            f"Base script not found: {path}"
        )
    path = Path(path)
    spec = importlib.util.spec_from_file_location("strict_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import base script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base = load_base_module(BASE_SCRIPT)
base.SEED = SEED
base.N_JOBS = N_JOBS
base.QUICK_MODE = QUICK_MODE
base.INNER_CV_SPLITS = INNER_CV_SPLITS

STATIC_SAR = list(base.STATIC_SAR)
DYNAMIC_SAR = list(base.DYNAMIC_SAR)
ENV = list(base.ENV)
TIME_PROGRESS = list(base.TIME_PROGRESS)

FEATURE_SETS: Dict[str, Dict[str, List[str]]] = {
    "T0_phenology_only": {
        "numeric": TIME_PROGRESS,
        "categorical": ["operational_phase"],
    },
    "S0_current_SAR_only": {
        "numeric": STATIC_SAR,
        "categorical": ["orbit_pass"],
    },
    "U0_static": {
        "numeric": STATIC_SAR + TIME_PROGRESS,
        "categorical": ["operational_phase", "orbit_pass"],
    },
    "U1_dynamic": {
        "numeric": STATIC_SAR + DYNAMIC_SAR + TIME_PROGRESS,
        "categorical": ["operational_phase", "orbit_pass"],
    },
    "U2_dynamic_env": {
        "numeric": STATIC_SAR + DYNAMIC_SAR + ENV + TIME_PROGRESS,
        "categorical": ["operational_phase", "orbit_pass"],
    },
}

FEATURE_GROUPS = {
    "phenology_time": set(TIME_PROGRESS + ["operational_phase"]),
    "current_SAR": set(STATIC_SAR + ["orbit_pass"]),
    "dynamic_SAR": set(DYNAMIC_SAR),
    "environment": set(ENV),
}

CORE_COMPARISONS = {
    "LORYO": [
        ("T0_phenology_only", "U0_static"),
        ("S0_current_SAR_only", "U0_static"),
        ("U0_static", "U1_dynamic"),
        ("U1_dynamic", "U2_dynamic_env"),
        ("U0_static", "U2_dynamic_env"),
    ],
    "LORO": [
        ("U0_static", "U1_dynamic"),
        ("U1_dynamic", "U2_dynamic_env"),
        ("U0_static", "U2_dynamic_env"),
    ],
    "LOYO": [
        ("U0_static", "U1_dynamic"),
        ("U1_dynamic", "U2_dynamic_env"),
        ("U0_static", "U2_dynamic_env"),
    ],
}




def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def bootstrap_mean_ci(values: Sequence[float], reps: int = BOOTSTRAP_REPS) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(SEED)
    boot = np.empty(reps, dtype=float)
    for i in range(reps):
        boot[i] = np.mean(rng.choice(x, size=len(x), replace=True))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return adjusted
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def linear_calibration(y, pred) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    y = y[valid]
    pred = pred[valid]
    if len(y) < 3 or np.std(y) <= 1e-12:
        return {
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
            "calibration_r2": np.nan,
        }
    slope, intercept = np.polyfit(y, pred, 1)
    fitted = intercept + slope * y
    return {
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "calibration_r2": float(r2_score(pred, fitted)),
    }


def add_causal_dynamic_features_strict(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["region", "year", "sample_id", "orbit_pass"]
    out = out.sort_values(group_cols + ["date"]).copy()
    for feature in ["VV_dB", "VH_dB", "RVI", "VH_VV_dB"]:
        if feature not in out.columns:
            continue
        g = out.groupby(group_cols, observed=True, sort=False)[feature]
        lag1 = g.shift(1)
        out[f"{feature}_lag1"] = lag1
        out[f"{feature}_delta1"] = out[feature] - lag1
        out[f"{feature}_delta2"] = out[feature] - g.shift(2)
        rolling = g.rolling(window=3, min_periods=1)
        out[f"{feature}_roll3_mean"] = rolling.mean().reset_index(level=group_cols, drop=True)
        out[f"{feature}_roll3_std"] = rolling.std(ddof=0).reset_index(level=group_cols, drop=True)
    return out.sort_index()


def available_features(df: pd.DataFrame, requested: Sequence[str]) -> List[str]:
    generated = {"season_progress", "GDD_progress", "phase_progress", "doy_sin", "doy_cos"}
    return [c for c in requested if c in df.columns or c in generated]


def required_raw_columns(df: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str]) -> List[str]:
    generated = {
        "season_progress", "GDD_progress", "phase_progress", "doy_sin", "doy_cos",
        "operational_phase",
    }
    required = {"date", "DAP_estimated", "GDD_cum", "orbit_pass"}
    required.update(c for c in numeric if c not in generated)
    required.update(c for c in categorical if c not in generated)
    missing = sorted(c for c in required if c not in df.columns)
    if missing:
        raise KeyError(f"Required raw columns are missing: {missing}")
    return sorted(required)


def feature_group(feature: str) -> str:
    for group, names in FEATURE_GROUPS.items():
        if feature in names:
            return group
    return "other"


def safe_text(x) -> str:
    return str(x).replace(" ", "_").replace("/", "-").replace("\\", "-")




def make_outer_splits(df: pd.DataFrame, scheme: str):
    if scheme == "LORYO":
        domains = list(
            df[["region", "year"]].drop_duplicates().sort_values(["region", "year"])
            .itertuples(index=False, name=None)
        )
        if QUICK_MODE:
            domains = domains[:2]
        for region, year in domains:
            mask = df["region"].eq(region) & df["year"].eq(year)
            yield {
                "test_id": f"{region}__{int(year)}",
                "test_region": str(region),
                "test_year": int(year),
                "train_mask": ~mask,
                "test_mask": mask,
            }
    elif scheme == "LORO":
        regions = sorted(df["region"].dropna().astype(str).unique())
        if QUICK_MODE:
            regions = regions[:2]
        for region in regions:
            mask = df["region"].astype(str).eq(region)
            yield {
                "test_id": str(region),
                "test_region": str(region),
                "test_year": np.nan,
                "train_mask": ~mask,
                "test_mask": mask,
            }
    elif scheme == "LOYO":
        years = sorted(df["year"].dropna().astype(int).unique())
        if QUICK_MODE:
            years = years[:2]
        for year in years:
            mask = df["year"].eq(year)
            yield {
                "test_id": str(int(year)),
                "test_region": "ALL_REGIONS",
                "test_year": int(year),
                "train_mask": ~mask,
                "test_mask": mask,
            }
    else:
        raise ValueError(f"Unknown scheme: {scheme}")




def checkpoint_path(kind: str, scheme: str, feature_set: str, test_id: str) -> Path:
    name = f"{kind}__{scheme}__{feature_set}__{safe_text(test_id)}.csv"
    return CHECKPOINT_DIR / name


def completed_metric_exists(scheme: str, feature_set: str, test_id: str) -> bool:
    return checkpoint_path("metrics", scheme, feature_set, test_id).exists()


def read_checkpoints(kind: str) -> pd.DataFrame:
    files = sorted(CHECKPOINT_DIR.glob(f"{kind}__*.csv"))
    frames = []
    for p in files:
        try:
            frames.append(pd.read_csv(p, encoding="utf-8-sig"))
        except Exception as exc:
            print(f"[WARNING] Cannot read checkpoint {p.name}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()




def extract_test_permutation_importance(
    fitted_pipeline: Pipeline,
    X_test_raw: pd.DataFrame,
    y_test: pd.Series,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    test_region: str,
    test_year: int,
    selected_model: str,
) -> pd.DataFrame:
    progress = fitted_pipeline.named_steps["strict_progress"]
    transformed = progress.transform(X_test_raw)
    columns = list(numeric_features) + list(categorical_features)
    X_eval = transformed[columns].copy().reset_index(drop=True)
    y_eval = y_test.reset_index(drop=True).copy()

    if len(X_eval) > PERMUTATION_MAX_ROWS:
        rng = np.random.default_rng(SEED + int(test_year))
        take = np.sort(rng.choice(len(X_eval), size=PERMUTATION_MAX_ROWS, replace=False))
        X_eval = X_eval.iloc[take].reset_index(drop=True)
        y_eval = y_eval.iloc[take].reset_index(drop=True)

    fitted_tail = Pipeline([
        ("preprocessor", fitted_pipeline.named_steps["preprocessor"]),
        ("model", fitted_pipeline.named_steps["model"]),
    ])
    result = permutation_importance(
        fitted_tail,
        X_eval,
        y_eval,
        scoring="neg_root_mean_squared_error",
        n_repeats=PERMUTATION_REPEATS,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    out = pd.DataFrame({
        "feature": columns,
        "importance_rmse_increase": result.importances_mean,
        "importance_sd": result.importances_std,
    })
    out["importance_positive"] = out["importance_rmse_increase"].clip(lower=0)
    total = out["importance_positive"].sum()
    out["importance_normalized"] = out["importance_positive"] / total if total > 0 else np.nan
    out["feature_group"] = out["feature"].map(feature_group)
    out["scheme"] = "LORYO"
    out["test_region"] = test_region
    out["test_year"] = int(test_year)
    out["feature_set"] = "U2_dynamic_env"
    out["selected_model"] = selected_model
    out["n_importance_rows"] = len(X_eval)
    return out.sort_values("importance_rmse_increase", ascending=False).reset_index(drop=True)




def run_scheme(df: pd.DataFrame, scheme: str, feature_names: Sequence[str]):
    splits = list(make_outer_splits(df, scheme))
    for i, split in enumerate(splits, start=1):
        test_id = split["test_id"]
        train = df.loc[split["train_mask"]].dropna(subset=["NDVI"]).copy()
        test = df.loc[split["test_mask"]].dropna(subset=["NDVI"]).copy()
        if len(train) == 0 or len(test) == 0:
            continue

        
        groups = train["region"].astype(str) + "__" + train["year"].astype(str)

        print(f"\n[{scheme} {i}/{len(splits)}] target={test_id}; train={len(train)}, test={len(test)}")

        for feature_set in feature_names:
            metric_cp = checkpoint_path("metrics", scheme, feature_set, test_id)
            if RESUME and metric_cp.exists():
                print(f"  [SKIP] {feature_set}: checkpoint exists")
                continue

            definition = FEATURE_SETS[feature_set]
            numeric = available_features(df, definition["numeric"])
            categorical = [c for c in definition["categorical"] if c in df.columns or c == "operational_phase"]
            raw_columns = required_raw_columns(df, numeric, categorical)
            X_train = train[raw_columns].copy()
            X_test = test[raw_columns].copy()
            y_train = train["NDVI"].astype(float)
            y_test = test["NDVI"].astype(float)

            print(f"  {feature_set}: {len(numeric)} numeric, {len(categorical)} categorical")
            try:
                model, model_name, params, source_cv_rmse = base.choose_best_source_only(
                    X_train, y_train, groups, numeric, categorical
                )
                pred = np.clip(np.asarray(model.predict(X_test), dtype=float), 0.0, 1.0)

                row = {
                    "analysis": "strict_operational_v3",
                    "scheme": scheme,
                    "test_id": test_id,
                    "test_region": split["test_region"],
                    "test_year": split["test_year"],
                    "feature_set": feature_set,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "selected_model": model_name,
                    "selected_params": json.dumps(params, ensure_ascii=False, default=str),
                    "source_cv_rmse": float(source_cv_rmse),
                    "rmse": rmse(y_test.to_numpy(), pred),
                    "mae": float(mean_absolute_error(y_test, pred)),
                    "r2": float(r2_score(y_test, pred)),
                    "bias": float(np.mean(pred - y_test.to_numpy())),
                    "target_sd": float(np.std(y_test, ddof=1)),
                    "status": "PASS",
                }
                row.update(linear_calibration(y_test.to_numpy(), pred))
                pd.DataFrame([row]).to_csv(metric_cp, index=False, encoding="utf-8-sig")

                progress = model.named_steps["strict_progress"]
                threshold = pd.DataFrame([{
                    "scheme": scheme,
                    "test_id": test_id,
                    "test_region": split["test_region"],
                    "test_year": split["test_year"],
                    "feature_set": feature_set,
                    "progress_mode": progress.mode,
                    "source_only_max_dap": float(progress.max_dap_),
                    "source_only_max_gdd": float(progress.max_gdd_),
                    "source_quantile": float(progress.quantile),
                }])
                threshold.to_csv(
                    checkpoint_path("thresholds", scheme, feature_set, test_id),
                    index=False, encoding="utf-8-sig"
                )

                meta = [c for c in ["sample_id", "date", "region", "year", "orbit_pass"] if c in test.columns]
                p = test[meta].copy().reset_index(drop=True)
                p["scheme"] = scheme
                p["test_id"] = test_id
                p["test_region"] = split["test_region"]
                p["test_year"] = split["test_year"]
                p["observed_ndvi"] = y_test.to_numpy()
                p["predicted_ndvi"] = pred
                p["residual_pred_minus_obs"] = pred - y_test.to_numpy()
                p["feature_set"] = feature_set
                p["selected_model"] = model_name
                p.to_csv(
                    checkpoint_path("predictions", scheme, feature_set, test_id),
                    index=False, encoding="utf-8-sig"
                )

                if (
                    RUN_PERMUTATION_IMPORTANCE and scheme == "LORYO" and
                    feature_set == "U2_dynamic_env" and pd.notna(split["test_year"])
                ):
                    imp_cp = checkpoint_path("importance", scheme, feature_set, test_id)
                    if not (RESUME and imp_cp.exists()):
                        imp = extract_test_permutation_importance(
                            model, X_test, y_test, numeric, categorical,
                            str(split["test_region"]), int(split["test_year"]), model_name,
                        )
                        imp.to_csv(imp_cp, index=False, encoding="utf-8-sig")

                print(
                    f"      R2={row['r2']:.4f}, RMSE={row['rmse']:.4f}, "
                    f"slope={row['calibration_slope']:.3f}, model={model_name}"
                )

            except Exception as exc:
                fail = pd.DataFrame([{
                    "analysis": "strict_operational_v3",
                    "scheme": scheme,
                    "test_id": test_id,
                    "test_region": split["test_region"],
                    "test_year": split["test_year"],
                    "feature_set": feature_set,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "status": "FAIL",
                    "message": repr(exc),
                }])
                fail.to_csv(metric_cp, index=False, encoding="utf-8-sig")
                print(f"      [FAIL] {exc}")
                raise
            finally:
                for name in ["model", "X_train", "X_test", "y_train", "y_test"]:
                    if name in locals():
                        try:
                            del locals()[name]
                        except Exception:
                            pass
                gc.collect()




def summarize_all(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics[metrics.get("status", "PASS").eq("PASS")].copy()
    rows = []
    for (scheme, feature_set), g in metrics.groupby(["scheme", "feature_set"], sort=False):
        p = predictions[(predictions["scheme"].eq(scheme)) & (predictions["feature_set"].eq(feature_set))]
        row = {
            "analysis": "strict_operational_v3",
            "scheme": scheme,
            "feature_set": feature_set,
            "n_tests": int(len(g)),
            "mean_rmse": float(g["rmse"].mean()),
            "median_rmse": float(g["rmse"].median()),
            "mean_mae": float(g["mae"].mean()),
            "median_mae": float(g["mae"].median()),
            "mean_r2": float(g["r2"].mean()),
            "median_r2": float(g["r2"].median()),
            "negative_r2_rate": float(np.mean(g["r2"] < 0)),
            "mean_calibration_slope": float(g["calibration_slope"].mean()),
            "mean_calibration_intercept": float(g["calibration_intercept"].mean()),
        }
        for metric in ["rmse", "mae", "r2", "calibration_slope"]:
            lo, hi = bootstrap_mean_ci(g[metric])
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        if not p.empty:
            row.update({
                "pooled_rmse": rmse(p["observed_ndvi"], p["predicted_ndvi"]),
                "pooled_mae": float(mean_absolute_error(p["observed_ndvi"], p["predicted_ndvi"])),
                "pooled_r2": float(r2_score(p["observed_ndvi"], p["predicted_ndvi"])),
                "pooled_bias": float(p["residual_pred_minus_obs"].mean()),
                "pooled_n": int(len(p)),
            })
            row.update({f"pooled_{k}": v for k, v in linear_calibration(
                p["observed_ndvi"].to_numpy(), p["predicted_ndvi"].to_numpy()
            ).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests_all(metrics: pd.DataFrame) -> pd.DataFrame:
    m = metrics[metrics.get("status", "PASS").eq("PASS")].copy()
    rows = []
    for scheme, comparisons in CORE_COMPARISONS.items():
        d = m[m["scheme"].eq(scheme)].copy()
        if d.empty:
            continue
        for a, b in comparisons:
            ga = d[d["feature_set"].eq(a)].set_index("test_id")
            gb = d[d["feature_set"].eq(b)].set_index("test_id")
            common = ga.index.intersection(gb.index)
            if len(common) == 0:
                continue
            for metric in ["rmse", "mae", "r2", "calibration_slope"]:
                a_vals = ga.loc[common, metric].to_numpy(dtype=float)
                b_vals = gb.loc[common, metric].to_numpy(dtype=float)
                delta = b_vals - a_vals
                ci_low, ci_high = bootstrap_mean_ci(delta)
                try:
                    statistic, p_value = wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
                except ValueError:
                    statistic, p_value = np.nan, np.nan
                if metric in {"rmse", "mae"}:
                    better = delta < 0
                elif metric == "calibration_slope":
                    better = np.abs(b_vals - 1) < np.abs(a_vals - 1)
                else:
                    better = delta > 0
                rows.append({
                    "scheme": scheme,
                    "metric": metric,
                    "comparison": f"{b} vs {a}",
                    "model_A": a,
                    "model_B": b,
                    "n_pairs": int(len(delta)),
                    "mean_delta_B_minus_A": float(np.mean(delta)),
                    "median_delta_B_minus_A": float(np.median(delta)),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_value_raw": p_value,
                    "fraction_B_better": float(np.mean(better)),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["wilcoxon_p_value_holm"] = np.nan
    
    for (_, _), idx in out.groupby(["scheme", "metric"]).groups.items():
        out.loc[idx, "wilcoxon_p_value_holm"] = holm_adjust(out.loc[idx, "wilcoxon_p_value_raw"])
    return out




def cluster_bootstrap_bias_curve(predictions: pd.DataFrame) -> pd.DataFrame:
    p = predictions[
        predictions["scheme"].eq("LORYO") & predictions["feature_set"].eq("U2_dynamic_env")
    ].copy()
    if p.empty:
        return pd.DataFrame()
    p["domain"] = p["region"].astype(str) + "__" + p["year"].astype(str)
    bins = np.arange(0.0, 1.0001, 0.05)
    p["bin_left"] = pd.cut(
        p["observed_ndvi"], bins=bins, include_lowest=True, right=False, labels=bins[:-1]
    )
    p = p.dropna(subset=["bin_left"])
    p["bin_left"] = p["bin_left"].astype(float)

    domain_bin = (
        p.groupby(["domain", "bin_left"], observed=True)
        .agg(
            n=("observed_ndvi", "size"),
            sum_observed=("observed_ndvi", "sum"),
            sum_predicted=("predicted_ndvi", "sum"),
            sum_residual=("residual_pred_minus_obs", "sum"),
            sum_abs_residual=("residual_pred_minus_obs", lambda x: float(np.abs(x).sum())),
            sum_sq_residual=("residual_pred_minus_obs", lambda x: float(np.square(x).sum())),
        )
        .reset_index()
    )
    domains = sorted(p["domain"].unique())
    rng = np.random.default_rng(SEED)
    rows = []
    for left, g in domain_bin.groupby("bin_left", sort=True):
        n = int(g["n"].sum())
        if n < MIN_BIN_N:
            continue
        lookup = g.set_index("domain")
        boot_bias = np.empty(BIAS_BOOTSTRAP_REPS)
        for rep in range(BIAS_BOOTSTRAP_REPS):
            sampled = rng.choice(domains, size=len(domains), replace=True)
            sampled_rows = lookup.reindex(sampled).dropna()
            total_n = sampled_rows["n"].sum()
            boot_bias[rep] = sampled_rows["sum_residual"].sum() / total_n if total_n > 0 else np.nan
        finite = boot_bias[np.isfinite(boot_bias)]
        ci_low, ci_high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
        rows.append({
            "feature_set": "U2_dynamic_env",
            "bin_left": float(left),
            "bin_right": float(left + 0.05),
            "n": n,
            "n_domains": int(g["domain"].nunique()),
            "mean_observed": float(g["sum_observed"].sum() / n),
            "mean_predicted": float(g["sum_predicted"].sum() / n),
            "bias": float(g["sum_residual"].sum() / n),
            "bias_ci_low": float(ci_low),
            "bias_ci_high": float(ci_high),
            "rmse": float(np.sqrt(g["sum_sq_residual"].sum() / n)),
            "mae": float(g["sum_abs_residual"].sum() / n),
        })
    return pd.DataFrame(rows)




def summarize_importance(importance: pd.DataFrame):
    if importance.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ranked = importance.copy()
    ranked["rank_within_domain"] = ranked.groupby(["test_region", "test_year"])[
        "importance_normalized"
    ].rank(ascending=False, method="min")
    ranked["in_top10"] = ranked["rank_within_domain"] <= 10

    feature_summary = (
        ranked.groupby(["feature", "feature_group"], observed=True)
        .agg(
            mean_importance=("importance_normalized", "mean"),
            median_importance=("importance_normalized", "median"),
            sd_importance=("importance_normalized", "std"),
            mean_rmse_increase=("importance_rmse_increase", "mean"),
            top10_frequency=("in_top10", "mean"),
            n_domains=("test_year", "size"),
        )
        .reset_index()
        .sort_values("mean_importance", ascending=False)
    )

    group_domain = (
        ranked.groupby(["test_region", "test_year", "feature_group"], observed=True)[
            "importance_normalized"
        ].sum().reset_index()
    )
    group_summary = (
        group_domain.groupby("feature_group", observed=True)
        .agg(
            mean_group_importance=("importance_normalized", "mean"),
            median_group_importance=("importance_normalized", "median"),
            sd_group_importance=("importance_normalized", "std"),
            n_domains=("test_year", "size"),
        )
        .reset_index()
        .sort_values("mean_group_importance", ascending=False)
    )
    return ranked, feature_summary, group_summary




def build_domain_environment_pca(df: pd.DataFrame):
    candidate = [
        "ERA5_T2M_C", "ERA5_SM", "Rain_10d_mm", "GDD_10d", "GDD_cum",
        "VV_dB", "VH_dB", "RVI", "VH_VV_dB",
    ]
    features = [c for c in candidate if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()]
    if len(features) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    work = df[["region", "year"] + features].copy()
    for c in features:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    agg_dict = {}
    for c in features:
        agg_dict[f"{c}_mean"] = (c, "mean")
        agg_dict[f"{c}_sd"] = (c, "std")
    domain = work.groupby(["region", "year"], observed=True).agg(**agg_dict).reset_index()
    pca_cols = [c for c in domain.columns if c not in ["region", "year"]]
    X = domain[pca_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    pca = PCA(n_components=min(4, Z.shape[1], Z.shape[0]), random_state=SEED)
    scores = pca.fit_transform(Z)
    for i in range(scores.shape[1]):
        domain[f"PC{i+1}"] = scores[:, i]
    domain["domain"] = domain["region"].astype(str) + "__" + domain["year"].astype(str)

    loadings = pd.DataFrame({"variable": pca_cols})
    for i in range(pca.components_.shape[0]):
        loadings[f"PC{i+1}_loading"] = pca.components_[i]

    variance = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    })
    return domain, loadings, variance






MANTEL_COLORS = {
    "heatmap_negative": "#A96A46",   
    "heatmap_zero": "#F7F4EE",       
    "heatmap_positive": "#2D8C82",   
    "nodes": "#8B4F16",
    "p_lt_001": "#BE8733",           
    "p_001_005": "#75C9BF",          
    "p_ge_005": "#C8C9C9",           
    "ink": "#171717",
}


def mantel_feature_spec(df: pd.DataFrame):

















    phenology = [c for c in ["DAP_estimated", "GDD_cum"] if c in df.columns]

    current_sar = [
        c for c in ["VV_dB", "VH_dB", "RVI", "VH_VV_dB"]
        if c in df.columns
    ]

    
    
    dynamic_sar = [
        c for c in [
            "VV_dB_delta1", "VH_dB_delta1",
            "RVI_delta1", "VH_VV_dB_delta1",
            "VV_dB_lag1", "VH_dB_lag1",
        ]
        if c in df.columns
    ]

    environment = [
        c for c in ["ERA5_T2M_C", "ERA5_SM", "Rain_10d_mm", "GDD_10d"]
        if c in df.columns
    ]

    response = ["NDVI"] if "NDVI" in df.columns else []

    
    left_features = phenology + current_sar + dynamic_sar + environment
    left_features = list(dict.fromkeys(left_features))[:18]

    blocks = {
        "P": phenology,
        "S": current_sar,
        "D": dynamic_sar,
        "E": environment,
        "Y": response,
    }
    blocks = {k: v for k, v in blocks.items() if len(v) > 0}

    pretty = {
        "DAP_estimated": "DAP",
        "GDD_cum": "GDDcum",
        "VV_dB": "VV",
        "VH_dB": "VH",
        "RVI": "RVI",
        "VH_VV_dB": "VH/VV",
        "VV_dB_delta1": "ΔVV",
        "VH_dB_delta1": "ΔVH",
        "RVI_delta1": "ΔRVI",
        "VH_VV_dB_delta1": "ΔVH/VV",
        "VV_dB_lag1": "VV lag1",
        "VH_dB_lag1": "VH lag1",
        "ERA5_T2M_C": "T2m",
        "ERA5_SM": "Soil M.",
        "Rain_10d_mm": "Rain10d",
        "GDD_10d": "GDD10d",
    }

    block_labels = {
        "P": "Phenology",
        "S": "Current SAR",
        "D": "Dynamic SAR",
        "E": "Environment",
        "Y": "NDVI",
    }

    return left_features, blocks, pretty, block_labels


def _numeric_complete(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    z = df[list(columns)].copy()
    for c in columns:
        z[c] = pd.to_numeric(z[c], errors="coerce")
    return z.replace([np.inf, -np.inf], np.nan).dropna()


def _balanced_region_year_sample(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    max_rows: int = MANTEL_MAX_ROWS,
) -> pd.DataFrame:






    meta = [c for c in ["region", "year", "sample_id", "date"] if c in df.columns]
    columns = list(dict.fromkeys(meta + list(required_columns)))
    work = df[columns].copy()

    for c in required_columns:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=list(required_columns)).copy()

    if len(work) <= max_rows:
        return work.reset_index(drop=True)

    if "region" not in work.columns or "year" not in work.columns:
        return work.sample(n=max_rows, random_state=SEED).reset_index(drop=True)

    work["_domain"] = work["region"].astype(str) + "__" + work["year"].astype(str)
    domains = sorted(work["_domain"].unique())
    per_domain = max(2, max_rows // max(1, len(domains)))

    parts = []
    for i, domain in enumerate(domains):
        g = work[work["_domain"].eq(domain)]
        n_take = min(len(g), per_domain)
        if n_take > 0:
            parts.append(g.sample(n=n_take, random_state=SEED + i))

    sampled = pd.concat(parts, ignore_index=True) if parts else work.iloc[0:0].copy()

    
    if len(sampled) < max_rows:
        used_index = set(sampled.index.tolist())
        remaining = work.drop(index=[i for i in used_index if i in work.index], errors="ignore")
        if len(remaining) > 0:
            n_more = min(max_rows - len(sampled), len(remaining))
            sampled = pd.concat(
                [sampled, remaining.sample(n=n_more, random_state=SEED + 999)],
                ignore_index=True,
            )

    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=SEED).reset_index(drop=True)

    return sampled.drop(columns=["_domain"], errors="ignore").reset_index(drop=True)


def _pearson_matrix(data: pd.DataFrame):
    n = data.shape[1]
    corr = np.full((n, n), np.nan)
    pvals = np.full((n, n), np.nan)

    for i in range(n):
        xi = pd.to_numeric(data.iloc[:, i], errors="coerce").to_numpy(dtype=float)
        for j in range(n):
            xj = pd.to_numeric(data.iloc[:, j], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(xi) & np.isfinite(xj)
            if valid.sum() < 3:
                continue
            if np.std(xi[valid]) <= 1e-12 or np.std(xj[valid]) <= 1e-12:
                continue
            r, p = pearsonr(xi[valid], xj[valid])
            corr[i, j] = float(r)
            pvals[i, j] = float(p)

    return corr, pvals


def _zscore_array(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd <= 1e-12] = 1.0
    return (X - mu) / sd


def _upper_triangle_vector(D: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(D, k=1)
    return D[idx]


def _mantel_test(
    D1: np.ndarray,
    D2: np.ndarray,
    permutations: int = MANTEL_PERMUTATIONS,
    seed: int = SEED,
):



    if D1.shape != D2.shape:
        raise ValueError("Distance matrices must have identical shape.")

    v1 = _upper_triangle_vector(D1)
    v2 = _upper_triangle_vector(D2)

    valid = np.isfinite(v1) & np.isfinite(v2)
    v1 = v1[valid]
    v2 = v2[valid]

    if len(v1) < 3 or np.std(v1) <= 1e-12 or np.std(v2) <= 1e-12:
        return np.nan, np.nan

    r_obs = float(pearsonr(v1, v2)[0])
    rng = np.random.default_rng(seed)
    n = D1.shape[0]
    extreme = 0
    valid_reps = 0

    tri = np.triu_indices(n, k=1)

    for _ in range(permutations):
        perm = rng.permutation(n)
        vp = D2[np.ix_(perm, perm)][tri]
        if np.std(vp) <= 1e-12:
            continue
        rp = float(pearsonr(v1, vp)[0])
        valid_reps += 1
        if abs(rp) >= abs(r_obs):
            extreme += 1

    if valid_reps == 0:
        return r_obs, np.nan

    p = (extreme + 1) / (valid_reps + 1)
    return r_obs, float(p)


def run_mantel_feature_analysis(df: pd.DataFrame):







    left_features, blocks, pretty, block_labels = mantel_feature_spec(df)

    if len(left_features) < 4 or len(blocks) < 2:
        print("[Mantel] Too few available features/blocks; analysis skipped.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    required = list(dict.fromkeys(left_features + [
        c for cols in blocks.values() for c in cols
    ]))

    sampled = _balanced_region_year_sample(df, required, MANTEL_MAX_ROWS)

    if len(sampled) < 30:
        print(f"[Mantel] Only {len(sampled)} complete rows; analysis skipped.")
        return sampled, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(
        f"[Mantel] Complete balanced sample: n={len(sampled)}, "
        f"predictors={len(left_features)}, blocks={list(blocks)}"
    )

    left_numeric = sampled[left_features].copy()
    corr, corr_p = _pearson_matrix(left_numeric)

    mantel_rows = []
    mantel_r = np.full((len(left_features), len(blocks)), np.nan)
    mantel_p = np.full((len(left_features), len(blocks)), np.nan)
    block_keys = list(blocks.keys())

    for i, feature in enumerate(left_features):
        for j, block_key in enumerate(block_keys):
            
            
            effective_block = [c for c in blocks[block_key] if c != feature]
            if len(effective_block) == 0:
                continue

            cols = [feature] + effective_block
            z = _numeric_complete(sampled, cols)

            if len(z) < 30:
                continue

            x = _zscore_array(z[[feature]].to_numpy(dtype=float))
            B = _zscore_array(z[effective_block].to_numpy(dtype=float))

            D_x = squareform(pdist(x, metric="euclidean"))
            D_b = squareform(pdist(B, metric="euclidean"))

            r, p = _mantel_test(
                D_x,
                D_b,
                permutations=MANTEL_PERMUTATIONS,
                seed=SEED + i * 101 + j,
            )

            mantel_r[i, j] = r
            mantel_p[i, j] = p

            mantel_rows.append({
                "predictor": feature,
                "predictor_label": pretty.get(feature, feature),
                "block": block_key,
                "block_label": block_labels.get(block_key, block_key),
                "block_features_used": "|".join(effective_block),
                "n_rows": int(len(z)),
                "mantel_r": r,
                "mantel_p": p,
                "n_permutations": int(MANTEL_PERMUTATIONS),
            })

    mantel_results = pd.DataFrame(mantel_rows)

    pearson_labels = [pretty.get(c, c) for c in left_features]
    pearson_r_table = pd.DataFrame(corr, columns=pearson_labels)
    pearson_r_table.insert(0, "variable", pearson_labels)

    pearson_p_table = pd.DataFrame(corr_p, columns=pearson_labels)
    pearson_p_table.insert(0, "variable", pearson_labels)

    _plot_mantel_network(
        left_features=left_features,
        block_keys=block_keys,
        pretty=pretty,
        block_labels=block_labels,
        corr=corr,
        corr_p=corr_p,
        mantel_r=mantel_r,
        mantel_p=mantel_p,
    )

    return sampled, mantel_results, pearson_r_table, pearson_p_table


def _plot_mantel_network(
    left_features,
    block_keys,
    pretty,
    block_labels,
    corr,
    corr_p,
    mantel_r,
    mantel_p,
):



    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["mathtext.fontset"] = "stix"

    n_left = len(left_features)
    n_right = len(block_keys)

    fig, ax = plt.subplots(figsize=(16, 10.8), facecolor="white")

    
    norm = Normalize(vmin=-1, vmax=1)
    corr_cmap = LinearSegmentedColormap.from_list(
        "mantel_corr",
        [
            MANTEL_COLORS["heatmap_negative"],
            MANTEL_COLORS["heatmap_zero"],
            MANTEL_COLORS["heatmap_positive"],
        ],
        N=256,
    )

    
    for i in range(n_left):
        for j in range(i + 1):
            rect = patches.Rectangle(
                (j, i), 1, 1,
                facecolor="white",
                edgecolor="#202020",
                linewidth=0.85,
                zorder=1,
            )
            ax.add_patch(rect)

    
    for i in range(n_left):
        for j in range(i):
            r = corr[i, j]
            if not np.isfinite(r):
                continue

            side = max(0.06, 0.86 * abs(r))
            color = corr_cmap(norm(r))
            ax.add_patch(
                patches.Rectangle(
                    (j + 0.5 - side / 2, i + 0.5 - side / 2),
                    side, side,
                    facecolor=color,
                    edgecolor="none",
                    zorder=2,
                )
            )

            p = corr_p[i, j]
            stars = ""
            if np.isfinite(p):
                if p < 0.001:
                    stars = "***"
                elif p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"

            if stars:
                ax.text(
                    j + 0.5, i + 0.5, stars,
                    ha="center", va="center",
                    fontsize=10.8, fontweight="bold",
                    color=MANTEL_COLORS["ink"],
                    zorder=3,
                )

    
    left_nodes = [(i + 0.5, i + 0.5) for i in range(n_left)]

    
    right_start_x = n_left + 1.2
    right_step_x = 2.8
    right_y = np.linspace(0.35, n_left - 0.65, n_right)
    right_nodes = [
        (right_start_x + j * right_step_x, right_y[j])
        for j in range(n_right)
    ]

    
    for i in range(n_left):
        for j in range(n_right):
            r = mantel_r[i, j]
            p = mantel_p[i, j]

            if not np.isfinite(r) or not np.isfinite(p):
                continue

            if (not MANTEL_SHOW_NONSIGNIFICANT) and p >= 0.05:
                continue

            if p < 0.01:
                color = MANTEL_COLORS["p_lt_001"]
                alpha = 0.90
            elif p < 0.05:
                color = MANTEL_COLORS["p_001_005"]
                alpha = 0.88
            else:
                color = MANTEL_COLORS["p_ge_005"]
                alpha = 0.45

            ar = abs(r)
            if ar < 0.10:
                lw = 0.75
            elif ar < 0.20:
                lw = 1.7
            else:
                lw = 3.1

            con = patches.ConnectionPatch(
                xyA=left_nodes[i],
                xyB=right_nodes[j],
                coordsA="data",
                coordsB="data",
                axesA=ax,
                axesB=ax,
                arrowstyle="-",
                connectionstyle="arc3,rad=0.075",
                color=color,
                linewidth=lw,
                alpha=alpha,
                zorder=4,
            )
            ax.add_patch(con)

    
    ax.scatter(
        [x for x, _ in left_nodes],
        [y for _, y in left_nodes],
        s=110,
        facecolor=MANTEL_COLORS["nodes"],
        edgecolor="black",
        linewidth=0.8,
        zorder=8,
    )

    ax.scatter(
        [x for x, _ in right_nodes],
        [y for _, y in right_nodes],
        s=145,
        facecolor=MANTEL_COLORS["nodes"],
        edgecolor="black",
        linewidth=0.9,
        zorder=8,
    )

    
    for j, key in enumerate(block_keys):
        ax.text(
            right_nodes[j][0] + 0.38,
            right_nodes[j][1],
            f"{key}  {block_labels.get(key, key)}",
            ha="left", va="center",
            fontsize=13.2, fontweight="bold",
            color=MANTEL_COLORS["ink"],
        )

    
    labels = [pretty.get(c, c) for c in left_features]
    ax.set_xticks(np.arange(n_left) + 0.5)
    ax.set_xticklabels(labels, rotation=90, fontsize=11.5)
    ax.set_yticks(np.arange(n_left) + 0.5)
    ax.set_yticklabels(labels, fontsize=11.5)
    ax.tick_params(axis="both", length=0, pad=3)

    ax.set_xlim(-1.8, right_nodes[-1][0] + 4.6)
    ax.set_ylim(n_left + 0.8, -0.8)

    for spine in ax.spines.values():
        spine.set_visible(False)

    
    cax = fig.add_axes([0.050, 0.17, 0.018, 0.29])
    cb = colorbar.ColorbarBase(
        cax,
        cmap=corr_cmap,
        norm=norm,
        orientation="vertical",
    )
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=11)
    cb.set_label("Pearson's r", fontsize=13, labelpad=10)

    
    p_handles = [
        Line2D([0], [0], color=MANTEL_COLORS["p_lt_001"], lw=3, label="< 0.01"),
        Line2D([0], [0], color=MANTEL_COLORS["p_001_005"], lw=3, label="0.01–0.05"),
        Line2D([0], [0], color=MANTEL_COLORS["p_ge_005"], lw=3, label="≥ 0.05"),
    ]
    leg1 = ax.legend(
        handles=p_handles,
        title="Mantel's p",
        loc="upper left",
        bbox_to_anchor=(-0.03, 0.95),
        frameon=False,
        fontsize=11.2,
        title_fontsize=12.5,
    )
    ax.add_artist(leg1)

    
    r_handles = [
        Line2D([0], [0], color="#777777", lw=0.75, label="< 0.10"),
        Line2D([0], [0], color="#777777", lw=1.7, label="0.10–0.20"),
        Line2D([0], [0], color="#777777", lw=3.1, label="≥ 0.20"),
    ]
    ax.legend(
        handles=r_handles,
        title="|Mantel's r|",
        loc="upper left",
        bbox_to_anchor=(-0.03, 0.76),
        frameon=False,
        fontsize=11.2,
        title_fontsize=12.5,
    )

    fig.suptitle(
        "Predictor correlation and information-block associations",
        fontsize=16, fontweight="bold", y=0.975,
    )

    png = OUT / "16_Mantel_Pearson_feature_network_v3.png"
    pdf = OUT / "16_Mantel_Pearson_feature_network_v3.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[Mantel] Figure saved: {png.name}")
    print(f"[Mantel] Figure saved: {pdf.name}")





def sample_overlap_audit(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s = df[["sample_id", "region", "year"]].dropna().drop_duplicates().copy()
    by_sample = (
        s.groupby("sample_id", observed=True)
        .agg(
            n_regions=("region", "nunique"),
            n_years=("year", "nunique"),
            first_region=("region", "first"),
            min_year=("year", "min"),
            max_year=("year", "max"),
        )
        .reset_index()
    )
    by_sample["reappears_across_years"] = by_sample["n_years"] > 1
    by_sample["reappears_across_regions"] = by_sample["n_regions"] > 1

    summary = pd.DataFrame([{
        "n_unique_samples": int(by_sample["sample_id"].nunique()),
        "fraction_samples_across_years": float(by_sample["reappears_across_years"].mean()),
        "fraction_samples_across_regions": float(by_sample["reappears_across_regions"].mean()),
        "max_years_per_sample": int(by_sample["n_years"].max()),
        "max_regions_per_sample": int(by_sample["n_regions"].max()),
    }])
    return by_sample, summary




def make_audit(overlap_summary: pd.DataFrame) -> pd.DataFrame:
    overlap_detail = "Not computed"
    if not overlap_summary.empty:
        r = overlap_summary.iloc[0]
        overlap_detail = (
            f"Samples across years={r['fraction_samples_across_years']:.3f}; "
            f"across regions={r['fraction_samples_across_regions']:.3f}. "
            "Interpret LORYO as unseen region-year unless a stricter location-holdout is added."
        )
    return pd.DataFrame([
        {"check": "primary_result_version", "status": "PASS", "detail": "All primary outputs use strict source-only progress normalization."},
        {"check": "target_excluded", "status": "PASS", "detail": "NDVI is absent from predictor sets."},
        {"check": "outer_target_excluded", "status": "PASS", "detail": "LORYO/LORO/LOYO targets are excluded from training and source-only tuning."},
        {"check": "inner_validation_excluded_from_progress_fit", "status": "PASS", "detail": "Progress normalization is fitted inside each source GroupKFold fold."},
        {"check": "dynamic_group_boundary", "status": "PASS", "detail": "Dynamic features are grouped by region, year, sample_id and orbit_pass."},
        {"check": "future_dynamic_information", "status": "PASS", "detail": "Only lagged/backward differences/backward rolling windows are used."},
        {"check": "weather_window_causality", "status": "CHECK", "detail": "Verify Rain_10d_mm and GDD_10d are strictly backward-looking [t-9,t] or equivalent."},
        {"check": "weather_product_latency", "status": "CHECK", "detail": "Document ERA5/ERA5-Land latency; use near-operational wording if retrospective reanalysis is used."},
        {"check": "sample_identity_overlap", "status": "CHECK", "detail": overlap_detail},
        {"check": "LORYO_claim_scope", "status": "PASS", "detail": "LORYO is interpreted as unseen region-year environment, not necessarily a completely unseen region."},
        {"check": "LORO_sensitivity", "status": "PASS" if RUN_LORO else "NOT_RUN", "detail": "Entire regions are held out across all years."},
        {"check": "LOYO_sensitivity", "status": "PASS" if RUN_LOYO else "NOT_RUN", "detail": "Entire years are held out across all regions."},
        {"check": "permutation_importance_role", "status": "PASS", "detail": "Held-out labels are used post hoc for interpretation only, never for fitting/selection."},
    ])




def write_outputs(outputs: Dict[str, pd.DataFrame]):
    for filename, table in outputs.items():
        if table is None:
            continue
        table.to_csv(OUT / filename, index=False, encoding="utf-8-sig")

    workbook = OUT / "strict_operational_v3_complete.xlsx"
    temp = OUT / "strict_operational_v3_complete.tmp.xlsx"
    if temp.exists():
        temp.unlink()
    with pd.ExcelWriter(temp, engine="openpyxl") as writer:
        wrote = False
        for filename, table in outputs.items():
            if table is None or table.empty:
                continue
            sheet = filename.split(".")[0][:31]
            table.head(1_048_575).to_excel(writer, sheet_name=sheet, index=False)
            wrote = True
        if not wrote:
            pd.DataFrame({"message": ["No outputs"]}).to_excel(writer, sheet_name="README", index=False)
    if workbook.exists():
        workbook.unlink()
    temp.replace(workbook)




def main():
    print("=" * 88)
    print("STRICT OPERATIONAL V3 / IJAEOG-STYLE ANALYSIS / LOW-MEMORY")
    print(f"Output: {OUT}")
    print(f"N_JOBS={N_JOBS}, INNER_CV_SPLITS={INNER_CV_SPLITS}, RESUME={RESUME}")
    print("=" * 88)

    input_file = base.locate_input_file()
    print("Input:", input_file)
    raw = base.read_csv_robust(input_file)
    df = base.standardize_columns(raw)
    df = base.add_static_sar_features(df)
    df = add_causal_dynamic_features_strict(df)
    df = df.dropna(subset=["date", "region", "year", "NDVI", "VV_dB", "VH_dB"]).copy()
    df["year"] = df["year"].astype(int)

    
    sample_detail, sample_summary = sample_overlap_audit(df)
    if RUN_DOMAIN_PCA:
        pca_scores, pca_loadings, pca_variance = build_domain_environment_pca(df)
    else:
        pca_scores, pca_loadings, pca_variance = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    
    
    
    if RUN_MANTEL_NETWORK:
        (
            mantel_sample,
            mantel_results,
            pearson_r_table,
            pearson_p_table,
        ) = run_mantel_feature_analysis(df)
    else:
        mantel_sample = pd.DataFrame()
        mantel_results = pd.DataFrame()
        pearson_r_table = pd.DataFrame()
        pearson_p_table = pd.DataFrame()

    if RUN_LORYO:
        run_scheme(df, "LORYO", LORYO_FEATURES)
    if RUN_LORO:
        run_scheme(df, "LORO", SENSITIVITY_FEATURES)
    if RUN_LOYO:
        run_scheme(df, "LOYO", SENSITIVITY_FEATURES)

    
    metrics = read_checkpoints("metrics")
    predictions = read_checkpoints("predictions")
    thresholds = read_checkpoints("thresholds")
    importance = read_checkpoints("importance")

    
    if not metrics.empty:
        metrics = metrics.sort_values(["scheme", "feature_set", "test_id"]).drop_duplicates(
            ["scheme", "feature_set", "test_id"], keep="last"
        )
    if not thresholds.empty:
        thresholds = thresholds.sort_values(["scheme", "feature_set", "test_id"]).drop_duplicates(
            ["scheme", "feature_set", "test_id"], keep="last"
        )

    summary = summarize_all(metrics, predictions) if not metrics.empty and not predictions.empty else pd.DataFrame()
    significance = paired_tests_all(metrics) if not metrics.empty else pd.DataFrame()
    bias_curve = cluster_bootstrap_bias_curve(predictions) if not predictions.empty else pd.DataFrame()
    importance_ranked, feature_importance, group_importance = summarize_importance(importance)
    audit = make_audit(sample_summary)

    outputs = {
        "00_operational_audit_v3.csv": audit,
        "01_model_metrics_all_schemes_v3.csv": metrics,
        "02_overall_summary_all_schemes_v3.csv": summary,
        "03_predictions_all_schemes_v3.csv": predictions,
        "04_source_only_thresholds_v3.csv": thresholds,
        "05_paired_tests_holm_all_schemes_v3.csv": significance,
        "06_U2_LORYO_bias_curve_cluster_CI_v3.csv": bias_curve,
        "07_U2_LORYO_permutation_importance_by_domain_v3.csv": importance_ranked,
        "08_U2_LORYO_permutation_importance_summary_v3.csv": feature_importance,
        "09_U2_LORYO_feature_group_importance_v3.csv": group_importance,
        "10_region_year_environment_PCA_scores_v3.csv": pca_scores,
        "11_region_year_environment_PCA_loadings_v3.csv": pca_loadings,
        "12_region_year_environment_PCA_variance_v3.csv": pca_variance,
        "13_sample_identity_overlap_detail_v3.csv": sample_detail,
        "14_sample_identity_overlap_summary_v3.csv": sample_summary,
        "15_model_ready_features_for_mantel_v3.csv": mantel_sample,
        "16_mantel_test_results_v3.csv": mantel_results,
        "17_pearson_r_matrix_v3.csv": pearson_r_table,
        "18_pearson_p_matrix_v3.csv": pearson_p_table,
    }
    write_outputs(outputs)

    print("\nCompleted / current consolidated summary:")
    if not summary.empty:
        cols = [
            "scheme", "feature_set", "n_tests", "mean_r2", "r2_ci_low", "r2_ci_high",
            "pooled_r2", "mean_rmse", "mean_mae", "mean_calibration_slope",
        ]
        print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
    print("\nMATLAB input folder:", OUT)
    print("Checkpoint folder:", CHECKPOINT_DIR)
    print("If execution stops, rerun the same script; completed scheme/feature/test combinations are skipped.")


if __name__ == "__main__":
    main()
