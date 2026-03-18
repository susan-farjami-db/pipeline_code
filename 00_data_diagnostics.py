# Databricks notebook source
# MAGIC %md
# MAGIC # Data Anonymization Diagnostics
# MAGIC
# MAGIC Analyzes a data table to determine what kind of anonymization was applied
# MAGIC and which transformations are destroying predictive signal.
# MAGIC
# MAGIC **Tests:**
# MAGIC 1. Basic stats and value ranges
# MAGIC 2. Moving average consistency (are BILLING_MOVING_AVERAGE columns derived from BB?)
# MAGIC 3. Cross-column correlations (Pearson vs Spearman)
# MAGIC 4. Temporal autocorrelation of BB
# MAGIC 5. Cross-column ratio stability (multiplicative masking detection)
# MAGIC 6. Lag consistency (do -3_MONTH columns match 0_MONTH shifted by 1 quarter?)
# MAGIC 7. Distribution shape analysis
# MAGIC 8. Summary verdict

# COMMAND ----------

# MAGIC %pip install -r requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
import os

notebook_path = os.getcwd()
if notebook_path not in sys.path:
    sys.path.insert(0, notebook_path)

# COMMAND ----------

import numpy as np
import pandas as pd
from scipy import stats

from pyspark.sql import functions as F

from modeling_dev.data_utils import load_latest_version

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "dash_workshop")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("data_table", "sci_masked_horizon_data")
dbutils.widgets.text("partition_columns", "MARKET_SEGMENT,SCENARIO")

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
DATA_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('data_table')}"
PARTITION_COLS = [c.strip() for c in dbutils.widgets.get("partition_columns").split(",")]

print(f"Analyzing: {DATA_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

SELECT_COLUMNS = [
    "QUARTER", "QUARTER_PARSED", "MARKET_SEGMENT", "SCENARIO",
    "VERIFICATION_HORIZON", "PREDICTION_HORIZON", "BB",
    "BILLING_MOVING_AVERAGE_2Q", "BILLING_MOVING_AVERAGE_3Q",
    "BILLING_MOVING_AVERAGE_4Q", "BILLING_MOVING_AVERAGE_5Q",
    "BB_VERSION", "MOF_VERSION", "BB_VERSION_LATEST", "MOF_VERSION_LATEST",
    "MOF_0_MONTH", "MOF_PLUS_BUFFER_0_MONTH", "SELLIN_0_MONTH", "SR_0_MONTH",
    "SELLIN_MINUS3_MONTH", "MOF_MINUS3_MONTH", "SR_MINUS3_MONTH", "MOF_PLUS_BUFFER_MINUS3_MONTH",
]

df_spark, DATA_VERSION = load_latest_version(spark, DATA_TABLE, SELECT_COLUMNS)
df_spark = df_spark.withColumn("PARTITION_COLUMN", F.concat_ws("_", *[F.col(c) for c in PARTITION_COLS]))

df = df_spark.toPandas()
print(f"Data version: {DATA_VERSION}")
print(f"Shape: {df.shape}")
print(f"Segments: {df['MARKET_SEGMENT'].nunique()} unique")
print(f"Partitions: {df['PARTITION_COLUMN'].nunique()} unique")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Test 1: Basic Stats and Value Ranges
# MAGIC
# MAGIC Per-column descriptive stats. Look for:
# MAGIC - Negative values (unusual for demand/billing data)
# MAGIC - All-zero or near-constant columns
# MAGIC - Suspiciously uniform distributions (synthetic generation)
# MAGIC - Null rates

# COMMAND ----------

numeric_cols = [
    "BB", "BILLING_MOVING_AVERAGE_2Q", "BILLING_MOVING_AVERAGE_3Q",
    "BILLING_MOVING_AVERAGE_4Q", "BILLING_MOVING_AVERAGE_5Q",
    "MOF_0_MONTH", "MOF_PLUS_BUFFER_0_MONTH", "SELLIN_0_MONTH", "SR_0_MONTH",
    "SELLIN_MINUS3_MONTH", "MOF_MINUS3_MONTH", "SR_MINUS3_MONTH", "MOF_PLUS_BUFFER_MINUS3_MONTH",
]
numeric_cols = [c for c in numeric_cols if c in df.columns]

basic_stats = []
for col in numeric_cols:
    s = df[col]
    basic_stats.append({
        "column": col,
        "null_pct": round(s.isna().mean() * 100, 1),
        "zero_pct": round((s == 0).mean() * 100, 1),
        "negative_pct": round((s < 0).mean() * 100, 1),
        "min": round(s.min(), 2) if s.notna().any() else None,
        "max": round(s.max(), 2) if s.notna().any() else None,
        "mean": round(s.mean(), 2) if s.notna().any() else None,
        "std": round(s.std(), 2) if s.notna().any() else None,
        "cv": round(s.std() / s.mean(), 3) if s.notna().any() and s.mean() != 0 else None,
    })

stats_df = pd.DataFrame(basic_stats)
print("=== BASIC STATS ===\n")
display(stats_df)

# Flag issues
constant_cols = [r["column"] for r in basic_stats if r["std"] is not None and r["std"] < 0.01]
high_zero_cols = [r["column"] for r in basic_stats if r["zero_pct"] > 50]
negative_cols = [r["column"] for r in basic_stats if r["negative_pct"] > 5]

if constant_cols:
    print(f"CONSTANT/NEAR-CONSTANT columns: {constant_cols}")
if high_zero_cols:
    print(f"HIGH ZERO RATE (>50%) columns: {high_zero_cols}")
if negative_cols:
    print(f"NEGATIVE VALUES (>5%) columns: {negative_cols}")

# Check if all columns have similar scale (would suggest uniform generation)
means = [r["mean"] for r in basic_stats if r["mean"] is not None and r["mean"] != 0]
if means:
    mean_cv = np.std(means) / np.mean(means) if np.mean(means) != 0 else 0
    if mean_cv < 0.3:
        print(f"WARNING: All columns have similar mean scale (CV={mean_cv:.3f}). May indicate uniform generation.")
    else:
        print(f"Column means vary across columns (CV={mean_cv:.3f}). Suggests differentiated values.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Test 2: Moving Average Consistency (KEY TEST)
# MAGIC
# MAGIC BILLING_MOVING_AVERAGE_2Q should be the 2-quarter rolling mean of BB within each scenario.
# MAGIC If this relationship holds, BB values are internally consistent.
# MAGIC If not, the moving averages were independently masked/generated.

# COMMAND ----------

print("=== MOVING AVERAGE CONSISTENCY ===\n")

# Group by scenario (each scenario is a single time series)
scenario_col = "SCENARIO" if "SCENARIO" in df.columns else "PARTITION_COLUMN"

ma_results = []
for name, group in df.groupby(scenario_col):
    group = group.sort_values("QUARTER_PARSED")
    bb = group["BB"].values

    for window, col_name in [(2, "BILLING_MOVING_AVERAGE_2Q"), (3, "BILLING_MOVING_AVERAGE_3Q"),
                              (4, "BILLING_MOVING_AVERAGE_4Q"), (5, "BILLING_MOVING_AVERAGE_5Q")]:
        if col_name not in group.columns:
            continue

        actual_ma = group[col_name].values
        # Compute expected rolling mean from BB
        computed_ma = pd.Series(bb).rolling(window, min_periods=1).mean().values

        # Compare where both are non-null and non-zero
        mask = ~np.isnan(actual_ma) & ~np.isnan(computed_ma) & (actual_ma != 0)
        if mask.sum() < 3:
            continue

        actual_vals = actual_ma[mask]
        computed_vals = computed_ma[mask]

        # Check exact match
        exact_match = np.allclose(actual_vals, computed_vals, rtol=1e-3, atol=1e-3)

        # Check correlation
        if len(actual_vals) > 2:
            corr = np.corrcoef(actual_vals, computed_vals)[0, 1]
        else:
            corr = np.nan

        # Check mean absolute relative error
        mare = np.mean(np.abs((actual_vals - computed_vals) / actual_vals)) * 100

        ma_results.append({
            "scenario": str(name)[:40],
            "ma_column": col_name,
            "n_compared": int(mask.sum()),
            "exact_match": exact_match,
            "correlation": round(corr, 4),
            "mean_abs_rel_error_pct": round(mare, 2),
        })

if ma_results:
    ma_df = pd.DataFrame(ma_results)
    display(ma_df)

    exact_pct = ma_df["exact_match"].mean() * 100
    avg_corr = ma_df["correlation"].mean()
    avg_mare = ma_df["mean_abs_rel_error_pct"].mean()

    print(f"\nExact matches: {exact_pct:.0f}%")
    print(f"Avg correlation (actual MA vs computed from BB): {avg_corr:.4f}")
    print(f"Avg relative error: {avg_mare:.1f}%")

    if exact_pct > 90:
        print("CONCLUSION: Moving averages ARE consistent with BB. BB history is internally coherent.")
    elif avg_corr > 0.9:
        print("CONCLUSION: Moving averages are APPROXIMATELY consistent (noise or rounding applied).")
    elif avg_corr > 0.5:
        print("CONCLUSION: Moving averages are WEAKLY consistent. Significant distortion applied.")
    else:
        print("CONCLUSION: Moving averages are NOT consistent with BB. They were independently generated/masked.")
else:
    print("Could not compute moving average comparisons (insufficient data).")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Test 3: Cross-Column Correlations
# MAGIC
# MAGIC Pearson (linear) vs Spearman (rank) correlation.
# MAGIC - If both strong: real relationships preserved
# MAGIC - If Spearman strong but Pearson weak: non-linear monotonic transform applied
# MAGIC - If both weak: independent masking destroyed relationships

# COMMAND ----------

print("=== CROSS-COLUMN CORRELATIONS ===\n")

corr_cols = [c for c in numeric_cols if c in df.columns and df[c].notna().sum() > 10]
corr_data = df[corr_cols].dropna()

print(f"Computing correlations on {len(corr_data)} rows across {len(corr_cols)} columns\n")

pearson_corr = corr_data.corr(method="pearson")
spearman_corr = corr_data.corr(method="spearman")

# Show key relationships with BB
print("--- BB vs each feature ---")
print(f"{'Feature':<35} {'Pearson':>10} {'Spearman':>10} {'Diff':>10}")
print("-" * 70)
for col in corr_cols:
    if col == "BB":
        continue
    p = pearson_corr.loc["BB", col]
    s = spearman_corr.loc["BB", col]
    diff = s - p
    flag = " <<<" if abs(diff) > 0.2 else ""
    print(f"{col:<35} {p:>10.4f} {s:>10.4f} {diff:>10.4f}{flag}")

print("\n(<<< = Spearman much higher than Pearson, suggesting non-linear monotonic transform)")

# COMMAND ----------

# MAGIC %md
# MAGIC Full Pearson correlation matrix:

# COMMAND ----------

display(pearson_corr.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC Full Spearman correlation matrix:

# COMMAND ----------

display(spearman_corr.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC Cross-signal correlations (MOF vs SR vs SELLIN -- should be correlated in real data):

# COMMAND ----------

signal_cols = ["MOF_0_MONTH", "SR_0_MONTH", "SELLIN_0_MONTH", "MOF_PLUS_BUFFER_0_MONTH"]
signal_cols = [c for c in signal_cols if c in corr_data.columns]

if len(signal_cols) >= 2:
    print("--- Inter-signal Pearson correlations ---")
    sig_corr = corr_data[signal_cols].corr(method="pearson")
    display(sig_corr.round(3))

    avg_inter = []
    for i, c1 in enumerate(signal_cols):
        for c2 in signal_cols[i+1:]:
            avg_inter.append(sig_corr.loc[c1, c2])
    avg_val = np.mean(avg_inter) if avg_inter else 0

    if avg_val > 0.7:
        print(f"Avg inter-signal correlation: {avg_val:.3f} -- STRONG (real relationships likely preserved)")
    elif avg_val > 0.3:
        print(f"Avg inter-signal correlation: {avg_val:.3f} -- MODERATE (partial signal)")
    else:
        print(f"Avg inter-signal correlation: {avg_val:.3f} -- WEAK/ABSENT (signals likely independently masked)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Test 4: Temporal Autocorrelation
# MAGIC
# MAGIC Check if BB has temporal structure (autocorrelation at lag 1-4).
# MAGIC Strong autocorrelation = time series structure preserved.
# MAGIC Weak/zero = rows shuffled or values independently generated.

# COMMAND ----------

print("=== TEMPORAL AUTOCORRELATION ===\n")

autocorr_results = []
for name, group in df.groupby(scenario_col):
    group = group.sort_values("QUARTER_PARSED")
    bb = group["BB"].dropna()
    if len(bb) < 6:
        continue

    for lag in [1, 2, 3, 4]:
        ac = bb.autocorr(lag=lag)
        autocorr_results.append({
            "scenario": str(name)[:40],
            "lag": lag,
            "autocorrelation": round(ac, 4) if not np.isnan(ac) else None,
        })

if autocorr_results:
    ac_df = pd.DataFrame(autocorr_results)

    # Summary by lag
    print("--- Average BB autocorrelation by lag ---")
    for lag in [1, 2, 3, 4]:
        lag_vals = ac_df[ac_df["lag"] == lag]["autocorrelation"].dropna()
        if len(lag_vals) > 0:
            avg_ac = lag_vals.mean()
            pct_positive = (lag_vals > 0).mean() * 100
            print(f"  Lag {lag}: avg={avg_ac:.4f}, positive in {pct_positive:.0f}% of partitions")

    avg_lag1 = ac_df[ac_df["lag"] == 1]["autocorrelation"].dropna().mean()
    if avg_lag1 > 0.5:
        print(f"\nCONCLUSION: Strong temporal autocorrelation (lag-1 avg={avg_lag1:.3f}). Time series structure PRESERVED.")
    elif avg_lag1 > 0.2:
        print(f"\nCONCLUSION: Moderate autocorrelation (lag-1 avg={avg_lag1:.3f}). Partial temporal structure.")
    else:
        print(f"\nCONCLUSION: Weak/no autocorrelation (lag-1 avg={avg_lag1:.3f}). Temporal structure DESTROYED or absent.")
else:
    print("Insufficient data for autocorrelation analysis.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Test 5: Cross-Column Ratio Stability
# MAGIC
# MAGIC If columns were scaled by the same multiplicative factor, BB/MOF ratio would be
# MAGIC roughly constant within each partition. High variance = independent masking.

# COMMAND ----------

print("=== CROSS-COLUMN RATIO STABILITY ===\n")

ratio_pairs = [
    ("BB", "MOF_0_MONTH"),
    ("BB", "SR_0_MONTH"),
    ("BB", "SELLIN_0_MONTH"),
    ("MOF_0_MONTH", "SR_0_MONTH"),
]

print(f"{'Ratio':<30} {'Mean':>10} {'Std':>10} {'CV':>10} {'Interpretation'}")
print("-" * 85)

for col_a, col_b in ratio_pairs:
    if col_a not in df.columns or col_b not in df.columns:
        continue

    mask = (df[col_a].notna() & df[col_b].notna() & (df[col_b] != 0) & (df[col_a] != 0))
    ratios = df.loc[mask, col_a] / df.loc[mask, col_b]

    if len(ratios) < 5:
        continue

    # Remove extreme outliers for cleaner stats
    q1, q99 = ratios.quantile(0.01), ratios.quantile(0.99)
    ratios_clipped = ratios[(ratios >= q1) & (ratios <= q99)]

    r_mean = ratios_clipped.mean()
    r_std = ratios_clipped.std()
    r_cv = r_std / abs(r_mean) if r_mean != 0 else np.inf

    if r_cv < 0.1:
        interp = "CONSTANT ratio -- same scaling factor"
    elif r_cv < 0.5:
        interp = "Low variance -- similar scaling"
    elif r_cv < 1.0:
        interp = "Moderate variance -- partial independence"
    else:
        interp = "HIGH variance -- independently masked"

    print(f"{col_a}/{col_b:<25} {r_mean:>10.3f} {r_std:>10.3f} {r_cv:>10.3f}   {interp}")

# COMMAND ----------

# MAGIC %md
# MAGIC Per-partition ratio stability (checks if masking is per-partition or global):

# COMMAND ----------

# Check if BB/MOF ratio is constant WITHIN each partition (even if different between partitions)
if "MOF_0_MONTH" in df.columns:
    per_partition_cv = []
    for name, group in df.groupby(scenario_col):
        mask = (group["BB"].notna() & group["MOF_0_MONTH"].notna() &
                (group["MOF_0_MONTH"] != 0) & (group["BB"] != 0))
        ratios = group.loc[mask, "BB"] / group.loc[mask, "MOF_0_MONTH"]
        if len(ratios) >= 3:
            cv = ratios.std() / abs(ratios.mean()) if ratios.mean() != 0 else np.inf
            per_partition_cv.append({"scenario": str(name)[:40], "ratio_cv": round(cv, 4), "n_rows": len(ratios)})

    if per_partition_cv:
        cv_df = pd.DataFrame(per_partition_cv)
        avg_within_cv = cv_df["ratio_cv"].mean()
        print(f"Avg WITHIN-partition CV of BB/MOF ratio: {avg_within_cv:.4f}")
        if avg_within_cv < 0.2:
            print("BB/MOF ratio is fairly stable WITHIN partitions (even if different between them).")
            print("This suggests per-partition multiplicative masking.")
        else:
            print("BB/MOF ratio varies even WITHIN partitions. Not a simple multiplicative mask.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Test 6: Lag Consistency
# MAGIC
# MAGIC Check if SELLIN_MINUS3_MONTH at quarter Q matches SELLIN_0_MONTH at quarter Q-1.
# MAGIC (Both should represent the same underlying value, just at different points in the data.)
# MAGIC If they match, the lag structure is real. If not, lags were independently masked.

# COMMAND ----------

print("=== LAG CONSISTENCY ===\n")

lag_pairs = [
    ("SELLIN_0_MONTH", "SELLIN_MINUS3_MONTH"),
    ("SR_0_MONTH", "SR_MINUS3_MONTH"),
    ("MOF_PLUS_BUFFER_0_MONTH", "MOF_PLUS_BUFFER_MINUS3_MONTH"),
]

for current_col, lag_col in lag_pairs:
    if current_col not in df.columns or lag_col not in df.columns:
        continue

    corrs = []
    exact_matches = []

    for name, group in df.groupby(scenario_col):
        group = group.sort_values("QUARTER_PARSED").reset_index(drop=True)
        if len(group) < 3:
            continue

        # Shift current_col forward by 1 to align with lag_col
        shifted = group[current_col].shift(1)
        lag_vals = group[lag_col]

        mask = shifted.notna() & lag_vals.notna() & (shifted != 0) & (lag_vals != 0)
        if mask.sum() < 2:
            continue

        s_vals = shifted[mask].values
        l_vals = lag_vals[mask].values

        # Check exact match
        exact = np.allclose(s_vals, l_vals, rtol=1e-3, atol=1e-3)
        exact_matches.append(exact)

        # Check correlation
        if len(s_vals) > 2:
            c = np.corrcoef(s_vals, l_vals)[0, 1]
            corrs.append(c)

    if corrs:
        avg_corr = np.mean(corrs)
        exact_pct = np.mean(exact_matches) * 100 if exact_matches else 0
        print(f"{current_col} (shifted) vs {lag_col}:")
        print(f"  Exact match: {exact_pct:.0f}% of partitions")
        print(f"  Avg correlation: {avg_corr:.4f}")
        if exact_pct > 80:
            print(f"  --> Lag structure is REAL (values match when shifted)")
        elif avg_corr > 0.8:
            print(f"  --> Lag structure is APPROXIMATELY preserved")
        elif avg_corr > 0.3:
            print(f"  --> Lag structure is WEAKLY preserved")
        else:
            print(f"  --> Lag structure DESTROYED (independently masked)")
        print()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Test 7: Distribution Shape Analysis
# MAGIC
# MAGIC Check if distributions look like real financial data (right-skewed) or synthetic (uniform/normal).

# COMMAND ----------

print("=== DISTRIBUTION SHAPE ===\n")

print(f"{'Column':<35} {'Skewness':>10} {'Kurtosis':>10} {'Shapiro p':>12} {'Shape'}")
print("-" * 85)

for col in numeric_cols:
    if col not in df.columns:
        continue
    vals = df[col].dropna()
    if len(vals) < 10:
        continue

    skew = vals.skew()
    kurt = vals.kurtosis()

    # Shapiro-Wilk on a sample (max 5000 for speed)
    sample = vals.sample(min(len(vals), 5000), random_state=42)
    try:
        _, shapiro_p = stats.shapiro(sample)
    except Exception:
        shapiro_p = np.nan

    # Classify shape
    if abs(skew) < 0.5 and abs(kurt) < 1:
        shape = "SYMMETRIC (normal-like)"
    elif skew > 1:
        shape = "RIGHT-SKEWED (typical for financial)"
    elif skew < -1:
        shape = "LEFT-SKEWED (unusual)"
    elif abs(kurt) > 3:
        shape = "HEAVY-TAILED"
    else:
        shape = "MODERATE"

    print(f"{col:<35} {skew:>10.3f} {kurt:>10.3f} {shapiro_p:>12.6f}   {shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC Per-segment BB distributions (check if different segments have different masking):

# COMMAND ----------

print("--- BB distribution per segment ---\n")
print(f"{'Segment':<25} {'N':>6} {'Mean':>12} {'Std':>12} {'Skew':>8} {'Min':>12} {'Max':>12}")
print("-" * 95)

for seg in sorted(df["MARKET_SEGMENT"].unique()):
    bb = df.loc[df["MARKET_SEGMENT"] == seg, "BB"].dropna()
    if len(bb) < 3:
        continue
    print(f"{seg:<25} {len(bb):>6} {bb.mean():>12.2f} {bb.std():>12.2f} {bb.skew():>8.3f} {bb.min():>12.2f} {bb.max():>12.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Test 8: Summary Verdict

# COMMAND ----------

print("=" * 70)
print("ANONYMIZATION DIAGNOSTIC SUMMARY")
print("=" * 70)

findings = {}

# Moving average consistency
if ma_results:
    ma_avg_corr = pd.DataFrame(ma_results)["correlation"].mean()
    ma_exact = pd.DataFrame(ma_results)["exact_match"].mean() * 100
    if ma_exact > 90:
        findings["moving_averages"] = ("CONSISTENT", "Moving averages match BB rolling means. BB history is internally coherent.")
    elif ma_avg_corr > 0.9:
        findings["moving_averages"] = ("APPROX CONSISTENT", "Moving averages approximately match (noise or rounding applied).")
    elif ma_avg_corr > 0.5:
        findings["moving_averages"] = ("WEAKLY CONSISTENT", "Moving averages weakly match BB. Significant distortion.")
    else:
        findings["moving_averages"] = ("INCONSISTENT", "Moving averages do NOT match BB. Independently generated/masked.")

# Cross-correlations
bb_corrs = []
for col in corr_cols:
    if col != "BB" and col in pearson_corr.columns:
        bb_corrs.append(abs(pearson_corr.loc["BB", col]))
avg_bb_corr = np.mean(bb_corrs) if bb_corrs else 0

if avg_bb_corr > 0.6:
    findings["cross_correlation"] = ("STRONG", f"Avg |corr| with BB = {avg_bb_corr:.3f}. Feature-target relationships preserved.")
elif avg_bb_corr > 0.3:
    findings["cross_correlation"] = ("MODERATE", f"Avg |corr| with BB = {avg_bb_corr:.3f}. Partial signal remains.")
else:
    findings["cross_correlation"] = ("WEAK/ABSENT", f"Avg |corr| with BB = {avg_bb_corr:.3f}. Cross-column relationships destroyed.")

# Temporal structure
if autocorr_results:
    ac_lag1 = pd.DataFrame(autocorr_results)
    ac_lag1 = ac_lag1[ac_lag1["lag"] == 1]["autocorrelation"].dropna().mean()
    if ac_lag1 > 0.5:
        findings["temporal"] = ("PRESERVED", f"Lag-1 autocorrelation avg = {ac_lag1:.3f}. Time series structure intact.")
    elif ac_lag1 > 0.2:
        findings["temporal"] = ("PARTIAL", f"Lag-1 autocorrelation avg = {ac_lag1:.3f}. Some temporal structure.")
    else:
        findings["temporal"] = ("DESTROYED", f"Lag-1 autocorrelation avg = {ac_lag1:.3f}. No temporal structure.")

# Print findings
print()
for test, (status, detail) in findings.items():
    print(f"  [{status:^20}] {test}")
    print(f"  {'':20}  {detail}")
    print()

# Determine likely anonymization method
print("-" * 70)
print("LIKELY ANONYMIZATION METHOD:")
print()

ma_status = findings.get("moving_averages", ("UNKNOWN",))[0]
corr_status = findings.get("cross_correlation", ("UNKNOWN",))[0]
temp_status = findings.get("temporal", ("UNKNOWN",))[0]

if ma_status in ("CONSISTENT", "APPROX CONSISTENT") and corr_status == "STRONG":
    print("  Columns appear to be consistently masked (same or proportional transform).")
    print("  Predictive signal should be largely PRESERVED.")
    print("  Model performance issues are likely due to other factors (small data, model config).")
elif ma_status in ("CONSISTENT", "APPROX CONSISTENT") and corr_status in ("MODERATE", "WEAK/ABSENT"):
    print("  BB and its moving averages are internally consistent, but cross-column")
    print("  correlations (BB vs MOF/SR/SELLIN) are weak. This means:")
    print("  --> BB history is real (or consistently masked)")
    print("  --> Demand signals (MOF, SR, SELLIN) were independently masked/generated")
    print("  --> Models can only learn from BB's own history (autoregressive signal)")
    print("  --> External demand signals are NOISE, not signal")
    print()
    print("  IMPLICATION: Only BILLING_MOVING_AVERAGE features carry information.")
    print("  MOF, SR, SELLIN features actively HURT predictions by adding noise.")
elif ma_status == "INCONSISTENT":
    print("  Moving averages don't match BB. All columns appear independently generated.")
    print("  NO predictive signal exists in this data.")
    print("  Model performance is expected to be random.")
else:
    print("  Mixed signals. Refer to individual test results above for details.")

print()
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Feature Signal Strength
# MAGIC
# MAGIC Rank features by their correlation with BB to show which ones carry signal
# MAGIC and which are noise.

# COMMAND ----------

print("=== FEATURE SIGNAL RANKING ===\n")
print(f"{'Feature':<35} {'|Pearson|':>10} {'|Spearman|':>10} {'Signal?'}")
print("-" * 70)

feature_signal = []
for col in corr_cols:
    if col == "BB":
        continue
    p = abs(pearson_corr.loc["BB", col])
    s = abs(spearman_corr.loc["BB", col])

    if p > 0.5 or s > 0.5:
        signal = "SIGNAL"
    elif p > 0.2 or s > 0.2:
        signal = "WEAK"
    else:
        signal = "NOISE"

    feature_signal.append((col, p, s, signal))

# Sort by Pearson
feature_signal.sort(key=lambda x: x[1], reverse=True)

for col, p, s, signal in feature_signal:
    print(f"{col:<35} {p:>10.4f} {s:>10.4f}   {signal}")
