#!/usr/bin/env python3
"""
Selection-associated allele-frequency analysis from sample-level files.

This public version starts from prepared sample-level allele-frequency and
REF/ALT count files plus a metadata file that maps samples to populations and
cohorts.

Required input files:
    1. example_metadata.csv
    2. example_sample_level_genotype_frequency.csv
    3. example_sample_level_allele_count_Ref_to_Alt.csv
    4. example_population_level_GRM_metadata_version.csv

All paths are relative to the working directory.
Edit the configuration section below if your file names differ.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, ttest_1samp
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


###############################################################################
# Configuration
###############################################################################

WORK_DIR = Path(".").resolve()
OUTPUT_DIR = WORK_DIR

METADATA_FILE = WORK_DIR / "example_metadata.csv"
FREQUENCY_FILE = WORK_DIR / "example_sample_level_genotype_frequency.csv"
COUNT_FILE = WORK_DIR / "example_sample_level_allele_count_Ref_to_Alt.csv"
POP_GRM_FILE = WORK_DIR / "example_population_level_GRM_metadata_version.csv"

FINAL_OUTPUT_FILE = OUTPUT_DIR / "selected_unselected_frequency_results.csv"
POP_GRM_USED_FILE = OUTPUT_DIR / "population_level_GRM_used_by_common_ids.csv"

# If your metadata has an Include column, only rows marked "yes", "true",
# "1", or "include" are used. If the column is absent, all rows are used.
METADATA_INCLUDE_COLUMN = "Include"

# Optional: restrict analysis to a set of populations, e.g. {"1", "2", "4"}.
# Set to None to include all populations present in metadata.
INCLUDED_GROUPS = None

# Primary candidate thresholds
FDR_THRESHOLD = 0.01
GST_THRESHOLD = 0.03

# Sensitivity-analysis settings
DIRECTION_CONSISTENCY_THRESHOLD = 0.70
LOO_P_THRESHOLD = 0.01
LOO_MIN_PASS_FRAC = 0.90

# Within-population bootstrap for primary candidates
RUN_BOOTSTRAP = True
BOOTSTRAP_N_REPS = 1000
BOOTSTRAP_RANDOM_SEED = 20260522
BOOTSTRAP_CI_LOW = 2.5
BOOTSTRAP_CI_HIGH = 97.5
BOOTSTRAP_DIRECTION_THRESHOLD = 0.95
BOOTSTRAP_MIN_VALID_POPS = 3

# GRM-informed GLS sensitivity analysis for primary candidates
RUN_GRM_GLS = True
GRM_RHOS = [0.25, 0.50, 0.75, 1.00]
GRM_FDR_THRESHOLD = 0.05

VAR_FLOOR = 1e-8


###############################################################################
# Helper functions
###############################################################################

def normalise_group_id(x):
    """
    Convert group labels such as 1, '1', '1.0', 'Pop 1', and 'Pop1' to '1'.
    Non-numeric labels are returned after removing spaces.
    """
    s = str(x).strip().replace(" ", "")
    nums = re.findall(r"\d+", s)

    if len(nums) > 0:
        return str(int(nums[0]))

    return s


def normalise_pop_label(x):
    """
    Convert population labels such as 1, '1', '1.0', 'Pop 1', and 'Pop1' to 'Pop1'.
    Non-numeric labels are returned after removing spaces.
    """
    s = str(x).strip().replace(" ", "")
    nums = re.findall(r"\d+", s)

    if len(nums) > 0:
        return f"Pop{int(nums[0])}"

    return s


def normalise_type(x):
    """Normalise cohort labels to 'selected' or 'raw'."""
    s = str(x).strip().lower()

    if s in {"selected", "selection", "sel"}:
        return "selected"

    if s in {"raw", "unselected", "unselect", "pool", "pooled"}:
        return "raw"

    raise ValueError(
        f"Unknown cohort type {x!r}. Use values such as 'Selected', 'Raw', or 'unselected'."
    )


def normalise_locus_dataframe(df, chrom_col="Chromosome", pos_col="Position"):
    """Normalise chromosome and position columns for safe matching."""
    out = df.copy()

    if chrom_col not in out.columns or pos_col not in out.columns:
        raise ValueError(f"Input file must contain {chrom_col!r} and {pos_col!r} columns.")

    out[chrom_col] = out[chrom_col].astype(str)
    out[pos_col] = pd.to_numeric(out[pos_col], errors="coerce")

    if out[pos_col].isna().any():
        bad = out[out[pos_col].isna()][[chrom_col, pos_col]].head()
        raise ValueError(f"Some Position values cannot be converted to numeric:\n{bad}")

    out[pos_col] = out[pos_col].astype(np.int64)
    return out


def load_metadata(path):
    """Load and validate metadata."""
    meta = pd.read_csv(path)

    required = {"Sample", "Group", "type"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"Metadata file is missing required columns: {sorted(missing)}")

    meta = meta.copy()
    meta["Sample"] = meta["Sample"].astype(str)
    meta["Group"] = meta["Group"].apply(normalise_group_id)
    meta["cohort"] = meta["type"].apply(normalise_type)

    if METADATA_INCLUDE_COLUMN in meta.columns:
        include_values = {"yes", "y", "true", "1", "include", "included"}
        include_mask = meta[METADATA_INCLUDE_COLUMN].astype(str).str.strip().str.lower().isin(include_values)
        meta = meta[include_mask].copy()

    if INCLUDED_GROUPS is not None:
        included = {normalise_group_id(x) for x in INCLUDED_GROUPS}
        meta = meta[meta["Group"].isin(included)].copy()

    if meta.empty:
        raise ValueError("No metadata rows remain after filtering.")

    if meta["Sample"].duplicated().any():
        dup = meta.loc[meta["Sample"].duplicated(keep=False), "Sample"].head()
        raise ValueError(f"Metadata contains duplicated sample names. Examples:\n{dup}")

    return meta


def get_group_sample_map(meta, data_columns):
    """
    Build a dictionary of selected and raw sample names for each population.
    """
    data_columns = set(map(str, data_columns))
    missing_samples = sorted([s for s in meta["Sample"].tolist() if s not in data_columns])

    if missing_samples:
        raise ValueError(
            "The following metadata samples are missing from the frequency file. "
            f"Examples: {missing_samples[:10]}"
        )

    group_map = {}

    for group, group_meta in meta.groupby("Group"):
        selected = group_meta.loc[group_meta["cohort"] == "selected", "Sample"].tolist()
        raw = group_meta.loc[group_meta["cohort"] == "raw", "Sample"].tolist()

        if len(selected) == 0 or len(raw) == 0:
            print(f"WARNING: Group {group} does not contain both selected and raw samples; skipping.")
            continue

        group_map[group] = {
            "selected": selected,
            "raw": raw
        }

    if len(group_map) < 2:
        raise ValueError("Need at least two populations with both selected and raw samples.")

    group_ids = sorted(
        group_map.keys(),
        key=lambda x: int(re.findall(r"\d+", str(x))[0]) if re.findall(r"\d+", str(x)) else str(x)
    )

    return group_ids, group_map


def parse_ref_alt_count_table(count_subset):
    """
    Parse a count table with cells formatted as REF/ALT.

    Returns
    -------
    ref_df : pandas.DataFrame
    alt_df : pandas.DataFrame
    depth_df : pandas.DataFrame
    ref_freq_df : pandas.DataFrame
    """
    ref_df = pd.DataFrame(index=count_subset.index)
    alt_df = pd.DataFrame(index=count_subset.index)

    for col in count_subset.columns:
        s = count_subset[col].astype("string").str.strip()
        s = s.replace({
            "": pd.NA,
            ".": pd.NA,
            "./.": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "NA": pd.NA
        })

        parts = s.str.split("/", n=1, expand=True)
        ref_df[col] = pd.to_numeric(parts[0], errors="coerce")

        if parts.shape[1] > 1:
            alt_df[col] = pd.to_numeric(parts[1], errors="coerce")
        else:
            alt_df[col] = np.nan

    depth_df = ref_df + alt_df

    with np.errstate(divide="ignore", invalid="ignore"):
        ref_freq_df = ref_df / depth_df

    ref_freq_df[depth_df <= 0] = np.nan
    depth_df[depth_df <= 0] = np.nan

    return ref_df, alt_df, depth_df, ref_freq_df


def cohort_mean_matrix(freq_indexed, group_ids, group_map):
    """Calculate selected and raw cohort mean allele frequencies for each population."""
    n_loci = freq_indexed.shape[0]
    selected_mat = np.full((n_loci, len(group_ids)), np.nan, dtype=float)
    raw_mat = np.full((n_loci, len(group_ids)), np.nan, dtype=float)

    for j, group in enumerate(group_ids):
        selected_samples = group_map[group]["selected"]
        raw_samples = group_map[group]["raw"]

        selected_values = freq_indexed[selected_samples].apply(pd.to_numeric, errors="coerce")
        raw_values = freq_indexed[raw_samples].apply(pd.to_numeric, errors="coerce")

        selected_mat[:, j] = selected_values.mean(axis=1, skipna=True).to_numpy(dtype=float)
        raw_mat[:, j] = raw_values.mean(axis=1, skipna=True).to_numpy(dtype=float)

    return selected_mat, raw_mat


def read_count_variance(count_indexed, freq_index, group_ids, group_map):
    """
    Estimate the variance of selected-minus-raw cohort mean allele-frequency shifts
    from sample-level REF/ALT read counts.
    """
    count_indexed = count_indexed.reindex(freq_index)
    n_loci = len(freq_index)
    var_delta_mat = np.full((n_loci, len(group_ids)), np.nan, dtype=float)

    for j, group in enumerate(group_ids):
        selected_samples = group_map[group]["selected"]
        raw_samples = group_map[group]["raw"]

        missing_count_cols = [
            s for s in selected_samples + raw_samples
            if s not in count_indexed.columns
        ]

        if missing_count_cols:
            raise ValueError(
                f"Count file is missing sample columns for group {group}. "
                f"Examples: {missing_count_cols[:10]}"
            )

        selected_count = count_indexed[selected_samples]
        raw_count = count_indexed[raw_samples]

        sel_ref, sel_alt, sel_depth, _ = parse_ref_alt_count_table(selected_count)
        raw_ref, raw_alt, raw_depth, _ = parse_ref_alt_count_table(raw_count)

        # Continuity correction for variance estimation only.
        sel_p_adj = (sel_ref + 0.5) / (sel_depth + 1.0)
        raw_p_adj = (raw_ref + 0.5) / (raw_depth + 1.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            var_each_sel = sel_p_adj * (1.0 - sel_p_adj) / sel_depth
            var_each_raw = raw_p_adj * (1.0 - raw_p_adj) / raw_depth

        valid_sel = np.isfinite(var_each_sel.to_numpy(dtype=float))
        valid_raw = np.isfinite(var_each_raw.to_numpy(dtype=float))

        n_sel = valid_sel.sum(axis=1).astype(float)
        n_raw = valid_raw.sum(axis=1).astype(float)

        var_sel_sum = np.nansum(var_each_sel.to_numpy(dtype=float), axis=1)
        var_raw_sum = np.nansum(var_each_raw.to_numpy(dtype=float), axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            var_sel_mean = var_sel_sum / (n_sel ** 2)
            var_raw_mean = var_raw_sum / (n_raw ** 2)

        var_delta = var_sel_mean + var_raw_mean
        var_delta[~np.isfinite(var_delta)] = np.nan
        var_delta[var_delta <= 0] = np.nan

        var_delta_mat[:, j] = var_delta

    return np.maximum(var_delta_mat, VAR_FLOOR)


def inverse_variance_weighted_test(d_row, var_row):
    """Fixed-effect inverse-variance weighted test for one SNP."""
    d_row = np.asarray(d_row, dtype=float)
    var_row = np.asarray(var_row, dtype=float)

    mask = np.isfinite(d_row) & np.isfinite(var_row) & (var_row > 0)

    if mask.sum() < 3:
        return np.nan, np.nan, np.nan

    d = d_row[mask]
    v = var_row[mask]

    w = 1.0 / v
    w_sum = np.sum(w)

    weighted_mean = np.sum(w * d) / w_sum
    se = np.sqrt(1.0 / w_sum)

    if se <= 0 or not np.isfinite(se):
        return weighted_mean, np.nan, np.nan

    z_stat = weighted_mean / se
    p_value = 2.0 * norm.sf(abs(z_stat))

    return weighted_mean, z_stat, p_value


def benjamini_hochberg(pvals):
    """Return Benjamini-Hochberg FDR-adjusted q-values."""
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)

    valid = np.isfinite(pvals)
    if valid.sum() == 0:
        return qvals

    qvals[valid] = multipletests(pvals[valid], method="fdr_bh")[1]
    return qvals


def force_correlation_psd(R, eps=1e-8):
    """Symmetrise and make a correlation-like matrix positive semi-definite."""
    R = np.asarray(R, dtype=float)
    R = (R + R.T) / 2.0
    np.fill_diagonal(R, 1.0)

    eigval, eigvec = np.linalg.eigh(R)
    eigval_clipped = np.clip(eigval, eps, None)

    R_psd = eigvec @ np.diag(eigval_clipped) @ eigvec.T
    R_psd = (R_psd + R_psd.T) / 2.0

    d = np.sqrt(np.diag(R_psd))
    R_psd = R_psd / np.outer(d, d)
    np.fill_diagonal(R_psd, 1.0)

    return R_psd


def grm_to_correlation(K):
    """Convert a population-level GRM to a correlation-like matrix."""
    K = np.asarray(K, dtype=float)
    K = (K + K.T) / 2.0

    diag = np.diag(K).copy()
    if np.any(~np.isfinite(diag)) or np.any(diag <= 0):
        raise ValueError("Population GRM has missing or non-positive diagonal values.")

    R = K / np.sqrt(np.outer(diag, diag))
    return force_correlation_psd(R)


def gls_intercept_test(y, var_y, R_pop, rho):
    """GLS test of y_i = mu + error_i, with Var(error) = S * R_rho * S."""
    y = np.asarray(y, dtype=float)
    var_y = np.asarray(var_y, dtype=float)

    valid = np.isfinite(y) & np.isfinite(var_y) & (var_y > 0)
    n = int(valid.sum())

    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, n

    yv = y[valid]
    vv = np.maximum(var_y[valid], VAR_FLOOR)

    Rv = R_pop[np.ix_(valid, valid)]
    I = np.eye(n)
    R_rho = (1.0 - rho) * I + rho * Rv

    S = np.diag(np.sqrt(vv))
    V = S @ R_rho @ S
    V = (V + V.T) / 2.0

    Vinv = np.linalg.pinv(V)
    X = np.ones((n, 1))

    XtVinvX = X.T @ Vinv @ X
    XtVinvy = X.T @ Vinv @ yv

    mu_hat = float(np.linalg.pinv(XtVinvX) @ XtVinvy)
    se_mu = float(np.sqrt(np.linalg.pinv(XtVinvX)[0, 0]))

    if se_mu <= 0 or not np.isfinite(se_mu):
        return mu_hat, np.nan, np.nan, np.nan, n

    t_stat = mu_hat / se_mu
    p_value = 2.0 * stats.t.sf(np.abs(t_stat), df=n - 1)

    return mu_hat, se_mu, t_stat, p_value, n


def within_population_bootstrap(freq_indexed, group_ids, group_map, candidate_idx, main_shift):
    """
    Resample selected individuals and raw pools within each population.

    Raw pools are treated as observed pool-level samples. Hidden individuals
    inside a pool are not resampled.
    """
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)

    bootstrap_mean = np.full(len(candidate_idx), np.nan, dtype=float)
    bootstrap_ci_low = np.full(len(candidate_idx), np.nan, dtype=float)
    bootstrap_ci_high = np.full(len(candidate_idx), np.nan, dtype=float)
    bootstrap_direction_frac = np.full(len(candidate_idx), np.nan, dtype=float)
    bootstrap_n_valid = np.zeros(len(candidate_idx), dtype=int)
    bootstrap_ci_excludes_zero = np.full(len(candidate_idx), False, dtype=bool)
    bootstrap_robust = np.full(len(candidate_idx), False, dtype=bool)

    sample_mats = {}

    for group in group_ids:
        selected_samples = group_map[group]["selected"]
        raw_samples = group_map[group]["raw"]

        sample_mats[group] = {
            "selected": freq_indexed[selected_samples].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float),
            "raw": freq_indexed[raw_samples].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        }

    for local_i, snp_i in enumerate(candidate_idx):
        original_sign = np.sign(main_shift[snp_i])
        boot_reps = np.full(BOOTSTRAP_N_REPS, np.nan, dtype=float)

        for b in range(BOOTSTRAP_N_REPS):
            pop_deltas = []

            for group in group_ids:
                sel_values = sample_mats[group]["selected"][snp_i, :]
                raw_values = sample_mats[group]["raw"][snp_i, :]

                sel_values = sel_values[np.isfinite(sel_values)]
                raw_values = raw_values[np.isfinite(raw_values)]

                if len(sel_values) == 0 or len(raw_values) == 0:
                    continue

                sel_boot = rng.choice(sel_values, size=len(sel_values), replace=True)
                raw_boot = rng.choice(raw_values, size=len(raw_values), replace=True)

                pop_deltas.append(np.mean(sel_boot) - np.mean(raw_boot))

            if len(pop_deltas) >= BOOTSTRAP_MIN_VALID_POPS:
                boot_reps[b] = np.mean(pop_deltas)

        valid = np.isfinite(boot_reps)
        n_valid = int(valid.sum())
        bootstrap_n_valid[local_i] = n_valid

        if n_valid == 0:
            continue

        values = boot_reps[valid]

        bootstrap_mean[local_i] = np.mean(values)
        bootstrap_ci_low[local_i] = np.percentile(values, BOOTSTRAP_CI_LOW)
        bootstrap_ci_high[local_i] = np.percentile(values, BOOTSTRAP_CI_HIGH)

        if original_sign != 0 and np.isfinite(original_sign):
            bootstrap_direction_frac[local_i] = np.mean(np.sign(values) == original_sign)

        ci_excludes_zero = (
            (bootstrap_ci_low[local_i] > 0 and bootstrap_ci_high[local_i] > 0) or
            (bootstrap_ci_low[local_i] < 0 and bootstrap_ci_high[local_i] < 0)
        )

        bootstrap_ci_excludes_zero[local_i] = ci_excludes_zero

        bootstrap_robust[local_i] = (
            ci_excludes_zero and
            np.isfinite(bootstrap_direction_frac[local_i]) and
            bootstrap_direction_frac[local_i] >= BOOTSTRAP_DIRECTION_THRESHOLD
        )

    return {
        "Bootstrap_N_Valid_Reps": bootstrap_n_valid,
        "Bootstrap_Mean_Shift_SelectedMinusRaw": bootstrap_mean,
        "Bootstrap_CI95_Low": bootstrap_ci_low,
        "Bootstrap_CI95_High": bootstrap_ci_high,
        "Bootstrap_Direction_Same_Frac": bootstrap_direction_frac,
        "Bootstrap_CI_Excludes_Zero": bootstrap_ci_excludes_zero,
        "Bootstrap_Robust": bootstrap_robust
    }


###############################################################################
# Main analysis
###############################################################################

def main():
    print(f"Reading metadata file: {METADATA_FILE}")
    meta = load_metadata(METADATA_FILE)

    print(f"Reading sample-level frequency file: {FREQUENCY_FILE}")
    freq_df = pd.read_csv(FREQUENCY_FILE)
    freq_df = normalise_locus_dataframe(freq_df)

    if freq_df.duplicated(["Chromosome", "Position"]).any():
        dup = freq_df[freq_df.duplicated(["Chromosome", "Position"], keep=False)][
            ["Chromosome", "Position"]
        ].head()
        raise ValueError(f"Duplicated Chromosome/Position in frequency file:\n{dup}")

    group_ids, group_map = get_group_sample_map(meta, freq_df.columns)

    print(f"Populations used: {group_ids}")

    freq_index = pd.MultiIndex.from_frame(freq_df[["Chromosome", "Position"]])
    freq_indexed = freq_df.set_index(["Chromosome", "Position"])

    # Check frequency range.
    sample_cols = meta["Sample"].tolist()
    freq_values = freq_indexed[sample_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid_freq = freq_values[np.isfinite(freq_values)]

    if np.any((valid_freq < 0) | (valid_freq > 1)):
        raise ValueError("Allele frequencies must be between 0 and 1.")

    selected_mat, raw_mat = cohort_mean_matrix(freq_indexed, group_ids, group_map)
    diff_mat = selected_mat - raw_mat

    result = freq_df[["Chromosome", "Position"]].copy()

    for j, group in enumerate(group_ids):
        result[f"Selected_{group}"] = selected_mat[:, j]
        result[f"Raw_{group}"] = raw_mat[:, j]
        result[f"Shift_SelectedMinusRaw_{group}"] = diff_mat[:, j]

    result["N_PopPairs"] = np.sum(np.isfinite(diff_mat), axis=1)
    result["Mean_Shift_SelectedMinusRaw"] = np.nanmean(diff_mat, axis=1)

    # Gst/Fst-like differentiation for each population and average across populations.
    gst_mat = np.full_like(diff_mat, np.nan, dtype=float)

    for j, group in enumerate(group_ids):
        sel = selected_mat[:, j]
        raw = raw_mat[:, j]

        total = (sel + raw) / 2.0
        H_T = 2.0 * total * (1.0 - total)
        H_S = (2.0 * sel * (1.0 - sel) + 2.0 * raw * (1.0 - raw)) / 2.0

        with np.errstate(divide="ignore", invalid="ignore"):
            gst = np.where(H_T > 0, (H_T - H_S) / H_T, 0.0)

        gst[~np.isfinite(gst)] = np.nan
        gst_mat[:, j] = gst
        result[f"Gst_{group}"] = gst

    result["Average_Gst"] = np.nanmean(gst_mat, axis=1)

    # Primary one-sample t-test across population-level shifts.
    t_stats, p_vals = ttest_1samp(diff_mat, popmean=0.0, axis=1, nan_policy="omit")
    result["T_Stat"] = t_stats
    result["P_Value"] = p_vals
    result["P_FDR_BH"] = benjamini_hochberg(p_vals)

    result["Candidate"] = (
        (result["P_FDR_BH"] < FDR_THRESHOLD) &
        (result["Average_Gst"] > GST_THRESHOLD)
    )

    # Count-based inverse-variance weighted sensitivity analysis.
    print(f"Reading sample-level REF/ALT count file: {COUNT_FILE}")
    count_df = pd.read_csv(COUNT_FILE)
    count_df = normalise_locus_dataframe(count_df)

    if count_df.duplicated(["Chromosome", "Position"]).any():
        dup = count_df[count_df.duplicated(["Chromosome", "Position"], keep=False)][
            ["Chromosome", "Position"]
        ].head()
        raise ValueError(f"Duplicated Chromosome/Position in count file:\n{dup}")

    count_index = pd.MultiIndex.from_frame(count_df[["Chromosome", "Position"]])
    missing_count_loci = ~freq_index.isin(count_index)

    if missing_count_loci.any():
        examples = freq_df.loc[missing_count_loci, ["Chromosome", "Position"]].head()
        raise ValueError(
            "Some loci in the frequency file are missing from the count file. "
            f"Examples:\n{examples}"
        )

    count_indexed = count_df.set_index(["Chromosome", "Position"])
    var_delta_mat = read_count_variance(count_indexed, freq_index, group_ids, group_map)

    weighted = np.array([
        inverse_variance_weighted_test(diff_mat[i, :], var_delta_mat[i, :])
        for i in range(diff_mat.shape[0])
    ])

    result["Weighted_Mean_Shift_SelectedMinusRaw"] = weighted[:, 0]
    result["Weighted_Z"] = weighted[:, 1]
    result["Weighted_P_Value"] = weighted[:, 2]
    result["Weighted_P_FDR_BH"] = benjamini_hochberg(result["Weighted_P_Value"].to_numpy())

    result["Candidate_weighted"] = (
        (result["Weighted_P_FDR_BH"] < FDR_THRESHOLD) &
        (result["Average_Gst"] > GST_THRESHOLD)
    )

    # Direction consistency.
    main_shift = result["Mean_Shift_SelectedMinusRaw"].to_numpy(dtype=float)
    overall_sign = np.sign(main_shift)
    diff_sign = np.sign(diff_mat)

    valid_direction = (
        np.isfinite(diff_mat) &
        (diff_sign != 0) &
        (overall_sign[:, None] != 0)
    )

    same_direction = valid_direction & (diff_sign == overall_sign[:, None])

    n_same = np.sum(same_direction, axis=1)
    n_valid = np.sum(valid_direction, axis=1)

    frac_same = np.full(diff_mat.shape[0], np.nan, dtype=float)
    valid_denom = n_valid > 0
    frac_same[valid_denom] = n_same[valid_denom] / n_valid[valid_denom]

    result["N_Pop_Same_Direction"] = n_same
    result["N_Pop_Direction_Informative"] = n_valid
    result["Frac_Pop_Same_Direction"] = frac_same
    result["Direction_Consistent"] = frac_same >= DIRECTION_CONSISTENCY_THRESHOLD

    # Leave-one-population-out weighted sensitivity analysis for primary candidates.
    candidate_idx = np.where(result["Candidate"].fillna(False).to_numpy())[0]
    loo_total = len(group_ids)

    result["LOO_Same_Direction_Frac"] = np.nan
    result["LOO_P001_Pass_Frac"] = np.nan
    result["LOO_Both_Pass_Frac"] = np.nan
    result["LOO_Robust"] = False

    if len(candidate_idx) > 0:
        loo_same_count = np.zeros(len(candidate_idx), dtype=int)
        loo_p_count = np.zeros(len(candidate_idx), dtype=int)
        loo_both_count = np.zeros(len(candidate_idx), dtype=int)

        for j in range(loo_total):
            keep = np.array([k for k in range(loo_total) if k != j])

            for local_i, snp_i in enumerate(candidate_idx):
                loo_mean, loo_z, loo_p = inverse_variance_weighted_test(
                    diff_mat[snp_i, keep],
                    var_delta_mat[snp_i, keep]
                )

                original_sign = overall_sign[snp_i]
                loo_sign = np.sign(loo_mean)

                same_dir = (
                    np.isfinite(loo_mean) and
                    original_sign != 0 and
                    loo_sign == original_sign
                )

                p_pass = (
                    np.isfinite(loo_p) and
                    loo_p < LOO_P_THRESHOLD
                )

                if same_dir:
                    loo_same_count[local_i] += 1

                if p_pass:
                    loo_p_count[local_i] += 1

                if same_dir and p_pass:
                    loo_both_count[local_i] += 1

        result.loc[candidate_idx, "LOO_Same_Direction_Frac"] = loo_same_count / loo_total
        result.loc[candidate_idx, "LOO_P001_Pass_Frac"] = loo_p_count / loo_total
        result.loc[candidate_idx, "LOO_Both_Pass_Frac"] = loo_both_count / loo_total
        result.loc[candidate_idx, "LOO_Robust"] = (
            result.loc[candidate_idx, "LOO_Both_Pass_Frac"] >= LOO_MIN_PASS_FRAC
        )

    # Within-population bootstrap.
    result["Bootstrap_N_Valid_Reps"] = np.nan
    result["Bootstrap_Mean_Shift_SelectedMinusRaw"] = np.nan
    result["Bootstrap_CI95_Low"] = np.nan
    result["Bootstrap_CI95_High"] = np.nan
    result["Bootstrap_Direction_Same_Frac"] = np.nan
    result["Bootstrap_CI_Excludes_Zero"] = False
    result["Bootstrap_Robust"] = False

    if RUN_BOOTSTRAP and len(candidate_idx) > 0:
        print(f"Running within-population bootstrap for {len(candidate_idx)} candidate loci.")
        boot = within_population_bootstrap(
            freq_indexed=freq_indexed,
            group_ids=group_ids,
            group_map=group_map,
            candidate_idx=candidate_idx,
            main_shift=main_shift
        )

        for col, values in boot.items():
            result.loc[candidate_idx, col] = values

    # GRM-informed GLS sensitivity analysis.
    if RUN_GRM_GLS:
        print(f"Reading population-level GRM file: {POP_GRM_FILE}")
        pop_grm = pd.read_csv(POP_GRM_FILE, index_col=0)

        pop_grm.index = [normalise_pop_label(x) for x in pop_grm.index]
        pop_grm.columns = [normalise_pop_label(x) for x in pop_grm.columns]

        common_pop_labels = [normalise_pop_label(x) for x in group_ids]

        missing_pops = [
            p for p in common_pop_labels
            if p not in pop_grm.index or p not in pop_grm.columns
        ]

        if len(missing_pops) > 0:
            raise ValueError(
                "The following populations are missing from the GRM file: "
                f"{missing_pops}"
            )

        K_common = pop_grm.loc[common_pop_labels, common_pop_labels].astype(float)
        K_common.to_csv(POP_GRM_USED_FILE)

        R_common = grm_to_correlation(K_common.to_numpy())

        for rho in GRM_RHOS:
            rho_tag = str(rho).replace(".", "p")

            col_mu = f"GRM_GLS_MeanShift_rho{rho_tag}"
            col_se = f"GRM_GLS_SE_rho{rho_tag}"
            col_t = f"GRM_GLS_T_rho{rho_tag}"
            col_p = f"GRM_GLS_P_rho{rho_tag}"
            col_q = f"GRM_GLS_FDR_rho{rho_tag}"
            col_n = f"GRM_GLS_N_Pops_rho{rho_tag}"
            col_same = f"GRM_GLS_Same_Direction_rho{rho_tag}"
            col_robust = f"GRM_GLS_Robust_rho{rho_tag}"

            mu_arr = np.full(len(candidate_idx), np.nan)
            se_arr = np.full(len(candidate_idx), np.nan)
            t_arr = np.full(len(candidate_idx), np.nan)
            p_arr = np.full(len(candidate_idx), np.nan)
            n_arr = np.zeros(len(candidate_idx), dtype=int)
            same_dir_arr = np.full(len(candidate_idx), False, dtype=bool)

            for local_i, snp_i in enumerate(candidate_idx):
                mu_hat, se_mu, t_stat, p_value, n_valid = gls_intercept_test(
                    y=diff_mat[snp_i, :],
                    var_y=var_delta_mat[snp_i, :],
                    R_pop=R_common,
                    rho=rho
                )

                mu_arr[local_i] = mu_hat
                se_arr[local_i] = se_mu
                t_arr[local_i] = t_stat
                p_arr[local_i] = p_value
                n_arr[local_i] = n_valid

                original_shift = result.loc[snp_i, "Mean_Shift_SelectedMinusRaw"]
                if np.isfinite(mu_hat) and np.isfinite(original_shift):
                    same_dir_arr[local_i] = np.sign(mu_hat) == np.sign(original_shift)

            q_arr = benjamini_hochberg(p_arr)
            robust_arr = same_dir_arr & np.isfinite(q_arr) & (q_arr < GRM_FDR_THRESHOLD)

            result[col_mu] = np.nan
            result[col_se] = np.nan
            result[col_t] = np.nan
            result[col_p] = np.nan
            result[col_q] = np.nan
            result[col_n] = np.nan
            result[col_same] = False
            result[col_robust] = False

            result.loc[candidate_idx, col_mu] = mu_arr
            result.loc[candidate_idx, col_se] = se_arr
            result.loc[candidate_idx, col_t] = t_arr
            result.loc[candidate_idx, col_p] = p_arr
            result.loc[candidate_idx, col_q] = q_arr
            result.loc[candidate_idx, col_n] = n_arr
            result.loc[candidate_idx, col_same] = same_dir_arr
            result.loc[candidate_idx, col_robust] = robust_arr

    result["Robust_Candidate"] = (
        result["Candidate"].fillna(False) &
        result["Candidate_weighted"].fillna(False) &
        result["Direction_Consistent"].fillna(False) &
        result["LOO_Robust"].fillna(False)
    )

    if "GRM_GLS_Robust_rho1p0" in result.columns:
        result["Robust_Candidate_With_GRM_GLS"] = (
            result["Robust_Candidate"].fillna(False) &
            result["GRM_GLS_Robust_rho1p0"].fillna(False)
        )

    result.to_csv(FINAL_OUTPUT_FILE, index=False)

    print("\nAnalysis completed.")
    print(f"Primary candidates: {int(result['Candidate'].sum())}")
    print(f"Weighted-supported candidates: {int((result['Candidate'] & result['Candidate_weighted']).sum())}")
    print(f"Robust candidates: {int(result['Robust_Candidate'].sum())}")
    print(f"Output written to: {FINAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
