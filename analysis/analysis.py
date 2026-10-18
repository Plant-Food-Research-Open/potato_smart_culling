#!/usr/bin/env python3
"""
Selection-associated allele-frequency analysis with three estimators of the
selection-cohort allele frequency.

For every locus and population, the reference-allele frequency of the
selection cohort (individually sequenced lines) is estimated in three ways and
compared with the frequency of the raw cohort (pooled seedlings):

    genotype_based    mean over the lines of the frequency implied by the called
                      tetraploid dosage (1 - dosage / ploidy)
    mean_read_ratio   mean over the lines of RO / (RO + AO); every line has the
                      same weight
    summed_reads      sum of RO over the lines / sum of (RO + AO) over the lines
                      (in-silico pooling; lines are weighted by their depth)

The raw-cohort frequency is the mean over the pools of RO / (RO + AO) and is
the same in the three analyses. The per-population differences (selection
minus raw) are combined across populations with inverse-variance weights and
tested against zero with a two-sided Z test; p-values are adjusted across loci
with the Benjamini-Hochberg procedure, separately for each estimator. A locus
is a candidate when it reaches weighted FDR < 0.01 with all three estimators
and the three weighted shifts have the same sign.

Sensitivity analyses: direction consistency across populations and a
leave-one-population-out analysis (all tested loci), a within-population
bootstrap of the weighted shift (candidate loci) and, when a population-level
GRM is supplied, a GRM-informed generalised least-squares model (all tested
loci, BH across loci).

Required input files (relative paths; change the names in the configuration
section below or pass them as command-line options):

    1. example_metadata.csv
           columns Sample, Group, type. "type" is "Selected" for an
           individually sequenced line of the selection cohort and
           "unselected" (or "Raw") for a raw pool; rows of any other type
           (e.g. "Parent") are ignored. "Group" is the population. An optional
           Include column (yes/no) restricts the rows that are used.
    2. example_sample_level_genotype_frequency.csv
           columns Chromosome, Position and one column per selection-cohort
           line holding the reference-allele frequency implied by its called
           dosage (1, 0.75, 0.5, 0.25 or 0 for a tetraploid; empty where there
           is no confident call). With GENOTYPE_VALUES = "dosage" the cells hold
           the alternate-allele dosage (0 .. ploidy) instead. Columns of samples
           that are not selection-cohort lines are ignored.
    3. example_sample_level_allele_count_Ref_to_Alt.csv
           columns Chromosome, Position and one column per sample (lines and
           pools) with cells "REF/ALT" (reference and alternate read counts).
    4. example_population_level_GRM_metadata_version.csv   (optional)
           population-level genomic relationship matrix with row and column
           labels Pop1, Pop2, ... (or 1, 2, ...). Without this file the GRM step
           is skipped.

Output files (written to the output directory):

    results_genotype_based.csv
    results_mean_read_ratio.csv
    results_summed_reads.csv       one row per locus: per-population
                                   frequencies, weighted test and sensitivity
                                   analyses with that estimator
    results_all_loci_overview.csv  weighted shift and FDR of every locus with
                                   each estimator, and the candidate flags
    results_candidates.csv         the candidate loci with the key statistics
                                   of each estimator
    results_summary.txt            numbers of loci tested, significant and robust

Usage:
    python analysis.py
    python analysis.py --metadata meta.csv --genotypes geno.csv --counts counts.csv \
                       --grm grm.csv --outdir results

Requires numpy, pandas, scipy and statsmodels.
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


###############################################################################
# Configuration
###############################################################################

# Input and output files, relative to the working directory.
METADATA_FILE = "example_metadata.csv"
GENOTYPE_FILE = "example_sample_level_genotype_frequency.csv"
COUNT_FILE = "example_sample_level_allele_count_Ref_to_Alt.csv"
POP_GRM_FILE = "example_population_level_GRM_metadata_version.csv"
OUTPUT_DIR = "results"

# If the metadata has this column, only rows marked "yes", "true", "1" or
# "include" are used. If the column is absent, all rows are used.
METADATA_INCLUDE_COLUMN = "Include"

# Optional: restrict the analysis to a set of populations, e.g. {"1", "2", "4"}.
# None uses every population of the metadata that has both cohorts.
INCLUDED_GROUPS = None

# "frequency": the genotype file holds reference-allele frequencies implied by
# the called dosage (1 - dosage / PLOIDY). "dosage": it holds the alternate-
# allele dosage (0 .. PLOIDY).
GENOTYPE_VALUES = "frequency"

PLOIDY = 4
PLANTS_PER_POOL = 20          # seedlings represented by each raw pool

# A line or pool contributes to a locus only if its read depth RO + AO is
# above MIN_DEPTH. A population is used at a locus if at least MIN_SELECTED
# lines and one raw pool have data; a locus is tested if at least
# MIN_POPULATIONS populations are used.
MIN_DEPTH = 0
MIN_SELECTED = 1
MIN_POPULATIONS = 3

# Candidate definition: weighted FDR below this with all three estimators and
# the same direction of shift.
FDR_THRESHOLD = 0.01

# Sensitivity analyses
DIRECTION_CONSISTENCY_THRESHOLD = 0.70
LOO_P_THRESHOLD = 0.01
LOO_MIN_PASS_FRAC = 0.90

RUN_BOOTSTRAP = True
BOOTSTRAP_N_REPS = 1000
BOOTSTRAP_RANDOM_SEED = 20261007
BOOTSTRAP_CI_LOW = 2.5
BOOTSTRAP_CI_HIGH = 97.5
BOOTSTRAP_DIRECTION_THRESHOLD = 0.95

RUN_GRM_GLS = True
GRM_RHOS = [0.25, 0.50, 0.75, 1.00]   # rho = 1 uses the full relationship matrix
GRM_FDR_THRESHOLD = 0.05

VAR_FLOOR = 1e-8

ESTIMATORS = ("genotype_based", "mean_read_ratio", "summed_reads")
ESTIMATOR_LABELS = {
    "genotype_based": "genotype-based",
    "mean_read_ratio": "mean read ratio",
    "summed_reads": "summed reads",
}
KEYS = ["Chromosome", "Position"]


###############################################################################
# Input files
###############################################################################

def normalise_group_id(x):
    """Group labels such as 1, '1', '1.0', 'Pop 1' and 'Pop1' become '1'."""
    s = str(x).strip().replace(" ", "")
    nums = re.findall(r"\d+", s)
    return str(int(nums[0])) if nums else s


def normalise_pop_label(x):
    """Population labels such as 1, '1', '1.0', 'Pop 1' and 'Pop1' become 'Pop1'."""
    s = str(x).strip().replace(" ", "")
    nums = re.findall(r"\d+", s)
    return "Pop%d" % int(nums[0]) if nums else s


def normalise_type(x):
    """Cohort labels become 'selected', 'raw' or 'other'."""
    s = str(x).strip().lower()
    if s in {"selected", "selection", "sel", "line", "individual"}:
        return "selected"
    if s in {"raw", "unselected", "unselect", "pool", "pooled"}:
        return "raw"
    return "other"          # e.g. parents: listed in the metadata but in neither cohort


def group_sort_key(g):
    nums = re.findall(r"\d+", str(g))
    return (0, int(nums[0]), str(g)) if nums else (1, 0, str(g))


def load_metadata(path, include_column, included_groups):
    """Metadata rows that are used, with normalised Group and cohort columns."""
    meta = pd.read_csv(path)
    missing = {"Sample", "Group", "type"}.difference(meta.columns)
    if missing:
        raise ValueError("Metadata file is missing required columns: %s" % sorted(missing))
    meta = meta.copy()
    meta["Sample"] = meta["Sample"].astype(str).str.strip()
    meta["Group"] = meta["Group"].apply(normalise_group_id)
    meta["cohort"] = meta["type"].apply(normalise_type)
    other = meta["cohort"] == "other"
    if other.any():
        print("Note: %d metadata rows of type %s are in neither cohort and are ignored."
              % (int(other.sum()), sorted(meta.loc[other, "type"].astype(str).unique())))
        meta = meta[~other].copy()
    if include_column and include_column in meta.columns:
        marks = meta[include_column].astype(str).str.strip().str.lower()
        meta = meta[marks.isin({"yes", "y", "true", "1", "include", "included"})].copy()
    if included_groups is not None:
        wanted = {normalise_group_id(x) for x in included_groups}
        meta = meta[meta["Group"].isin(wanted)].copy()
    if meta.empty:
        raise ValueError("No metadata rows remain after filtering.")
    if meta["Sample"].duplicated().any():
        dup = sorted(set(meta.loc[meta["Sample"].duplicated(), "Sample"]))
        print("WARNING: duplicated sample names in the metadata, first occurrence kept: %s" % dup[:10])
        meta = meta.drop_duplicates("Sample")
    return meta


def group_samples(meta, count_columns, genotype_columns, min_pops):
    """{population: (selection-cohort lines, raw pools)} for the populations with both cohorts."""
    count_columns, genotype_columns = set(count_columns), set(genotype_columns)
    missing = sorted(s for s in meta["Sample"] if s not in count_columns)
    if missing:
        raise ValueError("Metadata samples missing from the count file. Examples: %s" % missing[:10])
    selected_all = meta.loc[meta["cohort"] == "selected", "Sample"]
    missing = sorted(s for s in selected_all if s not in genotype_columns)
    if missing:
        raise ValueError("Selection-cohort lines missing from the genotype file. Examples: %s" % missing[:10])
    groups = {}
    for g in sorted(meta["Group"].unique(), key=group_sort_key):
        gm = meta[meta["Group"] == g]
        selected = list(dict.fromkeys(gm.loc[gm["cohort"] == "selected", "Sample"]))
        raw = list(dict.fromkeys(gm.loc[gm["cohort"] == "raw", "Sample"]))
        if not selected or not raw:
            print("WARNING: population %s does not have both cohorts; skipped." % g)
            continue
        groups[g] = (selected, raw)
    if len(groups) < min_pops:
        raise ValueError("Need at least %d populations with both cohorts; found %d." % (min_pops, len(groups)))
    return groups


def read_locus_table(path, columns):
    """A locus table (Chromosome, Position, samples) restricted to the wanted sample columns."""
    header = pd.read_csv(path, nrows=0).columns
    missing = [k for k in KEYS if k not in header]
    if missing:
        raise ValueError("%s must contain the columns %s." % (path, KEYS))
    wanted = set(KEYS) | {c for c in columns if c in set(header)}
    df = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
    df["Chromosome"] = df["Chromosome"].astype(str).str.strip()
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    if df["Position"].isna().any():
        raise ValueError("%s: some Position values are not numeric." % path)
    df["Position"] = df["Position"].astype(np.int64)
    if df.duplicated(KEYS).any():
        dup = df.loc[df.duplicated(KEYS), KEYS].head()
        raise ValueError("%s: duplicated Chromosome/Position rows:\n%s" % (path, dup))
    return df.set_index(KEYS)


def parse_ref_alt_counts(df):
    """Cells 'REF/ALT' -> REF and ALT count arrays (loci x samples). Invalid cells and zero depth are missing."""
    ref = np.full(df.shape, np.nan)
    alt = np.full(df.shape, np.nan)
    for j, col in enumerate(df.columns):
        parts = df[col].astype("string").str.strip().str.split("/", n=1, expand=True)
        if parts.shape[1] < 2:
            continue
        ref[:, j] = pd.to_numeric(parts[0], errors="coerce")
        alt[:, j] = pd.to_numeric(parts[1], errors="coerce")
    bad = ~(np.isfinite(ref) & np.isfinite(alt)) | ((ref + alt) <= 0)
    ref[bad] = np.nan
    alt[bad] = np.nan
    return ref, alt


def genotype_frequency(df, genotype_values, ploidy):
    """Reference-allele frequency implied by the called genotype of each line (loci x lines)."""
    values = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if genotype_values == "dosage":
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(values) & (values >= 0) & (values <= ploidy) & (np.mod(values, 1) == 0)
        return np.where(valid, 1.0 - values / ploidy, np.nan)
    if genotype_values == "frequency":
        with np.errstate(invalid="ignore"):
            out_of_range = np.isfinite(values) & ((values < 0) | (values > 1))
        if out_of_range.any():
            columns = [c for c, bad in zip(df.columns, out_of_range.any(axis=0)) if bad]
            raise ValueError("Genotype frequencies must lie between 0 and 1; values outside this range in %s. "
                             "Set GENOTYPE_VALUES = 'dosage' if the file holds dosages." % columns[:10])
        return values
    raise ValueError("GENOTYPE_VALUES must be 'frequency' or 'dosage'.")


def cohort_inputs(groups, geno, cnt, genotype_values, ploidy, min_depth):
    """
    Sample-level matrices of every population:
        genotype   reference frequency of each line from its called genotype
        ratio      RO / (RO + AO) of each line
        depth      RO + AO of each line
        raw        RO / (RO + AO) of each pool, with raw_ref and raw_depth
    A line or pool counts only where its depth is above min_depth. A genotype
    call is used only where the line also has reads at the locus, so that the
    three estimators draw on the same lines.
    """
    samples = {}
    for g, (selected, raw) in groups.items():
        ref, alt = parse_ref_alt_counts(cnt[selected])
        depth = ref + alt
        ok = np.isfinite(depth) & (depth > min_depth)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(ok, ref / depth, np.nan)
        pref, palt = parse_ref_alt_counts(cnt[raw])
        pdepth = pref + palt
        pok = np.isfinite(pdepth) & (pdepth > min_depth)
        with np.errstate(divide="ignore", invalid="ignore"):
            pratio = np.where(pok, pref / pdepth, np.nan)
        gfreq = genotype_frequency(geno[selected], genotype_values, ploidy)
        gfreq = np.where(ok & np.isfinite(gfreq), gfreq, np.nan)
        samples[g] = dict(genotype=gfreq, ratio=ratio, depth=np.where(ok, depth, np.nan),
                          raw=pratio, raw_ref=np.where(pok, pref, np.nan),
                          raw_depth=np.where(pok, pdepth, np.nan))
    return samples


###############################################################################
# Cohort frequencies of each estimator
###############################################################################

def nanmean_rows(a):
    n = np.isfinite(a).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.nansum(a, axis=1) / np.maximum(n, 1), np.nan)


def pool_read_variance(ref, depth):
    """Read-sampling variance of the mean pool frequency: (1/m^2) sum_j p*_j (1 - p*_j) / D_j,
    with the continuity-corrected p*_j = (RO_j + 0.5) / (D_j + 1)."""
    valid = np.isfinite(ref) & np.isfinite(depth) & (depth > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (ref + 0.5) / (depth + 1.0)
        each = np.where(valid, p * (1.0 - p) / depth, np.nan)
        m = valid.sum(axis=1)
        return np.where(m > 0, np.nansum(each, axis=1) / np.maximum(m, 1) ** 2, np.nan)


def cohort_matrices(estimator, index, samples):
    """
    Loci x populations matrices of one estimator: sel, n_sel, raw, n_raw and
    read_var, plus the sample-level values used by the bootstrap. For the
    summed-read estimator, n_sel is the effective number of lines
    (sum w)^2 / sum w^2 of the depth weights w, which enters the variance.
    """
    pops = list(samples)
    n = len(index)
    S = dict(name=estimator, index=index, pops=pops, sample={})
    for key in ("sel", "raw", "n_sel", "n_raw", "read_var"):
        S[key] = np.full((n, len(pops)), np.nan)
    for j, g in enumerate(pops):
        x = samples[g]
        if estimator == "genotype_based":
            values, w = x["genotype"], None
        elif estimator == "mean_read_ratio":
            values, w = x["ratio"], None
        else:
            values, w = x["ratio"], x["depth"]
        values = values.copy()
        if w is None:
            S["sel"][:, j] = nanmean_rows(values)
            S["n_sel"][:, j] = np.isfinite(values).sum(axis=1)
        else:
            w = np.where(np.isfinite(values) & np.isfinite(w), w, np.nan)
            wsum = np.nansum(w, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                S["sel"][:, j] = np.where(wsum > 0, np.nansum(values * w, axis=1) / wsum, np.nan)
                S["n_sel"][:, j] = np.where(wsum > 0, wsum ** 2 / np.nansum(w ** 2, axis=1), 0.0)
        S["raw"][:, j] = nanmean_rows(x["raw"])
        S["n_raw"][:, j] = np.isfinite(x["raw"]).sum(axis=1)
        S["read_var"][:, j] = pool_read_variance(x["raw_ref"], x["raw_depth"])
        S["sample"][g] = dict(sel=values, w=w, raw=x["raw"].copy(),
                              raw_ref=x["raw_ref"].copy(), raw_depth=x["raw_depth"].copy())
    return S


def apply_eligibility(strategies, min_selected, min_pops):
    """
    One eligibility decision for all estimators, made before any test:
    a population is used at a locus if at least min_selected lines have a
    genotype call and at least min_selected lines have reads, and at least one
    raw pool has reads; a locus is tested if at least min_pops populations are
    used. Populations that are not used are blanked in every estimator.
    """
    g, c = strategies["genotype_based"], strategies["mean_read_ratio"]
    use = ((g["n_sel"] >= min_selected) & (c["n_sel"] >= min_selected) & (g["n_raw"] >= 1)
           & np.isfinite(g["sel"]) & np.isfinite(c["sel"]))
    tested = use.sum(axis=1) >= min_pops
    for S in strategies.values():
        for key in ("sel", "raw", "n_sel", "n_raw", "read_var"):
            S[key] = np.where(use, S[key], np.nan)
        for j, pop in enumerate(S["pops"]):
            for values in S["sample"][pop].values():
                if values is not None:
                    values[~use[:, j], :] = np.nan
    return use, tested


###############################################################################
# Statistics
###############################################################################

def continuity_corrected(p, n):
    """p* = (p n + 0.5) / (n + 1); used for variance estimation only."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (p * n + 0.5) / (n + 1.0)


def sampling_variance(S, ploidy, plants_per_pool):
    """
    Var(Delta_i) = p_sel (1 - p_sel) / (ploidy n_sel)
                 + p_raw (1 - p_raw) / (ploidy x plants_per_pool x n_rawpool)
                 + (1 / n_rawpool^2) sum_j p_ij (1 - p_ij) / D_ij
    with continuity-corrected frequencies in the first two terms.
    """
    ns = ploidy * S["n_sel"]
    nr = ploidy * plants_per_pool * S["n_raw"]
    ps = continuity_corrected(S["sel"], ns)
    pr = continuity_corrected(S["raw"], nr)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = ps * (1.0 - ps) / ns + pr * (1.0 - pr) / nr + S["read_var"]
    v[(ns <= 0) | (nr <= 0) | ~np.isfinite(v) | (v <= 0)] = np.nan
    return v


def weighted_test(diff, var, min_pops):
    """
    Fixed-effect inverse-variance weighted mean of the population differences
    (loci x populations), its standard error, z and two-sided normal p-value.
    Loci with fewer than min_pops populations receive no result.
    """
    ok = np.isfinite(diff) & np.isfinite(var) & (var > 0)
    w = np.where(ok, 1.0 / np.where(ok, var, 1.0), 0.0)
    d = np.where(ok, diff, 0.0)
    enough = ok.sum(axis=1) >= min_pops
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(enough, (w * d).sum(axis=1) / w.sum(axis=1), np.nan)
        se = np.where(enough, np.sqrt(1.0 / w.sum(axis=1)), np.nan)
        z = mean / se
    p = 2.0 * stats.norm.sf(np.abs(z))
    return mean, se, z, p


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg adjusted p-values; missing values stay missing."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def analyse(S, tested, ploidy, plants_per_pool, min_pops):
    """Weighted test, direction consistency, leave-one-population-out and heterogeneity of one estimator."""
    diff = np.where(tested[:, None], S["sel"] - S["raw"], np.nan)
    var = sampling_variance(S, ploidy, plants_per_pool)
    mu, se, z, p = weighted_test(diff, var, min_pops)
    q = benjamini_hochberg(p)
    S["diff"], S["var"], S["mu"] = diff, var, mu
    n_used = np.isfinite(diff).sum(axis=1)

    cols = {"Tested": tested, "N_Populations_Used": n_used}
    for j, pop in enumerate(S["pops"]):
        cols["Selected_%s" % pop] = S["sel"][:, j]
        cols["Raw_%s" % pop] = S["raw"][:, j]
        cols["Shift_SelectedMinusRaw_%s" % pop] = diff[:, j]
        cols["N_Selected_%s" % pop] = S["n_sel"][:, j]
        cols["N_RawPools_%s" % pop] = S["n_raw"][:, j]
        cols["Var_Shift_%s" % pop] = var[:, j]
    cols["Mean_Shift_SelectedMinusRaw"] = nanmean_rows(diff)
    cols["Weighted_Mean_Shift_SelectedMinusRaw"] = mu
    cols["Weighted_SE"] = se
    cols["Weighted_Z"] = z
    cols["Weighted_P_Value"] = p
    cols["Weighted_P_FDR_BH"] = q
    cols["Significant"] = tested & (q < FDR_THRESHOLD)

    # Direction consistency: populations whose shift has the sign of the weighted mean.
    valid = np.isfinite(diff) & (diff != 0) & np.isfinite(mu[:, None]) & (mu[:, None] != 0)
    same = valid & (np.sign(diff) == np.sign(mu[:, None]))
    n_valid = valid.sum(axis=1)
    frac = np.where(n_valid > 0, same.sum(axis=1) / np.maximum(n_valid, 1), np.nan)
    cols["N_Pop_Same_Direction"] = same.sum(axis=1)
    cols["N_Pop_Direction_Informative"] = n_valid
    cols["Frac_Pop_Same_Direction"] = frac
    cols["Direction_Consistent"] = frac >= DIRECTION_CONSISTENCY_THRESHOLD

    # Leave one population out: the weighted test without each used population in turn.
    passes = np.zeros(len(tested))
    for j in range(diff.shape[1]):
        m, _, _, pj = weighted_test(np.delete(diff, j, axis=1), np.delete(var, j, axis=1), min_pops)
        passes += np.isfinite(diff[:, j]) & (np.sign(m) == np.sign(mu)) & (pj < LOO_P_THRESHOLD)
    loo = np.where(tested, passes / np.maximum(n_used, 1), np.nan)
    cols["LOO_Pass_Frac"] = loo
    cols["LOO_Robust"] = loo >= LOO_MIN_PASS_FRAC

    # Heterogeneity of the population differences around the weighted mean (diagnostic).
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = np.nansum((diff - mu[:, None]) ** 2 / var, axis=1)
    cols["Heterogeneity_Q"] = np.where(tested, Q, np.nan)
    cols["Heterogeneity_P_Value"] = np.where(tested, stats.chi2.sf(Q, np.maximum(n_used - 1, 1)), np.nan)
    return pd.DataFrame(cols, index=S["index"])


def bootstrap(S, rows, reps, seed, ploidy, plants_per_pool, min_pops):
    """
    Within-population bootstrap of the weighted shift for the loci in rows.
    Lines of the selection cohort are resampled with replacement; raw pools
    are resampled where more than one pool exists and kept fixed otherwise.
    Each replicate recalculates the population differences and their variances
    and combines them with inverse-variance weights.
    """
    n = len(S["index"])
    mean, lo, hi, same = (np.full(n, np.nan) for _ in range(4))
    n_valid = np.zeros(n)
    direction = np.sign(S["mu"])
    for i in rows:
        rng_raw = np.random.default_rng(seed + int(i))
        rng_sel = np.random.default_rng(seed + int(i) + 1000000)
        deltas, variances = [], []
        for pop in S["pops"]:
            x = S["sample"][pop]
            ok = np.isfinite(x["sel"][i])
            rp = np.isfinite(x["raw"][i])
            values, pools = x["sel"][i][ok], x["raw"][i][rp]
            if len(values) == 0 or len(pools) == 0:
                continue
            picks = rng_sel.integers(0, len(values), size=(reps, len(values)))
            if x["w"] is None:
                ps = values[picks].mean(axis=1)
                ns = np.full(reps, ploidy * len(values))
            else:
                weights = x["w"][i][ok][picks]
                ps = (values[picks] * weights).sum(axis=1) / weights.sum(axis=1)
                ns = ploidy * weights.sum(axis=1) ** 2 / (weights ** 2).sum(axis=1)
            picks = rng_raw.integers(0, len(pools), size=(reps, len(pools)))
            pr = pools[picks].mean(axis=1)
            nr = ploidy * plants_per_pool * len(pools)
            read_v = pool_read_variance(x["raw_ref"][i][rp][picks], x["raw_depth"][i][rp][picks])
            sm, rm = continuity_corrected(ps, ns), continuity_corrected(pr, nr)
            variances.append(sm * (1.0 - sm) / ns + rm * (1.0 - rm) / nr + read_v)
            deltas.append(ps - pr)
        if len(deltas) < min_pops:
            continue
        boot, _, _, _ = weighted_test(np.column_stack(deltas), np.column_stack(variances), min_pops)
        good = boot[np.isfinite(boot)]
        n_valid[i] = len(good)
        if len(good):
            mean[i] = good.mean()
            lo[i], hi[i] = np.percentile(good, [BOOTSTRAP_CI_LOW, BOOTSTRAP_CI_HIGH])
            if direction[i] != 0 and np.isfinite(direction[i]):
                same[i] = np.mean(np.sign(good) == direction[i])
    excludes = (lo > 0) | (hi < 0)
    return pd.DataFrame({
        "Bootstrap_N_Valid_Reps": n_valid,
        "Bootstrap_Mean_Shift_SelectedMinusRaw": mean,
        "Bootstrap_CI95_Low": lo,
        "Bootstrap_CI95_High": hi,
        "Bootstrap_Direction_Same_Frac": same,
        "Bootstrap_CI_Excludes_Zero": excludes,
        "Bootstrap_Robust": excludes & (same >= BOOTSTRAP_DIRECTION_THRESHOLD)}, index=S["index"])


def force_correlation_psd(R, eps=1e-8):
    """Symmetrise a correlation-like matrix and make it positive semi-definite."""
    R = (np.asarray(R, dtype=float) + np.asarray(R, dtype=float).T) / 2.0
    np.fill_diagonal(R, 1.0)
    val, vec = np.linalg.eigh(R)
    R = vec @ np.diag(np.clip(val, eps, None)) @ vec.T
    R = (R + R.T) / 2.0
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def read_population_grm(path, pops, used_file=None):
    """Correlation matrix of the used populations from the population-level GRM."""
    grm = pd.read_csv(path, index_col=0)
    grm.index = [normalise_pop_label(x) for x in grm.index]
    grm.columns = [normalise_pop_label(x) for x in grm.columns]
    labels = [normalise_pop_label(g) for g in pops]
    missing = [x for x in labels if x not in grm.index or x not in grm.columns]
    if missing:
        raise ValueError("Populations missing from the GRM file: %s" % missing)
    K = grm.loc[labels, labels].astype(float)
    if used_file is not None:
        K.to_csv(used_file)
    K = K.to_numpy()
    K = (K + K.T) / 2.0
    diag = np.diag(K)
    if np.any(~np.isfinite(diag)) or np.any(diag <= 0):
        raise ValueError("The population GRM has missing or non-positive diagonal values.")
    return force_correlation_psd(K / np.sqrt(np.outer(diag, diag)))


def grm_gls(S, rows, R, rhos, ploidy, plants_per_pool):
    """
    GRM-informed generalised least squares of the population shifts y for the
    loci in rows: y_i = mu + e_i with Var(e) = D R_rho D, where D = diag(sqrt(v))
    holds the binomial sampling variance v of both cohorts (without the
    read-depth term, floored at VAR_FLOOR) and R_rho = (1 - rho) I + rho R.
    With u = 1 / sqrt(v), the GLS estimate is mu = (u' Rinv (u y)) / (u' Rinv u)
    with standard error 1 / sqrt(u' Rinv u); it is tested with a t statistic
    on n - 1 degrees of freedom, and p-values are BH-adjusted across the loci
    for each rho. Loci with the same set of used populations share one inverse
    of R_rho.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (S["sel"] * (1.0 - S["sel"]) / (ploidy * S["n_sel"])
               + S["raw"] * (1.0 - S["raw"]) / (ploidy * plants_per_pool * S["n_raw"]))
    var = np.maximum(var, VAR_FLOOR)
    y, v = S["diff"][rows], var[rows]
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    n_ok = ok.sum(axis=1)
    patterns = {}
    for k, pattern in enumerate(map(tuple, ok)):
        patterns.setdefault(pattern, []).append(k)
    n_total = len(S["index"])
    out = {}
    for rho in rhos:
        tag = "rho" + str(rho).replace(".", "p")
        mu, se = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
        for pattern, idx in patterns.items():
            m = np.array(pattern, dtype=bool)
            n = int(m.sum())
            if n < 3:
                continue
            idx = np.array(idx)
            R_rho = (1.0 - rho) * np.eye(n) + rho * R[np.ix_(m, m)]
            Rinv = np.linalg.pinv((R_rho + R_rho.T) / 2.0)
            u = 1.0 / np.sqrt(v[idx][:, m])
            a = np.einsum("ia,ab,ib->i", u, Rinv, u)
            b = np.einsum("ia,ab,ib->i", u, Rinv, u * y[idx][:, m])
            with np.errstate(divide="ignore", invalid="ignore"):
                good = np.isfinite(a) & (a > 0)
                mu[idx[good]] = b[good] / a[good]
                se[idx[good]] = np.sqrt(1.0 / a[good])
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = mu / se
        p = np.where(np.isfinite(tstat), 2.0 * stats.t.sf(np.abs(tstat), df=np.maximum(n_ok - 1, 1)), np.nan)
        q = benjamini_hochberg(p)
        same = np.isfinite(mu) & (np.sign(mu) == np.sign(S["mu"][rows]))
        for name, values in (("MeanShift", mu), ("SE", se), ("T", tstat), ("P", p), ("N_Pops", n_ok.astype(float)),
                             ("FDR", q), ("Same_Direction", same),
                             ("Robust", same & np.isfinite(q) & (q < GRM_FDR_THRESHOLD))):
            col = np.zeros(n_total, dtype=bool) if values.dtype == bool else np.full(n_total, np.nan)
            col[rows] = values
            out["GRM_GLS_%s_%s" % (name, tag)] = col
    return pd.DataFrame(out, index=S["index"])


###############################################################################
# Candidates and output
###############################################################################

def candidate_table(results, grm_tag):
    """Weighted shift and FDR of every locus with each estimator, and the candidate flags."""
    first = results[ESTIMATORS[0]]
    cols = {"Tested": first["Tested"].to_numpy(), "N_Populations_Used": first["N_Populations_Used"].to_numpy()}
    for key in ESTIMATORS:
        cols["%s_Weighted_Shift" % key] = results[key]["Weighted_Mean_Shift_SelectedMinusRaw"].to_numpy()
        cols["%s_Weighted_FDR" % key] = results[key]["Weighted_P_FDR_BH"].to_numpy()
        cols["%s_Significant" % key] = results[key]["Significant"].to_numpy(dtype=bool)
    shifts = np.column_stack([cols["%s_Weighted_Shift" % key] for key in ESTIMATORS])
    same_direction = np.isfinite(shifts).all(axis=1) & ((shifts > 0).all(axis=1) | (shifts < 0).all(axis=1))
    all_three = np.logical_and.reduce([cols["%s_Significant" % key] for key in ESTIMATORS])
    cols["All_Three_Significant"] = all_three
    cols["Same_Direction_All_Three"] = same_direction
    cols["Candidate"] = all_three & same_direction
    for check, column in (("Direction_Consistent", "Direction_Consistent"), ("LOO_Robust", "LOO_Robust"),
                          ("Bootstrap_Robust", "Bootstrap_Robust"),
                          ("GRM_GLS_Robust", "GRM_GLS_Robust_%s" % grm_tag if grm_tag else None)):
        if column is None or column not in first.columns:
            continue
        flags = []
        for key in ESTIMATORS:
            flags.append(results[key][column].to_numpy(dtype=bool))
            cols["%s_%s" % (key, check)] = flags[-1]
        cols["Candidate_%s_All_Three" % check] = cols["Candidate"] & np.logical_and.reduce(flags)
    return pd.DataFrame(cols, index=first.index)


def write_summary(path, overview, results, n_loci, groups, args, grm_tag):
    tested = int(overview["Tested"].sum())
    cand = overview["Candidate"]
    lines = ["Selection-associated allele-frequency analysis: three estimators of the selection-cohort frequency",
             "",
             "Populations used: %s" % ", ".join(str(g) for g in groups),
             "Lines in the selection cohort: %d; raw pools: %d" % (
                 sum(len(s) for s, _ in groups.values()), sum(len(r) for _, r in groups.values())),
             "Loci read: %d; loci tested (at least %d populations used): %d" % (n_loci, args.min_pops, tested),
             "Depth filter: a line or pool counts at a locus if RO + AO > %g" % args.min_depth,
             "Sampling variance: ploidy %g, %g plants per raw pool, plus the read-sampling variance of the pools"
             % (args.ploidy, args.plants_per_pool),
             "",
             "Loci with weighted FDR < %g (BH across the tested loci, separately for each estimator):" % FDR_THRESHOLD]
    for key in ESTIMATORS:
        lines.append("    %-16s %d" % (ESTIMATOR_LABELS[key], int(results[key]["Significant"].sum())))
    lines += ["    all three        %d" % int(overview["All_Three_Significant"].sum()),
              "    candidates       %d (all three significant with the same direction of shift)" % int(cand.sum()),
              ""]
    if cand.any():
        up = int((overview.loc[cand, "%s_Weighted_Shift" % ESTIMATORS[0]] > 0).sum())
        lines.append("Candidates with a higher reference-allele frequency in the selection cohort: %d; lower: %d"
                     % (up, int(cand.sum()) - up))
        lines.append("")
        lines.append("Sensitivity analyses among the candidates (criterion met with all three estimators):")
        checks = (("Direction_Consistent", "direction consistent (>= %g of populations)" % DIRECTION_CONSISTENCY_THRESHOLD),
                  ("LOO_Robust", "leave-one-population-out (p < %g in >= %g of iterations)" % (LOO_P_THRESHOLD, LOO_MIN_PASS_FRAC)),
                  ("Bootstrap_Robust", "bootstrap (95%% CI excludes 0 and >= %g same direction)" % BOOTSTRAP_DIRECTION_THRESHOLD),
                  ("GRM_GLS_Robust", "GRM-informed GLS (FDR < %g, %s)" % (GRM_FDR_THRESHOLD, grm_tag or "not run")))
        for check, label in checks:
            col = "Candidate_%s_All_Three" % check
            if col in overview.columns:
                lines.append("    %-62s %d of %d" % (label, int(overview[col].sum()), int(cand.sum())))
            else:
                lines.append("    %-62s not run" % label)
    lines += ["",
              "Output files: results_<estimator>.csv (one per estimator), results_all_loci_overview.csv, "
              "results_candidates.csv"]
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    print(text)


def parse_arguments():
    ap = argparse.ArgumentParser(
        description="Three-estimator selection-cohort versus raw-cohort allele-frequency analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--metadata", default=METADATA_FILE)
    ap.add_argument("--genotypes", default=GENOTYPE_FILE, help="sample-level genotype frequency (or dosage) file")
    ap.add_argument("--counts", default=COUNT_FILE, help="sample-level REF/ALT read-count file")
    ap.add_argument("--grm", default=POP_GRM_FILE, help="population-level GRM; skipped if the file does not exist")
    ap.add_argument("--outdir", default=OUTPUT_DIR)
    ap.add_argument("--include-column", default=METADATA_INCLUDE_COLUMN)
    ap.add_argument("--groups", default=None,
                    help="comma-separated populations to use, e.g. 1,2,4 (default: every population of the metadata)")
    ap.add_argument("--genotype-values", default=GENOTYPE_VALUES, choices=("frequency", "dosage"))
    ap.add_argument("--ploidy", type=float, default=PLOIDY)
    ap.add_argument("--plants-per-pool", type=float, default=PLANTS_PER_POOL)
    ap.add_argument("--min-depth", type=float, default=MIN_DEPTH)
    ap.add_argument("--min-selected", type=int, default=MIN_SELECTED)
    ap.add_argument("--min-pops", type=int, default=MIN_POPULATIONS)
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_N_REPS if RUN_BOOTSTRAP else 0,
                    help="0 skips the bootstrap")
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_RANDOM_SEED)
    ap.add_argument("--no-grm", action="store_true", help="skip the GRM-informed GLS analysis")
    args = ap.parse_args()
    if args.min_pops < 3:
        raise SystemExit("--min-pops must be at least 3")
    if args.min_selected < 1 or args.plants_per_pool <= 0 or args.ploidy <= 0 or args.min_depth < 0:
        raise SystemExit("Require --min-selected >= 1, --plants-per-pool > 0, --ploidy > 0 and --min-depth >= 0")
    return args


def main():
    args = parse_arguments()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    included_groups = INCLUDED_GROUPS
    if args.groups:
        included_groups = [g for g in args.groups.split(",") if g.strip()]
    print("Reading metadata: %s" % args.metadata)
    meta = load_metadata(args.metadata, args.include_column, included_groups)
    samples = meta["Sample"].tolist()

    print("Reading genotype file: %s" % args.genotypes)
    geno = read_locus_table(args.genotypes, samples)
    print("Reading read-count file: %s" % args.counts)
    cnt = read_locus_table(args.counts, samples)
    missing = ~geno.index.isin(cnt.index)
    if missing.any():
        examples = geno.index[missing][:5].tolist()
        raise ValueError("%d loci of the genotype file are missing from the count file. Examples: %s"
                         % (int(missing.sum()), examples))
    cnt = cnt.reindex(geno.index)
    groups = group_samples(meta, cnt.columns, geno.columns, args.min_pops)
    print("Populations used: %s" % ", ".join(str(g) for g in groups))
    print("Loci: %d" % len(geno))

    inputs = cohort_inputs(groups, geno, cnt, args.genotype_values, args.ploidy, args.min_depth)
    strategies = {key: cohort_matrices(key, geno.index, inputs) for key in ESTIMATORS}
    use, tested = apply_eligibility(strategies, args.min_selected, args.min_pops)
    print("Loci tested in all three analyses: %d" % int(tested.sum()))

    results = {}
    for key in ESTIMATORS:
        print("Weighted test, direction consistency and leave-one-population-out: %s" % ESTIMATOR_LABELS[key])
        results[key] = analyse(strategies[key], tested, args.ploidy, args.plants_per_pool, args.min_pops)

    # Candidates: weighted FDR < threshold with all three estimators, same direction.
    grm_tag = None
    overview = candidate_table(results, grm_tag)
    rows = np.flatnonzero(overview["Candidate"].to_numpy())
    print("Candidates: %d" % len(rows))

    if args.bootstrap_reps > 0 and len(rows):
        for key in ESTIMATORS:
            print("Bootstrap (%d replicates) of %d candidates: %s" % (args.bootstrap_reps, len(rows), ESTIMATOR_LABELS[key]))
            boot = bootstrap(strategies[key], rows, args.bootstrap_reps, args.seed,
                             args.ploidy, args.plants_per_pool, args.min_pops)
            results[key] = pd.concat([results[key], boot], axis=1)

    grm_path = Path(args.grm) if args.grm else None
    if RUN_GRM_GLS and not args.no_grm and grm_path is not None and grm_path.exists():
        print("Reading population-level GRM: %s" % grm_path)
        R = read_population_grm(grm_path, list(groups), out / "population_level_GRM_used.csv")
        grm_tag = "rho" + str(GRM_RHOS[-1]).replace(".", "p")
        tested_rows = np.flatnonzero(tested)
        for key in ESTIMATORS:
            print("GRM-informed GLS of %d tested loci: %s" % (len(tested_rows), ESTIMATOR_LABELS[key]))
            gls = grm_gls(strategies[key], tested_rows, R, GRM_RHOS, args.ploidy, args.plants_per_pool)
            results[key] = pd.concat([results[key], gls], axis=1)
    else:
        print("GRM file not found or GRM step disabled; the GRM-informed GLS analysis is skipped.")

    overview = candidate_table(results, grm_tag)
    for key in ESTIMATORS:
        results[key]["Candidate"] = overview["Candidate"].to_numpy()
        results[key].reset_index().to_csv(out / ("results_%s.csv" % key), index=False)
    overview.reset_index().to_csv(out / "results_all_loci_overview.csv", index=False)
    overview[overview["Candidate"]].reset_index().to_csv(out / "results_candidates.csv", index=False)
    write_summary(out / "results_summary.txt", overview, results, len(geno), groups, args, grm_tag)
    print("Output written to: %s" % out.resolve())


if __name__ == "__main__":
    main()
