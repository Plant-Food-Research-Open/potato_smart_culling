# Selection-associated allele-frequency analysis from sample-level files

This analysis compares the reference-allele frequency of a selection cohort
(individually sequenced lines) with that of a raw cohort (pooled seedlings)
within each breeding population, and combines the populations in an
inverse-variance weighted test. The selection-cohort frequency is estimated in
three ways, and a locus is a candidate only when all three estimators agree.

## Script

```text
analysis.py
```

## Required input files

The script expects the four anonymised example files included in this package:

```text
example_metadata.csv
example_sample_level_genotype_frequency.csv
example_sample_level_allele_count_Ref_to_Alt.csv
example_population_level_GRM_metadata_version.csv
```

File names can be changed in the configuration section at the top of the
script or given as command-line options (`--metadata`, `--genotypes`,
`--counts`, `--grm`, `--outdir`).

## 1. Metadata file

```text
example_metadata.csv
```

The metadata file maps each sample column to a breeding population and cohort.

Required columns:

```text
Sample
Group
type
```

Optional column:

```text
Include
```

If the `Include` column is present, only rows marked `yes`, `true`, `1`, or
`include` are used. If the column is absent, all rows are used. Rows of any
other `type` (for example `Parent`) are ignored.

Example:

```csv
Sample,Group,type,Include
Sel1_ind01,1,Selected,yes
Sel1_ind02,1,Selected,yes
Raw1_pool01,1,Raw,yes
Raw1_pool02,1,Raw,yes
Sel2_ind01,2,Selected,yes
```

Accepted selection-cohort labels include `Selected`, `selection` and `sel`;
accepted raw-cohort labels include `Raw`, `unselected`, `pool` and `pooled`.

## 2. Sample-level genotype frequency file

```text
example_sample_level_genotype_frequency.csv
```

Required columns:

```text
Chromosome
Position
```

One further column per selection-cohort line, named as in the metadata. Each
cell holds the reference-allele frequency implied by the called tetraploid
dosage of that line (1, 0.75, 0.5, 0.25 or 0); a cell is left empty where the
line has no confident call. Columns of samples that are not selection-cohort
lines are ignored. If the file holds alternate-allele dosages (0 to 4) instead,
set `GENOTYPE_VALUES = "dosage"` or pass `--genotype-values dosage`.

Example:

```csv
Chromosome,Position,Sel1_ind01,Sel1_ind02,Sel1_ind03
ST4.03ch01,1490000,0.75,0.5,1.0
ST4.03ch01,1623000,0.25,,0.5
```

## 3. Sample-level REF/ALT count file

```text
example_sample_level_allele_count_Ref_to_Alt.csv
```

Required columns `Chromosome` and `Position`, and one column per sample (lines
and pools) named as in the metadata. Each cell is formatted as `REF/ALT`
(reference and alternate read counts).

Example:

```csv
Chromosome,Position,Sel1_ind01,Sel1_ind02,Raw1_pool01,Raw1_pool02
ST4.03ch01,1490000,41/18,30/31,96/104,104/96
ST4.03ch01,1623000,15/44,0/55,60/140,70/130
```

The read counts give the raw-cohort frequencies (RO / (RO + AO) of each pool),
the two read-based estimators of the selection-cohort frequency, and the
sampling variances of the weighted test.

## 4. Population-level GRM file (optional)

```text
example_population_level_GRM_metadata_version.csv
```

A square population-level relationship matrix. The labels should correspond
to the `Group` values of the metadata; `1`, `Pop1` and `Pop 1` are all
accepted. If the file is absent, the GRM-informed analysis is skipped.

```csv
,Pop1,Pop2,Pop3
Pop1,0.492,0.128,0.136
Pop2,0.128,0.593,0.096
Pop3,0.136,0.096,0.548
```

## Analyses performed

1. Sample-level grouping using the metadata file.
2. Raw-cohort frequency of each population: mean over its pools of
   RO / (RO + AO).
3. Selection-cohort frequency of each population with three estimators:
   - `genotype_based`: mean over the lines of the frequency implied by the
     called dosage (1 - dosage / 4);
   - `mean_read_ratio`: mean over the lines of RO / (RO + AO);
   - `summed_reads`: sum of RO over the lines / sum of (RO + AO) over the
     lines (in-silico pooling).
4. Selection-minus-raw shift for each population and its binomial-like
   sampling variance (ploidy 4, 20 seedlings per raw pool, plus the
   read-sampling variance of the pools).
5. Inverse-variance weighted mean shift across populations, two-sided Z test
   and Benjamini-Hochberg FDR across loci, separately for each estimator.
   A locus needs at least three populations with data.
6. Candidate definition:

```text
weighted FDR < 0.01 with all three estimators, same direction of shift
```

7. Sensitivity analyses: direction consistency (at least 70% of populations
   share the direction), leave-one-population-out (direction kept and
   p < 0.01 in at least 90% of iterations), within-population bootstrap of
   the weighted shift (1,000 replicates; 95% interval excludes zero and at
   least 95% of replicates keep the direction) and a GRM-informed generalised
   least-squares model (FDR < 0.05 at rho = 1). Each criterion is reported
   per estimator and as "met with all three estimators".

## Output files

Written to the output directory (`results` by default):

```text
results_genotype_based.csv
results_mean_read_ratio.csv
results_summed_reads.csv
results_all_loci_overview.csv
results_candidates.csv
results_summary.txt
population_level_GRM_used.csv
```

The three `results_<estimator>.csv` files hold one row per locus with the
per-population frequencies, the weighted test and the sensitivity analyses of
that estimator. `results_all_loci_overview.csv` lists the weighted shift and
FDR of every locus with each estimator and the candidate flags;
`results_candidates.csv` is its candidate subset; `results_summary.txt`
gives the counts.

## Dependencies

```text
pandas
numpy
scipy
statsmodels
```

Install using pip:

```bash
pip install -r requirements_sample_level.txt
```

## Example command

```bash
python analysis.py
```

With the files included in this package (six simulated populations, 400 loci,
six loci with a planted shift), the command runs in a few seconds and writes
the output files listed above. Other inputs:

```bash
python analysis.py --metadata meta.csv --genotypes geno.csv --counts counts.csv --grm grm.csv --outdir results
```

Use `--groups 1,2,4` to restrict the analysis to some populations,
`--min-depth` to require a minimum read depth per line or pool,
`--bootstrap-reps 0` to skip the bootstrap and `--no-grm` to skip the
GRM-informed analysis.

## Notes

The weighted test, the leave-one-population-out analysis and the GRM-informed
model all require at least three populations with both cohorts at a locus.
With exactly three populations, leaving one population out leaves too few for
a test, so the leave-one-population-out criterion can only be met with four or
more populations.
