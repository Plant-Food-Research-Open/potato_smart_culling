# Selection-associated allele-frequency analysis from sample-level files

This version is appropriate when each breeding population contains multiple selected individuals and one or more raw/unselected pools.

## Script

```text
selection_analysis_sample_level_with_metadata.py
```

## Required input files

The script expects the four anonymised example files included in this package:

```text
example_metadata.csv
example_sample_level_genotype_frequency.csv
example_sample_level_allele_count_Ref_to_Alt.csv
example_population_level_GRM_metadata_version.csv
```

File paths can be changed in the configuration section at the top of the script.

## 1. Metadata file

File name used by this supplementary example:

```text
example_metadata.csv
```

The metadata file maps each sample column to a breeding population and cohort type.

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

If the `Include` column is present, only rows marked `yes`, `true`, `1`, or `include` are used. If the column is absent, all rows are used.

Example:

```csv
Sample,Group,type,Include
Sel1_ind01,1,Selected,yes
Sel1_ind02,1,Selected,yes
Raw1_pool01,1,Raw,yes
Raw1_pool02,1,Raw,yes
Sel2_ind01,2,Selected,yes
Sel2_ind02,2,Selected,yes
Raw2_pool01,2,Raw,yes
Raw2_pool02,2,Raw,yes
```

Accepted selected-cohort labels include:

```text
Selected
selection
sel
```

Accepted raw-cohort labels include:

```text
Raw
unselected
pool
pooled
```

## 2. Sample-level genotype frequency file

File name used by this supplementary example:

```text
example_sample_level_genotype_frequency.csv
```

This file contains allele frequencies for each sample or raw pool.

Required columns:

```text
Chromosome
Position
```

Additional columns should match the `Sample` names in `example_metadata.csv`.

Example:

```csv
Chromosome,Position,Sel1_ind01,Sel1_ind02,Raw1_pool01,Raw1_pool02,Sel2_ind01,Sel2_ind02,Raw2_pool01,Raw2_pool02
ST4.03ch01,1000,0.75,0.50,0.48,0.52,0.75,0.50,0.45,0.47
ST4.03ch01,2000,0.25,0.25,0.30,0.35,0.25,0.50,0.40,0.38
ST4.03ch02,1500,0.50,0.75,0.54,0.56,0.50,0.50,0.52,0.53
```

Each value should be an allele frequency between 0 and 1.

In this analysis, the frequency is treated as reference-allele frequency. For each population and locus, the script calculates:

```text
mean selected frequency - mean raw frequency
```

## 3. Sample-level REF/ALT count file

File name used by this supplementary example:

```text
example_sample_level_allele_count_Ref_to_Alt.csv
```

This file provides REF/ALT read counts for the same samples and loci as the genotype frequency file.

Required columns:

```text
Chromosome
Position
```

Additional columns should match the `Sample` names in `example_metadata.csv`.

Each count cell should be formatted as:

```text
REF/ALT
```

Example:

```csv
Chromosome,Position,Sel1_ind01,Sel1_ind02,Raw1_pool01,Raw1_pool02,Sel2_ind01,Sel2_ind02,Raw2_pool01,Raw2_pool02
ST4.03ch01,1000,150/50,100/100,96/104,104/96,150/50,100/100,90/110,94/106
ST4.03ch01,2000,50/150,50/150,60/140,70/130,50/150,100/100,80/120,76/124
ST4.03ch02,1500,100/100,150/50,108/92,112/88,100/100,100/100,104/96,106/94
```

The script uses these counts to estimate variance for the inverse-variance weighted sensitivity analysis and the GRM-informed GLS sensitivity analysis.

## 4. Population-level GRM file

File name used by this supplementary example:

```text
example_population_level_GRM_metadata_version.csv
```

This file is a square population-level relationship matrix used in the GRM-informed GLS sensitivity analysis.

The population labels should correspond to the `Group` values in `example_metadata.csv`. Labels such as `1`, `Pop1`, and `Pop 1` are normalised by the script.

Example:

```csv
,Pop1,Pop2
Pop1,1.00,0.25
Pop2,0.25,1.00
```

## Main output

The main output file is:

```text
selected_unselected_frequency_results.csv
```

The script also writes:

```text
population_level_GRM_used_by_common_ids.csv
```

## Main analyses performed

The script performs:

1. Sample-level grouping using the metadata file.
2. Population-level selected and raw cohort mean allele-frequency calculation.
3. Selected-minus-raw allele-frequency shift calculation for each population.
4. Gst/Fst-like differentiation calculation between selected and raw cohorts.
5. One-sample t-test of selected-minus-raw shifts across populations.
6. Benjamini-Hochberg FDR correction.
7. Primary candidate identification using:

```text
FDR < 0.01
Average_Gst > 0.03
```

8. Inverse-variance weighted sensitivity analysis using sample-level REF/ALT counts.
9. Direction-consistency analysis across populations.
10. Leave-one-population-out sensitivity analysis for primary candidates.
11. Within-population bootstrap resampling of selected individuals and raw pools.
12. Population-level GRM-informed GLS sensitivity analysis.

## Dependencies

Required Python packages:

```text
pandas==2.2.3
numpy==2.0.2
scipy==1.13.1
statsmodels==0.14.4
```

Install using pip:

```bash
pip install -r requirements_sample_level.txt
```

or using conda:

```bash
conda install pandas numpy scipy statsmodels
```

## Example command

```bash
python selection_analysis_sample_level_with_metadata.py
```

With the files included in this package, the command above runs on the
anonymised example data and writes `selected_unselected_frequency_results.csv`.

## Notes

The primary one-sample t-test requires at least two populations with both
selected and raw samples. Some sensitivity analyses are more informative with
larger numbers of populations: the inverse-variance weighted test and
GRM-informed GLS require at least three valid population shifts, and
leave-one-population-out support is most useful when at least four populations
are available.
