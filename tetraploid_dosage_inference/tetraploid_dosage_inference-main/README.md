# Tetraploid Dosage Inference from REF/ALT Read Counts

This repository contains a Python script for inferring allele dosage genotypes in autotetraploid potato from sequencing-derived REF/ALT read counts.

The script implements a modified naïve binomial dosage-calling approach on the log-likelihood scale. It is designed for biallelic SNP data where each sample genotype is represented as a read-count string such as `39/0`, `79/25`, or `52/52`.

The method is intended to provide a fast, transparent, and practical genotype-calling approach for high-confidence dosage-aware analysis in tetraploid potato breeding populations.

---

## Overview

In an autotetraploid organism, a biallelic SNP can occur in five possible dosage states.

In this script:

- `A` represents the reference allele.
- `B` represents the alternate allele.
- Dosage refers to the number of ALT allele copies.

| ALT dosage | Genotype | Expected REF fraction | Expected ALT fraction |
|---:|---|---:|---:|
| 0 | `AAAA` | 1.00 | 0.00 |
| 1 | `AAAB` | 0.75 | 0.25 |
| 2 | `AABB` | 0.50 | 0.50 |
| 3 | `ABBB` | 0.25 | 0.75 |
| 4 | `BBBB` | 0.00 | 1.00 |

For each SNP and sample, the script uses the observed REF and ALT read counts to estimate the most likely dosage class.

Only genotype calls with normalized probability greater than or equal to the user-defined confidence threshold are reported. The default threshold is `0.95`.

---

## Key features

- Infers tetraploid dosage genotypes from `REF/ALT` read-count data.
- Supports the five expected autotetraploid dosage classes: `AAAA`, `AAAB`, `AABB`, `ABBB`, and `BBBB`.
- Uses log binomial likelihoods for numerical stability.
- Avoids underflow in high-depth sequencing data using log-scale calculations and `logsumexp`.
- Includes a small sequencing-error term to handle near-homozygous sites.
- Reports only high-confidence genotype calls.
- Allows user-defined error and confidence thresholds.
- Handles missing or malformed values safely.
- Removes duplicated marker positions based on the metadata columns.
- Produces a clean CSV output suitable for downstream genetic analysis.

---

## Input format

The input file should be a comma-separated CSV file.

The first two columns are treated as marker metadata. For example:

```text
Chromosome,Position
```

All remaining columns are assumed to contain sample-level REF/ALT read counts.

Example input:

```csv
Chromosome,Position,Sample_1,Sample_2,Sample_3,Sample_4
ST4.03ch01,12345,39/0,79/25,52/52,0/39
ST4.03ch01,67890,3036/56,25/79,40/3,./.
```

Each genotype cell should be formatted as:

```text
REF/ALT
```

Examples:

```text
39/0
79/25
52/52
25/79
0/39
3036/56
```

Missing or malformed values such as the following are treated as missing data:

```text
.
NA
NaN
nan
./.
empty cells
malformed strings
negative read counts
```

---

## Output format

The output file is a CSV containing the original metadata columns followed by inferred genotype calls for each sample.

Example output:

```csv
Chromosome,Position,Sample_1,Sample_2,Sample_3,Sample_4
ST4.03ch01,12345,AAAA (99.99%),AAAB (99.99%),AABB (99.99%),BBBB (99.99%)
ST4.03ch01,67890,AAAA (99.99%),ABBB (99.99%),Low confidence,No reads
```

Possible output labels include:

| Output label | Meaning |
|---|---|
| `AAAA (xx.xx%)` | Confident genotype call with normalized probability |
| `AAAB (xx.xx%)` | Confident genotype call with normalized probability |
| `AABB (xx.xx%)` | Confident genotype call with normalized probability |
| `ABBB (xx.xx%)` | Confident genotype call with normalized probability |
| `BBBB (xx.xx%)` | Confident genotype call with normalized probability |
| `Low confidence` | Best genotype probability is below the confidence threshold |
| `No reads` | Missing, malformed, or zero-depth genotype information |

Duplicated marker rows are removed using the first two metadata columns. Only the first occurrence is retained.

---

## Method

For each sample at each SNP, the script extracts the REF and ALT read counts:

```text
n = REF + ALT
```

where:

- `REF` is the number of reads supporting the reference allele.
- `ALT` is the number of reads supporting the alternate allele.
- `n` is the total read depth.

The model evaluates the likelihood of observing the REF read count under each of the five tetraploid dosage classes.

The expected REF fractions are:

```text
AAAA: 1.00
AAAB: 0.75
AABB: 0.50
ABBB: 0.25
BBBB: 0.00
```

A small sequencing-error term is used:

```text
error = 0.001
```

The homozygous classes are modelled as:

```text
AAAA: p_ref = 1 - error
BBBB: p_ref = error
```

The heterozygous classes are modelled using a symmetric two-component mixture around the expected REF fraction:

```text
0.5 × Binomial(p = f + error) + 0.5 × Binomial(p = f - error)
```

where `f` is the expected REF fraction for the genotype class.

All likelihoods are calculated on the log scale using the binomial log-probability mass function:

```text
log P(k | n, p)
```

where:

- `k` is the observed REF read count,
- `n` is the total read depth,
- `p` is the expected REF probability for a given genotype class.

The script then normalizes the log-likelihoods across all five genotype classes using `logsumexp`, generating posterior-like probabilities.

The genotype with the highest normalized probability is selected as the best call.

If the best probability is greater than or equal to the confidence threshold, the genotype is reported. Otherwise, the genotype is labelled as `Low confidence`.

---

## Usage

Run the script from the command line:

```bash
python dosage_inference.py \
  --input input_ref_alt_counts.csv \
  --output inferred_genotypes.csv
```

Short option names are also supported:

```bash
python dosage_inference.py \
  -i input_ref_alt_counts.csv \
  -o inferred_genotypes.csv
```

---

## Command-line options

| Option | Required | Default | Description |
|---|---:|---:|---|
| `-i`, `--input` | Yes | NA | Path to the input CSV file |
| `-o`, `--output` | Yes | NA | Path to the output CSV file |
| `--error` | No | `0.001` | Sequencing/error term used in genotype likelihoods |
| `--confidence` | No | `0.95` | Minimum normalized probability required to report a genotype |

Example with custom parameters:

```bash
python dosage_inference.py \
  -i potato_ref_alt_counts.csv \
  -o potato_dosage_calls.csv \
  --error 0.001 \
  --confidence 0.95
```

---

## Dependencies

The script requires Python 3 and the following Python packages:

```bash
pip install pandas tqdm
```

It also uses standard Python libraries:

```text
argparse
math
```

---

## Example

Input:

```csv
Chromosome,Position,Sample_1,Sample_2,Sample_3,Sample_4,Sample_5
ST4.03ch01,10001,39/0,79/25,52/52,25/79,0/39
```

Expected interpretation:

| Sample | REF/ALT count | Expected genotype |
|---|---:|---|
| `Sample_1` | `39/0` | `AAAA` |
| `Sample_2` | `79/25` | `AAAB` |
| `Sample_3` | `52/52` | `AABB` |
| `Sample_4` | `25/79` | `ABBB` |
| `Sample_5` | `0/39` | `BBBB` |

Example output:

```csv
Chromosome,Position,Sample_1,Sample_2,Sample_3,Sample_4,Sample_5
ST4.03ch01,10001,AAAA (99.99%),AAAB (99.99%),AABB (99.99%),ABBB (99.99%),BBBB (99.99%)
```
---

## Assumptions and limitations

This script assumes:

1. The organism is autotetraploid.
2. SNPs are biallelic.
3. Input values are REF/ALT read counts.
4. The five genotype classes follow expected tetraploid dosage ratios.
5. All genotype classes are treated with equal prior probability.
6. The sequencing error term is small and symmetric.
7. The confidence value is a normalized likelihood-based probability rather than a fully Bayesian posterior with external priors.


For more complex genotype calling, especially in low-depth or highly biased datasets, specialized polyploid genotype callers such as `updog` may provide a more detailed probabilistic framework.

However, for high-depth datasets with reliable REF/ALT counts, this simplified method provides a transparent and computationally efficient alternative.

---

