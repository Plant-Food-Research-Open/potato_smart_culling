import argparse
from math import lgamma, log, exp
import pandas as pd
from tqdm import tqdm


# Mapping of ALT dosage to genotype labels
# Here:
# 0 ALT copies = AAAA
# 1 ALT copy   = AAAB
# 2 ALT copies = AABB
# 3 ALT copies = ABBB
# 4 ALT copies = BBBB
dosage_to_genotype = {
    0: "AAAA",
    1: "AAAB",
    2: "AABB",
    3: "ABBB",
    4: "BBBB",
}


def extract_ref_alt(value):
    """
    Extract REF and ALT read counts from a cell formatted as REF/ALT.

    Examples:
        3036/56 -> ref=3036, alt=56
        0/31    -> ref=0, alt=31

    Missing or malformed values return None, None.
    """

    if pd.isna(value):
        return None, None

    value = str(value).strip()

    if value in {"", ".", "NA", "NaN", "nan", "./."}:
        return None, None

    parts = value.split("/")

    if len(parts) != 2:
        return None, None

    try:
        ref = int(parts[0])
        alt = int(parts[1])
    except ValueError:
        return None, None

    if ref < 0 or alt < 0:
        return None, None

    return ref, alt


def clip_probability(p, tiny=1e-15):
    """
    Avoid exact 0 or exact 1 probabilities.

    This prevents log(0) while keeping probabilities biologically close
    to 0 or 1.
    """

    if p <= 0.0:
        return tiny

    if p >= 1.0:
        return 1.0 - tiny

    return p


def log_binom_pmf(n, k, p):
    """
    Log probability of observing k REF reads out of n total reads
    under a Binomial(n, p) model.

    This replaces both exact raw binomial probability and normal approximation.

    It is stable for:
        - low depth
        - high depth
        - extreme REF/ALT ratios
        - near-homozygous cases
    """

    p = clip_probability(p)

    return (
        lgamma(n + 1)
        - lgamma(k + 1)
        - lgamma(n - k + 1)
        + k * log(p)
        + (n - k) * log(1.0 - p)
    )


def logsumexp(values):
    """
    Stable calculation of log(sum(exp(values))).
    """

    max_value = max(values)

    return max_value + log(sum(exp(v - max_value) for v in values))


def calculate_probabilities(ref_count, alt_count, error=0.001):
    """
    Calculate posterior probabilities for tetraploid genotype classes.

    The model assumes ideal REF fractions:

        AAAA: 1.00
        AAAB: 0.75
        AABB: 0.50
        ABBB: 0.25
        BBBB: 0.00

    Homozygous classes use the error-adjusted probabilities:

        AAAA: p_ref = 1 - error
        BBBB: p_ref = error

    Heterozygous classes use a symmetric two-component mixture:

        0.5 * Binomial(p = f + error)
      + 0.5 * Binomial(p = f - error)

    All likelihoods are calculated on the log scale.
    """

    n = ref_count + alt_count

    if n == 0:
        return {}

    # ALT dosage -> expected REF fraction
    f_values = {
        0: 1.0,       # AAAA
        1: 3 / 4,     # AAAB
        2: 2 / 4,     # AABB
        3: 1 / 4,     # ABBB
        4: 0.0,       # BBBB
    }

    log_likelihoods = {}

    for dosage, f in f_values.items():

        if dosage == 0:
            # AAAA: nearly all reads should support REF
            p_ref = 1.0 - error
            log_likelihoods[dosage] = log_binom_pmf(n, ref_count, p_ref)

        elif dosage == 4:
            # BBBB: nearly no reads should support REF
            p_ref = error
            log_likelihoods[dosage] = log_binom_pmf(n, ref_count, p_ref)

        else:
            # Heterozygous classes: symmetric mixture around expected REF fraction
            p_plus = clip_probability(f + error)
            p_minus = clip_probability(f - error)

            log_likelihood_plus = log(0.5) + log_binom_pmf(n, ref_count, p_plus)
            log_likelihood_minus = log(0.5) + log_binom_pmf(n, ref_count, p_minus)

            log_likelihoods[dosage] = logsumexp(
                [log_likelihood_plus, log_likelihood_minus]
            )

    # Convert log-likelihoods to normalized posterior-like probabilities
    denominator = logsumexp(list(log_likelihoods.values()))

    probabilities = {
        dosage: exp(log_likelihood - denominator)
        for dosage, log_likelihood in log_likelihoods.items()
    }

    return probabilities


def calculate_probabilities_for_row(row, columns, error=0.001, confidence_threshold=0.95):
    """
    Infer genotypes for one SNP row across all genotype/sample columns.
    """

    inferred_genotypes = {}

    for col in columns:

        ref, alt = extract_ref_alt(row[col])

        if ref is None or alt is None:
            inferred_genotypes[col] = "No reads"
            continue

        probabilities = calculate_probabilities(ref, alt, error=error)

        if not probabilities:
            inferred_genotypes[col] = "No reads"
            continue

        best_dosage = max(probabilities, key=probabilities.get)
        best_genotype = dosage_to_genotype[best_dosage]
        best_prob = probabilities[best_dosage]

        if best_prob >= confidence_threshold:
            inferred_genotypes[col] = f"{best_genotype} ({best_prob * 100:.2f}%)"
        else:
            inferred_genotypes[col] = "Low confidence"

    return inferred_genotypes


def main():
    parser = argparse.ArgumentParser(
        description="Infer tetraploid genotypes from REF/ALT read counts in a CSV file."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to the input CSV file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Path to the output CSV file.",
    )

    parser.add_argument(
        "--error",
        default=0.001,
        type=float,
        help="Sequencing/error term used for genotype likelihoods. Default: 0.001",
    )

    parser.add_argument(
        "--confidence",
        default=0.95,
        type=float,
        help="Minimum posterior probability required to report a genotype. Default: 0.95",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Assume first two columns are metadata, e.g. Chromosome and Position
    metadata_columns = list(df.columns[:2])

    # Assume remaining columns contain REF/ALT counts
    genotype_columns = list(df.columns[2:])

    inferred_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        inferred_data.append(
            calculate_probabilities_for_row(
                row,
                genotype_columns,
                error=args.error,
                confidence_threshold=args.confidence,
            )
        )

    inferred_df = pd.DataFrame(inferred_data, index=df.index)

    output_df = pd.concat([df[metadata_columns], inferred_df], axis=1)

    output_df = output_df.drop_duplicates(subset=metadata_columns, keep="first")

    output_df.to_csv(args.output, index=False)

    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
