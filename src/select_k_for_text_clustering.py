import argparse
import re
from itertools import combinations
from pathlib import Path
from typing import Optional

import jiwer
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score


ROMANIAN_STOPWORDS = {
    "a", "ai", "al", "ale", "am", "ar", "are", "as", "aș", "au",
    "ca", "că", "cam", "care", "cea", "ce", "cel", "cele", "cei",
    "ceva", "ci", "cu", "cum", "când", "cand", "cât", "cat",
    "da", "dacă", "daca", "dar", "de", "deci", "din", "dintre",
    "după", "dupa", "ea", "ei", "el", "ele", "era", "erau",
    "este", "e", "eu", "fie", "fi", "fost", "foarte",
    "în", "in", "îi", "ii", "îl", "il", "îmi", "imi", "îți", "iti",
    "își", "isi", "la", "le", "lor", "lui", "mai", "mă", "ma",
    "mi", "mie", "ne", "ni", "nici", "noi", "nu", "o", "pe",
    "pentru", "peste", "poate", "prin", "sa", "să", "se", "si", "și",
    "sunt", "suntem", "sunteți", "sau", "te", "tu", "un", "una",
    "unei", "unui", "unor", "vă", "va", "vi", "voi", "vor",

    # demonstratives / filler words that polluted the first text clusters
    "acest", "aceasta", "această", "aceste", "acestea", "acesta",
    "acela", "aceea", "acești", "acestui", "acestei", "acestor",
    "acelui", "acelei", "acele", "acolo", "aici", "asta", "ăsta",
    "ăștia", "lucru", "lucrul", "lucrului", "lucruri",
    "tot", "toate", "toți", "toate",
}


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()

    replacements = {
        "ş": "ș",
        "ţ": "ț",
        "ã": "ă",
        "–": "-",
        "—": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[.,!?;:\"'„”()\[\]]", "", text)
    text = " ".join(text.split())

    return text


def sample_word_error_rate(reference: str, prediction: str) -> float:
    reference = normalize_text(reference)
    prediction = normalize_text(prediction)

    if not reference:
        return 0.0 if not prediction else 1.0

    return float(jiwer.wer(reference, prediction))


def build_vectorizer(
    min_df: int,
    max_df: float,
    max_features: int,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        stop_words=list(ROMANIAN_STOPWORDS),
        strip_accents=None,
        sublinear_tf=True,
    )


def get_top_terms(
    model: KMeans,
    feature_names: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    rows = []

    for cluster_id, centroid in enumerate(model.cluster_centers_):
        top_indices = centroid.argsort()[::-1][:top_n]
        top_terms = feature_names[top_indices].tolist()

        rows.append({
            "cluster": cluster_id,
            "top_terms": ", ".join(top_terms),
        })

    return pd.DataFrame(rows)


def evaluate_error_separation(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    predictions: Optional[pd.DataFrame],
) -> dict[str, float]:
    if predictions is None:
        return {
            "error_cluster_coverage": np.nan,
            "wer_spread": np.nan,
            "min_cluster_mean_wer": np.nan,
            "max_cluster_mean_wer": np.nan,
        }

    label_df = metadata[["audio_path"]].copy()
    label_df["text_cluster"] = labels

    merged = predictions.merge(label_df, on="audio_path", how="inner")

    if merged.empty:
        return {
            "error_cluster_coverage": 0.0,
            "wer_spread": np.nan,
            "min_cluster_mean_wer": np.nan,
            "max_cluster_mean_wer": np.nan,
        }

    merged["sample_wer"] = merged.apply(
        lambda row: sample_word_error_rate(row["reference"], row["prediction"]),
        axis=1,
    )

    cluster_wer = merged.groupby("text_cluster")["sample_wer"].mean()

    if len(cluster_wer) == 0:
        return {
            "error_cluster_coverage": 0.0,
            "wer_spread": np.nan,
            "min_cluster_mean_wer": np.nan,
            "max_cluster_mean_wer": np.nan,
        }

    return {
        "error_cluster_coverage": float(len(merged) / len(predictions)),
        "wer_spread": float(cluster_wer.max() - cluster_wer.min()),
        "min_cluster_mean_wer": float(cluster_wer.min()),
        "max_cluster_mean_wer": float(cluster_wer.max()),
    }


def min_max_normalize(series: pd.Series, higher_is_better: bool) -> pd.Series:
    values = series.astype(float)

    if values.isna().all():
        return pd.Series([0.0] * len(values), index=values.index)

    min_value = values.min()
    max_value = values.max()

    if np.isclose(min_value, max_value):
        return pd.Series([1.0] * len(values), index=values.index)

    normalized = (values - min_value) / (max_value - min_value)

    if not higher_is_better:
        normalized = 1.0 - normalized

    return normalized


def evaluate_k_values(
    metadata_csv: Path,
    predictions_csv: Optional[Path],
    output_dir: Path,
    min_k: int,
    max_k: int,
    seeds: list[int],
    min_df: int,
    max_df: float,
    max_features: int,
    min_cluster_samples: int,
    max_balance_ratio: float,
    top_terms: int,
    silhouette_sample_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_csv)

    if "audio_path" not in metadata.columns or "transcript" not in metadata.columns:
        raise ValueError("metadata CSV must contain audio_path and transcript columns.")

    metadata = metadata.copy()
    metadata["normalized_transcript"] = metadata["transcript"].apply(normalize_text)

    predictions = None

    if predictions_csv is not None:
        predictions = pd.read_csv(predictions_csv)

        required_columns = {"audio_path", "reference", "prediction"}
        missing = required_columns - set(predictions.columns)

        if missing:
            raise ValueError(f"Missing required columns in predictions CSV: {missing}")

    vectorizer = build_vectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )

    tfidf = vectorizer.fit_transform(metadata["normalized_transcript"])
    feature_names = np.array(vectorizer.get_feature_names_out())

    summary_rows = []
    top_terms_rows = []

    n_samples = tfidf.shape[0]
    sample_size = min(silhouette_sample_size, n_samples)

    for k in range(min_k, max_k + 1):
        print(f"Evaluating k={k}")

        seed_labels = []
        seed_silhouettes = []
        seed_min_samples = []
        seed_max_samples = []
        seed_balance_ratios = []
        seed_error_separations = []

        first_model = None

        for seed in seeds:
            model = KMeans(
                n_clusters=k,
                random_state=seed,
                n_init=10,
            )

            labels = model.fit_predict(tfidf)

            if first_model is None:
                first_model = model

            seed_labels.append(labels)

            counts = np.bincount(labels, minlength=k)
            min_count = int(counts.min())
            max_count = int(counts.max())
            balance_ratio = float(max_count / max(min_count, 1))

            seed_min_samples.append(min_count)
            seed_max_samples.append(max_count)
            seed_balance_ratios.append(balance_ratio)

            try:
                silhouette = silhouette_score(
                    tfidf,
                    labels,
                    metric="cosine",
                    sample_size=sample_size,
                    random_state=seed,
                )
            except Exception:
                silhouette = np.nan

            seed_silhouettes.append(float(silhouette))

            error_info = evaluate_error_separation(
                metadata=metadata,
                labels=labels,
                predictions=predictions,
            )

            seed_error_separations.append(error_info)

        ari_scores = []

        for labels_a, labels_b in combinations(seed_labels, 2):
            ari_scores.append(adjusted_rand_score(labels_a, labels_b))

        if ari_scores:
            stability_ari_mean = float(np.mean(ari_scores))
            stability_ari_std = float(np.std(ari_scores))
        else:
            stability_ari_mean = 1.0
            stability_ari_std = 0.0

        error_df = pd.DataFrame(seed_error_separations)

        silhouette_mean = float(np.nanmean(seed_silhouettes))
        silhouette_std = float(np.nanstd(seed_silhouettes))

        min_cluster_samples_mean = float(np.mean(seed_min_samples))
        min_cluster_samples_min = int(np.min(seed_min_samples))
        max_cluster_samples_mean = float(np.mean(seed_max_samples))
        balance_ratio_mean = float(np.mean(seed_balance_ratios))

        passes_size_constraints = (
            min_cluster_samples_min >= min_cluster_samples
            and balance_ratio_mean <= max_balance_ratio
        )

        summary_rows.append({
            "k": k,
            "silhouette_cosine_mean": silhouette_mean,
            "silhouette_cosine_std": silhouette_std,
            "stability_ari_mean": stability_ari_mean,
            "stability_ari_std": stability_ari_std,
            "min_cluster_samples_mean": min_cluster_samples_mean,
            "min_cluster_samples_min": min_cluster_samples_min,
            "max_cluster_samples_mean": max_cluster_samples_mean,
            "balance_ratio_mean": balance_ratio_mean,
            "passes_size_constraints": passes_size_constraints,
            "error_cluster_coverage_mean": error_df["error_cluster_coverage"].mean()
            if "error_cluster_coverage" in error_df else np.nan,
            "wer_spread_mean": error_df["wer_spread"].mean()
            if "wer_spread" in error_df else np.nan,
            "min_cluster_mean_wer_mean": error_df["min_cluster_mean_wer"].mean()
            if "min_cluster_mean_wer" in error_df else np.nan,
            "max_cluster_mean_wer_mean": error_df["max_cluster_mean_wer"].mean()
            if "max_cluster_mean_wer" in error_df else np.nan,
        })

        terms_df = get_top_terms(
            model=first_model,
            feature_names=feature_names,
            top_n=top_terms,
        )

        terms_df.insert(0, "k", k)
        top_terms_rows.append(terms_df)

    summary = pd.DataFrame(summary_rows)

    summary["silhouette_score_norm"] = min_max_normalize(
        summary["silhouette_cosine_mean"],
        higher_is_better=True,
    )

    summary["stability_score_norm"] = min_max_normalize(
        summary["stability_ari_mean"],
        higher_is_better=True,
    )

    summary["min_cluster_size_score_norm"] = min_max_normalize(
        summary["min_cluster_samples_min"],
        higher_is_better=True,
    )

    summary["balance_score_norm"] = min_max_normalize(
        summary["balance_ratio_mean"],
        higher_is_better=False,
    )

    summary["practical_score"] = (
        0.35 * summary["silhouette_score_norm"]
        + 0.25 * summary["stability_score_norm"]
        + 0.20 * summary["min_cluster_size_score_norm"]
        + 0.20 * summary["balance_score_norm"]
    )

    summary["practical_score_with_constraints"] = summary["practical_score"]

    summary.loc[
        ~summary["passes_size_constraints"],
        "practical_score_with_constraints",
    ] *= 0.5

    summary = summary.sort_values(
        ["practical_score_with_constraints", "practical_score"],
        ascending=False,
    )

    top_terms_all = pd.concat(top_terms_rows, ignore_index=True)

    detailed_path = output_dir / "text_k_selection_summary.csv"
    top_terms_path = output_dir / "text_k_selection_top_terms.csv"

    summary.to_csv(detailed_path, index=False)
    top_terms_all.to_csv(top_terms_path, index=False)

    print()
    print("K selection summary:")
    print(summary.to_string(index=False))

    best_row = summary.iloc[0]

    print()
    print("Suggested k:")
    print(int(best_row["k"]))

    print()
    print(f"Saved summary to: {detailed_path}")
    print(f"Saved top terms to: {top_terms_path}")


def parse_seeds(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/processed/metadata_with_clusters_and_split.csv"),
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/text_k_selection_stopwords"),
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="13,42,77",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--min-cluster-samples",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max-balance-ratio",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--top-terms",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=2000,
    )

    args = parser.parse_args()

    evaluate_k_values(
        metadata_csv=args.metadata_csv,
        predictions_csv=args.predictions_csv,
        output_dir=args.output_dir,
        min_k=args.min_k,
        max_k=args.max_k,
        seeds=parse_seeds(args.seeds),
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        min_cluster_samples=args.min_cluster_samples,
        max_balance_ratio=args.max_balance_ratio,
        top_terms=args.top_terms,
        silhouette_sample_size=args.silhouette_sample_size,
    )


if __name__ == "__main__":
    main()