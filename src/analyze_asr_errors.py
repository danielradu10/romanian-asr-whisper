import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


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


def duration_bucket(seconds: float) -> str:
    if seconds < 3:
        return "short_<3s"
    if seconds < 6:
        return "medium_3_6s"
    return "long_>6s"


def word_edit_analysis(reference: str, prediction: str) -> dict[str, Any]:
    ref_words = reference.split()
    pred_words = prediction.split()

    n = len(ref_words)
    m = len(pred_words)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "delete"

    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "insert"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == pred_words[j - 1]:
                candidates = [
                    (dp[i - 1][j - 1], "equal"),
                    (dp[i - 1][j] + 1, "delete"),
                    (dp[i][j - 1] + 1, "insert"),
                ]
            else:
                candidates = [
                    (dp[i - 1][j - 1] + 1, "substitute"),
                    (dp[i - 1][j] + 1, "delete"),
                    (dp[i][j - 1] + 1, "insert"),
                ]

            best_cost, best_op = min(candidates, key=lambda item: item[0])
            dp[i][j] = best_cost
            back[i][j] = best_op

    substitutions = []
    deletions = []
    insertions = []

    i = n
    j = m

    while i > 0 or j > 0:
        op = back[i][j]

        if op == "equal":
            i -= 1
            j -= 1
        elif op == "substitute":
            substitutions.append((ref_words[i - 1], pred_words[j - 1]))
            i -= 1
            j -= 1
        elif op == "delete":
            deletions.append(ref_words[i - 1])
            i -= 1
        elif op == "insert":
            insertions.append(pred_words[j - 1])
            j -= 1
        else:
            break

    return {
        "substitutions": list(reversed(substitutions)),
        "deletions": list(reversed(deletions)),
        "insertions": list(reversed(insertions)),
        "num_substitutions": len(substitutions),
        "num_deletions": len(deletions),
        "num_insertions": len(insertions),
        "num_word_edits": len(substitutions) + len(deletions) + len(insertions),
        "num_reference_words": max(n, 1),
    }


def add_text_clusters(
    metadata: pd.DataFrame,
    n_text_clusters: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = metadata.copy()
    metadata["normalized_transcript"] = metadata["transcript"].apply(normalize_text)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
        strip_accents=None,
    )

    tfidf = vectorizer.fit_transform(metadata["normalized_transcript"])

    kmeans = KMeans(
        n_clusters=n_text_clusters,
        random_state=random_seed,
        n_init=10,
    )

    metadata["text_cluster"] = kmeans.fit_predict(tfidf)

    terms = np.array(vectorizer.get_feature_names_out())
    top_terms_rows = []

    for cluster_id, centroid in enumerate(kmeans.cluster_centers_):
        top_indices = centroid.argsort()[::-1][:15]
        top_terms = terms[top_indices].tolist()

        top_terms_rows.append({
            "text_cluster": cluster_id,
            "top_terms": ", ".join(top_terms),
        })

    top_terms_df = pd.DataFrame(top_terms_rows)

    return metadata, top_terms_df


def summarize_group(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    return (
        df.groupby(group_column)
        .agg(
            samples=("audio_path", "count"),
            duration_hours=("duration_seconds", lambda values: values.sum() / 3600),
            mean_duration_seconds=("duration_seconds", "mean"),
            mean_word_count=("reference_word_count", "mean"),
            mean_wer=("sample_wer", "mean"),
            mean_cer=("sample_cer", "mean"),
            mean_substitutions=("num_substitutions", "mean"),
            mean_deletions=("num_deletions", "mean"),
            mean_insertions=("num_insertions", "mean"),
        )
        .reset_index()
        .sort_values("mean_wer", ascending=False)
    )


def save_counter(counter: Counter, output_path: Path, columns: list[str]) -> None:
    rows = []

    for key, count in counter.most_common(50):
        if isinstance(key, tuple):
            row = {columns[index]: value for index, value in enumerate(key)}
            row["count"] = count
        else:
            row = {columns[0]: key, "count": count}

        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def analyze_errors(
    predictions_csv: Path,
    metadata_csv: Path,
    output_dir: Path,
    n_text_clusters: int,
    random_seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(predictions_csv)
    metadata = pd.read_csv(metadata_csv)

    required_prediction_columns = {"audio_path", "reference", "prediction"}
    missing = required_prediction_columns - set(predictions.columns)

    if missing:
        raise ValueError(f"Missing columns in predictions CSV: {missing}")

    metadata_with_text_clusters, top_terms_df = add_text_clusters(
        metadata=metadata,
        n_text_clusters=n_text_clusters,
        random_seed=random_seed,
    )

    columns_to_merge = [
        "audio_path",
        "cluster",
        "text_cluster",
        "duration_seconds",
        "speaker_id",
        "split",
        "transcript",
    ]

    available_columns = [
        column for column in columns_to_merge
        if column in metadata_with_text_clusters.columns
    ]

    df = predictions.merge(
        metadata_with_text_clusters[available_columns],
        on="audio_path",
        how="left",
        suffixes=("", "_metadata"),
    )

    if "cluster" in df.columns:
        df = df.rename(columns={"cluster": "audio_cluster"})

    if "audio_cluster" not in df.columns:
        df["audio_cluster"] = "unknown"

    if "duration_seconds" not in df.columns:
        df["duration_seconds"] = 0.0

    df["reference"] = df["reference"].apply(normalize_text)
    df["prediction"] = df["prediction"].apply(normalize_text)

    substitution_counter = Counter()
    deletion_counter = Counter()
    insertion_counter = Counter()

    rows = []

    for _, row in df.iterrows():
        reference = row["reference"]
        prediction = row["prediction"]

        edit_info = word_edit_analysis(reference, prediction)

        for ref_word, pred_word in edit_info["substitutions"]:
            substitution_counter[(ref_word, pred_word)] += 1

        for word in edit_info["deletions"]:
            deletion_counter[word] += 1

        for word in edit_info["insertions"]:
            insertion_counter[word] += 1

        sample_wer = edit_info["num_word_edits"] / edit_info["num_reference_words"]
        sample_cer = jiwer.cer(reference, prediction)

        dominant_error = "none"

        error_counts = {
            "substitution": edit_info["num_substitutions"],
            "deletion": edit_info["num_deletions"],
            "insertion": edit_info["num_insertions"],
        }

        if sum(error_counts.values()) > 0:
            dominant_error = max(error_counts, key=error_counts.get)

        rows.append({
            **row.to_dict(),
            "sample_wer": sample_wer,
            "sample_cer": sample_cer,
            "reference_word_count": len(reference.split()),
            "prediction_word_count": len(prediction.split()),
            "reference_char_count": len(reference),
            "prediction_char_count": len(prediction),
            "duration_bucket": duration_bucket(float(row.get("duration_seconds", 0.0))),
            "dominant_error": dominant_error,
            "num_substitutions": edit_info["num_substitutions"],
            "num_deletions": edit_info["num_deletions"],
            "num_insertions": edit_info["num_insertions"],
            "num_word_edits": edit_info["num_word_edits"],
        })

    errors_df = pd.DataFrame(rows)

    overall = pd.DataFrame([
        {
            "predictions_file": str(predictions_csv),
            "samples": len(errors_df),
            "overall_wer": jiwer.wer(errors_df["reference"].tolist(), errors_df["prediction"].tolist()),
            "overall_cer": jiwer.cer(errors_df["reference"].tolist(), errors_df["prediction"].tolist()),
            "mean_sample_wer": errors_df["sample_wer"].mean(),
            "mean_sample_cer": errors_df["sample_cer"].mean(),
        }
    ])

    errors_df.to_csv(output_dir / "per_sample_errors.csv", index=False)
    overall.to_csv(output_dir / "overall_error_metrics.csv", index=False)

    top_terms_df.to_csv(output_dir / "top_terms_per_text_cluster.csv", index=False)

    summarize_group(errors_df, "audio_cluster").to_csv(
        output_dir / "error_summary_by_audio_cluster.csv",
        index=False,
    )

    summarize_group(errors_df, "text_cluster").to_csv(
        output_dir / "error_summary_by_text_cluster.csv",
        index=False,
    )

    summarize_group(errors_df, "duration_bucket").to_csv(
        output_dir / "error_summary_by_duration_bucket.csv",
        index=False,
    )

    summarize_group(errors_df, "dominant_error").to_csv(
        output_dir / "error_summary_by_dominant_error.csv",
        index=False,
    )

    worst_samples = (
        errors_df.sort_values(["sample_wer", "sample_cer"], ascending=False)
        .head(50)
    )

    worst_samples[
        [
            "audio_path",
            "reference",
            "prediction",
            "sample_wer",
            "sample_cer",
            "audio_cluster",
            "text_cluster",
            "duration_seconds",
            "dominant_error",
            "num_substitutions",
            "num_deletions",
            "num_insertions",
        ]
    ].to_csv(output_dir / "worst_samples.csv", index=False)

    save_counter(
        substitution_counter,
        output_dir / "frequent_substitutions.csv",
        columns=["reference_word", "predicted_word"],
    )

    save_counter(
        deletion_counter,
        output_dir / "frequent_deletions.csv",
        columns=["deleted_word"],
    )

    save_counter(
        insertion_counter,
        output_dir / "frequent_insertions.csv",
        columns=["inserted_word"],
    )

    print("Overall:")
    print(overall.to_string(index=False))

    print()
    print("Error summary by audio cluster:")
    print(summarize_group(errors_df, "audio_cluster").to_string(index=False))

    print()
    print("Error summary by text cluster:")
    print(summarize_group(errors_df, "text_cluster").to_string(index=False))

    print()
    print("Top terms per text cluster:")
    print(top_terms_df.to_string(index=False))

    print()
    print(f"Saved error analysis to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/processed/metadata_with_clusters_and_split.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/error_analysis"),
    )
    parser.add_argument(
        "--n-text-clusters",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    analyze_errors(
        predictions_csv=args.predictions_csv,
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        n_text_clusters=args.n_text_clusters,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()