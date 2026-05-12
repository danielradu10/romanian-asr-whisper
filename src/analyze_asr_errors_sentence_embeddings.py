import argparse
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
    "acest", "aceasta", "această", "aceste", "acestea", "acesta",
    "acela", "aceea", "acești", "acestui", "acestei", "acestor",
    "acelui", "acelei", "acele", "acolo", "aici", "asta", "ăsta",
    "ăștia", "lucru", "lucrul", "lucrului", "lucruri",
    "tot", "toate", "toți",
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


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_or_load_embeddings(
    texts: list[str],
    model_name: str,
    cache_path: Path,
    batch_size: int,
    force_recompute: bool,
) -> np.ndarray:
    if cache_path.exists() and not force_recompute:
        embeddings = np.load(cache_path)

        if embeddings.shape[0] == len(texts):
            print(f"Loaded cached sentence embeddings from: {cache_path}")
            print(f"Embeddings shape: {embeddings.shape}")
            return embeddings

        print("Cached embeddings do not match metadata length. Recomputing.")

    device = choose_device()
    print(f"Using device for sentence embeddings: {device}")
    print(f"Loading sentence embedding model: {model_name}")

    model = SentenceTransformer(model_name, device=device)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)

    print(f"Saved sentence embeddings to: {cache_path}")
    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings


def maybe_apply_pca(
    embeddings: np.ndarray,
    use_pca: bool,
    pca_components: int,
    random_seed: int,
) -> np.ndarray:
    if not use_pca:
        return embeddings

    n_components = min(pca_components, embeddings.shape[1], embeddings.shape[0] - 1)

    print(f"Applying PCA to sentence embeddings: {embeddings.shape[1]} -> {n_components}")

    pca = PCA(
        n_components=n_components,
        random_state=random_seed,
    )

    reduced = pca.fit_transform(embeddings)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

    return reduced


def sample_wer(reference: str, prediction: str) -> float:
    reference = normalize_text(reference)
    prediction = normalize_text(prediction)

    if not reference:
        return 0.0 if not prediction else 1.0

    return float(jiwer.wer(reference, prediction))


def evaluate_error_separation(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    predictions: pd.DataFrame | None,
) -> dict[str, float]:
    if predictions is None:
        return {
            "error_cluster_coverage": np.nan,
            "wer_spread": np.nan,
            "min_cluster_mean_wer": np.nan,
            "max_cluster_mean_wer": np.nan,
        }

    label_df = metadata[["audio_path"]].copy()
    label_df["sentence_cluster"] = labels

    merged = predictions.merge(label_df, on="audio_path", how="inner")

    if merged.empty:
        return {
            "error_cluster_coverage": 0.0,
            "wer_spread": np.nan,
            "min_cluster_mean_wer": np.nan,
            "max_cluster_mean_wer": np.nan,
        }

    merged["sample_wer"] = merged.apply(
        lambda row: sample_wer(row["reference"], row["prediction"]),
        axis=1,
    )

    cluster_wer = merged.groupby("sentence_cluster")["sample_wer"].mean()

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


def select_k(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    predictions: pd.DataFrame | None,
    output_dir: Path,
    min_k: int,
    max_k: int,
    seeds: list[int],
    min_cluster_samples: int,
    max_balance_ratio: float,
    silhouette_sample_size: int,
) -> tuple[pd.DataFrame, int]:
    summary_rows = []
    n_samples = embeddings.shape[0]
    sample_size = min(silhouette_sample_size, n_samples)

    for k in range(min_k, max_k + 1):
        print(f"Evaluating sentence embedding k={k}")

        seed_labels = []
        seed_silhouettes = []
        seed_min_samples = []
        seed_max_samples = []
        seed_balance_ratios = []
        seed_error_rows = []

        for seed in seeds:
            model = KMeans(
                n_clusters=k,
                random_state=seed,
                n_init=10,
            )

            labels = model.fit_predict(embeddings)
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
                    embeddings,
                    labels,
                    metric="cosine",
                    sample_size=sample_size,
                    random_state=seed,
                )
            except Exception:
                silhouette = np.nan

            seed_silhouettes.append(float(silhouette))

            seed_error_rows.append(
                evaluate_error_separation(
                    metadata=metadata,
                    labels=labels,
                    predictions=predictions,
                )
            )

        ari_scores = []

        for labels_a, labels_b in combinations(seed_labels, 2):
            ari_scores.append(adjusted_rand_score(labels_a, labels_b))

        error_df = pd.DataFrame(seed_error_rows)

        min_cluster_samples_min = int(np.min(seed_min_samples))
        balance_ratio_mean = float(np.mean(seed_balance_ratios))

        passes_size_constraints = (
            min_cluster_samples_min >= min_cluster_samples
            and balance_ratio_mean <= max_balance_ratio
        )

        summary_rows.append({
            "k": k,
            "silhouette_cosine_mean": float(np.nanmean(seed_silhouettes)),
            "silhouette_cosine_std": float(np.nanstd(seed_silhouettes)),
            "stability_ari_mean": float(np.mean(ari_scores)) if ari_scores else 1.0,
            "stability_ari_std": float(np.std(ari_scores)) if ari_scores else 0.0,
            "min_cluster_samples_mean": float(np.mean(seed_min_samples)),
            "min_cluster_samples_min": min_cluster_samples_min,
            "max_cluster_samples_mean": float(np.mean(seed_max_samples)),
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

    summary_path = output_dir / "sentence_k_selection_summary.csv"
    summary.to_csv(summary_path, index=False)

    suggested_k = int(summary.iloc[0]["k"])

    print()
    print("Sentence embedding k-selection summary:")
    print(summary.to_string(index=False))
    print()
    print(f"Suggested k: {suggested_k}")
    print(f"Saved k-selection summary to: {summary_path}")

    return summary, suggested_k


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


def add_top_terms_per_cluster(
    assignments: pd.DataFrame,
    cluster_column: str,
    output_path: Path,
    top_n: int,
) -> None:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.80,
        max_features=5000,
        stop_words=list(ROMANIAN_STOPWORDS),
        strip_accents=None,
        sublinear_tf=True,
    )

    tfidf = vectorizer.fit_transform(assignments["normalized_transcript"])
    terms = np.array(vectorizer.get_feature_names_out())

    rows = []

    for cluster_id in sorted(assignments[cluster_column].unique()):
        indices = assignments.index[assignments[cluster_column] == cluster_id].tolist()

        if not indices:
            continue

        cluster_scores = np.asarray(tfidf[indices].mean(axis=0)).ravel()
        top_indices = cluster_scores.argsort()[::-1][:top_n]
        top_terms = terms[top_indices].tolist()

        rows.append({
            cluster_column: cluster_id,
            "top_terms": ", ".join(top_terms),
        })

    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_representative_samples(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    model: KMeans,
    output_path: Path,
    examples_per_cluster: int,
) -> None:
    centroids = model.cluster_centers_
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroid_norms[centroid_norms == 0] = 1.0
    normalized_centroids = centroids / centroid_norms

    rows = []

    for cluster_id in sorted(np.unique(labels)):
        cluster_indices = np.where(labels == cluster_id)[0]

        if len(cluster_indices) == 0:
            continue

        cluster_embeddings = embeddings[cluster_indices]
        similarities = cluster_embeddings @ normalized_centroids[cluster_id]

        top_local_indices = np.argsort(similarities)[::-1][:examples_per_cluster]

        for local_index in top_local_indices:
            global_index = cluster_indices[local_index]
            row = metadata.iloc[global_index]

            rows.append({
                "sentence_cluster": int(cluster_id),
                "similarity_to_centroid": float(similarities[local_index]),
                "audio_path": row["audio_path"],
                "split": row.get("split", "unknown"),
                "transcript": row["transcript"],
                "duration_seconds": row.get("duration_seconds", np.nan),
            })

    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_worst_samples_by_cluster(
    errors_df: pd.DataFrame,
    output_path: Path,
    cluster_column: str,
    examples_per_cluster: int,
) -> None:
    rows = []

    for cluster_id, group in errors_df.groupby(cluster_column):
        worst = (
            group.sort_values(["sample_wer", "sample_cer"], ascending=False)
            .head(examples_per_cluster)
        )

        for _, row in worst.iterrows():
            rows.append({
                cluster_column: cluster_id,
                "audio_path": row["audio_path"],
                "reference": row["reference"],
                "prediction": row["prediction"],
                "sample_wer": row["sample_wer"],
                "sample_cer": row["sample_cer"],
                "duration_seconds": row["duration_seconds"],
                "dominant_error": row["dominant_error"],
            })

    pd.DataFrame(rows).to_csv(output_path, index=False)


def analyze_selected_k(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    predictions: pd.DataFrame,
    output_dir: Path,
    selected_k: int,
    random_seed: int,
    examples_per_cluster: int,
) -> None:
    model = KMeans(
        n_clusters=selected_k,
        random_state=random_seed,
        n_init=10,
    )

    labels = model.fit_predict(embeddings)

    assignments = metadata.copy()
    assignments["sentence_cluster"] = labels

    assignments_path = output_dir / "sentence_cluster_assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    save_representative_samples(
        metadata=assignments,
        embeddings=embeddings,
        labels=labels,
        model=model,
        output_path=output_dir / "representative_samples_per_sentence_cluster.csv",
        examples_per_cluster=examples_per_cluster,
    )

    add_top_terms_per_cluster(
        assignments=assignments,
        cluster_column="sentence_cluster",
        output_path=output_dir / "top_terms_per_sentence_cluster.csv",
        top_n=20,
    )

    columns_to_merge = [
        "audio_path",
        "sentence_cluster",
        "cluster",
        "duration_seconds",
        "speaker_id",
        "split",
        "transcript",
    ]

    available_columns = [
        column for column in columns_to_merge
        if column in assignments.columns
    ]

    df = predictions.merge(
        assignments[available_columns],
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

        sample_word_error = edit_info["num_word_edits"] / edit_info["num_reference_words"]
        sample_character_error = jiwer.cer(reference, prediction)

        error_counts = {
            "substitution": edit_info["num_substitutions"],
            "deletion": edit_info["num_deletions"],
            "insertion": edit_info["num_insertions"],
        }

        dominant_error = "none"

        if sum(error_counts.values()) > 0:
            dominant_error = max(error_counts, key=error_counts.get)

        rows.append({
            **row.to_dict(),
            "sample_wer": sample_word_error,
            "sample_cer": sample_character_error,
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
            "samples": len(errors_df),
            "overall_wer": jiwer.wer(
                errors_df["reference"].tolist(),
                errors_df["prediction"].tolist(),
            ),
            "overall_cer": jiwer.cer(
                errors_df["reference"].tolist(),
                errors_df["prediction"].tolist(),
            ),
            "mean_sample_wer": errors_df["sample_wer"].mean(),
            "mean_sample_cer": errors_df["sample_cer"].mean(),
            "selected_k": selected_k,
            "clustering_variant": "sentence_embeddings",
        }
    ])

    errors_df.to_csv(output_dir / "per_sample_errors.csv", index=False)
    overall.to_csv(output_dir / "overall_error_metrics.csv", index=False)

    summarize_group(errors_df, "sentence_cluster").to_csv(
        output_dir / "error_summary_by_sentence_cluster.csv",
        index=False,
    )

    summarize_group(errors_df, "audio_cluster").to_csv(
        output_dir / "error_summary_by_audio_cluster.csv",
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
            "sentence_cluster",
            "duration_seconds",
            "dominant_error",
            "num_substitutions",
            "num_deletions",
            "num_insertions",
        ]
    ].to_csv(output_dir / "worst_samples.csv", index=False)

    save_worst_samples_by_cluster(
        errors_df=errors_df,
        output_path=output_dir / "worst_samples_by_sentence_cluster.csv",
        cluster_column="sentence_cluster",
        examples_per_cluster=examples_per_cluster,
    )

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

    print()
    print("Overall:")
    print(overall.to_string(index=False))

    print()
    print("Error summary by sentence cluster:")
    print(summarize_group(errors_df, "sentence_cluster").to_string(index=False))

    print()
    print("Representative samples saved to:")
    print(output_dir / "representative_samples_per_sentence_cluster.csv")

    print()
    print("Top terms saved to:")
    print(output_dir / "top_terms_per_sentence_cluster.csv")

    print()
    print(f"Saved sentence embedding error analysis to: {output_dir}")


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
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/error_analysis_sentence_embeddings"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument(
        "--embeddings-cache",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--force-recompute-embeddings",
        action="store_true",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--selected-k",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="13,42,77",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--min-cluster-samples",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max-balance-ratio",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--use-pca",
        action="store_true",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--examples-per-cluster",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.embeddings_cache is None:
        safe_model_name = args.model_name.replace("/", "__")
        args.embeddings_cache = args.output_dir / f"{safe_model_name}_embeddings.npy"

    metadata = pd.read_csv(args.metadata_csv)
    predictions = pd.read_csv(args.predictions_csv)

    if "audio_path" not in metadata.columns or "transcript" not in metadata.columns:
        raise ValueError("metadata CSV must contain audio_path and transcript columns.")

    required_prediction_columns = {"audio_path", "reference", "prediction"}
    missing_prediction_columns = required_prediction_columns - set(predictions.columns)

    if missing_prediction_columns:
        raise ValueError(f"Missing columns in predictions CSV: {missing_prediction_columns}")

    metadata = metadata.copy()
    metadata["normalized_transcript"] = metadata["transcript"].apply(normalize_text)

    embeddings = compute_or_load_embeddings(
        texts=metadata["normalized_transcript"].tolist(),
        model_name=args.model_name,
        cache_path=args.embeddings_cache,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute_embeddings,
    )

    clustering_embeddings = maybe_apply_pca(
        embeddings=embeddings,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        random_seed=args.random_seed,
    )

    _, suggested_k = select_k(
        metadata=metadata,
        embeddings=clustering_embeddings,
        predictions=predictions,
        output_dir=args.output_dir,
        min_k=args.min_k,
        max_k=args.max_k,
        seeds=parse_seeds(args.seeds),
        min_cluster_samples=args.min_cluster_samples,
        max_balance_ratio=args.max_balance_ratio,
        silhouette_sample_size=args.silhouette_sample_size,
    )

    selected_k = args.selected_k if args.selected_k is not None else suggested_k

    print()
    print(f"Using selected k for final sentence-cluster error analysis: {selected_k}")

    analyze_selected_k(
        metadata=metadata,
        embeddings=clustering_embeddings,
        predictions=predictions,
        output_dir=args.output_dir,
        selected_k=selected_k,
        random_seed=args.random_seed,
        examples_per_cluster=args.examples_per_cluster,
    )


if __name__ == "__main__":
    main()