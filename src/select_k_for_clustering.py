import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from tqdm import tqdm


def compute_cluster_stats(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> dict:
    temp = df.copy()
    temp["cluster"] = labels

    grouped = (
        temp.groupby("cluster")
        .agg(
            samples=("audio_path", "count"),
            duration_seconds=("duration_seconds", "sum"),
            speakers=("speaker_id", "nunique"),
        )
        .reset_index()
    )

    durations = grouped["duration_seconds"].to_numpy()
    samples = grouped["samples"].to_numpy()
    speakers = grouped["speakers"].to_numpy()

    return {
        "min_cluster_samples": int(samples.min()),
        "max_cluster_samples": int(samples.max()),
        "mean_cluster_samples": float(samples.mean()),
        "min_cluster_duration_hours": float(durations.min() / 3600),
        "max_cluster_duration_hours": float(durations.max() / 3600),
        "mean_cluster_duration_hours": float(durations.mean() / 3600),
        "min_cluster_speakers": int(speakers.min()),
        "max_cluster_speakers": int(speakers.max()),
        "duration_balance_ratio": float(durations.max() / max(durations.min(), 1e-12)),
    }


def evaluate_k(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    k: int,
    seeds: list[int],
) -> list[dict]:
    rows = []
    all_labels = []

    for seed in seeds:
        kmeans = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=10,
        )

        labels = kmeans.fit_predict(embeddings)
        all_labels.append(labels)

        silhouette = silhouette_score(embeddings, labels)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        calinski_harabasz = calinski_harabasz_score(embeddings, labels)

        cluster_stats = compute_cluster_stats(df, labels)

        row = {
            "k": k,
            "seed": seed,
            "silhouette": float(silhouette),
            "davies_bouldin": float(davies_bouldin),
            "calinski_harabasz": float(calinski_harabasz),
            **cluster_stats,
        }

        rows.append(row)

    stability_scores = []

    if len(all_labels) > 1:
        reference_labels = all_labels[0]

        for labels in all_labels[1:]:
            stability_scores.append(adjusted_rand_score(reference_labels, labels))

    stability = float(np.mean(stability_scores)) if stability_scores else 1.0

    for row in rows:
        row["stability_ari"] = stability

    return rows


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "silhouette": ["mean", "std"],
        "davies_bouldin": ["mean", "std"],
        "calinski_harabasz": ["mean", "std"],
        "stability_ari": "mean",
        "min_cluster_samples": "mean",
        "min_cluster_duration_hours": "mean",
        "min_cluster_speakers": "mean",
        "duration_balance_ratio": "mean",
    }

    aggregated = results.groupby("k").agg(aggregations)

    aggregated.columns = [
        "_".join(column).strip("_")
        for column in aggregated.columns.to_flat_index()
    ]

    aggregated = aggregated.reset_index()

    return aggregated


def mark_practical_candidates(
    aggregated: pd.DataFrame,
    min_cluster_duration_hours: float,
    min_cluster_samples: int,
    min_cluster_speakers: int,
) -> pd.DataFrame:
    df = aggregated.copy()

    df["passes_size_constraints"] = (
        (df["min_cluster_duration_hours_mean"] >= min_cluster_duration_hours)
        & (df["min_cluster_samples_mean"] >= min_cluster_samples)
        & (df["min_cluster_speakers_mean"] >= min_cluster_speakers)
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=Path("data/processed/common_voice_embeddings.npy"),
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/processed/common_voice_metadata_with_embeddings.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 999],
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
        "--min-cluster-duration-hours",
        type=float,
        default=0.15,
        help="0.15 hours = 9 minutes",
    )
    parser.add_argument(
        "--min-cluster-samples",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--min-cluster-speakers",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    embeddings = np.load(args.embeddings_path)
    df = pd.read_csv(args.metadata_csv)

    if len(df) != len(embeddings):
        raise ValueError(
            f"Metadata rows and embeddings count do not match: "
            f"{len(df)} metadata rows vs {len(embeddings)} embeddings"
        )

    clustering_input = embeddings

    if args.use_pca:
        effective_components = min(
            args.pca_components,
            embeddings.shape[1],
            len(embeddings),
        )

        print(f"Applying PCA: {embeddings.shape[1]} -> {effective_components}")

        pca = PCA(n_components=effective_components, random_state=42)
        clustering_input = pca.fit_transform(embeddings)

        explained = pca.explained_variance_ratio_.sum()
        print(f"PCA explained variance ratio: {explained:.4f}")

    all_rows = []

    for k in tqdm(range(args.min_k, args.max_k + 1), desc="Evaluating k"):
        rows = evaluate_k(
            embeddings=clustering_input,
            df=df,
            k=k,
            seeds=args.seeds,
        )
        all_rows.extend(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    detailed = pd.DataFrame(all_rows)
    aggregated = aggregate_results(detailed)

    aggregated = mark_practical_candidates(
        aggregated=aggregated,
        min_cluster_duration_hours=args.min_cluster_duration_hours,
        min_cluster_samples=args.min_cluster_samples,
        min_cluster_speakers=args.min_cluster_speakers,
    )

    detailed_path = args.output_dir / "k_selection_detailed.csv"
    aggregated_path = args.output_dir / "k_selection_summary.csv"

    detailed.to_csv(detailed_path, index=False)
    aggregated.to_csv(aggregated_path, index=False)

    print()
    print(f"Saved detailed results to: {detailed_path}")
    print(f"Saved summary results to: {aggregated_path}")

    print()
    print("Practical candidates:")
    practical = aggregated[aggregated["passes_size_constraints"]].copy()

    if practical.empty:
        print("No k passed all practical constraints.")
        print("Consider lowering constraints or using a smaller max_k.")
    else:
        display_columns = [
            "k",
            "silhouette_mean",
            "davies_bouldin_mean",
            "calinski_harabasz_mean",
            "stability_ari_mean",
            "min_cluster_duration_hours_mean",
            "min_cluster_samples_mean",
            "min_cluster_speakers_mean",
            "duration_balance_ratio_mean",
        ]

        practical = practical.sort_values(
            by=[
                "silhouette_mean",
                "stability_ari_mean",
                "duration_balance_ratio_mean",
            ],
            ascending=[False, False, True],
        )

        print(practical[display_columns].to_string(index=False))

        print()
        print("Suggested k based on this heuristic:")
        print(int(practical.iloc[0]["k"]))


if __name__ == "__main__":
    main()