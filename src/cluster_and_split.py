import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def compute_assignment_error(
    current_cluster_durations: dict[str, dict[int, float]],
    target_cluster_durations: dict[str, dict[int, float]],
) -> float:
    error = 0.0

    for split_name in target_cluster_durations:
        for cluster_id, target_duration in target_cluster_durations[split_name].items():
            current_duration = current_cluster_durations[split_name].get(cluster_id, 0.0)
            error += (target_duration - current_duration) ** 2

    return error


def assign_splits_group_aware(
    df: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> pd.DataFrame:
    """
    Assigns train/validation/test splits while avoiding speaker leakage.

    The same speaker_id is assigned to exactly one split globally.
    The assignment tries to preserve cluster duration distribution across splits.
    """
    df = df.copy()
    df["split"] = ""

    split_ratios = {
        "train": train_ratio,
        "validation": validation_ratio,
        "test": test_ratio,
    }

    cluster_total_durations = (
        df.groupby("cluster")["duration_seconds"]
        .sum()
        .to_dict()
    )

    target_cluster_durations = {
        split_name: {
            cluster_id: total_duration * split_ratio
            for cluster_id, total_duration in cluster_total_durations.items()
        }
        for split_name, split_ratio in split_ratios.items()
    }

    current_cluster_durations = {
        "train": {cluster_id: 0.0 for cluster_id in cluster_total_durations},
        "validation": {cluster_id: 0.0 for cluster_id in cluster_total_durations},
        "test": {cluster_id: 0.0 for cluster_id in cluster_total_durations},
    }

    speaker_cluster_durations = (
        df.groupby(["speaker_id", "cluster"])["duration_seconds"]
        .sum()
        .reset_index()
    )

    speaker_total_durations = (
        df.groupby("speaker_id")["duration_seconds"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    speaker_total_durations = speaker_total_durations.sample(
        frac=1.0,
        random_state=random_seed,
    ).sort_values(
        by="duration_seconds",
        ascending=False,
    ).reset_index(drop=True)

    speaker_to_cluster_durations: dict[str, dict[int, float]] = {}

    for _, row in speaker_cluster_durations.iterrows():
        speaker_id = row["speaker_id"]
        cluster_id = int(row["cluster"])
        duration = float(row["duration_seconds"])

        speaker_to_cluster_durations.setdefault(speaker_id, {})
        speaker_to_cluster_durations[speaker_id][cluster_id] = duration

    speaker_to_split = {}

    for _, row in speaker_total_durations.iterrows():
        speaker_id = row["speaker_id"]
        speaker_durations = speaker_to_cluster_durations[speaker_id]

        best_split = None
        best_error = None

        for candidate_split in ["train", "validation", "test"]:
            simulated_current = {
                split_name: cluster_map.copy()
                for split_name, cluster_map in current_cluster_durations.items()
            }

            for cluster_id, duration in speaker_durations.items():
                simulated_current[candidate_split][cluster_id] += duration

            error = compute_assignment_error(
                current_cluster_durations=simulated_current,
                target_cluster_durations=target_cluster_durations,
            )

            if best_error is None or error < best_error:
                best_error = error
                best_split = candidate_split

        speaker_to_split[speaker_id] = best_split

        for cluster_id, duration in speaker_durations.items():
            current_cluster_durations[best_split][cluster_id] += duration

    for speaker_id, selected_split in speaker_to_split.items():
        df.loc[df["speaker_id"] == speaker_id, "split"] = selected_split

    if (df["split"] == "").any():
        raise RuntimeError("Some rows were not assigned to any split.")

    leakage_check = df.groupby("speaker_id")["split"].nunique()
    leaking_speakers = leakage_check[leakage_check > 1]

    if len(leaking_speakers) > 0:
        raise RuntimeError("Speaker leakage detected after split assignment.")

    return df



def save_split_files(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    columns_to_save = [
        "audio_path",
        "transcript",
        "source",
        "duration_seconds",
        "difficulty_rating",
        "speaker_id",
        "recording_id",
        "cluster",
        "split",
        "observation",
    ]

    available_columns = [column for column in columns_to_save if column in df.columns]

    for split in ["train", "validation", "test"]:
        split_df = df[df["split"] == split].copy()
        output_path = output_dir / f"{split}.csv"
        split_df[available_columns].to_csv(output_path, index=False)

        print(
            f"{split}: {len(split_df)} samples, "
            f"{split_df['duration_seconds'].sum() / 3600:.2f} hours -> {output_path}"
        )


def save_reports(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_distribution = (
        df.groupby(["cluster", "split"])
        .agg(
            samples=("audio_path", "count"),
            duration_hours=("duration_seconds", lambda values: values.sum() / 3600),
            speakers=("speaker_id", "nunique"),
        )
        .reset_index()
        .sort_values(["cluster", "split"])
    )

    split_distribution = (
        df.groupby("split")
        .agg(
            samples=("audio_path", "count"),
            duration_hours=("duration_seconds", lambda values: values.sum() / 3600),
            speakers=("speaker_id", "nunique"),
        )
        .reset_index()
        .sort_values("split")
    )

    cluster_distribution.to_csv(output_dir / "cluster_distribution.csv", index=False)
    split_distribution.to_csv(output_dir / "split_distribution.csv", index=False)

    print()
    print("Split distribution:")
    print(split_distribution)

    print()
    print(f"Saved cluster report to: {output_dir / 'cluster_distribution.csv'}")
    print(f"Saved split report to: {output_dir / 'split_distribution.csv'}")


def cluster_and_split(
    embeddings_path: Path,
    metadata_csv: Path,
    output_data_dir: Path,
    output_results_dir: Path,
    n_clusters: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
    use_pca: bool,
    pca_components: int,
) -> None:
    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + validation_ratio + test_ratio must be 1.0")

    embeddings = np.load(embeddings_path)
    df = pd.read_csv(metadata_csv)

    if len(df) != len(embeddings):
        raise ValueError(
            f"Metadata rows and embeddings count do not match: "
            f"{len(df)} metadata rows vs {len(embeddings)} embeddings"
        )

    clustering_input = embeddings

    if use_pca:
        effective_components = min(pca_components, embeddings.shape[1], len(embeddings))
        print(f"Applying PCA: {embeddings.shape[1]} -> {effective_components}")

        pca = PCA(n_components=effective_components, random_state=random_seed)
        clustering_input = pca.fit_transform(embeddings)

        explained = pca.explained_variance_ratio_.sum()
        print(f"PCA explained variance ratio: {explained:.4f}")

    print(f"Running KMeans with n_clusters={n_clusters}")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_seed,
        n_init=10,
    )

    clusters = kmeans.fit_predict(clustering_input)
    df["cluster"] = clusters

    if n_clusters > 1 and len(df) > n_clusters:
        score = silhouette_score(clustering_input, clusters)
        print(f"Silhouette score: {score:.4f}")

    df = assign_splits_group_aware(
        df=df,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_results_dir.mkdir(parents=True, exist_ok=True)

    full_output = output_data_dir / "metadata_with_clusters_and_split.csv"
    df.to_csv(full_output, index=False)

    print(f"Saved full metadata with clusters and split to: {full_output}")
    print()

    save_split_files(df, output_data_dir)
    save_reports(df, output_results_dir)

    leakage_check = df.groupby("speaker_id")["split"].nunique()
    leaking_speakers = leakage_check[leakage_check > 1]

    print()
    if len(leaking_speakers) == 0:
        print("Leakage check passed: no speaker_id appears in multiple splits.")
    else:
        print("WARNING: speaker leakage detected:")
        print(leaking_speakers)


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
        "--output-data-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--output-results-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
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

    args = parser.parse_args()

    cluster_and_split(
        embeddings_path=args.embeddings_path,
        metadata_csv=args.metadata_csv,
        output_data_dir=args.output_data_dir,
        output_results_dir=args.output_results_dir,
        n_clusters=args.n_clusters,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
    )


if __name__ == "__main__":
    main()