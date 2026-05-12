import argparse
from pathlib import Path

import pandas as pd


def duration_bucket(seconds: float) -> str:
    if seconds < 3:
        return "short_<3s"
    if seconds < 6:
        return "medium_3_6s"
    return "long_>6s"


def safe_column(df: pd.DataFrame, column: str, default: str = "unknown") -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df))

    return (
        df[column]
        .fillna(default)
        .astype(str)
        .replace({"": default, "nan": default, "None": default})
    )


def categorical_distribution(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    grouped = (
        df.groupby([group_column, value_column])
        .agg(samples=("audio_path", "count"))
        .reset_index()
    )

    totals = (
        grouped.groupby(group_column)["samples"]
        .sum()
        .reset_index()
        .rename(columns={"samples": "total_samples"})
    )

    result = grouped.merge(totals, on=group_column)
    result["percentage"] = result["samples"] / result["total_samples"]

    return result.sort_values([group_column, "samples"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clustered-metadata-csv",
        type=Path,
        default=Path("data/processed/metadata_with_clusters_and_split.csv"),
    )
    parser.add_argument(
        "--common-voice-tsv",
        type=Path,
        default=Path("data/raw/common_voice/cv-corpus-25.0-2026-03-09/ro/validated.tsv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cluster_metadata_analysis"),
    )

    args = parser.parse_args()

    clustered = pd.read_csv(args.clustered_metadata_csv)
    cv = pd.read_csv(args.common_voice_tsv, sep="\t")

    clustered["cv_path"] = clustered["audio_path"].apply(lambda value: Path(value).name)
    cv["cv_path"] = cv["path"].astype(str)

    metadata_columns = [
        "cv_path",
        "age",
        "gender",
        "accents",
        "variant",
        "up_votes",
        "down_votes",
        "segment",
        "is_edited",
    ]

    available_metadata_columns = [
        column for column in metadata_columns
        if column in cv.columns
    ]

    merged = clustered.merge(
        cv[available_metadata_columns],
        on="cv_path",
        how="left",
    )

    merged["age"] = safe_column(merged, "age")
    merged["gender"] = safe_column(merged, "gender")
    merged["accents"] = safe_column(merged, "accents")
    merged["variant"] = safe_column(merged, "variant")
    merged["duration_bucket"] = merged["duration_seconds"].apply(duration_bucket)
    merged["word_count"] = merged["transcript"].astype(str).apply(lambda text: len(text.split()))
    merged["char_count"] = merged["transcript"].astype(str).apply(len)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cluster_profile = (
        merged.groupby("cluster")
        .agg(
            samples=("audio_path", "count"),
            duration_hours=("duration_seconds", lambda values: values.sum() / 3600),
            speakers=("speaker_id", "nunique"),
            mean_duration_seconds=("duration_seconds", "mean"),
            mean_word_count=("word_count", "mean"),
            mean_char_count=("char_count", "mean"),
        )
        .reset_index()
        .sort_values("cluster")
    )

    split_profile = (
        merged.groupby("split")
        .agg(
            samples=("audio_path", "count"),
            duration_hours=("duration_seconds", lambda values: values.sum() / 3600),
            speakers=("speaker_id", "nunique"),
            mean_duration_seconds=("duration_seconds", "mean"),
            mean_word_count=("word_count", "mean"),
            mean_char_count=("char_count", "mean"),
        )
        .reset_index()
        .sort_values("split")
    )

    cluster_profile.to_csv(args.output_dir / "cluster_profile.csv", index=False)
    split_profile.to_csv(args.output_dir / "split_profile.csv", index=False)

    for column in ["age", "gender", "accents", "variant", "duration_bucket"]:
        by_cluster = categorical_distribution(
            merged,
            group_column="cluster",
            value_column=column,
        )
        by_split = categorical_distribution(
            merged,
            group_column="split",
            value_column=column,
        )

        by_cluster.to_csv(args.output_dir / f"cluster_{column}_distribution.csv", index=False)
        by_split.to_csv(args.output_dir / f"split_{column}_distribution.csv", index=False)

    merged.to_csv(args.output_dir / "metadata_with_common_voice_fields.csv", index=False)

    print("Cluster profile:")
    print(cluster_profile.to_string(index=False))

    print()
    print("Split profile:")
    print(split_profile.to_string(index=False))

    print()
    print(f"Saved analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()