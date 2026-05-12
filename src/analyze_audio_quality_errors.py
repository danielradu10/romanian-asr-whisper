import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import jiwer
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLE_RATE = 16_000
EPS = 1e-10


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


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.std(values))


def db_from_amplitude(value: float) -> float:
    return float(20.0 * math.log10(max(value, EPS)))


def compute_silence_features(
    rms_frames: np.ndarray,
    duration_seconds: float,
    silence_top_db: float,
) -> dict[str, float]:
    if rms_frames.size == 0:
        return {
            "silence_ratio": float("nan"),
            "non_silent_duration_seconds": float("nan"),
            "leading_silence_ratio": float("nan"),
            "trailing_silence_ratio": float("nan"),
        }

    max_rms = float(np.max(rms_frames))
    threshold = max_rms * (10.0 ** (-silence_top_db / 20.0))

    silent = rms_frames < threshold
    silence_ratio = float(np.mean(silent))

    leading_silent_frames = 0
    for is_silent in silent:
        if is_silent:
            leading_silent_frames += 1
        else:
            break

    trailing_silent_frames = 0
    for is_silent in silent[::-1]:
        if is_silent:
            trailing_silent_frames += 1
        else:
            break

    total_frames = len(silent)

    return {
        "silence_ratio": silence_ratio,
        "non_silent_duration_seconds": duration_seconds * (1.0 - silence_ratio),
        "leading_silence_ratio": leading_silent_frames / total_frames,
        "trailing_silence_ratio": trailing_silent_frames / total_frames,
    }


def compute_pitch_features(
    audio: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
) -> dict[str, float]:
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=50,
            fmax=450,
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )

        voiced_f0 = f0[voiced_flag]
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

        if voiced_f0.size == 0:
            return {
                "pitch_mean_hz": float("nan"),
                "pitch_std_hz": float("nan"),
                "pitch_min_hz": float("nan"),
                "pitch_max_hz": float("nan"),
                "voiced_ratio": 0.0,
            }

        return {
            "pitch_mean_hz": float(np.mean(voiced_f0)),
            "pitch_std_hz": float(np.std(voiced_f0)),
            "pitch_min_hz": float(np.min(voiced_f0)),
            "pitch_max_hz": float(np.max(voiced_f0)),
            "voiced_ratio": float(np.mean(voiced_flag)),
        }
    except Exception:
        return {
            "pitch_mean_hz": float("nan"),
            "pitch_std_hz": float("nan"),
            "pitch_min_hz": float("nan"),
            "pitch_max_hz": float("nan"),
            "voiced_ratio": float("nan"),
        }


def compute_audio_features(
    audio_path: str,
    sample_rate: int,
    compute_pitch: bool,
    silence_top_db: float,
) -> dict[str, float]:
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    if audio.size == 0:
        raise ValueError("Empty audio.")

    duration_seconds = float(len(audio) / sr)

    frame_length = 1024
    hop_length = 256

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    peak_amplitude = float(np.max(np.abs(audio)))
    mean_abs_amplitude = float(np.mean(np.abs(audio)))
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    clipping_ratio = float(np.mean(np.abs(audio) >= 0.99))

    rms_p10 = float(np.percentile(rms, 10))
    rms_p50 = float(np.percentile(rms, 50))
    rms_p90 = float(np.percentile(rms, 90))

    snr_proxy_db = float(20.0 * np.log10((rms_p90 + EPS) / (rms_p10 + EPS)))

    dynamic_range_db = float(20.0 * np.log10((rms_p90 + EPS) / (rms_p10 + EPS)))

    silence_features = compute_silence_features(
        rms_frames=rms,
        duration_seconds=duration_seconds,
        silence_top_db=silence_top_db,
    )

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        hop_length=hop_length,
    )[0]

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr,
        hop_length=hop_length,
    )[0]

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        roll_percent=0.85,
    )[0]

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
        hop_length=hop_length,
    )

    features = {
        "actual_duration_seconds": duration_seconds,
        "peak_amplitude": peak_amplitude,
        "mean_abs_amplitude": mean_abs_amplitude,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_mean_db": db_from_amplitude(rms_mean),
        "rms_p10_db": db_from_amplitude(rms_p10),
        "rms_p50_db": db_from_amplitude(rms_p50),
        "rms_p90_db": db_from_amplitude(rms_p90),
        "clipping_ratio": clipping_ratio,
        "snr_proxy_db": snr_proxy_db,
        "dynamic_range_db": dynamic_range_db,
        "spectral_centroid_mean": safe_mean(spectral_centroid),
        "spectral_centroid_std": safe_std(spectral_centroid),
        "spectral_bandwidth_mean": safe_mean(spectral_bandwidth),
        "spectral_bandwidth_std": safe_std(spectral_bandwidth),
        "spectral_rolloff_mean": safe_mean(spectral_rolloff),
        "spectral_rolloff_std": safe_std(spectral_rolloff),
        "zero_crossing_rate_mean": safe_mean(zero_crossing_rate),
        "zero_crossing_rate_std": safe_std(zero_crossing_rate),
        **silence_features,
    }

    for index in range(mfcc.shape[0]):
        mfcc_values = mfcc[index]
        mfcc_number = index + 1
        features[f"mfcc_{mfcc_number}_mean"] = safe_mean(mfcc_values)
        features[f"mfcc_{mfcc_number}_std"] = safe_std(mfcc_values)

    if compute_pitch:
        features.update(
            compute_pitch_features(
                audio=audio,
                sample_rate=sr,
                frame_length=frame_length,
                hop_length=hop_length,
            )
        )

    return features


def create_bucket_summary(
    df: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
) -> pd.DataFrame:
    rows = []

    for column in feature_columns:
        if column not in df.columns:
            continue

        values = df[column]

        if values.dropna().nunique() < 3:
            continue

        try:
            bucket_series = pd.qcut(
                values,
                q=3,
                labels=["low", "medium", "high"],
                duplicates="drop",
            )
        except ValueError:
            continue

        temp = df.copy()
        temp[f"{column}_bucket"] = bucket_series

        grouped = (
            temp.groupby(f"{column}_bucket", observed=True)
            .agg(
                samples=("audio_path", "count"),
                mean_wer=("sample_wer", "mean"),
                mean_cer=("sample_cer", "mean"),
                mean_value=(column, "mean"),
                min_value=(column, "min"),
                max_value=(column, "max"),
            )
            .reset_index()
        )

        for _, row in grouped.iterrows():
            rows.append({
                "feature": column,
                "bucket": row[f"{column}_bucket"],
                "samples": row["samples"],
                "mean_wer": row["mean_wer"],
                "mean_cer": row["mean_cer"],
                "mean_value": row["mean_value"],
                "min_value": row["min_value"],
                "max_value": row["max_value"],
            })

    bucket_summary = pd.DataFrame(rows)
    bucket_summary.to_csv(output_path, index=False)

    return bucket_summary


def compute_correlations(
    df: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
) -> pd.DataFrame:
    rows = []

    for column in feature_columns:
        if column not in df.columns:
            continue

        valid = df[[column, "sample_wer", "sample_cer"]].dropna()

        if len(valid) < 5:
            continue

        if valid[column].nunique() < 2:
            continue

        rows.append({
            "feature": column,
            "pearson_corr_with_wer": valid[column].corr(valid["sample_wer"], method="pearson"),
            "spearman_corr_with_wer": valid[column].corr(valid["sample_wer"], method="spearman"),
            "pearson_corr_with_cer": valid[column].corr(valid["sample_cer"], method="pearson"),
            "spearman_corr_with_cer": valid[column].corr(valid["sample_cer"], method="spearman"),
            "samples": len(valid),
        })

    correlations = pd.DataFrame(rows)

    if not correlations.empty:
        correlations["abs_spearman_corr_with_wer"] = correlations[
            "spearman_corr_with_wer"
        ].abs()

        correlations = correlations.sort_values(
            "abs_spearman_corr_with_wer",
            ascending=False,
        )

    correlations.to_csv(output_path, index=False)

    return correlations


def summarize_group(
    df: pd.DataFrame,
    group_column: str,
    output_path: Path,
    min_samples: int = 1,
) -> pd.DataFrame:
    if group_column not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby(group_column)
        .agg(
            samples=("audio_path", "count"),
            duration_hours=("actual_duration_seconds", lambda values: values.sum() / 3600),
            mean_duration_seconds=("actual_duration_seconds", "mean"),
            mean_word_count=("reference_word_count", "mean"),
            mean_words_per_second=("words_per_second", "mean"),
            mean_rms_db=("rms_mean_db", "mean"),
            mean_silence_ratio=("silence_ratio", "mean"),
            mean_snr_proxy_db=("snr_proxy_db", "mean"),
            mean_wer=("sample_wer", "mean"),
            mean_cer=("sample_cer", "mean"),
            mean_substitutions=("num_substitutions", "mean"),
            mean_deletions=("num_deletions", "mean"),
            mean_insertions=("num_insertions", "mean"),
        )
        .reset_index()
    )

    summary = summary[summary["samples"] >= min_samples].copy()
    summary = summary.sort_values("mean_wer", ascending=False)

    summary.to_csv(output_path, index=False)

    return summary


def compare_high_error_samples(
    df: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
) -> pd.DataFrame:
    threshold = df["sample_wer"].quantile(0.80)

    low_or_medium_error = df[df["sample_wer"] < threshold]
    high_error = df[df["sample_wer"] >= threshold]

    rows = []

    for column in feature_columns:
        if column not in df.columns:
            continue

        rows.append({
            "feature": column,
            "high_error_mean": high_error[column].mean(),
            "low_or_medium_error_mean": low_or_medium_error[column].mean(),
            "difference_high_minus_rest": high_error[column].mean()
            - low_or_medium_error[column].mean(),
            "high_error_samples": len(high_error),
            "low_or_medium_error_samples": len(low_or_medium_error),
            "wer_threshold_80_percentile": threshold,
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_path, index=False)

    return comparison


def save_basic_plots(
    output_dir: Path,
    correlations: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not correlations.empty:
        top_corr = correlations.head(15).copy()

        ax = top_corr.set_index("feature")["spearman_corr_with_wer"].plot(
            kind="bar",
            figsize=(10, 5),
        )

        ax.set_title("Audio Feature Correlation with WER")
        ax.set_xlabel("Audio feature")
        ax.set_ylabel("Spearman correlation with WER")
        ax.grid(axis="y", alpha=0.3)

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(figures_dir / "audio_feature_correlations_with_wer.png", dpi=300)
        plt.close()

    important_features = [
        "actual_duration_seconds",
        "words_per_second",
        "rms_mean_db",
        "silence_ratio",
        "snr_proxy_db",
        "spectral_centroid_mean",
        "zero_crossing_rate_mean",
    ]

    for feature in important_features:
        subset = bucket_summary[bucket_summary["feature"] == feature]

        if subset.empty:
            continue

        ax = subset.set_index("bucket")[["mean_wer", "mean_cer"]].plot(
            kind="bar",
            figsize=(7, 4),
        )

        ax.set_title(f"Error by {feature} bucket")
        ax.set_xlabel(feature)
        ax.set_ylabel("Error rate")
        ax.grid(axis="y", alpha=0.3)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3)

        plt.tight_layout()
        plt.savefig(figures_dir / f"error_by_{feature}_bucket.png", dpi=300)
        plt.close()

    if "audio_cluster" in df.columns:
        audio_cluster_summary = (
            df.groupby("audio_cluster")
            .agg(
                mean_wer=("sample_wer", "mean"),
                mean_cer=("sample_cer", "mean"),
                samples=("audio_path", "count"),
            )
            .reset_index()
        )

        ax = audio_cluster_summary.set_index("audio_cluster")[["mean_wer", "mean_cer"]].plot(
            kind="bar",
            figsize=(7, 4),
        )

        ax.set_title("Error by Audio Cluster")
        ax.set_xlabel("Audio cluster")
        ax.set_ylabel("Error rate")
        ax.grid(axis="y", alpha=0.3)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3)

        plt.tight_layout()
        plt.savefig(figures_dir / "error_by_audio_cluster.png", dpi=300)
        plt.close()

    if "sentence_cluster" in df.columns:
        sentence_cluster_summary = (
            df.groupby("sentence_cluster")
            .agg(
                mean_wer=("sample_wer", "mean"),
                mean_cer=("sample_cer", "mean"),
                samples=("audio_path", "count"),
            )
            .reset_index()
        )

        ax = sentence_cluster_summary.set_index("sentence_cluster")[["mean_wer", "mean_cer"]].plot(
            kind="bar",
            figsize=(7, 4),
        )

        ax.set_title("Error by Sentence Cluster")
        ax.set_xlabel("Sentence cluster")
        ax.set_ylabel("Error rate")
        ax.grid(axis="y", alpha=0.3)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3)

        plt.tight_layout()
        plt.savefig(figures_dir / "error_by_sentence_cluster.png", dpi=300)
        plt.close()


def detect_optional_metadata_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "speaker_id",
        "client_id",
        "gender",
        "age",
        "accents",
        "accent",
        "locale",
        "audio_cluster",
        "sentence_cluster",
        "duration_bucket",
    ]

    return [column for column in candidates if column in df.columns]


def analyze_audio_quality(
    predictions_csv: Path,
    metadata_csv: Path,
    sentence_clusters_csv: Path | None,
    output_dir: Path,
    sample_rate: int,
    compute_pitch: bool,
    silence_top_db: float,
    max_samples: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(predictions_csv)
    metadata = pd.read_csv(metadata_csv)

    required_prediction_columns = {"audio_path", "reference", "prediction"}
    missing_prediction_columns = required_prediction_columns - set(predictions.columns)

    if missing_prediction_columns:
        raise ValueError(f"Missing columns in predictions CSV: {missing_prediction_columns}")

    if "audio_path" not in metadata.columns:
        raise ValueError("metadata CSV must contain audio_path column.")

    metadata = metadata.copy()

    if "cluster" in metadata.columns:
        metadata = metadata.rename(columns={"cluster": "audio_cluster"})

    merge_columns = [
        "audio_path",
        "audio_cluster",
        "duration_seconds",
        "speaker_id",
        "client_id",
        "gender",
        "age",
        "accents",
        "accent",
        "locale",
        "split",
        "transcript",
    ]

    available_merge_columns = [
        column for column in merge_columns
        if column in metadata.columns
    ]

    df = predictions.merge(
        metadata[available_merge_columns],
        on="audio_path",
        how="left",
    )

    if sentence_clusters_csv is not None and sentence_clusters_csv.exists():
        sentence_clusters = pd.read_csv(sentence_clusters_csv)

        if {"audio_path", "sentence_cluster"}.issubset(sentence_clusters.columns):
            df = df.merge(
                sentence_clusters[["audio_path", "sentence_cluster"]],
                on="audio_path",
                how="left",
            )

    if max_samples is not None:
        df = df.head(max_samples).copy()

    rows = []
    skipped_rows = []

    print(f"Analyzing audio files: {len(df)}")

    for index, row in df.iterrows():
        audio_path = row["audio_path"]

        try:
            reference = normalize_text(row["reference"])
            prediction = normalize_text(row["prediction"])

            edit_info = word_edit_analysis(reference, prediction)

            sample_wer = edit_info["num_word_edits"] / edit_info["num_reference_words"]
            sample_cer = jiwer.cer(reference, prediction)

            audio_features = compute_audio_features(
                audio_path=audio_path,
                sample_rate=sample_rate,
                compute_pitch=compute_pitch,
                silence_top_db=silence_top_db,
            )

            actual_duration = audio_features["actual_duration_seconds"]
            reference_word_count = len(reference.split())
            prediction_word_count = len(prediction.split())

            words_per_second = reference_word_count / max(actual_duration, EPS)
            chars_per_second = len(reference) / max(actual_duration, EPS)

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
                "reference": reference,
                "prediction": prediction,
                "sample_wer": sample_wer,
                "sample_cer": sample_cer,
                "reference_word_count": reference_word_count,
                "prediction_word_count": prediction_word_count,
                "reference_char_count": len(reference),
                "prediction_char_count": len(prediction),
                "words_per_second": words_per_second,
                "chars_per_second": chars_per_second,
                "dominant_error": dominant_error,
                "num_substitutions": edit_info["num_substitutions"],
                "num_deletions": edit_info["num_deletions"],
                "num_insertions": edit_info["num_insertions"],
                "num_word_edits": edit_info["num_word_edits"],
                **audio_features,
            })

        except Exception as error:
            skipped_rows.append({
                "audio_path": audio_path,
                "error": str(error),
            })

        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1}/{len(df)}")

    analysis_df = pd.DataFrame(rows)

    if analysis_df.empty:
        raise RuntimeError("No audio features were produced.")

    analysis_df["duration_bucket"] = analysis_df["actual_duration_seconds"].apply(duration_bucket)

    per_sample_path = output_dir / "per_sample_audio_quality_errors.csv"
    analysis_df.to_csv(per_sample_path, index=False)

    pd.DataFrame(skipped_rows).to_csv(output_dir / "skipped_audio_files.csv", index=False)

    feature_columns = [
        "actual_duration_seconds",
        "words_per_second",
        "chars_per_second",
        "peak_amplitude",
        "mean_abs_amplitude",
        "rms_mean",
        "rms_std",
        "rms_mean_db",
        "rms_p10_db",
        "rms_p50_db",
        "rms_p90_db",
        "clipping_ratio",
        "snr_proxy_db",
        "dynamic_range_db",
        "silence_ratio",
        "non_silent_duration_seconds",
        "leading_silence_ratio",
        "trailing_silence_ratio",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_std",
    ]

    if compute_pitch:
        feature_columns.extend([
            "pitch_mean_hz",
            "pitch_std_hz",
            "pitch_min_hz",
            "pitch_max_hz",
            "voiced_ratio",
        ])

    feature_columns.extend([
        column
        for column in analysis_df.columns
        if column.startswith("mfcc_")
    ])

    correlations = compute_correlations(
        df=analysis_df,
        feature_columns=feature_columns,
        output_path=output_dir / "audio_feature_correlations.csv",
    )

    bucket_summary = create_bucket_summary(
        df=analysis_df,
        feature_columns=feature_columns,
        output_path=output_dir / "error_by_audio_feature_buckets.csv",
    )

    compare_high_error_samples(
        df=analysis_df,
        feature_columns=feature_columns,
        output_path=output_dir / "high_error_vs_rest_audio_features.csv",
    )

    optional_group_columns = detect_optional_metadata_columns(analysis_df)

    for group_column in optional_group_columns:
        min_samples = 1

        if group_column in {"speaker_id", "client_id"}:
            min_samples = 2

        summarize_group(
            df=analysis_df,
            group_column=group_column,
            output_path=output_dir / f"error_by_{group_column}.csv",
            min_samples=min_samples,
        )

    if "audio_cluster" in analysis_df.columns and "sentence_cluster" in analysis_df.columns:
        combined = (
            analysis_df.groupby(["audio_cluster", "sentence_cluster"])
            .agg(
                samples=("audio_path", "count"),
                mean_wer=("sample_wer", "mean"),
                mean_cer=("sample_cer", "mean"),
                mean_duration_seconds=("actual_duration_seconds", "mean"),
                mean_words_per_second=("words_per_second", "mean"),
                mean_rms_db=("rms_mean_db", "mean"),
                mean_silence_ratio=("silence_ratio", "mean"),
                mean_snr_proxy_db=("snr_proxy_db", "mean"),
            )
            .reset_index()
            .sort_values("mean_wer", ascending=False)
        )

        combined.to_csv(output_dir / "error_by_audio_and_sentence_cluster.csv", index=False)

    worst_samples = (
        analysis_df.sort_values(["sample_wer", "sample_cer"], ascending=False)
        .head(50)
    )

    worst_columns = [
        "audio_path",
        "reference",
        "prediction",
        "sample_wer",
        "sample_cer",
        "audio_cluster",
        "sentence_cluster",
        "actual_duration_seconds",
        "words_per_second",
        "rms_mean_db",
        "silence_ratio",
        "snr_proxy_db",
        "spectral_centroid_mean",
        "zero_crossing_rate_mean",
        "dominant_error",
        "num_substitutions",
        "num_deletions",
        "num_insertions",
    ]

    available_worst_columns = [
        column for column in worst_columns
        if column in worst_samples.columns
    ]

    worst_samples[available_worst_columns].to_csv(
        output_dir / "worst_samples_with_audio_features.csv",
        index=False,
    )

    overall = pd.DataFrame([
        {
            "predictions_file": str(predictions_csv),
            "metadata_file": str(metadata_csv),
            "samples": len(analysis_df),
            "skipped_samples": len(skipped_rows),
            "overall_wer": jiwer.wer(
                analysis_df["reference"].tolist(),
                analysis_df["prediction"].tolist(),
            ),
            "overall_cer": jiwer.cer(
                analysis_df["reference"].tolist(),
                analysis_df["prediction"].tolist(),
            ),
            "mean_sample_wer": analysis_df["sample_wer"].mean(),
            "mean_sample_cer": analysis_df["sample_cer"].mean(),
            "mean_duration_seconds": analysis_df["actual_duration_seconds"].mean(),
            "mean_words_per_second": analysis_df["words_per_second"].mean(),
            "mean_rms_db": analysis_df["rms_mean_db"].mean(),
            "mean_silence_ratio": analysis_df["silence_ratio"].mean(),
            "mean_snr_proxy_db": analysis_df["snr_proxy_db"].mean(),
            "compute_pitch": compute_pitch,
        }
    ])

    overall.to_csv(output_dir / "overall_audio_quality_metrics.csv", index=False)

    save_basic_plots(
        output_dir=output_dir,
        correlations=correlations,
        bucket_summary=bucket_summary,
        df=analysis_df,
    )

    print()
    print("Overall audio quality analysis:")
    print(overall.to_string(index=False))

    print()
    print("Top audio feature correlations with WER:")
    if correlations.empty:
        print("No correlations computed.")
    else:
        print(correlations.head(15).to_string(index=False))

    print()
    if "audio_cluster" in analysis_df.columns:
        print("Error by audio cluster:")
        print(
            summarize_group(
                df=analysis_df,
                group_column="audio_cluster",
                output_path=output_dir / "error_by_audio_cluster.csv",
            ).to_string(index=False)
        )

    print()
    print(f"Saved per-sample audio features to: {per_sample_path}")
    print(f"Saved audio quality analysis to: {output_dir}")


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
        "--sentence-clusters-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/audio_quality_analysis"),
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
    )
    parser.add_argument(
        "--compute-pitch",
        action="store_true",
    )
    parser.add_argument(
        "--silence-top-db",
        type=float,
        default=40.0,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
    )

    args = parser.parse_args()

    max_samples = None if args.max_samples < 0 else args.max_samples

    analyze_audio_quality(
        predictions_csv=args.predictions_csv,
        metadata_csv=args.metadata_csv,
        sentence_clusters_csv=args.sentence_clusters_csv,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        compute_pitch=args.compute_pitch,
        silence_top_db=args.silence_top_db,
        max_samples=max_samples,
    )


if __name__ == "__main__":
    main()