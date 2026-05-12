import argparse
from pathlib import Path
from typing import Optional

import librosa
import pandas as pd
from tqdm import tqdm


COMMON_VOICE_DEFAULT_TSV_FILES = [
    "validated.tsv",
    "train.tsv",
    "dev.tsv",
    "test.tsv",
]


def normalize_transcript(text: str) -> str:
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

    punctuation_to_remove = [
        ".", ",", "!", "?", ":", ";", '"', "'", "„", "”", "(", ")", "[", "]"
    ]

    for symbol in punctuation_to_remove:
        text = text.replace(symbol, "")

    text = " ".join(text.split())
    return text


def get_audio_duration_seconds(audio_path: Path) -> float:
    try:
        return round(float(librosa.get_duration(path=str(audio_path))), 3)
    except Exception:
        return -1.0


def find_tsv_file(common_voice_dir: Path, preferred_tsv: Optional[str]) -> Path:
    if preferred_tsv:
        candidate = common_voice_dir / preferred_tsv
        if not candidate.exists():
            raise FileNotFoundError(f"TSV file not found: {candidate}")
        return candidate

    for filename in COMMON_VOICE_DEFAULT_TSV_FILES:
        candidate = common_voice_dir / filename
        if candidate.exists():
            return candidate

    found = list(common_voice_dir.glob("*.tsv"))
    if found:
        return found[0]

    raise FileNotFoundError(
        f"No Common Voice TSV file found in {common_voice_dir}. "
        f"Expected one of: {COMMON_VOICE_DEFAULT_TSV_FILES}"
    )


def build_common_voice_metadata(
    common_voice_dir: Path,
    output_csv: Path,
    tsv_file: Optional[str],
    max_hours: Optional[float],
    min_duration_seconds: float,
    max_duration_seconds: float,
) -> None:
    selected_tsv = find_tsv_file(common_voice_dir, tsv_file)
    clips_dir = common_voice_dir / "clips"

    if not clips_dir.exists():
        raise FileNotFoundError(
            f"Could not find clips directory: {clips_dir}. "
            "Common Voice usually stores audio files inside a clips/ folder."
        )

    df = pd.read_csv(selected_tsv, sep="\t")

    transcript_column = "sentence" if "sentence" in df.columns else "text"

    required_columns = {"path", transcript_column}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {selected_tsv}: {missing_columns}"
        )

    rows = []
    total_duration = 0.0
    max_duration_total = None if max_hours is None else max_hours * 3600.0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Common Voice"):
        relative_audio_path = str(row["path"])
        transcript = normalize_transcript(row[transcript_column])

        if not transcript:
            continue

        audio_path = clips_dir / relative_audio_path

        if not audio_path.exists():
            continue

        duration = get_audio_duration_seconds(audio_path)

        if duration <= 0:
            continue

        if duration < min_duration_seconds or duration > max_duration_seconds:
            continue

        if max_duration_total is not None and total_duration >= max_duration_total:
            break

        client_id = str(row["client_id"]) if "client_id" in df.columns else "unknown_speaker"

        rows.append({
            "audio_path": str(audio_path),
            "transcript": transcript,
            "source": "common_voice",
            "duration_seconds": duration,
            "difficulty_rating": 1,
            "speaker_id": f"cv_{client_id}",
            "recording_id": f"cv_{relative_audio_path}",
            "observation": "",
        })

        total_duration += duration

    if not rows:
        raise RuntimeError("No valid Common Voice samples were found.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_csv, index=False)

    print(f"Input TSV: {selected_tsv}")
    print(f"Saved metadata to: {output_csv}")
    print(f"Samples: {len(output_df)}")
    print(f"Total duration: {total_duration / 3600:.2f} hours")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--common-voice-dir",
        type=Path,
        default=Path("data/raw/common_voice"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/common_voice_metadata.csv"),
    )
    parser.add_argument(
        "--tsv-file",
        type=str,
        default=None,
        help="Example: validated.tsv, train.tsv, dev.tsv, test.tsv",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=5.0,
        help="Maximum number of hours to keep. Use -1 for all available data.",
    )
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=30.0,
    )

    args = parser.parse_args()

    max_hours = None if args.max_hours < 0 else args.max_hours

    build_common_voice_metadata(
        common_voice_dir=args.common_voice_dir,
        output_csv=args.output_csv,
        tsv_file=args.tsv_file,
        max_hours=max_hours,
        min_duration_seconds=args.min_duration_seconds,
        max_duration_seconds=args.max_duration_seconds,
    )


if __name__ == "__main__":
    main()