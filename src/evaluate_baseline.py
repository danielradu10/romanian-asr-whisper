import argparse
import re
from pathlib import Path
from typing import Optional

import evaluate
import librosa
import pandas as pd
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


SAMPLE_RATE = 16_000


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


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_processor(
    model_name: str,
    processor_name: Optional[str],
    base_processor_name: str,
    language: str,
    task: str,
) -> WhisperProcessor:
    candidates = []

    if processor_name:
        candidates.append(processor_name)

    candidates.append(model_name)
    candidates.append(base_processor_name)

    last_error = None

    for candidate in candidates:
        try:
            print(f"Loading processor: {candidate}")
            return WhisperProcessor.from_pretrained(
                candidate,
                language=language,
                task=task,
            )
        except Exception as exc:
            last_error = exc
            print(f"Could not load processor from {candidate}: {exc}")

    raise RuntimeError(f"Could not load any processor. Last error: {last_error}")


def configure_generation(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    language: str,
    task: str,
) -> None:
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task=task,
    )

    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.suppress_tokens = []

    # Keep generation params out of model.config for newer Transformers versions.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = None


def transcribe_audio(
    audio_path: str,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str,
    generation_max_length: int,
    num_beams: int,
) -> str:
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    inputs = processor.feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_features = inputs.input_features.to(device=device, dtype=model.dtype)

    generation_kwargs = {
        "input_features": input_features,
        "max_length": generation_max_length,
        "num_beams": num_beams,
    }

    if hasattr(inputs, "attention_mask"):
        generation_kwargs["attention_mask"] = inputs.attention_mask.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**generation_kwargs)

    prediction = processor.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return normalize_text(prediction)


def evaluate_model(
    model_name: str,
    processor_name: Optional[str],
    base_processor_name: str,
    test_csv: Path,
    output_csv: Path,
    metrics_csv: Path,
    max_samples: Optional[int],
    language: str,
    task: str,
    generation_max_length: int,
    num_beams: int,
) -> None:
    device = get_device()
    print(f"Using device: {device}")
    print(f"Evaluating model: {model_name}")
    print(f"Test CSV: {test_csv}")

    df = pd.read_csv(test_csv)

    required_columns = {"audio_path", "transcript"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns in {test_csv}: {missing_columns}")

    if max_samples is not None:
        df = df.head(max_samples).copy()

    print(f"Samples: {len(df)}")

    processor = load_processor(
        model_name=model_name,
        processor_name=processor_name,
        base_processor_name=base_processor_name,
        language=language,
        task=task,
    )

    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {model_name}")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )

    model.to(device)
    model.eval()

    configure_generation(
        model=model,
        processor=processor,
        language=language,
        task=task,
    )

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    predictions = []
    references = []
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating ASR"):
        audio_path = str(row["audio_path"])
        reference = normalize_text(row["transcript"])

        try:
            prediction = transcribe_audio(
                audio_path=audio_path,
                model=model,
                processor=processor,
                device=device,
                generation_max_length=generation_max_length,
                num_beams=num_beams,
            )

            predictions.append(prediction)
            references.append(reference)

            rows.append({
                "audio_path": audio_path,
                "reference": reference,
                "prediction": prediction,
                "source": row.get("source", ""),
                "duration_seconds": row.get("duration_seconds", ""),
                "speaker_id": row.get("speaker_id", ""),
                "cluster": row.get("cluster", ""),
                "split": row.get("split", ""),
            })

        except Exception as exc:
            print(f"Skipping {audio_path}: {exc}")

    if not predictions:
        raise RuntimeError("No predictions were produced.")

    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(rows)
    predictions_df.to_csv(output_csv, index=False)

    metrics_df = pd.DataFrame([
        {
            "model": model_name,
            "processor": processor_name or model_name,
            "test_csv": str(test_csv),
            "samples": len(predictions_df),
            "wer": wer,
            "cer": cer,
            "num_beams": num_beams,
            "generation_max_length": generation_max_length,
        }
    ])

    metrics_df.to_csv(metrics_csv, index=False)

    print()
    print("Evaluation completed.")
    print(f"WER: {wer:.4f}")
    print(f"CER: {cer:.4f}")
    print(f"Saved predictions to: {output_csv}")
    print(f"Saved metrics to: {metrics_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/whisper-small",
    )
    parser.add_argument(
        "--processor-name",
        type=str,
        default=None,
        help="Optional processor path/name. Useful for evaluating checkpoint folders.",
    )
    parser.add_argument(
        "--base-processor-name",
        type=str,
        default="openai/whisper-small",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/processed/test.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/baseline_predictions.csv"),
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("results/baseline_metrics.csv"),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--language",
        type=str,
        default="romanian",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
    )
    parser.add_argument(
        "--generation-max-length",
        type=int,
        default=225,
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    max_samples = None if args.max_samples < 0 else args.max_samples

    evaluate_model(
        model_name=args.model_name,
        processor_name=args.processor_name,
        base_processor_name=args.base_processor_name,
        test_csv=args.test_csv,
        output_csv=args.output_csv,
        metrics_csv=args.metrics_csv,
        max_samples=max_samples,
        language=args.language,
        task=args.task,
        generation_max_length=args.generation_max_length,
        num_beams=args.num_beams,
    )


if __name__ == "__main__":
    main()