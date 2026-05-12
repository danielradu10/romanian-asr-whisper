import argparse
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate
import librosa
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


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


def load_split(csv_path: Path, max_samples: int | None) -> Dataset:
    df = pd.read_csv(csv_path)

    required_columns = {"audio_path", "transcript"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in {csv_path}: {missing_columns}")

    if max_samples is not None:
        df = df.head(max_samples).copy()

    df["transcript"] = df["transcript"].apply(normalize_text)

    return Dataset.from_pandas(df, preserve_index=False)


def prepare_dataset(example: dict[str, Any], processor: WhisperProcessor) -> dict[str, Any]:
    audio_path = example["audio_path"]

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    input_features = processor.feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
    ).input_features[0]

    labels = processor.tokenizer(example["transcript"]).input_ids

    return {
        "input_features": input_features,
        "labels": labels,
    }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        label_features = [
            {"input_ids": feature["labels"]}
            for feature in features
        ]

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100,
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def build_training_arguments(args: argparse.Namespace) -> Seq2SeqTrainingArguments:
    kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": 2,
        "predict_with_generate": True,
        "generation_max_length": args.generation_max_length,
        "generation_num_beams": 1,
        "fp16": torch.cuda.is_available(),
        "report_to": "none",
        "remove_unused_columns": False,
        "push_to_hub": False,
        "load_best_model_at_end": False,
    }

    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)

    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    if "save_strategy" in signature.parameters:
        kwargs["save_strategy"] = "steps"

    return Seq2SeqTrainingArguments(**kwargs)


def configure_whisper_generation(
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

    # Important for newer Transformers versions:
    # generation parameters must not be stored in model.config,
    # otherwise save_pretrained() raises a ValueError.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = None


def train(args: argparse.Namespace) -> None:
    print("Loading datasets...")
    train_dataset = load_split(args.train_csv, args.max_train_samples)
    validation_dataset = load_split(args.validation_csv, args.max_validation_samples)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    print(f"Loading processor and model: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language=args.language,
        task=args.task,
    )

    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    configure_whisper_generation(
        model=model,
        processor=processor,
        language=args.language,
        task=args.task,
    )

    if args.freeze_encoder:
        print("Freezing encoder...")
        model.freeze_encoder()

    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    print("Preparing train dataset...")
    train_dataset = train_dataset.map(
        lambda example: prepare_dataset(example, processor),
        remove_columns=train_dataset.column_names,
        desc="Preparing train data",
    )

    print("Preparing validation dataset...")
    validation_dataset = validation_dataset.map(
        lambda example: prepare_dataset(example, processor),
        remove_columns=validation_dataset.column_names,
        desc="Preparing validation data",
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(prediction_output: Any) -> dict[str, float]:
        pred_ids = prediction_output.predictions
        label_ids = prediction_output.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        predictions = processor.tokenizer.batch_decode(
            pred_ids,
            skip_special_tokens=True,
        )

        references = processor.tokenizer.batch_decode(
            label_ids,
            skip_special_tokens=True,
        )

        predictions = [normalize_text(prediction) for prediction in predictions]
        references = [normalize_text(reference) for reference in references]

        wer = wer_metric.compute(predictions=predictions, references=references)
        cer = cer_metric.compute(predictions=predictions, references=references)

        return {
            "wer": wer,
            "cer": cer,
        }

    training_args = build_training_arguments(args)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    print("Running final validation evaluation...")
    metrics = trainer.evaluate()

    metrics_path = args.output_dir / "validation_metrics.txt"

    with metrics_path.open("w", encoding="utf-8") as file:
        for key, value in sorted(metrics.items()):
            file.write(f"{key}: {value}\n")

    print("Training completed.")
    print(f"Model saved to: {args.output_dir}")
    print(f"Validation metrics saved to: {metrics_path}")
    print(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/processed/train.csv"),
    )
    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=Path("data/processed/validation.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/whisper-small-ro-finetuned"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/whisper-small",
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
        "--max-train-samples",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--generation-max-length",
        type=int,
        default=225,
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
    )

    args = parser.parse_args()

    args.max_train_samples = None if args.max_train_samples < 0 else args.max_train_samples
    args.max_validation_samples = None if args.max_validation_samples < 0 else args.max_validation_samples

    train(args)


if __name__ == "__main__":
    main()