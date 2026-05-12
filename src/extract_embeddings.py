import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel


SAMPLE_RATE = 16_000


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_audio(audio_path: str) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    return audio


def mean_pool_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    hidden_states shape: [batch_size, sequence_length, hidden_size]
    Returns: [hidden_size]
    """
    return hidden_states.mean(dim=1).squeeze(0)


def l2_normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.clamp(torch.norm(vector, p=2), min=1e-12)


def extract_embeddings(
    metadata_csv: Path,
    output_embeddings: Path,
    output_metadata: Path,
    model_name: str,
    max_samples: int | None,
) -> None:
    device = get_device()
    print(f"Using device: {device}")

    df = pd.read_csv(metadata_csv)

    if max_samples is not None:
        df = df.head(max_samples).copy()

    if "audio_path" not in df.columns:
        raise ValueError("metadata_csv must contain an 'audio_path' column")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    embeddings = []
    valid_rows = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
            audio_path = row["audio_path"]

            try:
                audio = load_audio(audio_path)

                inputs = feature_extractor(
                    audio,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                )

                input_features = inputs.input_features.to(device)

                encoder_outputs = model.encoder(input_features=input_features)
                hidden_states = encoder_outputs.last_hidden_state

                embedding = mean_pool_hidden_states(hidden_states)
                embedding = l2_normalize(embedding)

                embeddings.append(embedding.detach().cpu().numpy())
                valid_rows.append(row)

            except Exception as exc:
                print(f"Skipping {audio_path}: {exc}")

    if not embeddings:
        raise RuntimeError("No embeddings were extracted.")

    embeddings_array = np.vstack(embeddings)
    output_df = pd.DataFrame(valid_rows)

    output_embeddings.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_embeddings, embeddings_array)
    output_df.to_csv(output_metadata, index=False)

    print(f"Saved embeddings to: {output_embeddings}")
    print(f"Saved aligned metadata to: {output_metadata}")
    print(f"Embeddings shape: {embeddings_array.shape}")
    print(f"Valid samples: {len(output_df)}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/processed/common_voice_metadata.csv"),
    )
    parser.add_argument(
        "--output-embeddings",
        type=Path,
        default=Path("data/processed/common_voice_embeddings.npy"),
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("data/processed/common_voice_metadata_with_embeddings.csv"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/whisper-small",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Use a small number for testing. Example: 20. Use -1 for all samples.",
    )

    args = parser.parse_args()

    max_samples = None if args.max_samples < 0 else args.max_samples

    extract_embeddings(
        metadata_csv=args.metadata_csv,
        output_embeddings=args.output_embeddings,
        output_metadata=args.output_metadata,
        model_name=args.model_name,
        max_samples=max_samples,
    )


if __name__ == "__main__":
    main()