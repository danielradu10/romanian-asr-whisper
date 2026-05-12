# Romanian ASR with Whisper Fine-Tuning

This repository contains a Romanian automatic speech recognition (ASR) pipeline based on OpenAI Whisper. The project covers dataset preparation, audio embedding extraction, audio-based clustering, speaker-aware train/validation/test splitting, baseline evaluation, Whisper fine-tuning, final evaluation, error analysis, sentence-level clustering, audio-quality analysis, and a simple web demo.

The current experimental setup uses Mozilla Common Voice Romanian data and fine-tunes `openai/whisper-small` for Romanian transcription.

---

## 1. Main Results

The final fine-tuned Whisper-small model was evaluated on a held-out Romanian Common Voice test split.

| Model                                    | Test samples |    WER |    CER |
| ---------------------------------------- | -----------: | -----: | -----: |
| `openai/whisper-small` baseline          |          408 | 0.2870 | 0.0828 |
| Whisper-small fine-tuned, checkpoint 400 |          408 | 0.1789 | 0.0487 |
| Whisper-small fine-tuned, 2 epochs       |          408 | 0.1766 | 0.0480 |

Relative improvement of the 2-epoch fine-tuned model over the baseline:

| Metric | Relative reduction |
| ------ | -----------------: |
| WER    |             ~38.5% |
| CER    |             ~42.0% |

The best current model is the full 2-epoch fine-tuned checkpoint.

---

## 2. Dataset Summary

The dataset is built from Mozilla Common Voice Romanian recordings. The processed split is speaker-aware, meaning the same `speaker_id` is assigned to only one split.

| Split      | Samples | Duration | Speakers |
| ---------- | ------: | -------: | -------: |
| Train      |    3450 |  ~4.00 h |      163 |
| Validation |     429 |  ~0.50 h |       71 |
| Test       |     408 |  ~0.50 h |       71 |
| Total      |    4287 |  ~5.00 h |        - |

The split was created after extracting audio embeddings and clustering the data to preserve audio-condition diversity across train, validation, and test.

---

## 3. Repository Structure

```text
romanian-asr-whisper/
├── data/
│   ├── README.md
│   ├── raw/                         # Local Common Voice data, not committed
│   └── processed/                   # Generated CSVs and embeddings
├── models/
│   └── whisper-tiny-ro-smoke-test/  # Optional local smoke-test model
├── results/
│   ├── baseline_metrics.csv
│   ├── baseline_predictions.csv
│   ├── cluster_metadata_analysis/
│   ├── k_selection_summary.csv
│   └── ...
├── src/
│   ├── prepare_common_voice.py
│   ├── extract_embeddings.py
│   ├── select_k_for_clustering.py
│   ├── cluster_and_split.py
│   ├── analyze_cluster_metadata.py
│   ├── train_whisper.py
│   ├── evaluate_baseline.py
│   ├── evaluate_finetuned.py
│   ├── analyze_asr_errors.py
│   ├── analyze_asr_errors_stopwords.py
│   ├── select_k_for_text_clustering.py
│   ├── analyze_asr_errors_sentence_embeddings.py
│   ├── analyze_audio_quality_errors.py
│   └── app.py
├── requirements.txt
└── README.md
```

Recommended: keep large datasets, checkpoints, `.venv`, `.idea`, and temporary result folders out of Git.

---

## 4. Environment Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU usage, make sure PyTorch is installed with CUDA support. In Google Colab, this is usually already available after selecting a GPU runtime.

Check the current device from Python:

```bash
python - <<'PY'
import torch
print('CUDA:', torch.cuda.is_available())
print('MPS:', torch.backends.mps.is_available())
PY
```

---

## 5. Data Preparation

Download Mozilla Common Voice Romanian and place it locally under a path similar to:

```text
data/raw/common_voice/cv-corpus-25.0-2026-03-09/ro/
```

The folder should contain:

```text
validated.tsv
clips/
```

Build the initial metadata file:

```bash
python src/prepare_common_voice.py \
  --common-voice-dir data/raw/common_voice/cv-corpus-25.0-2026-03-09/ro \
  --output-csv data/processed/common_voice_metadata.csv \
  --tsv-file validated.tsv \
  --max-hours 5 \
  --min-duration-seconds 1 \
  --max-duration-seconds 30
```

This produces:

```text
data/processed/common_voice_metadata.csv
```

The metadata contains normalized Romanian transcripts, audio paths, duration, speaker identifiers, and recording identifiers.

---

## 6. Audio Embedding Extraction

Extract Whisper encoder embeddings for all selected audio samples:

```bash
python src/extract_embeddings.py \
  --metadata-csv data/processed/common_voice_metadata.csv \
  --output-embeddings data/processed/common_voice_embeddings.npy \
  --output-metadata data/processed/common_voice_metadata_with_embeddings.csv \
  --model-name openai/whisper-small
```

Optional smoke test:

```bash
python src/extract_embeddings.py \
  --metadata-csv data/processed/common_voice_metadata.csv \
  --output-embeddings data/processed/common_voice_embeddings_smoke.npy \
  --output-metadata data/processed/common_voice_metadata_with_embeddings_smoke.csv \
  --model-name openai/whisper-small \
  --max-samples 20
```

---

## 7. Selecting the Number of Audio Clusters

Before splitting the data, evaluate candidate cluster counts:

```bash
python src/select_k_for_clustering.py \
  --embeddings-path data/processed/common_voice_embeddings.npy \
  --metadata-csv data/processed/common_voice_metadata_with_embeddings.csv \
  --output-dir results \
  --min-k 2 \
  --max-k 20 \
  --seeds 42 123 999
```

Outputs:

```text
results/k_selection_detailed.csv
results/k_selection_summary.csv
```

The selected audio clustering configuration used in the current experiments is based on practical constraints such as cluster duration, sample count, speaker count, and cluster balance.

---

## 8. Clustering and Speaker-Aware Splitting

Create final audio clusters and split the dataset into train, validation, and test sets:

```bash
python src/cluster_and_split.py \
  --embeddings-path data/processed/common_voice_embeddings.npy \
  --metadata-csv data/processed/common_voice_metadata_with_embeddings.csv \
  --output-data-dir data/processed \
  --output-results-dir results \
  --n-clusters 3 \
  --train-ratio 0.8 \
  --validation-ratio 0.1 \
  --test-ratio 0.1
```

Outputs:

```text
data/processed/metadata_with_clusters_and_split.csv
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
results/cluster_distribution.csv
results/split_distribution.csv
```

The split logic is speaker-aware. It checks that no speaker appears in more than one split.

---

## 9. Cluster Metadata Analysis

Analyze the distribution of duration, speaker count, age, gender, accents, and Common Voice variants across clusters and splits:

```bash
python src/analyze_cluster_metadata.py \
  --clustered-metadata-csv data/processed/metadata_with_clusters_and_split.csv \
  --common-voice-tsv data/raw/common_voice/cv-corpus-25.0-2026-03-09/ro/validated.tsv \
  --output-dir results/cluster_metadata_analysis
```

Important outputs:

```text
results/cluster_metadata_analysis/cluster_profile.csv
results/cluster_metadata_analysis/split_profile.csv
results/cluster_metadata_analysis/metadata_with_common_voice_fields.csv
```

---

## 10. Baseline Evaluation

Evaluate the original Whisper-small model on the test set:

```bash
python src/evaluate_baseline.py \
  --model-name openai/whisper-small \
  --test-csv data/processed/test.csv \
  --output-csv results/baseline_predictions.csv \
  --metrics-csv results/baseline_metrics.csv \
  --language romanian \
  --task transcribe
```

Outputs:

```text
results/baseline_predictions.csv
results/baseline_metrics.csv
```

Current baseline test performance:

```text
WER: 0.2870
CER: 0.0828
```

---

## 11. Fine-Tuning Whisper

Train Whisper-small on the Romanian train split and evaluate periodically on the validation split:

```bash
python src/train_whisper.py \
  --model-name openai/whisper-small \
  --train-csv data/processed/train.csv \
  --validation-csv data/processed/validation.csv \
  --output-dir models/whisper-small-ro-finetuned \
  --language romanian \
  --task transcribe \
  --num-train-epochs 2 \
  --train-batch-size 4 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-5 \
  --warmup-steps 50 \
  --logging-steps 10 \
  --eval-steps 50 \
  --save-steps 50 \
  --gradient-checkpointing
```

In Google Colab, the trained model was saved to Google Drive:

```text
/content/drive/MyDrive/romanian-asr-results/whisper-small-ro-finetuned-2epochs
```

The final validation metrics after 2 epochs were:

```text
eval_wer: 0.1835
eval_cer: 0.0544
eval_loss: 0.2474
```

---

## 12. Fine-Tuned Model Evaluation

Evaluate the final fine-tuned model:

```bash
python src/evaluate_finetuned.py \
  --model-name /content/drive/MyDrive/romanian-asr-results/whisper-small-ro-finetuned-2epochs \
  --processor-name /content/drive/MyDrive/romanian-asr-results/whisper-small-ro-finetuned-2epochs \
  --base-processor-name openai/whisper-small \
  --test-csv data/processed/test.csv \
  --output-csv results/finetuned_predictions_2epochs_final.csv \
  --metrics-csv results/finetuned_metrics_2epochs_final.csv \
  --language romanian \
  --task transcribe
```

Evaluate checkpoint 400:

```bash
python src/evaluate_finetuned.py \
  --model-name /content/drive/MyDrive/romanian-asr-results/whisper-small-ro-finetuned-2epochs/checkpoint-400 \
  --processor-name /content/drive/MyDrive/romanian-asr-results/whisper-small-ro-finetuned-2epochs \
  --base-processor-name openai/whisper-small \
  --test-csv data/processed/test.csv \
  --output-csv results/finetuned_predictions_checkpoint_400.csv \
  --metrics-csv results/finetuned_metrics_checkpoint_400.csv \
  --language romanian \
  --task transcribe
```

Final 2-epoch test performance:

```text
WER: 0.1766
CER: 0.0480
```

Checkpoint-400 test performance:

```text
WER: 0.1789
CER: 0.0487
```

---

## 13. Error Analysis

### 13.1 Basic ASR Error Analysis

Run word-level error analysis using substitutions, deletions, insertions, audio clusters, duration buckets, and text clusters:

```bash
python src/analyze_asr_errors.py \
  --predictions-csv results/finetuned_predictions_2epochs_final.csv \
  --metadata-csv data/processed/metadata_with_clusters_and_split.csv \
  --output-dir results/error_analysis_finetuned_2epochs \
  --n-text-clusters 5
```

Outputs include:

```text
per_sample_errors.csv
overall_error_metrics.csv
error_summary_by_audio_cluster.csv
error_summary_by_text_cluster.csv
error_summary_by_duration_bucket.csv
error_summary_by_dominant_error.csv
worst_samples.csv
frequent_substitutions.csv
frequent_deletions.csv
frequent_insertions.csv
```

### 13.2 TF-IDF Text Clustering with Romanian Stopwords

The first text-clustering version was affected by high-frequency Romanian words. A second version removes Romanian stopwords and uses TF-IDF with unigrams and bigrams.

```bash
python src/analyze_asr_errors_stopwords.py \
  --predictions-csv results/finetuned_predictions_2epochs_final.csv \
  --metadata-csv data/processed/metadata_with_clusters_and_split.csv \
  --output-dir results/error_analysis_finetuned_2epochs_stopwords_k3 \
  --n-text-clusters 3
```

### 13.3 Selecting k for Text Clustering

Evaluate candidate values of `k` for TF-IDF text clustering:

```bash
python src/select_k_for_text_clustering.py \
  --metadata-csv data/processed/metadata_with_clusters_and_split.csv \
  --predictions-csv results/finetuned_predictions_2epochs_final.csv \
  --output-dir results/text_k_selection_stopwords \
  --min-k 2 \
  --max-k 12
```

The TF-IDF clustering is useful for analysis, but the resulting clusters may still be dominated by repeated phrase templates.

### 13.4 Sentence Embedding Error Analysis

For stronger semantic clustering, use multilingual sentence embeddings:

```bash
python src/analyze_asr_errors_sentence_embeddings.py \
  --predictions-csv results/finetuned_predictions_2epochs_final.csv \
  --metadata-csv data/processed/metadata_with_clusters_and_split.csv \
  --output-dir results/error_analysis_sentence_embeddings \
  --min-k 2 \
  --max-k 10
```

The current run selected `k=2` as the most practical sentence-cluster configuration.

Current sentence-cluster error summary:

| Sentence cluster | Samples | Mean WER | Mean CER |
| ---------------: | ------: | -------: | -------: |
|                0 |     150 |   0.1952 |   0.0539 |
|                1 |     258 |   0.1588 |   0.0419 |

This suggests that semantic content influences the error distribution, not only audio quality.

---

## 14. Audio Quality Analysis

The project also analyzes audio-level features such as duration, speech rate, RMS amplitude, silence ratio, spectral centroid, spectral rolloff, zero-crossing rate, MFCC statistics, and a proxy SNR/dynamic range measure.

Run the analysis:

```bash
python src/analyze_audio_quality_errors.py \
  --predictions-csv results/finetuned_predictions_2epochs_final.csv \
  --metadata-csv data/processed/metadata_with_clusters_and_split.csv \
  --sentence-clusters-csv results/error_analysis_sentence_embeddings/sentence_cluster_assignments.csv \
  --output-dir results/audio_quality_analysis
```

Optional smoke test:

```bash
python src/analyze_audio_quality_errors.py \
  --predictions-csv results/finetuned_predictions_2epochs_final.csv \
  --metadata-csv data/processed/metadata_with_clusters_and_split.csv \
  --sentence-clusters-csv results/error_analysis_sentence_embeddings/sentence_cluster_assignments.csv \
  --output-dir results/audio_quality_analysis_smoke \
  --max-samples 20
```

Important outputs:

```text
overall_audio_quality_metrics.csv
audio_feature_correlations.csv
error_by_audio_feature_buckets.csv
high_error_vs_rest_audio_features.csv
error_by_audio_cluster.csv
error_by_sentence_cluster.csv
error_by_duration_bucket.csv
error_by_audio_and_sentence_cluster.csv
worst_samples_with_audio_features.csv
figures/
```

Main observation: simple audio-quality features have weak direct correlation with WER. The strongest Spearman correlations with WER are below 0.20, suggesting that error behavior is not explained by a single simple audio feature.

However, combining audio clusters and sentence clusters exposes a clearer pattern:

| Audio cluster | Sentence cluster | Samples | Mean WER |
| ------------: | ---------------: | ------: | -------: |
|             1 |                0 |      45 |   0.2255 |
|             2 |                0 |      40 |   0.2040 |
|             2 |                1 |      58 |   0.1841 |
|             0 |                0 |      65 |   0.1687 |
|             1 |                1 |     104 |   0.1597 |
|             0 |                1 |      96 |   0.1425 |

The highest-error region combines a more difficult audio cluster with the more difficult sentence cluster.

---

## 15. Web App

The repository includes a simple Python web application in:

```text
src/app.py
```

Run it locally with:

```bash
python src/app.py
```

Depending on the implementation, the app can be used to upload or select an audio sample and transcribe it using the fine-tuned Whisper model.

Before running the app, make sure the model path in `src/app.py` points to an available local or Google Drive checkpoint.

---

## 16. Google Colab Workflow

A Colab notebook is provided for GPU-based execution. The recommended Colab flow is:

1. Mount Google Drive.
2. Install dependencies from `requirements.txt`.
3. Prepare or load the processed dataset.
4. Run a smoke test on a small number of samples.
5. Fine-tune Whisper-small.
6. Evaluate the final model and selected checkpoints.
7. Run error analysis.
8. Save all results back to Google Drive.

Typical Drive output path:

```text
/content/drive/MyDrive/romanian-asr-results/
```

Recommended final model path:

```text
/content/drive/MyDrive/romanian-asr-results/whisper-small-ro-finetuned-2epochs
```

---

## 17. Notes for Future Experiments

This repository is prepared so additional experiments can be added without changing the core pipeline.

Useful next steps:

* Add LoRA / PEFT fine-tuning for Whisper.
* Compare full fine-tuning vs. parameter-efficient fine-tuning.
* Evaluate larger Whisper variants if GPU resources allow it.
* Add pitch-based audio analysis with `--compute-pitch`.
* Run error analysis on multiple model variants using the same test split.
* Add a model comparison script that automatically builds a before/after results table.

Suggested future result table:

| Model                  | Training strategy | Trainable parameters |    WER |    CER |
| ---------------------- | ----------------- | -------------------: | -----: | -----: |
| Whisper-small baseline | zero-shot         |                    0 | 0.2870 | 0.0828 |
| Whisper-small          | full fine-tuning  |                  all | 0.1766 | 0.0480 |
| Whisper-small          | LoRA              |                  TBD |    TBD |    TBD |

---

## 18. Reproducibility Checklist

To reproduce the current main experiment:

```bash
# 1. Prepare Common Voice metadata
python src/prepare_common_voice.py \
  --common-voice-dir data/raw/common_voice/cv-corpus-25.0-2026-03-09/ro \
  --output-csv data/processed/common_voice_metadata.csv \
  --tsv-file validated.tsv \
  --max-hours 5

# 2. Extract Whisper encoder embeddings
python src/extract_embeddings.py \
  --metadata-csv data/processed/common_voice_metadata.csv \
  --output-embeddings data/processed/common_voice_embeddings.npy \
  --output-metadata data/processed/common_voice_metadata_with_embeddings.csv \
  --model-name openai/whisper-small

# 3. Cluster and split
python src/cluster_and_split.py \
  --embeddings-path data/processed/common_voice_embeddings.npy \
  --metadata-csv data/processed/common_voice_metadata_with_embeddings.csv \
  --output-data-dir data/processed \
  --output-results-dir results \
  --n-clusters 3

# 4. Evaluate baseline
python src/evaluate_baseline.py \
  --model-name openai/whisper-small \
  --test-csv data/processed/test.csv \
  --output-csv results/baseline_predictions.csv \
  --metrics-csv results/baseline_metrics.csv

# 5. Train fine-tuned model
python src/train_whisper.py \
  --model-name openai/whisper-small \
  --train-csv data/processed/train.csv \
  --validation-csv data/processed/validation.csv \
  --output-dir models/whisper-small-ro-finetuned \
  --num-train-epochs 2 \
  --gradient-checkpointing

# 6. Evaluate fine-tuned model
python src/evaluate_finetuned.py \
  --model-name models/whisper-small-ro-finetuned \
  --processor-name models/whisper-small-ro-finetuned \
  --base-processor-name openai/whisper-small \
  --test-csv data/processed/test.csv \
  --output-csv results/finetuned_predictions_2epochs_final.csv \
  --metrics-csv results/finetuned_metrics_2epochs_final.csv

# 7. Sentence embedding error analysis
python src/analyze_asr_errors_sentence_embeddings.py \
  --predictions-csv results/finetuned_predictions_2epochs_final.csv \
  --metadata-csv data/processed/metadata_with_clusters_and_split.csv \
  --output-dir results/error_analysis_sentence_embeddings

# 8. Audio quality analysis
python src/analyze_audio_quality_errors.py \
  --predictions-csv results/finetuned_predictions_2epochs_final.csv \
  --metadata-csv data/processed/metadata_with_clusters_and_split.csv \
  --sentence-clusters-csv results/error_analysis_sentence_embeddings/sentence_cluster_assignments.csv \
  --output-dir results/audio_quality_analysis
```

---

## 19. Troubleshooting

### `Input type (float) and bias type (c10::Half) should be the same`

This happens when the model is loaded in FP16 but the input features remain FP32. The evaluation code must send features to the same dtype as the model:

```python
input_features = inputs.input_features.to(device=device, dtype=model.dtype)
```

### `No predictions were produced`

This usually means all samples failed during evaluation. Check the printed skip messages. Common causes include broken audio paths, dtype mismatch, missing audio files, or incorrect processor/model path.

### `num_frames` error

This can occur when using high-level pipelines or incompatible audio inputs. The current evaluation scripts use explicit `librosa.load` plus Whisper feature extraction to avoid this issue.

### Slow GPU evaluation warning

If using Hugging Face pipelines sequentially, evaluation can be inefficient. The current scripts avoid the pipeline API and call the model directly.

---

## 20. Citation / Dataset Note

This project uses Mozilla Common Voice Romanian speech data and OpenAI Whisper models through the Hugging Face Transformers ecosystem. If this repository is used in a paper or report, cite Mozilla Common Voice, OpenAI Whisper, Hugging Face Transformers, and any additional libraries used for evaluation or sentence embeddings.
