# Colab Fine-Tuning and Evaluation Results

This README documents the results produced in the Google Colab fine-tuning run for Romanian Automatic Speech Recognition using `openai/whisper-small`.

The purpose of this folder is to make the experimental outputs understandable without having to inspect every CSV manually. It explains what was trained, how the model was evaluated, what each result file contains, and what the main conclusions are.

---

## 1. What this result package contains

The attached result package contains two main groups of artifacts:

```text
colab-results/
├── results/
│   ├── baseline_metrics.csv
│   ├── baseline_predictions.csv
│   ├── finetuned_metrics_2epochs_final.csv
│   ├── finetuned_predictions_2epochs_final.csv
│   ├── finetuned_metrics_checkpoint_400.csv
│   ├── finetuned_predictions_checkpoint_400.csv
│   ├── finetuned_metrics_2epochs_final_smoke.csv
│   ├── finetuned_predictions_2epochs_final_smoke.csv
│   ├── split_distribution.csv
│   ├── cluster_distribution.csv
│   ├── k_selection_summary.csv
│   ├── k_selection_detailed.csv
│   └── cluster_metadata_analysis/
│
└── evaluation/
    ├── before_after_metrics.csv
    ├── finetuned_metrics_2epochs_final.csv
    ├── finetuned_predictions_2epochs_final.csv
    ├── finetuned_metrics_checkpoint_400.csv
    ├── finetuned_predictions_checkpoint_400.csv
    ├── error_analysis_finetuned_2epochs/
    ├── error_analysis_finetuned_2epochs_stopwords/
    ├── error_analysis_finetuned_2epochs_stopwords_k3/
    ├── audio_quality_analysis/
    └── figures/
```

The `results/` directory contains the main baseline, fine-tuned, split, clustering, and metadata outputs. The `evaluation/` directory contains the more detailed error analysis, comparison tables, and generated figures.

> Note: the Colab notebook also ran a sentence-embedding clustering analysis using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. The notebook output shows the sentence-cluster results, and `audio_quality_analysis` contains `sentence_cluster` columns, but the source folder `error_analysis_sentence_embeddings/` is not included in this uploaded ZIP. If that analysis should be archived fully, rerun the sentence-embedding cell and copy `results/error_analysis_sentence_embeddings/` to the final results folder.

---

## 2. Colab execution flow

The Colab notebook follows this pipeline:

1. Mount Google Drive.
2. Unzip the local project into `/content/romanian-asr-whisper`.
3. Install runtime dependencies, including `ffmpeg`, `transformers`, `datasets`, `evaluate`, `librosa`, `jiwer`, `accelerate`, `scikit-learn`, `matplotlib`, and `sentence-transformers`.
4. Verify that the prepared split files exist:
   - `data/processed/train.csv`
   - `data/processed/validation.csv`
   - `data/processed/test.csv`
5. Run an optional smoke-test training/evaluation path.
6. Fine-tune `openai/whisper-small` for 2 epochs.
7. Evaluate:
   - the final 2-epoch model;
   - checkpoint 400;
   - the pre-trained baseline.
8. Generate comparison metrics.
9. Generate ASR error analysis.
10. Generate text-cluster analysis.
11. Generate sentence-embedding analysis.
12. Generate audio-quality feature analysis.
13. Copy final artifacts back to Google Drive.

---

## 3. Training setup

The main fine-tuning run used the following configuration:

```bash
python src/train_whisper.py \
  --model-name openai/whisper-small \
  --output-dir /content/drive/MyDrive/romanian-asr-results/whisper-small-ro-finetuned-2epochs \
  --num-train-epochs 2 \
  --train-batch-size 4 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 \
  --warmup-steps 50 \
  --logging-steps 25 \
  --eval-steps 200 \
  --save-steps 200 \
  --generation-max-length 225 \
  --gradient-checkpointing
```

Important details:

| Parameter | Value | Meaning |
|---|---:|---|
| Base model | `openai/whisper-small` | Starting ASR model. |
| Epochs | 2 | The model saw the full training split twice. |
| Train batch size | 4 | Per-device batch size. |
| Gradient accumulation | 4 | Effective batch size is approximately `4 × 4 = 16`. |
| Learning rate | `5e-6` | Conservative learning rate for fine-tuning. |
| Warmup steps | 50 | Learning rate warmup before decay. |
| Gradient checkpointing | enabled | Saves GPU memory at the cost of additional computation. |
| Generation max length | 225 | Maximum generated token length during evaluation. |

Training summary from the notebook:

| Metric | Value |
|---|---:|
| Train samples | 3,450 |
| Validation samples | 429 |
| Total training steps | 432 |
| Train runtime | 2,547 seconds |
| Train loss | 1.067 |
| Final validation loss | 0.2474 |
| Final validation WER | 0.1835 |
| Final validation CER | 0.0544 |

The final validation WER of approximately **18.35%** is close to the final test WER of **17.66%**, which suggests that the test result is not obviously inconsistent with validation behavior.

---

## 4. Dataset split summary

The fixed dataset split used in this experiment is documented in `results/split_distribution.csv`.

| Split | Samples | Duration hours | Speakers |
|---|---:|---:|---:|
| Train | 3,450 | 4.0005 | 163 |
| Validation | 429 | 0.5011 | 71 |
| Test | 408 | 0.5000 | 71 |

The total prepared dataset contains **4,287 utterances** and about **5 hours of audio**. The test set contains **408 utterances**, which is the set used for all final reported test metrics.

The split was produced in a speaker-aware way, meaning that the same `speaker_id` should not appear in multiple splits. This is important because ASR models can otherwise overfit speaker-specific characteristics.

---

## 5. Main model comparison

The most important comparison is stored in `evaluation/before_after_metrics.csv`.

| Model | Samples | WER | CER | Relative WER reduction vs baseline | Relative CER reduction vs baseline |
|---|---:|---:|---:|---:|---:|
| `openai/whisper-small` baseline | 408 | 0.2870 | 0.0828 | 0.00% | 0.00% |
| Fine-tuned final model, 2 epochs | 408 | 0.1766 | 0.0480 | 38.47% | 42.03% |
| Fine-tuned checkpoint 400 | 408 | 0.1789 | 0.0487 | 37.67% | 41.18% |

Main conclusion: fine-tuning clearly improves Romanian ASR performance on this fixed test set. The final model reduces WER from **28.70%** to **17.66%**, and CER from **8.28%** to **4.80%**.

The final 2-epoch model is slightly better than checkpoint 400:

| Model | WER | CER |
|---|---:|---:|
| Checkpoint 400 | 0.1789 | 0.0487 |
| Final 2-epoch model | 0.1766 | 0.0480 |

The difference is small, but the final model is still the best among these evaluated checkpoints.

---

## 6. Exact-match and per-sample behavior

Besides global WER/CER, the per-sample behavior is useful because it shows whether the model improves broadly or only on a few examples.

Using per-sample WER:

| Model | Exact transcriptions | Exact transcription rate | Median sample WER | Mean sample WER |
|---|---:|---:|---:|---:|
| Baseline Whisper small | 91 / 408 | 22.30% | 0.250 | 0.2869 |
| Fine-tuned final model | 181 / 408 | 44.36% | 0.125 | 0.1722 |
| Fine-tuned checkpoint 400 | 180 / 408 | 44.12% | 0.125 | 0.1742 |

The fine-tuned model approximately doubles the number of exact test-set transcriptions: **91 exact samples before fine-tuning vs 181 after fine-tuning**.

When comparing the baseline and final model sample by sample:

| Outcome | Samples | Percentage |
|---|---:|---:|
| Improved after fine-tuning | 202 | 49.51% |
| Unchanged | 156 | 38.24% |
| Worse after fine-tuning | 50 | 12.25% |

This is a good sign: the model improves on many samples, keeps many samples unchanged, and regresses on a smaller subset. Still, the 50 regressions should be acknowledged, especially because some regressions are visible in the worst-sample table.

---

## 7. What the prediction files contain

The prediction files are:

| File | Meaning |
|---|---|
| `results/baseline_predictions.csv` | Predictions from the original `openai/whisper-small` model. |
| `results/finetuned_predictions_2epochs_final.csv` | Predictions from the final fine-tuned model. |
| `results/finetuned_predictions_checkpoint_400.csv` | Predictions from the intermediate checkpoint 400. |
| `results/finetuned_predictions_2epochs_final_smoke.csv` | Small 5-sample smoke-test output used only to verify that evaluation works. |

Typical columns:

| Column | Meaning |
|---|---|
| `audio_path` | Path to the evaluated audio file. |
| `reference` | Ground-truth normalized transcript. |
| `prediction` | Model output after normalization. |
| `source` | Dataset source, here Common Voice. |
| `duration_seconds` | Audio duration. |
| `speaker_id` | Speaker identifier. |
| `cluster` | Audio embedding cluster assigned before splitting. |
| `split` | Dataset split, usually `test` for these files. |

These files are the source for all later error analysis.

---

## 8. ASR error analysis: general version

The directory `evaluation/error_analysis_finetuned_2epochs/` contains the first detailed error analysis over the final fine-tuned predictions.

Important files:

| File | Meaning |
|---|---|
| `overall_error_metrics.csv` | Global and mean sample WER/CER. |
| `per_sample_errors.csv` | One row per evaluated test utterance, including WER, CER, error counts, clusters, and metadata. |
| `error_summary_by_audio_cluster.csv` | Error grouped by audio embedding cluster. |
| `error_summary_by_text_cluster.csv` | Error grouped by automatically discovered TF-IDF text clusters. |
| `error_summary_by_duration_bucket.csv` | Error grouped by audio duration bucket. |
| `error_summary_by_dominant_error.csv` | Error grouped by the dominant edit type. |
| `worst_samples.csv` | Top 50 worst transcribed samples. |
| `frequent_substitutions.csv` | Most frequent word-level substitutions. |
| `frequent_deletions.csv` | Most frequent deleted reference words. |
| `frequent_insertions.csv` | Most frequent inserted words. |
| `top_terms_per_text_cluster.csv` | Most representative terms for each text cluster. |

Overall metrics in this folder match the final test metrics:

| Metric | Value |
|---|---:|
| Samples | 408 |
| Overall WER | 0.1766 |
| Overall CER | 0.0480 |
| Mean sample WER | 0.1722 |
| Mean sample CER | 0.0463 |

`overall_wer` pools all word edits globally. `mean_sample_wer` first computes WER for each utterance, then averages across utterances. Both are useful: global WER is the standard headline metric, while mean sample WER shows the average per-utterance difficulty.

---

## 9. Error type analysis

The dominant error type analysis shows what kind of mistakes remain after fine-tuning.

| Dominant error | Samples | Mean WER | Mean CER | Mean substitutions | Mean deletions | Mean insertions |
|---|---:|---:|---:|---:|---:|---:|
| None | 181 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Substitution | 220 | 0.3100 | 0.0824 | 1.8955 | 0.1818 | 0.2318 |
| Deletion | 4 | 0.2232 | 0.0665 | 0.2500 | 1.2500 | 0.0000 |
| Insertion | 3 | 0.3836 | 0.1618 | 0.6667 | 0.0000 | 2.3333 |

Main conclusion: most remaining non-perfect samples are dominated by **substitutions**, not deletions or insertions. This means the model usually produces a transcript of roughly the right length, but confuses words or word forms.

Examples of observed substitution patterns include:

| Reference word | Predicted word | Count |
|---|---|---:|
| `iti` | `eți` | 3 |
| `ști` | `aștii` | 2 |
| `sa` | `să` | 2 |
| `ca` | `că` | 2 |
| `in` | `în` | 2 |
| `banii` | `bani` | 2 |
| `dvs` | `dumneavoastră` | 2 |

Some of these are real ASR confusions, while others are normalization/orthography issues. For example, `sa` vs `să`, `ca` vs `că`, and `in` vs `în` are diacritic-related differences. This should be discussed carefully: depending on the final evaluation policy, some of these differences may be considered less severe than completely wrong lexical substitutions.

Frequent deletions and insertions mostly involve short function words such as `de`, `a`, `o`, `să`, `nu`, `că`. This is typical for ASR because short unstressed words are acoustically weak and easy to miss or hallucinate.

---

## 10. Error by audio embedding cluster

The test set was previously assigned to audio clusters based on Whisper encoder embeddings. These clusters are not labels such as gender or accent; they are unsupervised groups of acoustically or representationally similar samples.

From `error_summary_by_audio_cluster.csv`:

| Audio cluster | Samples | Mean duration | Mean WER | Mean CER | Mean substitutions | Mean deletions | Mean insertions |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 161 | 4.13 s | 0.1531 | 0.0386 | 0.8758 | 0.0932 | 0.1366 |
| 1 | 149 | 3.99 s | 0.1796 | 0.0543 | 1.0738 | 0.1342 | 0.1007 |
| 2 | 98 | 5.52 s | 0.1922 | 0.0468 | 1.2143 | 0.1020 | 0.2143 |

Main conclusions:

- **Cluster 0 is easiest**, with mean WER around **15.31%**.
- **Cluster 2 is hardest**, with mean WER around **19.22%**.
- Cluster 2 also has the longest average duration and the highest average number of reference words, which likely contributes to the higher WER.
- Cluster 1 has the highest CER even though its WER is slightly lower than cluster 2. This suggests that in cluster 1 the word-level mistakes may also involve more character-level distortion.

Fine-tuning improved all audio clusters compared with the baseline. The largest per-sample WER improvement was in audio cluster 1:

| Audio cluster | Baseline mean sample WER | Fine-tuned mean sample WER | Mean WER reduction | Improved samples | Worse samples |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2406 | 0.1531 | 0.0875 | 44.72% | 10.56% |
| 1 | 0.3509 | 0.1796 | 0.1713 | 55.03% | 10.74% |
| 2 | 0.2656 | 0.1922 | 0.0733 | 48.98% | 17.35% |

This suggests that the baseline struggled especially with cluster 1, and fine-tuning helped that cluster substantially.

---

## 11. Error by utterance duration

Duration buckets are defined as:

| Bucket | Definition |
|---|---|
| `short_<3s` | Audio shorter than 3 seconds. |
| `medium_3_6s` | Audio between 3 and 6 seconds. |
| `long_>6s` | Audio longer than 6 seconds. |

From `error_summary_by_duration_bucket.csv`:

| Duration bucket | Samples | Mean duration | Mean word count | Mean WER | Mean CER |
|---|---:|---:|---:|---:|---:|
| Short `<3s` | 12 | 2.77 s | 5.50 | 0.2645 | 0.0737 |
| Medium `3-6s` | 373 | 4.31 s | 7.20 | 0.1671 | 0.0442 |
| Long `>6s` | 23 | 6.90 s | 9.09 | 0.2066 | 0.0661 |

Main conclusions:

- Medium-length utterances are the easiest group.
- Very short utterances are the hardest, but there are only 12 such samples, so this result should be interpreted cautiously.
- Long utterances are also harder than medium utterances, probably because longer sequences give the model more opportunities to accumulate substitutions.

For the paper, this can be summarized as: performance is best on medium-length utterances, while very short and long utterances remain more difficult.

---

## 12. Text-cluster analysis

Several variants of text clustering were tested.

### 12.1 Initial TF-IDF text clustering

The first version in `error_analysis_finetuned_2epochs/` used TF-IDF clustering over normalized transcripts without an extended Romanian stopword list. This produced clusters polluted by common function words such as `este`, `de`, `nu`, `să`, and `aceasta`.

This version is still useful as a baseline, but it is less interpretable because many top terms are grammatical fillers rather than semantic topics.

### 12.2 TF-IDF with Romanian stopwords, k=5

The improved version in `error_analysis_finetuned_2epochs_stopwords/` removes Romanian stopwords and common demonstratives/fillers.

With `k=5`, the clusters became more interpretable but very imbalanced:

| Text cluster | Samples | Mean WER | Mean CER | Interpretation |
|---:|---:|---:|---:|---|
| 0 | 8 | 0.2232 | 0.0725 | Very small cluster, harder but statistically fragile. |
| 2 | 342 | 0.1784 | 0.0478 | Large general cluster. |
| 4 | 23 | 0.1579 | 0.0394 | Small phrase/topic cluster. |
| 3 | 26 | 0.1281 | 0.0343 | `trebuie` / action-oriented sentences. |
| 1 | 9 | 0.0556 | 0.0171 | `cred` / formulaic statements; easiest but very small. |

The problem with `k=5` is not performance; it is interpretability and statistical balance. Some clusters have fewer than 10 test samples, so their WER is unstable.

### 12.3 TF-IDF with Romanian stopwords, k=3

The final simpler version in `error_analysis_finetuned_2epochs_stopwords_k3/` uses `k=3`.

| Text cluster | Samples | Mean WER | Mean CER | Top terms / meaning |
|---:|---:|---:|---:|---|
| 0 | 349 | 0.1773 | 0.0480 | General parliamentary/political statements: `avem`, `cred`, `există`, `acum`, `nevoie`, `raport`, `important`. |
| 1 | 30 | 0.1443 | 0.0360 | Phrases around `această`, `situație`, `problemă`, `mulțumesc`, `privință`. |
| 2 | 29 | 0.1394 | 0.0366 | Action/modality phrases around `trebuie`, `acționăm`, `facem`, `luăm`, `punem`. |

Main conclusion: the `k=3` version is easier to explain than the `k=5` version, but it still has one dominant cluster containing most of the test set. For the paper/report, this should be described as an exploratory text-cluster analysis, not as a definitive taxonomy of linguistic error types.

---

## 13. Text k-selection experiment

The notebook also evaluates different values of `k` for TF-IDF text clustering using:

- cosine silhouette score;
- stability across random seeds using Adjusted Rand Index;
- minimum cluster size;
- cluster balance ratio;
- WER spread between clusters.

The k-selection output showed that simple TF-IDF text clustering is not naturally well-separated for this dataset. The silhouette scores were very low, and larger k values produced imbalanced clusters. This explains why the text clusters were not as meaningful as expected.

Interpretation:

- Low silhouette means the text clusters overlap heavily.
- High imbalance means the clustering tends to create one very large generic cluster and several small phrase-specific clusters.
- Therefore, TF-IDF clustering should be treated as a lightweight exploratory analysis, not as a robust linguistic grouping method.

---

## 14. Sentence-embedding clustering experiment

The notebook tested sentence embeddings using:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

The model produced embeddings of shape:

```text
(4287, 384)
```

This means one 384-dimensional sentence embedding was created for each of the 4,287 transcript samples.

The k-selection summary suggested `k=2` as the most practical value:

| k | Silhouette cosine mean | Stability ARI mean | Minimum cluster samples | Balance ratio | Practical score |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.1123 | 0.9981 | 1,646 | 1.60 | 0.9782 |
| 3 | 0.0776 | 0.9718 | 1,276 | 1.26 | 0.7081 |
| 4 | 0.0760 | 0.9806 | 845 | 1.65 | 0.6252 |

For the final test-set error analysis, the two sentence clusters were:

| Sentence cluster | Samples | Mean WER | Mean CER | Main terms |
|---:|---:|---:|---:|---|
| 0 | 150 | 0.1952 | 0.0539 | `există`, `doar`, `trebuie`, `problemă`, `putem`, `situație`, `însă`, `fără`. |
| 1 | 258 | 0.1588 | 0.0419 | `trebuie`, `avem`, `nevoie`, `cred`, `important`, `două`, `raport`, `mulțumesc`. |

This is more stable than TF-IDF clustering and gives a clearer split. Sentence cluster 0 is harder than sentence cluster 1 by about **3.64 WER points**.

However, sentence-embedding clusters should still be interpreted carefully. They group sentences by semantic similarity, not necessarily by acoustic difficulty.

---

## 15. Audio-quality analysis

The directory `evaluation/audio_quality_analysis/` connects ASR errors with measurable audio features extracted from the waveform using `librosa`.

Important files:

| File | Meaning |
|---|---|
| `overall_audio_quality_metrics.csv` | Global ASR + audio feature summary. |
| `per_sample_audio_quality_errors.csv` | One row per test utterance with ASR errors and audio features. |
| `audio_feature_correlations.csv` | Correlation between each audio feature and WER/CER. |
| `error_by_audio_feature_buckets.csv` | WER/CER by low/medium/high buckets of each audio feature. |
| `high_error_vs_rest_audio_features.csv` | Compares high-error samples against the rest. |
| `error_by_audio_cluster.csv` | Audio quality + error grouped by audio cluster. |
| `error_by_sentence_cluster.csv` | Audio quality + error grouped by sentence cluster. |
| `error_by_audio_and_sentence_cluster.csv` | Joint grouping by audio cluster and sentence cluster. |
| `worst_samples_with_audio_features.csv` | Worst ASR samples enriched with audio features. |
| `skipped_audio_files.csv` | Files skipped during audio analysis. In this run it is effectively empty. |
| `figures/` | Audio-quality plots. |

Overall audio analysis:

| Metric | Value |
|---|---:|
| Samples | 408 |
| Skipped samples | 0 |
| Overall WER | 0.1766 |
| Overall CER | 0.0480 |
| Mean duration | 4.4042 s |
| Mean words per second | 1.6923 |
| Mean RMS dB | -26.8628 dB |
| Mean silence ratio | 0.3465 |
| Mean SNR proxy | 66.6448 dB |
| Pitch computed | No |

No files were skipped, so the audio-quality analysis covers the entire test set.

---

## 16. Most informative audio features

The strongest correlations with WER are still weak to moderate. The top features from `audio_feature_correlations.csv` are:

| Feature | Spearman correlation with WER | Interpretation |
|---|---:|---|
| `spectral_centroid_mean` | -0.1970 | Lower spectral centroid tends to be associated with higher WER. |
| `spectral_rolloff_mean` | -0.1948 | Lower rolloff tends to be associated with higher WER. |
| `mfcc_2_std` | 0.1938 | Variation in the second MFCC has a weak positive association with WER. |
| `spectral_bandwidth_mean` | -0.1863 | Narrower average bandwidth tends to be associated with higher WER. |
| `zero_crossing_rate_mean` | -0.1776 | Lower zero-crossing rate tends to be associated with higher WER. |
| `chars_per_second` | 0.1565 | Faster character rate tends to be associated with higher WER. |

Because these are correlations, not controlled experiments, they should not be stated as direct causes. The safe interpretation is:

> The model tends to make more errors on samples with lower spectral centroid/rolloff and on denser speech, but no single low-level audio feature alone explains the errors strongly.

---

## 17. Audio feature buckets

The bucket analysis is easier to explain than raw correlations because each audio feature is split into low, medium, and high ranges.

### Duration

| Duration bucket | Samples | Mean value | Mean WER | Mean CER |
|---|---:|---:|---:|---:|
| Low duration | 139 | 3.49 s | 0.1785 | 0.0513 |
| Medium duration | 133 | 4.28 s | 0.1518 | 0.0403 |
| High duration | 136 | 5.46 s | 0.1857 | 0.0471 |

Medium-duration samples are easiest. Very short and longer samples are harder.

### Speech density: words per second

| Words-per-second bucket | Samples | Mean value | Mean WER | Mean CER |
|---|---:|---:|---:|---:|
| Low | 142 | 1.33 | 0.1666 | 0.0373 |
| Medium | 131 | 1.69 | 0.1438 | 0.0408 |
| High | 135 | 2.07 | 0.2055 | 0.0611 |

High speech density is associated with worse ASR performance. This is one of the clearest practical findings from the audio analysis.

### Character rate: chars per second

| Chars-per-second bucket | Samples | Mean value | Mean WER | Mean CER |
|---|---:|---:|---:|---:|
| Low | 136 | 7.97 | 0.1338 | 0.0389 |
| Medium | 136 | 9.93 | 0.1748 | 0.0412 |
| High | 136 | 12.02 | 0.2080 | 0.0587 |

Character rate has an even clearer trend than words per second: denser utterances are harder.

### Loudness: RMS dB

| RMS dB bucket | Samples | Mean value | Mean WER | Mean CER |
|---|---:|---:|---:|---:|
| Low | 136 | -34.11 dB | 0.1833 | 0.0476 |
| Medium | 136 | -25.36 dB | 0.1842 | 0.0514 |
| High | 136 | -21.12 dB | 0.1490 | 0.0400 |

Louder samples are easier on average. This is plausible because very quiet recordings can reduce acoustic clarity.

### Silence ratio

| Silence-ratio bucket | Samples | Mean value | Mean WER | Mean CER |
|---|---:|---:|---:|---:|
| Low | 136 | 0.2226 | 0.1502 | 0.0385 |
| Medium | 136 | 0.3516 | 0.2135 | 0.0576 |
| High | 136 | 0.4655 | 0.1528 | 0.0427 |

The relationship with silence is not linear. Medium-silence samples are hardest, while high-silence samples are not necessarily worse. This may happen because silence ratio also interacts with duration, speech rate, speaker pacing, and dataset segmentation.

### SNR proxy

| SNR-proxy bucket | Samples | Mean value | Mean WER | Mean CER |
|---|---:|---:|---:|---:|
| Low | 136 | 45.47 dB | 0.1550 | 0.0439 |
| Medium | 136 | 58.26 dB | 0.1438 | 0.0353 |
| High | 136 | 96.20 dB | 0.2177 | 0.0597 |

The high SNR-proxy bucket has the highest WER. This does not necessarily mean that better SNR hurts ASR. In this script, `snr_proxy_db` is computed from the ratio between high and low RMS percentiles, so it is closer to a dynamic-range proxy than a true noise estimate. A high value may indicate uneven energy, long quiet regions, or unstable recording dynamics.

### Spectral centroid

| Spectral centroid bucket | Samples | Mean value | Mean WER | Mean CER |
|---|---:|---:|---:|---:|
| Low | 136 | 1378.06 | 0.2258 | 0.0700 |
| Medium | 136 | 1801.24 | 0.1492 | 0.0348 |
| High | 136 | 2232.76 | 0.1415 | 0.0341 |

Lower spectral centroid is associated with substantially higher WER. This may reflect muffled speech, lower-frequency recordings, microphone characteristics, or speaker/audio conditions.

### Zero-crossing rate

| Zero-crossing-rate bucket | Samples | Mean value | Mean WER | Mean CER |
|---|---:|---:|---:|---:|
| Low | 136 | 0.1024 | 0.2275 | 0.0705 |
| Medium | 136 | 0.1462 | 0.1339 | 0.0317 |
| High | 136 | 0.2023 | 0.1551 | 0.0367 |

The lowest zero-crossing-rate bucket is hardest. This is consistent with the spectral centroid result: lower-frequency or less spectrally active audio appears harder for the model.

---

## 18. Joint audio-cluster and sentence-cluster analysis

The file `error_by_audio_and_sentence_cluster.csv` combines acoustic grouping and sentence-level semantic grouping.

| Audio cluster | Sentence cluster | Samples | Mean WER | Mean CER |
|---:|---:|---:|---:|---:|
| 1 | 0 | 45 | 0.2255 | 0.0701 |
| 2 | 0 | 40 | 0.2040 | 0.0560 |
| 2 | 1 | 58 | 0.1841 | 0.0404 |
| 0 | 0 | 65 | 0.1687 | 0.0414 |
| 1 | 1 | 104 | 0.1597 | 0.0475 |
| 0 | 1 | 96 | 0.1425 | 0.0366 |

This is one of the most useful diagnostic tables.

Main conclusion:

- The hardest intersection is **audio cluster 1 + sentence cluster 0**, with WER **22.55%** and CER **7.01%**.
- The easiest intersection is **audio cluster 0 + sentence cluster 1**, with WER **14.25%** and CER **3.66%**.

This means the model errors are not explained by audio properties alone or text semantics alone. The worst cases appear when certain audio conditions and certain sentence types overlap.

---

## 19. Worst-sample analysis

The file `worst_samples.csv` and the audio-enriched version `worst_samples_with_audio_features.csv` show the most difficult utterances.

Examples from the worst set:

| Reference | Prediction | WER | CER | Notes |
|---|---|---:|---:|---|
| `elicopterul nu fusese anunțat` | `ericopterul nu puse să se anunță` | 1.25 | 0.276 | Multiple lexical substitutions and insertions. |
| `căutăm noi actori pentru a extinde baza` | `cotăm un actor pentru o acestini de bază` | 1.00 | 0.333 | Almost every content word is affected. |
| `aceste chestiuni țin de competența noastră` | `ar acestei chestiuni din lecompetența moastră` | 1.00 | 0.190 | Word-boundary and morphology errors. |
| `planeta ne transmite un semnal de alarmă` | `planeta a intratzmuit în semnul alarma` | 0.857 | 0.350 | Severe substitution around `transmite un semnal`. |
| `dubarbier cluj expediază primul șut pe poartă` | `dui barbieri cluji expediază primul șud pe poarta ea` | 0.857 | 0.200 | Proper names and sports phrase are difficult. |

Observed patterns:

- Proper names and less common words are difficult: e.g. `dubarbier`, `cluj`, `vasile cristea`.
- Morphologically rich words are often distorted: e.g. `competența`, `îndrumătoarea`, `supracapacitatea`.
- Some mistakes are diacritic or orthographic variants, but the worst cases are real lexical errors.
- Dense or acoustically challenging utterances often produce merged or split words, such as `îndetrimentul`, `delei`, `ungemiu`, or `lecompetența`.

---

## 20. Figures

The folder `evaluation/figures/` contains the main report-ready plots:

| Figure | Meaning |
|---|---|
| `baseline_vs_finetuned_wer_cer.png` | Direct comparison between baseline and fine-tuned WER/CER. |
| `audio_cluster_wer_cer.png` | WER/CER grouped by audio cluster. |
| `audio_cluster_duration_vs_wer.png` | Relation between cluster duration and WER. |
| `text_cluster_k3_wer_cer.png` | WER/CER grouped by k=3 stopword-based text cluster. |
| `text_cluster_k3_sizes.png` | Number of samples in each text cluster. |
| `dominant_error_distribution.png` | Distribution of dominant error types. |
| `top_worst_samples_by_wer.png` | Highest-WER samples. |
| `project_pipeline_diagram.png` | End-to-end project pipeline diagram. |

The folder `evaluation/audio_quality_analysis/figures/` contains audio-feature plots:

| Figure | Meaning |
|---|---|
| `audio_feature_correlations_with_wer.png` | Correlation ranking between audio features and WER. |
| `error_by_actual_duration_seconds_bucket.png` | Error by duration bucket. |
| `error_by_words_per_second_bucket.png` | Error by speech-rate bucket. |
| `error_by_rms_mean_db_bucket.png` | Error by loudness bucket. |
| `error_by_silence_ratio_bucket.png` | Error by silence-ratio bucket. |
| `error_by_snr_proxy_db_bucket.png` | Error by SNR-proxy/dynamic-range bucket. |
| `error_by_spectral_centroid_mean_bucket.png` | Error by spectral-centroid bucket. |
| `error_by_zero_crossing_rate_mean_bucket.png` | Error by zero-crossing-rate bucket. |
| `error_by_audio_cluster.png` | Error by audio cluster. |
| `error_by_sentence_cluster.png` | Error by sentence cluster. |



---

## 21. How to reproduce the important result files

### Evaluate final fine-tuned model

```bash
python src/evaluate_baseline.py \
  --model-name /content/drive/MyDrive/romanian-asr-results/whisper-small-ro-finetuned-2epochs \
  --test-csv data/processed/test.csv \
  --output-csv results/finetuned_predictions_2epochs_final.csv \
  --metrics-csv results/finetuned_metrics_2epochs_final.csv
```

### Evaluate checkpoint 400

```bash
python src/evaluate_baseline.py \
  --model-name /content/drive/MyDrive/romanian-asr-results/whisper-small-ro-finetuned-2epochs/checkpoint-400 \
  --processor-name /content/drive/MyDrive/romanian-asr-results/whisper-small-ro-finetuned-2epochs \
  --test-csv data/processed/test.csv \
  --output-csv results/finetuned_predictions_checkpoint_400.csv \
  --metrics-csv results/finetuned_metrics_checkpoint_400.csv
```

### Run basic error analysis

```bash
python src/analyze_asr_errors.py \
  --predictions-csv results/finetuned_predictions_2epochs_final.csv \
  --metadata-csv data/processed/metadata_with_clusters_and_split.csv \
  --output-dir results/error_analysis_finetuned_2epochs \
  --n-text-clusters 5
```

### Run stopword-based text-cluster analysis with k=3

```bash
python src/analyze_asr_errors_stopwords.py \
  --predictions-csv results/finetuned_predictions_2epochs_final.csv \
  --metadata-csv data/processed/metadata_with_clusters_and_split.csv \
  --output-dir results/error_analysis_finetuned_2epochs_stopwords_k3 \
  --n-text-clusters 3
```

### Run sentence-embedding clustering analysis

```bash
python src/analyze_asr_errors_sentence_embeddings.py \
  --predictions-csv results/finetuned_predictions_2epochs_final.csv \
  --metadata-csv data/processed/metadata_with_clusters_and_split.csv \
  --output-dir results/error_analysis_sentence_embeddings \
  --min-k 2 \
  --max-k 10
```

### Run audio-quality analysis

```bash
python src/analyze_audio_quality_errors.py \
  --predictions-csv results/finetuned_predictions_2epochs_final.csv \
  --metadata-csv data/processed/metadata_with_clusters_and_split.csv \
  --sentence-clusters-csv results/error_analysis_sentence_embeddings/sentence_cluster_assignments.csv \
  --output-dir results/audio_quality_analysis
```

---

## 22. Conclusions

A concise interpretation of our results:

> Fine-tuning Whisper-small on the prepared Romanian Common Voice subset substantially improved ASR performance on the held-out test set. The baseline model obtained 28.70% WER and 8.28% CER, while the 2-epoch fine-tuned model obtained 17.66% WER and 4.80% CER, corresponding to relative reductions of 38.47% in WER and 42.03% in CER. Per-sample analysis shows that exact transcriptions increased from 91 to 181 out of 408 test utterances. The remaining errors are dominated by substitutions rather than deletions or insertions, indicating that the model usually preserves the approximate utterance length but still confuses lexical or morphological forms. Error analysis by duration, audio cluster, sentence cluster, and audio quality features suggests that short utterances, long utterances, denser speech, and lower spectral-centroid recordings are more difficult, although individual low-level audio features only weakly correlate with WER.

---

## 23. Limitations

These results should be interpreted with the following limitations:

1. The test set contains 408 samples, about 0.5 hours of audio. This is enough for a project-level evaluation, but not enough for a production-grade ASR benchmark.
2. The dataset is from Common Voice and may not represent all Romanian accents, microphones, domains, or spontaneous speech conditions.
3. TF-IDF text clustering is exploratory. The low silhouette scores show that transcript clusters overlap significantly.
4. Audio feature correlations are not causal. They identify associations, not definitive reasons for model failure.
5. The SNR proxy is not a true SNR measurement. It is based on RMS percentile differences and should be described as a dynamic-range/noise proxy.
6. Some residual errors are affected by text normalization, especially diacritics and short function words.
