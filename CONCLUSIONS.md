# Results and Error Analysis

The Romanian ASR system was evaluated after fine-tuning `openai/whisper-small` for two epochs on a prepared Common Voice Romanian subset. The final model was evaluated on a held-out test set containing 408 samples. The model achieved 17.66% Word Error Rate (WER) and 4.80% Character Error Rate (CER) on the test set. On the validation set, after two training epochs, the model obtained 18.35% WER, 5.44% CER, and an evaluation loss of 0.2474. These values are close to the final test results, suggesting that the model generalizes reasonably well and does not show a large validation-test mismatch.

A comparison between the intermediate checkpoint and the final model shows a small but consistent improvement. The checkpoint at step 400 achieved 17.89% WER and 4.87% CER, while the final model after two epochs achieved 17.66% WER and 4.80% CER. The improvement is not large, but it indicates that the second epoch still helped the model slightly. A small smoke test on only 5 samples produced 37.14% WER and 10.24% CER, but this result should not be interpreted as representative because it was used only to verify that the evaluation pipeline worked correctly.

| Evaluation run | Samples | WER | CER | Observation |
|---|---:|---:|---:|---|
| Validation after 2 epochs | validation set | 18.35% | 5.44% | Stable validation performance |
| Final fine-tuned model | 408 | 17.66% | 4.80% | Best tested model |
| Checkpoint 400 | 408 | 17.89% | 4.87% | Slightly worse than final model |
| Smoke test | 5 | 37.14% | 10.24% | Pipeline sanity check only |

The final model therefore provides a solid Romanian ASR baseline. It produces generally readable transcriptions, but the detailed error analysis shows that errors are not uniformly distributed across the dataset. Some groups of utterances are significantly harder than others, depending on sentence content, audio characteristics, utterance length, and lexical difficulty.

## Word-Level Error Patterns

The most common error type was word substitution. This means that the model usually detects that speech is present and produces a Romanian-like transcription, but it often replaces the correct word with a phonetically similar or contextually plausible alternative. Deletions and insertions were less frequent than substitutions.

This is visible in the average edit counts. For example, in the hardest audio cluster, the model produced on average 1.21 substitutions, 0.10 deletions, and 0.21 insertions per sample. This confirms that the dominant failure mode is not silence detection or complete omission, but incorrect lexical selection.

Several typical error patterns were observed:

| Reference | Prediction | WER | CER | Error interpretation |
|---|---|---:|---:|---|
| elicopterul nu fusese anunțat | ericopterul nu puse să se anunță | 125.00% | 27.59% | Strong phonetic confusion and grammatical hallucination |
| căutăm noi actori pentru a extinde baza | cotăm un actor pentru o acestini de bază | 100.00% | 33.33% | Multiple substitutions; semantic meaning mostly lost |
| aceste chestiuni țin de competența noastră | ar acestei chestiuni din lecompetența moastră | 100.00% | 19.05% | Word boundary and morphology errors |
| planeta ne transmite un semnal de alarmă | planeta a intratzmuit în semnul alarma | 85.71% | 35.00% | Severe word substitution around “transmite un semnal” |
| iti promit ca vei primi dreptate | eți promit că vă-i primit dreptată | 83.33% | 21.88% | Diacritics, inflection, and verb-form confusion |
| iti poruncesc sa bei niste apa | iti porunce să bein niște apă | 83.33% | 20.00% | Partial recognition but incorrect word endings |
| legislația actuală în vigoare permite derogări | regislația actuală învigua re permite de rogări | 83.33% | 10.87% | Legal/formal vocabulary and word splitting errors |
| profesoara de engleză a dat două sute cincizeci de lei | profesora de embrez a dat 250 delei | 70.00% | 46.30% | Rare word confusion plus number normalization issue |
| a mers acolo vineri ne-a spus directorul grădinii vasile cristea | a merg să acolo vine în aspectul directorul glădinii vasile cristă | 70.00% | 25.00% | Long sentence, proper names, and accumulated substitutions |

These examples show that the model often remains acoustically close to the target but fails at the exact lexical level. Some errors are phonetic: `elicopterul` becomes `ericopterul`, `grădinii` becomes `glădinii`, and `șut` becomes `șud`. Other errors are caused by morphology or Romanian inflection, such as `primi dreptate` becoming `primit dreptată`. The model also struggles with word boundaries, producing outputs such as `învigua re`, `de rogări`, or `delei`.

One important case is the sentence:

```text
profesoara de engleză a dat două sute cincizeci de lei
```

predicted as:

```text
profesora de embrez a dat 250 delei
```

This example contains both a real ASR error and an evaluation artifact. The model incorrectly recognized `engleză` as `embrez`, but it also predicted `250` instead of `două sute cincizeci`. Semantically, the number is correct, but the WER metric penalizes it heavily because the reference writes the number as words while the prediction uses digits. This suggests that future evaluation could include Romanian text normalization for numbers before computing WER.

## Error Analysis by Audio Cluster

The dataset was also analyzed using audio-based clusters. These clusters were obtained from acoustic embeddings and group together samples with similar audio characteristics. The test set was distributed into three audio clusters:

| Audio cluster | Samples | Mean duration | Mean WER | Mean CER | Mean substitutions | Mean deletions | Mean insertions |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 98 | 5.52 s | 19.22% | 4.68% | 1.21 | 0.10 | 0.21 |
| 1 | 149 | 3.98 s | 17.96% | 5.43% | 1.07 | 0.13 | 0.10 |
| 0 | 161 | 4.12 s | 15.31% | 3.86% | 0.88 | 0.09 | 0.14 |

Audio cluster 2 was the hardest group, with 19.22% WER, while audio cluster 0 was the easiest, with 15.31% WER. This is a difference of almost 4 percentage points in WER between the easiest and hardest acoustic groups.

The hardest audio cluster also had the longest average duration, approximately 5.52 seconds, compared with 3.98 seconds and 4.12 seconds for the other clusters. This suggests that part of the difficulty may come from longer utterances, where there are more opportunities for word-level errors to accumulate. It also had the highest number of insertions, suggesting that the model sometimes adds extra words when the acoustic signal or sentence flow is harder to decode.

However, the differences between clusters are not explained by duration alone. Audio cluster 1 had a shorter average duration than cluster 0, but still had a higher WER. This indicates that cluster-level difficulty likely depends on a combination of speaker characteristics, signal quality, pronunciation, speaking rhythm, and sentence content.

## Error Analysis by Sentence Embedding Clusters

Textual and semantic difficulty was analyzed using multilingual sentence embeddings. A k-selection procedure was applied using silhouette score, clustering stability, minimum cluster size, and balance ratio. The best practical value was k = 2, with a silhouette score of 0.1123, stability ARI of 0.9981, and a balance ratio of 1.60. This made the sentence embedding clustering much more stable and interpretable than the TF-IDF clustering variants.

The final sentence-cluster error analysis showed the following:

| Sentence cluster | Samples | Mean WER | Mean CER | Mean substitutions | Mean deletions | Mean insertions |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 150 | 19.52% | 5.39% | 1.13 | 0.09 | 0.23 |
| 1 | 258 | 15.88% | 4.19% | 0.97 | 0.12 | 0.09 |

Sentence cluster 0 was clearly harder, with 19.52% WER, compared with 15.88% WER for sentence cluster 1. This shows that model performance is affected not only by audio quality, but also by the type of sentence being transcribed.

The harder sentence cluster contained examples and top terms such as:

```text
există, doar, trebuie, problemă, cred, putem, urmare, pot, situație, însă, totuși, fără
```

Representative examples from this cluster included:

| Representative sentence from harder cluster |
|---|
| considerăm că aici există o lipsă de control |
| nu este o soluție de moment |
| ei bine este corect însă nu suficient |
| situația nu stă însă așa dimpotrivă |
| din nefericire acest lucru este departe de adevăr |

These sentences are short to medium-length, but they contain abstract or argumentative language. They resemble political, institutional, or formal discourse. The model appears to have more difficulty with this kind of language, especially when the words are semantically abstract and acoustically similar to other Romanian expressions.

The easier sentence cluster contained top terms such as:

```text
trebuie, avem, nevoie, cred, multe, important, două, bun, raport, mulțumesc
```

Representative examples included:

| Representative sentence from easier cluster |
|---|
| este important să subliniem acest lucru |
| este important să ținem cont de acest lucru |
| trebuie să ținem cont de acest lucru |
| să ne folosim de acest aspect |
| haideți așadar să acționăm în consecință |

Although this cluster also contains formal language, many sentences follow common Romanian discourse patterns such as `este important să`, `trebuie să`, or `avem nevoie de`. These recurring structures may be easier for the model because they appear more frequently in the training data and provide stronger language-model context.

## TF-IDF Text Clustering versus Sentence Embeddings

A first version of text clustering used TF-IDF features without removing Romanian stopwords. This produced clusters dominated by frequent function words such as `este`, `de`, `nu`, `să`, `aceasta`, and `acest lucru`. Although the clusters had different WER values, they were not very semantically meaningful because the top terms mostly reflected common grammatical words rather than content.

After adding Romanian stopwords and removing demonstrative/filler words, the TF-IDF clusters became more interpretable. For example, with k = 3, the clusters were:

| TF-IDF stopword cluster | Samples | Mean WER | Mean CER | Main terms |
|---:|---:|---:|---:|---|
| 0 | 349 | 17.73% | 4.80% | acesta, avem, cred, există, acum, nevoie, raport |
| 1 | 30 | 14.43% | 3.60% | această, privință, situație, problemă, propunere |
| 2 | 29 | 13.94% | 3.66% | trebuie, acționăm, facem, fim, punem |

However, this TF-IDF clustering was highly imbalanced: one cluster contained 349 out of 408 test samples, while the other two clusters contained only 30 and 29 samples. Because of this imbalance, the TF-IDF clusters were less useful for robust error analysis.

Sentence embeddings produced a more balanced and stable split: 150 samples in the harder cluster and 258 samples in the easier cluster. This made the sentence-embedding analysis more useful for explaining semantic difficulty.

## Error Analysis by Duration

Utterance duration was another important factor. The samples were grouped into three duration buckets:

| Duration bucket | Samples | Mean duration | Mean WER | Mean CER | Mean substitutions | Mean deletions | Mean insertions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short: < 3 s | 12 | 2.76 s | 26.45% | 7.37% | 1.08 | 0.00 | 0.33 |
| Medium: 3–6 s | 373 | 4.30 s | 16.71% | 4.42% | 0.99 | 0.12 | 0.13 |
| Long: > 6 s | 23 | 6.89 s | 20.66% | 6.61% | 1.65 | 0.04 | 0.17 |

The highest WER was observed for very short utterances, under 3 seconds, with 26.45% WER. This suggests that very short clips may not provide enough acoustic context for the model. In short utterances, even one wrong word has a large effect on WER because the denominator is small.

Long utterances above 6 seconds also had a higher WER of 20.66%. In this case, the problem is different: longer utterances contain more words, and therefore more opportunities for substitutions to accumulate. The long bucket had the highest average number of substitutions, with 1.65 substitutions per sample.

The best performance was obtained for medium-length utterances between 3 and 6 seconds, with 16.71% WER and 4.42% CER. This suggests that medium-length speech segments provide enough context without becoming too long and error-prone.

## Audio Quality Feature Analysis

To investigate whether errors are related to audio signal properties, several low-level audio features were extracted for each test sample:

- RMS amplitude and RMS decibel statistics;
- silence ratio;
- leading and trailing silence;
- proxy SNR / dynamic range;
- spectral centroid;
- spectral bandwidth;
- spectral rolloff;
- zero-crossing rate;
- MFCC means and standard deviations;
- words per second and characters per second.

The overall test set had:

| Audio statistic | Value |
|---|---:|
| Samples analyzed | 408 |
| Skipped samples | 0 |
| Mean duration | 4.40 s |
| Mean words per second | 1.69 |
| Mean RMS level | -26.86 dB |
| Mean silence ratio | 34.65% |
| Mean SNR proxy | 66.64 dB |

The SNR value here should be interpreted carefully. It is a proxy based on RMS percentile differences, not a calibrated physical microphone SNR. It is useful for relative comparison between clips, but it should not be interpreted as an absolute acoustic measurement.

The strongest correlations between audio features and WER were weak:

| Feature | Spearman correlation with WER | Interpretation |
|---|---:|---|
| Spectral centroid mean | -0.197 | Slight tendency: lower spectral centroid may be associated with higher WER |
| Spectral rolloff mean | -0.195 | Slight tendency: less high-frequency energy may be associated with higher WER |
| MFCC 2 std | 0.194 | Slight relation with spectral envelope variability |
| Spectral bandwidth mean | -0.186 | Slight tendency: narrower spectral content may be harder |
| Zero-crossing rate mean | -0.178 | Slight tendency: lower ZCR may be associated with higher WER |
| Characters per second | 0.156 | Faster character rate may slightly increase errors |

These correlations are not strong enough to claim that a single audio feature explains the errors. Instead, they suggest that model errors are multifactorial. Acoustic properties influence performance, but they interact with sentence content, duration, vocabulary, speaking style, and possibly speaker-specific characteristics.

The negative correlation between WER and spectral centroid / rolloff suggests that samples with less high-frequency information may be slightly harder. In practical terms, this could correspond to speech that is more muffled, less clear, or recorded with poorer high-frequency detail. However, because the absolute correlation is below 0.20, this should be treated as a weak trend rather than a decisive cause.

## Combined Audio and Semantic Analysis

The most informative result comes from combining audio clusters and sentence clusters. This allows us to see whether errors are mainly caused by audio, mainly caused by sentence content, or by their interaction.

| Audio cluster | Sentence cluster | Samples | Mean WER | Mean CER | Interpretation |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | 45 | 22.55% | 7.01% | Hardest combination |
| 2 | 0 | 40 | 20.40% | 5.60% | Hard semantic cluster + harder audio group |
| 2 | 1 | 58 | 18.41% | 4.04% | Harder audio, easier semantic content |
| 0 | 0 | 65 | 16.87% | 4.14% | Easier audio but harder semantic content |
| 1 | 1 | 104 | 15.97% | 4.75% | Moderate |
| 0 | 1 | 96 | 14.25% | 3.66% | Easiest combination |

The hardest group was audio cluster 1 + sentence cluster 0, with 22.55% WER and 7.01% CER. The easiest group was audio cluster 0 + sentence cluster 1, with only 14.25% WER and 3.66% CER. This is a large difference of more than 8 percentage points in WER.

This combined analysis is important because it shows that neither audio nor text alone fully explains the model behavior. The model performs worst when difficult acoustic conditions and difficult sentence content appear together. Conversely, the best performance occurs when the audio belongs to an easier acoustic cluster and the sentence belongs to the easier semantic cluster.

This supports the idea that ASR performance should be analyzed across multiple dimensions, not only using an overall WER value.

## Interpretation of Difficult Examples

The worst samples reveal several concrete failure categories.

### 1. Rare or domain-specific vocabulary

Formal and institutional vocabulary often caused substitutions:

| Reference | Prediction |
|---|---|
| legislația actuală în vigoare permite derogări | regislația actuală învigua re permite de rogări |
| mecanismele sofisticate nu creează valoare și bogăție | mecanismele s-au festicat de nu crează valoare și văgăție |
| sprijinirea celor bogați în detrimentul celor săraci | spriginirea celor bugat îndetrimentul celor sărați |

These examples suggest that the model struggles with formal vocabulary, inflected forms, and less frequent Romanian words.

### 2. Proper nouns and named entities

Proper nouns or place names were also difficult:

| Reference | Prediction |
|---|---|
| dubarbier cluj expediază primul șut pe poartă | dui barbieri cluji expediază primul șud pe poarta ea |
| directorul grădinii vasile cristea | directorul glădinii vasile cristă |

The model captures parts of the acoustic signal but often approximates names or uncommon words.

### 3. Word boundary errors

The model sometimes splits or merges words incorrectly:

| Reference | Prediction |
|---|---|
| în vigoare | învigua re |
| derogări | de rogări |
| de lei | delei |
| în detrimentul | îndetrimentul |

These errors are important because they may not always destroy the acoustic similarity, but they strongly affect WER and readability.

### 4. Romanian diacritics and morphology

The model sometimes produces words that are close but morphologically wrong:

| Reference | Prediction |
|---|---|
| dreptate | dreptată |
| noastră | moastră |
| bogăție | văgăție |
| săraci | sărați |
| creșterea | greșterea |

These mistakes show that the model often approximates the sound but does not always recover the correct Romanian lexical form.

### 5. Numeric normalization

The sentence:

```text
profesoara de engleză a dat două sute cincizeci de lei
```

was transcribed as:

```text
profesora de embrez a dat 250 delei
```

This contains a real ASR error, but also a text-normalization issue. The numeric meaning of 250 is correct, but the reference uses words. Without number normalization, WER penalizes this heavily. This suggests that future evaluation should normalize numbers and possibly standardize common Romanian orthographic variants before computing final metrics.

## Main Takeaways

The fine-tuned Whisper-small model achieved 17.66% WER and 4.80% CER on the Romanian test set. This is a strong baseline for a relatively compact fine-tuning experiment. The validation result of 18.35% WER was close to the test result, which suggests stable generalization.

The model’s dominant error type was substitution, not deletion. This means that the model usually produces speech-like Romanian output, but it often selects the wrong word among acoustically or contextually similar alternatives.

The error analysis showed that model performance depends on several interacting factors:

- Sentence content: abstract, formal, institutional, and less frequent vocabulary produced more errors.
- Audio characteristics: different acoustic clusters had different WER values, from 15.31% to 19.22%.
- Utterance duration: short utterances under 3 seconds were the hardest, with 26.45% WER.
- Semantic difficulty: the harder sentence embedding cluster reached 19.52% WER, while the easier one achieved 15.88% WER.
- Combined effects: the hardest audio-semantic combination reached 22.55% WER, while the easiest achieved 14.25% WER.

The audio feature analysis showed that no single acoustic metric strongly explains WER. The strongest correlations were weak, around 0.18–0.20 in absolute Spearman correlation. This means that ASR errors are not caused by one simple factor such as volume, silence ratio, or spectral centroid. Instead, they emerge from a combination of acoustic quality, speaking rate, duration, lexical difficulty, and semantic complexity.

Overall, the model is usable as a Romanian ASR baseline, but future improvements should focus on:

- adding more Romanian speech data with diverse speakers and recording conditions;
- increasing coverage of formal, legal, institutional, and domain-specific vocabulary;
- applying data augmentation for noisy, short, or acoustically weak clips;
- using text normalization for numbers, diacritics, and common orthographic variants during evaluation;
- comparing full fine-tuning with parameter-efficient approaches such as LoRA.
