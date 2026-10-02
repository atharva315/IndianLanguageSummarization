\# Marathi Tokenizer Analysis



\## 1. Purpose



This analysis measures how efficiently different tokenizers represent Marathi text, with particular attention to Devanagari Marathi.



The analysis is a tokenizer-efficiency study. It does not measure or rank the overall summarization capability of the underlying models.



The analysis uses three distinct tokenizer families:



\- Gemma 4 E2B IT

\- Qwen3-4B

\- BLOOMZ-3B



Qwen3-1.7B was also checked during tokenizer compatibility testing. Its tokenizer configuration is the same tokenizer family used by Qwen3-4B, so it is not treated as a separate tokenizer in the fertility comparison.



\---



\## 2. Models and Tokenizers



| Reference model | Hugging Face ID | Tokenizer | Vocabulary size |

|---|---|---|---:|

| Gemma 4 E2B IT | `google/gemma-4-E2B-it` | `GemmaTokenizer` | 262,144 |

| Qwen3-4B | `Qwen/Qwen3-4B` | `Qwen2Tokenizer` | 151,643 |

| BLOOMZ-3B | `bigscience/bloomz-3b` | `TokenizersBackend` | 250,680 |



Tokenizer loading was verified successfully in the project Python environment.



Only tokenizer files were loaded during this analysis. Model weights were not loaded locally.



\---



\## 3. Diagnostic Dataset



A deterministic tokenizer diagnostic dataset containing 1,000 examples was created at:



`data/processed/marathi\_tokenizer\_diagnostic/tokenizer\_diagnostic.csv`



Composition:



| Category | Examples |

|---|---:|

| Native Devanagari | 500 |

| Native Latin | 200 |

| Romanized | 150 |

| Code-mixed Devanagari | 75 |

| Code-mixed Roman | 75 |

| \*\*Total\*\* | \*\*1,000\*\* |



The Romanized Marathi examples are controlled transliterations generated for diagnostic purposes. They are not naturally collected Roman Marathi user text.



The diagnostic dataset is analysis-only and is not used for model training.



\---



\## 4. Fertility Definition



Tokenizer fertility is defined as:



`fertility = tokenizer token count / whitespace-separated word count`



Tokenization was performed with:



`add\_special\_tokens=False`



Lower fertility means fewer tokenizer pieces per whitespace-separated word.



Fertility is a tokenizer segmentation measurement. Lower fertility does not by itself establish better summarization quality, better model reasoning, or overall model superiority.



\---



\## 5. Controlled Diagnostic Results



\### Native Devanagari



| Tokenizer | Mean fertility |

|---|---:|

| Gemma 4 E2B | 1.872 |

| Qwen3-4B | 6.828 |

| BLOOMZ-3B | 1.617 |



\### Native Latin



| Tokenizer | Mean fertility |

|---|---:|

| Gemma 4 E2B | 1.144 |

| Qwen3-4B | 1.144 |

| BLOOMZ-3B | 1.169 |



\### Romanized Marathi



| Tokenizer | Mean fertility |

|---|---:|

| Gemma 4 E2B | 3.772 |

| Qwen3-4B | 4.385 |

| BLOOMZ-3B | 3.867 |



\### Code-mixed Devanagari



| Tokenizer | Mean fertility |

|---|---:|

| Gemma 4 E2B | 1.849 |

| Qwen3-4B | 6.688 |

| BLOOMZ-3B | 1.601 |



\### Code-mixed Roman



| Tokenizer | Mean fertility |

|---|---:|

| Gemma 4 E2B | 3.670 |

| Qwen3-4B | 4.270 |

| BLOOMZ-3B | 3.762 |



\---



\## 6. Real Marathi Training-Corpus Measurement



To determine whether the diagnostic pattern also appears in the actual project data, fertility was measured over all 14,020 examples in:



`data/processed/marathi\_v1/04\_train\_sft.csv`



The SFT prompt structure was verified across all 14,020 rows.



Each prompt contains exactly one newline separator between the instruction and the source article.



The tokenizer measurement uses only the extracted Marathi source article, not the instruction wrapper and not the reference summary.



\### Corpus Results



| Tokenizer | N | Mean fertility | Median | P90 | P95 | Mean tokens | Mean words |

|---|---:|---:|---:|---:|---:|---:|---:|

| Gemma 4 E2B | 14,020 | 1.848 | 1.851 | 2.036 | 2.068 | 272.185 | 147.781 |

| Qwen3-4B | 14,020 | 6.821 | 6.841 | 7.094 | 7.430 | 1007.290 | 147.781 |

| BLOOMZ-3B | 14,020 | 1.584 | 1.594 | 1.726 | 1.761 | 233.780 | 147.781 |



The same 14,020 source examples were measured for all three tokenizers.



\---



\## 7. Interpretation



The real Marathi corpus shows a substantial tokenizer-segmentation difference between the three tokenizers.



Qwen3-4B produces substantially more tokenizer pieces per whitespace-separated word on this Marathi Devanagari corpus than Gemma 4 E2B and BLOOMZ-3B.



Gemma 4 E2B also has substantially lower fertility than Qwen3-4B on the Marathi corpus.



BLOOMZ-3B has the lowest measured fertility among the three tokenizers on this corpus.



The controlled diagnostic and the real-corpus measurement show the same broad pattern for Devanagari Marathi.



The Latin control category in the diagnostic has much smaller differences between the tokenizers, which provides a useful reference when interpreting the larger Devanagari differences.



These observations describe tokenizer segmentation only. They should not be interpreted as evidence that one underlying model will necessarily produce better Marathi summaries.



\---



\## 8. Generated Measurement Files



The following files contain the detailed measurements:



\- `gemma4\_e2b\_corpus\_fertility.csv`

\- `qwen3\_4b\_corpus\_fertility.csv`

\- `bloomz\_3b\_corpus\_fertility.csv`

\- `corpus\_fertility\_summary.csv`



All are stored under:



`data/processed/marathi\_tokenizer\_diagnostic/`



The per-example files contain the pair ID, source text, word count, token count, and fertility for the corresponding tokenizer.



\---



\## 9. Reproducibility



Measurement scripts:



\- `src/tokenizer/build\_diagnostic.py`

\- `src/tokenizer/measure\_fertility.py`

\- `src/tokenizer/measure\_corpus\_fertility.py`



The controlled diagnostic uses a deterministic construction.



The real-corpus measurement uses the frozen project training dataset version currently stored under:



`data/processed/marathi\_v1/`



No model weights are required for this analysis.



\---



\## 10. Limitations



1\. Tokenizer fertility is not a direct measure of summarization quality.

2\. Different models have different architectures, training data, objectives, and capabilities; fertility alone cannot be used to compare overall model strength.

3\. The Romanized Marathi diagnostic uses controlled transliteration rather than naturally occurring Roman Marathi.

4\. The corpus-level analysis uses the project's Marathi training corpus. It is therefore a measurement of tokenizer behavior on this dataset, not a universal measurement of all Marathi text.

5\. Vocabulary size alone does not determine tokenizer quality or fertility.

6\. The analysis does not establish that a lower-fertility tokenizer will necessarily produce better downstream summarization results.



\---



\## 11. Research Decision



Gemma 4 E2B IT remains the project's main model.



The additional tokenizers are used as reference points for tokenizer-efficiency analysis only.



The main summarization experiment remains:



`Gemma 4 E2B IT baseline`



versus



`Gemma 4 E2B IT + Marathi QLoRA`



Both will use the same frozen Marathi test set and identical evaluation conditions.



\---



\## 12. Status



Tokenizer compatibility verification: PASS



Controlled diagnostic fertility measurement: PASS



Real Marathi corpus fertility measurement: PASS



Documentation: COMPLETED

