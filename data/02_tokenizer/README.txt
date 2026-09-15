TOKENIZER FERTILITY DIAGNOSTIC — 1,000 EXAMPLES

Purpose
Measure tokenizer efficiency for Marathi in native Devanagari, Romanized Marathi,
and Marathi-English code-mixed text, with an English control.

Composition
500 native Marathi (Devanagari)
200 English (Latin; controlled diagnostic set)
150 Romanized Marathi (Latin; deterministic transliteration)
150 Marathi-English code-mixed (75 Devanagari+English + 75 Roman+English)

Use
This is a diagnostic dataset only. Do NOT train on it.

Native Marathi sampling
Sampled only from the TRAINING split of the curated corpus, not from the frozen test.
Selection favors domain, length, and numeric diversity and avoids unnecessary structural
template duplication.

English control
Controlled synthetic English sentences distributed across the same broad domain set.
They are NOT a benchmark, translation reference, or human-authored corpus. They exist to
measure tokenizer behavior under a controlled English condition.

Romanized Marathi
Generated with a deterministic Devanagari→Latin transliteration fallback. This gives a
controlled script comparison but may not match natural user spellings. Spot-check before
making claims about real-world Roman Marathi.

Code-mixed Marathi
Synthetic stress-test variants. 75 combine Devanagari Marathi with English tokens; 75 use
Romanized Marathi with injected English tokens. Diagnostic only.

Metric
word_count_whitespace = len(text.split())
fertility = tokenizer_token_count / word_count_whitespace

Run the included measure_fertility.py with:
  google/gemma-4-E2B-it
  Qwen/Qwen3-1.7B

IMPORTANT:
- use add_special_tokens=False
- tokenize only the raw text
- do not add the summarization prompt
- use the same script for both tokenizers
- save the resulting CSV as an immutable measurement artifact

Recommended outputs
mean, median, P90, P95 fertility
mean/median tokens per example
fertility by language variant
fertility by length bucket
fertility by domain
Marathi vs English fertility ratio
native vs Romanized vs code-mixed Marathi

Verified current model references:
Gemma 4 E2B-IT: https://huggingface.co/google/gemma-4-E2B-it
Qwen3 1.7B: https://huggingface.co/Qwen/Qwen3-1.7B

A 2026 ACL study explicitly investigates tokenizer metrics including fertility for
Hindi and Marathi:
https://aclanthology.org/2026.acl-long.1037/
