# Data Provenance, Domain Balance, and Template-Family Control

## Correction to the dataset description

The Marathi v1 dataset was **collected from external websites and article-summary resources**. It was not created synthetically from scratch. Sources known from the project notes include BBC Marathi content and the XL-Sum article-summary dataset, along with additional collected websites/resources. The collected material was then preprocessed and analysed.

The preparation procedure, as described by the project owner, was:

1. Collect Marathi article-summary pairs from multiple external sources/domains.
2. Preprocess the records: clean text, standardize Unicode/whitespace, remove invalid records and duplicates, and review article-summary alignment.
3. Use an AI tool to help identify recurring structural templates and categorize records by domain/template.
4. Balance the representation of template families to reduce overrepresentation of a few recurring patterns.
5. Shuffle the prepared rows in a reproducible manner and apply a template-aware train/validation/test split.
6. Keep the separate 100-example HumanEval set independent of model training.

**Important terminology:** using an AI tool to identify and label templates is not the same as generating the articles or summaries synthetically.

The exact complete inventory of every source URL, source dataset/version, collection date, licence/terms, and the AI tool/prompt used for template labelling should be retained in a provenance spreadsheet and completed before publication. Only sources verified in that inventory should be claimed as used in the final paper.

## Canonical dataset and split

Dataset version: `marathi_v1`

- Train: 14,020
- Validation: 796
- Frozen test: 794
- Total experimental pool: 15,610
- Template-family retention cap: 12
- Split method: template-aware

Frozen test file: `data/processed/marathi_v1/06_test_frozen.csv`

SHA-256:

`D93B3EE80E6AE306C1C1F02855A244AE2C38F46D587F3C39C8C415F0011A117D`

The numbers above describe the **completed experiment**. A newly designed project could choose another split ratio, but changing the official frozen test would create a different experiment and would not reproduce the current E0/E1 results.

## What is a template family?

A **domain** is the subject area, such as banking, education, politics, history, or environment.

A **template family** is a group of records that follow substantially the same structure or content pattern, even when the names, dates, amounts, or places differ. One domain can contain many different template families.

### Banking example

Suppose the source collection contains these 40 records:

- “Bank A changes its savings interest rate from 3% to 3.5%.”
- “Bank B changes its savings interest rate from 4% to 4.25%.”
- “Bank C changes its savings interest rate from 3.25% to 3.75%.”
- Many other records with the same announcement structure.

Although the banks and rates differ, the article pattern is very similar: **bank + interest-rate change + effective value/date**. The AI-assisted template analysis might group these records into one template family, subject to the project's labelling rules and human review.

A second banking family might describe a loan eligibility announcement. A third might describe a fraud alert. These are all banking records, but they are not necessarily the same template family.

### What does “maximum 12 retained examples per template family” mean?

It means that, after template families are identified, **no more than 12 retained records are taken from any single family** under this cap. It does NOT mean there are only 12 families in the entire dataset, and it does NOT mean only 12 banking records can be retained. Banking may have dozens of families, each with its own cap.

If a particular family has 40 similar records, at most 12 of them are retained from that family. The remaining records may be excluded or handled according to the documented curation policy. The selection should not be based on which rows make the model's score look better.

The reason for the cap is to prevent a few repeated structures from dominating training and to reduce the chance that evaluation rewards template memorization.

## Balancing, shuffling, and splitting do different jobs

- **Domain/template balancing** reduces excessive representation of a small number of domains or repeated structures. It helps reduce one source of imbalance; it cannot guarantee that all bias is eliminated.
- **Shuffling** changes row order so training is not accidentally organized by source, domain, or template. Shuffling must use a recorded seed when reproducibility matters.
- **Template-aware splitting** keeps examples from the same or closely related template family from being distributed across partitions in a way that makes validation/test artificially easy. The exact grouping/split code and family labels should be kept with the experiment artifacts.
- **Deduplication** detects identical or near-identical text. It complements, but does not replace, template analysis.

Simply shuffling all rows and randomly splitting them is not a substitute for template-aware splitting.

## Source pairs versus article-only websites

A dataset row needs both the source article and a trustworthy target summary. Many websites publish full articles but do not provide a separately written summary. A headline, search snippet, article lead, abstract, and executive summary are not automatically interchangeable; each has a different purpose and may omit or introduce information.

For each source, document how the target summary was obtained:

- paired article-summary dataset released by its authors;
- summary or abstract published with that source document;
- editor-written standfirst/description;
- human-written summary;
- AI-generated candidate summary that was reviewed and approved.

Do not label AI-generated summaries as human-written gold summaries. If AI-generated targets are used for training, record the model, prompt, date, validation procedure, and proportion of such targets.

## Example source register

The following are **source candidates for future collection or provenance verification**, not a claim that every item below was used in `marathi_v1`. Language and licence must be checked before inclusion. English-only data must not be silently mixed into a Marathi dataset as if it were native Marathi data.

| Domain | Candidate source/resource | What can be paired | Marathi suitability / caution |
|---|---|---|---|
| News | [BBC Marathi / XL-Sum](https://github.com/csebuetnlp/xl-sum) | Marathi BBC articles with reference summaries in XL-Sum | Directly relevant; verify the dataset release and licence. |
| Marathi news | [L3Cube-MahaSum / MarathiNLP](https://github.com/l3cube-pune/MarathiNLP) | Marathi news articles with abstract summaries as described by the dataset paper | Directly relevant; verify the exact version, access terms, and licence. |
| General news | [CNN/DailyMail dataset](https://huggingface.co/datasets/abisee/cnn_dailymail) | English news articles and highlight summaries | English; not direct Marathi training data. |
| Single-document news | [XSum dataset](https://huggingface.co/datasets/EdinburghNLP/xsum) | English BBC articles with concise reference summaries; XL-Sum is a separate multilingual dataset that includes Marathi | Use XSum only for English experiments; use the Marathi XL-Sum configuration for Marathi. |
| Multi-document news | [Multi-News dataset](https://huggingface.co/datasets/alexfabbri/multi_news) | Multiple English articles paired with a reference summary | English; a different multi-document task. |
| Law / legislation | [BillSum / GovInfo Bill Summaries](https://www.govinfo.gov/bulkdata/BILLSUM) | U.S. bill text with legislative summaries | English and U.S.-specific; not a direct source of Marathi legal pairs. |
| Government reports | [GovReport dataset](https://huggingface.co/datasets/launch/gov_report) | Government reports paired with expert-written summaries | English; check licence and long-document handling. |
| Scientific research | [arXiv papers](https://arxiv.org/) | Paper body paired with the paper's abstract | Often English. Abstracts are author-written, but are not always equivalent to task-specific summaries. |
| Biomedical research | [PubMed](https://pubmed.ncbi.nlm.nih.gov/) | Articles/abstracts and biomedical abstracts | Check access, copyright, and full-text availability; do not assume abstracts license the article body. |
| Patents / technology | [BIGPATENT dataset](https://aclanthology.org/P19-1212/) | Patent description paired with its human-written abstract | English/U.S. patent domain; abstracts have a particular technical style. |
| Agriculture / food systems | [FAO publications](https://www.fao.org/publications/en) | Some long reports contain executive summaries | A summary is available only for qualifying documents; verify it corresponds to the source text. |
| Banking / finance | [Reserve Bank of India reports](https://rbi.org.in/scripts/annualreportpublications.aspx) | Some reports contain overviews, chapter reviews, or executive-style summaries | Primarily English on these pages; not every report provides a directly paired short summary. |

### Source-selection rule

For the current Marathi experiment, prefer source-summary pairs that are genuinely in Marathi and whose reuse conditions allow the intended research. Other-language resources above are useful examples of how summarization datasets are constructed; they are not automatically appropriate for this Marathi training corpus. If either the article or summary is translated, document the translation process and check both sides for meaning preservation.

## Data provenance fields to retain

For every source record, maintain at least:

`id, source_name, source_url_or_dataset_id, source_version, collection_date, licence_or_terms, domain, template_family_id, article, reference_summary, preprocessing_status, exclusion_reason`

Keep raw-source metadata separately from the final model-ready columns `ID | article | summary` so that the training files remain simple while the collection can still be audited.

## Interpretation boundary

Template-aware curation reduces repetition and structural leakage; it does not prove that all sources are unbiased, all summaries are factually correct, or results generalize to every form of real-world Marathi. Continue to report these limitations clearly.
