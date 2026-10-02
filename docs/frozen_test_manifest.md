\# Frozen Marathi Test Set Manifest



\## Dataset Version



`marathi\_v1`



\## Frozen Evaluation File



`data/processed/marathi\_v1/06\_test\_frozen.csv`



\## Integrity



\- SHA-256:

&#x20; `D93B3EE80E6AE306C1C1F02855A244AE2C38F46D587F3C39C8C415F0011A117D`

\- Row count: `794`

\- Unique `pair\_id`: `794`



\## Columns



\- `pair\_id`

\- `source\_file`

\- `source\_row`

\- `template\_id`

\- `text\_chars`

\- `summary\_chars`

\- `compression\_ratio`

\- `length\_bucket`

\- `text`

\- `reference\_summary`



\## Usage



This file is the frozen Marathi evaluation set for the research experiments.



It is used for:



\- Marathi Gemma baseline evaluation (`MR-BM-001`)

\- Marathi QLoRA final evaluation (`MR-FT-001`)

\- automatic metric evaluation

\- final comparison between baseline and QLoRA

\- selection of examples for human evaluation



The model receives only the `text` field during prediction.



The `reference\_summary` field is used only after prediction generation

for evaluation.



\## Immutability Rule



`06\_test\_frozen.csv` must not be modified after baseline results are

generated.



Any change to the frozen test file requires creation of a new dataset

version and a new SHA-256 fingerprint.



\## Dataset Source



This frozen test set belongs to the validated `marathi\_v1` dataset package.

The package was produced using the documented cleaning, normalized

deduplication, structural QA, template-family analysis, template cap,

and template-aware split procedure.

