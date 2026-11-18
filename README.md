# tre-ocr

A [ternlabs](https://ternlabs.io) project.

OCR for Vietnamese documents, with the accuracy measured, not assumed.

Vietnamese is hard for OCR: small tone marks (ả, ạ, ặ, ữ) change the meaning, and
official documents carry signatures and stamps over the text. Models exist; a public
measurement of how well they read real Vietnamese business documents does not.

Status: v0.2.0, the first public release. Early work: one profile (scanned stock-exchange
disclosures) has passed; a second (financial statements) is measured and has not. See
[METHOD.md](METHOD.md) for how, and its limits. Developed for
our own use: issues are closed for now, and no support or schedule is promised.

Plan:
1. A benchmark of scanned Vietnamese disclosures, using bilingual filings as free
   ground truth, scored on numbers and tone marks separately.
2. A tool that picks an engine, checks the numbers, and abstains when unsure.

## Use

```
pip install -r requirements-reader.txt   # in its own environment; needs poppler-utils
bash tools/get_weights.sh                # the tuned recogniser, hash-checked
python -I tools/read_pdf.py some.pdf --out cache/drafts
```

On its first run PaddleOCR downloads its own stock models; `--device cpu` works, slowly.
Every number the tool is not sure of is left empty (`[?]`), never guessed. Financial
statements were measured on two test sets and have not passed (METHOD.md, "Financial
statements"): `tools/read.py` refuses them, and statement output from `read_pdf.py` is
unchecked.

## Weights

The tuned recogniser (PaddleOCR PP-OCRv6 medium, fine-tuned) is kept in its own
repository, so a clone stays small; `tools/get_weights.sh` fetches it. It was trained
only on typed public filings made to look scanned, with every test-set file and every
test-set company left out (METHOD.md, "Fine-tuning").

## Licence

- Code and weights: [Apache License 2.0](LICENSE).
- Label sets in `sets/`: [CC-BY-4.0](sets/LICENSE).
- The filings the sets point to belong to their publishers and are not covered.

The tool and its checks are open and stay open. ternlabs may offer paid versions later,
such as a hosted service, more statement forms or support; what is released here keeps its
licence.
