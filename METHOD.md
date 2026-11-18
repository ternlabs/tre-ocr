# Method

This document explains how OCR engines were evaluated on scanned Vietnamese documents,
how the pass marks were set, and how the tool built on the results uses the candidate
that passed. It is written for anyone who wants to know not only what the tool does, but
what testing has and has not established. Code comments refer to this document by step
number and section name.

What this repository provides:

1. **A method** for measuring OCR engines on Vietnamese stock-exchange disclosures, with
   the sets, the scoring code and the results.
2. **A tool** that reads disclosures with the candidate that passed, and leaves a number
   empty when it is not sure of it.
3. **A method for financial statements**, along the same lines, whose candidate has not
   passed yet; the tool refuses statements until one does.

Every pass mark below was fixed before its test ran, and each test set was scored once.
A set that has been looked at while choosing a change is never used to confirm that
change; a newly drawn set must be used instead.

How it is laid out: **What is measured** and **Steps** give the method; **Results** gives
the scores for disclosures; **The tool** describes what was built on them; **Financial
statements** gives the second profile's method and scores; **Limits** and **The sets**
close.

## What is measured

The goal is to measure how accurately OCR engines read **scanned Vietnamese documents**,
one **profile** (use case) at a time; to build a tool for a profile only if an engine meets
pass marks fixed in advance; and to have that tool **abstain** when it is not sure.

The first profile, and the only one that has passed, is **stock-exchange disclosures**:
resolutions, notices, explanations and similar filings by companies listed on HOSE and
HNX. The second, **financial statements** (mostly tables), is measured but has not
passed; see [Financial statements](#financial-statements-profile-2).

### Concepts

- **Truth.** The reference a reading is scored against (ground truth): a text layer, a
  twin's figures, or what a person typed from the scan.
- **Engine, reader, reading.** An engine is an OCR system under evaluation; a reader is
  one engine with its settings and weights; a reading is the output of a run.
- **Candidate.** The exact readers, settings and rules fixed before a test. Nothing in it
  changes between fixing it and scoring it.
- **Page, not file.** A PDF can join a scanned part and a typed part, so every label is
  per page: text layer, stamp, paper, skip.
- **Scan or text layer.** After the signature box is removed, fewer than 200 extracted
  characters on a page means a scan.
- **The signature box.** A digitally signed PDF carries a box of text (signer, time,
  "DN:" line, tax code). Left in, it makes a scan count as text and adds numbers that
  are not in the document. It is removed before a page is classified or truth is taken.
- **The heading.** A Vietnamese official document opens with two columns: company and
  document number ("Số: …") on the left; national title, motto and "place, ngày … tháng
  … năm …" on the right. Engines read these short lines in different orders. Company,
  document number, place and date are scored as **fields**, each right or wrong whole.
- **VI/EN twins.** Many disclosures are filed in Vietnamese and in English. When one is
  a scan and the other has a text layer, the text layer supplies the truth for the
  scan's **numbers** without anyone typing it, never for its tone marks.
- **Scores, never blended into one:**
  - **number recall:** the share of numbers in the truth (digit strings of 3+ digits,
    separators removed) found in the output;
  - **base-letter error rate:** character error rate with all tone marks removed;
  - **mark accuracy:** of words with the right base letters, the share with the right
    marks too;
  - **heading fields:** the share exactly right, scored without marks for the company.

  Before scoring, all runs of whitespace become one space; lines are matched in any
  order, then scored inside each match. A wrong tone mark can change a Vietnamese word,
  but the intended word can often be inferred from context; a wrong digit gives a
  different figure, and nothing around it shows the error. The two errors cost
  differently, so they are never reported as one accuracy.
- **Abstain.** Each line is marked sure or unsure, and every number on an unsure line is
  left empty. Measured by **wrong when sure** (the share of numbers on sure lines that
  are wrong) and **coverage** (the share of numbers on sure lines).
- **Random sampling.** Companies are drawn with a fixed seed from every stock on HOSE and
  HNX, with a cap per company, and filings are sorted by kind. Sets are never taken from
  someone's own collection: what someone collected shows what they follow.

## Steps

0. **Survey.** Seed 1, 30 companies, 2,087 PDF files. Of the disclosures filed in both
   languages, most pairs are scans on both sides; the signature box makes many scan
   pages look like text until it is removed.
1. **Page labeller** (`tools/label_pages.py`). Per page: remove the signature box, then
   text layer or not, stamp, national title, paper, and skip (member lists,
   signature-box-only pages, blank pages). Skipped pages keep their reason but are never
   scored.
2. **Sort filings by kind** (`tools/sort_kinds.py`). The title, lower-cased and without
   accents, is matched against phrases per kind; the earliest phrase wins, and when two
   phrases start at the same place, the longer one wins.
3. **Set A, numbers** (`tools/grow_set_a.py`). 100 scan pages with a text-layer English
   twin, at most 5 per company. Insider trades and personnel notices are left out.
4. **Number truth for A** (`tools/number_truth.py`, `tools/align_pages.py`). Digit
   strings from the English page, matched to the Vietnamese page they belong to. Spans
   were found with EasyOCR's help, so an engine close to EasyOCR may share its blind
   spots at page edges.
5. **Set B, tone marks** (`tools/grow_set_b.py`, `tools/truth_b.py`). 101 typed
   Vietnamese pages from 50 companies, rendered to images; the text layer is exact truth.
   A text layer that fails a syllable check (`tools/syllables.py`) is treated as broken
   and not used. B's images are cleaner than real scans.
6. **Set C, real scans checked by a person** (`tools/grow_set_c.py`, `tools/draft_c.py`,
   `tools/correct_c.py`). 30 scan pages drawn, 20 kept, at least 10 companies; drafted,
   then corrected by a person against the scan.
7. **Scoring** (`tools/score.py`). The four scores above, each on its own.
8. **Engines** (`tools/pages.py`, `tools/run_engine.py`). Tesseract (vie), VietOCR,
   PaddleOCR PP-OCRv6, PaddleOCR-VL 1.6, SenOCR-Vi and Qwen2.5-VL 7B, versions and
   settings recorded with each run.
9. **Decide.** Pass marks, fixed before step 8 ran: number recall at least 98% on A and
   C; wrong when sure at most 1% with coverage at least 85%; the company field right on
   at least 90% of pages. Heading number, date and place are reported, with no mark.

## Results

### Engines on sets A, C and B (step 8)

Number recall, % on A / C / B:

| Engine | A | C | B |
|---|---|---|---|
| Qwen2.5-VL 7B | 97.7 | 98.6 | 75.1 |
| PaddleOCR | 98.5 | 95.4 | 98.3 |
| VietOCR | 98.1 | 95.0 | 92.5 |
| PaddleOCR-VL 1.6 | 98.8 | 91.1 | 74.8 |
| SenOCR-Vi | 98.3 | 91.5 | 74.6 |
| Tesseract | 88.6 | 89.7 | 94.0 |

No engine reaches 98% on both A and C. The page-reading models share two failures: one
dense page of B makes each repeat until its output limit (without it they reach 98.6 to
98.8 on B), and their pipeline merges two table cells' figures into one token. Both are
scored as they came out.

### Fine-tuning

PaddleOCR's recogniser lacks many Vietnamese letters, so it was fine-tuned on typed pages
made to look scanned (`tools/train_lines.py`, `tools/train_rec.sh`). Any file in a set,
and any page naming a set's company, was left out of training. The fine-tuned reader
recognised tone marks more accurately but lost handwritten digits that the stock reader
got right. So the candidate is **two readers on one set of boxes**
(`tools/two_readers.py`): the fine-tuned and stock recognisers. When their digits differ,
the stock reader's line is used. A line is marked sure only when both readers produce
the same numbers and each reports a confidence of at least 0.95.

### The decision (step 9)

On C and a new draw D (seed 43, 20 kept pages, none seen in training), the candidate
**did not pass**: recall on C 97.9% (275 of 281), and wrong when sure on D 1.8% (5 of
284). D's five turned out to be numbers missing from its truth. With them added, D
passes, but the corrected truth was made after seeing the output, so the failed result
stands. Set E (seed 44) confirmed a heading rule change: company field 16 of 17, against
12 of 17 before.

A final test was fixed before it was drawn: **set F**, seed 45, 40 kept pages, at most 2
per company. The first reading of each page was made by a commercial vision-language
model (Claude), not one of the engines under test. Numbers where it and either engine
differed were typed by a person from the scan without seeing either value; 100 agreed
numbers and every company field were checked against the scan. The candidate was run
once.

| Mark | Needed | On F | 95% interval |
|---|---|---|---|
| Number recall | at least 98% | **98.7%** (607 of 615) | 97.5 to 99.3 |
| Wrong when sure | at most 1% | **0.19%** (1 of 535) | 0.0 to 1.1 |
| Coverage | at least 85% | **87.4%** | |
| Company field | at least 90% | **91.7%** (22 of 24) | 74.2 to 97.7 |

**Passed**, under the rule fixed in advance: each measured value meets its mark. That is
not 95% confidence that the true rates do: the intervals for number recall, wrong when
sure and the company field all reach past their marks. F shows the candidate is close
to the marks, not clear of them.

The one number wrong when sure is the date in a digital signature box; the truth
excludes the signature box by design (see Concepts). None of the 100 agreed numbers
checked was wrong. That bounds misreads shared by both readings below 3% of agreed
numbers (95% bound); it does not show there are none.

Reported with no pass mark: short numbers wrong when sure 2 of 761; whole dates 110 of
120; heading number 16 of 25, date 21 of 25, place 24 of 25; marks 98.8% of all words
and 98.3% of marked words. **Tone marks were scored against a reading that no person
checked**, so the marks figures are reported, never used as a pass.

## The tool

`tools/read_pdf.py` reads one disclosure PDF into a draft, using exactly the candidate
that passed on F, and nothing else. It is checked against F's scored run, byte for byte,
before use. The tuned recogniser's weights are published in their own repository,
fetched and checked by `tools/get_weights.sh`; `read_pdf.py` refuses any other weights.

- **Pages** are labelled as at step 1. Text-layer pages are taken as they are, unless
  the syllable check of step 5 finds the layer broken, or the layer came from OCR: those
  are read as scans. A skipped page keeps its reason and no text.
- **Scan pages** are rendered at 300 dpi with the signature box painted white and read by
  the two readers. About 3.9 s per page (median) on an 8 GB RTX 2060 SUPER.
- **Unsure lines** keep their words, and every number on them is left empty (`[?]`).
- **English pages** (30+ words, under 40% of syllables marked) are read but flagged as
  unmeasured.
- **Heading fields** (`tools/fields.py`) are filled only from sure lines, and only when
  the rule finds one value. The company field only just passed on F, then was wrong 2
  of 5 times in a spot check of other files, so a field found by the rule alone is never
  trusted.
- **A person** can fill what is empty, and confirm the fields, on a local review page
  (`tools/review.py`).

### The company check (set G)

`tools/listing.py` compares a heading's company with the official listing's names for
the filing's symbol. Set G (seed 46, 330 files from 165 companies, 80 of them with a
match) tested whether, when a company **read by OCR matches the listing**, the page
really prints that name. A person compared each image with the matched name, one item
per company: 60 test units, with 10 decoys mixed in, each changed by one letter. Item ids
gave no hint; "cannot tell" counted as wrong. The mark: at most 1 of 60 wrong; the check
is void if decoys are judged the same.

The first round was void (9 of 10 decoys passed). A second round with new decoys passed:
no decoy passed, and 1 of 60 test units was marked wrong (one-sided 95% bound 7.7%). That
one differs only by a missing space, which the match ignores.

The evidence is weaker than the count: the decoys in the second round were the only
items the person had not seen before, and several gave themselves away without being
read: their change made a syllable that does not exist in Vietnamese. One person
answered every round. So a listing match on an OCR page stands **provisionally**.
Further validation is required: new test units, decoys made only of real syllables and
shown as often as the test units, and a second person. A match on a text layer was not
tested.

### The read entry point

`tools/read.py` is the tool for a caller with no person to review. Its output format is
defined by `schemas/read.schema.json`.

- **Refused before reading** when the title's kind was never measured (insider trades,
  personnel, financial statements, annual reports, prospectuses, unsorted titles), or
  when there is no title: the output says `measured: false` and why, with no text.
- **Each heading field says how it was obtained.** The company is given only when it
  matches the listing (provisional on OCR pages, untested on a text layer). The number
  and place are "rule, unchecked". A date after the publication day "disagrees with
  meta" and is left empty; that check was never measured. An empty value is final.
- **Nothing is kept or fetched:** images go to a temporary folder; the caller's meta and
  paths are never written or logged; no network.

## Financial statements (profile 2)

**Not passed.** Two fresh sets were each scored once, and the candidate failed both. The
tool refuses financial statements until one passes (see The read entry point).

### What is measured

A statement's figures are read as **cells**: a figure at a statement (balance sheet B01,
income statement B02, cash flow statement B03), a row code and a column. The scored
column is the base period's.

- **Truth without typing every figure.** A period's figures are printed again as the
  comparative column of a later filing: the year-end balance sheet in later quarters, a
  period's income and cash flows a year on. That gives two readings of the same figures
  from different scans, which makes it less likely that both share an error; this was
  not measured. Each statement is paired with the later filing that repeats it. A cell
  both readings agree on is truth; every disagreement, and a random 100 agreed cells,
  are typed blind by a person from the scan.
- **A statement that could not be matched** across the pair is typed too: the person
  says on which pages it is printed and which columns hold the base period, without
  seeing the reading. So a statement the candidate misses counts against it.
- **Sums are never truth.** They may make a cell unsure, never sure, and the truth never
  comes from them, or "wrong when sure" would be near zero by construction.

Scores, never blended:

| Score | Mark |
|---|---|
| Cell recall: right value, sign and place | at least 98% |
| Wrong when sure, of sure cells | at most 0.5% |
| Coverage, cells answered sure | at least 85% |
| Statements found | at least 95% |
| Figures under the wrong statement (from set K) | at most 0.5% |
| Figures under codes their statement does not use (from set K) | at most 0.5% |

0.5% is half the disclosures' mark: a wrong figure in a statement feeds calculations.
Cells on one page are not independent, so intervals are also given by resampling
statements.

### Sets H, I and J

Drawn by `tools/grow_set_h.py` at random from the whole listing, one statement per
company, 30 companies each, non-financial forms, fiscal 2025 paired with 2026. Each set
skips the companies drawn for earlier sets.

- **H** (seed 47) is the development set: looked at, tuned on.
- **I** (seed 48) tested the candidate fixed before its draw. Scored once, 2026-11-13:

  | | On I | Mark |
  |---|---|---|
  | Cell recall | 3245 of 3660 = 88.66% | **misses** |
  | Coverage | 88.77% | meets |
  | Wrong when sure | 4 of 3249 = 0.12% | meets |
  | Statements found | 85 of 100 | **misses** |

  The statements it found were read accurately (recall 2581 of 2592 on those matched
  across the pair); the loss came from statements it did not find. Of the 15 missed or
  only partly found, 7 were income statements whose title the rule did not cover.
- **J** (seed 49) tested the candidate fixed after I. Scored once, 2026-11-16:

  | | On J | Mark | 95% interval |
  |---|---|---|---|
  | Cell recall | 2784 of 3063 = 90.89% | **misses** | 89.82 to 91.86 |
  | Coverage | 91.54% | meets | |
  | Wrong when sure | 20 of 2804 = 0.71% | **misses** | 0.46 to 1.10 |
  | Statements found | 90 of 98 | **misses** | 84.71 to 95.81 |

  4 of the 20 wrong answers differ in sign only; none is a figure given where the page
  prints none. 9 came from one page where the reading filed one statement under another
  that shares its codes.

**Faults found later, in the truth and the scorer, not the candidate.** The scorer looked
matched cells up in the reading the truth was built on, not the one being scored (fixed;
I's score was unchanged, and J's record was not affected). In one cell typed by the
person, the figure came from the column next to the one asked for.
58 of J's truth cells were filed under the statement the old reading named, not the one
the page's printed form label names; they were re-filed under the label, which a person
confirmed on each page, and the truth tool now asks a person whenever a page's own print
names another statement.

**Rescored after those fixes and later changes to the candidate, information only.** These
sets have been looked at, so this is not a pass, and the records above stand:

| | H | I | J |
|---|---|---|---|
| Cell recall | 99.83% | 99.43% | 98.60% |
| Wrong when sure | 4 cells | 4 cells | 5 of 3025 = 0.17% |
| Statements found | not rescored | 98 of 100 | 97 of 98 |

The next test, **set K**, is drawn only after its candidate and its marks are fixed.

## Limits

- One profile has passed: scanned Vietnamese disclosures. Financial statements are
  measured but have not passed. Other tables and English pages are not measured.
- The sets are small: 20 to 40 kept pages for disclosures, 30 companies for statements.
  The intervals above are wide.
- A number can be missed without leaving anything the system can flag: a line the
  detector never boxes leaves no trace.
- Tone marks were never checked by a person on a test set.
- Processing time was measured on one machine.

## The sets

`sets/disclosures-*.jsonl` and `sets/statements-*.jsonl` hold each file's source URL,
SHA-256 and labels, never the PDFs: the filings are public, but the hosts' copies are not
ours to redistribute.

**Truth.** For the statement tests I and J, `sets/truth-statements-*.jsonl` holds the
whole truth the scores were made from: the cells and their figures, the person's typed
figures and page answers, and each matched cell's column place, so anyone with the PDFs,
the code and the weights can score a reading against it
(`tools/fs_truth_bundle.py unpack`, then `tools/score_fs.py`). Unpacked and scored, it
gives the rescores above exactly. H's truth, in an older form made before the where round,
is not published. The disclosures' person-corrected truth (C to F) is the documents'
text typed out, which comes too close to redistributing them, so it is not published;
A's and B's truth can be rebuilt from the PDFs by the steps above.

| Set | Seed | Pages | Scores |
|---|---|---|---|
| A | 7 | 100 | numbers, from English twins |
| B | 10 | 101 | tone marks, from text layers |
| C | 13 | 20 kept of 30 | everything, person-corrected |
| D | 43 | 20 kept of 30 | the step 9 test, again |
| E | 44 | 20 kept of 30 | a heading rule change |
| F | 45 | 40 kept of 60 | the final test |
| G | 46 | 330 files | the company check |
| H | 47 | 60 files | statements, development |
| I | 48 | 60 files | statements, test (failed) |
| J | 49 | 60 files | statements, test (failed) |

**A set is published after it is scored, never before.** A draw that is still a test
stays private, so that nothing can be tuned to it, ours or anyone's; its seed and size are
fixed before it is drawn, and published with it.
