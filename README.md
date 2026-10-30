# AI Customer Support Assistant (Evidence-Aware RAG)

An offline Retrieval-Augmented Generation (RAG) customer-support assistant. It answers
questions from a knowledge base, shows which articles the answer came from, and declines
to answer (referring the customer to support) when the knowledge base does not contain
enough evidence.
** Live Demo:** https://ai-support-assistant-rag-vcraedqq5kejm8sjzpfltn.streamlit.app/
> **DEMO DATA.** The knowledge base (22 FAQ articles) and the ticket set (186 tickets)
> are synthetic examples, not the verified policies of any real company. This project
> demonstrates an architecture. It is not production-ready and has not been tested with
> real customers.

## Problem statement

A support chatbot that always answers can present unsupported claims as facts. In the
reference system this project started from, "What is the status of my specific refund?"
returned the generic refund policy, and an empty or unrelated question returned an
arbitrary article (a similarity of 0.0 was still turned into an answer).

Objective: build a lightweight, free, offline RAG assistant that answers when the
knowledge base provides evidence, shows its sources, and declines otherwise, and measure
the improvement on a labeled test set.

## Architecture

```
customer question
      |
      +--> Intent classifier  (TF-IDF + Logistic Regression)   shown in Developer view only
      +--> Tone scorer        (VADER)                          adjusts reply tone only
      +--> KnowledgeBase      (TF-IDF -> LSA -> FAISS cosine search, top 3)
                |
                v
        Answerability check   (src/answerability.py), in this order:
          1. empty question                             -> decline
          2. needs the customer's own records           -> decline
             (order numbers, "status of my...", "where is my order")
          3. no evidence (top similarity 0.0)           -> decline
          4. low term coverage (under 30% of the question's meaningful
             words appear in the retrieved articles)    -> decline
                |
                v
        Reply composer        (src/reply.py, offline template mode)
                |
                v
        Streamlit UI          (answer + the FAQ articles used, or a
                               "please contact our support team" message)
```

Retrieval similarity is NOT treated as answer confidence or correctness: a related
article can score high without answering the question. The rules in step 2 are simple
keyword patterns, crude by design, and evaluated rather than trusted. Intent does not
influence the answer.

## Evaluation

**Test set** (`eval/eval_set.json`): 41 questions, 22 answerable from the knowledge base
and 19 not (customer-specific, not in the knowledge base, ambiguous, sensitive, empty).
Each has the relevant FAQ ids and the expected behaviour.

- **dev (19 questions):** used while designing the rules.
- **held-out (22 questions):** examined only in aggregate; rules were frozen before the
  final run. 10 were written by the project author in their own words; the other 31
  questions in the set were drafted with AI assistance.

**Metrics:** recall@k (a relevant article is in the top k), MRR (mean of 1/rank of the
first relevant article), share of unanswerable questions declined, share of answerable
questions declined (the cost of caution), and correct top-1 answers (the top source is a
labeled relevant article; this measures source relevance, not factual correctness).

**Results, held-out split (12 answerable, 10 unanswerable)**

| Stage | Unanswerable declined | Answerable declined | recall@1 | recall@3 | MRR |
|---|---|---|---|---|---|
| Reference system (baseline) | 0/10 | 0/12 | 0.58 | 0.75 | 0.64 |
| + account-record and no-evidence rules | 4/10 (40%) | 0/12 | 0.58 | 0.75 | 0.64 |
| + term-coverage rule | 6/10 (60%) | 3/12 (25%) | 0.58 | 0.75 | 0.64 |

**Results, all 41 questions (22 answerable, 19 unanswerable)**

| Stage | Unanswerable declined | Answerable declined | recall@1 | recall@3 | MRR |
|---|---|---|---|---|---|
| Reference system (baseline) | 0/19 | 0/22 | 0.64 | 0.73 | 0.67 |
| + account-record and no-evidence rules | 11/19 (58%) | 1/22 (5%) | 0.64 | 0.73 | 0.67 |
| + term-coverage rule | 15/19 (79%) | 4/22 (18%) | 0.64 | 0.73 | 0.67 |

**How to read these**
- Retrieval metrics are identical in every row because retrieval was not changed.
  Retrieval is the weaker part: recall@1 is about 0.6.
- Correct top-1 answers stayed the same at every stage (7/12 held-out, 14/22 overall), so
  the answerable questions that were declined had not been answered with a correct
  article before: wrong answers became referrals and no correct answer was lost.
- On the dev split the final rule declines 9/9 unanswerable questions, but the rules were
  tuned on those questions, so use the held-out figures.
- 4 of the 10 unanswerable held-out questions still receive an answer.

**Intent classifier.** The 186 tickets contain only 119 unique texts once numbers are
ignored. 5-fold cross-validated accuracy on the unique tickets is 0.899. The average
confidence score is 0.334, far below the accuracy, so the score is not a calibrated
probability. The tickets are template-generated, so this figure is probably optimistic for
real customer language.

## Limitations

- Tiny synthetic data: 22 FAQ articles and 119 unique tickets.
- Small test sets: one question moves a rate by about 4-5 percentage points, so results
  are indicative, not statistically strong. The held-out set was scored in aggregate
  several times, so it is not a perfectly untouched test.
- The coverage threshold (0.3) was chosen from two dev questions and may not generalise.
- Keyword rules for customer-specific questions will miss other phrasings.
- Retrieval (TF-IDF + LSA) cannot match synonyms such as "broken" and "damaged".
- Template mode copies retrieved FAQ text. The system does not verify that the text is true.
- No access to real orders or accounts, no multi-turn memory, no real escalation. The UI
  tells the customer to contact support; nothing is sent.
- Most test questions and the rules were drafted with AI assistance.

## Ethics and privacy

- All data is synthetic. Do not use real customer data without authorization and safeguards.
- A deployment handling personal data would need data minimisation, access control,
  retention limits and careful handling of secrets (never commit API keys).
- Sentiment describes tone only. It does not establish urgency or correctness.

## Setup and run (Windows, Python 3.11)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Tick **Developer view** in the sidebar to see intent, tone, scores, retrieved evidence,
decline reasons and the evaluation tab.

```
python -m pytest -q                                   # 26 tests
python eval\run_eval.py --split all --label mytest    # use a NEW label; do not overwrite saved results
python eval\report_table.py                           # prints the summary tables above
```

## Project layout

```
app.py                  Streamlit UI (customer view + Developer view)
train.py                builds the knowledge-base index and trains the intent classifier
data/                   tickets.csv, faq_knowledge_base.json (synthetic), generate_data.py
src/
  kb_retriever.py       KnowledgeBase: TF-IDF -> LSA -> FAISS, validation, stale-index check
  answerability.py      the four answer-or-decline rules
  reply.py              offline template reply composer (records the sources it used)
  tone.py               VADER tone scoring
  intent_classifier.py  TF-IDF + Logistic Regression, deduplicated cross-validated evaluation
  pipeline.py           wires the pieces together
eval/                   eval_set.json, run_eval.py, report_table.py, compare_runs.py,
                        coverage_dev.py, results/ (saved baseline and step results)
tests/                  test_pipeline.py (13), test_rewritten_modules.py (8), test_intent.py (5)
models/                 generated by train.py (not tracked in git)
```

## Future scope

- Compare the LSA retriever with a small neural embedding model (size, speed, recall).
- Let an organisation supply its own approved documents, with source metadata.
- Replace the keyword rules with an evaluated learned answerability classifier.
- Larger, independently labeled test sets with confidence intervals.
- Multi-turn conversation and real human escalation.
- A tool-using agent (order lookup, ticket creation) as a separate extension.

## Credits and license

The design was informed by an existing open-source (MIT-licensed) support-RAG
demonstration, whose synthetic ticket and FAQ data were reused here. Its license notice
is kept in `LICENSE`. The retriever, answerability rules, reply composer, tone scorer,
intent classifier, pipeline, interface, tests and evaluation described above were written
for this project.