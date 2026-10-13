# AI Customer Support Assistant (Evidence-Aware RAG)

An offline Retrieval-Augmented Generation (RAG) customer-support assistant. It answers
questions from a knowledge base, shows which article the answer came from, and declines
to answer (referring the customer to support) when the knowledge base does not contain
enough evidence.

> **DEMO DATA.** The knowledge base (22 FAQ articles) and the ticket set (186 tickets)
> are synthetic examples. They are NOT the verified policies of any real company.
> This project demonstrates an architecture. It is not production-ready and has not
> been tested with real customers.

## Problem statement

A support chatbot that always answers can present unsupported claims as facts. In the
starter project, the question "What is the status of my specific refund?" returned the
generic refund policy, and an empty or unrelated question returned an arbitrary article
(a retrieval similarity of 0.0 was still turned into an answer).

Objective: build a lightweight, free, offline RAG assistant that answers when the
knowledge base provides evidence, shows its source, and declines otherwise, and measure
the improvement on a labeled test set.

## What was added to the starter project

| Starter project | This project |
|---|---|
| Always answers with the top retrieved article | Answerability checks decide whether to answer or refer to support |
| "I'm escalating to a human agent" (no such mechanism exists) | Accurate wording: "Please contact our support team" |
| Confidence label on the intent score | Labeled as an uncalibrated classifier score, shown only in Developer view |
| No source shown | Source FAQ shown with every answer |
| 7 smoke tests | 12 tests, including tests for correct abstention |
| No evaluation | 41 labeled questions, dev/held-out split, saved baseline and results |
| Single UI | Simple customer view plus a Developer view with scores and evaluation tables |

## Architecture

```
customer question
      |
      +--> Intent classifier  (TF-IDF + Logistic Regression)   shown in Developer view only
      +--> Sentiment scorer   (VADER)                          adjusts reply tone
      +--> Retriever          (TF-IDF -> LSA -> FAISS cosine search, top 3)
                |
                v
        Answerability check   (src/answerability.py)
          1. empty question                         -> decline
          2. needs the customer's own records       -> decline
             (order numbers, "status of my...", "where is my order")
          3. no evidence (top similarity 0.0)       -> decline
          4. low term coverage (fewer than 30% of the question's
             meaningful words appear in the retrieved articles) -> decline
                |
                v
        Answer generator     (template mode, offline, default)
                |
                v
        Streamlit UI         (answer + source, or a "please contact support" message)
```

Retrieval similarity is NOT treated as answer confidence or correctness. A related
article can score high without answering the question. The rules in step 2 above are
simple keyword patterns: crude by design, and evaluated rather than trusted.

The optional OpenAI generation mode (enabled only if `OPENAI_API_KEY` is set) is off by
default, was not evaluated, and is not needed for any result below. A prompt instruction
is not a guarantee against hallucination.

## Evaluation

**Test set** (`eval/eval_set.json`): 41 questions, 22 answerable from the knowledge base
and 19 not (customer-specific, not in the knowledge base, ambiguous, sensitive, empty).
Each has the relevant FAQ ids and the expected behaviour (answer or decline).

- **dev (19 questions):** used while designing the rules.
- **held-out (22 questions):** scored only in aggregate, and the rules were frozen before
  the final run. 10 were written by the project author in their own words without looking
  at the FAQ answers; the other 12 were drafted with AI assistance.

**Metrics**
- *recall@k:* share of answerable questions with a relevant article in the top k results.
- *MRR:* mean of 1 / (rank of the first relevant article).
- *Unanswerable correctly declined:* share of unanswerable questions the system declined.
- *Answerable declined:* share of answerable questions the system declined (the cost of
  being cautious).
- *Correct top-1 answers:* answerable questions answered with a labeled relevant article
  as the top source. This measures source relevance, not the factual correctness of the text.

**Results, held-out split (the fairer view; 12 answerable, 10 unanswerable)**

| Stage | Unanswerable correctly declined | Answerable declined | recall@1 | recall@3 | MRR |
|---|---|---|---|---|---|
| Baseline (starter project) | 0/10 | 0/12 | 0.58 | 0.75 | 0.64 |
| + account-record and no-evidence rules | 4/10 (40%) | 0/12 | 0.58 | 0.75 | 0.64 |
| + term-coverage rule | 6/10 (60%) | 3/12 (25%) | 0.58 | 0.75 | 0.64 |

**Results, all 41 questions (22 answerable, 19 unanswerable)**

| Stage | Unanswerable correctly declined | Answerable declined | recall@1 | recall@3 | MRR |
|---|---|---|---|---|---|
| Baseline (starter project) | 0/19 | 0/22 | 0.64 | 0.73 | 0.67 |
| + account-record and no-evidence rules | 11/19 (58%) | 1/22 (5%) | 0.64 | 0.73 | 0.67 |
| + term-coverage rule | 15/19 (79%) | 4/22 (18%) | 0.64 | 0.73 | 0.67 |

**How to read these**
- Retrieval metrics are identical in every row because retrieval was not changed.
  Retrieval is the weaker part: recall@1 is about 0.6.
- Correct top-1 answers stayed the same at every stage (7/12 held-out, 14/22 overall). So
  the answerable questions that were declined had not been answered with the correct
  article before: wrong answers became referrals, and no correct answer was lost.
- On the dev split the final rule declines 9/9 unanswerable questions, but the rules were
  tuned on those questions, so that number is partly built in. Use the held-out figures.
- 4 of the 10 unanswerable held-out questions still receive an answer.

## Limitations

- Tiny synthetic data: 22 FAQ articles; 186 tickets, of which 35 texts are duplicates.
- Small test sets: one question moves a rate by about 4-5 percentage points. The results
  are indicative, not statistically strong.
- The coverage threshold (0.3) was chosen from two dev questions and the gap around it is
  small. It may not generalise.
- Keyword rules for customer-specific questions will miss other phrasings.
- Retrieval (TF-IDF + LSA) cannot match synonyms. For example "broken" does not match
  "damaged", so some correct questions get no relevant article.
- The intent classifier was not independently evaluated (duplicate texts can leak between
  train and test). Its score is not calibrated and does not affect the answer.
- Template mode copies the retrieved FAQ answer and adds a short tone prefix.
- No access to real orders or accounts, no multi-turn memory, and no real escalation to
  a human. The UI says to contact support; nothing is sent.
- Part of the test set and the rules were drafted with AI assistance (see Evaluation).

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

Tick **Developer view** in the sidebar to see intent, sentiment, scores, retrieved
evidence and the evaluation tab.

```
python -m pytest -q                                   # 12 tests
python eval\run_eval.py --split all --label mytest    # use a NEW label, do not overwrite saved results
python eval\report_table.py                           # prints the summary tables above
```

## Project layout

```
app.py                  Streamlit UI (customer view + Developer view)
train.py                trains the intent classifier and builds the retrieval index
data/                   generate_data.py, tickets.csv, faq_knowledge_base.json (synthetic)
src/                    intent_classifier, sentiment, retriever, answerability,
                        generator, pipeline
eval/                   eval_set.json, run_eval.py, coverage_dev.py, report_table.py,
                        results/ (saved baseline and step results)
tests/test_pipeline.py  12 tests
models/                 generated by train.py (not tracked in git)
```

## Future scope

- Compare the LSA retriever with a small neural embedding model, measuring size, speed and recall.
- Allow an organisation to upload its own approved documents, with source metadata.
- Evaluate the intent classifier on deduplicated data.
- Enlarge and independently label the test set.
- A tool-using agent (order lookup, ticket creation) as a separate extension.
- Real human escalation, and multi-turn conversation.
