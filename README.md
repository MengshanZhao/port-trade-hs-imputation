# Port Trade HS Code Imputation

**Goal:** Assign accurate Harmonized System (HS) commodity codes to U.S.
port trade records when codes are missing or inconsistent.
**Scale:** ~10 years of U.S. export/import data; >20% missing HS codes
in some product categories
**Keywords:** Text classification, LinearSVC, BayesSearchCV, TF-IDF,
administrative data, trade data, ML pipeline

---

## Research context

This pipeline was built to support analysis of U.S. agricultural trade
flows over a decade. HS codes are the international standard for
classifying traded goods: without them, product-level trade analysis
is impossible. But real customs records frequently have missing, vague,
or inconsistent codes due to incomplete filing, product description
variation, and reporting errors.

Manual recoding at this scale is not feasible. This project develops
a hybrid automated approach that combines a fast deterministic lookup
with a text classification fallback, with explicit confidence scoring
to flag low-certainty cases for human review.

---

## Two approaches

### 1) Dictionary (deterministic)

Notebook: `Code_example/Datamyne_00matching.ipynb`

Maps product descriptions to HS codes via an authoritative phrase
dictionary. Fast and fully auditable. Coverage limited to descriptions
that closely match known canonical phrases.

### 2) Machine Learning classifier

Notebook: `Code_example/predictHS_version0728.ipynb`

- Text preprocessing → TF-IDF vectorization
- LinearSVC classifier
- Bayesian hyperparameter optimization (`BayesSearchCV`) for `C` tuning
- Decision function margins used as confidence proxy

Handles free-text descriptions, misspellings, and non-canonical phrasing
that the dictionary misses.

---

## Recommended hybrid policy

if dictionary match:
assign HS, log dict_version
elif ML margin >= threshold:
assign HS, log model_version + margin
else:
flag for manual QA


This tiered approach maximizes automation while maintaining auditability —
the same logic used in clinical coding validation workflows.

---

## Confidence proxy (LinearSVC)

```python
margins = best_model.decision_function(X_test)
conf = margins.max(axis=1)  # larger = more confident
```

Cases with low `conf` are routed to manual review rather than
automatically assigned.

---

## Transferability

The problem structure here, large-scale administrative records with
missing or miscoded categorical identifiers, where ground truth exists
but coverage is incomplete: maps directly onto clinical data challenges
such as missing ICD codes, NDC code inconsistencies, or lab result
classification in EHR data.

---

## Author

Mengshan Zhao | mengshan.zhao@wsu.edu | [www.mengshanzhao.com](https://www.mengshanzhao.com)
