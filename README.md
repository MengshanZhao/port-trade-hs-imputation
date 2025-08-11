# Port Trade HS Code Imputation

**Goal:** Assign accurate Harmonized System (HS) codes to U.S. port trade records when codes are missing or incomplete.  
This repository contains two approaches — a deterministic dictionary match and a machine learning (ML) classifier — plus a recommended hybrid strategy.

---

## Why this matters
Many customs and trade datasets contain incomplete or inconsistent HS codes, making longitudinal or comparative trade analysis difficult.  
This project shows two scalable ways to impute codes:

1. **Dictionary match** — Fast, fully explainable, but only works when the description is close to a known phrase.
2. **Machine Learning (ML)** — Uses text features to classify even when there’s no exact match, but is less interpretable.

---

## Approaches

### 1) Dictionary (Deterministic)
Notebook: `Code_example/Datamyne_00matching.ipynb`  
- Uses an authoritative **phrase → HS mapping** table.  
- **Pros:** Fast, auditable, easy to explain to policy or compliance teams.  
- **Cons:** No coverage when product description deviates from the canonical phrase.

### 2) Machine Learning
Notebook: `Code_example/predictHS_version0728.ipynb`  
- Text preprocessing → TF-IDF vectorization → LinearSVC classifier.  
- Bayesian hyperparameter optimization (`skopt.BayesSearchCV`) to tune `C`.  
- **Pros:** Covers free-text, misspellings, and non-canonical descriptions.  
- **Cons:** Requires feature engineering, training data, and interpretability safeguards.


---

## Operational policy suggestion

1. If dictionary match → assign HS + log `dict_version`.
2. Else if ML margin ≥ τ → assign HS + log `model_version`, `margin`.
3. Else → flag for manual QA.

Example margin calculation in LinearSVC:
```python
# margin as a confidence proxy
margins = best_model.decision_function(X_test)
conf = margins.max(axis=1)  # larger = more confident
