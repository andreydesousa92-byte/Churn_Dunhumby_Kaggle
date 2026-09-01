# Customer Churn Prediction — dunnhumby "Complete Journey"

A churn model for a grocery retailer, built on the [dunnhumby "Complete Journey"](https://www.dunnhumby.com/source-files/) dataset (~2,500 households, 711 days of transactions). The project's focus is less "fit a classifier" and more "get the label, the split, and the evaluation right first" — most of what's in the notebook is about making a silently-broken model *detectable* before it ships.

**Objective:** predict which households are at risk of churning — defined as making no purchase in the 28 days following a cutoff point — from their prior shopping behaviour.

## Results

| | PR-AUC |
|---|---|
| **model on held-out test set** | **0.652** |
| baseline (recency-only rule) on test | 0.516 |
| chance (test churn rate) | 0.157 |
| model in cross-validation | 0.512 |

- **4.2x chance**, **+0.136 PR-AUC over a one-column recency rule**, on the same households. ROC-AUC is 0.897, reported for comparability, but PR-AUC is the honest number on a 15.7% positive class.
- **In operational terms:** ranking test households by predicted risk and contacting the top 50 catches 37 churners (74% precision) — half of all churners in the set, 4.7x better than a random 50.
- **Out-of-time validation:** the pipeline was rebuilt at three independent cutoffs (non-overlapping outcome windows) and every model scored against every period. Lift over the baseline is stable across all six cross-period cells, +0.057 to +0.080 (avg +0.071) — the model's edge does not depend on when it was fitted.

The final model is a plain **logistic regression**, chosen over a depth/leaf-restricted random forest after the two landed in a technical tie under cross-validation (4 of 5 paired folds, lower fold-to-fold variance) — resolved on simplicity, not on 0.013 of PR-AUC.

## Why this is more than a classifier fit

A few of the decisions the notebook walks through in detail:

- **The leakage trap.** Labelling churn over the final weeks of data while building features from the full history makes `recency` and the label the same fact written twice — the model would score near-perfect and learn nothing. Fixed by a hard cutoff `T`: features only from data `≤ T`, label only from the 28 days after it.
- **Population vs. event.** "At risk of churning" and "already gone" are different questions. Households with no purchase in the 90 days before `T` are excluded from the eligible population before labelling — otherwise a retention budget spent on the highest-risk decile goes mostly to people who will never come back.
- **Two cuts, two purposes.** The time cutoff (`T`) separates past from future within each customer; the train/test split separates customers from customers. They defend against different failures (predicting the present with the present vs. memorising individuals) and are applied in that order.
- **Metric fixed before any model is trained.** Accuracy is unusable at a 15.6% positive rate (predicting "nobody leaves" scores 84%+). PR-AUC is primary, ROC-AUC secondary — decided in writing before results existed, since the two metrics disagree by more than a third of the scale on the same baseline rule.
- **A one-column baseline as the unit of measurement.** Ranking households by `recency` alone gets PR-AUC 0.516 on test. Every model has to clear that bar to justify its complexity.
- **Four rejected features.** Fuel purchases, coupon redemption, campaign exposure, and store concentration all looked like strong retention signals on raw association — and all four either were confounded with visit frequency (Simpson's paradox: e.g. gas-kiosk buyers churn at 5.7% vs. 19.5% overall, but the gap vanishes once controlled for visit frequency) or added nothing once tested inside the model. A real effect is not the same as new information.
- **Out-of-time validation**, because a single train/test split only answers "does this generalise to unseen customers," not "does a model fitted in one period still work in another."

## Repository contents

| File | Description |
|---|---|
| [`customer-churn-prediction.ipynb`](customer-churn-prediction.ipynb) | The full analysis: data prep, label construction, EDA, feature engineering, model selection, hyperparameter tuning, final evaluation, out-of-time validation. |
| [`churn_model_methodology.docx`](churn_model_methodology.docx) | A written companion covering the reasoning behind the notebook — why the cutoff is not optional, why the metric is chosen before results, why raw feature associations mislead. |
| [`Predicting Customer Churn - Figo Data.pptx`](Predicting%20Customer%20Churn%20-%20Figo%20Data.pptx) | A slide summary of the project and its findings. |

## Data

This project uses the **dunnhumby "Complete Journey"** dataset — household-level transactions, product, demographic, coupon, and campaign tables for ~2,500 households over 711 days. It is not included in this repository (large, third-party licensed).

To reproduce locally:

1. Download the dataset, e.g. via Kaggle: [`frtgnn/dunnhumby-the-complete-journey`](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey).
2. Place the CSVs in a local folder, e.g. `data/raw/`.
3. In the notebook, update `RAW_PATH` (currently set to a Kaggle-notebook path, `/kaggle/input/...`) to point at that folder:

   ```python
   from pathlib import Path
   RAW_PATH = Path("data/raw")
   ```

The notebook was originally authored and run on Kaggle; the rest of the code has no other Kaggle-specific dependency.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
jupyter lab
```

Built with Python 3.12. Core dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

## Method summary

1. **Data prep** — load and validate 8 raw tables (keys, nulls, referential integrity between them).
2. **Defining churn** — cutoff `T = 683`; eligible population = households active in the 90 days before `T`; label = no purchase in the 28 days after `T`.
3. **Train/test split** — stratified on `churn`, at the household level.
4. **Feature engineering** — 7 behavioural features (recency, tenure, visit counts at 90/180d, spend, distinct products, basket size, visit-trend), all computed strictly from data `≤ T`.
5. **EDA** — distributions, medians, and correlations of features against churn, on the training set only.
6. **Baseline** — recency-only ranking rule, scored before any model.
7. **Cross-validation** — logistic regression, decision tree, and random forest compared on 5-fold stratified PR-AUC.
8. **Hyperparameter tuning** — grid search over the random forest's regularising parameters.
9. **Feature re-engineering** — four candidate features tested against the visit-frequency confound and rejected.
10. **Final evaluation** — logistic regression scored once against the held-out test set.
11. **Out-of-time validation** — the full pipeline rebuilt at three cutoffs to check the model's edge holds across periods, not just across customers.

## Author

Andrey de Sousa Figueiredo — [LinkedIn](https://www.linkedin.com/in/andrey-de-sousa-figueiredo)

## License

[MIT](LICENSE) for the code in this repository. The underlying dataset is subject to dunnhumby's own terms — see the source link above.
