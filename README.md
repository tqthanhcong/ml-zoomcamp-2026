# Machine Learning Zoomcamp 2026

Solutions and projects for the live 2026 cohort of DataTalks.Club's Machine Learning Zoomcamp.

## Progress

| Module | Deliverable | Status |
|---|---|---|
| 1. Introduction to Machine Learning | `01-intro/homework.ipynb` | Complete |
| 2. Machine Learning for Regression | `02-regression/homework.ipynb` | Complete |
| 3. Machine Learning for Classification | `03-classification/homework.ipynb` | Complete |
| 4. Evaluation Metrics for Classification | `04-evaluation/homework.ipynb` | Complete |
| 6. Decision Trees and Ensemble Learning | `06-trees/homework.ipynb` | Complete |

## Reproducing a homework notebook

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --execute --to notebook --inplace 01-intro/homework.ipynb
```
