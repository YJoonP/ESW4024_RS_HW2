# SKKU ESW4024 Recommender System — HW2

Kaggle: https://www.kaggle.com/competitions/skku-esw-4024-recommender-system/

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Put `train.csv`, `test.csv`, `sample_submission.csv` from Kaggle in `data/`.

## Run
```bash
python -m rs.run_baselines --submit   # Popularity, EASE (Q1)
python -m rs.run_neural neumf --submit  # NeuMF (Q2)
```
Submissions are written to `outputs/`.

## Validation
For each test user, the last 20% of interactions by `time_idx` are held out; models are tuned by NDCG@20 on this split.

## Results (NDCG@20)
| Model | Validation | Public LB |
|---|---|---|
| EASE (λ=200) | 0.1551 | 0.11637 |
| EASE (λ=200, recency τ=10) | 0.1729 | 0.12411 |
| NeuMF (dim=32, 40 epochs) | 0.1252 | 0.09489 |
