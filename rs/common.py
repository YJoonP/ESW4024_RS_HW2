from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
K = 20


def load():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    n_users = int(train.userId.max()) + 1
    n_items = int(train.movieId.max()) + 1
    return train, test, n_users, n_items


def split(train, eval_users, mode="last", frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    df = train.sort_values(["userId", "time_idx"]).reset_index(drop=True)
    is_eval = df.userId.isin(set(eval_users)).to_numpy()
    held = np.zeros(len(df), dtype=bool)
    sub_pos = np.flatnonzero(is_eval)
    for _, pos in df[is_eval].groupby("userId").indices.items():
        rows = sub_pos[pos]
        n_hold = max(1, int(round(len(rows) * frac)))
        if mode == "last":
            pick = rows[-n_hold:]
        elif mode == "random":
            pick = rng.choice(rows, n_hold, replace=False)
        else:
            raise ValueError(mode)
        held[pick] = True
    return df[~held].reset_index(drop=True), df[held].reset_index(drop=True)


def seen_mask(df, users, n_items):
    row = {u: i for i, u in enumerate(users)}
    m = np.zeros((len(users), n_items), dtype=bool)
    d = df[df.userId.isin(row)]
    m[d.userId.map(row).to_numpy(), d.movieId.to_numpy()] = True
    return m


def topk(scores, exclude, k=K):
    s = np.where(exclude, -np.inf, scores.astype(np.float64))
    part = np.argpartition(-s, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, part, 1), axis=1)
    return np.take_along_axis(part, order, 1)


def ndcg_at_k(recs, targets, k=K):
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    vals = []
    for r, t in zip(recs, targets):
        if not t:
            continue
        dcg = sum(disc[i] for i, it in enumerate(r[:k]) if it in t)
        idcg = disc[: min(len(t), k)].sum()
        vals.append(dcg / idcg)
    return float(np.mean(vals))


def evaluate(model_factory, train, test, n_items, mode="last", frac=0.2, seed=0):
    users = test.userId.to_numpy()
    fit_df, val_df = split(train, users, mode=mode, frac=frac, seed=seed)
    model = model_factory().fit(fit_df)
    scores = model.score(users)
    recs = topk(scores, seen_mask(fit_df, users, n_items))
    tgt = val_df.groupby("userId").movieId.apply(set)
    return ndcg_at_k(recs, [tgt.get(u, set()) for u in users])


def write_submission(model, train, test, n_items, name):
    users = test.userId.to_numpy()
    model.fit(train)
    recs = topk(model.score(users), seen_mask(train, users, n_items))
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.csv"
    pd.DataFrame({"id": test.id, "recommendations": [" ".join(map(str, r)) for r in recs]}).to_csv(path, index=False)
    return path
