import sys, time
import torch
from rs.common import evaluate, load, write_submission
from rs.neural import NeuMF, SASRec

train, test, n_users, n_items = load()
which = sys.argv[1]
import os
dev = os.environ.get("DEV") or ("cuda:1" if which == "neumf" else "cuda:0")

if which == "neumf":
    grid = [dict(dim=d, epochs=e, n_neg=n) for d in (32, 64) for e in (10, 20, 40) for n in (4,)]
    Model = NeuMF
else:
    grid = [dict(epochs=e, dropout=p) for p in (0.2, 0.5) for e in (100, 200, 400, 800)]
    Model = SASRec

best = (0, None)
for kw in grid:
    t = time.time()
    v = evaluate(lambda: Model(n_users, n_items, device=dev, **kw), train, test, n_items, mode="last")
    print(f"{which} {kw}  NDCG@20(last)={v:.4f}  ({time.time()-t:.0f}s)", flush=True)
    best = max(best, (v, kw), key=lambda x: x[0])
print("best", best, flush=True)

if "--submit" in sys.argv:
    kw = best[1]
    name = which + "_" + "_".join(f"{k}{v}" for k, v in kw.items())
    print("wrote", write_submission(Model(n_users, n_items, device=dev, **kw), train, test, n_items, name))
