import sys

from rs.common import evaluate, load, write_submission
from rs.models import EASE, Popularity

train, test, n_users, n_items = load()
modes = ["last", "random"]

print(f"{'model':28s}" + "".join(f"{m:>10s}" for m in modes))
print(f"{'Popularity':28s}" + "".join(f"{evaluate(lambda: Popularity(n_users, n_items), train, test, n_items, mode=m):10.4f}" for m in modes))

best = {}
for lam in [50, 100, 200, 500, 1000, 2000]:
    row = {m: evaluate(lambda: EASE(n_users, n_items, lam=lam), train, test, n_items, mode=m) for m in modes}
    for m in modes:
        if row[m] > best.get(m, (0, None))[0]:
            best[m] = (row[m], lam)
    print(f"{'EASE lam=' + str(lam):28s}" + "".join(f"{row[m]:10.4f}" for m in modes))

for m in modes:
    print(f"best on {m}: NDCG@20={best[m][0]:.4f} at lambda={best[m][1]}")

if "--submit" in sys.argv:
    lam = best["last"][1]
    print("wrote", write_submission(EASE(n_users, n_items, lam=lam), train, test, n_items, f"ease_lam{lam}"))
