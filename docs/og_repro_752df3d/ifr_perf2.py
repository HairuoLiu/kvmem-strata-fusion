"""Clean old-vs-new ifr.py scoring path comparison (fixed np.where bug)."""
import time
import numpy as np

rng = np.random.RandomState(0)
D = 64
CAND = 12_500
SPLIT_FRAC = 0.5
REPS = 10
TOP_K = 8

q = rng.randn(D)
mean_ks = rng.randn(CAND, D)
is_split = rng.rand(CAND) < SPLIT_FRAC
needles = rng.randn(CAND, D)
resid = rng.randn(CAND, D)
scale = 1.0 / np.sqrt(D)
block_ids = [f"b{i}" for i in range(CAND)]

def compute_logits(query, cands):
    return np.dot(cands, query) * scale

# ---- OLD (pre-752df3d): build flat list, ONE vectorized dot, top-2K, dedup = MAX pooling
def old_path():
    cand_keys, cand_mapping = [], []
    for i in range(CAND):
        if is_split[i]:
            cand_keys.append(needles[i]); cand_mapping.append(i)
            cand_keys.append(resid[i]);   cand_mapping.append(i)
        else:
            cand_keys.append(mean_ks[i]); cand_mapping.append(i)
    logits = compute_logits(q, np.array(cand_keys))
    k = min(len(logits), TOP_K * 2)
    top = np.argpartition(-logits, k)[:k]
    top = top[np.argsort(-logits[top])]
    seen, sel = set(), []
    for ci in top:
        bid = cand_mapping[ci]
        if bid not in seen:
            seen.add(bid); sel.append(bid)
            if len(sel) >= TOP_K: break
    return sel

# ---- NEW (752df3d): per-sub-centroid python loop + np.mean + list.sort = MEAN pooling
def new_path():
    sub_scores = {}
    for i in range(CAND):
        blk_logits = []
        if is_split[i]:
            for sc in (needles[i], resid[i]):
                blk_logits.append(float(compute_logits(q, np.array([sc]))[0]))
        else:
            blk_logits.append(float(compute_logits(q, np.array([mean_ks[i]]))[0]))
        sub_scores[i] = float(np.mean(blk_logits))
    ranked = sorted(sub_scores.items(), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked[:TOP_K]]

# ---- vectorized mean (semantically identical to new, one dot over collapsed vectors)
def vec_mean_path():
    eff = np.where(is_split[:, None], 0.5 * (needles + resid), mean_ks)
    logits = compute_logits(q, eff)
    top = np.argpartition(-logits, TOP_K)[:TOP_K]
    return list(top[np.argsort(-logits[top])])

a, b, c = old_path(), new_path(), vec_mean_path()
print("old(max-pool) top8 :", a)
print("new(mean-pool) top8:", b)
print("vec-mean top8      :", c)
print("new == vec-mean (as sets)?", set(b) == set(c))
print("overlap old vs new:", len(set(a) & set(b)), "/ 8")

for name, fn in [("OLD vectorized max-pool", old_path),
                 ("NEW per-sub-centroid loop", new_path),
                 ("vectorized mean (equiv.)", vec_mean_path)]:
    fn()
    t0 = time.perf_counter()
    for _ in range(REPS): fn()
    dt = (time.perf_counter() - t0) / REPS
    print(f"{name:26s} {dt*1000:8.2f} ms/query")
