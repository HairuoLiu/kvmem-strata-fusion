"""Measure ifr.py retrieve() scoring loop: current per-sub-centroid loop vs vectorized."""
import time
import numpy as np

rng = np.random.RandomState(0)
D = 64
N = 50_000          # total unique blocks
CAND = 12_500       # IVF probe candidates (n_probe=4 of 8 clusters)
SPLIT_FRAC = 0.5    # anti-collapse fires on ~50% of blocks
REPS = 20

q = rng.randn(D).astype(np.float64)
mean_ks = rng.randn(CAND, D)
is_split = rng.rand(CAND) < SPLIT_FRAC
needles = rng.randn(CAND, D)
resid = rng.randn(CAND, D)
# effective per-block vector: (needle + residual_mean)/2 if split else mean_k
vecs = np.where(is_split[:, None], 0.5 * (needles + resid), mean_ks)

scale = 1.0 / np.sqrt(D)

# ---- (a) current production path: per-sub-centroid python loop, np.array([sc]) each call
def current_path():
    scores = np.empty(CAND)
    for i in range(CAND):
        if is_split[i]:
            s1 = float(np.dot(np.array([needles[i]]), q)[0]) * scale
            s2 = float(np.dot(np.array([resid[i]]), q)[0]) * scale
            scores[i] = (s1 + s2) / 2.0
        else:
            scores[i] = float(np.dot(np.array([mean_ks[i]]), q)[0]) * scale
    return scores

# ---- (b) vectorized: one dot over stacked sub-centroids
sub = np.stack([needles, resid], axis=1)          # (C,2,D)
def vectorized_path():
    logits = np.einsum("cd,d->c", sub.reshape(-1, D), q).reshape(CAND, 2) * scale
    merged = np.where(is_split[:, None], logits.mean(axis=1), np.einsum("cd,d->c", mean_ks, q) * scale)
    return merged

# ---- (c) fully collapsed single vector (what mean-of-logits is mathematically equal to)
def collapsed_path():
    return np.einsum("cd,d->c", vecs, q) * scale

a = current_path(); b = vectorized_path(); c = collapsed_path()
print("max |a-b| =", np.abs(a - b).max())
print("max |a-c| =", np.abs(a - c).max(), " (mean-of-logits == logit-of-mean-of-subcentroids?)")
print("max |a-argmax shift| :", np.argmax(a), np.argmax(b), np.argmax(c))

for name, fn in [("current per-sub-centroid loop", current_path),
                 ("vectorized stacked", vectorized_path),
                 ("collapsed single vector", collapsed_path)]:
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn()
    dt = (time.perf_counter() - t0) / REPS
    print(f"{name:34s} {dt*1000:8.2f} ms/query  ({CAND} candidates)")
