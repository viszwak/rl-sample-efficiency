# inspect_dataset.py
import pickle
import numpy as np
from collections import Counter

with open('lunarlander_online_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

obs_lens = Counter()
act_lens = Counter()
examples = {}

# Only scan the first 50k for speed
for idx, (obs, act, *_) in enumerate(data[:50_000]):
    L = len(obs) if hasattr(obs, '__len__') else None
    A = len(act) if hasattr(act, '__len__') else None
    obs_lens[L] += 1
    act_lens[A] += 1
    key = (L, A)
    if key not in examples:
        examples[key] = obs

print("=== Observation lengths ===")
for L, cnt in obs_lens.most_common():
    print(f"  len(obs)={L}: {cnt}")

print("\n=== Action lengths ===")
for A, cnt in act_lens.most_common():
    print(f"  len(act)={A}: {cnt}")

print("\n=== Sample obs for each (obs_len, act_len) ===")
for (L, A), obs in examples.items():
    print(f"\nCase len(obs)={L}, len(act)={A}")
    if L == 8:
        arr = np.asarray(obs, dtype=np.float32)
        print("  Flat obs → first 8 values:", arr.flatten()[:8])
    elif L == 2:
        print("  Nested obs pieces:")
        for i, part in enumerate(obs):
            try:
                arr_part = np.asarray(part, dtype=np.float32)
                print(f"    Part {i}: shape={arr_part.shape}, sample={arr_part.flatten()[:4]}")
            except Exception as e:
                print(f"    Part {i}: could not convert to array ({e}), repr={repr(part)[:100]}")
    else:
        print("  (Unexpected obs length)")
