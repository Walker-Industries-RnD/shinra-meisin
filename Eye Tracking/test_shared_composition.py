"""Smoke test: verify SharedDataset interleaving composition for phase 2.

synthetic = 0, real = 1 — no dataset classes involved.
"""

import random

PHASE = 2
SYN_FRAC = 1.0 - 0.7 * PHASE / 3  # ~0.533
EPOCH_SIZE = 1000
TRIALS = 200
SEED_BASE = 42


def simulate_epoch(total: int, syn_frac: float, seed: int) -> list[int]:
    """Returns a sequence of 0 (synthetic) or 1 (real) for one epoch."""
    rng = random.Random(seed)
    ramp_end = 0.50 * max(total - 1, 1)
    order = []
    for i in range(total):
        p_syn = 1.0 - (1.0 - syn_frac) * min(i / ramp_end, 1.0)
        order.append(0 if rng.random() < p_syn else 1)
    return order


def window_real_frac(order: list[int], lo: float, hi: float) -> float:
    a, b = int(lo * len(order)), int(hi * len(order))
    chunk = order[a:b]
    return sum(chunk) / len(chunk) if chunk else 0.0


# Aggregate over many trials to smooth out randomness.
buckets = {"0–25%": [], "25–50%": [], "50–75%": [], "75–100%": [], "overall": []}

for t in range(TRIALS):
    order = simulate_epoch(EPOCH_SIZE, SYN_FRAC, seed=SEED_BASE + t)
    buckets["0–25%"].append(window_real_frac(order, 0.00, 0.25))
    buckets["25–50%"].append(window_real_frac(order, 0.25, 0.50))
    buckets["50–75%"].append(window_real_frac(order, 0.50, 0.75))
    buckets["75–100%"].append(window_real_frac(order, 0.75, 1.00))
    buckets["overall"].append(sum(order) / len(order))

print(f"Phase {PHASE} composition  (syn_frac target = {SYN_FRAC:.3f}  →  real target = {1-SYN_FRAC:.3f})")
print(f"epoch_size={EPOCH_SIZE}, trials={TRIALS}\n")
print(f"{'window':<12}  {'mean real %':>11}  {'expected':>9}")

# Theoretical window averages assuming a linear ramp from 0 → (1-syn_frac)
# over [0, 0.5], then flat at (1-syn_frac) for [0.5, 1.0].
# Average real% over [a, b] in ramp region = (1-syn_frac) * (a+b) / (2 * 0.5)
#                                           = (1-syn_frac) * (a+b)
expected = {
    "0–25%":    (1 - SYN_FRAC) * (0.00 + 0.25),   # ~11.7%
    "25–50%":   (1 - SYN_FRAC) * (0.25 + 0.50),   # ~35.0%
    "50–75%":   1 - SYN_FRAC,                       # ~46.7%
    "75–100%":  1 - SYN_FRAC,                       # ~46.7%
    "overall":  None,
}

for label, vals in buckets.items():
    mean_real = sum(vals) / len(vals) * 100
    exp = expected[label]
    exp_str = f"{exp*100:6.1f}%" if exp is not None else "    n/a"
    print(f"{label:<12}  {mean_real:>10.1f}%  {exp_str:>9}")

# Assertions
avg = {k: sum(v) / len(v) for k, v in buckets.items()}
tol = 0.03
for label, exp in expected.items():
    if exp is None:
        continue
    assert abs(avg[label] - exp) < tol, \
        f"{label}: expected {exp:.3f}, got {avg[label]:.3f} (tol={tol})"
assert abs(avg["75–100%"] - avg["50–75%"]) < tol, \
    "post-ramp should be flat (50–75% ≈ 75–100%)"

print("\nAll assertions passed.")
