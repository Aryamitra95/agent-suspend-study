import math
import random
from sim.model import *

rng = random.Random(0)

def draw(tool):
    if tool == "ls":
        return rng.lognormvariate(math.log(0.1), 0.5)
    if tool == "pytest":
        return rng.lognormvariate(math.log(20), 1.0)
    if rng.random() < 0.7:
        return rng.lognormvariate(math.log(0.1), 0.5)
    return rng.lognormvariate(math.log(40), 0.6)

events = []
for i in range(20000):
    tool = rng.choice(["ls", "ls", "pytest", "git"])
    events.append(({"tool": tool, "session": i // 20}, draw(tool)))

cost = Cost()
print(f"break-even wait B = {cost.B:.2f} s\n")

policies = {
    "never_suspend": Fixed(INF), "always_suspend": Fixed(0),
    "fixed_1s": Fixed(1), "fixed_5s": Fixed(5), "fixed_15s": Fixed(15),
    "break_even": BreakEven(cost), "continuum_style": ContinuumStyle(cost),
    "pure_predict": PurePredict(cost),
    "learn_aug_0.3": LearningAugmented(cost, 0.3), "learn_aug_0.7": LearningAugmented(cost, 0.7),
}

print(f"{'policy':16s} {'active_s':>10s} {'ratio_to_best':>14s} {'lat_p99':>8s}")
for name, p in policies.items():
    r = evaluate(events, p, cost)
    print(f"{name:16s} {r['active_s']:10.0f} {r['ratio_to_oracle']:14.3f} {r['lat_p99']:8.2f}")