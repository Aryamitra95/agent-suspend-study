import numpy as np
from collections import defaultdict

INF = float("inf")

class Cost:
    def __init__(self, s=0.5, r=1.0, rho=0.02, mu=0.5):
        self.s, self.r, self.rho, self.mu = s, r, rho, mu
        self.B = (s + r * (1 + mu)) / (1 - rho)

    def total(self, wait, T):
        if wait <= T:
            return wait, wait, 0.0
        active = T + self.s + self.r + (wait - T) * self.rho
        return active + self.mu * self.r, active, self.r

    def mean_total(self, waits, T):
        w = np.asarray(waits, float)
        with np.errstate(all="ignore"):
            c = np.where(w <= T, w, T + self.s + self.r + (w - T) * self.rho + self.mu * self.r)
        return float(c.mean())

class Fixed:
    def __init__(self, T): self.T = T
    def threshold(self, ev): return self.T
    def update(self, ev, wait): pass

class BreakEven:
    def __init__(self, cost): self.T = cost.B
    def threshold(self, ev): return self.T
    def update(self, ev, wait): pass

class ContinuumStyle:
    GRID = (0, 1, 2, 5, 10, 20, 60, 300, INF)
    def __init__(self, cost, K=100, window=1500, refresh=50):
        self.c, self.K, self.window, self.refresh = cost, K, window, refresh
        self.hist, self.all, self.cache = defaultdict(list), [], {}
        
    def _best(self, h):
        h = h[-self.window:]
        return min(self.GRID, key=lambda T: self.c.mean_total(h, T))
        
    def threshold(self, ev):
        k, n = ev["tool"], len(self.hist[ev["tool"]])
        if n >= self.K:
            if k not in self.cache or n % self.refresh == 0:
                self.cache[k] = self._best(self.hist[k])
            return self.cache[k]
        if len(self.all) >= self.K:
            if "_g" not in self.cache or len(self.all) % (self.refresh * 10) == 0:
                self.cache["_g"] = self._best(self.all)
            return self.cache["_g"]
        return self.c.B
        
    def update(self, ev, wait):
        self.hist[ev["tool"]].append(wait)
        self.all.append(wait)

class PurePredict:
    def __init__(self, cost, q=0.5, window=300):
        self.c, self.q, self.window, self.h = cost, q, window, defaultdict(list)
        
    def threshold(self, ev):
        h = self.h[ev["tool"]][-self.window:]
        if len(h) < 5:
            return self.c.B
        return 0.0 if np.quantile(h, self.q) >= self.c.B else INF
        
    def update(self, ev, wait): self.h[ev["tool"]].append(wait)

class LearningAugmented:
    def __init__(self, cost, lam=0.5, q=0.5, window=300):
        self.c, self.lam, self.q, self.window = cost, lam, q, window
        self.h = defaultdict(list)
        
    def threshold(self, ev):
        h = self.h[ev["tool"]][-self.window:]
        if len(h) < 5:
            return self.c.B
        y = float(np.quantile(h, self.q))
        return self.lam * self.c.B if y >= self.c.B else self.c.B / self.lam
        
    def update(self, ev, wait): self.h[ev["tool"]].append(wait)

def evaluate(events, policy, cost):
    tot = act = opt = 0.0
    lats = []
    for ev, w in events:
        t, a, l = cost.total(w, policy.threshold(ev))
        tot += t; act += a; lats.append(l)
        opt += min(cost.total(w, 0)[0], cost.total(w, INF)[0])
        policy.update(ev, w)
    lats = np.array(lats)
    return {"total_cost": tot, "active_s": act, "ratio_to_oracle": tot / opt,
            "lat_mean": float(lats.mean()), "lat_p99": float(np.percentile(lats, 99))}