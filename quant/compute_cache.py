"""一次性计算因子矩阵并缓存(F/health/Fz 方向对齐版),后续脚本直接读。"""
import os
import pickle, time
import numpy as np
import panel, factors, research

BASE = os.environ.get("FACTORLAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
t0 = time.time()
p = panel.build_panel()
F = research.direction_fix(factors.compute_factors(p))
health = research.ichealth(p, F)
F = research.empirical_direction(F, health)
Fz = research.zscore(F)
chosen_sign = {r['factor']: r['ic_mean'] for r in health}
with open(f"{BASE}/data/fz_cache.pkl", 'wb') as f:
    pickle.dump({'F': F, 'Fz': Fz, 'health_ic': chosen_sign}, f)
print(f"cache saved · {time.time()-t0:.0f}s · ic:", {k: round(v,3) for k,v in list(chosen_sign.items())[:4]})
