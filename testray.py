import numpy as np
from scipy.optimize import linear_sum_assignment as lsa
#import ray

print("[info] starting init ray...")
#ray.init()

print("[info] allocating memory...")
foo = np.random.rand(10_000, 10_000)
#foo_id = ray.put(foo)

#@ray.remote
def solve(arg):
    return lsa(arg)

print("[info] starting solve...")
#row, col = ray.get(solve.remote(foo_id))
row, col = solve(foo)

print(f"sum of costs = {np.sum(foo[row, col])}")
