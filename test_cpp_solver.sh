./cpp_solver/objects/scheduler.o ./instances/huge.json --method=heuristic --lookahead=5 --gurobi_threads=6 --time_limit=10.0

# Sweet spots:
# huge: lookahead=2
# large: lookahead=3
# medium: lookahead=4 (heuristic does better anyway)
# tiny: 20 (heuristic provides the optimal value anyway)