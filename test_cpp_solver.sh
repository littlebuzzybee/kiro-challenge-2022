#!/bin/bash

INSTANCE=$1
METHOD=$2
LOOKAHEAD=$3

./cpp_solver/objects/scheduler.o ./instances/$INSTANCE.json --method=$METHOD --lookahead=$LOOKAHEAD --gurobi_threads=6 --time_limit=20.0

# Sweet spots for solver mode:
# huge: lookahead=2
# large: lookahead=3
# medium: lookahead=4 (heuristic does better anyway)
# tiny: 20 (heuristic provides the optimal value anyway)