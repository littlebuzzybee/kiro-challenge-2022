#ifndef SOLVE_ORTOOLS_H
#define SOLVE_ORTOOLS_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <set>
#include <map>
#include <deque>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include "utils.h"
#include "breakdown.h"
#include "ortools/linear_solver/linear_solver.h"
#include <memory>




void resolve_traverse(
    Instance&,
    Solution&,
    std::map<int, std::deque<int>>&,
    std::ostream&
);


#endif