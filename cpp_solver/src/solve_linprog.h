#ifndef SOLVE_LINPROG_H
#define SOLVE_LINPROG_H

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
#include <memory>
#include <cmath>
#include "ortools/linear_solver/linear_solver.h"





void resolve_linprog(
    Instance&,
    Solution&,
    std::map<int, std::deque<int>>&,
    std::ostream&
);


#endif
