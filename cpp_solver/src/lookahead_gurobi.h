#ifndef LOOKAHEAD_GUROBI_H
#define LOOKAHEAD_GUROBI_H

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
#include "gurobi_c++.h"
#include "lookahead_breakdown.h"





void resolve_lookahead(
    Instance&,
    Solution&,
    std::map<int, std::deque<int>>&,
    std::unordered_map<int, int>&,
    int,
    double,
    int,
    bool,
    bool,
    std::ostream&,
    std::ostream&
);

#endif