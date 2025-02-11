#ifndef SOLVE_H
#define SOLVE_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <set>
#include <map>
#include <deque>
#include <set>


#include "gurobi_c++.h"
#include "utils.h"






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

void resolve_simple(
    Instance&,
    Solution&,
    std::map<int, std::deque<int>>&,
    std::ostream&
);

#endif