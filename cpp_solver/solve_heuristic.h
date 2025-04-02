#ifndef SOLVE_HEURISTIC_H
#define SOLVE_HEURISTIC_H

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






void resolve_simple(
    Instance&,
    Solution&,
    std::map<int, std::deque<int>>&,
    std::ostream&
);


#endif