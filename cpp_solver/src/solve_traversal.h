#ifndef SOLVE_TRAVERSAL_H
#define SOLVE_TRAVERSAL_H

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
#include <armadillo>
#include "utils.h"






void resolve_traversal(
    Instance&,
    Solution&,
    std::map<int, std::deque<int>>&,
    std::ostream&
);


#endif
