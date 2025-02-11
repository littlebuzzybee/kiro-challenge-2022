#ifndef BREAKDOWN_H
#define BREAKDOWN_H


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



void get_relevant_tasks(
    Instance&,
    Solution&,
    int,
    int,
    std::map<int, std::deque<int>>&,
    std::vector<int>&,
    std::vector<int>&,
    std::map<int, std::vector<int>>&,
    std::map<int, int>&,
    std::unordered_map<int, int>&
);


void display_lookahead_program(
    int,
    std::vector<int>&,
    std::vector<int>&,
    std::map<int, std::vector<int>>&,
    std::map<int, std::deque<int>>&,
    std::unordered_map<int, int>&,
    std::ostream&
);




void set_begin_variables_and_ordering_constraints(
    Instance&,
    Solution&,
    GRBModel&,
    std::map<int, std::map<int, GRBVar>>&,
    std::map<int, std::vector<int>>&,
    std::vector<int>&,
    std::unordered_map<int, int>&,
    int
);

void set_slack_and_penalty_variables(
    GRBModel&,
    std::map<int, GRBVar>&,
    std::map<int, GRBVar>&,
    std::vector<int>&
);


void set_assignment_variables(
    Instance&,
    GRBModel&,
    std::map<int, std::map<int, GRBVar>>&,
    std::map<int, std::map<int, GRBVar>>&,
    std::vector<int>&
);


void set_workers_ubiquity_constraints(
    Instance&,
    GRBModel&,
    std::map<int, std::map<int, GRBVar>>&,
    std::map<int, std::map<int, GRBVar>>&,
    std::vector<int>&
);

void set_workers_uniqueness_constraints(
    Instance&,
    GRBModel&,
    std::map<int, std::map<int, GRBVar>>&,
    std::map<int, std::map<int, GRBVar>>&,
    std::vector<int>&
);

void set_workers_time_exclusion_constraints(
    Instance&,
    Solution&,
    GRBModel&,
    std::map<int, std::map<int, GRBVar>>&,
    std::map<int, std::map<int, GRBVar>>&,
    std::map<int, std::map<int, GRBVar>>&,
    std::vector<int>&,
    std::map<int, std::vector<int>>&,
    std::unordered_map<int, int>&,
    std::ostream&
);


void set_completion_time_and_penalty_constraints(
    Instance&,
    GRBModel&,
    std::map<int, GRBVar>&,
    std::map<int, GRBVar>&,
    std::map<int, GRBLinExpr>&,
    std::map<int, std::map<int, GRBVar>>&,
    std::vector<int>&,
    std::map<int, std::vector<int>>&,
    std::map<int, int>&
);


void set_objective_function(
    Instance&,
    GRBModel&,
    std::vector<int>&,
    std::map<int, GRBLinExpr>&,
    std::map<int, GRBVar>&,
    std::map<int, GRBVar>&
    );



void greedy_partial_solve_lookahead(
    Instance&,
    Solution&,
    int,
    int,
    std::map<int, std::deque<int>>&,
    std::unordered_map<int, int>&,
    std::ostream&
);

void warmup_solution(
    Instance&,
    Solution&,
    std::vector<int>,
    std::map<int, std::map<int, GRBVar>>,
    std::map<int, std::map<int, GRBVar>>,
    std::map<int, std::map<int, GRBVar>>
);

#endif