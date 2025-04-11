#ifndef UTILS_H
#define UTILS_H

#include <unordered_set>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <deque>
#include <unordered_map>
#include <iostream>
#include <boost/dynamic_bitset.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <armadillo>
#include "json.hpp"


struct Job {
    int id;
    int release_date;
    int due_date;
    int weight;
    std::vector<int> sequence; // sequence of task indexes (not ids)
};

struct Task {
    int id;
    int processing_time;
    int job_parent;
    std::set<int> machines; // possible machine indexes
    std::map<int, std::set<int>> compatibility; // machine -> operators
    std::set<int> operators; // possible operator indexes
};

struct Instance {
    // size
    int nb_jobs;
    int nb_tasks;
    int nb_machines;
    int nb_operators;

    // costs
    int unit_penalty;
    int tardiness;
    int interim;

    // jobs
    std::vector<Job> jobs;
    std::vector<Task> tasks;
};

struct Solution {
    // time variables
    std::vector<int> begin_time_tasks;
    std::vector<int> completion_date_jobs;
    // choice variables
    std::vector<int> machine_choice_tasks;
    std::vector<int> operator_choice_tasks;

    bool is_valid{ false };
};

struct ExplorationNode {
    boost::dynamic_bitset<> available_machines;
    boost::dynamic_bitset<> available_operators;
    int overhead_tardiness_score; // total overhead score added by the node's decision variables
    int nb_addressed_tasks; // number of tasks addressed by the node, that is, that are actually scheduled with a machine and an operator
    int next_task_vec_idx; // integer pointer to the next index of the task vector to address; if > #tasks, the node is a terminal leaf
    std::vector<int> assigned_tasks;
    std::vector<int> chosen_machines; // task index -> machine index
    std::vector<int> chosen_operators; // task index -> operator index
};

nlohmann::json read_json_file(std::string);
Instance import_instance(std::string, std::ostream&);

void print_set(std::set<int>, int, std::ostream&);

void print_sequence(std::vector<int>, int, std::ostream&);

void display_cstr_matrix(std::map<int, std::map<int, int>>&, std::ostream&);

void displayMatrix(const std::map<int, std::map<int, int>>&, std::ostream&);

bool all_stacks_are_empty(const std::map<int, std::deque<int>>&);

void get_sort_tasks_and_scores(
    std::vector<std::tuple<float, int>>&,
    Instance&,
    std::map<int, std::deque<int>>&,
    std::map<int, int>&,
    std::map<int, int>&,
    int
);

void get_cumulative_remaining_time_per_job(
    std::map<int, int>&,
    Instance&,
    std::map<int, std::deque<int>>&,
    std::map<int, int>&
);

int compute_loss(const Instance&, const Solution&);

void print_job_stacks(const std::map<int, std::deque<int>>& job_stacks, std::ostream& log_stream);

bool check_validity(const Instance&, const Solution&);

void node_analysis(const ExplorationNode&, const Instance&, const Solution&);

#endif