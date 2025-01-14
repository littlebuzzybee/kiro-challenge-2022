#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include "nlohmann/json.hpp"


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
};

nlohmann::json read_json_file(std::string);
Instance import_instance(std::string, std::ostream&);
void print_set(std::set<int>, int, std::ostream&);
void print_sequence(std::vector<int>, int, std::ostream&);
void display_cstr_matrix(std::map<int, std::map<int, int>>& matrix, std::ostream& out_stream);
void displayMatrix(const std::map<int, std::map<int, int>>& matrix, std::ostream& out_stream);


#endif