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
#include "solve.h"
#include "nlohmann/json.hpp"




int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <instance_filename> <lookahead_duration> <options>" << std::endl;
        return 1;
    }

    // Detect flags
    // Detect --writeproblemfile flag
    bool write_problem_file = false;
    for (int i = 3; i < argc; i++) {
        if (std::string(argv[i]) == "--writeproblemfile") {
            write_problem_file = true;
        }
    }


    // Import instance
    std::string instance_filename = argv[1];
    const int lookahead_duration = std::stoi(argv[2]);
    std::ofstream log_file("solve.log");
    Instance inst = import_instance(instance_filename, log_file);


    // Begin solving procedure
    log_file << "Beginning solving procedure with lookahead duration " << lookahead_duration << "..." << std::endl;

    Solution sol;
    // Initialize the solution's vectors
    // time variables
    sol.begin_time_tasks.assign(inst.nb_tasks, 0);
    sol.end_time_tasks.assign(inst.nb_tasks, 0);
    sol.completion_date_jobs.assign(inst.nb_jobs, 0);
    // choice variables
    sol.machine_choice_tasks.assign(inst.nb_tasks, 0);
    sol.operator_choice_tasks.assign(inst.nb_tasks, 0);


    greedy_initialize_time_scheduling(inst, sol, log_file);
    int time_cursor = 0;

    std::map<int, std::deque<int>> job_stacks;
    for (int j_idx = 0; j_idx < inst.nb_jobs; j_idx++) {
        job_stacks[j_idx] = std::deque<int>(inst.jobs[j_idx].sequence.begin(), inst.jobs[j_idx].sequence.end());
    }
    // Declare the set of pending tasks (should be a small set of indices between each iteration since tasks durations
    // are quite limited in comparison to the lookahead duration)
    std::unordered_map<int, int> pending_tasks_per_job{}; // empty at first

    // Declare and initialize the indicators of resource pool intersections for operators and machines
    std::vector<std::vector<int>> intersect_operators_; // binary indicator if non-empty intersection
    std::map<std::pair<int, int>, std::set<int>> intersect_operators; // actual intersection

    std::vector<std::vector<int>> intersect_machines_;
    std::map<std::pair<int, int>, std::set<int>> intersect_machines;

    // Initialize the integer indicators
    intersect_operators_.assign(inst.nb_operators, std::vector<int>(inst.nb_operators, -1)); // -1 for not yet computed
    intersect_machines_.assign(inst.nb_machines, std::vector<int>(inst.nb_machines, -1));


    resolve_lookahead(inst, sol, job_stacks, pending_tasks_per_job, time_cursor, lookahead_duration, std::cout);

    log_file.close();
    return 0;
}


