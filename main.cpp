#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <set>
#include <map>
#include <deque>
#include <set>
#include <execution>
#include <numeric>


#include "gurobi_c++.h"
#include "utils.h"
#include "solve.h"
#include "nlohmann/json.hpp"




int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <instance_filename> <options>" << std::endl;
        return 1;
    }


    // Detect flags
    // Detect --writeproblemfile flag



    // Import instance
    std::string instance_filename = argv[1];

    double time_limit = 30.0;
    int lookahead_duration = 5;
    bool report_all_solutions = false;
    int max_threads = 3;
    bool write_problem_file = false;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--gurobi_threads=") == 0) {
            std::string value = arg.substr(17);
            max_threads = std::stoi(value);
            std::cout << "Gurobi threads set to " << max_threads << "." << std::endl;
        }
        if (arg.find("--time_limit=") == 0) {
            std::string value = arg.substr(13);
            time_limit = std::stod(value);
            std::cout << "Time limit set to " << time_limit << " seconds." << std::endl;
        }
        if (arg.find("--lookahead=") == 0) {
            std::string value = arg.substr(12);
            lookahead_duration = std::stoi(value);
            std::cout << "Lookahead duration set to " << lookahead_duration << " time units." << std::endl;
        }
        if (arg.find("--write_problem_file") == 0) {
            write_problem_file = true;
            std::cout << "Problem files will be written." << std::endl;
        }
    }





    std::ofstream log_file("solve.log");
    Instance inst = import_instance(instance_filename, log_file);

    std::map<int, std::deque<int>> job_stacks;
    std::vector<int> total_time_per_job(inst.nb_jobs, 0);
    for (int j_idx = 0; j_idx < inst.nb_jobs; j_idx++) {

        std::vector<int>& task_sequence = inst.jobs[j_idx].sequence;

        job_stacks[j_idx] = std::deque<int>(task_sequence.begin(), task_sequence.end());
        // Front of the deque has the lowest task indexes, i.e. the first to process in order

        int total_processing_time = std::accumulate(
            task_sequence.begin(),
            task_sequence.end(),
            0, // Initial value of the sum
            [&inst](int total_sum, int t_idx) {
                return total_sum + inst.tasks[t_idx].processing_time;
            }
        );

        total_time_per_job[j_idx] = total_processing_time;
    }

    log_file << std::setw(5) << "J" << std::setw(8) << "W_J" << std::setw(8) << "\u03A3d_T" << std::setw(8) << "t_rel" << std::setw(8) << "t_due" << std::endl;
    for (int j_idx = 0; j_idx < inst.nb_jobs; j_idx++) {
        log_file << std::setw(5) << j_idx + 1
            << std::setw(8) << inst.jobs[j_idx].weight
            << std::setw(8) << total_time_per_job[j_idx]
            << std::setw(8) << inst.jobs[j_idx].release_date
            << std::setw(8) << inst.jobs[j_idx].due_date << std::endl;
    }


    // Begin solving procedure
    log_file << "Beginning solving procedure with lookahead duration " << lookahead_duration << "..." << std::endl;

    Solution sol;
    // Initialize the solution's vectors
    // time variables
    sol.begin_time_tasks.assign(inst.nb_tasks, 0);
    // sol.end_time_tasks.assign(inst.nb_tasks, 0);
    sol.completion_date_jobs.assign(inst.nb_jobs, 0);
    // choice variables
    sol.machine_choice_tasks.assign(inst.nb_tasks, 0);
    sol.operator_choice_tasks.assign(inst.nb_tasks, 0);



    int time_cursor = 0;



    // Declare the set of pending tasks (should be a small set of indices between each iteration since tasks durations are quite limited in comparison to the lookahead duration)
    std::unordered_map<int, int> pending_tasks_per_job{};

    // Declare and initialize the indicators of resource pool intersections for operators and machines
    // std::vector<std::vector<int>> intersect_operators_; // binary indicator if non-empty intersection
    // std::map<std::pair<int, int>, std::set<int>> intersect_operators; // actual intersection

    // std::vector<std::vector<int>> intersect_machines_;
    // std::map<std::pair<int, int>, std::set<int>> intersect_machines;

    // Initialize the integer indicators
    // intersect_operators_.assign(inst.nb_operators, std::vector<int>(inst.nb_operators, -1)); // -1 for not yet computed
    // intersect_machines_.assign(inst.nb_machines, std::vector<int>(inst.nb_machines, -1));



    resolve_lookahead(
        inst, sol, job_stacks, pending_tasks_per_job,
        time_cursor, lookahead_duration, time_limit, max_threads, report_all_solutions,
        write_problem_file, log_file
    );


    // TODO: set time limit as a parameter
    // Implement Warm start
    // Set the log stream as a parameter
    log_file.close();
    return 0;
}


