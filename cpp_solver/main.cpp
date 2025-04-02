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
#include <chrono>

#include "gurobi_c++.h"
#include "utils.h"
#include "solve_gurobi.h"
#include "solve_heuristic.h"
#include "solve_ortools.h"
#include "json.hpp"





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
    bool report_all_decisions = false;
    int max_threads = 3;
    bool write_problem_file = false;
    int method_code = 2;


    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--gurobi_threads=") == 0) {
            std::string value = arg.substr(17);
            max_threads = std::stoi(value);
            assert(max_threads > 0);
        }
        if (arg.find("--time_limit=") == 0) {
            std::string value = arg.substr(13);
            time_limit = std::stod(value);
            assert(time_limit > 0);
        }
        if (arg.find("--lookahead=") == 0) {
            std::string value = arg.substr(12);
            lookahead_duration = std::stoi(value);
            assert(lookahead_duration > 0);
        }
        if (arg.find("--write_problem_file") == 0) {
            write_problem_file = true;
        }
        if (arg.find("--report_all_decisions") == 0) {
            report_all_decisions = true;
        }
        if (arg.find("--method=") == 0) {
            std::string method = arg.substr(9);
            if (method == "solver") {
                method_code = 1;
            }
            else if (method == "heuristic") {
                method_code = 2;
            }
            else if (method == "heuristic_smart") {
                method_code = 3;
            }
            else {
                std::cerr << "Invalid method provided." << std::endl;
                exit(1);
            }
        }
    }



    std::ofstream log_solve("./log_solve.log");
    std::ofstream log_import("./log_import.log");
    std::ostream& log_inform = std::cout;


    Instance inst = import_instance(instance_filename, log_import);

    // Fill the jobs stacks with the tasks
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

    log_inform << std::endl << "===== Instance Details: =====" << std::endl;
    log_inform << std::setw(5) << "J" << std::setw(8) << "W_J" << std::setw(8) << "\u03A3d_T" << std::setw(8) << "t_rel" << std::setw(8) << "t_due" << std::endl;
    for (int j_idx = 0; j_idx < inst.nb_jobs; j_idx++) {
        log_inform << std::setw(5) << j_idx + 1
            << std::setw(8) << inst.jobs[j_idx].weight
            << std::setw(8) << total_time_per_job[j_idx]
            << std::setw(8) << inst.jobs[j_idx].release_date
            << std::setw(8) << inst.jobs[j_idx].due_date << std::endl;
    }


    Solution sol;
    // Initialize the solution's vectors

    // time variables
    sol.begin_time_tasks.assign(inst.nb_tasks, -1);
    sol.completion_date_jobs.assign(inst.nb_jobs, -1);
    // decision variables
    sol.machine_choice_tasks.assign(inst.nb_tasks, -1);
    sol.operator_choice_tasks.assign(inst.nb_tasks, -1);


    // Declare the set of pending tasks (should be a small set of indices between each iteration since tasks durations are quite limited compared to the lookahead duration)
    std::unordered_map<int, int> pending_tasks_per_job{};

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point stop;

    switch (method_code)
    {
    case 1:
        start = std::chrono::high_resolution_clock::now();
        resolve_lookahead(
            inst, sol, job_stacks, pending_tasks_per_job,
            lookahead_duration, time_limit, max_threads, write_problem_file, report_all_decisions, log_solve, log_inform
        );
        stop = std::chrono::high_resolution_clock::now();
        break;
    case 2:
        start = std::chrono::high_resolution_clock::now();
        resolve_simple(
            inst,
            sol,
            job_stacks,
            log_solve
        );
        stop = std::chrono::high_resolution_clock::now();
        break;
    case 3:
        start = std::chrono::high_resolution_clock::now();
        resolve_traverse(
            inst,
            sol,
            job_stacks,
            std::cout
        );
        stop = std::chrono::high_resolution_clock::now();
        break;
    }



    // Now display all decisions in the solution
    log_inform << std::endl << "===== Solution Details: =====" << std::endl;
    log_inform << std::setw(5) << "Task" << std::setw(8) << "Begin" << std::setw(10) << "Machine" << std::setw(10) << "Operator" << std::endl;
    for (int t_idx = 0; t_idx < inst.nb_tasks; t_idx++) {
        log_inform << std::setw(5) << "T" << t_idx + 1
            << std::setw(8) << sol.begin_time_tasks[t_idx]
            << std::setw(10) << "M" << sol.machine_choice_tasks[t_idx] + 1
            << std::setw(10) << "O" << sol.operator_choice_tasks[t_idx] + 1
            << std::endl;
    }

    int total_loss = compute_loss(inst, sol);
    log_inform << "Total loss: " << total_loss << std::endl;

    std::chrono::duration<double> duration;
    duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    if (duration.count() >= 5) {
        log_inform << "Solution computed solution in " << duration.count() << " s." << std::endl;
    }
    else {
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        log_inform << "Solution computed solution in " << duration.count() << " \u33B2." << std::endl;
    }

    // Check the validity of the solution
    if (check_validity(inst, sol)) {
        log_inform << "The solution is valid." << std::endl;
    }
    else {
        log_inform << "The solution is invalid." << std::endl;
    }

    log_solve.close();
    log_import.close();
    return 0;
}


