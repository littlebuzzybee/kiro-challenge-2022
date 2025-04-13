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
#include "solve_greedy.h"
#include "solve_linprog.h"
#include "solve_traversal.h"
#include "json.hpp"
#include "CLI/CLI.hpp"






int main(int argc, char* argv[]) {
    std::string greetings = 
        "================ Kiro 2022 Solver ================\n"
        "       A solver for the Kiro 2022 problem\n"
        "      with several algorithms to choose from.\n"
        "==================================================";
    CLI::App app{ greetings };


    std::filesystem::path input_filename;
    CLI::Option* ifo = app.add_option("-f,--file", input_filename, "The JSON instance file to read from and solve");
    ifo->required();
    ifo->check(CLI::ExistingFile.description("File must exist").active(false).name("file"));
    ifo->get_validator("file")->active();

    std::string output_filename;
    CLI::Option* ofo = app.add_option("-o,--output", output_filename, "The log file to write the solution to");

    bool write_problem_file{ false };
    app.add_flag("-w,--write_solution", write_problem_file, "Write problem solution to file");

    double solver_time_limit { 10.0 };
    CLI::Option* stl = app.add_option("-t,--time_limit", solver_time_limit, "Time limit for each of Gurobi's steps in seconds");
    stl->default_val(10.0);
    stl->check(CLI::Number.description("Time limit must be a number").active(false).name("type"));
    stl->check(CLI::PositiveNumber.description("Time limit must be positive").active(false).name("sign"));
    stl->get_validator("type")->active();
    stl->get_validator("sign")->active();

    int lookahead_duration { 2 };
    CLI::Option* lad = app.add_option("-l,--lookahead", lookahead_duration, "Lookahead duration in seconds");
    lad->default_val(2);
    lad->check(CLI::Number.description("Lookahead duration must be a number").active(false).name("type"));
    lad->check(CLI::PositiveNumber.description("Lookahead duration must be positive").active(false).name("sign"));
    lad->get_validator("type")->active();
    lad->get_validator("sign")->active();

    bool report_all_decisions = false;
    app.add_flag("-r,--report_all_decisions", report_all_decisions, "Report all decisions made by the solver");

    int gurobi_max_threads { 4 };
    CLI::Option* gto = app.add_option("-g,--gurobi_threads", gurobi_max_threads, "Number of threads for Gurobi solver");
    gto->default_val(4);
    gto->check(CLI::Number.description("Number of threads must be a number").active(false).name("type"));
    gto->check(CLI::PositiveNumber.description("Number of threads must be positive").active(false).name("sign"));
    gto->get_validator("type")->active();
    gto->get_validator("sign")->active();


    std::string method;
    CLI::Option* mth = app.add_option("-m,--method", method, "Method to use for solving the problem");
    mth->required();
    mth->check(CLI::IsMember({ "solver", "greedy", "linprog", "traversal" }, CLI::ignore_case).description("Method must be one of: solver, greedy, linprog, traversal").active(false).name("method"));
    mth->get_validator("method")->active();



    app.footer("Kiro 2022 Solver - A solver for the Kiro 2022 problem");
    CLI11_PARSE(app, argc, argv);


    int method_code = 2;


    // Define the mapping
    std::map<std::string, int> method_map = {
        { "solver", 1 },
        { "greedy", 2 },
        { "linprog", 3 },
        { "traversal", 4 }
    };

    // Look up the method code
    auto it = method_map.find(method);
    if (it != method_map.end()) {
        method_code = it->second;
    }
    else {
        std::cerr << "Invalid method provided." << std::endl;
        exit(1);
    }




    std::ofstream log_solve("./cpp_solve.log");
    std::ostream& log_inform = std::cout;


    Instance inst = import_instance(input_filename, log_solve);

    // Fill the jobs stacks with the tasks
    std::map<int, std::deque<int>> job_stacks;
    std::vector<int> total_time_per_job(inst.nb_jobs, 0);
    fill_job_stacks_and_compute_time(
        inst,
        job_stacks,
        total_time_per_job
    );

    log_inform << std::endl << "===== Instance Details: =====" << std::endl;
    log_inform << std::setw(5) << "J" << std::setw(8) << "W_J" << std::setw(8) << "\u03A3d_T" << std::setw(8) << "t_rel" << std::setw(8) << "t_due" << std::endl;
    for (int j_idx = 0; j_idx < inst.nb_jobs; j_idx++) {
        log_inform << std::setw(5) << j_idx + 1
            << std::setw(8) << inst.jobs[j_idx].weight
            << std::setw(8) << total_time_per_job[j_idx]
            << std::setw(8) << inst.jobs[j_idx].release_date
            << std::setw(8) << inst.jobs[j_idx].due_date << std::endl;
    }

    print_job_stacks(job_stacks, log_inform);


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
            lookahead_duration, solver_time_limit, gurobi_max_threads, write_problem_file, report_all_decisions, log_solve, log_inform
        );
        stop = std::chrono::high_resolution_clock::now();
        break;
    case 2:
        start = std::chrono::high_resolution_clock::now();
        resolve_greedy(
            inst,
            sol,
            job_stacks,
            log_solve
        );
        stop = std::chrono::high_resolution_clock::now();
        break;
    case 3:
        start = std::chrono::high_resolution_clock::now();
        resolve_linprog(
            inst,
            sol,
            job_stacks,
            log_solve
        );
        stop = std::chrono::high_resolution_clock::now();
        break;
    case 4:
        start = std::chrono::high_resolution_clock::now();
        resolve_traversal(
            inst,
            sol,
            job_stacks,
            log_solve
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
    log_inform << "Solution computed solution under ";
    if (duration.count() >= 5) {
        log_inform << duration.count() << " s." << std::endl;
    }
    else {
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        log_inform << duration.count() << " \u33B2." << std::endl;
    }

    // Check the validity of the solution
    if (check_validity(inst, sol)) {
        log_inform << "The solution is valid." << std::endl;
        sol.is_valid = true;
    }
    else {
        log_inform << "The solution is invalid." << std::endl;
    }

    log_solve.close();

    return 0;
}


