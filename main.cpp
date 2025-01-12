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
#include "nlohmann/json.hpp"

// #define QUADRATIC



void greedy_initialize_time_scheduling(Instance& inst, Solution& sol, std::ostream& log_stream = std::cout) {
    log_stream << "Stacking greedily tasks in time..." << std::endl;
    for (int j_idx = 0; j_idx < inst.nb_jobs; j_idx++) {
        int total_task_offset = 0; // cumulative sum of all task processing times for the current job
        log_stream << "~~~~~ Processing job " << j_idx + 1 << " ~~~~~" << std::endl;
        for (int t_idx = 0; t_idx < (int)inst.jobs[j_idx].sequence.size(); t_idx++) {
            int processed_task = inst.jobs[j_idx].sequence[t_idx];

            sol.begin_time_tasks[processed_task] = inst.jobs[j_idx].release_date
                + total_task_offset;

            sol.end_time_tasks[processed_task] = inst.jobs[j_idx].release_date
                + total_task_offset
                + inst.tasks[processed_task].processing_time;

            log_stream << "Set T" << processed_task + 1 << ": slot [b,e] = [" << sol.begin_time_tasks[processed_task] << "," << sol.end_time_tasks[processed_task] << "]" << std::endl;
            // update time offset with current task
            total_task_offset += inst.tasks[processed_task].processing_time;
        }
        sol.completion_date_jobs[j_idx] = inst.jobs[j_idx].release_date + total_task_offset;
        log_stream << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
        log_stream << "Job " << j_idx << " completion date: " << sol.completion_date_jobs[j_idx] << std::endl << std::endl;
    }
}



void resolve_lookahead(Instance& inst, Solution& sol, std::map<int, std::deque<int>> job_stacks, std::unordered_map<int, int>& pending_tasks_per_job, int time_cursor, int lookahead_duration, std::ostream& log_stream = std::cout) {


    const int time_horizon = time_cursor + lookahead_duration;
    log_stream << "Resolving lookahead on time window [" << time_cursor << ", " << time_horizon << "]..." << std::endl;


    std::map<int, std::vector<int>> processed_tasks_of_jobs; // ordered set of tasks for each job
    std::set<int> processed_jobs;
    // std::set<int> processed_tasks; // ordered redundancy of tasks used for assignment consraints between tasks later on
    std::vector<int> processed_tasks;



    // Begin by identifying the tasks that are relevant in the lookahead window
    for (int job_idx = 0; job_idx < inst.nb_jobs; ++job_idx) {
        if (job_stacks[job_idx].empty() || inst.jobs[job_idx].release_date >= time_horizon) {
            continue;
        }
        else {
            // Only the tasks that are fully comprised in the time window are considered.
            // Those which started before the time window but end in the time window are considered processed, fixed and pending and would not benefit the problem
            // by being postponed again. Theirs implications are taken into account by the pending_tasks set and not recomputed here.
            processed_jobs.insert(job_idx); // inserts an index, not an id
            while (!job_stacks[job_idx].empty()
                && time_cursor <= sol.begin_time_tasks[job_stacks[job_idx].front()]
                && sol.begin_time_tasks[job_stacks[job_idx].front()] < time_horizon
                ) {
                int task_idx = job_stacks[job_idx].front();
                job_stacks[job_idx].pop_front();
                processed_tasks_of_jobs[job_idx].push_back(task_idx);
                processed_tasks.push_back(task_idx);
            }
            // Finally also insert the tasks to be processed together in the processed_tasks set
            // processed_tasks.insert(processed_tasks_of_jobs[job_idx].begin(), processed_tasks_of_jobs[job_idx].end());
        }
    }

    processed_tasks.shrink_to_fit();
    std::sort(processed_tasks.begin(), processed_tasks.end());



    int nb_processed_jobs = processed_jobs.size();
    int nb_pending_tasks = pending_tasks_per_job.size();
    int nb_processed_tasks = processed_tasks.size();



    // Print the list of processed tasks and jobs in the log stream
    log_stream << "There are " << nb_processed_jobs << " jobs processed." << std::endl;

    log_stream << "They comprise the following processed tasks in the lookahead: ";
    for (int task_idx : processed_tasks) {
        log_stream << ";" << task_idx;
    }
    log_stream << " distributed as follows: " << std::endl << std::endl;

    for (int job_idx : processed_jobs) {
        log_stream << "Job " << job_idx << " entails " << processed_tasks_of_jobs[job_idx].size() << " tasks ordered as:   ";
        for (int task_idx : processed_tasks_of_jobs[job_idx]) {
            log_stream << "|" << task_idx;
        }
        log_stream << "|" << std::endl;
    }
    log_stream << std::endl << "There are " << nb_pending_tasks << " pending tasks in total." << std::endl;

    // Compute the maximum duration of the processed tasks
    int max_duration = 0;
    max_duration = *std::max_element(processed_tasks.begin(), processed_tasks.end(),
        [&inst](int t1, int t2) {
            return inst.tasks[t1].processing_time < inst.tasks[t2].processing_time;
        });
    log_stream << "Max duration of processed tasks: " << max_duration << std::endl;


    // Initialize Gurobi environment and model
    log_stream << "Initializing Gurobi environment and model..." << std::endl;
    GRBEnv env = GRBEnv(true);
    env.set("LogFile", "gurobi.log");
    env.start();
    GRBModel model = GRBModel(env);
    model.set(GRB_StringAttr_ModelName, "Time_Scheduling_Round_" + std::to_string(1));
    model.set(GRB_IntParam_OutputFlag, 1);
    model.set(GRB_IntParam_Threads, 5);
    //model.set(GRB_DoubleParam_TimeLimit, 30.0);


    // Declare the begin times of each task and set the ordering constraints
    std::map<int, std::map<int, GRBVar>> begin_times_tasks_per_job;

    log_stream << "Declaring scheduling variables and constraints for processed job..." << std::endl;
    for (int job_idx : processed_jobs) {

        // First, Declare the begin times variables of each task
        for (int task_idx : processed_tasks_of_jobs[job_idx]) {

            begin_times_tasks_per_job[job_idx].emplace(
                task_idx,
                model.addVar(
                    time_cursor, GRB_INFINITY, 0.0, GRB_INTEGER,
                    "begin_time_T" + std::to_string(task_idx + 1)
                )
            );
        }



        // Then, set the ordering constraints
        for (int i = 0; i < int(processed_tasks_of_jobs[job_idx].size()); i++) {
            int task_idx = processed_tasks_of_jobs[job_idx][i];

            if (task_idx == processed_tasks_of_jobs[job_idx].front()) {
                // If this is the first task of the job being optimized in the window
                if (pending_tasks_per_job.contains(job_idx)) {
                    // If there is a pending task for this job overlapping the window's beginning,
                    // its end time is greater than the beginning of the window,
                    // so we set instead the beginning of the current task in loop after the end of the pending task at the soonest

                    int previous_fixed_task = pending_tasks_per_job[job_idx];
                    model.addConstr(
                        GRBLinExpr(sol.end_time_tasks[previous_fixed_task]),
                        GRB_LESS_EQUAL,
                        begin_times_tasks_per_job[job_idx][task_idx],
                        "earliest_begin_T" + std::to_string(task_idx + 1) // EST = earliest start time
                    );
                }
                else {
                    // If there is no pending task for this job overlapping the window's beginning,
                    // then we set the lower bound of the beginning of the task at the time cursor posution (beginning of the window)
                    model.addConstr(
                        GRBLinExpr(time_cursor),
                        GRB_LESS_EQUAL,
                        begin_times_tasks_per_job[job_idx][task_idx],
                        "earliest_begin_T" + std::to_string(task_idx + 1) // EST = earliest start time
                    );
                }
            }
            else if (
                (task_idx != processed_tasks_of_jobs[job_idx].front())
                && (task_idx != processed_tasks_of_jobs[job_idx].back())
                ) {
                // If this is not the first nor the last task of the job being optimized in the window,
                // we prevent the task from starting before the end of the previous task (no overlapping)
                int next_task_idx = processed_tasks_of_jobs[job_idx][i + 1];


                model.addConstr(
                    GRBLinExpr(begin_times_tasks_per_job[job_idx][task_idx] + inst.tasks[task_idx].processing_time),
                    GRB_LESS_EQUAL,
                    begin_times_tasks_per_job[job_idx][next_task_idx],
                    "precedence_T" + std::to_string(task_idx + 1) + "_T" + std::to_string(next_task_idx + 1)
                );
            }
        }
    }


    // Declare slacks and unit penalties variables for each job
    std::map<int, GRBVar> tardiness_post_slacks;
    std::map<int, GRBVar> unit_penalties;

    log_stream << "Adding slack and penalty variables for all jobs. " << std::endl;
    for (int job_idx : processed_jobs) {
        tardiness_post_slacks[job_idx] = model.addVar(
            0.0, GRB_INFINITY, 0.0, GRB_INTEGER,
            "due_date_slack_J" + std::to_string(job_idx + 1)
        );
        unit_penalties[job_idx] = model.addVar(
            0.0, 1.0, 0.0, GRB_BINARY,
            "unit_overdue_pen_J" + std::to_string(job_idx + 1)
        );
    }



    // Define the resulting completion dates variable of each processed job
    // as the sum of its current completion date and the postponement due to task delays
    std::map<int, GRBLinExpr> additional_delays_jobs;
    std::map<int, GRBLinExpr> new_completion_dates_jobs;
    log_stream << "Adding completion date variable for all jobs..." << std::endl;
    for (int job_idx : processed_jobs) {
        int last_task_of_job = processed_tasks_of_jobs[job_idx].back();

        additional_delays_jobs[job_idx] = GRBLinExpr(begin_times_tasks_per_job[job_idx][last_task_of_job] - sol.begin_time_tasks[last_task_of_job]);
        new_completion_dates_jobs[job_idx] = GRBLinExpr(sol.completion_date_jobs[job_idx] + additional_delays_jobs[job_idx]);
    }


    // Set constraint for completion time: completion_time <= due_date + slack
    log_stream << "Adding completion date constraint for all jobs..." << std::endl;
    for (int job_idx : processed_jobs) {
        model.addConstr(
            new_completion_dates_jobs[job_idx] <= inst.jobs[job_idx].due_date + tardiness_post_slacks[job_idx],
            "slack_deadline_J" + std::to_string(job_idx + 1)
        );
    }


    // Set unit penalty variables
    log_stream << "Adding unit penalty constraint on all jobs..." << std::endl;
    for (int job_idx : processed_jobs) {
        // completion_time > due_date => unit_penalty = 1
        // https://docs.gurobi.com/projects/optimizer/en/current/reference/cpp/model.html#_CPPv4N8GRBModel21addGenConstrIndicatorE6GRBVariRK10GRBLinExprcd6string
        model.addGenConstrIndicator(
            unit_penalties[job_idx],
            0, // false
            new_completion_dates_jobs[job_idx], GRB_LESS_EQUAL, inst.jobs[job_idx].due_date,
            "bind_unit_pen_J" + std::to_string(job_idx + 1)
        );
    }



    // Declare the assigned machines and operators variables for every task
    log_stream << "Declaring assignment variables..." << std::endl;
    std::map<int, std::map<int, GRBVar>> assigned_operators_per_task;
    std::map<int, std::map<int, GRBVar>> assigned_machines_per_task;
    // first index is the task index, second index is the operator/machine index
    // resulting function is T_idx --> {operator_idx/machine_idx} --> GRBVar



    for (int task_idx : processed_tasks) {
        // Declare the machine assignment binary variables for the task
        for (int machine_idx : inst.tasks[task_idx].machines) {
            assigned_machines_per_task[task_idx].emplace(
                machine_idx,
                model.addVar(
                    0, 1, 0.0, GRB_BINARY,
                    "T" + std::to_string(task_idx + 1) + "_uses_M" + std::to_string(machine_idx + 1))
            );
        }

        // Declare the operator assignment binary variable for the task
        for (int operator_idx : inst.tasks[task_idx].operators) {
            assigned_operators_per_task[task_idx].emplace(
                operator_idx,
                model.addVar(
                    0, 1, 0.0, GRB_BINARY,
                    "T" + std::to_string(task_idx + 1) + "_uses_O" + std::to_string(operator_idx + 1))
            );
        }
    }


    // Set the assignments physical overlap constraints
    log_stream << "Setting assignments physical overlap constraints..." << std::endl;

    for (int task_idx : processed_tasks) {
        GRBLinExpr sum_of_machines = GRBLinExpr(0.0);
        GRBLinExpr sum_of_operators = GRBLinExpr(0.0);
        int n_poss_mach = inst.tasks[task_idx].machines.size();
        int n_poss_oper = inst.tasks[task_idx].operators.size();

        std::vector<double> coeffs_mach(n_poss_mach, 1.0);
        std::vector<double> coeffs_oper(n_poss_oper, 1.0);

        std::vector<GRBVar> mach_use_var_pointers;
        std::vector<GRBVar> oper_use_var_pointers;

        // Building the vectors of pointers to the assignment variables for each machine and each operator
        for (auto& [t_idx, assigned_mach_var] : assigned_machines_per_task[task_idx]) {
            mach_use_var_pointers.emplace_back(assigned_mach_var);
        }
        for (auto& [t_idx, assigned_oper_var] : assigned_operators_per_task[task_idx]) {
            oper_use_var_pointers.emplace_back(assigned_oper_var);
        }

        // Summing the assignment variables for each machine (resp. operator) for the task
        sum_of_machines.addTerms(coeffs_mach.data(), mach_use_var_pointers.data(), n_poss_mach);
        sum_of_operators.addTerms(coeffs_oper.data(), oper_use_var_pointers.data(), n_poss_oper);

        // Exactly one machine and one operator can be assigned to a task: the sum of the assignment variables is 1
        model.addConstr(sum_of_machines == 1, "one_machine_only_T" + std::to_string(task_idx + 1));
        model.addConstr(sum_of_operators == 1, "one_operator_only_T" + std::to_string(task_idx + 1));
        // The equality constraint forces to address each task assignment and leav no orphan task in the middle of a job sequence.
        // In the worst case, the task will be postponed by a significant delay and fall outside the window, in which case it will be reoptimized in the next lookahead window.
    }

    // Set the OP-MA assignments compatibility constraints
    log_stream << "Setting assignment OP-MA compatibility constraints..." << std::endl;

    for (int task_idx : processed_tasks) {
        
        std::vector<GRBVar> machine_terms;
        std::vector<GRBVar> operator_terms;

        for (int ma_idx : inst.tasks[task_idx].machines) {
            for (int op_idx : inst.tasks[task_idx].compatibility[ma_idx]) {
                operator_terms.emplace_back(
                    assigned_operators_per_task[task_idx][op_idx]
                );

                machine_terms.emplace_back(
                    assigned_machines_per_task[task_idx][ma_idx]
                );
            }
        }

        int nb_terms = operator_terms.size();
        std::vector<double> coeffs_terms(nb_terms, 1.0);

        GRBQuadExpr cross_compat = GRBQuadExpr();
        // This quadratic expression contains the normalized disjunctive form of the compatibility constraints for this task i.e. O1.M1 + O1.M2 + ... for all compatible pairs
        cross_compat.addTerms(
            coeffs_terms.data(),
            machine_terms.data(),
            operator_terms.data(),
            nb_terms
        );

        model.addQConstr(cross_compat <= 1, "compatibility_T" + std::to_string(task_idx + 1));
    }




    // TODO: freeze machines and operators that are currently used by pending tasks


    // Set the assignments time overlap constraints

    log_stream << "Setting assignments time overlap constraints..." << std::endl;

    std::map <std::pair<int, int>, GRBVar> tasks_overlapping_machines;
    std::map <std::pair<int, int>, GRBVar> tasks_overlapping_operators;

    // Iterate over all pairs of different jobs, unordered, once, to set the exclusion constraints
    for (auto j_idx1_ptr = processed_jobs.begin();
        j_idx1_ptr != processed_jobs.end();
        ++j_idx1_ptr) {

        auto j_idx2_ptr = j_idx1_ptr; ++j_idx2_ptr;

        for (; j_idx2_ptr != processed_jobs.end(); ++j_idx2_ptr) {

            log_stream << std::endl << std::endl << "*** ...between respective tasks of J" << *j_idx1_ptr + 1 << " and J" << *j_idx2_ptr + 1 << " ***" << std::endl;

            // Given a pair of jobs, iterate over all pairs of tasks (one from each job)
            // This means we consider each edge of the corresponding bipartite graph
            for (int t_idx1 : processed_tasks_of_jobs[*j_idx1_ptr]) {
                for (int t_idx2 : processed_tasks_of_jobs[*j_idx2_ptr]) {

                    log_stream << "T" << t_idx1 + 1 << "-T" << t_idx2 + 1 << "; ";
                    // Compute temporarily the intersection of possible machines for the two tasks
                    std::vector<int> intersection_operators;
                    std::set_intersection(
                        inst.tasks[t_idx1].operators.begin(), inst.tasks[t_idx1].operators.end(),
                        inst.tasks[t_idx2].operators.begin(), inst.tasks[t_idx2].operators.end(),
                        std::back_inserter(intersection_operators)
                    );
                    // Compute the intersection of possible operators for the two tasks
                    std::vector<int> intersection_machines;
                    std::set_intersection(
                        inst.tasks[t_idx1].machines.begin(), inst.tasks[t_idx1].machines.end(),
                        inst.tasks[t_idx2].machines.begin(), inst.tasks[t_idx2].machines.end(),
                        std::back_inserter(intersection_machines)
                    );

                    // Define the suffix indicator for the two tasks to set in constraint names
                    std::string task_pair_str = "T" + std::to_string(t_idx1 + 1) + "_T" + std::to_string(t_idx2 + 1);

                    // Retrieve the begin times and durations for the two tasks
                    GRBVar& bt1 = begin_times_tasks_per_job[*j_idx1_ptr][t_idx1];
                    GRBVar& bt2 = begin_times_tasks_per_job[*j_idx2_ptr][t_idx2];
                    int& pt1 = inst.tasks[t_idx1].processing_time;
                    int& pt2 = inst.tasks[t_idx2].processing_time;

                    // Create both time overlap indicators for the two tasks
                    GRBVar ind1 = model.addVar(
                        0.0, 1.0, 0.0, GRB_BINARY,
                        task_pair_str + "_overlap_ind1"
                    );
                    GRBVar ind2 = model.addVar(
                        0.0, 1.0, 0.0, GRB_BINARY,
                        task_pair_str + "_overlap_ind2"
                    );

                    // Two tasks overlap if and only if (e1 - b2 >= 1) AND (e2 - b1 >= 1)
                    model.addGenConstrIndicator(
                        ind1,
                        0,              // (end1 - begin2 <= 0) => first overlap trigger
                        bt1 + pt1 - bt2, GRB_GREATER_EQUAL, 1,
                        "bind_intersect_ind1_" + task_pair_str
                    );
                    model.addGenConstrIndicator(
                        ind2,
                        0,              // (end2 - begin1 <= 0) => second overlap trigger
                        bt2 + pt2 - bt1, GRB_GREATER_EQUAL, 1,
                        "bind_intersect_ind2_" + task_pair_str
                    );
                    // At this point, we have set the following implication constraint:
                    //      IF the two tasks overlap in time, THEN both indicators ind1 and ind2 are TRUE
                    // Meaning we modelled the contrapositive:
                    //      IF one of the two indicators is FALSE, THEN the two tasks do not overlap in time
                    // Consequently, the overlap in time is the logical AND of the two indicators, which we can use

                    // Prevent that the assigned operators overlap between the two tasks if they overlap in time
                    for (int op_idx : intersection_operators) {
                        GRBVar& t1_uses_op = assigned_operators_per_task[t_idx1][op_idx];
                        GRBVar& t2_uses_op = assigned_operators_per_task[t_idx2][op_idx];
                        model.addQConstr(
                            ind1 * ind2 + t1_uses_op * t2_uses_op <= 1,
                            "bind_overlap_O" + std::to_string(op_idx + 1) + "_" + task_pair_str);
                    }

                    // Prevent that the assigned machines overlap between the two tasks if they overlap in time
                    for (int ma_idx : intersection_machines) {
                        GRBVar& t1_uses_ma = assigned_machines_per_task[t_idx1][ma_idx];
                        GRBVar& t2_uses_ma = assigned_machines_per_task[t_idx2][ma_idx];
                        model.addQConstr(
                            ind1 * ind2 + t1_uses_ma * t2_uses_ma <= 1,
                            "bind_overlap_M" + std::to_string(ma_idx + 1) + "_" + task_pair_str);
                    }
                }
            }

        }
    }


    // Set and declare the objective function
    log_stream << "Setting objective function." << std::endl;

    GRBLinExpr objective = 0;
    log_stream << "Setting the objective function..." << std::endl;
    for (int job_idx : processed_jobs) {
        // Set interim costs
        objective += inst.jobs[job_idx].weight * new_completion_dates_jobs[job_idx];
        // Set tardiness costs
        objective += inst.tardiness * inst.jobs[job_idx].weight * tardiness_post_slacks[job_idx];
        // Set unit penalty costs
        objective += inst.unit_penalty * inst.jobs[job_idx].weight * unit_penalties[job_idx];
    }


    model.setObjective(objective, GRB_MINIMIZE);
    log_stream << "Tuning model parameters..." << std::endl;
    // model.tune();



    log_stream << "Writing model to file..." << std::endl;
    model.write("model.mps");
    model.write("model.lp");
    model.optimize();



    // TODO: Delay all upcoming tasks coming after the horizon by the job's delay that was just resolved
    // TODO: Update the choices made by the optimization
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <instance_filename> <lookahead_duration>" << std::endl;
        return 1;
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


    resolve_lookahead(inst, sol, job_stacks, pending_tasks_per_job, time_cursor, lookahead_duration, log_file);

    log_file.close();
    return 0;
}


