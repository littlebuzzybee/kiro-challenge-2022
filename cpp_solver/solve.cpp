#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <set>
#include <map>
#include <deque>
#include <set>
#include <queue>


#include "utils.h"
#include "breakdown.h"
#include "gurobi_c++.h"



/* ========== MAIN FUNCTION TO ITERATE AND SOLVE ========== */



void resolve_lookahead(
    Instance& inst,
    Solution& sol,
    std::map<int, std::deque<int>>& job_stacks,
    std::unordered_map<int, int>& pending_task_per_job,
    int lookahead,
    double time_limit,
    int max_threads,
    bool write_problem_file,
    bool report_all_decisions,
    std::ostream& log_stream,
    std::ostream& inform_stream
) {
    // Begin by identifying the tasks that are relevant in the lookahead window
    std::map<int, std::vector<int>> processed_tasks_of_jobs; // ordered set of tasks for each job
    std::map<int, int> cumulative_remaining_time_per_job;
    std::vector<int> processed_jobs;
    std::vector<int> processed_tasks;


    int time_horizon{ 0 };
    int time_cursor{ 0 };
    int round = 1;
    inform_stream << "Beginning solving procedure with solver-augmented iterative heuristic..." << std::endl;
    inform_stream << "Lookahead duration set to " << lookahead << " time units." << std::endl;
    inform_stream << "Solve time limit set to " << time_limit << " seconds." << std::endl;
    inform_stream << "Gurobi threads set to " << max_threads << std::endl;

    log_stream << "Initializing Gurobi environment..." << std::endl;
    GRBEnv gurobi_env = GRBEnv(true);
    gurobi_env.set("LogFile", "gurobi.log");
    gurobi_env.start();

    while (!all_stacks_are_empty(job_stacks)) {
        // While there are tasks to process in the lookahead window
        // We process the tasks of the jobs
        time_horizon = time_cursor + lookahead;
        log_stream << std::endl << "**********************************************************" << std::endl;
        inform_stream << std::endl << "**********************************************************" << std::endl;
        log_stream << "Resolving lookahead on time window [" << time_cursor << ", " << time_horizon << "]..." << std::endl;
        inform_stream << "Resolving lookahead on time window [" << time_cursor << ", " << time_horizon << "]..." << std::endl;



        // Update the map of pending tasks
        for (auto it = pending_task_per_job.begin(); it != pending_task_per_job.end();) {
            auto& [parent_job, task_idx] = *it;
            if (sol.begin_time_tasks[task_idx] + inst.tasks[task_idx].processing_time <= time_cursor) {
                // The pending task has been processed in the lookahead window
                it = pending_task_per_job.erase(it); // Erase and get the next iterator
                log_stream << "Removed T" << task_idx + 1 << " (J" << parent_job + 1 << ") from pending." << std::endl;
            }
            else {
                ++it; // Advance the iterator if no erasure
            }
        }

        get_relevant_tasks(
            inst,
            sol,
            time_cursor,
            time_horizon,
            job_stacks,
            processed_tasks,
            processed_jobs,
            processed_tasks_of_jobs,
            cumulative_remaining_time_per_job,
            pending_task_per_job
        );

        processed_tasks.shrink_to_fit();
        processed_jobs.shrink_to_fit();
        std::sort(processed_tasks.begin(), processed_tasks.end());
        std::sort(processed_jobs.begin(), processed_jobs.end());


        if (processed_tasks.empty()) {
            log_stream << "No tasks to process in the lookahead window. Moving to the next window." << std::endl;
            time_cursor += lookahead;
            continue;
        }



        // Compute the maximum duration of the processed tasks
        int max_duration_tasks = *std::max_element(processed_tasks.begin(), processed_tasks.end(),
            [&inst](int t1, int t2) {
                return inst.tasks[t1].processing_time < inst.tasks[t2].processing_time;
            });

        // Print the list of processed tasks and jobs in the log stream
        display_lookahead_program(
            max_duration_tasks,
            processed_tasks,
            processed_jobs,
            processed_tasks_of_jobs,
            job_stacks,
            pending_task_per_job,
            log_stream
        );


        std::map<int, std::deque<int>> j_s{};
        for (int j_idx : processed_jobs) {
            for (int t_idx : processed_tasks_of_jobs[j_idx]) {
                j_s[j_idx].push_back(t_idx);
            }
        }

        // Warm up the solution with the greedy solution
        log_stream << "Computing a greedy partial solution to warm up the model..." << std::endl;
        greedy_partial_solve_lookahead(
            inst,
            sol,
            time_cursor,
            time_horizon,
            j_s,
            pending_task_per_job,
            log_stream
        );

        log_stream << std::endl << "Improving the greedy pre-solution with Gurobi..." << std::endl;
        // Initialize Gurobi environment and model
        // inform_stream << "Creating model..." << std::endl;
        GRBModel model = GRBModel(gurobi_env);
        model.set(GRB_IntParam_LogToConsole, 1); // Do not log to console
        model.set(GRB_StringAttr_ModelName, "Time_Scheduling_Round_" + std::to_string(1));
        model.set(GRB_IntParam_OutputFlag, 1);
        model.set(GRB_IntParam_Threads, max_threads);
        model.set(GRB_DoubleParam_TimeLimit, time_limit);
        model.set(GRB_IntParam_MIPFocus, 3); // Focus on finding new feasible solutions
        model.set(GRB_IntParam_Presolve, 2); // Aggressive presolve
        model.set(GRB_IntParam_PrePasses, 1); // One presolve pass only to limit the time spent on presolve
        model.set(GRB_DoubleParam_Heuristics, 0.7); // Increase the proportion of time spent on heuristics from 5% (default) to 50%
        model.set(GRB_DoubleParam_MIPGap, 0.01); // 1% optimality gap
        model.set(GRB_IntParam_Method, -1); // Automatic method selection




        // Declare the begin times of each task and set the ordering constraints
        std::map<int, std::map<int, GRBVar>> begin_times_tasks_per_job;
        log_stream << "Declaring scheduling variables and constraints for processed job..." << std::endl;

        set_begin_variables_and_ordering_constraints(
            inst,
            sol,
            model,
            begin_times_tasks_per_job,
            processed_tasks_of_jobs,
            processed_jobs,
            pending_task_per_job,
            time_cursor
        );

        // Declare slacks and unit penalties variables for each job
        std::map<int, GRBVar> tardiness_slacks;
        std::map<int, GRBVar> unit_penalties;

        log_stream << "Adding slack and penalty variables for all jobs..." << std::endl;
        set_slack_and_penalty_variables(
            model,
            tardiness_slacks,
            unit_penalties,
            processed_jobs
        );

        // Declare the assigned machines and operators variables for every task
        log_stream << "Declaring assignment variables..." << std::endl;
        std::map<int, std::map<int, GRBVar>> assigned_operators_per_task;
        std::map<int, std::map<int, GRBVar>> assigned_machines_per_task;
        // first index is the task index, second index is the operator/machine index
        // resulting function is T_idx --> {operator_idx/machine_idx} --> GRBVar

        set_assignment_variables(
            inst,
            model,
            assigned_operators_per_task,
            assigned_machines_per_task,
            processed_tasks
        );


        // Warm up the solution with the greedy search's results
        warmup_solution(
            inst,
            sol,
            processed_tasks,
            assigned_operators_per_task,
            assigned_machines_per_task,
            begin_times_tasks_per_job
        );


        // Set the assignments physical overlap constraints
        log_stream << "Setting workers physical overlap constraints..." << std::endl;
        set_workers_uniqueness_constraints(
            inst,
            model,
            assigned_operators_per_task,
            assigned_machines_per_task,
            processed_tasks
        );

        // Set the OP-MA assignments compatibility constraints
        log_stream << "Setting workers OP-MA compatibility constraints..." << std::endl;
        // These constraints are redundant but might help with constraint propagation, I guess
        set_workers_ubiquity_constraints(
            inst,
            model,
            assigned_operators_per_task,
            assigned_machines_per_task,
            processed_tasks
        );

        // Set the assignments time overlap constraints
        log_stream << "Setting assignments time overlap constraints..." << std::endl;
        set_workers_time_exclusion_constraints(
            inst,
            sol,
            model,
            begin_times_tasks_per_job,
            assigned_operators_per_task,
            assigned_machines_per_task,
            processed_jobs,
            processed_tasks_of_jobs,
            pending_task_per_job,
            log_stream
        );

        // Define the resulting completion dates variable of each processed job
        // as the sum of its current completion date and the postponement due to task delays
        std::map<int, GRBLinExpr> new_completion_dates_jobs;
        log_stream << "Adding completion date variables for all jobs..." << std::endl;
        set_completion_time_and_penalty_constraints(
            inst,
            model,
            tardiness_slacks,
            unit_penalties,
            new_completion_dates_jobs,
            begin_times_tasks_per_job,
            processed_jobs,
            processed_tasks_of_jobs,
            cumulative_remaining_time_per_job
        );






        // Set and declare the objective function
        log_stream << "Setting objective function..." << std::endl;
        set_objective_function(
            inst,
            model,
            processed_jobs,
            new_completion_dates_jobs,
            tardiness_slacks,
            unit_penalties
        );
        log_stream << std::endl;


        if (write_problem_file) {
            log_stream << "Writing model to file..." << std::endl;
            model.write("lp_problems/model" + std::to_string(round) + ".mps");
            model.write("lp_problems/model" + std::to_string(round) + ".lp");
        }

        model.update();
        int num_Vars = model.get(GRB_IntAttr_NumVars);
        int num_Constrs = model.get(GRB_IntAttr_NumConstrs);
        int num_SOS = model.get(GRB_IntAttr_NumSOS);
        int num_QConstrs = model.get(GRB_IntAttr_NumQConstrs);
        int num_GenConstrs = model.get(GRB_IntAttr_NumGenConstrs);
        int num_IntVars = model.get(GRB_IntAttr_NumIntVars);
        int num_BinVars = model.get(GRB_IntAttr_NumBinVars);
        log_stream << "Model has " << num_Vars << " variables" << std::endl;
        log_stream << "Model has " << num_IntVars << " integer variables" << std::endl;
        log_stream << "Model has " << num_BinVars << " binary variables" << std::endl;
        log_stream << "Model has " << num_Constrs << " linear constraints" << std::endl;
        log_stream << "Model has " << num_SOS << " SOS constraints" << std::endl;
        log_stream << "Model has " << num_QConstrs << " quadratic constraints" << std::endl;
        log_stream << "Model has " << num_GenConstrs << " general constraints" << std::endl;
        int is_MIP = model.get(GRB_IntAttr_IsMIP);
        int is_QP = model.get(GRB_IntAttr_IsQP);
        int is_QCP = model.get(GRB_IntAttr_IsQCP);

        log_stream << "Model is a MIP: " << is_MIP << std::endl;
        log_stream << "Model is a QP: " << is_QP << std::endl;
        log_stream << "Model is a QCP: " << is_QCP << std::endl << std::endl;


        // log_stream << "Tuning model parameters..." << std::endl;
        // model.tune();
        model.optimize();
        assert(model.get(GRB_IntAttr_SolCount) > 0);
        int status_code = model.get(GRB_IntAttr_Status);
        auto obj = model.get(GRB_DoubleAttr_ObjVal);
        log_stream << "Model status code: " << status_code << std::endl;
        log_stream << "Partial objective value on window: " << obj << std::endl;


        if (report_all_decisions) {
            std::cout << "Complete set of decision variables:" << std::endl;
            GRBVar* vars = NULL;
            double* values = NULL;
            std::string* names = NULL;

            int numVars = model.get(GRB_IntAttr_NumVars);

            vars = model.getVars();
            values = model.get(GRB_DoubleAttr_X, vars, numVars);
            names = model.get(GRB_StringAttr_VarName, vars, numVars);
            // Print the values of all variables

            for (int i = 0; i < numVars; i++) {
                std::cout << names[i] << " = " << values[i] << std::endl;
            }
        }



        // empty the map of pending tasks
        // pending_task_per_job.clear();

        // Display the decisions and update pending tasks
        log_stream << "~~~ Decisions made by the solver on this window: ~~~" << std::endl;
        for (int i = processed_tasks.size() - 1; i >= 0; --i) {
            // Reverse order to catch the tasks that are postponed and push them back in the same order in the job stacks
            int task_idx = processed_tasks[i];
            // Get the begin time of task according to the solver
            int parent_job = inst.tasks[task_idx].job_parent;
            int begin_time = begin_times_tasks_per_job[parent_job][task_idx].get(GRB_DoubleAttr_X);

            // Find the machine and operator assigned to the task
            int operator_choice = -1;
            int machine_choice = -1;

            for (auto& [op, var] : assigned_operators_per_task[task_idx]) {
                if (var.get(GRB_DoubleAttr_X) > 0.5) {
                    operator_choice = op;
                    break;
                }
            }
            for (auto& [ma, var] : assigned_machines_per_task[task_idx]) {
                if (var.get(GRB_DoubleAttr_X) > 0.5) {
                    machine_choice = ma;
                    break;
                }
            }

            if (begin_time >= time_horizon) {
                // Task begin time falls back after the time horizon, we do not consider it scheduled and it will be reoptimized in the next lookahead window
                job_stacks[parent_job].emplace_front(task_idx);
                // We traversed the tasks in reverse order to ensure that the tasks are pushed back here in the same order as they were popped off
                // Because of the precedence constraints, we know that all tasks that are pushed back here are adjacent and follow each other in the job sequence
                log_stream << "* T" << task_idx + 1 << " postponed" << std::endl;
            }
            else {
                // Task begin time falls within the window and ends within or after the time horizon, we schedule it and fix it
                // Update and fix the decision variables
                sol.begin_time_tasks[task_idx] = begin_time;
                sol.machine_choice_tasks[task_idx] = machine_choice;
                sol.operator_choice_tasks[task_idx] = operator_choice;
                log_stream << "* T" << task_idx + 1 << " scheduled at t=" << begin_time << " with M" << machine_choice + 1 << " & O" << operator_choice + 1;

                if (begin_time + inst.tasks[task_idx].processing_time > time_horizon && begin_time < time_horizon) {
                    // Task begin time falls within the window but ends after the window span, we additionally mark it as pending for the next window to optimize given the added constraint for concomitant jobs' tasks
                    pending_task_per_job[parent_job] = task_idx; // there can be only one
                    log_stream << " (pending in subsequent window)";
                }
                log_stream << std::endl;
            }
        }


        for (int job_idx : processed_jobs) {
            int tardiness = tardiness_slacks[job_idx].get(GRB_DoubleAttr_X);
            int unit_penalty = unit_penalties[job_idx].get(GRB_DoubleAttr_X);
            log_stream << "Job J" << job_idx + 1 << " tardiness: " << tardiness << " / penalty: " << unit_penalty << std::endl;
        }
        log_stream << std::endl;

        processed_jobs.clear();
        processed_tasks.clear();
        for (auto& [j_idx, task_list] : processed_tasks_of_jobs) {
            task_list.clear();
        }
        time_cursor += lookahead;
        round++;
    }
}




void resolve_simple(
    Instance& inst,
    Solution& sol,
    std::map<int, std::deque<int>>& job_stacks,
    std::ostream& log_stream
) {

    log_stream << "Beginning solving procedure with an elementary iterative heuristic..." << std::endl;

    int time_cursor{ 0 };
    // Report the contents of the stacks
    log_stream << "Job stacks at time " << time_cursor << ":" << std::endl;
    for (auto& [j_idx, task_stack] : job_stacks) {
        log_stream << "J" << j_idx + 1 << ": ";
        for (int t_idx : task_stack) {
            log_stream << "T" << t_idx + 1 << " ";
        }
        log_stream << std::endl;
    }

    std::set<int> available_machines;
    std::set<int> available_operators;

    // Create a pool of resources
    for (int m_idx = 0; m_idx < inst.nb_machines; ++m_idx) {
        available_machines.insert(m_idx);
    }
    for (int o_idx = 0; o_idx < inst.nb_operators; ++o_idx) {
        available_operators.insert(o_idx);
    }

    // Create data structures for the release of resources by key time position
    std::map<int, std::set<int>> release_calendar_machines;
    std::map<int, std::set<int>> release_calendar_operators;

    // Create a data structure to enforce the precedence
    std::map<int, int> next_time_persue_job;
    for (int j_idx = 0; j_idx < inst.nb_jobs; ++j_idx) {
        next_time_persue_job[j_idx] = inst.jobs[j_idx].release_date;
    }

    std::map<int, int> cumulative_remaining_time_per_job;
    for (auto& [j_idx, _] : job_stacks) {
        std::deque<int>& remaining_tasks_sequence = job_stacks[j_idx];
        int total_processing_time = std::reduce(
            remaining_tasks_sequence.begin(),
            remaining_tasks_sequence.end(),
            0, // Initial value of the sum
            [&inst](int total_sum, int t_idx) {
                return total_sum + inst.tasks[t_idx].processing_time;
            }
        );
        cumulative_remaining_time_per_job[j_idx] = total_processing_time;
    }

    int time_pos{ time_cursor };


    while (!all_stacks_are_empty(job_stacks)) { // && time_pos < time_horizon
        log_stream << std::endl << "*** Time " << time_pos << " ***" << std::endl;


        // First release the resources that are no longer used
        log_stream << "  removed: ";
        for (int m_idx : release_calendar_machines[time_pos]) {
            available_machines.insert(m_idx);
            log_stream << "M" << m_idx + 1 << " ";
        }
        for (int o_idx : release_calendar_operators[time_pos]) {
            available_operators.insert(o_idx);
            log_stream << "O" << o_idx + 1 << " ";
        }


        std::priority_queue<std::tuple<int, int>> task_queue{};
        // first integer is the score, second integer is the job index
        // score is a function of the tardiness of the job and its weight
        // to act as a proxy of the to-be objective terms

        // First, we insert the tasks that are ready to be processed in a sorted priority queue to be ranked and compared
        // there is at most only one task per job in the stack and we decide which to adress in order of priority

        for (auto& [j_idx, task_stack] : job_stacks) {
            // Insert the first task in line for the job if it exists and if the processing time of its predecessor is over
            if (task_stack.empty() || time_pos < next_time_persue_job[j_idx]) {
                continue;
            }
            int t_idx = task_stack.front();
            int tardiness = std::max(0, time_pos + cumulative_remaining_time_per_job[j_idx] - inst.jobs[j_idx].due_date);
            int score = inst.jobs[j_idx].weight * tardiness;
            std::tuple<int, int> task_entry = std::make_tuple(score, t_idx);
            task_queue.emplace(task_entry);
        }




        while (!task_queue.empty() && !available_machines.empty() && !available_operators.empty()) {
            // Heuristic: iterate through tasks in order of priority at that time step and
            // assign tasks and resources one at a time

            auto [tardiness, t_idx] = task_queue.top();
            task_queue.pop();
            int j_idx = inst.tasks[t_idx].job_parent;

            // Compute intersection of available machines and authorized machines fot that task
            // POSSIBLE = AVAILABLE INTERSECT AUTHORIZED
            std::vector<int> possible_machines;
            std::set<int>& authorized_machines = inst.tasks[t_idx].machines;
            std::set_intersection(
                available_machines.begin(), available_machines.end(),
                authorized_machines.begin(), authorized_machines.end(),
                std::back_inserter(possible_machines)
            );

            // If no machine is available, we skip the task
            if (possible_machines.empty()) {
                continue;
            }

            // Otherwise, we look for a qualified operator on one of the available machines for that task
            int chosen_machine{ -1 };
            int chosen_operator{ -1 };
            std::vector<int> possible_operators;
            std::set<int>& authorized_operators = inst.tasks[t_idx].operators;

            for (int m_idx : possible_machines) {
                possible_operators.clear();
                std::set_intersection(
                    available_operators.begin(), available_operators.end(),
                    authorized_operators.begin(), authorized_operators.end(),
                    std::back_inserter(possible_operators)
                );

                if (!possible_operators.empty()) {
                    // Greedily assign the first compatible pair to the task
                    chosen_operator = possible_operators.front();
                    chosen_machine = m_idx;
                    break;
                }
                else {
                    continue;
                }
            }

            // If no machine or operator is available, we skip the task
            if (chosen_machine == -1 || chosen_operator == -1) {
                continue;
            }


            // Assign the workers to the task
            sol.machine_choice_tasks[t_idx] = chosen_machine;
            sol.operator_choice_tasks[t_idx] = chosen_operator;

            // Schedule the task
            sol.begin_time_tasks[t_idx] = time_pos;

            // Prevent the subsequent assignment of any other task of the same job before the end of the current task
            next_time_persue_job[j_idx] = time_pos + inst.tasks[t_idx].processing_time;
            available_machines.erase(chosen_machine);
            available_operators.erase(chosen_operator);

            // Update the release calendar for the chosen machine and operator
            release_calendar_machines[time_pos + inst.tasks[t_idx].processing_time].insert(chosen_machine);
            release_calendar_operators[time_pos + inst.tasks[t_idx].processing_time].insert(chosen_operator);
            cumulative_remaining_time_per_job[j_idx] -= inst.tasks[t_idx].processing_time;

            // Remove the task from the stack
            job_stacks[j_idx].pop_front();
            log_stream << "Task T" << t_idx + 1 << " (J" << j_idx + 1 << ") assigned to M" << chosen_machine + 1 << " & O" << chosen_operator + 1 << " at time " << time_pos << std::endl;
        }
        time_pos++;
    }
    log_stream << std::endl << "End of solving procedure." << std::endl << std::endl;
}
