#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <set>
#include <map>
#include <deque>


#include "gurobi_c++.h"
#include "utils.h"
#include "nlohmann/json.hpp"



void partial_initialize_time_scheduling_greedy(Instance& inst, Solution& sol, std::ostream& log_stream = std::cout) {
    log_stream << "Stacking greedily tasks in time..." << std::endl;
    for (int j_idx = 0; j_idx < inst.nb_jobs; j_idx++) {
        int total_task_offset = 0; // cumulative sum of all task processing times for the current job
        log_stream << "=== Processing job " << j_idx << " ===" << std::endl;
        for (int t_idx = 0; t_idx < (int)inst.jobs[j_idx].sequence.size(); t_idx++) {
            int processed_task = inst.jobs[j_idx].sequence[t_idx];
            log_stream << "* Processing task " << processed_task << std::endl;

            sol.begin_time_tasks[processed_task] = inst.jobs[j_idx].release_date
                + total_task_offset;

            sol.end_time_tasks[processed_task] = inst.jobs[j_idx].release_date
                + total_task_offset
                + inst.tasks[processed_task].processing_time;

            log_stream << "Set task " << processed_task << " begin time: " << sol.begin_time_tasks[processed_task] << std::endl;
            log_stream << "Set task " << processed_task << " end time: " << sol.end_time_tasks[processed_task] << std::endl;
            // update time offset with current task
            total_task_offset += inst.tasks[processed_task].processing_time;
        }
        sol.completion_date_jobs[j_idx] = inst.jobs[j_idx].release_date + total_task_offset;
        log_stream << "Job " << j_idx << " ** completion date: " << sol.completion_date_jobs[j_idx] << std::endl << std::endl;
    }
}



void resolve_lookahead(Instance& inst, Solution& sol, std::map<int, std::deque<int>> job_stacks, std::unordered_map<int, int> pending_tasks_per_job, int time_cursor, int lookahead_duration, std::ostream& log_stream = std::cout) {
    const int time_horizon = time_cursor + lookahead_duration;
    log_stream << "Resolving lookahead on time window [" << time_cursor << ", " << time_horizon << "]..." << std::endl;


    std::map<int, std::vector<int>> processed_tasks_of_jobs; // ordered set of tasks for each job
    std::set<int> processed_jobs;
    // std::set<int> processed_tasks; // ordered redundancy of tasks used for assignment consraints between tasks later on
    std::vector<int> processed_tasks;



    // Begin by identifying the tasks that are concerned by the lookahead
    for (int job_idx = 0; job_idx < inst.nb_jobs; ++job_idx) {
        if (job_stacks[job_idx].empty()) {
            continue;
        }
        else {
            // Only the tasks that are fully comprised in the time window are considered.
            // Those which started before the time window but end in the time window are considered processed, fixed and pending and would not benefit the problem
            // by being postponed again. Theirs implications are taken into account by the pending_tasks set and not recomputed here.
            processed_jobs.insert(job_idx); // inserts an index, not an id
            while (!job_stacks[job_idx].empty()
                && time_cursor <= sol.begin_time_tasks[job_stacks[job_idx].front()]
                && sol.begin_time_tasks[job_stacks[job_idx].front()] <= time_horizon
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




    log_stream << "There are " << nb_processed_jobs << " jobs processed." << std::endl;

    // print ordered list of concerned tasks
    log_stream << "They comprise the following processed tasks in the lookahead: ";
    for (int task_idx : processed_tasks) {
        log_stream << "|" << task_idx;
    }
    log_stream << "| distributed as follows: " << std::endl;


    for (int job_idx : processed_jobs) {
        log_stream << "Job " << job_idx << " entails " << processed_tasks_of_jobs[job_idx].size() << " tasks ordered as:   ";
        for (int task_idx : processed_tasks_of_jobs[job_idx]) {
            log_stream << "|" << task_idx;
        }
        log_stream << "|" << std::endl;
    }
    log_stream << "There are " << nb_pending_tasks << " pending tasks in total." << std::endl;




    // Initialize Gurobi environment and model
    log_stream << "Initializing Gurobi environment and model..." << std::endl;
    GRBEnv env = GRBEnv(true);
    env.set("LogFile", "gurobi.log");
    env.start();
    GRBModel model = GRBModel(env);
    model.set(GRB_StringAttr_ModelName, "time_scheduling_round_" + std::to_string(1));


    // Declare the begin times of each task and set the ordering constraints
    std::map<int, std::map<int, GRBVar>> begin_times_tasks_per_job;

    log_stream << "Declaring scheduling variables and constraints for processed job..." << std::endl;
    for (int job_idx : processed_jobs) {
        // First, Declare the begin times variables of each task

        for (int task_idx : processed_tasks_of_jobs[job_idx]) {

            begin_times_tasks_per_job[job_idx].emplace(
                task_idx,
                model.addVar(time_cursor, GRB_INFINITY, 0.0, GRB_INTEGER,
                    "begin_time_task_" + std::to_string(task_idx))
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
                    model.addConstr(GRBLinExpr(sol.end_time_tasks[previous_fixed_task]),
                        GRB_LESS_EQUAL,
                        begin_times_tasks_per_job[job_idx][task_idx],
                        "thresh_begin_time_task_" + std::to_string(task_idx));
                }
                else {
                    // If there is no pending task for this job overlapping the window's beginning,
                    // then we set the lower bound of the beginning of the task at the time cursor posution (beginning of the window)
                    model.addConstr(GRBLinExpr(time_cursor),
                        GRB_LESS_EQUAL,
                        begin_times_tasks_per_job[job_idx][task_idx],
                        "rel_ordering_task_" + std::to_string(task_idx));
                }
            }
            else if ((task_idx != processed_tasks_of_jobs[job_idx].front()) && (task_idx != processed_tasks_of_jobs[job_idx].back())) {
                // If this is not the first nor the last task of the job being optimized in the window,
                // we prevent the task from starting before the end of the previous task (no overlapping)
                int next_task_idx = processed_tasks_of_jobs[job_idx][i + 1];


                model.addConstr(GRBLinExpr(begin_times_tasks_per_job[job_idx][task_idx] + inst.tasks[task_idx].processing_time),
                    GRB_LESS_EQUAL,
                    begin_times_tasks_per_job[job_idx][next_task_idx],
                    "rel_ordering_task_" + std::to_string(task_idx));
            }
        }
    }


    // Declare slacks and unit penalties variables for each job
    std::map<int, GRBVar> tardiness_post_slacks;
    std::map<int, GRBVar> unit_penalties;

    log_stream << "Adding slack variable for all jobs. " << std::endl;
    for (int job_idx : processed_jobs) {
        tardiness_post_slacks[job_idx] = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER,
            "slack_" + std::to_string(job_idx));
        unit_penalties[job_idx] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
            "unit_penalty_" + std::to_string(job_idx));
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
            new_completion_dates_jobs[job_idx] <= inst.jobs[job_idx].due_date + tardiness_post_slacks[job_idx]
        );
    }


    // Set unit penalty variables
    log_stream << "Adding unit penalty constraint all jobs..." << std::endl;
    for (int job_idx : processed_jobs) {
        // unit_penalty = 1 if completion_time > due_date
        model.addGenConstrIndicator(
            unit_penalties[job_idx],
            0, // false
            new_completion_dates_jobs[job_idx], GRB_LESS_EQUAL, inst.jobs[job_idx].due_date
        );
    }



    // Declare the assigned machines and operators variables for every task
    std::map<int, std::map<int, GRBVar>> assigned_tasks_operators_per_job;
    std::map<int, std::map<int, GRBVar>> assigned_tasks_machines_per_job;

    log_stream << "Declaring assignment variables..." << std::endl;
    for (int job_idx : processed_jobs) {
        for (int task_idx : processed_tasks_of_jobs[job_idx]) {

            assigned_tasks_operators_per_job[job_idx].emplace(
                task_idx,
                model.addVar(0, inst.nb_operators, 0.0, GRB_INTEGER,
                    "operator_task" + std::to_string(task_idx))
            );

            assigned_tasks_machines_per_job[job_idx].emplace(
                task_idx,
                model.addVar(0, inst.nb_operators, 0.0, GRB_INTEGER,
                    "machine_task" + std::to_string(task_idx))
            );
        }
    }

    // Set the assignment exclusion constraints
    log_stream << "Setting assignment exclusion constraints..." << std::endl;

    std::map <std::pair<int, int>, GRBVar> tasks_overlapping_machines;
    std::map <std::pair<int, int>, GRBVar> tasks_overlapping_operators;
    log_stream << "Setting exclusion constraints for overlapping machines and operators..." << std::endl;

    for (int t_idx_id1 = 0; t_idx_id1 < nb_processed_tasks; t_idx_id1++) {
        for (int t_idx_id2 = t_idx_id1 + 1; t_idx_id2 < nb_processed_tasks; t_idx_id2++) {


            int task_idx1 = processed_tasks[t_idx_id1];
            int task_idx2 = processed_tasks[t_idx_id2];

            int parent_job1 = inst.tasks[task_idx1].job_parent;
            int parent_job2 = inst.tasks[task_idx2].job_parent;

            if (parent_job1 != parent_job2) {
                // If the tasks belong to the same job, no need to set exclusion constraints since time scheduling constraints already prevent overlapping
                // If not, we set the exclusion constraints for the tasks' respective operators and machines


                //  ******************************************************************************************************
                // ************ Declare common variables for both machine and operator overlapping exclusions ************
                //  ******************************************************************************************************
                GRBVar& begin_time_task1 = begin_times_tasks_per_job[parent_job1][task_idx1];
                GRBVar& begin_time_task2 = begin_times_tasks_per_job[parent_job2][task_idx2];


                GRBVar diff_ti_aft = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
                GRBVar diff_ti_bef = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);

                model.addGenConstrIndicator(diff_ti_aft, 1, begin_time_task1 + inst.tasks[task_idx1].processing_time <= begin_time_task2);
                model.addGenConstrIndicator(diff_ti_bef, 1, begin_time_task2 + inst.tasks[task_idx2].processing_time <= begin_time_task1);
                //  diff_ti_aft activates => task2 starts after task1 ends
                //  diff_ti_bef activates => task1 starts after task2 ends

                //  ******************************************************
                //  ************ Prevent overlapping operators ***********
                //  ******************************************************
                GRBVar& operator_task1 = assigned_tasks_operators_per_job[parent_job1][task_idx1];
                GRBVar& operator_task2 = assigned_tasks_operators_per_job[parent_job2][task_idx2];

                GRBVar diff_op_lte = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
                GRBVar diff_op_gte = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);


                model.addGenConstrIndicator(diff_op_lte, 1, operator_task1 <= operator_task2 - 1);
                model.addGenConstrIndicator(diff_op_gte, 1, operator_task1 >= operator_task2 + 1);
                // diff_op_lte activates => operator_task1 < operator_task2
                // diff_op_gte activates => operator_task1 > operator_task2


                // The true disjunction of the below 4 literals is equivalent to
                // operators of the two tasks do not overlap
                const GRBVar exclusion_same_operator[4] = {
                    diff_op_lte,
                    diff_op_gte,
                    diff_ti_aft,
                    diff_ti_bef
                };

                // Create the overlapping variable for operators
                tasks_overlapping_operators[{task_idx1, task_idx2}] = model.addVar(
                    0.0,
                    1.0,
                    0.0,
                    GRB_BINARY,
                    "operators_overlap_for_tasks_" + std::to_string(task_idx1) + "_" + std::to_string(task_idx2)
                );


                // Force the overlapping variable to be equal to the disjunction of the above literals and store it in the map
                model.addGenConstrOr(
                    tasks_overlapping_operators[{task_idx1, task_idx2}],
                    exclusion_same_operator,
                    4, // number of literals in the above array of boolean variables
                    "activates_if_same_operator_" + std::to_string(task_idx1) + "_" + std::to_string(task_idx2)
                );

                // Finally constrain it to be equal to 0 (false) in the model
                model.addConstr(
                    tasks_overlapping_operators[{task_idx1, task_idx2}] == 0,
                    "exclusion_same_operator_" + std::to_string(task_idx1) + "_" + std::to_string(task_idx2)
                );


                //  ******************************************************
                //  ************ Prevent overlapping machines ************
                //  ******************************************************
                GRBVar& machine_task1 = assigned_tasks_machines_per_job[parent_job1][task_idx1];
                GRBVar& machine_task2 = assigned_tasks_machines_per_job[parent_job2][task_idx2];

                GRBVar diff_ma_lte = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
                GRBVar diff_ma_gte = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);

                model.addGenConstrIndicator(diff_ma_lte, 1, machine_task1 <= machine_task2 - 1);
                model.addGenConstrIndicator(diff_ma_gte, 1, machine_task1 >= machine_task2 + 1);
                // diff_ma_lte activates => machine_task1 < machine_task2
                // diff_ma_gte activates => machine_task1 > machine_task2

                // The true disjunction of the below 4 literals is equivalent to
                // machines of the two tasks do not overlap
                const GRBVar exclusion_same_machine[4] = {
                    diff_ma_lte,
                    diff_ma_gte,
                    diff_ti_aft,
                    diff_ti_bef
                };

                // Create the overlapping variable for machines
                tasks_overlapping_machines[{task_idx1, task_idx2}] = model.addVar(
                    0.0,
                    1.0,
                    0.0,
                    GRB_BINARY,
                    "machines_overlap_for_tasks_" + std::to_string(task_idx1) + "_" + std::to_string(task_idx2)
                );


                // Force the overlapping variable to be equal to the disjunction of the above literals and store it in the map
                model.addGenConstrOr(
                    tasks_overlapping_machines[{task_idx1, task_idx2}],
                    exclusion_same_machine,
                    4, // number of literals in the above array of boolean variables
                    "activates_if_same_operator_" + std::to_string(task_idx1) + "_" + std::to_string(task_idx2)
                );

                // Finally constrain it to be equal to 0 (false) in the model
                model.addConstr(
                    tasks_overlapping_machines[{task_idx1, task_idx2}] == 0,
                    "exclusion_same_operator_" + std::to_string(task_idx1) + "_" + std::to_string(task_idx2)
                );


            }
        }
    }



    // Set objective function
        // Declare objective function
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

    log_stream << "Setting objective function." << std::endl;
    model.setObjective(objective, GRB_MINIMIZE);



    // Delay all upcoming tasks coming after the horizon by the job's delay that was just resolved
    // Update the choices made by the optimization
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <instance_filename>" << std::endl;
        return 1;
    }

    // Import instance
    std::string instance_filename = argv[1];
    std::ofstream log_file("solve.log");
    Instance inst = import_instance(instance_filename, log_file);


    // Begin solving procedure
    Solution sol;
    // Initialize the solution's vectors
    // time variables
    sol.begin_time_tasks.assign(inst.nb_tasks, 0);
    sol.end_time_tasks.assign(inst.nb_tasks, 0);
    sol.completion_date_jobs.assign(inst.nb_jobs, 0);
    // choice variables
    sol.machine_choice_tasks.assign(inst.nb_tasks, 0);
    sol.operator_choice_tasks.assign(inst.nb_tasks, 0);


    partial_initialize_time_scheduling_greedy(inst, sol, log_file);
    int time_cursor = 0;
    const int lookahead_duration = 10;
    std::map<int, std::deque<int>> job_stacks;
    for (int j_idx = 0; j_idx < inst.nb_jobs; j_idx++) {
        job_stacks[j_idx] = std::deque<int>(inst.jobs[j_idx].sequence.begin(), inst.jobs[j_idx].sequence.end());
    }
    // Declare the set of pending tasks (should be a small set of indices between each iteration since tasks durations
    // are quite limited in comparison to the lookahead duration)
    std::unordered_map<int, int> pending_tasks_per_job{}; // empty at first


    resolve_lookahead(inst, sol, job_stacks, pending_tasks_per_job, time_cursor, lookahead_duration, log_file);

    log_file.close();
    return 0;
}


