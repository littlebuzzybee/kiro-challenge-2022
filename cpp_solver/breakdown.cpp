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


#include "gurobi_c++.h"
#include "utils.h"



#define QUADRA 0



void get_relevant_tasks(
    Instance& inst,
    Solution& sol,
    int time_cursor,
    int time_horizon,
    std::map<int, std::deque<int>>& job_stacks,
    std::vector<int>& processed_tasks,
    std::vector<int>& processed_jobs,
    std::map<int, std::vector<int>>& processed_tasks_of_jobs,
    std::map<int, int>& cumulative_remainder_time_per_job,
    std::unordered_map<int, int>& pending_task_per_job
) {
    for (int job_idx = 0; job_idx < inst.nb_jobs; ++job_idx) {
        if (job_stacks[job_idx].empty() || inst.jobs[job_idx].release_date >= time_horizon) {
            continue;
            // There are no tasks to schedule on the time window for this job
        }
        else {
            // Only the tasks that are fully comprised in the time window are considered.
            // Those which started before the time window but end in the time window are considered processed, fixed and pending and would not benefit the problem
            // by being postponed again. Their implications are taken into account by the pending_tasks set and not recomputed here.
            int earliest_begin_time = time_cursor;
            if (inst.jobs[job_idx].release_date > time_cursor) {
                earliest_begin_time = inst.jobs[job_idx].release_date;
            }
            // a pending task overrides the earliest begin time because the job was already commenced
            if (pending_task_per_job.contains(job_idx)) {
                // earliest begin time is > time_cursor if there is a pending task for this job overlapping the window's beginning
                int pending_task = pending_task_per_job[job_idx];
                earliest_begin_time = sol.begin_time_tasks[pending_task] + inst.tasks[pending_task].processing_time;
            }

            bool job_is_processed = false;
            while (!job_stacks[job_idx].empty() && earliest_begin_time < time_horizon) {
                int task_idx = job_stacks[job_idx].front();
                job_stacks[job_idx].pop_front();
                earliest_begin_time += inst.tasks[task_idx].processing_time; // update the earliest begin time for the next task in line
                processed_tasks_of_jobs[job_idx].emplace_back(task_idx);
                processed_tasks.emplace_back(task_idx);
                job_is_processed = true;
            }
            if (job_is_processed) {
                processed_jobs.emplace_back(job_idx);
            }
        }
    }



    for (int j_idx = 0; j_idx < inst.nb_jobs; j_idx++) {
        std::deque<int>& remaining_tasks_sequence = job_stacks[j_idx];
        int total_processing_time = std::accumulate(
            remaining_tasks_sequence.begin(),
            remaining_tasks_sequence.end(),
            0, // Initial value of the sum
            [&inst](int total_sum, int t_idx) {
                return total_sum + inst.tasks[t_idx].processing_time;
            }
        );
        cumulative_remainder_time_per_job[j_idx] = total_processing_time;
    }
}



void display_lookahead_program(
    int max_duration_tasks,
    std::vector<int>& processed_tasks,
    std::vector<int>& processed_jobs,
    std::map<int, std::vector<int>>& processed_tasks_of_jobs,
    std::map<int, std::deque<int>>& job_stacks,
    std::unordered_map<int, int>& pending_task_per_job,
    std::ostream& log_stream
) {
    log_stream << "There are " << processed_jobs.size() << " jobs and " << processed_tasks.size() << " tasks being processed this round." << std::endl;

    log_stream << "They comprise the following processed tasks in the lookahead window: ";
    for (int task_idx : processed_tasks) {
        log_stream << task_idx + 1 << "; ";
    }
    log_stream << " distributed as follows: " << std::endl << std::endl;

    for (int job_idx : processed_jobs) {
        log_stream << "J" << job_idx + 1 << " entails " << processed_tasks_of_jobs[job_idx].size() << " tasks ordered as:" << std::setw(2);
        for (int task_idx : processed_tasks_of_jobs[job_idx]) {
            log_stream << "|" << task_idx + 1;
        }
        log_stream << "|";
        log_stream << "   (+ " << job_stacks[job_idx].size() << " remaining)" << std::endl;
    }
    log_stream << std::endl << "There are " << pending_task_per_job.size() << " pending tasks in total: ";
    for (auto& [j_idx, t_idx] : pending_task_per_job) {
        log_stream << "T" << t_idx + 1 << " (J" << j_idx + 1 << ");  ";
    }
    log_stream << std::endl;
    log_stream << "Max duration of processed tasks: " << max_duration_tasks << std::endl << std::endl;
}


void set_begin_variables_and_ordering_constraints(
    Instance& inst,
    Solution& sol,
    GRBModel& model,
    std::map<int, std::map<int, GRBVar>>& begin_times_tasks_per_job,
    std::map<int, std::vector<int>>& processed_tasks_of_jobs,
    std::vector<int>& processed_jobs,
    std::unordered_map<int, int>& pending_task_per_job,
    int time_cursor
) {
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

            // If this is the first task of the job altogether
            if (task_idx == inst.jobs[job_idx].sequence.front()) {
                model.addConstr(
                    GRBLinExpr(inst.jobs[job_idx].release_date),
                    GRB_LESS_EQUAL,
                    begin_times_tasks_per_job[job_idx][task_idx],
                    "earliest_begin_T" + std::to_string(task_idx + 1)
                );
            }


            // Else if this is the first task of the job being optimized in the window
            // but not the first task of the job altogether, and there is a pending task for this job that overlaps the window's beginning
            else if (task_idx == processed_tasks_of_jobs[job_idx].front()) {
                if (pending_task_per_job.contains(job_idx)) { // look for the job index (key) in the map
                    // If there is a pending task for this job overlapping the window's beginning,
                    // its end time is greater than the beginning of the window,
                    // so we set instead the beginning of the current task in loop after the end of the pending task at the soonest

                    int previous_fixed_task = pending_task_per_job[job_idx];
                    model.addConstr(
                        GRBLinExpr(
                            sol.begin_time_tasks[previous_fixed_task] + inst.tasks[previous_fixed_task].processing_time
                        ),
                        GRB_LESS_EQUAL,
                        begin_times_tasks_per_job[job_idx][task_idx],
                        "earliest_begin_T" + std::to_string(task_idx + 1) // EST = earliest start time
                    );
                }

                // Else there is no pending task for this job overlapping the window's beginning:
                else {
                    // so we set the lower bound of the beginning of the task at the time cursor posution (beginning of the window)
                    model.addConstr(
                        GRBLinExpr(time_cursor),
                        GRB_LESS_EQUAL,
                        begin_times_tasks_per_job[job_idx][task_idx],
                        "earliest_begin_T" + std::to_string(task_idx + 1) // EST = earliest start time
                    );
                }
            }
            if (task_idx != processed_tasks_of_jobs[job_idx].back()) {
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
}


void set_slack_and_penalty_variables(
    GRBModel& model,
    std::map<int, GRBVar>& tardiness_slacks,
    std::map<int, GRBVar>& unit_penalties,
    std::vector<int>& processed_jobs
) {
    for (int job_idx : processed_jobs) {
        tardiness_slacks[job_idx] = model.addVar(
            0.0, GRB_INFINITY, 0.0, GRB_INTEGER,
            "due_date_slack_J" + std::to_string(job_idx + 1)
        );
        unit_penalties[job_idx] = model.addVar(
            0.0, 1.0, 0.0, GRB_BINARY,
            "unit_overdue_pen_J" + std::to_string(job_idx + 1)
        );
    }
}



void set_assignment_variables(
    Instance& inst,
    GRBModel& model,
    std::map<int, std::map<int, GRBVar>>& assigned_operators_per_task,
    std::map<int, std::map<int, GRBVar>>& assigned_machines_per_task,
    std::vector<int>& processed_tasks
) {

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
}



void set_workers_ubiquity_constraints(
    Instance& inst,
    GRBModel& model,
    std::map<int, std::map<int, GRBVar>>& assigned_operators_per_task,
    std::map<int, std::map<int, GRBVar>>& assigned_machines_per_task,
    std::vector<int>& processed_tasks
) {

    for (int task_idx : processed_tasks) {

        // Build the list of variables that are involved in the quadratic terms
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

        model.addQConstr(cross_compat <= 1, "workers_exclusion_T" + std::to_string(task_idx + 1));
    }
}



void set_workers_uniqueness_constraints(
    Instance& inst,
    GRBModel& model,
    std::map<int, std::map<int, GRBVar>>& assigned_operators_per_task,
    std::map<int, std::map<int, GRBVar>>& assigned_machines_per_task,
    std::vector<int>& processed_tasks
) {

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
        for (auto& [t_idx, assigned_oper_var] : assigned_operators_per_task[task_idx]) {
            oper_use_var_pointers.emplace_back(assigned_oper_var);
        }
        for (auto& [t_idx, assigned_mach_var] : assigned_machines_per_task[task_idx]) {
            mach_use_var_pointers.emplace_back(assigned_mach_var);
        }

        // Summing the assignment variables for each machine (resp. operator) for the task
        sum_of_machines.addTerms(coeffs_mach.data(), mach_use_var_pointers.data(), n_poss_mach);
        sum_of_operators.addTerms(coeffs_oper.data(), oper_use_var_pointers.data(), n_poss_oper);

        // Exactly one machine and one operator can be assigned to a task: the sum of the assignment variables is 1
        model.addConstr(sum_of_machines == 1, "one_machine_only_T" + std::to_string(task_idx + 1));
        model.addConstr(sum_of_operators == 1, "one_operator_only_T" + std::to_string(task_idx + 1));
        // The equality constraint forces to address each task assignment and leave no orphan task in the middle of a job sequence.
        // In the worst case, the task will be postponed by a significant delay and fall outside the window, in which case it will be reoptimized in the next lookahead window.
    }
}



void set_workers_time_exclusion_constraints(
    Instance& inst,
    Solution& sol,
    GRBModel& model,
    std::map<int, std::map<int, GRBVar>>& begin_times_tasks_per_job,
    std::map<int, std::map<int, GRBVar>>& assigned_operators_per_task,
    std::map<int, std::map<int, GRBVar>>& assigned_machines_per_task,
    std::vector<int>& processed_jobs,
    std::map<int, std::vector<int>>& processed_tasks_of_jobs,
    std::unordered_map<int, int>& pending_task_per_job,
    std::ostream& log_stream
) {

    std::map <std::pair<int, int>, GRBVar> tasks_overlapping_machines;
    std::map <std::pair<int, int>, GRBVar> tasks_overlapping_operators;



    // Iterate over all pairs of different jobs, unordered, once, to set the exclusion constraints
    for (auto j_idx1_ptr = processed_jobs.begin(); j_idx1_ptr != processed_jobs.end(); ++j_idx1_ptr) {

        auto j_idx2_ptr = j_idx1_ptr; ++j_idx2_ptr;
        for (; j_idx2_ptr != processed_jobs.end(); ++j_idx2_ptr) {
            // Given a pair of jobs, iterate over all pairs of tasks (one from each job)
            // This means we consider each edge of the corresponding bipartite graph
            for (int t_idx1 : processed_tasks_of_jobs[*j_idx1_ptr]) {
                for (int t_idx2 : processed_tasks_of_jobs[*j_idx2_ptr]) {
                    // log_stream << "T" << t_idx1 + 1 << "-T" << t_idx2 + 1 << "; ";
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
                    // We name those two conditions "overlap triggers" and set them as binary variables
                    // So that (e1 - b2 >= 1) => (trigg1 = TRUE)
                    // and     (e2 - b1 >= 1) => (trigg2 = TRUE)
                    // Two triggers TRUE means the two tasks overlap in time
                    model.addGenConstrIndicator(
                        ind1,
                        0, // (trigg1 = FALSE) => (end1 - begin2 <= 0)
                        bt1 + pt1 - bt2, GRB_LESS_EQUAL, 0,
                        "bind_intersect_indL_" + task_pair_str // left
                    );
                    model.addGenConstrIndicator(
                        ind2,
                        0, // (trigg2 = FALSE) = > (end2 - begin1 <= 0)
                        bt2 + pt2 - bt1, GRB_LESS_EQUAL, 0,
                        "bind_intersect_indR_" + task_pair_str // right
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
#if QUADRA
                        model.addQConstr(
                            ind1 * ind2 + t1_uses_op * t2_uses_op <= 1,
                            "bind_overlap_O" + std::to_string(op_idx + 1) + "_" + task_pair_str
                        );
#else
                        model.addConstr(
                            ind1 + ind2 + t1_uses_op + t2_uses_op <= 3,
                            "bind_overlap_O" + std::to_string(op_idx + 1) + "_" + task_pair_str
                        );
#endif
                    }

                    // Prevent that the assigned machines overlap between the two tasks if they overlap in time
                    for (int ma_idx : intersection_machines) {
                        GRBVar& t1_uses_ma = assigned_machines_per_task[t_idx1][ma_idx];
                        GRBVar& t2_uses_ma = assigned_machines_per_task[t_idx2][ma_idx];

#if QUADRA
                        model.addQConstr(
                            ind1 * ind2 + t1_uses_ma * t2_uses_ma <= 1,
                            "bind_overlap_M" + std::to_string(ma_idx + 1) + "_" + task_pair_str
                        );
#else
                        model.addConstr(
                            ind1 + ind2 + t1_uses_ma + t2_uses_ma <= 3,
                            "bind_overlap_M" + std::to_string(ma_idx + 1) + "_" + task_pair_str
                        );
#endif
                    }
                }
            }
        }
    }

    // If this job has a pending task at the beginning of the window, we must take it into account
    for (int j_idx1 = 0; j_idx1 < inst.nb_jobs; j_idx1++) {

        if (!pending_task_per_job.contains(j_idx1)) {
            continue;
            // No pending task for this job, move on
        }


        // get the pending task for the job
        int pend_t_idx = pending_task_per_job[j_idx1];

        for (int& j_idx2 : processed_jobs) {
            if (j_idx1 == j_idx2) {
                continue;
                // in such case, constraints are taken care of by precedence constraints
            }


            for (int proc_t_idx : processed_tasks_of_jobs[j_idx2]) {
                // iterate over all tasks of the second job != first job


                std::set<int>& proc_t_operators = inst.tasks[proc_t_idx].operators;
                std::set<int>& proc_t_machines = inst.tasks[proc_t_idx].machines;

                int machine_of_pending_task = sol.machine_choice_tasks[pend_t_idx];
                int operator_of_pending_task = sol.operator_choice_tasks[pend_t_idx];

                // If the pending task's operator is not in the possible operators for the processed task
                // AND the pending task's machine is not in the possible machines for the processed task
                // THEN we pass

                if (!proc_t_operators.contains(operator_of_pending_task) && !proc_t_machines.contains(machine_of_pending_task)) {
                    continue;
                }

                // ELSE, we exclude the possibility of overlapping resources between the two tasks

                // Retrieve the begin time of the processed task and the end time of the pending task
                int end_time_pend = sol.begin_time_tasks[pend_t_idx] + inst.tasks[pend_t_idx].processing_time;
                GRBVar& begin_time_proc = begin_times_tasks_per_job[j_idx2][proc_t_idx];

                std::string task_pair_str = "T" + std::to_string(pend_t_idx + 1) + "_T" + std::to_string(proc_t_idx + 1);

                // Prevent the assigned operators from overlapping between the two tasks if they overlap in time
                if (proc_t_operators.contains(operator_of_pending_task)) {
                    GRBVar& proc_task_uses_pend_op = assigned_operators_per_task[proc_t_idx][operator_of_pending_task];

                    // Add the implication constraint: (proc_task uses pend_op) => (begin_processed - end_pending >= 0)
                    model.addGenConstrIndicator(
                        proc_task_uses_pend_op,
                        1,
                        begin_time_proc - end_time_pend, GRB_GREATER_EQUAL, 0,
                        "no_pending_overlap_operator_" + task_pair_str
                    );

                }

                // Prevent the assigned machines from overlapping between the two tasks if they overlap in time
                if (proc_t_machines.contains(machine_of_pending_task)) {
                    GRBVar& proc_task_uses_pend_ma = assigned_machines_per_task[proc_t_idx][machine_of_pending_task];
                    model.addGenConstrIndicator(
                        proc_task_uses_pend_ma,
                        1,
                        begin_time_proc - end_time_pend, GRB_GREATER_EQUAL, 0,
                        "no_pending_overlap_machine_" + task_pair_str
                    );

                }
            }
        }
    }
}


void set_completion_time_and_penalty_constraints(
    Instance& inst,
    GRBModel& model,
    std::map<int, GRBVar>& tardiness_slacks,
    std::map<int, GRBVar>& unit_penalties,
    std::map<int, GRBLinExpr>& new_completion_dates_jobs,
    std::map<int, std::map<int, GRBVar>>& begin_times_tasks_per_job,
    std::vector<int>& processed_jobs,
    std::map<int, std::vector<int>>& processed_tasks_of_jobs,
    std::map<int, int>& cumulative_remainder_time_per_job
) {
    for (int job_idx : processed_jobs) {
        int last_task_of_job = processed_tasks_of_jobs[job_idx].back();

        new_completion_dates_jobs[job_idx] = GRBLinExpr(begin_times_tasks_per_job[job_idx][last_task_of_job] + cumulative_remainder_time_per_job[job_idx]);
    }


    // Set constraint for completion time: completion_time <= due_date + slack
    for (int job_idx : processed_jobs) {
        model.addConstr(
            new_completion_dates_jobs[job_idx] <= inst.jobs[job_idx].due_date + tardiness_slacks[job_idx],
            "slack_deadline_J" + std::to_string(job_idx + 1)
        );
    }


    // Set unit penalty constraints
    for (int job_idx : processed_jobs) {
        // [completion_time > due_date] => [unit_penalty = 1]
        // [unit_penalty = 0] => [completion_time <= due_date] i.e. tardiness_slack <= 0
        model.addGenConstrIndicator(
            unit_penalties[job_idx],
            0, // false
            tardiness_slacks[job_idx], GRB_LESS_EQUAL, 0,
            "bind_unit_pen_J" + std::to_string(job_idx + 1)
        );
    }
}





void set_objective_function(
    Instance& inst,
    GRBModel& model,
    std::vector<int>& processed_jobs,
    std::map<int, GRBLinExpr>& new_completion_dates_jobs,
    std::map<int, GRBVar>& tardiness_slacks,
    std::map<int, GRBVar>& unit_penalties
) {
    GRBLinExpr objective = 0;
    for (int job_idx : processed_jobs) {
        // Set interim costs
        objective += inst.jobs[job_idx].weight * new_completion_dates_jobs[job_idx];
        // Set tardiness costs
        objective += inst.jobs[job_idx].weight * inst.tardiness * tardiness_slacks[job_idx];
        // Set unit penalty costs
        objective += inst.jobs[job_idx].weight * inst.unit_penalty * unit_penalties[job_idx];
    }
    model.setObjective(objective, GRB_MINIMIZE);
}




void greedy_partial_solve_lookahead(
    Instance& inst,
    Solution& sol,
    int time_cursor,
    int time_horizon,
    std::map<int, std::deque<int>>& job_stacks,
    std::unordered_map<int, int>& pending_task_per_job,
    std::ostream& log_stream
) {
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
        next_time_persue_job[j_idx] = std::max(time_cursor, inst.jobs[j_idx].release_date); // ICI METTRE LE RELEASE DATE
    }


    // Remove resources currently used by pending tasks
    for (auto it = pending_task_per_job.begin(); it != pending_task_per_job.end(); ++it) {
        auto& [parent_job, pend_task_idx] = *it;
        available_machines.erase(sol.machine_choice_tasks[pend_task_idx]);
        available_operators.erase(sol.operator_choice_tasks[pend_task_idx]);
        int end_of_pending_task = sol.begin_time_tasks[pend_task_idx] + inst.tasks[pend_task_idx].processing_time;
        assert(end_of_pending_task > time_cursor);

        release_calendar_machines[end_of_pending_task].insert(sol.machine_choice_tasks[pend_task_idx]);
        release_calendar_operators[end_of_pending_task].insert(sol.operator_choice_tasks[pend_task_idx]);

        next_time_persue_job[parent_job] = end_of_pending_task;
    }

    int time_pos{ time_cursor };

    log_stream << "~~~ Decisions made by the heuristic on this window: ~~~" << std::endl;
    while (!all_stacks_are_empty(job_stacks)) { // && time_pos < time_horizon
        // TODO: take into account the tardiness of each job in a priority queue
        // First release the resources that are no longer used
        for (int m_idx : release_calendar_machines[time_pos]) {
            available_machines.insert(m_idx);
        }
        for (int o_idx : release_calendar_operators[time_pos]) {
            available_operators.insert(o_idx);
        }

        for (auto& [j_idx, task_stack] : job_stacks) {
            // Heuristic: iterate through jobs in order at each time step and
            // assign tasks and resources one at a time

            // Process the first task in line for the job if it exists and if the processsing time of its predecessor is over
            if (task_stack.empty() || time_pos < next_time_persue_job[j_idx]) {
                continue;
            }
            int t_idx = task_stack.front();

            if (available_machines.empty() || available_operators.empty()) {
                continue;
            }

            // Look for the first available machine and operator available for that task
            // std::set<int> authorized_machines = inst.tasks[t_idx].machines;
            // std::set<int> authorized_operators = inst.tasks[t_idx].operators;

            std::vector<int> possible_machines;
            std::set_intersection(
                available_machines.begin(), available_machines.end(),
                inst.tasks[t_idx].machines.begin(), inst.tasks[t_idx].machines.end(),
                std::back_inserter(possible_machines)
            );

            if (possible_machines.empty()) {
                continue;
            }

            int chosen_machine{ -1 };
            int chosen_operator{ -1 };
            std::vector<int> possible_operators;

            for (int m_idx : possible_machines) {
                possible_operators.clear();
                std::set_intersection(
                    available_operators.begin(), available_operators.end(),
                    inst.tasks[t_idx].operators.begin(), inst.tasks[t_idx].operators.end(),
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
            // Remove the task from the stack
            task_stack.pop_front();

            // Display the assignment in the log stream
            log_stream << "* T" << t_idx + 1 << " (J" << j_idx + 1 << ") scheduled at t=" << time_pos << " with M" << chosen_machine + 1 << " & O" << chosen_operator + 1;
            if (time_cursor >= time_horizon) {
                log_stream << " (postponed)";
            }
            else if (time_pos + inst.tasks[t_idx].processing_time > time_horizon) {
                log_stream << " (pending in subsequent window)";
            }
            log_stream << std::endl;

        }
        time_pos++;
    }
}


void warmup_solution(
    Instance& inst,
    Solution& sol,
    std::vector<int> processed_tasks,
    std::map<int, std::map<int, GRBVar>> assigned_operators_per_task,
    std::map<int, std::map<int, GRBVar>> assigned_machines_per_task,
    std::map<int, std::map<int, GRBVar>> begin_times_tasks_per_job
) {
    for (auto& t_idx : processed_tasks) {
        // Preset the start time
        int default_start_time = sol.begin_time_tasks[t_idx];
        GRBVar& start_time_task = begin_times_tasks_per_job[inst.tasks[t_idx].job_parent][t_idx];
        start_time_task.set(GRB_DoubleAttr_Start, double(default_start_time));

        // Preset the machine and operator assignments
        int default_machine = sol.machine_choice_tasks[t_idx];
        GRBVar& machine_assignment = assigned_machines_per_task[t_idx][default_machine];
        machine_assignment.set(GRB_DoubleAttr_Start, double(1.0));

        int default_operator = sol.operator_choice_tasks[t_idx];
        GRBVar& operator_assignment = assigned_operators_per_task[t_idx][default_operator];
        operator_assignment.set(GRB_DoubleAttr_Start, double(1.0));
    }
}
