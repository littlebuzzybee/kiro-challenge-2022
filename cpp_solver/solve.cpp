#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <set>
#include <map>
#include <deque>
#include <set>


#include "utils.h"
#include "gurobi_c++.h"


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
        // completion_time > due_date => unit_penalty = 1
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
    std::map<int, int> cumulative_remainder_time_per_job;
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
            cumulative_remainder_time_per_job,
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
            cumulative_remainder_time_per_job
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

        for (auto& [j_idx, task_stack] : job_stacks) {
            // Heuristic: iterate through jobs in order at each time step and
            // assign tasks and resources one at a time


            // Process the first task in line for the job if it exists and if the processsing time of its predecessor is over
            if (task_stack.empty() || time_pos < next_time_persue_job[j_idx]) {
                continue;
            }

            if (available_machines.empty() || available_operators.empty()) {
                continue;
            }

            int t_idx = task_stack.front();

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
            log_stream << "Task T" << t_idx + 1 << " (J" << j_idx + 1 << ") assigned to M" << chosen_machine + 1 << " & O" << chosen_operator + 1 << " at time " << time_pos << std::endl;
        }
        time_pos++;
    }
    log_stream << std::endl << "End of solving procedure." << std::endl << std::endl;
}
