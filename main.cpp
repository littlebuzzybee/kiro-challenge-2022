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
            int processed_task = inst.jobs[j_idx].sequence[t_idx] ;
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
        assert(sol.completion_date_jobs[j_idx] == sol.end_time_tasks[inst.jobs[j_idx].sequence.back()]);
        // Print job completion date
        log_stream << "Job " << j_idx << " ** completion date: " << sol.completion_date_jobs[j_idx] << std::endl << std::endl;
        }
}


void resolve_lookahead(Instance &inst, Solution &sol, std::map<int, std::deque<int>> job_stacks, std::unordered_map<int, int> pending_tasks_per_job, int time_cursor, int lookahead_duration, std::ostream& log_stream = std::cout) {
    const int time_horizon = time_cursor + lookahead_duration;
    log_stream << "Resolving lookahead on time window [" << time_cursor << ", " << time_horizon << "]..." << std::endl;


    std::map<int, std::vector<int>> processed_tasks_of_jobs; // ordered set of tasks for each job
    std::set<int> processed_jobs;



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
            while (    !job_stacks[job_idx].empty()
                    && time_cursor <= sol.begin_time_tasks[job_stacks[job_idx].front()]
                    &&                sol.begin_time_tasks[job_stacks[job_idx].front()] <= time_horizon
                    ) {
                int task_idx = job_stacks[job_idx].front();
                job_stacks[job_idx].pop_front();
                processed_tasks_of_jobs[job_idx].push_back(task_idx);
            }
        }

    }

    // int nb_processed_tasks = processed_tasks_of_jobs.size();
    int nb_processed_jobs  = processed_jobs.size();
    int nb_pending_tasks   = pending_tasks_per_job.size();

    std::cout << "There are " << nb_processed_jobs << " jobs processed." << std::endl;
    for (int job_idx : processed_jobs) {
        std::cout << "Job " << job_idx << " has " << processed_tasks_of_jobs[job_idx].size() << " tasks processed:" << std::endl;
        for (int task_idx : processed_tasks_of_jobs[job_idx]) {
            std::cout << task_idx << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "There are " << nb_pending_tasks << " pending tasks in total." << std::endl;




    // Initialize Gurobi environment and model
    GRBEnv env = GRBEnv(true);
    env.set("LogFile", "gurobi.log");
    env.start();
    GRBModel model = GRBModel(env);
    model.set(GRB_StringAttr_ModelName, "time_scheduling_round_"+std::to_string(1));


    // Declare the begin times of each task and set the ordering constraints
    std::map<int, std::map<int, GRBVar>> begin_times_tasks_per_job;

    for (int job_idx : processed_jobs) {
        // First, Declare the begin times variables of each task
        std::cout << "Declaring variables for job " << job_idx << std::endl;

        for (int task_idx : processed_tasks_of_jobs[job_idx]) {

            GRBVar new_var = model.addVar(time_cursor, GRB_INFINITY, 0.0, GRB_INTEGER,
                                                               "begin_time_task_" + std::to_string(task_idx));
            begin_times_tasks_per_job[job_idx].emplace(task_idx, new_var);
            assert(begin_times_tasks_per_job[job_idx].contains(task_idx));

            std::cout << "Successfully added begin time variable for task " << task_idx << std::endl;
        }

        for (int task_idx : processed_tasks_of_jobs[job_idx]) {
            std::cout << "Task " << task_idx << " is processed in job " << job_idx << std::endl;
        }
        std::cout << "Front: " << processed_tasks_of_jobs[job_idx].front() << std::endl;
        std::cout << "Back: " << processed_tasks_of_jobs[job_idx].back() << std::endl;

        for (int task_idx : processed_tasks_of_jobs[job_idx]) {
            std::cout << "Task " << task_idx << " will be met." << std::endl;
        }

        // Then, set the ordering constraints
        for (int &task_idx : processed_tasks_of_jobs[job_idx]) {
            if (task_idx == processed_tasks_of_jobs[job_idx].front()) {
                std::cout << "Task " << task_idx << " is the first task of job " << job_idx << std::endl;
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
            else if (task_idx == processed_tasks_of_jobs[job_idx].back()) {
                std::cout << "Task " << task_idx << " is neither the last task of job " << job_idx << std::endl;
                continue;
            }
            else if ((task_idx != processed_tasks_of_jobs[job_idx].front()) && (task_idx != processed_tasks_of_jobs[job_idx].back())) {
                std::cout << "Task " << task_idx << " is neither the first nor the last task of job " << job_idx << std::endl;
                // If this is not the first nor the last task of the job being optimized in the window,
                // we prevent the task from starting before the end of the previous task (no overlapping)
                std::cout << "Adding constraint between tasks " << task_idx << " and " << task_idx + 1 << std::endl;
                try {
                        assert(begin_times_tasks_per_job[job_idx].contains(task_idx));
                        assert(begin_times_tasks_per_job[job_idx].contains(task_idx + 1));
                        model.addConstr(GRBLinExpr(begin_times_tasks_per_job[job_idx][task_idx] + inst.tasks[task_idx].processing_time),
                                        GRB_LESS_EQUAL,
                                        begin_times_tasks_per_job[job_idx][task_idx + 1],
                                        "rel_ordering_task_" + std::to_string(task_idx));
                    } catch (GRBException& e) {
                        std::cerr << "Error code = " << e.getErrorCode() << std::endl;
                        std::cerr << e.getMessage() << std::endl;
                    } catch (...) {
                        std::cerr << "Exception during optimization" << std::endl;
                    }
            }
            std::cout << "Added time ordering constraint for task " << task_idx << std::endl;
        }
    }


    // Declare slacks and unit penalties variables for each job
    std::map<int, GRBVar> tardiness_post_slacks;
    std::map<int, GRBVar> unit_penalties;


    for (int job_idx : processed_jobs) {
        tardiness_post_slacks[job_idx] = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER,
                                                    "slack_" + std::to_string(job_idx));
        unit_penalties[job_idx] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                                               "unit_penalty_" + std::to_string(job_idx));
        log_stream << "Added slack variable for job " << job_idx << std::endl;
    }


    // Define the resulting completion dates variable of each processed job
    // as the sum of its current completion date and the postponement due to task delays
    std::map<int, GRBLinExpr> additional_delays_jobs;
    std::map<int, GRBLinExpr> new_completion_dates_jobs;
    for (int job_idx : processed_jobs) {
            int last_task_of_job = processed_tasks_of_jobs[job_idx].back();

            additional_delays_jobs[job_idx]    = GRBLinExpr(begin_times_tasks_per_job[job_idx][last_task_of_job] - sol.begin_time_tasks[last_task_of_job]);
            new_completion_dates_jobs[job_idx] = GRBLinExpr(sol.completion_date_jobs[job_idx] + additional_delays_jobs[job_idx]);
            log_stream << "Added completion date variable for job " << job_idx << std::endl;
        }


    // Set constraint for completion time: completion_time <= due_date + slack
    std::map<int, GRBConstr> tardiness_time_constraints;
    for (int job_idx : processed_jobs) {
        tardiness_time_constraints[job_idx] = model.addConstr(new_completion_dates_jobs[job_idx] <= inst.jobs[job_idx].due_date 
                        + tardiness_post_slacks[job_idx]);
        log_stream << "Added completion date constraint for job " << job_idx << std::endl;
    }


    // Set unit penalty variables
    std::map<int, GRBGenConstr> unit_penalty_constraints;
    for (int job_idx : processed_jobs) {
        // unit_penalty = 1 if completion_time > due_date
        unit_penalty_constraints[job_idx] = model.addGenConstrIndicator(unit_penalties[job_idx], 0,
                        new_completion_dates_jobs[job_idx], GRB_LESS_EQUAL, inst.jobs[job_idx].due_date);
        log_stream << "Added unit penalty constraint for job " << job_idx << std::endl;
    }


    // Declare objective function
    GRBLinExpr objective = 0;

    
    for (int job_idx : processed_jobs) {
        // Set interim costs
        objective += inst.jobs[job_idx].weight * new_completion_dates_jobs[job_idx];
        // Set tardiness costs
        objective += inst.tardiness * inst.jobs[job_idx].weight * tardiness_post_slacks[job_idx];
        // Set unit penalty costs
        objective += inst.unit_penalty * inst.jobs[job_idx].weight * unit_penalties[job_idx];

        log_stream << "Added costs for job " << job_idx << std::endl;
    }

    // Set objective function
    model.setObjective(objective, GRB_MINIMIZE);
    log_stream << "Set objective function" << std::endl;


    // Delay all upcoming tasks coming after the horizon by the job's delay that was just resolved

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
    std::unordered_map<int, int> pending_tasks_per_job {}; // empty at first


    resolve_lookahead(inst, sol, job_stacks, pending_tasks_per_job, time_cursor, lookahead_duration, log_file);

    log_file.close();
    return 0;
}


