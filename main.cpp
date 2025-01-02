#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <set>


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


void resolve_lookahead(Instance &inst, Solution &sol, int time_cursor, int lookahead_duration, std::ostream& log_stream = std::cout) {
    const int time_horizon = time_cursor + lookahead_duration;
    log_stream << "Resolving lookahead on time window [" << time_cursor << ", " << time_horizon << "]..." << std::endl;


    std::map<int, std::set<int>> concerned_tasks;
    std::set<int> concerned_jobs;

    for (int task_idx = 0; task_idx < inst.nb_tasks; ++task_idx) {
        if (time_cursor <= sol.begin_time_tasks[task_idx] && sol.begin_time_tasks[task_idx] < time_horizon) {
            int corresponding_job = inst.tasks[task_idx].job_parent;
            concerned_tasks[corresponding_job].insert(task_idx);
            concerned_jobs.insert(corresponding_job); // inserts an index, not an id
        }
    }

    int nb_concerned_tasks = concerned_tasks.size();
    int nb_concerned_jobs  = concerned_jobs.size();


    // Initialize Gurobi environment and model
    GRBEnv env = GRBEnv(true);
    env.set("LogFile", "gurobi.log");
    env.start();
    GRBModel model = GRBModel(env);
    model.set(GRB_StringAttr_ModelName, "time_scheduling_round_"+std::to_string(1));

    

    // Declare postpone values and slacks and unit penalties
    std::map<int, GRBVar> postponement_vals;
    std::map<int, GRBVar> postponement_slacks;
    std::map<int, GRBVar> unit_penalties;

    // Changer cela: postponement_vals devra être la somme des postponements des tâches concernées
    for (int job_idx : concerned_jobs) {
        postponement_vals[job_idx]   = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER,
                                                    "postpone_" + std::to_string(job_idx));
        postponement_slacks[job_idx] = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER,
                                                    "slack_" + std::to_string(job_idx));
        unit_penalties[job_idx] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                                               "unit_penalty_" + std::to_string(job_idx));
        log_stream << "Added postponement, slack & unit penalty variables for job " << job_idx << std::endl;
    }

    // Declare completion times as variables
    std::map<int, GRBLinExpr> completion_times_vals;
    for (int job_idx : concerned_jobs) {
            completion_times_vals[job_idx] = GRBLinExpr(sol.completion_date_jobs[job_idx]
                                + postponement_vals[job_idx]);
            log_stream << "Added completion time variable for job " << job_idx << std::endl;
        }

    // Set constraint for completion time: completion_time <= due_date + slack
    std::map<int, GRBConstr> tardiness_time_constraints;
    for (int job_idx : concerned_jobs) {
        tardiness_time_constraints[job_idx] = model.addConstr(completion_times_vals[job_idx] <= inst.jobs[job_idx].due_date 
                        + postponement_slacks[job_idx]);
        log_stream << "Added completion date constraint for job " << job_idx << std::endl;
    }


    // Set unit penalty variables
    std::map<int, GRBGenConstr> unit_penalty_constraints;
    for (int job_idx : concerned_jobs) {
        // unit_penalty = 1 if completion_time > due_date
        unit_penalty_constraints[job_idx] = model.addGenConstrIndicator(unit_penalties[job_idx], 0,
                        completion_times_vals[job_idx], GRB_LESS_EQUAL, inst.jobs[job_idx].due_date);
        log_stream << "Added unit penalty constraint for job " << job_idx << std::endl;
    }


    // Declare objective function
    GRBLinExpr objective = 0;

    
    for (int job_idx : concerned_jobs) {
        // Set interim costs
        objective += inst.jobs[job_idx].weight * completion_times_vals[job_idx];
        // Set tardiness costs
        objective += inst.tardiness * inst.jobs[job_idx].weight * postponement_slacks[job_idx];
        // Set unit penalty costs
        objective += inst.unit_penalty * inst.jobs[job_idx].weight * unit_penalties[job_idx];
    }

    // Set objective function
    model.setObjective(objective, GRB_MINIMIZE);
    log_stream << "Set objective function" << std::endl;
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
    resolve_lookahead(inst, sol, time_cursor, lookahead_duration, log_file);

    log_file.close();
    return 0;
}


