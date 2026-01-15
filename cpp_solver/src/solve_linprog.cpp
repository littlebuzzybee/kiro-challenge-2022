#include "solve_linprog.h"


void resolve_linprog(
    Instance& inst,
    Solution& sol,
    std::map<int, std::deque<int>>& job_stacks,
    std::ostream& log_stream
) {
    log_stream << "Beginning solving procedure with a local LP timestep search heuristic...\n";


    std::set<int> available_machines{};
    std::set<int> available_operators{};

    // Create a pool of resources
    for (int m_idx = 0; m_idx < inst.nb_machines; ++m_idx) {
        available_machines.insert(m_idx);
    }
    for (int o_idx = 0; o_idx < inst.nb_operators; ++o_idx) {
        available_operators.insert(o_idx);
    }

    // Create data structures for the release of resources by key time position
    std::map<int, std::set<int>> release_calendar_machines{};
    std::map<int, std::set<int>> release_calendar_operators{};

    // Create a data structure to enforce the precedence
    std::map<int, int> next_time_persue_job;
    for (int j_idx = 0; j_idx < inst.nb_jobs; ++j_idx) {
        next_time_persue_job[j_idx] = inst.jobs[j_idx].release_date;
    }

    // Compute the cumulative remaining time for each job
    std::map<int, int> cumulative_remaining_time_per_job;
    get_cumulative_remaining_time_per_job(
        cumulative_remaining_time_per_job,
        inst,
        job_stacks
    );

    int time_pos{ 0 };

    int fail_safe{ 0 };
    while (!all_stacks_are_empty(job_stacks) && fail_safe < 100) {
        fail_safe++;
        log_stream << "\n================== *** Time " << time_pos << " *** ==================\n";


        // First release the resources that are no longer used
        release_idle_resources(
            available_machines,
            available_operators,
            release_calendar_machines[time_pos],
            release_calendar_operators[time_pos]
        );

        // Display the released resources
        log_stream << "Released resources:\n";
        log_stream << "M";
        print_set(release_calendar_machines[time_pos], 1, log_stream);
        log_stream << "\nO";
        print_set(release_calendar_operators[time_pos], 1, log_stream);
        log_stream << '\n';

        // Display the available resources
        log_stream << "Available resources:\n";
        log_stream << "M";
        print_set(available_machines, 1, log_stream);
        log_stream << "\nO";
        print_set(available_operators, 1, log_stream);
        log_stream << '\n';

        // ====== Organize the tasks to be processed per order of priority ======
        // We insert the tasks that are ready to be processed from the job stacks into in a new priority queue to be ranked and compared
        // There is at most only one task addressed for each job in the queue and we decide which of them are going to be processed at this time position to minimize the upcoming tardiness score with a BFS exploration 

        // Create a vector of candidate tasks indexes to be processed, and sort them by priority w.r.t. the score depending on the current tardiness of the job
        std::vector<std::tuple<float, int>> candidate_tasks{}; // first integer is the tardiness score, second integer is the task index
        get_sort_tasks_and_scores(
            candidate_tasks,
            inst,
            job_stacks,
            cumulative_remaining_time_per_job,
            next_time_persue_job,
            time_pos
        );


        
        if (candidate_tasks.empty()) {
            log_stream << "No candidate tasks to process at this time position. Continuing.\n";
            time_pos++;
            continue;
        }



        // Display the candidate tasks
        log_stream << "Candidate tasks for processing (priority score): { ";
        for (auto& [score, t_idx] : candidate_tasks) {
            log_stream << "T" << t_idx + 1 << "(" << score << ") ";
        }
        log_stream << " }\n";

        // ====== Create the GLOP solver ======
        std::unique_ptr<operations_research::MPSolver> solver(operations_research::MPSolver::CreateSolver("GLOP"));
        if (!solver) {
            log_stream << "Solver not created. Exiting...\n";
            return;
        }

        // ====== Create decision variables ======
        // Only one decision variable per exact combination (task, machine, operator) is created


        std::map<int, std::vector<operations_research::MPVariable*>> tasks_assignments_vars{}; // task -> vector of assignment variables for a given (Task, Machine, Operator)
        std::map<int, std::vector<std::tuple<int, int>>> tasks_assignments{}; // task -> tuple of (machine, operator) for the assignment
        std::map<int, operations_research::MPConstraint*> tasks_assignments_constr{}; // task -> constraint on the number of assignments addressing it
        std::map<int, operations_research::MPConstraint*> machines_assignments_constr{}; // machine -> constraint on the number of assignments using it
        std::map<int, operations_research::MPConstraint*> operators_assignments_constr{}; // operator -> constraint on the number of assignments using it



        for (auto& [_, t_idx] : candidate_tasks) {
            tasks_assignments_constr[t_idx] = solver->MakeRowConstraint(
                0.0,
                1.0
            ); // Exlude the task from being assigned several times by different resources (op, ma)

            for (auto& [m_idx, op_list] : inst.tasks[t_idx].compatibility) {
                if (!available_machines.contains(m_idx)) { continue; }

                for (auto& o_idx : op_list) {
                    if (!available_operators.contains(o_idx)) { continue; }

                    tasks_assignments_vars[t_idx].emplace_back(
                        solver->MakeIntVar(
                            0.0,
                            1.0,
                            "T" + std::to_string(t_idx) + "_M" + std::to_string(m_idx) + "_O" + std::to_string(o_idx)
                        ) // Binary variable for the assignment of a given task to a machine and an operator
                    );
                    tasks_assignments[t_idx].emplace_back(std::make_tuple(m_idx, o_idx));

                    tasks_assignments_constr[t_idx]->SetCoefficient(tasks_assignments_vars[t_idx].back(), 1.0);

                    if (!machines_assignments_constr.contains(m_idx)) {
                        machines_assignments_constr[m_idx] = solver->MakeRowConstraint(
                            0.0,
                            1.0
                        ); // Exclude the machine from being assigned several times to one given task
                    }
                    if (!operators_assignments_constr.contains(o_idx)) {
                        operators_assignments_constr[o_idx] = solver->MakeRowConstraint(
                            0.0,
                            1.0
                        ); // Exclude the operator from being assigned several times to one given task
                    }

                    machines_assignments_constr[m_idx]->SetCoefficient(tasks_assignments_vars[t_idx].back(), 1.0);
                    operators_assignments_constr[o_idx]->SetCoefficient(tasks_assignments_vars[t_idx].back(), 1.0);
                }
            }
        }



        // ====== Create the objective function ======
        operations_research::MPObjective* const objective = solver->MutableObjective();

        for (auto& [score, t_idx] : candidate_tasks) {
            for (auto& var : tasks_assignments_vars[t_idx]) {
                // The objective function is to maximize the sum of the scores of the tasks that are addressed
                objective->SetCoefficient(var, score);
            }
        }
        objective->SetMaximization();

        log_stream << "Number of constraints = " << solver->NumConstraints() << '\n';
        log_stream << "Number of variables = " << solver->NumVariables() << '\n';
        const operations_research::MPSolver::ResultStatus result_status = solver->Solve();

        // Check that the problem has an optimal solution.
        if (result_status != operations_research::MPSolver::FEASIBLE && result_status != operations_research::MPSolver::OPTIMAL) {
            log_stream << "The problem does not have a feasible or optimal solution!\n";
        }

        log_stream << "Problem solved in " << solver->wall_time() << " milliseconds & " << solver->iterations() << " iterations\n";


        std::map <int, std::tuple<int, int>> tasks_chosen_resources{};
        for (auto& [t_idx, candidate_resources] : tasks_assignments) {
            for (size_t i = 0; i < candidate_resources.size(); ++i) {
                operations_research::MPVariable* var = tasks_assignments_vars[t_idx][i];
                if (var->solution_value() > 0.5) {
                    tasks_chosen_resources[t_idx] = candidate_resources[i];
                }
            }
        }

        
        log_stream << "Overhead tardiness score: " << objective->Value() << '\n';
        log_stream << "\nAssigned tasks are:\n";


        for (auto & [t_idx, resources] : tasks_chosen_resources) {
            int j_idx = inst.tasks[t_idx].job_parent;
            int chosen_machine = std::get<0>(resources);
            int chosen_operator = std::get<1>(resources);

            // Assign the workers to the task
            sol.machine_choice_tasks[t_idx] = chosen_machine;
            sol.operator_choice_tasks[t_idx] = chosen_operator;

            // Remove the workers from the available pool
            available_machines.erase(chosen_machine);
            available_operators.erase(chosen_operator);

            // Schedule the task
            sol.begin_time_tasks[t_idx] = time_pos;

            // Prevent the subsequent assignment of any other task of the same job before the end of the current task
            next_time_persue_job[j_idx] = time_pos + inst.tasks[t_idx].processing_time;

            // Update the release calendar for the chosen machine and operator
            release_calendar_machines[time_pos + inst.tasks[t_idx].processing_time].insert(chosen_machine);
            release_calendar_operators[time_pos + inst.tasks[t_idx].processing_time].insert(chosen_operator);
            cumulative_remaining_time_per_job[j_idx] -= inst.tasks[t_idx].processing_time;

            // Remove the task from the stack
            job_stacks[j_idx].pop_front();
            log_stream << " - Task T" << t_idx + 1 << " (J" << j_idx + 1 << ") assigned to M" << chosen_machine + 1 << " & O" << chosen_operator + 1 << '\n';
        }
        log_stream << "\nAssigned " << tasks_chosen_resources.size() << " tasks of " << candidate_tasks.size() << " this round (" << static_cast<float>(tasks_chosen_resources.size()) / static_cast<float>(candidate_tasks.size()) * 100.0f << "%).\n";
        time_pos++;
    }
    log_stream << "\n================== *** End of procedure *** ==================\n";
}
