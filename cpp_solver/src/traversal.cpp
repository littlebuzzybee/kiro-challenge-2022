#include "traversal.h"
#include "traversal_breakdown.h"

#define DEBUG 0

#if DEBUG
#include "ortools/linear_solver/linear_solver.h"
#endif

void resolve_traversal(
    Instance& inst,
    Solution& sol,
    std::map<int, std::deque<int>>& job_stacks,
    std::ostream& log_stream
) {
    log_stream << "Beginning solving procedure with a priority dispatch + graph-pruning + greedy matching search heuristic...\n";


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
        std::cout << "Beginning solving procedure at time " << time_pos << '\n';

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

        // Create a vector of candidate tasks indexes to be processed, and sort them by priority
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

        // Compute the characteristics of the available workers pool this round
        const int nb_candidate_tasks = static_cast<int>(candidate_tasks.size());

        // Sort the candidate tasks by highest priority to get the most urgent tasks first (those with the highest tardiness score so far)
        std::sort(candidate_tasks.begin(), candidate_tasks.end(), std::greater<std::tuple<int, int>>());

        // Display the candidate tasks
        log_stream << "\nThere are " << nb_candidate_tasks << " candidate tasks for processing (priority score):\n { ";
        for (auto& [score, t_idx] : candidate_tasks) {
            log_stream << "T" << t_idx + 1 << "(" << score << ") ";
        }
        log_stream << "}\n\n";



        // Enumerate the pool of available resources/workers onve for all this time step
        std::vector<std::tuple<int, int>> workers_pool{};

        // Also create a map for finding the workers indexes by their (machine, operator) pair
        std::map<std::tuple<int, int>, int> worker2poolindex_map{};

        build_workers_pool(
            inst,
            candidate_tasks,
            available_machines,
            available_operators,
            workers_pool,
            worker2poolindex_map
        );
        const int nb_workers = static_cast<int>(workers_pool.size());

        if (nb_workers == 0) {
            log_stream << "No workers available for the candidate tasks. Continuing.\n";
            time_pos++;
            continue;
        }

#if DEBUG
        /* BUILDING A DECISION PROCESS USING LINEAR PROGRAMMING */
        log_stream << "Selecting workers from the pool with a linear programming approach...\n";
        // ====== Create the GLOP solver ======
        std::unique_ptr<operations_research::MPSolver> solver(operations_research::MPSolver::CreateSolver("GLOP"));
        if (!solver) {
            log_stream << "Solver not created. Exiting...\n";
            continue;
        }

        // ====== Create decision variables ======
        std::vector<operations_research::MPVariable*> workers_vars{};
        for (int i = 0; i < nb_workers; i++) {

            workers_vars.emplace_back(
                solver->MakeIntVar(
                    0.0,
                    1.0,
                    "W" + std::to_string(i)
                )
            );
        }

        // ====== Create constraints ======
        std::vector<operations_research::MPConstraint*> workers_cstr{};
        for (int i = 0; i < nb_workers; i++) {
            for (int j = i + 1; j < nb_workers; j++) {

                int m_idx1 = std::get<0>(workers_pool[i]);
                int o_idx1 = std::get<1>(workers_pool[i]);
                int m_idx2 = std::get<0>(workers_pool[j]);
                int o_idx2 = std::get<1>(workers_pool[j]);

                if (m_idx1 == m_idx2 || o_idx1 == o_idx2) {
                    // Create one constraint for the pair, they are incompatible
                    workers_cstr.emplace_back(
                        solver->MakeRowConstraint(
                            0.0,
                            1.0
                        )
                    );

                    // No more than one of both conflicting workers can be used at the same time

                    workers_cstr.back()->SetCoefficient(workers_vars[i], 1.0);
                    workers_cstr.back()->SetCoefficient(workers_vars[j], 1.0);
                }
            }
        }


        // ====== Create the objective function ======
        operations_research::MPObjective* const objective = solver->MutableObjective();
        for (int i = 0; i < nb_candidate_tasks; i++) {
            int t_idx = std::get<1>(candidate_tasks[i]); // pas bon
            float score = std::get<0>(candidate_tasks[i]);

            for (auto& m_idx : inst.tasks[t_idx].machines) {
                for (auto& o_idx : inst.tasks[t_idx].compatibility[m_idx]) {

                    int w_idx = worker2poolindex_map[std::make_tuple(m_idx, o_idx)];
                    double coef = objective->GetCoefficient(workers_vars[w_idx]);
                    coef += static_cast<double>(score);
                    objective->SetCoefficient(workers_vars[i], coef);
                }
            }
        }

        objective->SetMaximization();


        log_stream << "Linear Program has " << solver->NumVariables() << " variables & " << solver->NumConstraints() << " constraints.\n";

        const operations_research::MPSolver::ResultStatus result_status = solver->Solve();

        // Check that the problem has an optimal solution.
        if (result_status != operations_research::MPSolver::FEASIBLE && result_status != operations_research::MPSolver::OPTIMAL) {
            log_stream << "The problem does not have a feasible or optimal solution!\n";
            return;
        }

        log_stream << "Problem solved in " << solver->wall_time() << " milliseconds & " << solver->iterations() << " iterations\n";
        log_stream << "Total objective: " << objective->Value() << '\n';

        int nb_selected_workers = 0;
        for (int i = 0; i < nb_workers; i++) {
            operations_research::MPVariable* w_var_ptr = workers_vars[i];
            if (w_var_ptr->solution_value() > 0.5) {
                nb_selected_workers++;
            }
        }

        log_stream << "\nSolving selected " << nb_selected_workers << "/" << nb_workers << " compatible workers (" << static_cast<float>(nb_selected_workers) / static_cast<float>(nb_workers) * 100.0f << "%):\n { ";

        for (int i = 0; i < nb_workers; i++) {
            operations_research::MPVariable* w_var_ptr = workers_vars[i];

            if (w_var_ptr->solution_value() > 0.5) {
                int w_m_idx = std::get<0>(workers_pool[i]);
                int w_o_idx = std::get<1>(workers_pool[i]);
                log_stream << "M" << w_m_idx + 1 << "_O" << w_o_idx + 1 << " ";
            }
        }
        log_stream << "}\n";
#endif


        /* BUILDING A DECISION PROCESS USING LINEAR ALGEBRA */
        log_stream << "Selecting workers from the pool with a weighted pruning approach...\n";

        // The iterative linear-algebra loop finds a good conflict-free subset of feasible worker pairs, and then a greedy pass uses those
        // workers to schedule as many high-priority tasks as possible at the current time step.

        // Create a conflict matrix between all workers this round
        arma::SpMat<float> W_conflicts;
        build_workers_conflict_matrix(workers_pool, W_conflicts);

        // Create a compatibility matrix between all tasks and workers
        arma::Mat<float> T_W_compat;

        // Also create a map for finding the tasks indexes by their (task, worker) pair
        std::map<int, int> task2poolindex_map{};


        build_task_worker_compatibility_matrix(
            inst,
            candidate_tasks,
            worker2poolindex_map,
            T_W_compat,
            task2poolindex_map
        );

        // Display number of entries in the compatibility matrix
        log_stream << "Compatibility matrix has " << arma::accu(T_W_compat) << " nonzero entries.\n";


        // Create a task scores vector
        arma::Col<float> T_scores = build_task_scores_vector(candidate_tasks);

        // Define the indicator vector of active workers
        arma::Col<float> active_workers = arma::Col<float>(nb_workers, arma::fill::ones); // all workers are in use by default until conflicts are resolved


        int multiplicity = compute_and_log_conflict_graph_connectivity(
            W_conflicts,
            nb_workers,
            log_stream
        );
        (void)multiplicity;

        prune_conflicting_workers(
            W_conflicts,
            T_W_compat,
            T_scores,
            active_workers,
            log_stream
        );

        // We have now eliminated all conflicts, all remaining workers are compatible
        // We can now assign them to tasks

        int nb_active_workers = static_cast<int>(arma::sum(active_workers));
        // log_stream << "Pruning has eliminated " << nb_workers - nb_active_workers << " workers." << std::endl;
        log_stream << "\nPruning selected " << nb_active_workers << "/" << nb_workers << " compatible workers (" << static_cast<float>(nb_active_workers) / static_cast<float>(nb_workers) * 100.0f << "%):\n { ";

        for (int i = 0; i < nb_workers; i++) {
            if (active_workers(i) > 0.5f) {
                int w_m_idx = std::get<0>(workers_pool[i]);
                int w_o_idx = std::get<1>(workers_pool[i]);
                log_stream << "M" << w_m_idx + 1 << "_O" << w_o_idx + 1 << " ";
            }
        }
        log_stream << "}\n";



        /* ASSIGNING WORKERS TO TASKS WITH THE DECIDED WORKERS */
        arma::Col<float> T_mult; // Assignment multiplicity of the tasks

        // Update the compatibility matrix with the remaining active workers
        T_W_compat.each_row() % active_workers.t();

        // Update the task multiplicities
        T_mult = T_W_compat * active_workers;

        /*         for (int i = 0; i < nb_workers; i++) {
                    int w_m_idx = std::get<0>(workers_pool[i]);
                    int w_o_idx = std::get<1>(workers_pool[i]);
                    log_stream << "M" << w_m_idx + 1 << "_O" << w_o_idx + 1 << "    ";
                }
                log_stream << std::endl << active_workers.t() << std::endl; */


                // Compute the remaing active workers' versatilities: counting the number of tasks each one of them can address
        std::vector<int> W_versatility_indexes_ = workers_sorted_by_versatility(T_W_compat, active_workers);

        greedy_assign_tasks_to_workers(
            inst,
            sol,
            job_stacks,
            candidate_tasks,
            workers_pool,
            task2poolindex_map,
            W_versatility_indexes_,
            T_W_compat,
            active_workers,
            time_pos,
            available_machines,
            available_operators,
            next_time_persue_job,
            release_calendar_machines,
            release_calendar_operators,
            cumulative_remaining_time_per_job,
            log_stream
        );
        time_pos++;
    }
    log_stream << "\n================== *** End of procedure *** ==================\n";
}
