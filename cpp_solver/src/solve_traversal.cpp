#include "solve_traversal.h"

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

        // Detect all workers
        for (auto& [_, t_idx] : candidate_tasks) {
            for (auto& m_idx : inst.tasks[t_idx].machines) {
                // Check if the machine is available
                if (!available_machines.contains(m_idx)) { continue; }
                for (auto& o_idx : inst.tasks[t_idx].compatibility[m_idx]) {
                    // Check if the operator is available
                    if (!available_operators.contains(o_idx)) { continue; }

                    // Check if the worker was not already added
                    if (!worker2poolindex_map.contains(std::make_tuple(m_idx, o_idx))) {
                        workers_pool.emplace_back(std::make_tuple(m_idx, o_idx));
                        worker2poolindex_map[std::make_tuple(m_idx, o_idx)] = static_cast<int>(workers_pool.size() - 1);
                    }
                }
            }
        }
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
        arma::SpMat<float> W_conflicts = arma::zeros<arma::SpMat<float>>(nb_workers, nb_workers);


        for (int i = 0; i < nb_workers; i++) {
            for (int j = i + 1; j < nb_workers; j++) {
                int m1_idx = std::get<0>(workers_pool[i]);
                int o1_idx = std::get<1>(workers_pool[i]);
                int m2_idx = std::get<0>(workers_pool[j]);
                int o2_idx = std::get<1>(workers_pool[j]);

                if (m1_idx == m2_idx || o1_idx == o2_idx) {
                    // The two workers are not compatible, we set the conflict in the matrix
                    W_conflicts(i, j) = 1.0f;
                    W_conflicts(j, i) = 1.0f;
                }
            }
        }

        // Create a compatibility matrix between all tasks and workers
        arma::Mat<float> T_W_compat = arma::zeros<arma::Mat<float>>(nb_candidate_tasks, nb_workers);

        // Also create a map for finding the tasks indexes by their (task, worker) pair
        std::map<int, int> task2poolindex_map{};


        for (int i = 0; i < nb_candidate_tasks; i++) {
            int t_idx = std::get<1>(candidate_tasks[i]);
            task2poolindex_map[t_idx] = i;
            // Only iterate over the task's known workers instead of going through all those the workers pool
            for (auto& m_idx : inst.tasks[t_idx].machines) {
                for (auto& o_idx : inst.tasks[t_idx].compatibility[m_idx]) {
                    // Check if the worker is available
                    auto worker = std::make_tuple(m_idx, o_idx);

                    int w_idx = worker2poolindex_map[worker];
                    T_W_compat(i, w_idx) = 1.0f;
                }
            }
        }

        // Display number of entries in the compatibility matrix
        log_stream << "Compatibility matrix has " << arma::accu(T_W_compat) << " nonzero entries.\n";


        // Create a task scores vector
        arma::Col<float> T_scores = arma::zeros<arma::Col<float>>(nb_candidate_tasks);
        for (int i = 0; i < nb_candidate_tasks; i++) {
            T_scores(i) = std::get<0>(candidate_tasks[i]);
        }

        // Define the indicator vector of active workers
        arma::Col<float> active_workers = arma::Col<float>(nb_workers, arma::fill::ones); // all workers are in use by default until conflicts are resolved


        // Computing the number or conflicts
        int nb_conflicts = static_cast<int>(.5f * arma::dot(active_workers, W_conflicts * active_workers));

        // Compute the laplacian of the conflict matrix
        arma::SpMat<double> W_conflicts_laplacian = -1.0 * arma::conv_to<arma::SpMat<double>>::from(W_conflicts);
        W_conflicts_laplacian.diag() = -1.0 * arma::sum(W_conflicts_laplacian, 1);

        arma::vec eigval;
        arma::mat eigvec;
        arma::eigs_opts opts;
        opts.maxiter = 1000;
        opts.tol = 1e-5;
        int nb_eigenvalues = W_conflicts_laplacian.n_rows > 5 ? 5 : W_conflicts_laplacian.n_rows - 1;
        log_stream << "\nComputing " << nb_eigenvalues << " eigenvalues of the laplacian matrix...";
        arma::eigs_sym(eigval, eigvec, W_conflicts_laplacian, nb_eigenvalues, "sa");

        int multiplicity = 0;
        for (int i = 0; i < static_cast<int>(eigval.n_elem); i++) {
            if (eigval(i) < 1e-5) { multiplicity++; }
        }


        log_stream << " Done.\n";
        log_stream << "There are " << nb_workers << " workers available with given resources in >= " << multiplicity << " connected components.\n";


        // Defining some variables to store for the elimination heuristics loop
        arma::Col<float> T_mult; // Assignment multiplicity of the tasks
        arma::Col<float> W_conflict_scores; // Conflict scores of the workers
        arma::Col<float> W_deletion_impact_scores; // Deletion impact scores of the workers
        arma::Col<float> W_compound_scores; // Compound scores of the workers
        arma::Col<arma::uword> W_compound_scores_indexes; // Sorted indexes of the compound scores

        log_stream << "Resolving " << nb_conflicts << " conflicts among them...\n";
        int nb_elim_rounds = 0;
        while (nb_conflicts > 0) {
            // Compute the multiplicity of the tasks: i.e. the number of workers capable of addressing them
            T_mult = T_W_compat * active_workers;

            log_stream << "\rConflicts remaining: " << nb_conflicts << ".. " << std::flush;

            // Compute the conflict score of each worker as a gradient of the conflict function 1/2 x.T G x
            // Each entry of that gradient approximates a 'contribution' value of the corresponding worker activation to the overall number of conflicts
            W_conflict_scores = (W_conflicts * active_workers) % active_workers; // Project the gradient onto the vector



            /*
            arma::Col<float> T_multipliers = T_scores / (T_mult + .1f);
            arma::Mat<float> T_redundancy_fractions = T_W_compat.each_col() % T_multipliers;
            arma::Col<float> W_deletion_impact_scores = arma::sum(T_redundancy_fractions, 0).t();
            */

            // Compute the delete impact score of each worker
            W_deletion_impact_scores = arma::sum(T_W_compat.each_col() % (T_scores / (T_mult + .1f)), 0).t();

            /*
            Now we have in our possession two metrics for each worker:
                - a conflict score [conflict_scores]
                - a redundancy score [W_deletion_impact_scores]
            We aim at eliminating conflicts as fast as possible while retaining the most redundancy for all tasks

            We can explore deactivating workers based on this data to quickly prune the search space
            We can now explore the tree by deactivating the workers with the highest conflict score and the lowest redundancy score
            */


            // Compute the compound score
            W_compound_scores = W_conflict_scores / (W_deletion_impact_scores + .1f);

            // Sort workers by their compound score and get the ones having the most [conflicting tendency / deletion impact] score
            W_compound_scores_indexes = arma::sort_index(W_compound_scores, "descend");
            // std::cout << arma::sort(W_compound_scores, "descend") << std::endl;



            // Begin eliminating problematic workers until we have removed enough of them
            // We adopt a logarithmic approach to the number of conflicts to be eliminated:
            // If there are more than 10, we will delete half of them before updating the gradients,
            // otherwise we will delete one worker at a time and update the rankings every time
            int planned_deletions = nb_conflicts > 100000 ? 10000 : (nb_conflicts > 10000 ? 5000 : (nb_conflicts > 1000 ? 100 : (nb_conflicts > 100 ? 10 : 1)));
            float nb_eliminated = 0.0f;

            // Get the iterator to the worker index that has the highest compound score
            for (auto w_r : W_compound_scores_indexes) {
                // Deactivate the worker
                if (nb_eliminated >= planned_deletions || nb_eliminated >= nb_conflicts || W_compound_scores(w_r) == .0f) {
                    break;
                }
                // Compute the number of conflicts eliminated as the dot product of the row of the conflict matrix and the active workers
                nb_eliminated += arma::dot(W_conflicts.row(w_r).t(), active_workers);
                active_workers(w_r) = 0.0f; // Deactivate the worker
            }
            // Updating the number or conflicts
            nb_conflicts = static_cast<int>(.5f * arma::dot(active_workers, W_conflicts * active_workers));
            nb_elim_rounds++;
        }
        log_stream << "\rConflicts remaining: " << nb_conflicts << std::flush;
        log_stream << " Done. (" << nb_elim_rounds << " rounds)\n";

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
        arma::Row<float> W_versatility = arma::sum(T_W_compat.each_row() % active_workers.t(), 0);
        assert(W_versatility.size() == nb_workers);

        // Sort the workers by their versatility and get the ones having the least versatility
        // We choose the least versatile workers first because we will address the tasks by order of ascending scores
        // Therefore, it is better to give the "most important" tasks the first compatible, least versatile worker first so that some
        // other workers can be freed up for the less important tasks and maximize the number of addressed tasks overall
        arma::Col<arma::uword> W_versatility_indexes = arma::sort_index(W_versatility, "ascend");

        std::vector<int> W_versatility_indexes_ = arma::conv_to<std::vector<int>>::from(W_versatility_indexes);



        int nb_assigned_tasks = 0;
        log_stream << "\nAssigned tasks are:\n";
        // Now we assign the tasks to the workers
        for (auto& [_, t_idx] : candidate_tasks) {
            // they are already ordered so the greatest score is first
            int t_rank = task2poolindex_map[t_idx]; // index of the task in the candidate tasks vector

            // Get the first worker that has the least versatility and that is active and compatible for this task
            int w_idx_ptr = 0;
            while (
                w_idx_ptr < nb_workers
                && (T_W_compat(t_rank, W_versatility_indexes_[w_idx_ptr]) < 0.5f
                    || active_workers(W_versatility_indexes_[w_idx_ptr]) < 0.5f)
                ) {
                w_idx_ptr++;
            }

            int j_idx = inst.tasks[t_idx].job_parent;

            // If we have no more workers available, we stop
            if (w_idx_ptr >= nb_workers) {
                log_stream << " - Task T" << t_idx + 1 << " (J" << j_idx + 1 << ") postponed.\n";
                continue;
            }
            nb_assigned_tasks++;
            // Get the worker's index
            int w_idx = W_versatility_indexes_[w_idx_ptr];
            // Choose that worker for task t_idx and reset the compatibility column of that worker since it is not available anymore
            T_W_compat.col(w_idx).zeros();
            active_workers(w_idx) = 0.0f; // Deactivate the worker

            // Now officially address the task
            int chosen_machine = std::get<0>(workers_pool[w_idx]);
            int chosen_operator = std::get<1>(workers_pool[w_idx]);
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
        log_stream << "\nAssigned " << nb_assigned_tasks << " tasks of " << nb_candidate_tasks << " this round (" << static_cast<float>(nb_assigned_tasks) / static_cast<float>(nb_candidate_tasks) * 100.0f << "%).\n";
        time_pos++;
    }
    log_stream << "\n================== *** End of procedure *** ==================\n";
}
