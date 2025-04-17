#include "solve_traversal.h"


void resolve_traversal(
    Instance& inst,
    Solution& sol,
    std::map<int, std::deque<int>>& job_stacks,
    std::ostream& log_stream
) {
    log_stream << "Beginning solving procedure with a tree traversal search heuristic..." << std::endl;

    // Display the job_stacks
    print_job_stacks(job_stacks, log_stream);


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
        job_stacks,
        next_time_persue_job
    );

    int time_pos{ 0 };

    int fail_safe{ 0 };
    while (!all_stacks_are_empty(job_stacks) && fail_safe < 100) {
        fail_safe++;
        log_stream << std::endl << "================== *** Time " << time_pos << " *** ==================" << std::endl;

        // First release the resources that are no longer used
        release_idle_resources(
            time_pos,
            available_machines,
            available_operators,
            release_calendar_machines[time_pos],
            release_calendar_operators[time_pos]
        );

        // Display the released resources
        log_stream << "Released resources:" << std::endl;
        log_stream << "M";
        print_set(release_calendar_machines[time_pos], 1, log_stream);
        log_stream << std::endl << "O";
        print_set(release_calendar_operators[time_pos], 1, log_stream);
        log_stream << std::endl;

        // Display the available resources
        log_stream << "Available resources:" << std::endl;
        log_stream << "M";
        print_set(available_machines, 1, log_stream);
        log_stream << std::endl << "O";
        print_set(available_operators, 1, log_stream);
        log_stream << std::endl;

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
            log_stream << "No candidate tasks to process at this time position. Continuing." << std::endl;
            time_pos++;
            continue;
        }

        // Sort the candidate tasks by highest priority to get the most urgent tasks first (those with the highest tardiness score so far)
        std::sort(candidate_tasks.begin(), candidate_tasks.end(), std::greater<std::tuple<int, int>>());

        // Display the candidate tasks
        log_stream << "Candidate tasks for processing (priority score): { ";
        for (auto& [score, t_idx] : candidate_tasks) {
            log_stream << "T" << t_idx + 1 << "(" << score << ") ";
        }
        log_stream << " }" << std::endl;

        // Compute the characteristics of the available workers pool this round
        int nb_candidate_tasks = static_cast<int>(candidate_tasks.size());


        /* BUILDING A DECISION PROCESS USING LINEAR ALGEBRA */

        // Enumerate the pool of available resources/workers onve for all this time step
        std::vector<std::tuple<int, int>> workers_pool{};

        // Also create a map for finding the workers indexes by their (machine, operator) pair
        std::map<std::tuple<int, int>, int> worker2poolindex_map{};

        for (auto& [_, t_idx] : candidate_tasks) {
            for (auto& m_idx : inst.tasks[t_idx].machines) {
                // Check if the machine is available
                if (!available_machines.contains(m_idx)) { continue; }
                for (auto& o_idx : inst.tasks[t_idx].compatibility[m_idx]) {
                    // Check if the operator is available
                    if (!available_operators.contains(o_idx)) { continue; }
                    workers_pool.emplace_back(m_idx, o_idx);
                    worker2poolindex_map[std::make_tuple(m_idx, o_idx)] = static_cast<int>(workers_pool.size() - 1);
                }
            }
        }
        int nb_workers = static_cast<int>(workers_pool.size());

        if (nb_workers == 0) {
            log_stream << "No workers available for the candidate tasks. Continuing." << std::endl;
            time_pos++;
            continue;
        }


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
            for (auto& m_idx : inst.tasks[t_idx].machines) {
                for (auto& o_idx : inst.tasks[t_idx].compatibility[m_idx]) {
                    // Check if the worker is available
                    auto worker = std::make_tuple(m_idx, o_idx);
                    if (!worker2poolindex_map.contains(worker)) { continue; }
                    int w_idx = worker2poolindex_map[worker];
                    T_W_compat(i, w_idx) = 1.0f;
                }
            }
        }


        // Create a task scores vector
        arma::Col<float> T_scores = arma::zeros<arma::Col<float>>(nb_candidate_tasks);
        for (int i = 0; i < nb_candidate_tasks; i++) {
            T_scores(i) = std::get<0>(candidate_tasks[i]);
        }

        // TODO : add a vector of workers in a Task's representation to change the nested for loops every time
        // TODO : Change the (task_idx, task_score) representation to use the armadillo's constructors from vectors directly



        arma::Col<float> active_workers = arma::Col<float>(nb_workers, arma::fill::ones); // all workers are in use by default until conflicts are resolved
        log_stream << "There are " << nb_workers << " workers available." << std::endl;

        // Computing the number or conflicts
        int nb_conflicts = static_cast<int>((double).5 * arma::dot(active_workers, W_conflicts * active_workers));

        arma::Col<float> T_mult; // Assignment multiplicity of the tasks
        arma::Col<float> W_conflict_scores; // Conflict scores of the workers
        arma::Col<float> W_deletion_impact_scores; // Deletion impact scores of the workers
        arma::Col<float> W_compound_scores; // Compound scores of the workers
        arma::Col<arma::uword> W_compound_scores_indexes; // Sorted indexes of the compound scores

        log_stream << "There are " << nb_conflicts << " conflicts to resolve among them." << std::endl;
        while (nb_conflicts > 0) {
            T_mult = T_W_compat * active_workers;


            // Compute the conflict score of each worker as a gradient of the conflict function 1/2 x.T G x
            // Each entry of that gradient approximates a 'contribution' value of the corresponding worker activation to the overall number of conflicts
            W_conflict_scores = W_conflicts * active_workers;

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
            W_compound_scores = W_deletion_impact_scores / (W_conflict_scores + .1f);

            // Sort workers by their compound score and get the ones having the least deletion impact / conflicting tendency score
            W_compound_scores_indexes = arma::sort_index(W_compound_scores, "ascend");

            // Get the iterator to the worker index that has the highest compound score
            arma::uword* w_it = W_compound_scores_indexes.begin();

            // Begin eliminating problematic workers until we have removed enough of them
            // We adopt a logarithmic approach to the number of conflicts to be eliminated:
            // If there are more than 10, we will delete half of them before updating the gradients,
            // otherwise we will delete one worker at a time and update the rankings every time
            float planned_deletions = nb_conflicts > 1000 ? 100.0f : (nb_conflicts > 100 ? 10.0f : 1.0f);
            int nb_eliminated = 0;

            while (nb_eliminated < planned_deletions && w_it != W_compound_scores_indexes.end()) {
                // elimite half of the currently ordered conflicts before updating the actual number of conflicts
                int w_idx = W_compound_scores_indexes(*w_it);
                // Deactivate the worker
                if (active_workers(w_idx) < 0.5f) {
                    w_it++;
                    continue;
                }
                active_workers(w_idx) = 0.0f; // Deactivate the worker
                nb_eliminated++;
                w_it++;
            }
            // Updating the number or conflicts
            nb_conflicts = static_cast<int>(.5f * arma::dot(active_workers, W_conflicts * active_workers));
        }

        // We have now eliminated all conflicts, all remaining workers are compatible
        // We can now assign them to tasks

        log_stream << "Presolve has eliminated " << nb_workers - static_cast<int>(arma::sum(active_workers)) << " workers." << std::endl;
        log_stream << "There are " << arma::sum(active_workers) << " selected potential workers remaining: { ";
        for (int i = 0; i < nb_workers; i++) {
            if (active_workers(i) > 0.5f) {
                int w_m_idx = std::get<0>(workers_pool[i]);
                int w_o_idx = std::get<1>(workers_pool[i]);
                log_stream << "M" << w_m_idx + 1 << "_O" << w_o_idx + 1 << " ";
            }
        }
        log_stream << " }" << std::endl;
        log_stream << "There are " << static_cast<int>(arma::sum(active_workers)) << " workers and " << candidate_tasks.size() << " candidate tasks remaning." << std::endl;


        // Recompute the task multiplicities
        T_mult = T_W_compat * active_workers;

        // Compute the remaing active workers' versatilities: counting the number of tasks each one of them can address
        arma::Col<float> W_versatility = arma::sum(T_W_compat.each_row() % active_workers.t(), 0).t();
        
        // Sort the workers by their versatility and get the ones having the least versatility
        arma::Col<arma::uword> W_versatility_indexes = arma::sort_index(W_versatility, "ascend");
        std::vector<float> W_versatility_indexes_ = arma::conv_to<std::vector<float>>::from(W_versatility_indexes);



        log_stream << "Assigned tasks are:" << std::endl;
        // Now we assign the tasks to the workers
        for (auto& [_, t_idx] : candidate_tasks) {
            // they are already ordered so the greatest score is first
            int t_rank = task2poolindex_map[t_idx]; // index of the task in the candidate tasks vector
            
            // Get the first worker that has the least versatility and that is active and compatible for this task
            int w_idx_ptr = 0;
            while (w_idx_ptr < nb_workers && T_W_compat(t_rank, W_versatility_indexes(w_idx_ptr)) < 0.5f) {
                w_idx_ptr++;
            }
            // If we have no more workers available, we stop
            if (w_idx_ptr >= nb_workers) {
                continue;
            }
            // Get the worker's index
            int w_idx = W_versatility_indexes(w_idx_ptr);
            // Choose that worker for task t_idx and reset the compatibility column of that worker since it is not available anymore
            T_W_compat.col(w_idx).zeros();

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
            int j_idx = inst.tasks[t_idx].job_parent;
            next_time_persue_job[j_idx] = time_pos + inst.tasks[t_idx].processing_time;

            // Update the release calendar for the chosen machine and operator
            release_calendar_machines[time_pos + inst.tasks[t_idx].processing_time].insert(chosen_machine);
            release_calendar_operators[time_pos + inst.tasks[t_idx].processing_time].insert(chosen_operator);
            cumulative_remaining_time_per_job[j_idx] -= inst.tasks[t_idx].processing_time;

            // Remove the task from the stack
            job_stacks[j_idx].pop_front();
            log_stream << " - Task T" << t_idx + 1 << " (J" << j_idx + 1 << ") assigned to M" << chosen_machine + 1 << " & O" << chosen_operator + 1 << std::endl;
            
        }
        time_pos++;
    }
    log_stream << std::endl << "End of solving procedure." << std::endl << std::endl;
}
