#include "traversal_breakdown.h"

#include <cassert>

void build_workers_pool(
    Instance& inst,
    const std::vector<std::tuple<float, int>>& candidate_tasks,
    const std::set<int>& available_machines,
    const std::set<int>& available_operators,
    std::vector<std::tuple<int, int>>& workers_pool,
    std::map<std::tuple<int, int>, int>& worker2poolindex_map
) {
    // Enumerate the pool of available resources/workers onve for all this time step
    // Also create a map for finding the workers indexes by their (machine, operator) pair

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
}

void build_workers_conflict_matrix(
    const std::vector<std::tuple<int, int>>& workers_pool,
    arma::SpMat<float>& W_conflicts
) {
    const int nb_workers = static_cast<int>(workers_pool.size());

    // Create a conflict matrix between all workers this round
    W_conflicts = arma::zeros<arma::SpMat<float>>(nb_workers, nb_workers);

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
}

void build_task_worker_compatibility_matrix(
    Instance& inst,
    const std::vector<std::tuple<float, int>>& candidate_tasks,
    std::map<std::tuple<int, int>, int>& worker2poolindex_map,
    arma::Mat<float>& T_W_compat,
    std::map<int, int>& task2poolindex_map
) {
    const int nb_candidate_tasks = static_cast<int>(candidate_tasks.size());
    const int nb_workers = static_cast<int>(worker2poolindex_map.size());

    // Create a compatibility matrix between all tasks and workers
    T_W_compat = arma::zeros<arma::Mat<float>>(nb_candidate_tasks, nb_workers);

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
}

arma::Col<float> build_task_scores_vector(
    const std::vector<std::tuple<float, int>>& candidate_tasks
) {
    const int nb_candidate_tasks = static_cast<int>(candidate_tasks.size());

    // Create a task scores vector
    arma::Col<float> T_scores = arma::zeros<arma::Col<float>>(nb_candidate_tasks);
    for (int i = 0; i < nb_candidate_tasks; i++) {
        T_scores(i) = std::get<0>(candidate_tasks[i]);
    }
    return T_scores;
}

int count_worker_conflicts(
    const arma::SpMat<float>& W_conflicts,
    const arma::Col<float>& active_workers
) {
    // Computing the number or conflicts
    return static_cast<int>(.5f * arma::dot(active_workers, W_conflicts * active_workers));
}

int compute_and_log_conflict_graph_connectivity(
    const arma::SpMat<float>& W_conflicts,
    int nb_workers,
    std::ostream& log_stream
) {
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
    return multiplicity;
}

int prune_conflicting_workers(
    const arma::SpMat<float>& W_conflicts,
    arma::Mat<float>& T_W_compat,
    const arma::Col<float>& T_scores,
    arma::Col<float>& active_workers,
    std::ostream& log_stream
) {
    int nb_conflicts = count_worker_conflicts(W_conflicts, active_workers);

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
        nb_conflicts = count_worker_conflicts(W_conflicts, active_workers);
        nb_elim_rounds++;
    }
    log_stream << "\rConflicts remaining: " << nb_conflicts << std::flush;
    log_stream << " Done. (" << nb_elim_rounds << " rounds)\n";

    return nb_elim_rounds;
}

std::vector<int> workers_sorted_by_versatility(
    const arma::Mat<float>& T_W_compat,
    const arma::Col<float>& active_workers
) {
    // Compute the remaing active workers' versatilities: counting the number of tasks each one of them can address
    arma::Row<float> W_versatility = arma::sum(T_W_compat.each_row() % active_workers.t(), 0);
    assert(W_versatility.size() == active_workers.n_elem);

    // Sort the workers by their versatility and get the ones having the least versatility
    // We choose the least versatile workers first because we will address the tasks by order of ascending scores
    // Therefore, it is better to give the "most important" tasks the first compatible, least versatile worker first so that some
    // other workers can be freed up for the less important tasks and maximize the number of addressed tasks overall
    arma::Col<arma::uword> W_versatility_indexes = arma::sort_index(W_versatility, "ascend");

    return arma::conv_to<std::vector<int>>::from(W_versatility_indexes);
}

int greedy_assign_tasks_to_workers(
    Instance& inst,
    Solution& sol,
    std::map<int, std::deque<int>>& job_stacks,
    const std::vector<std::tuple<float, int>>& candidate_tasks,
    const std::vector<std::tuple<int, int>>& workers_pool,
    const std::map<int, int>& task2poolindex_map,
    const std::vector<int>& worker_choice_order,
    arma::Mat<float>& T_W_compat,
    arma::Col<float>& active_workers,
    int time_pos,
    std::set<int>& available_machines,
    std::set<int>& available_operators,
    std::map<int, int>& next_time_persue_job,
    std::map<int, std::set<int>>& release_calendar_machines,
    std::map<int, std::set<int>>& release_calendar_operators,
    std::map<int, int>& cumulative_remaining_time_per_job,
    std::ostream& log_stream
) {
    const int nb_candidate_tasks = static_cast<int>(candidate_tasks.size());
    const int nb_workers = static_cast<int>(workers_pool.size());

    int nb_assigned_tasks = 0;
    log_stream << "\nAssigned tasks are:\n";
    // Now we assign the tasks to the workers
    for (auto& [_, t_idx] : candidate_tasks) {
        // they are already ordered so the greatest score is first
        int t_rank = task2poolindex_map.at(t_idx); // index of the task in the candidate tasks vector

        // Get the first worker that has the least versatility and that is active and compatible for this task
        int w_idx_ptr = 0;
        while (
            w_idx_ptr < nb_workers
            && (T_W_compat(t_rank, worker_choice_order[w_idx_ptr]) < 0.5f
                || active_workers(worker_choice_order[w_idx_ptr]) < 0.5f)
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
        int w_idx = worker_choice_order[w_idx_ptr];
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
    return nb_assigned_tasks;
}
