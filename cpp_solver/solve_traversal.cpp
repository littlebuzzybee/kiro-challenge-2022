#include "solve_traversal.h"


void resolve_traversal(
    Instance& inst,
    Solution& sol,
    std::map<int, std::deque<int>>& job_stacks,
    std::ostream& log_stream
) {
    log_stream << "Beginning solving procedure with a decision tree based search heuristic..." << std::endl;

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
        log_stream << "Released workers:    ";
        for (int m_idx : release_calendar_machines[time_pos]) {
            available_machines.insert(m_idx);
            log_stream << "M" << m_idx + 1 << " ";
        }
        for (int o_idx : release_calendar_operators[time_pos]) {
            available_operators.insert(o_idx);
            log_stream << "O" << o_idx + 1 << " ";
        }
        log_stream << std::endl;

        // Display the available resources
        log_stream << "Available machines:  ";
        for (int m_idx : available_machines) {
            log_stream << "M" << m_idx + 1 << " ";
        }
        log_stream << std::endl;
        log_stream << "Available operators: ";
        for (int o_idx : available_operators) {
            log_stream << "O" << o_idx + 1 << " ";
        }
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


        // ====== Create the initial node for the BFS: ======
        // For each task, we will explore the possible assignments of machines and operators. Some tasks may be addressable with several pairs (M, O) and we explore them all so that we may be able to minimize the overhead added tardiness score this round
        ExplorationNode initial_node;

        // Copy the available resources as dynamic bitsets to use in the nodes of the tree traversal 
        initial_node.available_machines = boost::dynamic_bitset<>(inst.nb_machines);
        initial_node.available_operators = boost::dynamic_bitset<>(inst.nb_operators);
        initial_node.available_machines.reset();
        initial_node.available_operators.reset();
        for (int m_idx : available_machines) {
            initial_node.available_machines.set(m_idx, true);
        }
        for (int o_idx : available_operators) {
            initial_node.available_operators.set(o_idx, true);
        }

        // We start with the first task in the stack whatever happens
        initial_node.next_task_vec_idx = 0;
        initial_node.nb_addressed_tasks = 0;
        initial_node.overhead_tardiness_score = 0;
        initial_node.assigned_tasks = std::vector<int>();
        initial_node.chosen_machines = std::vector<int>();
        initial_node.chosen_operators = std::vector<int>();

        // Create the first node of the exploration tree
        std::deque<ExplorationNode> node_queue{};
        node_queue.emplace_back(initial_node);


        int current_best_overhead_score = std::numeric_limits<int>::max();

        ExplorationNode best_node{}; // copy of the current best node in the exploration tree


        // ====== Perform BFS exploration of the tree ======

        while (!node_queue.empty()) {
            ExplorationNode node = node_queue.front();
            node_queue.pop_front();

            // LEAF (TERMINAL CASE)
            if (node.next_task_vec_idx >= static_cast<int>(candidate_tasks.size())) {
                // the node is a leaf, we must compare it to the current best solution and update it if necessary
                if (node.overhead_tardiness_score < current_best_overhead_score) {
                    current_best_overhead_score = node.overhead_tardiness_score;
                    best_node = node;
                }
                else if (node.overhead_tardiness_score == current_best_overhead_score && node.nb_addressed_tasks > best_node.nb_addressed_tasks) {
                    best_node = node;
                }
                continue;
            }

            // NODE (NON-TERMINAL CASE)
            std::tuple<float, int>& processed_task = candidate_tasks[node.next_task_vec_idx];
            int t_idx = std::get<1>(processed_task);
            if (t_idx == 14 || t_idx == 17) {
                int debug = 1;
            }

            bool task_can_be_assigned = false;
            // Look for all pairs of operators and machines that can address the task
            for (auto& m_idx : inst.tasks[t_idx].machines) {
                if (!node.available_operators[m_idx]) {
                    continue;
                }

                for (auto& o_idx : inst.tasks[t_idx].compatibility[m_idx]) {
                    if (!node.available_operators[o_idx]) {
                        continue;
                    }
                    task_can_be_assigned = true;
                    // We can assign the task to the operator and machine
                    // copy the parent node, and update its fields afterwards
                    ExplorationNode child_node = node;
                    child_node.available_operators.set(o_idx, false);
                    child_node.available_machines.set(m_idx, false);
                    child_node.assigned_tasks.emplace_back(t_idx);
                    child_node.chosen_operators.emplace_back(o_idx);
                    child_node.chosen_machines.emplace_back(m_idx);
                    child_node.nb_addressed_tasks++;
                    child_node.next_task_vec_idx++;
                    // Emplace the child node in the queue for further exploration
                    node_queue.emplace_back(child_node);
                    // log_stream << "T" << t_idx + 1 << "[M" << m_idx + 1 << "+O" << o_idx + 1 << "][p" << node.next_task_vec_idx << "]::";
                }
            }
            // log_stream << std::endl;

            if (!task_can_be_assigned) {
                // The task cannot be assigned to any machine or operator, we skip it and create a branch for that scenario
                ExplorationNode child_node = node;
                int parent_job = inst.tasks[t_idx].job_parent;
                // recompute the tardiness contribution score for the job of that skipped task with postponed processing by 1 time unit
                child_node.overhead_tardiness_score += inst.jobs[parent_job].weight * std::max(0, 1 + time_pos + cumulative_remaining_time_per_job[parent_job] - inst.jobs[parent_job].due_date);
                child_node.next_task_vec_idx++;
                node_queue.emplace_back(child_node);
            }
        }


        // We have explored all the nodes of the tree, we can now assign the best node to the solution;
        // that is the one that was found to minimize the overhead tardiness score
        log_stream << "Best node assigns " << best_node.nb_addressed_tasks << " task(s): { ";
        for (int idx = 0; idx < static_cast<int>(best_node.assigned_tasks.size()); ++idx) {
            log_stream << "T" << best_node.assigned_tasks[idx] + 1 << " ";
        }
        log_stream << "} with overhead score " << best_node.overhead_tardiness_score << std::endl;

        for (int idx = 0; idx < static_cast<int>(best_node.assigned_tasks.size()); ++idx) {
            int t_idx = best_node.assigned_tasks[idx];
            int j_idx = inst.tasks[t_idx].job_parent;

            // Assign the workers to the task
            int chosen_machine = best_node.chosen_machines[idx];
            int chosen_operator = best_node.chosen_operators[idx];
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
            log_stream << " - Task T" << t_idx + 1 << " (J" << j_idx + 1 << ") assigned to M" << chosen_machine + 1 << " & O" << chosen_operator + 1 << std::endl;
        }
        time_pos++;
    }
    log_stream << std::endl << "End of solving procedure." << std::endl << std::endl;
}
