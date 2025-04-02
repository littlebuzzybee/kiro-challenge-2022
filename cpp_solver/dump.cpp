        // Create data structures to map the shared resources (values) between two given tasks (keys)
        std::map<std::tuple<int, int>, std::unordered_set<int>> shared_machines; // map (T1, T2) --> set of machines shared between T1 and T2
        std::map<std::tuple<int, int>, std::unordered_set<int>> shared_operators; // map (T1, T2) --> set of operators shared between T1 and T2

        for (int t1_idx = 0; t1_idx < inst.nb_tasks; t1_idx++) {
            for (int t2_idx = t1_idx++; t2_idx < inst.nb_tasks; t2_idx++) {

                std::tuple<int, int> identifier = std::make_tuple(t1_idx, t2_idx);

                shared_machines[identifier] = std::unordered_set<int>();
                std::set_intersection(
                    inst.tasks[t1_idx].machines.begin(), inst.tasks[t1_idx].machines.end(),
                    inst.tasks[t2_idx].machines.begin(), inst.tasks[t2_idx].machines.end(),
                    std::inserter(
                        shared_machines[identifier],
                        shared_machines[identifier].begin()
                    )
                );

                shared_operators[identifier] = std::unordered_set<int>();
                std::set_intersection(
                    inst.tasks[t1_idx].operators.begin(), inst.tasks[t1_idx].operators.end(),
                    inst.tasks[t2_idx].operators.begin(), inst.tasks[t2_idx].operators.end(),
                    std::inserter(
                        shared_operators[identifier],
                        shared_operators[identifier].begin()
                    )
                );
            }
        }










        std::map<int, operations_research::MPVariable*> machines_usage_vars{}; // task -> machine usage variable
        std::map<int, operations_research::MPConstraint*> machines_usage_constr{}; // task -> machine usage constraint
        for (int m_idx : available_machines) {
            machines_usage_vars[m_idx] = solver->MakeIntVar(
                0.0,
                infinity,
                "U_M" + std::to_string(m_idx) 
            ); // Integer variable for the number of times a machine is used
            
            machines_usage_constr[m_idx] = solver->MakeRowConstraint(
                0.0,
                1.0
            ); // Exclude using a machine several times

            machines_usage_constr[m_idx]->SetCoefficient(machines_usage_vars[m_idx], 1.0);
        }

        std::map<int, operations_research::MPVariable*> operators_usage_vars{}; // task -> operator usage variable
        std::map<int, operations_research::MPConstraint*> operators_usage_constr{}; // task -> operator usage constraint
        for (int o_idx : available_operators) {
            operators_usage_vars[o_idx] = solver->MakeIntVar(
                0.0, 
                infinity,
                "U_O" + std::to_string(o_idx)
            ); // Integer variable for the number of times an operator is used

            operators_usage_constr[o_idx] = solver->MakeRowConstraint(
                0.0,
                1.0
            ); // Exclude using an operator several times
            
            operators_usage_constr[o_idx]->SetCoefficient(operators_usage_vars[o_idx], 1.0);
        }












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