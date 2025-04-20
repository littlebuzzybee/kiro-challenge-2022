
        // ====== Create the initial node for the BFS: ======
        // For each task, we will explore the possible assignments of machines and operators.
        // Some tasks may be addressable with several pairs (M, O) and we explore them all so that we may be able to minimize the overhead added tardiness score this round
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
        initial_node.workers = arma::Col<float>(nb_workers, arma::fill::ones); // all workers are in use by default until confilcts are resolved

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
        if (nb_cfl <= 0) {
            // the node is a leaf with no conflicts left
            // we must compute an assignment that maximizes the number of addressed tasks or overall score (mminimize the tardiness score)

            // compare it to the current best solution and update it if necessary
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








        void node_analysis(const ExplorationNode& node, const Instance& inst) {
            // int nb_machines = static_cast<int>(node.available_machines.count());
            // int nb_operators = static_cast<int>(node.available_operators.count());
            int nb_tasks = static_cast<int>(node.assigned_tasks.size());
        
        
            // Laplacian of overlapping resources
            arma::fmat tasks_ma_laplacian(nb_tasks, nb_tasks, arma::fill::zeros);
            arma::fmat tasks_op_laplacian(nb_tasks, nb_tasks, arma::fill::zeros);
        
            // Number of overlapping resources
            arma::Mat<float> tasks_ma_overlap = arma::zeros<arma::Mat<float>>(nb_tasks, nb_tasks);
            arma::Mat<float> tasks_op_overlap = arma::zeros<arma::Mat<float>>(nb_tasks, nb_tasks);
        
            // Assemble adjacency and laplacian matrices
            
            for (int i = 0; i < nb_tasks; i++) {
                int t1_idx = node.assigned_tasks[i];
        
                int adjacency_degree_ma = 0;
                int adjacency_degree_op = 0;
        
                for (int j = i + 1; j < nb_tasks; j++) {
                    int t2_idx = node.assigned_tasks[j];
        
                    std::vector<int> common_machines{};
                    std::set_intersection(
                        inst.tasks[t1_idx].machines.begin(),
                        inst.tasks[t1_idx].machines.end(),
                        inst.tasks[t2_idx].machines.begin(),
                        inst.tasks[t2_idx].machines.end(),
                        std::back_inserter(common_machines)
                    );
        
                    std::vector<int> common_operators{};
                    std::set_intersection(
                        inst.tasks[t1_idx].machines.begin(),
                        inst.tasks[t1_idx].machines.end(),
                        inst.tasks[t2_idx].machines.begin(),
                        inst.tasks[t2_idx].machines.end(),
                        std::back_inserter(common_operators)
                    );
        
                    // Upper triangular part
                    tasks_ma_overlap(i, j) = static_cast<float>(common_machines.size());
                    tasks_op_overlap(i, j) = static_cast<float>(common_operators.size());
                    tasks_ma_laplacian(i, j) = common_machines.size() > 0 ? -1.0f : 0.0f;
                    tasks_op_laplacian(i, j) = common_operators.size() > 0 ? -1.0f : 0.0f;
                    // Lower triangular part
                    tasks_ma_overlap(j, i) = tasks_ma_overlap(i, j);
                    tasks_op_overlap(j, i) = tasks_ma_overlap(i, j);
                    tasks_ma_laplacian(j, i) = tasks_ma_laplacian(i, j);
                    tasks_op_laplacian(j, i) = tasks_op_laplacian(i, j);
        
                    adjacency_degree_ma += common_machines.size() > 0 ? -1 : 0;
                    adjacency_degree_op += common_machines.size() > 0 ? -1 : 0;
                }
        
                // Diagonal part
                tasks_ma_overlap(i, i) = static_cast<float>(adjacency_degree_ma);
                tasks_op_overlap(i, i) = static_cast<float>(adjacency_degree_op);
            }
            
        
            arma::fmat U;
            arma::fvec s;
            arma::fmat V;
        
            arma::svd(U, s, V, tasks_ma_laplacian);
            // detect multiplicity of 0 as the number of zero eigenvalues
            int multiplicity = 0;
            for (int i = 0; i < static_cast<int>(s.n_elem); i++) {
                if (s(i) < 1e-5) { multiplicity++; }
            }
        
        
        
        }