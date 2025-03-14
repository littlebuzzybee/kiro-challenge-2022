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