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














