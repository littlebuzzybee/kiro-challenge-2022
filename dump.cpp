                        if (parent_job1 != parent_job2) {
                            // If the tasks belong to the same job, no need to set exclusion constraints since time scheduling constraints already prevent overlapping
                            // If not, we set the exclusion constraints for the tasks' respective operators and machines


                            //  ******************************************************************************************************
                            // ************ Declare common variables for both machine and operator overlapping exclusions ************
                            //  ******************************************************************************************************
                            GRBVar& begin_time_task1 = begin_times_tasks_per_job[parent_job1][task_idx1];
                            GRBVar& begin_time_task2 = begin_times_tasks_per_job[parent_job2][task_idx2];


                            GRBVar diff_ti_aft = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
                            GRBVar diff_ti_bef = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);

                            model.addGenConstrIndicator(diff_ti_aft, 1, begin_time_task1 + inst.tasks[task_idx1].processing_time <= begin_time_task2);
                            model.addGenConstrIndicator(diff_ti_bef, 1, begin_time_task2 + inst.tasks[task_idx2].processing_time <= begin_time_task1);
                            //  diff_ti_aft activates => task2 starts after task1 ends
                            //  diff_ti_bef activates => task1 starts after task2 ends

                            //  ******************************************************
                            //  ************ Prevent overlapping operators ***********
                            //  ******************************************************
                            GRBVar& operator_task1 = assigned_tasks_operators_per_job[parent_job1][task_idx1];
                            GRBVar& operator_task2 = assigned_tasks_operators_per_job[parent_job2][task_idx2];

                            GRBVar diff_op_lte = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
                            GRBVar diff_op_gte = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);


                            model.addGenConstrIndicator(diff_op_lte, 1, operator_task1 <= operator_task2 - 1);
                            model.addGenConstrIndicator(diff_op_gte, 1, operator_task1 >= operator_task2 + 1);
                            // diff_op_lte activates => operator_task1 < operator_task2
                            // diff_op_gte activates => operator_task1 > operator_task2


                            // The true disjunction of the below 4 literals is equivalent to
                            // operators of the two tasks do not overlap
                            const GRBVar exclusion_same_operator[4] = {
                                diff_op_lte,
                                diff_op_gte,
                                diff_ti_aft,
                                diff_ti_bef
                            };

                            // Create the overlapping variable for operators
                            tasks_overlapping_operators[{task_idx1, task_idx2}] = model.addVar(
                                0.0,
                                1.0,
                                0.0,
                                GRB_BINARY,
                                "operators_overlap_for_tasks_" + std::to_string(task_idx1) + "_" + std::to_string(task_idx2)
                            );


                            // Force the overlapping variable to be equal to the disjunction of the above literals and store it in the map
                            model.addGenConstrOr(
                                tasks_overlapping_operators[{task_idx1, task_idx2}],
                                exclusion_same_operator,
                                4, // number of literals in the above array of boolean variables
                                "activates_if_same_operator_" + std::to_string(task_idx1) + "_" + std::to_string(task_idx2)
                            );

                            // Finally constrain it to be equal to 0 (false) in the model
                            model.addConstr(
                                tasks_overlapping_operators[{task_idx1, task_idx2}] == 0,
                                "exclusion_same_operator_" + std::to_string(task_idx1) + "_" + std::to_string(task_idx2)
                            );


                            //  ******************************************************
                            //  ************ Prevent overlapping machines ************
                            //  ******************************************************
                            GRBVar& machine_task1 = assigned_tasks_machines_per_job[parent_job1][task_idx1];
                            GRBVar& machine_task2 = assigned_tasks_machines_per_job[parent_job2][task_idx2];

                            GRBVar diff_ma_lte = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
                            GRBVar diff_ma_gte = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);

                            model.addGenConstrIndicator(diff_ma_lte, 1, machine_task1 <= machine_task2 - 1);
                            model.addGenConstrIndicator(diff_ma_gte, 1, machine_task1 >= machine_task2 + 1);
                            // diff_ma_lte activates => machine_task1 < machine_task2
                            // diff_ma_gte activates => machine_task1 > machine_task2

                            // The true disjunction of the below 4 literals is equivalent to
                            // machines of the two tasks do not overlap
                            const GRBVar exclusion_same_machine[4] = {
                                diff_ma_lte,
                                diff_ma_gte,
                                diff_ti_aft,
                                diff_ti_bef
                            };!

                            // Create the overlapping variable for machines
                            tasks_overlapping_machines[{task_idx1, task_idx2}] = model.addVar(
                                0.0,
                                1.0,
                                0.0,
                                GRB_BINARY,
                                "machines_overlap_for_tasks_" + std::to_string(task_idx1) + "_" + std::to_string(task_idx2)
                            );


                            // Force the overlapping variable to be equal to the disjunction of the above literals and store it in the map
                            model.addGenConstrOr(
                                tasks_overlapping_machines[{task_idx1, task_idx2}],
                                exclusion_same_machine,
                                4, // number of literals in the above array of boolean variables
                                "activates_if_same_operator_" + std::to_string(task_idx1) + "_" + std::to_string(task_idx2)
                            );

                            // Finally constrain it to be equal to 0 (false) in the model
                            model.addConstr(
                                tasks_overlapping_machines[{task_idx1, task_idx2}] == 0,
                                "exclusion_same_operator_" + std::to_string(task_idx1) + "_" + std::to_string(task_idx2)
                            );


                        }






    for (int t_idx_id1 = 0; t_idx_id1 < nb_processed_tasks; t_idx_id1++) {
        for (int t_idx_id2 = t_idx_id1 + 1; t_idx_id2 < nb_processed_tasks; t_idx_id2++) {


            int task_idx1 = processed_tasks[t_idx_id1];
            int task_idx2 = processed_tasks[t_idx_id2];

            int parent_job1 = inst.tasks[task_idx1].job_parent;
            int parent_job2 = inst.tasks[task_idx2].job_parent;

            if (parent_job1 != parent_job2) {
                GRBVar& bt1 = begin_times_tasks_per_job[parent_job1][task_idx1];
                GRBVar& bt2 = begin_times_tasks_per_job[parent_job2][task_idx2];

                int& d1 = inst.tasks[task_idx1].processing_time;
                int& d2 = inst.tasks[task_idx2].processing_time;

                GRBVar& o1 = assigned_tasks_operators_per_job[parent_job1][task_idx1];
                GRBVar& o2 = assigned_tasks_operators_per_job[parent_job2][task_idx2];
                GRBVar& m1 = assigned_tasks_machines_per_job[parent_job1][task_idx1];
                GRBVar& m2 = assigned_tasks_machines_per_job[parent_job2][task_idx2];

                model.addQConstr(-(bt1 + d1 - bt2) * (bt2 + d2 - bt1) + (o1 - o2) * (o1 - o2) + (m1 - m2) * (m1 - m2) >= 0, "exclusion_constraints_tasks_" + std::to_string(task_idx1) + "_" + std::to_string(task_idx2));
            }

        }
    }






                    // Construct the variable encoding operator overlap between the two tasks
                GRBVar e1 = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "operator_task_" + std::to_string(task_idx1) + "_is_greater_thanthatof_task_" + std::to_string(task_idx2));
                GRBVar e2 = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "operator_task_" + std::to_string(task_idx1) + "_is_smaller_thanthatof_task_" + std::to_string(task_idx2));
                
                model.addConstr(o1 >= o2 + eps - big_M_operator * (1 - e1));
                model.addConstr(o1 <= o2 + big_M_operator * e1);

                model.addConstr(o2 >= o1 + eps - big_M_operator * (1 - e2));
                model.addConstr(o2 <= o1 + big_M_operator * e2);

                GRBLinExpr operator_not_overlap = e1 + e2; // in {0, 1}, 0 IF overlap; 1 IF NO overlap



                // Construct the variable encoding machine overlap between the two tasks
                GRBVar f1 = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "machine_task_" + std::to_string(task_idx1) + "_is_greater_thanthatof_task_" + std::to_string(task_idx2));
                GRBVar f2 = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "machine_task_" + std::to_string(task_idx1) + "_is_smaller_thanthatof_task_" + std::to_string(task_idx2));
                
                model.addConstr(m1 >= m2 + eps - big_M_machine * (1 - f1));
                model.addConstr(m1 <= m2 + big_M_machine * f1);

                model.addConstr(m2 >= m1 + eps - big_M_machine * (1 - f2));
                model.addConstr(m2 <= m1 + big_M_machine * f2);

                GRBLinExpr machine_not_overlap = f1 + f2; // in {0, 1}, 0 IF overlap; 1 IF NO overlap

                // Construct the variable encoding the overlap of the above three variables
                model.addConstr(time_not_overlap + machine_not_overlap >= 1);
                model.addConstr(time_not_overlap + operator_not_overlap >= 1);