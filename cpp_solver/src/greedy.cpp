#include "greedy.h"


void resolve_greedy(
    Instance& inst,
    Solution& sol,
    std::map<int, std::deque<int>>& job_stacks,
    std::ostream& log_stream
) {

    log_stream << "Beginning solving procedure with an elementary greedy iterative heuristic...\n";

    int time_cursor{ 0 };
    // Print the job stacks
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

    // Create a data structure to enforce the precedence by remembering when to pursue the next task of a job
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

    int time_pos{ time_cursor };


    while (!all_stacks_are_empty(job_stacks)) {
        log_stream << "\n================== *** Time " << time_pos << " *** ==================\n";


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


        std::priority_queue<std::tuple<int, int>> task_queue{};
        // first integer is the score, second integer is the job index
        // score is a function of the tardiness of the job and its weight
        // to act as a proxy of the to-be objective terms

        // First, we insert the tasks that are ready to be processed in a sorted priority queue to be ranked and compared
        // there is at most only one task per job in the stack and we decide which to adress in order of priority

        for (auto& [j_idx, task_stack] : job_stacks) {
            // Insert the first task in line for the job if it exists and if the processing time of its predecessor is over
            if (task_stack.empty() || time_pos < next_time_persue_job[j_idx]) {
                continue;
            }
            int t_idx = task_stack.front();
            int tardiness = std::max(0, time_pos + cumulative_remaining_time_per_job[j_idx] - inst.jobs[j_idx].due_date);
            int score = inst.jobs[j_idx].weight * tardiness;
            std::tuple<int, int> task_entry = std::make_tuple(score, t_idx);
            task_queue.emplace(task_entry);
        }




        while (!task_queue.empty() && !available_machines.empty() && !available_operators.empty()) {
            // Heuristic: iterate through tasks in order of priority at that time step and
            // assign tasks and resources one at a time

            auto [tardiness, t_idx] = task_queue.top();
            task_queue.pop();
            int j_idx = inst.tasks[t_idx].job_parent;

            // Compute intersection of available machines and authorized machines for that task
            // POSSIBLE = AVAILABLE INTERSECT AUTHORIZED
            std::vector<int> possible_machines;
            std::set<int>& authorized_machines = inst.tasks[t_idx].machines;
            std::set_intersection(
                available_machines.begin(), available_machines.end(),
                authorized_machines.begin(), authorized_machines.end(),
                std::back_inserter(possible_machines)
            );

            // If no machine is available, we skip the task
            if (possible_machines.empty()) {
                continue;
            }

            // Otherwise, we look for a qualified operator on one of the available machines for that task
            // Analytics showed that tasks usually have far more overlapping common machines than operators, so we assign operators first in the hope that machines can click in easily thereafter
            int chosen_machine{ -1 };
            int chosen_operator{ -1 };
            std::vector<int> possible_operators;
            std::set<int>& authorized_operators = inst.tasks[t_idx].operators;

            for (int m_idx : possible_machines) {
                possible_operators.clear();
                std::set_intersection(
                    available_operators.begin(), available_operators.end(),
                    authorized_operators.begin(), authorized_operators.end(),
                    std::back_inserter(possible_operators)
                );

                if (!possible_operators.empty()) {
                    // Greedily assign the first compatible pair to the task
                    chosen_operator = possible_operators.front();
                    chosen_machine = m_idx;
                    break;
                }
                else {
                    continue;
                }
            }

            // If no machine or operator is available, we skip the task
            if (chosen_machine == -1 || chosen_operator == -1) {
                continue;
            }


            // Assign the workers to the task
            sol.machine_choice_tasks[t_idx] = chosen_machine;
            sol.operator_choice_tasks[t_idx] = chosen_operator;

            // Schedule the task
            sol.begin_time_tasks[t_idx] = time_pos;

            // Prevent the subsequent assignment of any other task of the same job before the end of the current task
            next_time_persue_job[j_idx] = time_pos + inst.tasks[t_idx].processing_time;
            available_machines.erase(chosen_machine);
            available_operators.erase(chosen_operator);

            // Update the release calendar for the chosen machine and operator
            release_calendar_machines[time_pos + inst.tasks[t_idx].processing_time].insert(chosen_machine);
            release_calendar_operators[time_pos + inst.tasks[t_idx].processing_time].insert(chosen_operator);
            cumulative_remaining_time_per_job[j_idx] -= inst.tasks[t_idx].processing_time;

            // Remove the task from the stack
            job_stacks[j_idx].pop_front();
            log_stream << " - Task T" << t_idx + 1 << " (J" << j_idx + 1 << ") assigned to M" << chosen_machine + 1 << " & O" << chosen_operator + 1 << " at time " << time_pos << '\n';
        }
        time_pos++;
    }
    log_stream << "\nEnd of solving procedure.\n\n";
}
