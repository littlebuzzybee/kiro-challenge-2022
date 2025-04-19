#include "utils.h"

nlohmann::json read_json_file(std::string filename) {
    std::ifstream f(filename);
    nlohmann::json data = nlohmann::json::parse(f);
    f.close();
    return data;
}

void print_set(std::set<int> s, int offset, std::ostream& out_stream) {
    out_stream << "{";
    for (auto& e : s) {
        out_stream << e + offset;
        if (e != *s.rbegin()) {
            out_stream << ",";
        }
    }
    out_stream << "}";
}


void print_sequence(std::vector<int> v, int offset, std::ostream& out_stream) {
    out_stream << "[";
    for (auto& t : v) {
        out_stream << t + offset;
        if (t != v.back()) {
            out_stream << "->";
        }
    }
    out_stream << "]";
}


void display_cstr_matrix(std::map<int, std::map<int, int>>& matrix, std::ostream& out_stream) {
    // Print column headers
    out_stream << "   ";
    for (const auto& col : matrix.begin()->second) {
        out_stream << std::setw(4) << col.first + 1;
    }
    out_stream << std::endl;

    // Print rows
    for (const auto& row : matrix) {
        out_stream << std::setw(4) << row.first + 1 << ": ";
        for (const auto& cell : row.second) {
            if (cell.second == 3) {
                out_stream << "\u00d7   ";
            }
            else if (cell.second == 2) {
                out_stream << "o   ";
            }
            else if (cell.second == 1) {
                out_stream << "m   ";
            }
            else {
                out_stream << "    "; // Two spaces for proper alignment
            }
        }
        out_stream << std::endl; // New line after each row
    }
}


void displayMatrix(const std::map<int, std::map<int, int>>& matrix, std::ostream& out_stream) {
    if (matrix.empty()) {
        out_stream << "Matrix is empty.\n";
        return;
    }

    // Determine the set of all column headers (keys of the innermost maps)
    std::set<int> columns;
    for (const auto& [outer_key, inner_map] : matrix) {
        for (const auto& [inner_key, _] : inner_map) {
            columns.insert(inner_key);
        }
    }

    // Print the header row
    out_stream << std::setw(4) << " "; // Leave space for row headers
    for (int col : columns) {
        out_stream << std::setw(4) << col;
    }
    out_stream << std::endl;

    // Print each row
    for (const auto& [row_key, inner_map] : matrix) {
        out_stream << std::setw(4) << row_key; // Row header
        for (int col : columns) {
            auto it = inner_map.find(col);
            if (it != inner_map.end()) {
                out_stream << std::setw(4) << it->second; // Value in the cell
            }
            else {
                out_stream << std::setw(4) << 0; // Default to 0 if no value exists
            }
        }
        out_stream << std::endl;
    }
}

Instance import_instance(std::filesystem::path filename, std::ostream& out_stream) {
    nlohmann::json inst_descriptor = read_json_file(filename.string());

    Instance inst_object;

    inst_object.nb_jobs = inst_descriptor["parameters"]["size"]["nb_jobs"];
    inst_object.nb_tasks = inst_descriptor["parameters"]["size"]["nb_tasks"];
    inst_object.nb_machines = inst_descriptor["parameters"]["size"]["nb_machines"];
    inst_object.nb_operators = inst_descriptor["parameters"]["size"]["nb_operators"];

    inst_object.unit_penalty = inst_descriptor["parameters"]["costs"]["unit_penalty"];
    inst_object.tardiness = inst_descriptor["parameters"]["costs"]["tardiness"];
    inst_object.interim = inst_descriptor["parameters"]["costs"]["interim"];


    // Iterate over tasks, import and print details

    for (auto& task_descriptor : inst_descriptor["tasks"]) {
        // int task_id = task_descriptor["task"];
        // int task_idx = task_id - 1; // ids are offset by 1 in the JSON file

        Task task_object;
        task_object.id = task_descriptor["task"];
        task_object.processing_time = task_descriptor["processing_time"];
        task_object.machines = std::set<int>();
        task_object.operators = std::set<int>();

        for (auto& machine_descr : task_descriptor["machines"]) {
            int machine_id = static_cast<int>(machine_descr["machine"]);
            int machine_idx = machine_id - 1; // ids are offset by 1 in the JSON file

            std::set<int> partial_ops = {};
            for (auto& operator_id : machine_descr["operators"]) {
                int operator_idx = static_cast<int>(operator_id) - 1; // ids are offset by 1 in the JSON file
                partial_ops.emplace(operator_idx);    // add the operator to the set of possible operators for that task
                task_object.operators.emplace(operator_idx); // add the operator to the set of all possible operators for the task
                task_object.compatible_workers.emplace(std::make_tuple(machine_idx, operator_idx)); // add the machine/operator pair to the set of compatible workers
            }
            task_object.compatibility.emplace(machine_idx, partial_ops);
            task_object.machines.emplace(machine_idx); // add the machine to the set of possible machines
        }

        // Commit log task details
        out_stream << "=== Imported task T" << task_object.id << " ===" << std::endl;
        out_stream << "* processing time: " << task_object.processing_time << std::endl;
        out_stream << "* possible supports: " << std::endl;

        for (auto& m : task_object.machines) {
            out_stream << "  - M" << m + 1 << " with O";
            print_set(task_object.compatibility[m], 1, out_stream);
            out_stream << std::endl;
        }
        out_stream << "* Combined Machines: ";
        print_set(task_object.machines, 1, out_stream);
        out_stream << std::endl;
        out_stream << "* Combined Operators: ";
        print_set(task_object.operators, 1, out_stream);
        out_stream << std::endl << std::endl;

        // Commit task object to instance
        inst_object.tasks.push_back(task_object);
    }

    out_stream << std::endl;
    // Iterate over jobs, import and print details

    for (auto& job_descriptor : inst_descriptor["jobs"]) {
        int job_id = job_descriptor["job"];
        int job_idx = job_id - 1; // ids are offset by 1 in the JSON file

        Job job_object;
        job_object.id = job_id;
        job_object.release_date = job_descriptor["release_date"];
        job_object.due_date = job_descriptor["due_date"];
        job_object.weight = job_descriptor["weight"];
        job_object.sequence = std::vector<int>();

        for (int task_id : job_descriptor["sequence"]) {
            int task_idx = task_id - 1; // ids are offset by 1 in the JSON file
            job_object.sequence.push_back(task_idx); // ids are offset by 1 in the JSON file
            inst_object.tasks[task_idx].job_parent = job_idx;
        }

        // Commit log job details
        out_stream << "=== Imported job J" << job_object.id << " ===" << std::endl;
        out_stream << "* release date: " << job_object.release_date << std::endl;
        out_stream << "* due date: " << job_object.due_date << std::endl;
        out_stream << "* weight: " << job_object.weight << std::endl;
        out_stream << "* task sequence: ";
        print_sequence(job_object.sequence, 1, out_stream);
        out_stream << std::endl << std::endl;

        // Add job object to instance
        inst_object.jobs.push_back(job_object);
    }



    return inst_object;
}


bool all_stacks_are_empty(const std::map<int, std::deque<int>>& stacks) {
    for (const auto& pair : stacks) {
        if (!pair.second.empty()) {
            return false;
        }
    }
    return true;
}



void get_sort_tasks_and_scores(
    std::vector<std::tuple<float, int>>& candidate_tasks,
    Instance& inst,
    std::map<int, std::deque<int>>& job_stacks,
    std::map<int, int>& cumulative_remaining_time_per_job,
    std::map<int, int>& next_time_persue_job,
    int time_pos
) {
    for (auto& [j_idx, task_stack] : job_stacks) {
        // Insert the first task in line for the job if it exists and if the processing time of its predecessor is over
        if (task_stack.empty() || time_pos < next_time_persue_job[j_idx]) {
            continue;
        }
        int t_idx = task_stack.front();
        int overhead = time_pos + cumulative_remaining_time_per_job[j_idx] - inst.jobs[j_idx].due_date;
        int tardiness = std::max(0, overhead);
        float score = inst.jobs[j_idx].weight * tardiness + 1.0; // Add 1 to avoid zero scores for linear programming; square root tends to even relative score discrepancies slightly
        // the higher the score, the higher the priority to avoid accumulation of tardiness
        candidate_tasks.emplace_back(std::make_tuple(score, t_idx));
    }

    // Sort the candidate tasks by highest priority to get the most urgent tasks first (those with the highest tardiness score so far)
    std::sort(candidate_tasks.begin(), candidate_tasks.end(), std::greater<std::tuple<float, int>>());
}


void get_cumulative_remaining_time_per_job(
    std::map<int, int>& cumulative_remaining_time_per_job,
    Instance& inst,
    std::map<int, std::deque<int>>& job_stacks,
    std::map<int, int>& next_time_persue_job
) {
    for (auto& [j_idx, j_queue] : job_stacks) {
        int total_processing_time = std::reduce(
            j_queue.begin(),
            j_queue.end(),
            0, // Initial value of the sum
            [&inst](int total_sum, int t_idx) {
                return total_sum + inst.tasks[t_idx].processing_time;
            }
        );
        cumulative_remaining_time_per_job[j_idx] = total_processing_time;
    }
}


void fill_job_stacks_and_compute_time(
    Instance& inst,
    std::map<int, std::deque<int>>& job_stacks,
    std::vector<int>& total_time_per_job
) {
    for (int j_idx = 0; j_idx < inst.nb_jobs; j_idx++) {

        std::vector<int>& task_sequence = inst.jobs[j_idx].sequence;

        job_stacks[j_idx] = std::deque<int>(task_sequence.begin(), task_sequence.end());
        // Front of the deque has the lowest task indexes, i.e. the first to process in order

        int total_processing_time = std::accumulate(
            task_sequence.begin(),
            task_sequence.end(),
            0, // Initial value of the sum
            [&inst](int total_sum, int t_idx) {
                return total_sum + inst.tasks[t_idx].processing_time;
            }
        );

        total_time_per_job[j_idx] = total_processing_time;
    }
}

void release_idle_resources(
    int time_pos,
    std::set<int>& available_machines,
    std::set<int>& available_operators,
    const std::set<int>& machines_to_release,
    const std::set<int>& operators_to_release
) {
    for (int m_idx : machines_to_release) {
        available_machines.insert(m_idx);
    }
    for (int o_idx : operators_to_release) {
        available_operators.insert(o_idx);
    }
}

int compute_loss(const Instance& inst, const Solution& sol) {
    int loss = 0;

    for (int j_idx = 0; j_idx < inst.nb_jobs; j_idx++) {
        int last_task = inst.jobs[j_idx].sequence.back();
        int wj = inst.jobs[j_idx].weight;
        int Cj = sol.begin_time_tasks[last_task] + inst.tasks[last_task].processing_time;
        int dj = inst.jobs[j_idx].due_date;
        int Tj = std::max(Cj - dj, 0);
        int Uj = Tj > 0 ? 1 : 0;
        loss += wj * (Cj + inst.tardiness * Tj + inst.unit_penalty * Uj);
    }
    return loss;
}

void print_job_stacks(const std::map<int, std::deque<int>>& job_stacks, std::ostream& log_stream) {
    // Print the job stacks
    log_stream << std::endl << "Job stacks current state:" << std::endl;
    for (auto& [j_idx, task_stack] : job_stacks) {
        log_stream << "J" << j_idx + 1 << ": |";
        for (int t_idx : task_stack) {
            log_stream << "T" << t_idx + 1 << "|";
        }
        log_stream << std::endl;
    }
}

bool check_validity(const Instance& inst, const Solution& sol) {

    // Check that all tasks are processed and that begin times are valid
    int max_time = 0;
    for (int t_idx = 0; t_idx < inst.nb_tasks; t_idx++) {
        if (sol.begin_time_tasks[t_idx] < 0) {
            return false;
        }

        int end_time = sol.begin_time_tasks[t_idx] + inst.tasks[t_idx].processing_time;
        if (end_time > max_time) {
            max_time = end_time;
        }
    }

    // Check that tasks are processed in order
    for (int j_idx = 0; j_idx < inst.nb_jobs; j_idx++) {
        auto& sequence = inst.jobs[j_idx].sequence;
        for (int k = 0; k < static_cast<int>(sequence.size()); k++) {
            int t_idx = sequence[k];
            int lower_bound = k == 0 ? inst.jobs[j_idx].release_date : sol.begin_time_tasks[sequence[k - 1]] + inst.tasks[sequence[k - 1]].processing_time;
            if (lower_bound > sol.begin_time_tasks[t_idx]) {
                std::cout << "Task " << t_idx + 1 << " of job " << j_idx + 1 << " is processed before the end of the previous task." << std::endl;
                std::cout << "Previous task end time: " << lower_bound << ", task begin time: " << sol.begin_time_tasks[t_idx] << std::endl;
                return false;
            }
        }
    }

    std::vector<bool> processed_tasks = std::vector<bool>(inst.nb_tasks, false);
    std::set<int> busy_machines = std::set<int>();
    std::set<int> busy_operators = std::set<int>();

    std::map<int, int> task_of_machine = std::map<int, int>();
    std::map<int, int> task_of_operator = std::map<int, int>();
    // Check that resources are not overlaping
    for (int t = 0; t < max_time; t++) {
        for (int t_idx = 0; t_idx < inst.nb_tasks; t_idx++) {
            if (sol.begin_time_tasks[t_idx] <= t && t < sol.begin_time_tasks[t_idx] + inst.tasks[t_idx].processing_time) {

                int m_idx = sol.machine_choice_tasks[t_idx];
                int o_idx = sol.operator_choice_tasks[t_idx];


                if (busy_machines.contains(m_idx)) {
                    std::cout << "M" << m_idx + 1 << " is busy at time " << t << " from T" << task_of_machine[m_idx] + 1 << " but T" << t_idx + 1 << " uses it again." << std::endl;
                    return false;
                }
                if (busy_operators.contains(o_idx)) {
                    std::cout << "O" << o_idx + 1 << " is busy at time " << t << " from T" << task_of_operator[o_idx] + 1 << " but T" << t_idx + 1 << " uses it again." << std::endl;
                    return false;
                }

                task_of_machine[m_idx] = t_idx;
                task_of_operator[o_idx] = t_idx;

                busy_machines.insert(m_idx);
                busy_operators.insert(o_idx);
                processed_tasks[t_idx] = true;
            }
        }
        busy_machines.clear();
        busy_operators.clear();
    }
    return true;
}


void node_analysis(const ExplorationNode& node, const Instance& inst, const Solution& sol) {
    int nb_machines = static_cast<int>(node.available_machines.count());
    int nb_operators = static_cast<int>(node.available_operators.count());
    int nb_tasks = static_cast<int>(node.assigned_tasks.size());

    // Detect whether or not the problem is overconstrained (i.e. if there are more tasks than machines or operators)
    int ma_deficit = std::max(0, nb_tasks - nb_machines);
    int op_deficit = std::max(0, nb_tasks - nb_operators);


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
    for (int i = 0; i < s.n_elem; i++) {
        if (s(i) < 1e-5) { multiplicity++; }
    }



}