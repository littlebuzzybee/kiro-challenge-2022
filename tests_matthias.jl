using JSON;
using LinearAlgebra;
using DataStructures;

instance = JSON.parsefile("C:/Users/matth/Documents/GitHub/kiro-challenge-2022/instances/tiny.json")

parameters = instance["parameters"]
tasks      = instance["tasks"]
jobs       = instance["jobs"]

param_size       = parameters["size"]
parameters_costs = parameters["costs"]

α = parameters_costs["unit_penalty"];
β = parameters_costs["tardiness"];

nb_machines, nb_tasks, nb_jobs, nb_operators = param_size["nb_machines"], param_size["nb_tasks"], param_size["nb_jobs"], param_size["nb_operators"]

interim, unit_penalty, tardiness = parameters_costs["interim"], parameters_costs["unit_penalty"], parameters_costs["tardiness"]

duration_task = zeros(Int32, nb_tasks)

compat_machine_operator_per_task = zeros(Bool, (nb_tasks, nb_machines, nb_operators))

for i=1:nb_tasks
    task = tasks[i];
    duration_task[i] = task["processing_time"];
    possible_machines = task["machines"];
    for machine in possible_machines
        machine_index = machine["machine"];
        possible_operators = Vector{Int64}(machine["operators"]);
        compat_machine_operator_per_task[Int64(i), machine_index, possible_operators] .= true;
    end
end



todo_tasks        = Set{Int64}(1:nb_tasks);
todo_nb_tasks     = zeros(Int64, nb_jobs);
running_tasks     = Set{Int64}();
done_tasks        = Set{Int64}();
busy_machines     = Set{Int64}();
busy_operators    = Set{Int64}();
idle_operators    = Set{Int64}(1:nb_operators);


jobs_task_sequences = Dict{Int64, Queue{Int}}();
jobs_weights        = zeros(Int64, nb_jobs)
jobs_release_date   = zeros(Int64, nb_jobs)
jobs_due_date       = zeros(Int64, nb_jobs)

job_of_task         = zeros(Int64,   nb_tasks)
score_of_task       = zeros(Float64, nb_tasks)
start_time_of_task  = zeros(Int64, nb_tasks)

operator_choice_of_task = zeros(Int64, nb_tasks)
machine_choice_of_task  = zeros(Int64, nb_tasks)
curr_busy_mach_op = rand(Bool, nb_machines, nb_operators); # remplacer par des zéros à la fin



for j=1:nb_jobs # 
    jobs_task_sequences[j] = Queue{Int64}();
    for task in Vector{Int64}(jobs[j]["sequence"])
        enqueue!(jobs_task_sequences[j], task);
        job_of_task[task] = j
    end
    jobs_weights[j] = jobs[j]["weight"];
    jobs_release_date[j] = jobs[j]["release_date"];
    jobs_due_date[j] = jobs[j]["due_date"];
end




t = 0; # time
# while length(done) >= nb_jobs
# mettre à jour les statuts des tâches déjà démarrées
    for i in running_tasks
        if start_time_of_task[i] - duration_task[i] + 1 >= t
            delete(running_tasks, i);
            push!(done_tasks, i);
            delete!(busy_operators, operator_choice_of_task[i]);
            push!(idle_operators, operator_choice_of_task[i]);
            curr_busy_mach_op[machine_choice_of_task[i], operator_choice_of_task[i]] = 0; 
        end
    
    
        push!(todo_tasks, τ);

    for τ in todo_tasks: 
        jt = job_of_task[τ];
        Δt = jobs_due_date[jt] - t;
        if Δt <= 0
            λ = 1/Δt * todo_nb_tasks[jt] * jobs_weights[jt];
        else
            λ = -Δt * todo_nb_tasks[jt] * jobs_weights[jt]; 
        end
        score_of_task[τ] = λ;
        # mxval, mxindx = findmax(collect(score_of_task));
        priority = reverse(sortperm(score_of_task));

        assign = true; i = 1; # commencer par la tâche la plus importante (numéro 1)
        while assign # pour chaque tâche en attente
            task_to_assign = priority[i];
            compat_machine_operator_per_task[task_to_assign,:,:]; # difficulté: assigner un couple opérateur-machine qui optimisera les ressources à l'étape suivantes
            available_resources =  compat_machine_operator_per_task[task_to_assign,:,:] .& .~ curr_busy_mach_op;
            if any(available_resources)
                # c'est possible: on l'assigne
            end
        end 

    end