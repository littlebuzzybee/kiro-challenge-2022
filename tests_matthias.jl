using JSON;
using LinearAlgebra;
using DataStructures;

instance = JSON.parsefile("C:/Users/matth/Documents/GitHub/kiro-challenge-2022/instances/tiny.json")

parameters = instance["parameters"]
tasks      = instance["tasks"]
jobs       = instance["jobs"]

param_size       = parameters["size"]
parameters_costs = parameters["costs"]
nb_machines, nb_tasks, nb_jobs, nb_operators = param_size["nb_machines"], param_size["nb_tasks"], param_size["nb_jobs"], param_size["nb_operators"]

interim, unit_penalty, tardiness = parameters_costs["interim"], parameters_costs["unit_penalty"], parameters_costs["tardiness"]

duration_task = zeros(Int32, nb_tasks)

compat_machine_operator_per_task = zeros(Int32, (nb_tasks, nb_machines, nb_operators))

for i=1:nb_tasks
    task = tasks[i];
    duration_task[i] = task["processing_time"];
    possible_machines = task["machines"];
    for machine in possible_machines
        machine_index = machine["machine"];
        possible_operators = Vector{Int64}(machine["operators"]);
        compat_machine_operator_per_task[Int64(i), machine_index, possible_operators] .= 1;
    end
end



todo_task        = Set{Int64}(1:nb_tasks);
running_task     = Set{Int64}();
done_task        = Set{Int64}();
busy_machine     = Set{Int64}();
busy_operator    = Set{Int64}();
idle_operator    = Set{Int64}(1:nb_operators);


jobs_task_sequences = Dict{Int64, Queue{Int}}();
jobs_weights        = zeros(Int64, nb_jobs)
jobs_release_date   = zeros(Int64, nb_jobs)
jobs_due_date       = zeros(Int64, nb_jobs)

job_of_task         = zeros(Int64,   nb_tasks);
score_of_task       = zeros(Float64, nb_tasks);
start_time_of_task  = zeros(Int64, nb_tasks);

operator_choice_of_task = zeros(Int64, nb_tasks);



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
    for i in running
        if start_time_of_task[i] - duration_task[i] + 1 >= t
            delete(running_task, i);
            push!(done_task, i);
            delete!(busy_operator, operator_choice_of_task[i]);
            push!(idle_operator, operator_choice_of_task[i]);
        end

    for λ in todo:
        jt = job_of_task[λ];
        score = 0;
        Δt = jobs_due_date[job_of_task[λ]] - t;
        if Δt <= 0
            score = 1/Δt * jobs_weights[job_of_task[λ]];
        else
            score = 
        end
        score_of_task[λ] = 
    end
# end


