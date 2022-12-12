using JSON;
using LinearAlgebra;
using DataStructures;
# using DataFrames;



function import_init(file::String)
    instance = JSON.parsefile(file)

    parameters = instance["parameters"]
    tasks      = instance["tasks"]
    jobs       = instance["jobs"]

    param_size       = parameters["size"]
    parameters_costs = parameters["costs"]



    nb_machines, nb_tasks, nb_jobs, nb_operators = param_size["nb_machines"], param_size["nb_tasks"], param_size["nb_jobs"], param_size["nb_operators"]

    interim, unit_penalty, tardiness = parameters_costs["interim"], parameters_costs["unit_penalty"], parameters_costs["tardiness"]

    α = parameters_costs["unit_penalty"]
    β = parameters_costs["tardiness"]

    duration_task        = zeros(Int64, nb_tasks)
    job_of_task          = zeros(Int64, nb_tasks)
    jobs_task_sequences  = Dict{Int64, Queue{Int}}()

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

    jobs_weights         = zeros(Int64, nb_jobs)
    jobs_release_date    = zeros(Int64, nb_jobs)
    jobs_due_date        = zeros(Int64, nb_jobs)
    last_task_of_jobs    = zeros(Int64, nb_jobs)
    
    

    for γ=1:nb_jobs 
        jobs_task_sequences[γ] = Queue{Int64}();
        for τ in Vector{Int64}(jobs[γ]["sequence"]) # remplir les queues de tâches pour tous les jobs
            enqueue!(jobs_task_sequences[γ], τ);
            job_of_task[τ] = γ;
        end
        jobs_weights[γ] = jobs[γ]["weight"];
        jobs_release_date[γ] = jobs[γ]["release_date"];
        jobs_due_date[γ] = jobs[γ]["due_date"];
        last_task_of_jobs[γ] = last(jobs_task_sequences[γ]);
    end
    


    return duration_task, compat_machine_operator_per_task,
            α, β, nb_machines, nb_tasks, nb_jobs, nb_operators,
            jobs_task_sequences,
            jobs_weights,
            jobs_release_date,
            jobs_due_date,
            last_task_of_jobs,
            job_of_task
end




function importance(Δt::Int, γ::Int, α::Int, β::Int,
    jobs_task_sequences::Dict{Int64, Queue{Int64}}, jobs_weights::Vector{Int64})

    # multiples à ajuster: hyperparamètre
    if Δt ≤ 0 # job pas en retard
        return  15*jobs_weights[γ]; # * 1/Δt * β * sum(collect(jobs_task_sequences[γ]));
        # importance ∝ au poids du job, paramètre β, et inversement ∝ au temps restant pour finir le job
    else # job en retard
        return jobs_weights[γ]; # * 1 * abs(Δt) * α * sum(collect(jobs_task_sequences[γ]));
        # importance ∝ au poids du job, paramètre α, et ∝ au retard abs(Δt)
    end
end


function solution_cost(nb_jobs::Int64,
        jobs_weights::Vector{Int64},
        start_time_of_task::Vector{Int64},
        duration_task::Vector{Int64},
        jobs_due_date::Vector{Int64},
        jobs_completion_time::Vector{Int64},
        last_task_of_jobs::Vector{Int64},
        α::Int64, β::Int64)
    S = 0;
    for γ in 1:nb_jobs
        τ = last_task_of_jobs[γ];
        completed_job_time = start_time_of_task[τ] + duration_task[τ];
        # @assert completed_job_time == jobs_completion_time[γ];
        if completed_job_time > jobs_due_date[γ]
            u = 1
        else
            u = 0
        end
        T = max(0, completed_job_time - jobs_due_date[γ]);
        S += jobs_weights[γ]*(completed_job_time + α*u + β*T);
    end
    return S
end




windows_path = "C:/Users/matth/Documents/GitHub/kiro-challenge-2022/";
linux_path   = "/home/matthias/Documents/GitHub/kiro-challenge-2022/";

path = linux_path;