include("ImportAndMetrics.jl")

include("LocalDecision.jl")

ENV["GUROBI_JL_USE_GUROBI_JLL"] = "false"
ENV["GUROBI_HOME"] = "/opt/gurobi1200/linux64"

using JuMP
using Gurobi

# fonction principale du programme qui calcule la stratégie à adopter
function main_strategy(
        compat_machine_operator_per_task::Array{Bool, 3},
        α                    ::Int,
        β                    ::Int,
        duration_task        ::Vector{TimeUnit},
        nb_machines          ::MachineId,
        nb_tasks             ::TaskId,
        nb_jobs              ::JobId,
        nb_operators         ::OperatorId,
        jobs_task_sequences  ::Dict{JobId, Queue{TaskId}},
        jobs_weights         ::Vector{Weight},
        jobs_release_date    ::Vector{TimeUnit},
        jobs_due_date        ::Vector{TimeUnit},
        last_task_of_jobs    ::Vector{TaskId},
        job_of_task          ::Vector{JobId}
    )


    start_time_of_task      = zeros(TimeUnit,   nb_tasks)
    complete_time_of_task   = zeros(TimeUnit,   nb_tasks)
    operator_choice_of_task = zeros(OperatorId, nb_tasks)
    machine_choice_of_task  = zeros(MachineId,  nb_tasks)


    for t=1:nb_jobs
        total_task_time_offset = 0
        for τ in jobs_task_sequences[t]
            start_time_of_task[τ] = jobs_release_date[t] + total_task_time_offset
            complete_time_of_task[τ] = jobs_release_date[t] + duration_task[τ] + total_task_time_offset
            total_task_time_offset += duration_task[τ]
        end
        jobs_release_date[t] = jobs_release_date[t] + total_task_time_offset
    end


    operator_choice_of_task[τ] = 0
    machine_choice_of_task[τ] = 0

    
    model = Model(Gurobi.Optimizer)
    @variable(model, start_time_of_task[1:nb_tasks] >= 0, Int);



    sol_cost = solution_cost(nb_jobs, jobs_weights, start_time_of_task, duration_task, jobs_due_date, jobs_completion_time, last_task_of_jobs, α, β);

    return sol_cost, start_time_of_task, busy_resources, jobs_release_date, compat_machine_operator_per_task
end



duration_task,
compat_machine_operator_per_task,
α, β,
nb_machines,
nb_tasks,
nb_jobs,
nb_operators,
jobs_task_sequences,
jobs_weights,
jobs_release_date,
jobs_due_date,
last_task_of_jobs,
job_of_task = import_init(path*"instances/tiny.json");



@time sol_cost, start_time_of_task, busy_resources, jobs_release_date, compat_machine_operator_per_task = main_strategy(compat_machine_operator_per_task,
                α, β, duration_task,
                nb_machines,
                nb_tasks,
                nb_jobs,
                nb_operators,
                jobs_task_sequences,
                jobs_weights,
                jobs_release_date,
                jobs_due_date,
                last_task_of_jobs,
                job_of_task);

