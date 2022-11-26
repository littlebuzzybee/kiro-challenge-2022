using JSON;
using LinearAlgebra;
using DataStructures;
using DataFrames;

function import_init()
    instance = JSON.parsefile("C:/Users/matth/Documents/GitHub/kiro-challenge-2022/instances/huge.json")

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





function distribution(duration_task::Vector{Int64},
    compat_machine_operator_per_task::Vector{Int64},
    α::Int, β::Int, nb_machines::Int, nb_tasks::Int, nb_jobs::Int, nb_operators::Int,
    jobs_task_sequences::Dict{Int64, Queue{Int64}},
    jobs_weights::Vector{Int64},
    jobs_release_date::Vector{Int64},
    jobs_due_date::Vector{Int64},
    last_task_of_jobs::Vector{Int64},
    job_of_task::Vector{Int64})


    jobs_completion_time   = zeros(Int64, nb_jobs)
    todo_tasks             = Set{Int64}()
    nb_tasks_per_job       = zeros(Int64, nb_jobs)

    running_tasks          = Set{Int64}()
    done_tasks             = Set{Int64}()
    done_jobs              = Set{Int64}()

    busy_machines          = zeros(Bool, nb_machines)
    busy_operators         = zeros(Bool, nb_operators)
    running_jobs           = zeros(Bool, nb_jobs)

    busy_resources         = zeros(Bool, nb_machines, nb_operators)
    # busy_resources encodera en un seul tableau les couples machine_opérateur disponibles à chaque étape d'assignation des tâches

    score_of_task           = zeros(Float64, nb_tasks)
    # ce tableau est réévalué à chaque étape de temps t pour les tâches τ envisagées

    start_time_of_task      = zeros(Int64, nb_tasks)
    complete_time_of_task   = zeros(Int64, nb_tasks)
    operator_choice_of_task = zeros(Int64, nb_tasks)
    machine_choice_of_task  = zeros(Int64, nb_tasks)


    t = 1; # time
    io = open("C:/Users/matth/Documents/GitHub/kiro-challenge-2022/log_algo.txt", "w");
    while (length(done_tasks) < nb_tasks) & (t < 100)
        write(io, "=== Time $t ===\n");
        for τ in running_tasks # mettre à jour les statuts des tâches déjà démarrées: ont-elles terminé ?
            if t - start_time_of_task[τ] >= duration_task[τ] + 1
                delete!(running_tasks, τ);
                push!(done_tasks, τ);

                running_jobs[job_of_task[τ]] = false;

                busy_machines[machine_choice_of_task[τ]]   = false;
                busy_operators[operator_choice_of_task[τ]] = false;
                write(io, "Finishing task $τ\n");
                if τ in last_task_of_jobs
                    γ = job_of_task[τ];
                    jobs_completion_time[γ] = t - 1;
                    push!(done_jobs, γ);
                    write(io, "Completing job $γ\n");
                end
            end
        end
        

        for γ in 1:nb_jobs # mettre à jour les tâches nouvelles à faire en dépilant les séquences des jobs
            if ~running_jobs[γ] && ~isempty(jobs_task_sequences[γ]) # dernière tâche du job finie ou bien job pas encore commencé et il reste des tâches: on les ajoute à la liste des todo
                τ = dequeue!(jobs_task_sequences[γ]);
                push!(todo_tasks, τ);  # on passe à la tâche suivante
                write(io, "Adding task $τ to the queue\n");
            end
        end



        for τ in todo_tasks
            γ  = job_of_task[τ];
            Δt = jobs_due_date[γ] - t;
            score_of_task[τ] = importance(Δt, γ, α,  β, jobs_task_sequences, jobs_weights);
        end
            
            
        todo_tasks_vec = collect(todo_tasks);
        priority = reverse(sortperm(score_of_task[todo_tasks_vec])); # trié dans l'ordre croissant sans le rev
        # mxval, mxindx = findmax(collect(score_of_task));
        tasks_to_assign = todo_tasks_vec[priority];
        write(io, "Tasks to assign: $tasks_to_assign\n\n");

        for τ in tasks_to_assign # for i=1:size(tasks_to_assign)[1]
            # en itérant sur les tâches les plus importantes par ordre décroissantà mesure que l'on parcourt les index (numéro i, i ∈ 1,...)

            busy_resources .= Matrix{Bool}(ones(Bool, nb_machines)*busy_operators' .| busy_machines*ones(Bool, nb_operators)')[:,:]; # calcul opérateur occupé OU LOGIQUE machine occupée
            compatible_resources = compat_machine_operator_per_task[τ,:,:];
            possible_resources   = compatible_resources .& .~busy_resources; # matrice (machine, opérateur) ressource dispo si et seulement si elle est compatible (avec la tâche) et disponible

            
            if any(possible_resources)
                solutions = findall(x -> x == true, possible_resources); # renvoie un vecteur de coordonnées cartésiennes encodant tous les choix possibles de couples (machine, opérateur)
                s = size(solutions)[1]; # nombre de solutions pour cette tâche
                c =  rand(1:s);         # on en choisit une au hasard # À AMÉLIORER

                # pour une amélioration, calculer le sous ensemble maximisant les tâches réalisées en fonction /  option 1: du nombre de tâches démarrées option 2: 
                choice_machine  = solutions[c][1];
                choice_operator = solutions[c][2];

                machine_choice_of_task[τ]  = choice_machine;
                operator_choice_of_task[τ] = choice_operator;
                start_time_of_task[τ]      = t;
                delete!(todo_tasks, τ);

                write(io, "= Commencing task $τ of Job $(job_of_task[τ])\n");
                push!(running_tasks, τ);
                running_jobs[job_of_task[τ]] = true;

                busy_operators[choice_operator]  = true;
                busy_machines[choice_machine]    = true;
                write(io, "operator $(choice_operator) on machine $(choice_machine)\n");
            else
                write(io, "= Postponed task $τ of Job $(job_of_task[τ]) for lack of resource\n");
            end
        end 
        t += 1;
        write(io, "\n\n");
    end
    close(io);

    
    

    sol_cost = solution_cost(nb_jobs, jobs_weights, start_time_of_task, duration_task, jobs_due_date, jobs_completion_time, last_task_of_jobs, α, β);
    return sol_cost, start_time_of_task, busy_resources, jobs_release_date, compat_machine_operator_per_task
end



duration_task,
compat_machine_operator_per_task,
α, β, nb_machines, nb_tasks, nb_jobs, nb_operators,
jobs_task_sequences,
jobs_weights,
jobs_release_date,
jobs_due_date,
last_task_of_jobs,
job_of_task = import_init();



@time sol_cost, start_time_of_task, busy_resources, jobs_release_date, compat_machine_operator_per_task = distribution(duration_task,
compat_machine_operator_per_task,
α, β, nb_machines, nb_tasks, nb_jobs, nb_operators,
jobs_task_sequences,
jobs_weights,
jobs_release_date,
jobs_due_date,
last_task_of_jobs,
job_of_task);


# survey les collisions de couples opérateurs_machines et faire un histogramme pour ensuite optimiser le code

intersection = Float64[];
for i in 1:nb_tasks
    for j in i+1:nb_tasks
        push!(intersection, norm(compat_machine_operator_per_task[i,:,:] .&&  compat_machine_operator_per_task[j,:,:], 1));
    end
end


using Plots;
histogram(intersection)
# on constate que dans une immense majorité de cas, les tâches n'ont quasi aucun couple possible opérateur-machine en commun.
# Ceci veut dire que "souvent" on n'aura pas à se préoccuper de telles intersections pour assigner les tâches avec les ressources disponibles et donc qu'on peut simplement essayer de parcourir le produit cartésien des couples possibles à chaque tâche à assigner lors de l'étape de temps en supprimant les solutions qui collisionnent.