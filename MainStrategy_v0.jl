include("ImportAndMetrics.jl")






# fonction principale du programme qui calcule la stratégie à adopter
function main_strategy(duration_task::Vector{Int64},
    compat_machine_operator_per_task::Array{Bool, 3},
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


    t = 1; # Initialisation du temps
    log_file = open(path*"/log_algo.txt", "w"); # ouverture du fichier Log
    
    # BOUCLE PRINCIPALE DE DÉCISION
    while (length(done_tasks) < nb_tasks) & (t < 100)
        write(log_file, "=== Time $t ===\n");
        for τ in running_tasks # mettre à jour les statuts des tâches déjà démarrées: ont-elles terminé ?
            if t - start_time_of_task[τ] >= duration_task[τ] + 1
                delete!(running_tasks, τ);
                push!(done_tasks, τ);

                running_jobs[job_of_task[τ]] = false;

                busy_machines[machine_choice_of_task[τ]]   = false;
                busy_operators[operator_choice_of_task[τ]] = false;
                write(log_file, "Finishing task $τ\n");
                if τ in last_task_of_jobs
                    γ = job_of_task[τ];
                    jobs_completion_time[γ] = t - 1;
                    push!(done_jobs, γ);
                    write(log_file, "Completing job $γ\n");
                end
            end
        end
        

        for γ in 1:nb_jobs # mettre à jour les tâches nouvelles à faire en dépilant les séquences des jobs
            if ~running_jobs[γ] && ~isempty(jobs_task_sequences[γ]) # dernière tâche du job finie ou bien job pas encore commencé et il reste des tâches: on les ajoute à la liste des todo
                τ = dequeue!(jobs_task_sequences[γ]);
                push!(todo_tasks, τ);  # on passe à la tâche suivante
                write(log_file, "Adding task $τ to the queue\n");
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
        write(log_file, "Tasks to assign: $tasks_to_assign\n\n");

        for τ in tasks_to_assign # for i=1:size(tasks_to_assign)[1]
            # en itérant sur les tâches les plus importantes par ordre décroissant à mesure que l'on parcourt les index (numéro i, i ∈ 1,...)

            busy_resources .= Matrix{Bool}(ones(Bool, nb_machines)*busy_operators' .| busy_machines*ones(Bool, nb_operators)')[:,:]; # calcul [opérateur occupé] OU LOGIQUE [machine occupée]
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

                write(log_file, "= Commencing task $τ of Job $(job_of_task[τ])\n");
                push!(running_tasks, τ);
                running_jobs[job_of_task[τ]] = true;

                busy_operators[choice_operator]  = true;
                busy_machines[choice_machine]    = true;
                write(log_file, "operator $(choice_operator) on machine $(choice_machine)\n");
            else
                write(log_file, "= Postponed task $τ of Job $(job_of_task[τ]) for lack of resource\n");
            end
        end 
        t += 1;
        write(log_file, "\n\n");
    end
    close(log_file);

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
job_of_task = import_init(path*"instances/tiny.json");



@time sol_cost, start_time_of_task, busy_resources, jobs_release_date, compat_machine_operator_per_task = main_strategy(duration_task,
compat_machine_operator_per_task,
α, β, nb_machines, nb_tasks, nb_jobs, nb_operators,
jobs_task_sequences,
jobs_weights,
jobs_release_date,
jobs_due_date,
last_task_of_jobs,
job_of_task);



