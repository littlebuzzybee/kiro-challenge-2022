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


# Idée pendant le parcours de l'arbre: pour éviter une explosion en complexité trop problématique, on ne mémorise que les tâches adressées, opérateurs et machines assignés indépendemment de tout mappage entre les tâches et ces derniers. C'est possible car une tâche adressée n'est plus à traiter/décider, et les machines et les opérateurs n'ont pas le don d'ubiquité. Donc pour aller plus loin en profondeur, seules ces informations sont suffisantes pour créer les nœuds.

# Pour créer les assignations des machines et des opérateurs à leurs tâches une fois ces ensembles (machines, opérateurs, tâches) détérminés [sous réserve qu'ils soient entièrement compatibles], il faudra créer un petit solveur en aval une fois l'exploration des possibles effectuée.

# Pour tout état, on pourra ainsi définir un ensemble d'assignations (fonctions machine -> tâche et opérateur -> tâche) respectant la situation. Et choisir celle qui favorise l'étape suivante / ou laisse le plus de choix/flexibilité à l'étape suivante (pour itérer le processus). Quand on parle de flexibilité ici, on parle d'éléments choisis d'opérateurs et machines pour un nombre fixé à l'avance respectivement, pas de leur quantité puisqu'à situation fixée, toute tâche adressée aura besoin exactement d'une machine unique et d'un opérateur unique.

using Distributions; # pour générer des exemples pour les tests
using SparseArrays;

# deux solutions:
dist = Bernoulli(.1);
C = rand(dist, 10, 10);
C = sprand(Bool, 10, 10, .1);


using AutoHashEquals;
@auto_hash_equals struct State
    nb_todo    ::Int64         # le nombre de tâches à adresser
    todo       ::Vector{Bool}   # les tâches à adresser 
    score      ::Number   # score de la configuration (Int ou Float indéfini)
    av_m       ::UInt128  # n'encode pas l'application m2t:  machine ↦ tâche
    av_o       ::UInt128  # n'encode pas l'application m2o:opérateur ↦ tâche
    timestep   ::Int64    # temps
end


# test ssh



function local_decision(duration_task::Vector{Int64},
    compat_m_o           ::Array{Bool, 3}, # alias pour compat_machine_operator_per_task
    α                    ::Number,
    β                    ::Number,
    nb_machines          ::Int,
    nb_tasks             ::Int,
    nb_jobs              ::Int,
    nb_operators         ::Int,
    jobs_task_sequences  ::Dict{Int64, Queue{Int64}}, # à ce stade, n'ont pas été dépilées !
    # ajouter un dict de vecteurs pour avoir l'accès rapide aux tâches  qui arrivent ensuite
    jobs_weights         ::Vector{Int64},
    jobs_release_date    ::Vector{Int64},
    jobs_due_date        ::Vector{Int64},
    last_task_of_jobs    ::Vector{Int64},
    job_of_task          ::Vector{Int64},
    todo_tasks           ::Set{Int64},    # paramètres spécifiques au flow dans la décision (à t fixé)
    av_m                 ::UInt128,       # alias de .~ busy_machines  (machines disponibles)
    av_o                 ::UInt128)       # alias de .~ busy_operators (opérateurs disponibles)

    # SPARSIFIER LES COMPATIBILITÉS POUR CHAQUE TÂCHE
    compat_sparse = Dict{Int64, SparseMatrixCSC{Bool, Int64}}; # créer un dictionnaire ayant pour clé les tâches, et en valeurs les tranches sparsifiées de leur compatibilité (pour itérer dessus ensuite)
    adressable_tasks = Set{Int64}();
    for τ in  todo_tasks                  # remplir le dictionnaire en itérant sur les tâches
        compat_sparse[τ] = sparse(view(compat_m_o, τ,:,:)); # ajouter la tranche
        if @views dot(av_m, compat_sparse[τ], av_o) ≥ 1
            push!(adressable_tasks, τ);
        end
    end

    nb_adr_tasks         = length(adressable_tasks);
    adressable_tasks_vec = Vector(adressable_tasks);


    # CRÉER LES INDICATEURS DE COLLISIONS
    collision_machines   = zeros(Bool, nb_adr_tasks, nb_adr_tasks);
    # collision_machines[i,j] = true ssi les tâches i et j partagent au moins une machine disponible étant capable de les réaliser
    collision_operators  = zeros(Bool, nb_adr_tasks, nb_adr_tasks);
    # collision_operators[i,j] = true ssi les tâches i et j partagent au moins un opérateur disponible étant capable de les réaliser

    for (t1,t2) in Iterators.product(1:nb_adr_tasks, 1:nb_adr_tasks)
        collisions_machines[t1,t2]  = @views any(av_m' * (compat_m_o[s[1],:,:] .| C[s[2],:,:]));
        collisions_operators[t1,t2] = @views any((compat_m_o[s[1],:,:] .| compat_m_o[s[2],:,:]) * av_o);
    end


    # CRÉER LE SETUP POUR LE PARCOURS DE L'ARBRE
    encoding_kernel = 2 .^ (0:nb_adr_tasks-1);

    q = Queue{State}(); # vérifier au moment d'insérer ou remplacer par un ensemble éventuellement (pas besoin de multiplicité dans la file)

    current_state = State(nb_adr_tasks,0.0)
    enqueue!(q, [current_state])

    # PARCOURS
    while ~isempty(q)
        instance       = dequeue!(q);
        # Vector{Int64} transformés en Vector{Bool}
        av_m           = Vector{Bool}(digits(instance.av_m,       base=2, pad=nb_machines));
        av_o           = Vector{Bool}(digits(instance.av_m,       base=2, pad=nb_operators));
        adressable     = Vector{Bool}(digits(instance.adressable, base=2, pad=nb_adr_tasks));
        coonf_score    = instance.conf_score;
        
        for τ=1:nb_adr_tasks # pour toute tâche encore adressable: prendre les bits positifs dans l'ordre du plus 
            if @views dot(av_m, compat_sparse[τ], av_o) ≥ 1 # si adressable avec au moins une des ressources restantes
                c_poss = @views av_m * ones(Bool, nb_operators)' .& compat_sparse[τ] .& ones(Bool, nb_machines) * av_o';
                new_adressable = adressable - encoding_kernel[τ];
                for (m,o,~) in zip(findnz(c_poss)) # parcours sur la tranche de compatibilité là où sont les 1
                end 
            end
        end
    end
end



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
    log_file = open("C:/Users/matth/Documents/GitHub/kiro-challenge-2022/log_algo.txt", "w"); # ouverture du fichier Log
    
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
job_of_task = import_init("C:/Users/matth/Documents/GitHub/kiro-challenge-2022/instances/tiny.json");



@time sol_cost, start_time_of_task, busy_resources, jobs_release_date, compat_machine_operator_per_task = main_strategy(duration_task,
compat_machine_operator_per_task,
α, β, nb_machines, nb_tasks, nb_jobs, nb_operators,
jobs_task_sequences,
jobs_weights,
jobs_release_date,
jobs_due_date,
last_task_of_jobs,
job_of_task);






### BROUILLON


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




using CUDA;

function inplace_kernel!(vec::CuDeviceVector{ComplexF32}, α::ComplexF32, ϵ::Float32, maxiter::Int32, res::CuDeviceVector{Float16})
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(vec) # verify if the index i actually corresponds to a index of the vectors
        ind = Int32(0);
        iterexp = 3.f0*vec[i];
        while (CUDA.abs(iterexp) > ϵ && ind < maxiter) # dummy algorithm to illustrate kernel writing
            iterexp *= α;
            vec[i] += CUDA.exp(-CUDA.abs(2.0f0*vec[i]));
            ind += 1;
        end
        res[i] = convert(Float16, CUDA.abs(iterexp));
    end
    return nothing;
end
