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
    todo       ::Vector{Bool}   # les tâches à adresser <" vecteur de booléens ? Set ?
    score      ::Number   # score de la configuration (Int ou Float indéfini)
    av_m       ::UInt128  # n'encode pas l'application m2t:  machine ↦ tâche
    av_o       ::UInt128  # n'encode pas l'application m2o:opérateur ↦ tâche
    timestep   ::Int64    # temps
end







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
