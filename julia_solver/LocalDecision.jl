# Idée pendant le parcours de l'arbre: pour éviter une explosion en complexité trop problématique, on ne mémorise que les tâches adressées, opérateurs et machines assignés indépendemment de tout mappage entre les tâches et ces derniers. C'est possible car une tâche adressée n'est plus à traiter/décider, et les machines et les opérateurs n'ont pas le don d'ubiquité. Donc pour aller plus loin en profondeur, seules ces informations sont suffisantes pour créer les nœuds. On pourra ensuite se donner un seuil (par exemple; À CHAQUE ÉTAPE DE TEMPS: ne propager en utilisant seulement les noeuds correspondant aux 20% des assignments couvrant le + de tâches // ou bien avec un nombre fixe max de noeuds pour majorer l'explosion combinatoire mémoire), et le maximum si pas assez de ressources) de nouvelles tâches à effectuer à chaque étape de temps

# Pour créer les assignations des machines et des opérateurs à leurs tâches une fois ces ensembles (machines, opérateurs, tâches) détérminés [sous réserve qu'ils soient entièrement compatibles], il faudra créer un petit solveur en aval une fois l'exploration des possibles effectuée.

# Pour tout état, on pourra ainsi définir un ensemble d'assignations (fonctions machine -> tâche et opérateur -> tâche) respectant la situation. Et choisir celle qui favorise l'étape suivante / ou laisse le plus de choix/flexibilité à l'étape suivante (pour itérer le processus). Quand on parle de flexibilité ici, on parle d'éléments choisis d'opérateurs et machines pour un nombre fixé à l'avance respectivement, pas de leur quantité puisqu'à situation fixée, toute tâche adressée aura besoin exactement d'une machine unique et d'un opérateur unique.

# Si on itère loin dans le futur, il faut créer alors une fonction d'évaluation de la situation (comme pour l'algorithme du minimax aux échecs, les coups de l'adversaire étant comparables à l'ensemble des tâches pouvant être ajoutées (l'information en plus qu'on ne maîtrise pas et qui nous est imposée) par le planning et l'arrivage des job releases)

using Distributions; # pour générer des exemples pour les tests
using SparseArrays;

# deux solutions:
dist = Bernoulli(.1);
C = rand(dist, 10, 10);
C = sprand(Bool, 10, 10, .1);


using AutoHashEquals;
@auto_hash_equals struct State
    nb_todo    ::Int64           # le nombre de tâches à adresser
    todo       ::Set{Int64}      # les tâches à adresser: Set plutôt que Vector{Bool} car ensemble réduit
    score      ::Number          # score de la configuration (Int ou Float indéfini)
    av_m       ::UInt128         # n'encode pas l'application m2t:  machine ↦ tâche pris seul (nécessite la branche complète)
    av_o       ::UInt128         # n'encode pas l'application m2o:opérateur ↦ tâche
    timestep   ::Int64           # temps (profondeur du nœud)
end


function evaluate_state(
        nb_todo::Int64,
#=         jobs_weights::Vector{Int64},
        duration_task::Vector{Int64},
        jobs_due_date::Vector{Int64},
        α::Int64, β::Int64 =#
    )::Number # à spécifier
    return nb_todo; # à modifier pour tester différentes fonctions d'évaluation des nœuds
end





function local_decision(duration_task::Vector{Int64},
    compat_m_o           ::Array{Bool, 3}, # alias pour compat_machine_operator_per_task
    α                    ::Number,
    β                    ::Number,
    nb_machines          ::Int,
    nb_tasks             ::Int,
    nb_jobs              ::Int,
    nb_operators         ::Int,
    jobs_task_sequences  ::Dict{Int64, Queue{Int64}}, # sera modifiée en place si profondeur temporelle > 1
    # ajouter un dict de vecteurs pour avoir l'accès rapide aux tâches  qui arrivent ensuite
    jobs_weights         ::Vector{Int64},
    jobs_release_date    ::Vector{Int64},
    jobs_due_date        ::Vector{Int64},
    last_task_of_jobs    ::Vector{Int64},
    job_of_task          ::Vector{Int64},
    todo_tasks           ::Set{Int64},    # paramètres spécifiques au flow dans la décision (à t fixé)
    av_m                 ::UInt128,       # alias de .~ busy_machines  (machines disponibles)
    av_o                 ::UInt128,       # alias de .~ busy_operators (opérateurs disponibles)
    timestep             ::Int64)
    

    # SPARSIFIER LES COMPATIBILITÉS POUR CHAQUE TÂCHE ET CONSTRUIRE UN ENSEMBLE (type Set) DE TÂCHES ADRESSABLES

    compat_sparse = Dict{Int64, SparseMatrixCSC{Bool, Int64}}; # créer un dictionnaire ayant pour clé les tâches, et en valeurs les tranches sparsifiées de leur compatibilité (pour ITÉRER dessus ensuite)
    adressable_tasks = Set{Int64}();
    for τ in  todo_tasks                  # remplir le dictionnaire en itérant sur les tâches
        compat_sparse[τ] = sparse(view(compat_m_o, τ,:,:)); # ajouter la tranche
        if @views dot(av_m, compat_sparse[τ], av_o) ≥ 1
            push!(adressable_tasks, τ);
        end
    end

    nb_adr_tasks         = length(adressable_tasks);
    adressable_tasks_vec = Vector(adressable_tasks);


    # CRÉER LES INDICATEURS DE COLLISIONS (Matrices de booléens)

    collisions_machines   = zeros(Bool, nb_adr_tasks, nb_adr_tasks);
    collisions_operators  = zeros(Bool, nb_adr_tasks, nb_adr_tasks);

    for (t1,t2) in Iterators.product(1:nb_adr_tasks, 1:nb_adr_tasks)
        collisions_machines[t1,t2]  = @views any(av_m' * (compat_m_o[t1,:,:] .| C[t2,:,:]));
        collisions_operators[t1,t2] = @views any((compat_m_o[t1,:,:] .| compat_m_o[t2,:,:]) * av_o);
    end
    # collision_machines[i,j] = true ssi les tâches i et j partagent au moins une machine disponible étant capable de les réaliser toutes les 2
    # collision_operators[i,j] = true ssi les tâches i et j partagent au moins un opérateur disponible étant capable de les réaliser toutes les 2



    # CRÉER LE SETUP POUR LE PARCOURS DE L'ARBRE
    encoding_kernel = 2 .^ (0:nb_adr_tasks-1);

    queue_states = Queue{State}(); # vérifier au moment d'insérer ou remplacer par un ensemble éventuellement (pas besoin de multiplicité dans la file)


    σ = evaluate_state(nb_adr_tasks); # évaluer le score du premier nœud
    current_state = State(nb_adr_tasks, adressable_tasks, σ, av_m, av_o, timestep); # créer le premier nœud
    enqueue!(queue_states, current_state);

    # PARCOURS

    # réflexions pour la suite: quid du dictionnaire pour les noeuds traités
    # cf idée de ne pas stocker les associations entre tâches, machines et opérateurs, comment alors évaluer le score d'un noeud ? prendre le min selon les associations ? mais ne change rien tant qu'on a pas passé l'étape suivante ! oblige à recalculer ensuite à cause de la fonction pas stockée ? et donc ajouter ça à la fonction d'évaluation ?
    # tout réside dans la fonction d'évaluation évoquée + haut ! Comment évaluer la fonction sur un Noeud intermédiaire sans recourir à de la récursivité ! Cela requière automatiquement que cette fonction puisse être évaluée progressivement sur une branche de l'arbre (par exemple en sommant des termes...)
    while ~isempty(q)
        instance       = dequeue!(queue_states);
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
