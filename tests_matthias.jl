using JSON;
using LinearAlgebra;
using DataStructures;

instance = JSON.parsefile("C:/Users/matth/Documents/GitHub/kiro-challenge-2022/instances/tiny.json")

parameters = instance["parameters"]
tasks      = instance["tasks"]
jobs       = instance["jobs"]

param_size       = parameters["size"]
parameters_costs = parameters["costs"]

α = parameters_costs["unit_penalty"]
β = parameters_costs["tardiness"]

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


task_set               = Set{Int64}(1:nb_tasks)
todo_tasks             = Set{Int64}()
nb_tasks_per_job       = zeros(Int64, nb_jobs)

running_tasks          = Set{Int64}()
done_tasks             = Set{Int64}()
done_jobs              = Set{Int64}()

busy_machines          = zeros(Bool, nb_machines)
busy_operators         = zeros(Bool, nb_operators)
running_jobs           = zeros(Bool, nb_jobs)

busy_resources         = zeros(Bool, nb_machines, nb_operators)

# idle_operators
# busy_mach_op           = zeros(Bool, nb_machines, nb_operators)
# busy_mach_op encode en un seul tableau les couples machine_opérateur disponibles


jobs_task_sequences = Dict{Int64, Queue{Int}}()
jobs_weights        = zeros(Int64, nb_jobs)
jobs_release_date   = zeros(Int64, nb_jobs)
jobs_due_date       = zeros(Int64, nb_jobs)

job_of_task         = zeros(Int64, nb_tasks)

score_of_task       = zeros(Float64, nb_tasks)
# ce tableau est réévalué à chaque étape de temps t pour les tâches τ envisagées

start_time_of_task      = zeros(Int64, nb_tasks)
operator_choice_of_task = zeros(Int64, nb_tasks)
machine_choice_of_task  = zeros(Int64, nb_tasks)




for j=1:nb_jobs 
    jobs_task_sequences[j] = Queue{Int64}();
    for task in Vector{Int64}(jobs[j]["sequence"]) # remplir les queues de tâches pour tous les jobs
        enqueue!(jobs_task_sequences[j], task);
        job_of_task[task] = j
    end
    jobs_weights[j] = jobs[j]["weight"];
    jobs_release_date[j] = jobs[j]["release_date"];
    jobs_due_date[j] = jobs[j]["due_date"];
end



function importance( 
        Δt::Int,
        γ::Int,
        β::Int,
        jobs_task_sequences::Dict{Int64, Queue{Int64}},
        jobs_weights::Vector{Int64}
        )
    
    # multiples à ajuster: hyperparamètre
    if Δt ≤ 0 # job pas en retard
        return 1/Δt * β * sum(collect(jobs_task_sequences[γ])) * jobs_weights[γ];
        # importance ∝ au poids du job, paramètre β, et inversement ∝ au temps restant pour finir le job
    else # job en retard
        return return abs(Δt) * α * sum(collect(jobs_task_sequences[γ])) * jobs_weights[γ];
        # importance ∝ au poids du job, paramètre α, et ∝ au retard abs(Δt)
    end
end




t = 1; # time
# while length(done) >= nb_jobs


    for τ in running_tasks # mettre à jour les statuts des tâches déjà démarrées: ont-elles terminé ?
        if start_time_of_task[τ] - duration_task[τ] + 1 >= t
            delete(running_tasks, τ);
            push!(done_tasks,    τ);

            running_jobs[job_of_task[τ]] = false;

            busy_machines[machine_choice_of_task[τ]]   = false;
            busy_operators[operator_choice_of_task[τ]] = false;
        end
    end
    

    for γ in 1:nb_jobs # mettre à jour les tâches nouvelles à faire en dépilant les séquences des jobs
        if ~running_jobs[γ] && ~isempty(jobs_task_sequences[γ]) # dernière tâche du job finie ou bien job pas encore commencé et il reste des tâches
            τ = dequeue!(jobs_task_sequences[γ]);
            push!(todo_tasks, τ);  # on passe à la tâche suivante

        end
    end



    for τ in todo_tasks
        γ  = job_of_task[τ];
        Δt = jobs_due_date[γ] - t;
        λ  = importance_on_time(Δt, γ, β, jobs_task_sequences, jobs_weights);
        score_of_task[τ] = λ;
    end
        
        
    todo_tasks_vec = collect(todo_tasks);
    priority = reverse(sortperm(score_of_task[todo_tasks_vec])); # trié dans l'ordre croissant sans le rev
    # mxval, mxindx = findmax(collect(score_of_task));
    tasks_to_assign = todo_tasks_vec[priority]

    for τ in tasks_to_assign # for i=1:size(tasks_to_assign)[1]
        # en itérant sur les tâches les plus importantes par ordre décroissantà mesure que l'on parcourt les index (numéro i, i ∈ 1,...)

        busy_resources       .= Matrix{Bool}(ones(Bool, nb_machines)*busy_operators' .| busy_machines*ones(Bool, nb_operators)')[:,:]; # calcul opérateur occupé OU LOGIQUE machine occupée
        compatible_resources = compat_machine_operator_per_task[τ,:,:];
        possible_resources  = compatible_resources .& .~busy_resources; # couple (machine, opérateur) dispo SSI c'est compatible et disponible

        
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

            println("Task $τ");
            push!(running_tasks, τ);
            running_jobs[job_of_task[τ]] = true;

            busy_operators[choice_operator]  = true;
            busy_machines[choice_machine]    = true;
            println("operator $(choice_operator) starting machine $(choice_machine)")
        end
    end 
