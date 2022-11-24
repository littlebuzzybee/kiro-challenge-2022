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

proc_time_task = zeros(Int32, nb_tasks)

compat_machine_operator_per_task = zeros(Int32, (nb_tasks, nb_machines, nb_operators))

for i=1:nb_tasks
    task = tasks[i];
    proc_time_task[i] = task["processing_time"];
    possible_machines = task["machines"];
    for machine in possible_machines
        machine_index = machine["machine"];
        possible_operators = Vector{Int64}(machine["operators"]);
        compat_machine_operator_per_task[Int64(i), machine_index, possible_operators] .= 1;
    end
end




running    = Set{Int64}();
done       = Set{Int64}();
busy_op    = Set{Int64}();
buzy_mac   = Set{Int64}();


jobs_task_sequences = Dict{Int64, Vector{Int64}}
jobs_weights        = zeros(Int64, nb_jobs)


s = Stack{Int}()

for i=1:nb_jobs
    jobs_task_sequences[i] = jobs[i]["sequence"];
    println(typeof(jobs[i]["sequence"]))

end


