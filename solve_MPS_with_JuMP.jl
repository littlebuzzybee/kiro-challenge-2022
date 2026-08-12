using JuMP
# using Ipopt
# using Gurobi
# using SCIP
# using KNITRO
using CPLEX

# Import the model

println("Importing model")
model = read_from_file("lp_problems/gurobi_tiny.mps")
# print(model) # beware if the model is too large

# Select the solver and optimize the model
println("setting optimizer")
set_optimizer(model, CPLEX.Optimizer)
println("optimizing")
optimize!(model)
